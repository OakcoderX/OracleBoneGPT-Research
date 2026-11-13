"""Whole-class hidden matching on new, disjoint query and reference components."""
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np

from cipher_rare_holdout import known_classes, select
from experiment import Corpus, mark_duplicates, split
from prepare import write

SEED = 20261115
PAST = (17, 29, 43, 20261113, 20261114)
SIZE = 16
METHODS = ("gpt", "adjacent", "bag")


def freeze():
    data, _ = split(SEED)
    old = {r["component"] for seed in PAST for r in split(seed)[0]["test"]}
    data["test"] = [r for r in data["test"] if r["component"] not in old]
    fit, refs, test = (data[n] for n in ("train", "validation", "test"))
    freq = Corpus(fit).frequency
    hidden = {u for u in known_classes() if 1 <= freq[u] <= 99}
    mark_duplicates(fit, refs)
    mark_duplicates(fit + refs, test)
    reference = select(refs, hidden, freq)
    ordered = sorted(reference, key=lambda u: (freq[u], sha256(u.encode()).hexdigest()))
    blocks = [ordered[i:i+SIZE] for i in range(0, len(ordered)-SIZE+1, SIZE)]
    lookup = {u: i for i, block in enumerate(blocks) for u in block}
    possible = select(test, hidden, freq)
    supported = select(test, set(lookup), freq)
    q_components = [obs[0][0]["component"] for obs in supported.values()]
    r_components = [reference[u][0][0]["component"] for u in lookup]
    assert len(set(q_components + r_components)) == len(q_components) + len(r_components)
    assert not old.intersection(q_components)
    return data, hidden, reference, blocks, lookup, possible, supported, freq


def masked_context(row, i, hidden):
    seq = ["@hidden_class" if u in hidden else u for u in row["tokens"]]
    return seq[:i], seq[i+1:][::-1]


def fit_models(raw, hidden, path):
    from blind_decipher import reversed_rows, load_model
    from blind_unseen_usage import mask
    from train import train_gpt

    masked = mask(raw, hidden)
    train, dev = [], []
    for row in masked:
        (dev if sha256(row["component"].encode()).digest()[0] < 26 else train).append(row)
    assert not ({r["component"] for r in train} & {r["component"] for r in dev})
    assert all(not hidden.intersection(r["tokens"]) for r in train + dev)
    corpus = Corpus(train)
    assert not hidden.intersection(corpus.input_index)
    models, info = [], {}
    for direction, (a, b) in enumerate(((train, dev), (reversed_rows(train), reversed_rows(dev)))):
        folder = path / str(direction)
        folder.mkdir()
        record, _ = train_gpt(corpus, a, b, [], SEED+900*direction, folder)
        weights = folder / "gpt.safetensors"
        models.append(load_model(corpus, weights))
        info[direction] = record | {"sha256": sha256(weights.read_bytes()).hexdigest()}
    return corpus, models, train, info


def profiles(obs, hidden, corpus, models, idf):
    from blind_unseen_usage import score

    ctx = [masked_context(r, i, hidden) for r, i in obs]
    p = score(ctx, corpus, *models)
    width = len(corpus.input_index)
    bag = np.zeros((len(ctx), 2 * width))
    for i, pair in enumerate(ctx):
        for side, tokens in enumerate(pair):
            indices = np.array(corpus.encode(tokens), dtype=int)
            np.add.at(bag[i], side * width + indices, idf[indices])
    return dict(gpt=p["gpt_usage"], adjacent=p["adjacent_anchors"], bag=bag)


def compare(query, reference, blocks, lookup, freq, query_obs, ref_obs):
    from blind_unseen_usage import unit

    rows = []
    for n, uid in enumerate(query_obs):
        block_id = lookup[uid]
        block = blocks[block_id]
        target = block.index(uid)
        methods = {}
        for name in METHODS:
            gallery = np.stack([reference[u][name] for u in block])
            similarity = unit(np.sqrt(query[name][n])) @ unit(np.sqrt(gallery)).T
            rank = int(np.count_nonzero(similarity >= similarity[target] - 1e-10))
            present = np.sort(similarity)
            absent = np.sort(np.delete(similarity, target))
            methods[name] = dict(rank=rank, present_score=float(present[-1]), absent_score=float(absent[-1]),
                                 present_margin=float(present[-1]-present[-2]),
                                 absent_margin=float(absent[-1]-absent[-2]))
        rows.append(dict(uid=uid, frequency=freq[uid], block=block_id, methods=methods,
                         query=query_obs[uid][0][0]["component"],
                         reference=ref_obs[uid][0][0]["component"]))
    return rows


def metrics(rows):
    return {m: [sum(r["methods"][m]["rank"] <= k for r in rows) for k in (1, 5)] for m in METHODS}


def reject(rows):
    # Entire candidate lists are assigned to calibration or test to isolate their profiles.
    dev = [r for r in rows if r["block"] % 3 == 0]
    test = [r for r in rows if r["block"] % 3 != 0]
    result = {}
    for m in METHODS:
        a = [r["methods"][m] for r in dev]
        b = [r["methods"][m] for r in test]
        result[m] = {}
        for field in ("score", "margin"):
            thresholds = sorted({r[f"{side}_{field}"] for r in a for side in ("present", "absent")})
            safe = [t for t in thresholds if sum(r[f"absent_{field}"] > t for r in a) <= len(a)//10]
            t = max(safe, key=lambda t: (sum(r["rank"] == 1 and r[f"present_{field}"] > t for r in a), t))
            result[m][field] = dict(threshold=t,
                                    accepted=sum(r[f"present_{field}"] > t for r in b),
                                    correct=sum(r["rank"] == 1 and r[f"present_{field}"] > t for r in b),
                                    false_accept=sum(r[f"absent_{field}"] > t for r in b))
    return dict(calibration=len(dev), test=len(test), methods=result)


def run(out):
    data, hidden, refs, blocks, lookup, possible, queries, freq = freeze()
    out.mkdir(parents=True, exist_ok=False)
    corpus, models, train, info = fit_models(data["train"], hidden, out)
    df = Counter(t for r in train for t in set(r["tokens"]))
    idf = np.ones(len(corpus.input_index))
    for t, i in corpus.input_index.items():
        idf[i] += np.log((1+len(train))/(1+df[t]))
    ref_items = list(lookup)
    ref_profiles = profiles([refs[u][0] for u in ref_items], hidden, corpus, models, idf)
    reference = {u: {m: ref_profiles[m][i] for m in METHODS} for i, u in enumerate(ref_items)}
    rows = compare(profiles([v[0] for v in queries.values()], hidden, corpus, models, idf),
                   reference, blocks, lookup, freq, queries, refs)
    from verify_cipher import dependency_interval

    contrasts = {}
    for name in ("adjacent", "bag"):
        for k in (1, 5):
            changes = [([r["block"]], int(r["methods"]["gpt"]["rank"] <= k) -
                        int(r["methods"][name]["rank"] <= k)) for r in rows]
            contrasts[f"{name}/{k}"] = dict(net=sum(d for _, d in changes),
                                             interval=dependency_interval(changes))
    summary = dict(all=metrics(rows), rare=metrics([r for r in rows if r["frequency"] <= 9]),
                   no_block_reference=sorted(possible.keys()-lookup.keys()),
                   n=len(rows), rare_n=sum(r["frequency"] <= 9 for r in rows))
    write(out / "results.json", dict(hidden=len(hidden), training=info, blocks=blocks, rows=rows,
                                    summary=summary, contrasts=contrasts, rejection=reject(rows)))
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    out = parser.parse_args().output
    if out:
        run(out)
    else:
        data, hidden, refs, blocks, lookup, possible, queries, freq = freeze()
        print(len(hidden), len(refs), len(blocks), len(possible), len(queries),
              Counter("rare" if freq[u] <= 9 else "mid" for u in queries),
              Counter(lookup[u] % 3 for u in queries))

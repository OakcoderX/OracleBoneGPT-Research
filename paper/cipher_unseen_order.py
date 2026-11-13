"""Replay whole-class matching with order interventions and equal-size lists."""
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np

from cipher_unseen_blocks import METHODS, compare, freeze, masked_context, profiles, reject
from experiment import Corpus
from prepare import ordinary, write
from verify_cipher import dependency_interval, read

SEEDS = range(64)


def digest(value):
    return sha256(value.encode()).hexdigest()


def perturb(context, rng, near):
    result = []
    for side in context:
        permuted = side.copy()
        indices = [i for i, t in enumerate(side) if ordinary(t)]
        if near:
            indices = indices[-2:]
        if len(indices) >= 2:
            for i, token in zip(indices, rng.permutation([side[i] for i in indices]), strict=True):
                permuted[i] = str(token)
        result.append(permuted)
    return result


def summary(similarity, gold, replacement):
    present = np.sort(similarity)
    absent = np.sort(np.append(np.delete(similarity, gold), replacement))
    return dict(rank=int(np.count_nonzero(similarity >= similarity[gold]-1e-10)),
                present_score=float(present[-1]), absent_score=float(absent[-1]),
                present_margin=float(present[-1]-present[-2]), absent_margin=float(absent[-1]-absent[-2]))


def run(archive, output):
    from blind_decipher import load_model
    from blind_unseen_usage import mask, score, unit

    data, hidden, refs, blocks, lookup, possible, queries, freq = freeze()
    saved = read(archive / "results.json")
    assert saved["blocks"] == blocks
    training = [r for r in mask(data["train"], hidden) if sha256(r["component"].encode()).digest()[0] >= 26]
    corpus = Corpus(training)
    assert not hidden.intersection(corpus.input_index)
    models = []
    for direction in (0, 1):
        path = archive / str(direction) / "gpt.safetensors"
        assert sha256(path.read_bytes()).hexdigest() == saved["training"][str(direction)]["sha256"]
        models.append(load_model(corpus, path))
    df = Counter(t for r in training for t in set(r["tokens"]))
    idf = np.ones(len(corpus.input_index))
    for token, i in corpus.input_index.items():
        idf[i] += np.log((1+len(training))/(1+df[token]))

    # Keep replacement profiles within the calibration or testing partition.
    replacements = {}
    for block_id, block in enumerate(blocks):
        candidates = [u for u, b in lookup.items() if b != block_id and (b % 3 == 0) == (block_id % 3 == 0)]
        replacements[block_id] = min(candidates,
                                     key=lambda u: (abs(np.log(freq[u]/freq[block[7]])), digest(f"{block_id}/{u}")))
    ref_items = list(lookup)
    features = profiles([refs[u][0] for u in ref_items], hidden, corpus, models, idf)
    reference = {u: {m: features[m][i] for m in METHODS} for i, u in enumerate(ref_items)}
    query_items = [v[0] for v in queries.values()]
    features = profiles(query_items, hidden, corpus, models, idf)
    original = compare(features, reference, blocks, lookup, freq, queries, refs)
    assert original == saved["rows"]
    context = [masked_context(row, i, hidden) for row, i in query_items]
    result = []
    for i, row in enumerate(original):
        block_id = row["block"]
        block = blocks[block_id]
        gold = block.index(row["uid"])
        names = [*block, replacements[block_id]]
        methods = {}
        for name in METHODS:
            gallery = np.stack([reference[u][name] for u in names])
            scores = unit(np.sqrt(features[name][i])) @ unit(np.sqrt(gallery)).T
            methods[name] = summary(scores[:-1], gold, scores[-1])
            assert methods[name]["rank"] == row["methods"][name]["rank"]
        result.append(row | {"methods": methods, "replacement": names[-1]})

    galleries = {b: unit(np.sqrt(np.stack([reference[u]["gpt"] for u in block])))
                 for b, block in enumerate(blocks)}
    interventions = {}
    for mode in ("near", "all"):
        rng = np.random.default_rng(65329)
        ranks = []
        changed = np.zeros(len(context), dtype=bool)
        for _ in SEEDS:
            batch = [perturb(ctx, rng, mode == "near") for ctx in context]
            changed |= [a != b for a, b in zip(context, batch, strict=True)]
            for before, after in zip(context, batch, strict=True):
                for a, b in zip(before, after, strict=True):
                    assert Counter(a) == Counter(b)
                    assert all(x == y for x, y in zip(a, b, strict=True) if not ordinary(x))
            shuffled = score(batch, corpus, *models)["gpt_usage"]
            sample = []
            for i, row in enumerate(result):
                scores = unit(np.sqrt(shuffled[i])) @ galleries[row["block"]].T
                gold = blocks[row["block"]].index(row["uid"])
                sample.append(int(np.count_nonzero(scores >= scores[gold]-1e-10)))
            ranks.append(sample)
        ranks = np.array(ranks)
        base = np.array([r["methods"]["gpt"]["rank"] for r in result])
        for i, row in enumerate(result):
            row[mode] = dict(changed=bool(changed[i]), ranks=ranks[:, i].tolist())
        cohorts = {}
        for key, included in (("all", np.ones(len(result), dtype=bool)), ("changed", changed)):
            values = {}
            for k in (1, 5):
                counts = (ranks[:, included] <= k).sum(axis=1)
                avg = (ranks <= k).mean(axis=0)
                changes = [([r["block"]], int(base[i] <= k) - avg[i])
                           for i, r in enumerate(result) if included[i]]
                values[k] = dict(original=int((base[included] <= k).sum()),
                                 shuffled_mean=float(counts.mean()),
                                 interval=dependency_interval(changes))
            cohorts[key] = dict(n=int(included.sum()), hits=values)
        interventions[mode] = cohorts

    # Actual absence from this archive. Pick a test-only list by pre-existing
    # frequency band, without considering similarity or using these queries in calibration.
    used = {r["query"] for r in result}
    test_blocks = [i for i in range(len(blocks)) if i % 3 != 0]
    absent_obs = [(u, obs[0]) for u, obs in possible.items()
                  if u not in lookup and obs[0][0]["component"] not in used]
    x = profiles([o for _, o in absent_obs], hidden, corpus, models, idf)
    abstention = reject(result)
    natural = []
    for i, (uid, (row, position)) in enumerate(absent_obs):
        b = min(test_blocks, key=lambda j: (abs(np.log(freq[uid]/freq[blocks[j][7]])), j))
        methods = {}
        for name in METHODS:
            gallery = np.stack([reference[u][name] for u in blocks[b]])
            scores = np.sort(unit(np.sqrt(x[name][i])) @ unit(np.sqrt(gallery)).T)
            methods[name] = dict(score=float(scores[-1]), margin=float(scores[-1]-scores[-2]))
        natural.append(dict(uid=uid, block=b, query=row["component"], position=position, methods=methods))
    real_absence = {name: {key: sum(r["methods"][name][key] > abstention["methods"][name][key]["threshold"]
                                    for r in natural) for key in ("score", "margin")} for name in METHODS}
    write(output, dict(rows=result, interventions=interventions, equal_size=abstention,
                       absent_rows=natural, absent_accept=real_absence))
    print(interventions, abstention, len(natural), real_absence, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.archive, args.output)

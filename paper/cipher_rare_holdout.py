"""Test low-frequency cipher recovery on unique classes and bone components."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from math import comb, log
from pathlib import Path

import numpy as np

from experiment import Corpus, mark_duplicates, split
from prepare import write

SEED = 20261114
OLD = (17, 29, 43, 20261113)
BANDS = ((1, 9), (10, 29), (30, 99))
METHODS = ("frequency", "gpt", "gpt_prior_once", "trigram", "trigram_prior_once")


def order_key(*values):
    return sha256("/".join(map(str, (SEED, *values))).encode()).hexdigest()


def known_classes():
    from verify_cipher import ROOT, read

    meta = read(ROOT / "source/Main-character.json")
    refs = {uid: values[0] for uid, entry in meta.items() if len(values := entry.get("transcription", [])) == 1
            and len(values[0]) == 1 and "\u4e00" <= values[0] <= "\u9fff"}
    counts = Counter(refs.values())
    return {uid for uid, char in refs.items() if counts[char] == 1}


def select(rows, eligible, freq):
    available = defaultdict(dict)
    for row in rows:
        if row["overlap"] == "unflagged":
            for i, uid in enumerate(row["tokens"]):
                if uid in eligible:
                    available[uid].setdefault(row["component"], (row, i))
    taken = set()
    selected = {}
    for uid in sorted(available, key=lambda u: (freq[u], len(available[u]), order_key(u))):
        for component, obs in sorted(available[uid].items(), key=lambda x: order_key(uid, x[0])):
            if component not in taken:
                selected[uid] = [obs]
                taken.add(component)
                break
    for size in (2, 3):
        for uid, used in selected.items():
            if len(used) != size - 1:
                continue
            for component, obs in sorted(available[uid].items(), key=lambda x: order_key(uid, x[0])):
                if component not in taken:
                    used.append(obs)
                    taken.add(component)
                    break
    assert len(taken) == sum(map(len, selected.values()))
    return selected


def freeze():
    data, _ = split(SEED)
    previous = {r["component"] for seed in OLD for r in split(seed)[0]["test"]}
    data["test"] = [r for r in data["test"] if r["component"] not in previous]
    mark_duplicates(data["train"] + data["validation"], data["test"])
    mark_duplicates(data["train"], data["validation"])
    corpus = Corpus(data["train"])
    eligible = {u for u in known_classes() if 1 <= corpus.frequency[u] <= 99}
    test = select(data["test"], eligible, corpus.frequency)
    validation = select(data["validation"], eligible.difference(test), corpus.frequency)
    assert not test.keys() & validation.keys()
    return data, corpus, eligible, test, validation


def candidate_blocks(eligible, freq):
    ordered = sorted(eligible, key=lambda u: (freq[u], order_key("block", u)))
    lookup = {}
    for start in range(0, len(ordered), 16):
        block = ordered[start:start+16]
        if len(block) != 16:
            break
        outside = [u for u in ordered if u not in block]
        replacement = min(outside, key=lambda u: (abs(log(freq[u] / freq[block[7]])), order_key("swap", start, u)))
        labels = sorted([*block, replacement], key=lambda u: order_key("tie", start, u))
        for uid in block:
            lookup[uid] = (labels, [i for i, u in enumerate(labels) if u != replacement],
                           [i for i, u in enumerate(labels) if u != uid])
    return lookup


def score(items, blocks, corpus, models, ngrams):
    from blind_decipher import neural, ngram_scores
    from cipher_context_controls import contexts

    results = []
    for uid, used in items.items():
        if uid not in blocks:
            continue
        labels, present, absent = blocks[uid]
        ids = np.array([corpus.index[u] for u in labels])
        sides = list(zip(*(contexts(r["tokens"], i, set(labels)) for r, i in used), strict=True))
        gpt = np.mean([neural(m, side, corpus, ids) for m, side in zip(models, sides, strict=True)], axis=0)
        trigram = np.mean([ngram_scores(m, side, corpus, ids) for m, side in zip(ngrams, sides, strict=True)], axis=0)
        prior = np.log([corpus.frequency[u] for u in labels])
        views = {}
        gold = labels.index(uid)
        for count in range(1, len(used) + 1):
            g = gpt[:count].sum(axis=0)
            t = trigram[:count].sum(axis=0)
            correction = (count - 1) * prior
            methods = {}
            for name, vector in zip(METHODS, (prior, g, g-correction, t, t-correction), strict=True):
                p = np.array(present)[np.argsort(-vector[present], kind="stable")]
                a = np.array(absent)[np.argsort(-vector[absent], kind="stable")]
                methods[name] = dict(rank=int(np.flatnonzero(p == gold)[0])+1, pick=labels[p[0]],
                                     present_margin=float((vector[p[0]] - vector[p[1]]) / count),
                                     absent_margin=float((vector[a[0]] - vector[a[1]]) / count))
            views[count] = methods
        results.append(dict(uid=uid, frequency=corpus.frequency[uid],
                            candidates=labels, present=present, absent=absent, views=views,
                            observations=[dict(bone=r["bone"], component=r["component"], group=r["index"], position=i)
                                          for r, i in used]))
    return results


def gate(validation, method, count):
    rows = [r["views"][count][method] for r in validation if count in r["views"]]
    if not rows:
        return None
    thresholds = sorted({r["absent_margin"] for r in rows} | {r["present_margin"] for r in rows})
    safe = [t for t in thresholds if sum(r["absent_margin"] > t for r in rows) <= len(rows) // 10]
    return max(safe, key=lambda t: (sum(r["rank"] == 1 and r["present_margin"] > t for r in rows), t))


def compare(test, validation, count):
    rows = [r["views"][count] for r in test if count in r["views"]]
    results = {}
    for method in METHODS:
        trial = [r[method] for r in rows]
        threshold = gate(validation, method, count)
        results[method] = dict(top1=sum(r["rank"] == 1 for r in trial),
                               top5=sum(r["rank"] <= 5 for r in trial), threshold=threshold,
                               accepted=sum(threshold is not None and r["present_margin"] > threshold for r in trial),
                               correct_accepted=sum(threshold is not None and r["present_margin"] > threshold
                                                    and r["rank"] == 1 for r in trial),
                               false_accept=sum(threshold is not None and r["absent_margin"] > threshold for r in trial))
    paired = {}
    for k in (1, 5):
        diffs = [int(r["gpt_prior_once"]["rank"] <= k) - int(r["trigram_prior_once"]["rank"] <= k) for r in rows]
        wins, losses = diffs.count(1), diffs.count(-1)
        n = wins + losses
        p = min(1, 2 * sum(comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n)
        paired[k] = dict(gpt_only=wins, trigram_only=losses, exact_p=p)
    return dict(n=len(rows), methods=results, paired=paired)


def run(output, archive):
    from blind_decipher import load_model, reversed_rows
    from experiment import Ngram

    data, corpus, eligible, test, validation = freeze()
    from verify_cipher import read

    output.mkdir(parents=True, exist_ok=False)
    models, ngrams = [], []
    metadata = read(archive / "results.json")["training"]
    for direction, fit in enumerate((data["train"], reversed_rows(data["train"]))):
        path = archive / str(direction) / "gpt.safetensors"
        assert sha256(path.read_bytes()).hexdigest() == metadata[str(direction)]["sha256"]
        models.append(load_model(corpus, path))
        ngrams.append(Ngram(corpus, fit))
    blocks = candidate_blocks(eligible, corpus.frequency)
    evaluation = score(test, blocks, corpus, models, ngrams)
    calibration = score(validation, blocks, corpus, models, ngrams)
    cohorts = {"all": evaluation} | {f"{lo}-{hi}": [r for r in evaluation if lo <= r["frequency"] <= hi]
                                     for lo, hi in BANDS}
    results = {key: {n: compare(rows, calibration, n) for n in (1, 2, 3)} for key, rows in cohorts.items()}
    matched = {n: {m: [sum(r["views"][1][m]["rank"] <= k for r in evaluation if n in r["views"])
                      for k in (1, 5)] for m in METHODS} for n in (2, 3)}
    write(output / "results.json", dict(seed=SEED, training=metadata, test=evaluation, validation=calibration,
                                         results=results, matched_one=matched))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.output:
        if not args.archive:
            parser.error("--output requires --archive")
        run(args.output, args.archive)
    else:
        data, corpus, eligible, test, validation = freeze()
        for name, selected in (("test", test), ("validation", validation)):
            print(name, len(data[name]), len(selected), [
                [sum(lo <= corpus.frequency[u] <= hi and len(obs) >= n for u, obs in selected.items())
                 for n in (1, 2, 3)] for lo, hi in BANDS])

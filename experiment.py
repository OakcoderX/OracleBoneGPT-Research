"""Shared splits, duplicate screening, statistical baselines, and metrics."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from prepare import OUT, ordinary, write

SEEDS = (17, 29, 43)
ORDERS = {"frequency": 0, "bigram": 1, "trigram": 2, "retrieval": 8}
ALPHAS = (0.5, 2.0, 8.0, 32.0)


def split(seed):
    rows = [json.loads(line) for line in (OUT / "rows.jsonl").read_text().splitlines()]
    parent = {row["bone"]: row["bone"] for row in rows}

    def root(bone):
        while bone != parent[bone]:
            bone = parent[bone]
        return bone

    joins = OUT / "verified_joins.tsv"
    applied = []
    if joins.exists():
        with joins.open() as stream:
            for edge in csv.DictReader(stream, delimiter="\t"):
                a, b = edge["bone_a"], edge["bone_b"]
                parent.setdefault(a, a)
                parent.setdefault(b, b)
                low, high = sorted((root(a), root(b)))
                parent[high] = low
                applied.append(edge)
    result = {key: [] for key in ("train", "validation", "test")}
    for row in rows:
        row["component"] = root(row["bone"])
        value = int.from_bytes(hashlib.sha256(f"{seed}/{row['component']}".encode()).digest()[:8], "big") / 2**64
        name = "train" if value < 0.8 else "validation" if value < 0.9 else "test"
        result[name].append(row)
    assert all(
        {r["component"] for r in a}.isdisjoint(r["component"] for r in b)
        for a, b in combinations(result.values(), 2)
    )
    return result, applied


def mark_duplicates(train, test):
    def key(row):
        return tuple(t for t in row["tokens"] if ordinary(t))

    def deletions(seq):
        return {seq[:i] + seq[i+1:] for i in range(len(seq))}

    def grams(seq):
        return {seq[i:i+5] for i in range(len(seq) - 4)}

    known = {key(row) for row in train}
    five = set().union(*(grams(seq) for seq in known))
    deletion_index = set()
    replacement_index = set()
    for seq in known:
        if len(seq) >= 5:
            deletion_index.update(deletions(seq))
            replacement_index.update((i, seq[:i] + seq[i+1:]) for i in range(len(seq)))
    for row in test:
        seq = key(row)
        if seq in known:
            row["overlap"] = "exact"
        elif (len(seq) >= 5 and (
            five.intersection(grams(seq))
            or seq in deletion_index
            or any((i, seq[:i] + seq[i+1:]) in replacement_index for i in range(len(seq)))
            or (len(seq) >= 6 and any(s in known for s in deletions(seq)))
        )):
            row["overlap"] = "near"
        else:
            row["overlap"] = "unflagged"


class Corpus:
    def __init__(self, train):
        self.frequency = Counter(
            token for row in train for token in row["tokens"] if ordinary(token)
        )
        self.labels = sorted(self.frequency)
        self.index = {t: i for i, t in enumerate(self.labels)}
        special = sorted({t for row in train for t in row["tokens"] if not ordinary(t)})
        self.input_index = self.index | {
            t: i for i, t in enumerate(["@pad", "@oov", *special], len(self.labels))
        }

    def encode(self, tokens):
        return [self.input_index.get(t, self.input_index["@oov"]) for t in tokens]

    def positions(self, rows):
        for row in rows:
            for i, token in enumerate(row["tokens"][1:], 1):
                if ordinary(token):
                    yield row, i, self.index.get(token, -1)


class Ngram:
    def __init__(self, corpus, train):
        self.size = len(corpus.labels)
        self.counts = defaultdict(Counter)
        for row, position, target in corpus.positions(train):
            prefix = tuple(corpus.encode(row["tokens"][:position]))
            for n in range(min(8, len(prefix)) + 1):
                self.counts[prefix[-n:] if n else ()][target] += 1
        self.tables = {}
        for key, counter in self.counts.items():
            ids, counts = zip(*counter.items())
            self.tables[key] = (np.array(ids), np.array(counts), sum(counts))
        ids, counts, total = self.tables[()]
        self.prior = np.full(self.size, 0.5)
        self.prior[ids] += counts
        self.prior /= total + 0.5 * self.size

    def distribution(self, prefix, order, alpha):
        probs = self.prior.copy()
        for n in range(1, min(order, len(prefix)) + 1):
            table = self.tables.get(prefix[-n:])
            if table is not None:
                ids, counts, total = table
                probs *= alpha
                probs[ids] += counts
                probs /= total + alpha
        return probs

    def tune(self, corpus, validation):
        points = [
            (tuple(corpus.encode(row["tokens"][:pos])), target)
            for row, pos, target in corpus.positions(validation) if target >= 0
        ]
        return {
            method: min(ALPHAS, key=lambda alpha: sum(
                -math.log(self.distribution(prefix, order, alpha)[target]) for prefix, target in points
            )) if order else ALPHAS[0]
            for method, order in ORDERS.items()
        }


def top5(scores):
    # Explicit stable tie-break by lexicographic UID (encoded in increasing order).
    return np.argsort(-scores, axis=-1, kind="stable")[..., :5]


def prediction(indices, target, nll):
    indices = [int(i) for i in indices]
    return [int(indices[0] == target), int(target in indices), nll, indices]


def strata(row, token, corpus):
    frequency = corpus.frequency[token]
    length = len(row["tokens"])
    return [
        "all",
        row["overlap"],
        "short_2_4" if length <= 4 else "length_5_plus",
        "oov" if frequency == 0 else "rare_1_9" if frequency < 10 else "mid_10_99" if frequency < 100 else "common_100_plus",
        "clean_context" if all(map(ordinary, row["tokens"])) else "marked_context",
    ]


def summarize(records):
    result = defaultdict(lambda: defaultdict(list))
    for row in records:
        for category in row["strata"]:
            for method, (one, five, nll, _) in row["predictions"].items():
                result[category][method].append((one, five, nll))
    return {
        category: {
            method: dict(n=len(values), in_vocab=sum(v[2] is not None for v in values),
                         top1=sum(v[0] for v in values) / len(values),
                         top5=sum(v[1] for v in values) / len(values),
                         nll=float(np.mean([v[2] for v in values if v[2] is not None]))
                         if any(v[2] is not None for v in values) else None)
            for method, values in methods.items()
        }
        for category, methods in result.items()
    }


def paired_interval(records, reference, category, seed):
    by_bone = defaultdict(lambda: np.zeros(5))
    for row in records:
        if category not in row["strata"]:
            continue
        a, b = (row["predictions"][m] for m in ("gpt", reference))
        by_bone[row["component"]] += (
            1, a[0]-b[0], a[1]-b[1], int(a[2] is not None),
            0 if a[2] is None else a[2]-b[2],
        )
    counts = np.array(list(by_bone.values()))
    rng = np.random.default_rng(seed)
    totals = counts[rng.integers(len(counts), size=(2000, len(counts)))].sum(axis=1)
    deltas = totals[:, [1, 2, 4]] / totals[:, [0, 0, 3]]
    return dict(components=len(counts), top1_top5_nll_95=np.quantile(deltas, [0.025, 0.975], axis=0).T.tolist())


def run(seed, baseline_only, root):
    splits, joins = split(seed)
    train, validation, test = (splits[name] for name in ("train", "validation", "test"))
    out = root / str(seed)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {name: {r["bone"]: r["component"] for r in rows} for name, rows in splits.items()}
    write(out / "splits.json", manifest)
    corpus = Corpus(train)
    mark_duplicates(train, test)
    baseline_started = time.perf_counter()
    baseline = Ngram(corpus, train)
    alphas = baseline.tune(corpus, validation)
    valid_losses = {}
    for method, order in ORDERS.items():
        valid_losses[method] = float(np.mean([
            -math.log(baseline.distribution(tuple(corpus.encode(row["tokens"][:i])), order, alphas[method])[y])
            for row, i, y in corpus.positions(validation) if y >= 0
        ]))
    reference = min(valid_losses, key=valid_losses.get)
    records = []
    for row, i, target in corpus.positions(test):
        prefix = tuple(corpus.encode(row["tokens"][:i]))
        uid = row["tokens"][i]
        scores = {}
        for method, order in ORDERS.items():
            probs = baseline.distribution(prefix, order, alphas[method])
            scores[method] = prediction(top5(probs), target, -math.log(probs[target]) if target >= 0 else None)
        records.append(dict(bone=row["bone"], component=row["component"], group=row["group"],
                            index=row["index"], position=i, prefix=row["tokens"][:i], target=uid,
                            strata=strata(row, uid, corpus), predictions=scores))
    baseline_seconds = time.perf_counter() - baseline_started
    gpt = None
    if not baseline_only:
        from train import train_gpt
        gpt, predictions = train_gpt(corpus, train, validation, test, seed + 100, out)
        for record, result in zip(records, predictions, strict=True):
            record["predictions"]["gpt"] = result
    write(out / "vocabulary.json", dict(labels=corpus.labels, input_index=corpus.input_index))
    summary = {
        name: dict(bones=len({r["bone"] for r in rows}), components=len({r["component"] for r in rows}),
                   rows=len(rows), targets=sum(1 for _ in corpus.positions(rows)))
        for name, rows in splits.items()
    }
    with gzip.open(out / "predictions.jsonl.gz", "wt", encoding="utf-8") as stream:
        for record in records:
            print(json.dumps(record, ensure_ascii=False), file=stream)
    write(out / "results.json", dict(seed=seed, splits=summary, joins=joins,
          overlap_rows=Counter(row["overlap"] for row in test), baseline_alphas=alphas,
          baseline_seconds=baseline_seconds, baseline_validation_nll=valid_losses,
          validation_selected_baseline=reference, gpt=gpt, metrics=summarize(records),
          paired_intervals={c: paired_interval(records, reference, c, seed)
                            for c in ("all", "unflagged", "rare_1_9", "short_2_4")} if gpt else {}))
    print(seed, summarize(records)["all"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs"))
    args = parser.parse_args()
    for seed in args.seed or SEEDS:
        run(seed, args.baseline_only, args.output)

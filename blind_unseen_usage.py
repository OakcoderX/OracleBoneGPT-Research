"""Retrieve anonymous usage examples for classes removed from GPT training."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from blind_decipher import load_model, neural, reversed_rows
from experiment import Corpus, SEEDS, mark_duplicates, split
from prepare import write
from train import train_gpt

OUT = Path("用法预测结果")
RUNS = Path("experiments/unseen-usage")
REFERENCE_COUNT = 2
QUERY_COUNTS = (1, 2, 5)


def mask(rows, hidden):
    return [dict(r, tokens=["@hidden_class" if u in hidden else u for u in r["tokens"]])
            for r in rows]


def occurrences(rows, hidden):
    groups = defaultdict(dict)
    for row in rows:
        if row["overlap"] != "unflagged":
            continue
        words = row["tokens"]
        masked = ["@hidden_class" if t in hidden else t for t in words]
        for i, uid in enumerate(words):
            if uid in hidden:
                groups[uid].setdefault(row["component"], (masked[:i], masked[i + 1:][::-1]))
    return groups


def fit(corpus, fit_rows, check, seed, root, reverse):
    path = root / ("reverse" if reverse else "forward")
    path.mkdir(parents=True, exist_ok=True)
    if reverse:
        fit_rows, check = reversed_rows(fit_rows), reversed_rows(check)
    metadata, _ = train_gpt(corpus, fit_rows, check, check, seed, path)
    write(path / "training.json", metadata)
    return load_model(corpus, path / "gpt.safetensors"), metadata


def unit(matrix):
    return matrix / np.maximum(np.linalg.norm(matrix, axis=-1, keepdims=True), 1e-12)


def score(rows, corpus, forward, reverse):
    left, right = zip(*rows, strict=True)
    ids = list(range(len(corpus.labels)))
    predictions = [np.exp(neural(model, ctx, corpus, ids)) for model, ctx in
                   ((forward, left), (reverse, right))]
    model_profiles = np.concatenate(predictions, axis=1)
    width = len(corpus.input_index)
    anchors = np.zeros((len(rows), 2 * width))
    for i, pair in enumerate(rows):
        for direction, ctx in enumerate(pair):
            key = corpus.encode(ctx[-1:])[0] if ctx else corpus.input_index["@pad"]
            anchors[i, direction * width + key] = 1
    return {"gpt_usage": model_profiles, "adjacent_anchors": anchors}


def evaluate(ref, test, reference_uids, query_uids, count, seed):
    ref = ref.reshape(len(reference_uids), REFERENCE_COUNT, -1).mean(axis=1)
    start = 0
    query = []
    for uid, length in query_uids:
        query.append(test[start:start + count].mean(axis=0))
        start += length
    scores = unit(np.sqrt(query)) @ unit(np.sqrt(ref)).T
    gold = np.array([reference_uids.index(uid) for uid, _ in query_uids])
    def ranks(matrix):
        truth = matrix[np.arange(len(gold)), gold]
        return (matrix >= truth[:, None] - 1e-10).sum(axis=1)
    observed = ranks(scores)
    rng = np.random.default_rng(seed)
    shuffled = []
    for _ in range(100):
        order = rng.permutation(len(reference_uids))
        shuffled.append(int((ranks(scores[:, order]) == 1).sum()))
    return dict(n=len(observed), top1=int((observed == 1).sum()),
                top5=int((observed <= 5).sum()), mrr=float(np.mean(1 / observed)),
                shuffled_top1=float(np.mean(shuffled)))


def run(seed):
    splits, _ = split(seed)
    training = splits["train"]
    original = Corpus(training)
    hidden = {u for u, n in original.frequency.items() if 5 <= n <= 60}
    # No reference or query component is used for model selection.
    fit_rows, check = [], []
    for row in training:
        key = hashlib.sha256(str(row["component"]).encode()).digest()[0]
        (check if key < 26 else fit_rows).append(row)
    assert not {r["component"] for r in check} & {r["component"] for r in fit_rows}
    fit_rows, check = mask(fit_rows, hidden), mask(check, hidden)
    corpus = Corpus(fit_rows)
    assert not hidden & corpus.input_index.keys()
    root = RUNS / str(seed)
    forward, fmeta = fit(corpus, fit_rows, check, seed, root, False)
    reverse, rmeta = fit(corpus, fit_rows, check, seed, root, True)
    refs, test = splits["validation"], splits["test"]
    mark_duplicates(training, refs)
    mark_duplicates(training + refs, test)
    references, queries = occurrences(refs, hidden), occurrences(test, hidden)
    reference_uids = sorted(u for u in references if len(references[u]) >= REFERENCE_COUNT)
    ref_context = [ctx for u in reference_uids for ctx in list(references[u].values())[:REFERENCE_COUNT]]
    reference_scores = score(ref_context, corpus, forward, reverse)
    results = {}
    for count in QUERY_COUNTS:
        eligible = [(u, list(queries[u].items())[:count]) for u in sorted(queries)
                    if u in reference_uids and len(queries[u]) >= count]
        query_contexts = [ctx for _, items in eligible for _, ctx in items]
        query_uids = [(u, len(items)) for u, items in eligible]
        result = {}
        if eligible:
            for name, test_scores in score(query_contexts, corpus, forward, reverse).items():
                result[name] = evaluate(reference_scores[name], test_scores, reference_uids,
                                        query_uids, count, seed)
        results[count] = dict(methods=result,
                              query_components={u: [c for c, _ in items] for u, items in eligible})
    return dict(hidden_count=len(hidden), output_vocab=len(corpus.labels),
                references={u: list(references[u])[:REFERENCE_COUNT] for u in reference_uids},
                query_classes_without_reference=sum(u not in reference_uids for u in queries),
                checkpoints=dict(forward=fmeta, reverse=rmeta), results=results)


def main():
    output = {seed: run(seed) for seed in SEEDS}
    write(OUT / "训练未见字匿名辞例.json", output)
    for seed, result in output.items():
        print(seed, len(result["references"]), {k: r["methods"] for k, r in result["results"].items()})


if __name__ == "__main__":
    main()

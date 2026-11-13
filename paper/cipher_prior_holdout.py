"""Test the fixed prior correction on bone components outside earlier cipher tests."""
import argparse
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np

from blind_decipher import load_model, neural, ngram_scores, reversed_rows
from cipher_context_controls import contexts, rank
from experiment import Corpus, Ngram, mark_duplicates, split
from prepare import write
from train import train_gpt
from verify_cipher import BLIND, SEEDS, dependency_interval, read

HOLDOUT_SEED = 20261113
METHODS = ("gpt", "gpt_prior_once", "trigram", "trigram_prior_once")


def score(seed, data, corpus, models, ngrams):
    reveal = read(BLIND / f"{seed}-reveal.json")
    labels = [r["uid"] for r in reveal["candidate_order"]]
    hidden = set(labels)
    known = np.array([uid in corpus.index for uid in labels])
    ids = np.array([corpus.index[uid] for uid, present in zip(labels, known, strict=True) if present])
    prior = np.log([max(1, corpus.frequency[uid]) for uid in labels])
    seen = set()
    for row in data["train"] + data["validation"]:
        for i, uid in enumerate(row["tokens"]):
            if uid in hidden and 0 < i < len(row["tokens"]) - 1:
                left, right = contexts(row["tokens"], i, hidden)
                seen.add((uid, left[-1], right[-1]))
    examples = defaultdict(dict)
    for row in data["test"]:
        if row["overlap"] == "unflagged":
            for i, uid in enumerate(row["tokens"]):
                if uid in hidden:
                    examples[uid].setdefault(row["component"], (row, i))
    selected = [(r, list(examples[r["uid"]].values())[:5]) for r in reveal["candidate_order"]]
    selected = [(r, obs) for r, obs in selected if obs]
    examples = [(row, i) for _, obs in selected for row, i in obs]
    sides = list(zip(*(contexts(row["tokens"], i, hidden) for row, i in examples), strict=True))
    # A fixed-list UID outside the training vocabulary has no model score.
    gpt = np.full((len(examples), len(labels)), -np.inf)
    trigram = np.full_like(gpt, -np.inf)
    gpt[:, known] = np.mean([neural(model, side, corpus, ids)
                             for model, side in zip(models, sides, strict=True)], axis=0)
    trigram[:, known] = np.mean([ngram_scores(model, side, corpus, ids)
                                 for model, side in zip(ngrams, sides, strict=True)], axis=0)
    result = []
    start = 0
    for label, obs in selected:
        uid = label["uid"]
        stop = start + len(obs)
        pairs = [(sides[0][i][-1], sides[1][i][-1]) for i in range(start, stop)
                 if sides[0][i] and sides[1][i]]
        ranks = {}
        for count in sorted({1, len(obs)}):
            g = gpt[start:start+count].sum(axis=0)
            t = trigram[start:start+count].sum(axis=0)
            correction = (count-1) * prior
            ranks[count] = dict(zip(METHODS, (rank(x, labels.index(uid)) for x in
                                             (g, g-correction, t, t-correction)), strict=True))
        result.append(dict(uid=uid, bin=label["bin"], frequency=corpus.frequency[uid],
                           observations=[dict(bone=r["bone"], component=r["component"], index=r["index"], position=i)
                                         for r, i in obs], both_sides=len(pairs),
                           seen_pairs=sum((uid, *pair) in seen for pair in pairs), ranks=ranks))
        start = stop
    assert start == len(examples)
    return result, [uid for uid, present in zip(labels, known, strict=True) if not present]


def summarize(rows):
    counts = {}
    for n in (1, 5):
        complete = [r for r in rows if 5 in r["ranks"]]
        counts[n] = {m: [sum(r["ranks"][n][m] <= k for r in complete) for k in (1, 5)] for m in METHODS}
    complete = [r for r in rows if 5 in r["ranks"]]
    intervals = {}
    if complete:
        for baseline in ("gpt", "trigram_prior_once"):
            intervals[baseline] = dependency_interval([
                ([("uid", r["uid"]), *(("bone", o["component"]) for o in r["observations"])],
                 int(r["ranks"][5]["gpt_prior_once"] <= 5) - int(r["ranks"][5][baseline] <= 5))
                for r in complete])
    return dict(observed=len(rows), five=len(complete), counts=counts, intervals=intervals)


def run(output):
    old = {r["component"] for seed in SEEDS for r in split(seed)[0]["test"]}
    data, _ = split(HOLDOUT_SEED)
    data["test"] = [r for r in data["test"] if r["component"] not in old]
    assert {r["component"] for r in data["test"]}.isdisjoint(old)
    mark_duplicates(data["train"] + data["validation"], data["test"])
    assert sum(r["overlap"] == "unflagged" for r in data["test"]) == 835
    corpus = Corpus(data["train"])
    output.mkdir(parents=True, exist_ok=False)
    metadata, models, ngrams = {}, [], []
    for name, seed in (("forward", HOLDOUT_SEED), ("reverse", HOLDOUT_SEED + 900)):
        train, val = data["train"], data["validation"]
        if name == "reverse":
            train, val = reversed_rows(train), reversed_rows(val)
        directory = output / name
        directory.mkdir()
        info, _ = train_gpt(corpus, train, val, [], seed, directory)
        path = directory / "gpt.safetensors"
        metadata[name] = info | {"sha256": sha256(path.read_bytes()).hexdigest()}
        models.append(load_model(corpus, path))
        ngrams.append(Ngram(corpus, train))
    results = {}
    for seed in SEEDS:
        rows, unseen = score(seed, data, corpus, models, ngrams)
        groups = dict(all=rows, no_seen_immediate_pair=[r for r in rows if r["both_sides"] and not r["seen_pairs"]])
        groups.update({label: [r for r in rows if r["bin"] == label] for label in ("common", "mid", "rare")})
        summaries = {name: summarize(group) for name, group in groups.items()}
        results[seed] = dict(summaries=summaries, rows=rows, training_oov_candidates=unseen)
        print(seed, summaries["all"], flush=True)
    assert [results[s]["summaries"]["all"]["five"] for s in SEEDS] == [24, 29, 27]
    write(output / "results.json", dict(split_seed=HOLDOUT_SEED, training=metadata, results=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

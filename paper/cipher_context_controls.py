"""Paired, exploratory context controls for the frozen 64-candidate cipher task."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from blind_decipher import load_model, neural, ngram_scores, reversed_rows
from experiment import Corpus, Ngram, split
from prepare import CONTEXT, write
from verify_cipher import BLIND, ROOT, SEEDS, dependency_interval, read


def weights(path, manifest, key):
    expected = dict(line.split()[::-1] for line in manifest.read_text().splitlines())
    assert sha256(path.read_bytes()).hexdigest() == expected[key], path
    return path


def contexts(tokens, position, hidden):
    coded = ["@oov" if t in hidden else t for t in tokens]
    return coded[:position][-CONTEXT:], coded[position+1:][::-1][-CONTEXT:]


def rank(scores, target):
    return int(np.flatnonzero(np.argsort(-scores, kind="stable") == target)[0]) + 1


def run(seed, archive):
    data, _ = split(seed)
    corpus = Corpus(data["train"])
    assert read(ROOT / f"runs/{seed}/splits.json") == {
        name: {r["bone"]: r["component"] for r in rows} for name, rows in data.items()
    }
    reveal = read(BLIND / f"{seed}-reveal.json")
    observations = read(BLIND / f"{seed}-blind_inputs.json")
    labels = [x["uid"] for x in reveal["candidate_order"]]
    hidden = set(labels)
    ids = np.array([corpus.index[uid] for uid in labels])
    alias = set(reveal["truth"])
    selected = [r for r in reveal["results"] if r["cohort"] == "unflagged"
                and r["budget"] == 5 and r["n_contexts"] == 5]
    used = sorted({i for row in selected for i in row["observations"]})
    index = {i: j for j, i in enumerate(used)}
    sides = list(zip(*(contexts(observations[i]["coded"], observations[i]["position"], alias)
                       for i in used), strict=True))
    seen = set()
    for row in data["train"] + data["validation"]:
        tokens = row["tokens"]
        for i, uid in enumerate(tokens):
            if uid in hidden:
                left, right = contexts(tokens, i, hidden)
                if left and right:
                    seen.add((uid, left[-1], right[-1]))

    full = []
    local = []
    trigram = []
    shuffled = []
    mutable = np.zeros(len(used), dtype=bool)
    for direction, side in enumerate(sides):
        name = f"{seed}/reverse/gpt.safetensors"
        path = (weights(archive / name, BLIND / "运行文件SHA256.txt", f"experiments/blind/run-001/{name}")
                if direction else weights(ROOT / f"runs/{seed}/gpt.safetensors", ROOT / "runs/权重校验.txt",
                                          f"runs/{seed}/gpt.safetensors"))
        model = load_model(corpus, path)
        full.append(neural(model, side, corpus, ids))
        nearest = [ctx[-2:] for ctx in side]
        local.append(neural(model, nearest, corpus, ids))
        rows = reversed_rows(data["train"]) if direction else data["train"]
        trigram.append(ngram_scores(Ngram(corpus, rows), nearest, corpus, ids))
        # Shuffle ordinary words beyond the nearest two on each side. Keep length,
        # side, special markers and the nearest two positions fixed.
        positions = [[j for j, t in enumerate(ctx[:-2]) if not t.startswith("@")] for ctx in side]
        mutable |= [len({ctx[j] for j in js}) >= 2 for ctx, js in zip(side, positions, strict=True)]
        rng = np.random.default_rng(seed * 2 + direction)
        permutations = []
        for _ in range(64):
            batch = []
            for ctx, js in zip(side, positions, strict=True):
                changed = ctx.copy()
                if len(js) > 1:
                    for j, token in zip(js, rng.permutation([ctx[j] for j in js]), strict=True):
                        changed[j] = str(token)
                batch.append(changed)
            permutations.append(neural(model, batch, corpus, ids))
        shuffled.append(np.array(permutations))
    prior = np.log([corpus.frequency[uid] for uid in labels])
    full = np.mean(full, axis=0)
    local = np.mean(local, axis=0)
    trigram = np.mean(trigram, axis=0)
    shuffled = np.mean(shuffled, axis=0)
    first = {r["code"]: r for r in reveal["results"] if r["cohort"] == "unflagged" and r["budget"] == 1}
    output = []
    for row in selected:
        uid = row["gold_uid"]
        target = labels.index(uid)
        ii = [index[i] for i in row["observations"]]
        assert first[row["code"]]["observations"] == row["observations"][:1]
        score = full[ii].sum(axis=0)
        ranks = dict(full_one=rank(full[ii[0]], target), full_five=rank(score, target),
                     local_one=rank(local[ii[0]], target), local_five=rank(local[ii].sum(axis=0), target),
                     trigram_one=rank(trigram[ii[0]], target), trigram_five=rank(trigram[ii].sum(axis=0), target),
                     prior_only=rank(prior, target), prior_once=rank(score - (len(ii)-1) * prior, target),
                     trigram_prior_once=rank(trigram[ii].sum(axis=0) - (len(ii)-1) * prior, target))
        assert ranks["full_one"] == first[row["code"]]["metrics"]["gpt_both"]["rank"]
        assert ranks["full_five"] == row["metrics"]["gpt_both"]["rank"]
        assert ranks["trigram_five"] == row["metrics"]["trigram_both"]["rank"]
        pairs = [(sides[0][i][-1], sides[1][i][-1]) for i in ii if sides[0][i] and sides[1][i]]
        output.append(dict(seed=seed, uid=uid, frequency=corpus.frequency[uid], bin=row["bin"], ranks=ranks,
                           bones=[observations[i]["component"] for i in row["observations"]],
                           shuffled_ranks=[rank(s, target) for s in shuffled[:, ii].sum(axis=1)],
                           mutable=int(mutable[ii].sum()), both_sides=len(pairs),
                           unique_pairs=len(set(pairs)), seen_pairs=sum((uid, *pair) in seen for pair in pairs)))
    return output


def summarize(rows):
    counts = {}
    for name in rows[0]["ranks"]:
        counts[name] = [sum(r["ranks"][name] <= k for r in rows) for k in (1, 5)]
    perm = np.array([r["shuffled_ranks"] for r in rows])
    counts["permuted_mean"] = [(perm <= k).sum(axis=0).mean().item() for k in (1, 5)]
    intervals = {}
    for a, b in [("full_five", "local_five"), ("full_five", "trigram_five"),
                 ("full_five", "prior_once"), ("local_five", "local_one")]:
        intervals[f"{a}-{b}"] = dependency_interval([
            ([("uid", r["uid"]), *(("bone", bone) for bone in r["bones"])],
             int(r["ranks"][a] <= 5) - int(r["ranks"][b] <= 5)) for r in rows])
    return dict(n=len(rows), counts=counts, intervals=intervals,
                permuted_top5_range=np.quantile((perm <= 5).sum(axis=0), [.025, .975]).tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for seed in SEEDS for row in run(seed, args.archive)]
    groups = dict(all=rows, movable=[r for r in rows if r["mutable"]],
                  no_seen_immediate_pair=[r for r in rows if r["both_sides"] and not r["seen_pairs"]])
    groups.update({str(seed): [r for r in rows if r["seed"] == seed] for seed in SEEDS})
    groups.update({label: [r for r in rows if r["bin"] == label] for label in ("common", "mid", "rare")})
    summaries = {name: summarize(group) for name, group in groups.items() if group}
    write(args.output, dict(summaries=summaries, rows=rows))
    for name, summary in summaries.items():
        print(name, summary)

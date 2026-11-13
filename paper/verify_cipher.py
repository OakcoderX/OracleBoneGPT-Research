"""Check the manuscript's anonymous-glyph counts from archived trial records."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BLIND = ROOT / "盲猜结果"
SEEDS = (17, 29, 43)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dependency_interval(records):
    """Group paired trials connected by any shared target or known bone component."""
    parents = list(range(len(records)))

    def root(index):
        while parents[index] != index:
            index = parents[index]
        return index

    owners = {}
    for i, (keys, _) in enumerate(records):
        for key in keys:
            other = owners.setdefault(key, i)
            parents[root(i)] = root(other)
    groups = defaultdict(list)
    for i, (_, delta) in enumerate(records):
        groups[root(i)].append(delta)
    values = np.array([(sum(g), len(g)) for g in groups.values()])
    rng = np.random.default_rng(1913)
    samples = values[rng.integers(len(values), size=(10000, len(values)))].sum(axis=1)
    interval = (100 * np.quantile(samples[:, 0] / samples[:, 1], [.025, .975])).tolist()
    return len(groups), max(map(len, groups.values())), interval


def verify_recovery():
    matched = []
    paired = []
    coverage = 0
    for seed in SEEDS:
        reveal = read(BLIND / f"{seed}-reveal.json")
        observations = read(BLIND / f"{seed}-blind_inputs.json")
        truth = reveal["truth"]
        assert len(truth) == len(set(truth.values())) == 64
        hidden = set(truth.values())
        for item in observations:
            assert not hidden.intersection(item["coded"])
            assert item["coded"][item["position"]] == item["code"]
        trials = {(r["cohort"], r["code"], r["budget"]): r for r in reveal["results"]}
        assert len(trials) == len(reveal["results"])
        for (cohort, code, budget), row in trials.items():
            used = [observations[i] for i in row["observations"]]
            assert row["gold_uid"] == truth[code]
            assert len({o["component"] for o in used}) == len(used) == row["n_contexts"]
            assert all(o["code"] == code for o in used)
            if cohort == "unflagged" and budget == 5:
                coverage += 1
                if len(used) == 5:
                    first = trials[(cohort, code, 1)]
                    assert first["observations"] == row["observations"][:1]
                    matched.append((first, row))
                    keys = [("uid", row["gold_uid"]), *(("bone", o["component"]) for o in used)]
                    paired.append((keys, row["metrics"]["gpt_both"]["top5"] -
                                   first["metrics"]["gpt_both"]["top5"]))
    assert coverage == 141 and len(matched) == 86
    counts = {
        "one_context_gpt": sum(a["metrics"]["gpt_both"]["top5"] for a, _ in matched),
        "five_context_gpt": sum(b["metrics"]["gpt_both"]["top5"] for _, b in matched),
        "five_context_trigram": sum(b["metrics"]["trigram_both"]["top5"] for _, b in matched),
        "five_context_shuffled": sum(b["metrics"]["shuffled_context"]["top5"] for _, b in matched),
    }
    assert tuple(counts.values()) == (49, 66, 58, 10)
    print("64 classes; 86 matched trials; top-five counts:", counts)
    print("One vs five, connected dependency groups:", dependency_interval(paired))


def verify_whole_class_holdout():
    data = read(ROOT / "匿名密码字袋对照结果.json")
    prior = ROOT / "匿名密码难候选结果.json"
    assert sha256(prior.read_bytes()).hexdigest() == data["previous_sha256"]
    expected = {17: (157, 57, 54), 29: (144, 60, 58), 43: (131, 52, 59)}
    pooled = []
    for seed in SEEDS:
        run = data["results"][str(seed)]
        rows = run["rows"]
        groups = defaultdict(lambda: [0, 0])
        gpt = bag = 0
        for row in rows:
            assert len(set(row["decoys"])) == 15 and row["uid"] not in row["decoys"]
            assert all(1 <= rank <= 16 for rank in row["ranks"].values())
            a = int(row["ranks"]["ordered/ordered"] <= 5)
            b = int(row["ranks"]["sided_idf"] <= 5)
            pooled.append(([("uid", row["uid"]), ("bone", row["component"])], a-b))
            gpt += a
            bag += b
            counts = groups[row["component"]]
            counts[0] += a-b
            counts[1] += 1
        assert (len(rows), gpt, bag) == expected[seed]
        values = np.asarray(list(groups.values()))
        rng = np.random.default_rng(902)
        draws = values[rng.integers(len(values), size=(5000, len(values)))].sum(axis=1)
        interval = 100 * np.quantile(draws[:, 0] / draws[:, 1], [.025, .975])
        np.testing.assert_allclose(interval, run["gpt_minus_bag_pp95"]["sided_idf"], atol=1e-9)
        assert interval[0] < 0 < interval[1]
        print(f"{seed}: {len(rows)} trials, GPT {gpt}, weighted bag {bag}, interval {interval.tolist()}")

    print("GPT vs bag, pooled descriptive contrast:", sum(delta for _, delta in pooled), len(pooled),
          dependency_interval(pooled))


if __name__ == "__main__":
    verify_recovery()
    verify_whole_class_holdout()

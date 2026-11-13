"""Train and score single- and double-position masked oracle bone infill."""
import argparse
import gzip
import hashlib
import json
import math
import platform
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from experiment import ALPHAS, SEEDS, Corpus, mark_duplicates, split, top5
from prepare import CONTEXT, ordinary, write
from train import BATCH, LR, GPT, nll

STEPS = 3000
CHECK_EVERY = 300
LEFT = ((0, 0), (1, 0), (2, 0), (3, 0))
BOTH = ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3))
METHODS = {"frequency": LEFT[:1], "left_lookup": LEFT, "both_lookup": BOTH}
MASK = "@infill_mask"


@dataclass
class Window:
    row: dict
    start: int
    width: int

    @property
    def tokens(self):
        return self.row["tokens"]

    @property
    def target(self):
        return self.tokens[self.start:self.start + self.width]

    def hidden(self):
        return self.tokens[:self.start] + [MASK] * self.width + self.tokens[self.start + self.width:]


def windows(rows, real=False):
    for row in rows:
        tokens = row["tokens"]
        for width in (1, 2):
            for start in range(1, len(tokens) - width):
                end = start + width
                if not (ordinary(tokens[start - 1]) and ordinary(tokens[end])):
                    continue
                if real:
                    accept = all(t.startswith("@missing:") for t in tokens[start:end])
                else:
                    accept = all(map(ordinary, tokens[start:end]))
                if accept:
                    yield Window(row, start, width)


def arrays(examples, corpus, real=False):
    x = np.full((len(examples), CONTEXT), corpus.input_index["@pad"], dtype=np.int32)
    positions = np.zeros((len(examples), 2), dtype=np.int32)
    y = np.full_like(positions, -1)
    lengths = []
    for i, example in enumerate(examples):
        tokens = example.hidden()
        lengths.append(len(tokens))
        x[i, :len(tokens)] = corpus.encode(tokens)
        positions[i] = [example.start, example.start + example.width - 1]
        if not real:
            y[i, :example.width] = [corpus.index.get(t, -1) for t in example.target]
    return x, positions, y, np.array(lengths)


def batch(data, indices):
    x, positions, y, lengths = data
    length = min(CONTEXT, math.ceil(int(max(lengths[indices])) / 8) * 8)
    return mx.array(x[indices, :length]), mx.array(positions[indices]), mx.array(y[indices])


class Infill(GPT):
    def __init__(self, corpus, bidirectional):
        super().__init__(len(corpus.input_index), len(corpus.labels))
        self.bidirectional = bidirectional
        self.pad = corpus.input_index["@pad"]

    def __call__(self, ids, positions):
        x = self.embedding(ids) + self.position(mx.arange(ids.shape[1]))
        # In causal mode padding only occurs after every scored position.
        mask = (ids != self.pad)[:, None, None, :] if self.bidirectional else "causal"
        for block in self.blocks:
            x = block(x, mask)
        at_gap = x[mx.arange(ids.shape[0])[:, None], positions]
        return self.norm(at_gap) @ self.embedding.weight[:self.classes].T


def evaluation(model, data):
    total, count = 0.0, 0
    for start in range(0, len(data[0]), BATCH):
        x, pos, y = batch(data, slice(start, start + BATCH))
        valid = y >= 0
        total += mx.sum(mx.where(valid, nll(model(x, pos), y), 0)).item()
        count += mx.sum(valid).item()
    return total / count


def fit(corpus, train, validation, seed, path, bidirectional):
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = Infill(corpus, bidirectional)
    mx.eval(model.parameters())
    initial_hash = hashlib.sha256(b"".join(
        np.array(v).tobytes() for _, v in tree_flatten(model.parameters())
    )).hexdigest()
    optimizer = optim.AdamW(learning_rate=LR, weight_decay=0.01)

    def loss(net, x, pos, y):
        valid = y >= 0
        return mx.sum(mx.where(valid, nll(net(x, pos), y), 0)) / mx.sum(valid)

    loss_grad = nn.value_and_grad(model, loss)
    batches = []
    history = []
    best = float("inf")
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        if not batches:
            indices = rng.permutation(len(train[0]))
            batches = list(np.array_split(indices, math.ceil(len(indices) / BATCH)))
        x, pos, y = batch(train, batches.pop())
        value, grad = loss_grad(model, x, pos, y)
        grad, _ = optim.clip_grad_norm(grad, 1.0)
        optimizer.update(model, grad)
        mx.eval(model.parameters(), optimizer.state, value)
        if step % CHECK_EVERY == 0:
            valid = evaluation(model, validation)
            assert math.isfinite(valid)
            history.append({"step": step, "validation_nll": valid})
            if valid < best:
                best, best_step = valid, step
                model.save_weights(str(path))
            print(seed, "both" if bidirectional else "left", step, round(valid, 4), flush=True)
    model.load_weights(str(path))
    model.eval()
    return model, dict(parameters=sum(v.size for _, v in tree_flatten(model.parameters())),
                       initial_weights_sha256=initial_hash, selected_step=best_step,
                       seconds=time.perf_counter() - started, history=history)


class Lookup:
    def __init__(self, corpus, examples):
        self.corpus = corpus
        counts = defaultdict(Counter)
        for example in examples:
            left, right = self.context(example)
            for i, uid in enumerate(example.target):
                for key in self.keys(left, right, example.width, i, set(LEFT + BOTH)):
                    counts[key][corpus.index[uid]] += 1
        self.tables = {
            key: (np.array(list(c)), np.array(list(c.values())), c.total())
            for key, c in counts.items()
        }

    def context(self, example):
        # Keep real OOV UID distinct for matching, rather than equating two unknown UIDs.
        return tuple(example.tokens[:example.start]), tuple(example.tokens[example.start + example.width:])

    @staticmethod
    def keys(left, right, width, offset, ladder):
        for a, b in ladder:
            if len(left) >= a and len(right) >= b:
                yield width, offset, left[-a:] if a else (), right[:b]

    def scores(self, example, offset, ladder, alpha):
        left, right = self.context(example)
        prior, *levels = self.keys(left, right, example.width, offset, ladder)
        size = len(self.corpus.labels)
        ids, counts, n = self.tables[prior]
        probs = np.full(size, 0.5)
        probs[ids] += counts
        probs /= n + 0.5 * size
        for key in levels:
            if key in self.tables:
                ids, counts, n = self.tables[key]
                probs *= alpha
                probs[ids] += counts
                probs /= n + alpha
        return probs

    def tune(self, examples):
        points = [
            (ex, i, self.corpus.index[uid])
            for ex in examples for i, uid in enumerate(ex.target) if uid in self.corpus.index
        ]
        return {name: min(ALPHAS, key=lambda alpha: sum(
            -math.log(self.scores(ex, i, ladder, alpha)[target]) for ex, i, target in points
        )) if len(ladder) > 1 else 0.5 for name, ladder in METHODS.items()}


def predict_model(model, data):
    for start in range(0, len(data[0]), BATCH):
        x, pos, y = batch(data, slice(start, start + BATCH))
        logits = model(x, pos)
        indices = top5(np.array(logits))
        losses = np.array(nll(logits, y))
        yield from zip(indices, losses, strict=True)


def pack(indices, loss, target):
    guesses = list(map(int, indices))
    return [int(guesses[0] == target), int(target in guesses),
            float(loss) if target >= 0 else None, guesses]


def measure(records, seed):
    groups = defaultdict(list)
    for record in records:
        groups["all"].append(record)
        groups[f'width_{record["width"]}'].append(record)
        groups[record["overlap"]].append(record)
        if 1 <= record["frequency"] <= 9:
            groups["rare"].append(record)
        if record["frequency"] == 0:
            groups["oov"].append(record)
    names = (*METHODS, "left_model", "both_model")
    stats, intervals = {}, {}
    for label, items in groups.items():
        stats[label] = {}
        for method in names:
            values = [r["predictions"][method] for r in items]
            known = [v[2] for v in values if v[2] is not None]
            stats[label][method] = dict(
                n=len(values), in_vocab=len(known), top1=np.mean([v[0] for v in values]).item(),
                top5=np.mean([v[1] for v in values]).item(),
                nll=np.mean(known).item() if known else None,
            )
        if label in ("all", "unflagged", "width_1", "width_2"):
            intervals[label] = {}
            for reference in ("both_lookup", "left_model"):
                by_bone = defaultdict(lambda: np.zeros(5))
                for r in items:
                    a, b = (r["predictions"][m] for m in ("both_model", reference))
                    by_bone[r["component"]] += [1, a[0]-b[0], a[1]-b[1],
                        int(a[2] is not None), 0 if a[2] is None else a[2]-b[2]]
                counts = np.array(list(by_bone.values()))
                rng = np.random.default_rng(seed)
                totals = np.array([counts[rng.integers(len(counts), size=len(counts))].sum(axis=0)
                                   for _ in range(2000)])
                delta = totals[:, [1, 2, 4]] / totals[:, [0, 0, 3]]
                intervals[label][reference] = np.quantile(delta, [0.025, 0.975], axis=0).T.tolist()
    full_windows = defaultdict(list)
    for r in records:
        full_windows[(r["bone"], r["index"], r["start"], r["width"])].append(r)
    exact = {}
    for width in (1, 2):
        chunks = [chunk for key, chunk in full_windows.items() if key[-1] == width]
        exact[width] = dict(n=len(chunks), **{
            method: np.mean([all(r["predictions"][method][0] for r in c) for c in chunks]).item()
            for method in names
        })
    return stats, intervals, exact


def run(seed, directory):
    started = time.perf_counter()
    out = directory / str(seed)
    out.mkdir(parents=True, exist_ok=False)
    rows, _ = split(seed)
    reference = json.loads((Path("runs") / str(seed) / "splits.json").read_text())
    assert reference == {k: {r["bone"]: r["component"] for r in v} for k, v in rows.items()}
    mark_duplicates(rows["train"], rows["test"])
    corpus = Corpus(rows["train"])
    corpus.input_index[MASK] = len(corpus.input_index)
    samples = {name: list(windows(part)) for name, part in rows.items()}
    encoded = {name: arrays(part, corpus) for name, part in samples.items()}
    lookup = Lookup(corpus, samples["train"])
    alpha = lookup.tune(samples["validation"])
    models, log = {}, {}
    for name, both in (("left_model", False), ("both_model", True)):
        models[name], log[name] = fit(corpus, encoded["train"], encoded["validation"],
                                      seed + 200, out / f"{name}.safetensors", both)
    assert log["left_model"]["initial_weights_sha256"] == log["both_model"]["initial_weights_sha256"]
    test = samples["test"]
    predictions = {name: predict_model(model, encoded["test"]) for name, model in models.items()}
    records = []
    for ex in test:
        neural = {name: next(iterator) for name, iterator in predictions.items()}
        for i, uid in enumerate(ex.target):
            target = corpus.index.get(uid, -1)
            result = {name: pack(top[i], loss[i], target) for name, (top, loss) in neural.items()}
            for name, ladder in METHODS.items():
                probs = lookup.scores(ex, i, ladder, alpha[name])
                result[name] = pack(top5(probs), -math.log(probs[target]) if target >= 0 else None, target)
            records.append(dict(bone=ex.row["bone"], component=ex.row["component"], index=ex.row["index"],
                                group=ex.row["group"], start=ex.start, width=ex.width, offset=i,
                                overlap=ex.row["overlap"], target=uid, frequency=corpus.frequency[uid],
                                predictions=result))
    stats, intervals, exact = measure(records, seed)
    with gzip.open(out / "predictions.jsonl.gz", "wt", encoding="utf-8") as stream:
        for record in records:
            print(json.dumps(record), file=stream)
    # A small fixed sample for expert review. Prefer maximal explicit gaps.
    real = [ex for ex in windows(rows["test"], real=True)
            if not any(t == "@order_gap" for t in ex.tokens)]
    selected = []
    for width in (1, 2):
        unique = {}
        for ex in real:
            if ex.width == width:
                unique.setdefault(ex.row["bone"], ex)
        selected.extend(list(unique.values())[:4])
    candidates = []
    if selected:
        for ex, (indices, _) in zip(selected, predict_model(
                models["both_model"], arrays(selected, corpus, real=True)), strict=True):
            candidates.append(dict(bone=ex.row["bone"], group=ex.row["group"], index=ex.row["index"],
                start=ex.start, width=ex.width, tokens=ex.hidden(), original_marks=ex.target,
                model=[[corpus.labels[j] for j in indices[i]] for i in range(ex.width)],
                lookup=[[corpus.labels[j] for j in top5(lookup.scores(ex, i, BOTH, alpha["both_lookup"]))]
                        for i in range(ex.width)]))
    write(out / "real_gap_candidates.json", dict(eligible=len(real), cases=candidates))
    write(out / "vocabulary.json", dict(labels=corpus.labels, input_index=corpus.input_index))
    write(out / "results.json", dict(seed=seed, initialization_seed=seed + 200, steps=STEPS,
        hardware=mx.device_info()["device_name"], device=str(mx.default_device()), mlx=version("mlx"),
        numpy=np.__version__, python=platform.python_version(), macos=platform.mac_ver()[0],
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        data_sha256=hashlib.sha256(Path("source/data.json").read_bytes()).hexdigest(),
        dataset={key: dict(windows=dict(Counter(ex.width for ex in values)),
                           positions=sum(ex.width for ex in values)) for key, values in samples.items()},
        tuning=alpha, models=log, metrics=stats, intervals=intervals, exact_window=exact,
        elapsed_seconds=time.perf_counter() - started))
    print(seed, stats["all"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", choices=SEEDS, action="append", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for seed in args.seed or SEEDS:
        run(seed, args.output)

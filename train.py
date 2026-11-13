"""A small randomly initialized causal Transformer for Apple MLX."""
import math
import platform
import subprocess
import time
from importlib.metadata import version

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from experiment import prediction, top5
from prepare import CONTEXT

WIDTH = 64
HEADS = 4
LAYERS = 2
BATCH = 64
STEPS = 2000
VALID_EVERY = 250
LR = 0.002


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm = nn.LayerNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.project = nn.Linear(WIDTH, WIDTH)
        self.mlp_norm = nn.LayerNorm(WIDTH)
        self.up = nn.Linear(WIDTH, 2 * WIDTH)
        self.down = nn.Linear(2 * WIDTH, WIDTH)

    def __call__(self, x, mask="causal"):
        b, t, _ = x.shape
        q, k, v = self.qkv(self.attention_norm(x)).reshape(
            b, t, 3, HEADS, WIDTH // HEADS
        ).transpose(2, 0, 3, 1, 4)
        attention = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=(WIDTH // HEADS) ** -0.5, mask=mask
        )
        x = x + self.project(attention.transpose(0, 2, 1, 3).reshape(b, t, WIDTH))
        return x + self.down(nn.gelu(self.up(self.mlp_norm(x))))


class GPT(nn.Module):
    def __init__(self, inputs, classes):
        super().__init__()
        self.classes = classes
        self.embedding = nn.Embedding(inputs, WIDTH)
        self.position = nn.Embedding(CONTEXT, WIDTH)
        self.blocks = [Block() for _ in range(LAYERS)]
        self.norm = nn.LayerNorm(WIDTH)
        self.embedding.weight = 0.02 * mx.random.normal(self.embedding.weight.shape)
        self.position.weight = 0.02 * mx.random.normal(self.position.weight.shape)

    def hidden(self, ids):
        x = self.embedding(ids) + self.position(mx.arange(ids.shape[1]))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def __call__(self, ids):
        return self.hidden(ids) @ self.embedding.weight[:self.classes].T


def arrays(rows, corpus):
    x = np.full((len(rows), CONTEXT), corpus.input_index["@pad"], dtype=np.int32)
    y = np.full_like(x, -1)
    lengths = []
    for i, row in enumerate(rows):
        tokens = row["tokens"]
        n = len(tokens) - 1
        x[i, :n] = corpus.encode(tokens[:-1])
        y[i, :n] = [corpus.index.get(t, -1) for t in tokens[1:]]
        lengths.append(n)
    return x, y, np.array(lengths)


def batch(data, indices):
    x, y, lengths = data
    length = min(CONTEXT, math.ceil(int(max(lengths[indices])) / 8) * 8)
    return mx.array(x[indices, :length]), mx.array(y[indices, :length])


def nll(logits, y):
    return nn.losses.cross_entropy(logits, mx.maximum(y, 0), reduction="none")


def validate(model, data):
    total, count = 0, 0
    for start in range(0, len(data[0]), BATCH):
        x, y = batch(data, slice(start, start + BATCH))
        valid = y >= 0
        total += mx.sum(mx.where(valid, nll(model(x), y), 0)).item()
        count += mx.sum(valid).item()
    return total / count


def predict(model, data, rows, corpus):
    results = []
    for start in range(0, len(rows), BATCH):
        x, y = batch(data, slice(start, start + BATCH))
        logits = model(x)
        losses = np.array(nll(logits, y))
        mask = np.array(y) >= 0
        for j, row in enumerate(rows[start:start + BATCH]):
            for _, i, target in corpus.positions([row]):
                scores = np.array(logits[j, i-1])
                assert bool(mask[j, i-1]) == (target >= 0)
                results.append(prediction(top5(scores), target, float(losses[j, i-1]) if target >= 0 else None))
    return results


def train_gpt(corpus, train, validation, test, seed, out, *, input_transform=None, make_training_arrays=arrays):
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    mask_rng = np.random.default_rng(seed + 2000) if input_transform is not None else None
    model = GPT(len(corpus.input_index), len(corpus.labels))
    mx.eval(model.parameters())
    params = sum(value.size for _, value in tree_flatten(model.parameters()))
    optimizer = optim.AdamW(learning_rate=LR, weight_decay=0.01)

    def loss(model, x, y):
        valid = y >= 0
        return mx.sum(mx.where(valid, nll(model(x), y), 0)) / mx.sum(valid)

    loss_and_grad = nn.value_and_grad(model, loss)
    train_data = make_training_arrays(train, corpus)
    valid_data = make_training_arrays(validation, corpus)
    batches = []
    steps = []
    best_loss, best_step = float("inf"), None
    started = time.perf_counter()
    model.train()
    for step in range(1, STEPS + 1):
        if not batches:
            indices = rng.permutation(len(train))
            batches = list(np.array_split(indices, math.ceil(len(indices) / BATCH)))
        x, y = batch(train_data, batches.pop())
        if input_transform is not None:
            x = input_transform(x, mask_rng)
        loss_value, gradients = loss_and_grad(model, x, y)
        gradients, _ = optim.clip_grad_norm(gradients, 1.0)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss_value)
        if step % VALID_EVERY == 0:
            model.eval()
            valid_loss = validate(model, valid_data)
            assert math.isfinite(valid_loss)
            elapsed = time.perf_counter() - started
            steps.append(dict(step=step, batch_nll=loss_value.item(), validation_nll=valid_loss, elapsed_seconds=elapsed))
            print(f"seed {seed}: step {step}  validation NLL {valid_loss:.4f}  {elapsed:.1f}s", flush=True)
            if valid_loss < best_loss:
                best_loss, best_step = valid_loss, step
                model.save_weights(str(out / "gpt.safetensors"))
            model.train()
    seconds = time.perf_counter() - started
    model.load_weights(str(out / "gpt.safetensors"))
    model.eval()
    metadata = dict(
        seed=seed, parameters=params, context=CONTEXT, width=WIDTH, layers=LAYERS,
        heads=HEADS, max_batch=BATCH, steps=STEPS, lr=LR, optimizer="AdamW",
        dtype="float32", training_and_validation_seconds=seconds,
        selected_step=best_step, validation_nll=best_loss, checkpoints=steps,
        device=str(mx.default_device()), hardware=subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(), mlx=version("mlx"), numpy=np.__version__, python=platform.python_version(),
        macos=platform.mac_ver()[0], gpu=mx.device_info()["device_name"],
    )
    return metadata, predict(model, arrays(test, corpus), test, corpus)

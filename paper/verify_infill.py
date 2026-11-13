"""Replay masked-position checkpoints and verify the paper's infill results."""
import argparse
import gzip
from hashlib import file_digest
import json
from pathlib import Path
from statistics import mean

import numpy as np

from experiment import SEEDS, Corpus, mark_duplicates, split
from infill import Infill, MASK, arrays, measure, pack, predict_model, windows

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "阶段二结果"
DEFAULT_ARCHIVE = ROOT.parent / "甲骨文GPT-阶段二-双向补字"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(archive):
    checks = REPORT / "运行文件SHA256.txt"
    for line in checks.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        with (archive / name).open("rb") as file:
            assert file_digest(file, "sha256").hexdigest() == expected, name
    directory = archive / "experiments/phase2-infill/run-001"
    results = []
    for seed in SEEDS:
        run = directory / str(seed)
        result = read(run / "results.json")
        assert result == read(REPORT / f"seed-{seed}.json")
        with gzip.open(run / "predictions.jsonl.gz", "rt", encoding="utf-8") as file:
            records = [json.loads(line) for line in file]
        data, _ = split(seed)
        mark_duplicates(data["train"], data["test"])
        corpus = Corpus(data["train"])
        corpus.input_index[MASK] = len(corpus.input_index)
        assert read(run / "vocabulary.json") == dict(labels=corpus.labels, input_index=corpus.input_index)
        test = list(windows(data["test"]))
        encoded = arrays(test, corpus)
        for method, both in (("left_model", False), ("both_model", True)):
            model = Infill(corpus, both)
            model.load_weights(str(run / f"{method}.safetensors"))
            model.eval()
            observed = iter(records)
            for example, (top, loss) in zip(test, predict_model(model, encoded), strict=True):
                for i, uid in enumerate(example.target):
                    saved = next(observed)
                    row = example.row
                    assert (saved["bone"], saved["index"], saved["start"], saved["width"], saved["offset"]) == (
                        row["bone"], row["index"], example.start, example.width, i)
                    assert saved["target"] == uid and saved["component"] == row["component"]
                    assert saved["overlap"] == row["overlap"] and saved["frequency"] == corpus.frequency[uid]
                    expected = saved["predictions"][method]
                    current = pack(top[i], loss[i], corpus.index.get(uid, -1))
                    assert current[:2] == expected[:2] and current[3] == expected[3]
                    if expected[2] is None:
                        assert current[2] is None
                    else:
                        np.testing.assert_allclose(current[2], expected[2], rtol=1e-6, atol=1e-6)
            assert next(observed, None) is None
        metrics, intervals, exact = measure(records, seed)
        assert metrics == result["metrics"]
        assert intervals == result["intervals"]
        assert {str(k): v for k, v in exact.items()} == result["exact_window"]
        results.append(metrics)
        print(f"{seed}: replay verified {len(records)} masked positions")
    expected = {"left_lookup": ("28.69%", "52.32%"), "both_lookup": ("35.16%", "58.40%"),
                "both_model": ("27.35%", "47.50%")}
    for method, values in expected.items():
        computed = tuple(f"{mean(r['all'][method][k] for r in results):.2%}" for k in ("top1", "top5"))
        assert computed == values
        print(method, *computed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    verify(parser.parse_args().archive)

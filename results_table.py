"""Export comparable results and a small, deterministic case gallery."""
import argparse
import csv
import gzip
import json
import statistics
import unicodedata
from pathlib import Path

from experiment import ORDERS, SEEDS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--runs", type=Path, default=Path("runs"))
RUNS = parser.parse_args().runs
METHODS = {
    "frequency": "高频字",
    "bigram": "前一字",
    "trigram": "前两字",
    "retrieval": "训练辞例后缀",
    "gpt": "小 GPT",
}
STRATA = {
    "all": "全部",
    "unflagged": "未触发重复筛查",
    "exact": "精确重复",
    "near": "近重复",
    "short_2_4": "短句（2–4 个位置）",
    "length_5_plus": "5 个位置及以上",
    "common_100_plus": "常见字（训练出现 ≥100 次）",
    "mid_10_99": "中频字（10–99 次）",
    "rare_1_9": "少见字（1–9 次）",
    "oov": "训练未见字",
    "clean_context": "整句无异常标记",
    "marked_context": "句中有异常标记",
}
runs = {seed: json.loads((RUNS / str(seed) / "results.json").read_text()) for seed in SEEDS}


def average(group, method):
    return {
        metric: statistics.mean(run["metrics"][group][method][metric] for run in runs.values())
        if runs[SEEDS[0]]["metrics"][group][method][metric] is not None else None
        for metric in ("top1", "top5", "nll")
    }


def cell(metrics):
    nll = f'{metrics["nll"]:.3f}' if metrics["nll"] is not None else "—"
    return f'{metrics["top1"]:.2%} / {metrics["top5"]:.2%} / {nll}'


with (RUNS / "对照结果.csv").open("w", encoding="utf-8-sig", newline="") as out:
    writer = csv.writer(out)
    writer.writerow(["split_seed", "stratum", "method", "positions", "in_vocab", "top1", "top5", "nll_nats"])
    for seed, run in runs.items():
        for category in STRATA:
            for method, m in run["metrics"][category].items():
                writer.writerow([seed, category, method, m["n"], m["in_vocab"], m["top1"], m["top5"], m["nll"]])

lines = ["# 下一字预测对照结果", "", "每格为 Top-1 / Top-5 / NLL。三次均值等权，测试材料可能跨次重叠。", "",
         "| 方法 | 种子 17 | 种子 29 | 种子 43 | 三次均值 |", "|---|---:|---:|---:|---:|"]
for method, label in METHODS.items():
    cells = [cell(run["metrics"]["all"][method]) for run in runs.values()]
    lines.append("| " + " | ".join([label, *cells, cell(average("all", method))]) + " |")
lines += ["", "## 分层结果", "",
          "| 分层 | 三次样本数 | 训练辞例后缀 | GPT |", "|---|---:|---:|---:|"]
for category, label in STRATA.items():
    sizes = " / ".join(str(r["metrics"][category]["gpt"]["n"]) for r in runs.values())
    lines.append(f'| {label} | {sizes} | {cell(average(category, "retrieval"))} | {cell(average(category, "gpt"))} |')
lines += ["", "## GPT 减去验证集选定基线的差值", "",
          "每次以缀合分组为重抽样单位，配对重抽 2,000 次。命中率差用百分点，损失差用 nats。区间为该次划分内的描述性 95% 区间。", "",
          "| 分层 | 种子 | Top-1 差及区间 | Top-5 差及区间 | NLL 差及区间 |", "|---|---:|---:|---:|---:|"]
for category in ("all", "unflagged", "rare_1_9", "short_2_4"):
    for seed, run in runs.items():
        g = run["metrics"][category]["gpt"]
        b = run["metrics"][category][run["validation_selected_baseline"]]
        intervals = run["paired_intervals"][category]["top1_top5_nll_95"]
        cells = []
        for metric, (lo, hi) in zip(("top1", "top5", "nll"), intervals):
            scale = 1 if metric == "nll" else 100
            cells.append(f"{(g[metric]-b[metric])*scale:+.3f} [{lo*scale:+.3f}, {hi*scale:+.3f}]")
        lines.append("| " + " | ".join([STRATA[category], str(seed), *cells]) + " |")
(RUNS / "对照结果.md").write_text("\n".join(lines) + "\n")


seed_dir = RUNS / "17"
labels = json.loads((seed_dir / "vocabulary.json").read_text())["labels"]
meta = json.loads(Path("source/Main-character.json").read_bytes())
with gzip.open(seed_dir / "predictions.jsonl.gz", "rt") as f:
    records = [json.loads(line) for line in f]


def describe(uid):
    if uid.startswith("@"):
        special = {"@order_gap": "读序缺号", "@unlabeled": "有框无字类"}
        return f"{uid}（{special.get(uid, '缺位或异常标记')}）"
    refs = [
        "".join(f"U+{ord(c):04X}" if unicodedata.category(c) == "Co" else c for c in ref)
        for ref in meta.get(uid, {}).get("transcription", [])
    ]
    return f'{uid}（{"／".join(refs) or "无平台释读"}）'


def pick(match):
    return next(r for r in records if match(r))


def one(row, model):
    return row["predictions"][model][0]


cases = [
    ("未触发重复筛查，GPT 首选命中、检索首选失手",
     pick(lambda r: "unflagged" in r["strata"] and len(r["prefix"]) >= 3 and one(r, "gpt") and not one(r, "retrieval"))),
    ("未触发重复筛查，检索首选命中、GPT 首选失手",
     pick(lambda r: "unflagged" in r["strata"] and len(r["prefix"]) >= 3 and one(r, "retrieval") and not one(r, "gpt"))),
    ("训练仅出现 1–9 次，两种方法均未进前五",
     pick(lambda r: "rare_1_9" in r["strata"] and not any(r["predictions"][m][1] for m in ("gpt", "retrieval")))),
    ("训练词表未见目标，无法命中",
     pick(lambda r: "oov" in r["strata"])),
]
lines = ["# 真实预测案例", "", "固定种子 17。按原文件记录顺序，选每个预定条件下首个案例。括号内是平台参考释读，可缺失、可能有多个，未做古文字专家复核。专用字库的私用区字符显示为 U+ 编码。评估按 UID。位置和原始句组索引从 0 起。"]
for title, row in cases:
    lines.extend([
        "", f'## {title}', "",
        f'- {row["bone"]} · {row["group"]} · 原始组索引 {row["index"]} · 目标位置 {row["position"]}',
        "- 已见前文 " + " → ".join(map(describe, row["prefix"])),
        "- OBIMD 标注目标 " + describe(row["target"]),
    ])
    for method in ("retrieval", "gpt"):
        guesses = [labels[i] for i in row["predictions"][method][3]]
        lines.append(f"- {METHODS[method]} 前五 " + "、".join(map(describe, guesses)))
(RUNS / "预测案例.md").write_text("\n".join(lines) + "\n")

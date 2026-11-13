"""Export reproducible Phase 2 result tables and separate expert review cases."""
import argparse
import gzip
import hashlib
import json
import statistics
import unicodedata

from pathlib import Path

from experiment import SEEDS
from prepare import BOXES, GROUPS, write

NAMES = dict(frequency="字频", left_lookup="左侧检索", both_lookup="双侧检索",
             left_model="单侧模型", both_model="双向模型")
GROUPS_DISPLAY = dict(all="全部", width_1="单字", width_2="双字", unflagged="未触发重复筛查",
                      exact="精确重复", near="近重复", rare="少见字（1–9 次）", oov="训练未见字")


def cell(m):
    nll = f'{m["nll"]:.3f}' if m["nll"] is not None else "—"
    return f'{m["top1"]:.2%} / {m["top5"]:.2%} / {nll}'


def mean(rows):
    return {k: statistics.mean(row[k] for row in rows) if rows[0][k] is not None else None
            for k in ("top1", "top5", "nll")}


def render(root, out):
    out.mkdir(parents=True, exist_ok=True)
    runs = [json.loads((root / str(seed) / "results.json").read_text()) for seed in SEEDS]
    lines = ["# 双向补字对照结果", "", "每格为逐位置 Top-1 / Top-5 / 词表内 NLL。三次结果等权平均。",
             "同一原始位置可进入多个不同遮挡窗口，因此样本数按遮挡后的预测位置统计。", "",
             "| 方法 | 种子 17 | 种子 29 | 种子 43 | 均值 |", "|---|---:|---:|---:|---:|"]
    for method, name in NAMES.items():
        values = [r["metrics"]["all"][method] for r in runs]
        lines.append("| " + " | ".join([name, *map(cell, values), cell(mean(values))]) + " |")
    lines.extend(["", "## 分层", "", "| 分层 | 各轮目标数 | 双侧检索 | 单侧模型 | 双向模型 |",
                  "|---|---:|---:|---:|---:|"])
    for key, title in GROUPS_DISPLAY.items():
        ms = [r["metrics"][key] for r in runs]
        sizes = " / ".join(str(m["both_model"]["n"]) for m in ms)
        values = [cell(mean([m[method] for m in ms])) for method in ("both_lookup", "left_model", "both_model")]
        lines.append("| " + " | ".join([title, sizes, *values]) + " |")
    lines.extend(["", "## 全窗首选命中", "",
                  "| 缺口长度 | 各轮窗口数 | 字频 | 左侧检索 | 双侧检索 | 单侧模型 | 双向模型 |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for width in ("1", "2"):
        ms = [r["exact_window"][width] for r in runs]
        sizes = " / ".join(str(m["n"]) for m in ms)
        lines.append("| " + " | ".join([width, sizes, *(
            f"{statistics.mean(m[name] for m in ms):.2%}" for name in NAMES
        )]) + " |")
    lines.extend(["", "## 配对 95% 区间", "",
                  "以缀合连通组为抽样单位，重抽 2,000 次。前两列为百分点，最后一列为 nats。",
                  "", "| 分层 | 种子 | 对照 | 双向模型 Top-1 差 | Top-5 差 | NLL 差 |",
                  "|---|---:|---|---:|---:|---:|"])
    for key in ("all", "unflagged", "width_1", "width_2"):
        for r in runs:
            for method in ("both_lookup", "left_model"):
                a = r["metrics"][key]["both_model"]
                b = r["metrics"][key][method]
                cells = []
                for metric, (lo, hi) in zip(("top1", "top5", "nll"), r["intervals"][key][method]):
                    scale = 1 if metric == "nll" else 100
                    cells.append(f"{(a[metric]-b[metric])*scale:+.3f} [{lo*scale:+.3f}, {hi*scale:+.3f}]")
                lines.append("| " + " | ".join([GROUPS_DISPLAY[key], str(r["seed"]), NAMES[method], *cells]) + " |")
    lines.extend(["", "## 缺口位置补充分析", "",
                  "按缺口中心相对句长的前三分之一、中三分之一、后三分之一分层。此表在主实验后追加。",
                  "", "| 位置 | 各轮目标数 | 双侧检索 | 单侧模型 | 双向模型 |",
                  "|---|---:|---:|---:|---:|"])
    rows = [json.loads(line) for line in Path("prepared/rows.jsonl").read_text().splitlines()]
    lengths = {(row["bone"], row["index"]): len(row["tokens"]) for row in rows}
    by_seed = []
    for seed in SEEDS:
        bins = [[], [], []]
        with gzip.open(root / str(seed) / "predictions.jsonl.gz", "rt") as stream:
            for line in stream:
                r = json.loads(line)
                length = lengths[r["bone"], r["index"]]
                part = int(3 * (r["start"] + r["width"] / 2) / length)
                bins[part].append(r["predictions"])
        by_seed.append(bins)
    for part, title in enumerate(("靠前", "居中", "靠后")):
        sizes = " / ".join(str(len(bins[part])) for bins in by_seed)
        cells = []
        for method in ("both_lookup", "left_model", "both_model"):
            results = []
            for bins in by_seed:
                values = [r[method] for r in bins[part]]
                results.append(dict(top1=statistics.mean(v[0] for v in values),
                                    top5=statistics.mean(v[1] for v in values),
                                    nll=statistics.mean(v[2] for v in values if v[2] is not None)))
            cells.append(cell(mean(results)))
        lines.append("| " + " | ".join([title, sizes, *cells]) + " |")
    (out / "结果对照.md").write_text("\n".join(lines) + "\n")

    raw = {x["RubbingName"]: x for x in json.loads(Path("source/data.json").read_bytes())}
    meta = json.loads(Path("source/Main-character.json").read_bytes())

    def describe(uid):
        if uid.startswith("@"):
            return "□" if uid == "@infill_mask" else uid
        refs = ["".join(f"U+{ord(c):X}" if unicodedata.category(c) in {"Co", "Cn"} else c for c in ref)
                for ref in meta.get(uid, {}).get("transcription", [])]
        return uid + ("（" + "／".join(refs) + "）" if refs else "")

    files = [json.loads((root / str(seed) / "real_gap_candidates.json").read_text()) for seed in SEEDS]
    lines = ["# 真实缺损候选 · 待专家复核", "",
             "以下位置是 OBIMD 的显式缺位框。框数和读序沿用原标注。候选按位置独立排序，双字候选未给出成对排序。括号为平台参考释读，可能缺失或多值。图像路径对应 OBIMD 数据包成员。",
             "复核可登记图上有无残笔、缺位数是否可信、可接受的多个 UID、证据所在位置及仍不能确定的原因。",
             "", "各轮合格窗口数 " + " / ".join(str(f["eligible"]) for f in files) + "。同一缺口可能出现在不同划分的测试集。"]
    seen = set()
    for seed, file in zip(SEEDS, files):
        for case in file["cases"]:
            key = tuple(case[k] for k in ("bone", "index", "start", "width"))
            if key in seen:
                continue
            seen.add(key)
            bone = raw[case["bone"]]
            group = bone[GROUPS][case["index"]]
            positions = range(case["start"], case["start"] + case["width"])
            boxes = [b["Position"] for b in sorted(group[BOXES], key=lambda b: b["OrderNumber"])
                     if b["OrderNumber"] in positions]
            lines.extend(["", f'## {case["bone"]} · {case["group"]} · 缺口 {case["start"]}（零起）',
                          "", f'- 种子 {seed}，组索引 {case["index"]}，缺位框 {case["width"]} 个。',
                          f'- 拓片 `{bone["Rubbing"]}`，摹本 `{bone["Facsimile"]}`，缺位框坐标 {" / ".join(boxes)}。',
                          "- 上下文 " + " → ".join(map(describe, case["tokens"]))])
            for method, label in (("lookup", "双侧检索"), ("model", "双向模型")):
                for i, predictions in enumerate(case[method], 1):
                    lines.append(f"- {label} · 缺位 {i} " + "、".join(map(describe, predictions)))
            lines.append("- 专家复核　待记录")
    (out / "真实缺损候选.md").write_text("\n".join(lines) + "\n")
    for seed, r in zip(SEEDS, runs):
        write(out / f"seed-{seed}.json", r)
    checksums = []
    for path in sorted(root.glob("*/*")):
        checksums.append(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + str(path))
    (out / "运行文件SHA256.txt").write_text("\n".join(checksums) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.run, args.output)

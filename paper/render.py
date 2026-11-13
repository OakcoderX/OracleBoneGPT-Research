"""Rebuild the archived manuscript and paired-results figure."""
import argparse
from html import escape
import json
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
MANUSCRIPT = ROOT / "甲骨文GPT论文初稿.md"
RESULTS = ROOT / "常见字图文复验结果.json"
TITLE = "把甲骨文当作密码之后"
SECTION = "#### 3.4 外部原拓说明哪些证据值得回查"
MAC_CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def results_figure(report):
    panels = (("原拓图库", "rubbing"), ("摹本图库", "facsimile"))
    methods = (("纯字形", "visual"), ("统计图文", "stats_fusion"), ("GPT 图文", "gpt_fusion"))
    assert report["classes"] == 378
    lines = ['<figure><div class="chart"><div></div><b>正确框命中前五</b><b>邻框也列出目标字</b>']
    for label, domain in panels:
        lines.append(f'<div class="chart-section">{label}</div>')
        summary = report["domains"][domain]["added"]["summary"]
        for name, method in methods:
            result = summary[method]
            assert result["n"] == 80
            lines.append(f"<span>{name}</span>")
            for key, color in (("top5", "blue"), ("wrong_frame_top5", "rust")):
                n = result[key]
                assert 0 <= n <= 20
                lines.append(f'<div class="track"><i class="{color}" style="width:{n * 5}%"></i><span>{n}</span></div>')
    lines.append(
        "</div><figcaption>图 1　固定 378 类图库。每种方法均测试同一批 80 个留出骨片组。"
        "邻框指标只追踪原题目标字类。</figcaption></figure>"
    )
    return "\n".join(lines)


def manuscript_html(english=False):
    if english:
        body = subprocess.run(["pandoc", "--from=markdown", "--to=html5"],
                              input=(ROOT / "OracleBoneGPT-EN.md").read_text(encoding="utf-8"),
                              encoding="utf-8", capture_output=True, check=True).stdout
        css = Path(__file__).with_name("print.css").read_text(encoding="utf-8")
        return ('<html lang="en"><meta charset="UTF-8"><title>Treating Oracle Bone Inscriptions as a Cipher</title>'
                f"<style>{css}</style><body>{body}</body></html>")
    source = MANUSCRIPT.read_text(encoding="utf-8")
    if source.count(SECTION) != 1:
        raise ValueError("找不到唯一的图表插入位置")
    report = json.loads(RESULTS.read_text(encoding="utf-8"))
    body = source.replace(SECTION, results_figure(report) + "\n\n" + SECTION)
    references = {}

    def local_reference(match):
        name, href = match.groups()
        if "://" in href:
            return match.group(0)
        if not (ROOT / href).is_file():
            raise FileNotFoundError(href)
        number = references.setdefault(href, len(references) + 1)
        return f"{name}〔附{number}〕"

    body = re.sub(r"\[([^\]\n]+)\]\(([^)\n]+)\)", local_reference, body)
    body += "\n### 附录　实验记录索引\n\n"
    body += "\n".join(f"{number}. `{path}`" for path, number in references.items()) + "\n"
    html = subprocess.run(["pandoc", "--from=markdown", "--to=html5"], input=body, encoding="utf-8",
                          capture_output=True, check=True).stdout
    css = Path(__file__).with_name("print.css").read_text(encoding="utf-8")
    return (f'<!doctype html><html lang="zh-CN"><head><meta charset="UTF-8">'
            f"<title>{escape(TITLE)}</title><style>{css}</style></head><body>{html}</body></html>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Path for a new PDF")
    parser.add_argument("--chrome", default=str(MAC_CHROME) if MAC_CHROME.is_file() else shutil.which("chromium"))
    parser.add_argument("--english", action="store_true")
    args = parser.parse_args()
    if args.chrome is None:
        parser.error("需要 Chrome 或 Chromium，请用 --chrome 指定路径")
    output = args.output.resolve()
    if output.exists():
        parser.error("输出已存在，请指定新文件名")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="oracle-paper-") as tmp:
        cache = Path(tmp)
        html = cache / "paper.html"
        html.write_text(manuscript_html(args.english), encoding="utf-8")
        subprocess.run([
            args.chrome, "--headless", "--no-first-run", "--disable-gpu", "--disable-background-networking",
            f"--user-data-dir={cache / 'chrome'}", "--no-pdf-header-footer",
            f"--print-to-pdf={output}", html.as_uri(),
        ], check=True)


if __name__ == "__main__":
    main()

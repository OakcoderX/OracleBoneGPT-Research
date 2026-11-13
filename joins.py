"""Extract the published OBID-ACR join groups; use only catalogue membership."""
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path

from prepare import OUT, SOURCE, write

path = Path("join-source/rejoining_data.json")
raw = path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "a085353223cd6a91cb03858ac48a17931fae60f45d665bd05f629fd9e8372856"
cases = json.loads(raw)
bones = {r["RubbingName"] for r in json.loads((SOURCE / "data.json").read_bytes())}
modeled = {json.loads(line)["bone"] for line in (OUT / "rows.jsonl").read_text().splitlines()}
edges = []
for i, case in enumerate(cases, 1):
    names = sorted(set(case["RubbingName"]))
    assert len(names) == case["RubbingCount"]
    for a, b in combinations(names, 2):
        edges.append((a, b, f"OBID-ACR:17749727:rejoining_data.json:{i}"))
with (OUT / "verified_joins.tsv").open("w") as stream:
    writer = csv.writer(stream, delimiter="\t")
    writer.writerow(("bone_a", "bone_b", "source"))
    writer.writerows(edges)
all_names = {n for case in cases for n in case["RubbingName"]}
write(OUT / "joins_audit.json", dict(
    cases=len(cases), unique_source_names=len(all_names), matched_raw_names=len(all_names & bones),
    matched_modeled_names=len(all_names & modeled), absent_raw_names=sorted(all_names - bones),
    modeled_cases_with_at_least_two=sum(len(set(c["RubbingName"]) & modeled) >= 2 for c in cases),
    edges=len(edges), edges_with_both_modeled=sum(a in modeled and b in modeled for a, b, _ in edges),
))

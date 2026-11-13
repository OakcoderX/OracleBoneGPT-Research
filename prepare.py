"""Audit pinned OBIMD annotations, preserving one record per source group."""
import hashlib
import json
import re
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

SOURCE = Path("source")
OUT = Path("prepared")
GROUPS = "RecordUtilSentenceGroupVoList"
BOXES = "RecordUtilOracleCharVoList"
CONTEXT = 80


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def ordinary(token):
    return not token.startswith("@")


def token(box):
    if box["SeatFont"]:
        return f"@missing:{box['Mark']}"
    if box["Mark"] != -1:
        return f"@mark:{box['Mark']}"
    return box["Label"] or "@unlabeled"


def prepare():
    OUT.mkdir(exist_ok=True)
    raw = (SOURCE / "data.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == "b504b0d4e7a0126d494c161f5445c5ee4225659ff5e94182685fce35d261aa19"
    bones = json.loads(raw)
    meta = json.loads((SOURCE / "Main-character.json").read_bytes())
    counts = Counter()
    types = Counter()
    statuses = Counter()
    sentence_statuses = Counter()
    plans = defaultdict(Counter)
    uids, subuids = set(), set()
    parents = defaultdict(set)
    groups_by_bone = defaultdict(list)
    rows = []
    lengths = Counter()
    examples = defaultdict(list)

    def add_plan(plan, sequence, bone):
        target_count = sum(map(ordinary, sequence[1:]))
        if target_count:
            plans[plan]["sequences"] += 1
            plans[plan]["symbols"] += len(sequence)
            plans[plan]["scored_targets"] += target_count
            groups_by_bone[plan].append(bone)

    for bone in bones:
        name = bone["RubbingName"]
        counts["bones"] += 1
        seen_groups = set()
        seen_boxes = set()
        for index, group in enumerate(bone[GROUPS]):
            category = group["GroupCategory"]
            boxes = group[BOXES]
            key = re.sub(r"\d+$", "", category)
            types[key] += 1
            counts["reused_group_categories"] += category in seen_groups
            seen_groups.add(category)
            for box in boxes:
                counts["boxes"] += 1
                status = f"{box['SeatFont']},{box['Mark']},{bool(box['Label'])}"
                statuses[status] += 1
                if key == "InscriptionSentence":
                    sentence_statuses[status] += 1
                signature = tuple(sorted(box.items()))
                counts["duplicate_boxes_same_record"] += signature in seen_boxes
                seen_boxes.add(signature)
                if box["Label"]:
                    uids.add(box["Label"])
                    subuids.add(box["SubLabel"])
                    parents[box["SubLabel"]].add(box["Label"])
                    counts["different_subuid"] += box["SubLabel"] != box["Label"]
                    counts["labeled_with_reference"] += bool(meta.get(box["Label"], {}).get("transcription"))
            if key != "InscriptionSentence":
                continue
            orders = [b["OrderNumber"] for b in boxes]
            counts["sentence_order_unsorted"] += orders != sorted(orders)
            if not orders or any(type(i) is not int or i < 0 for i in orders) or len(set(orders)) != len(orders):
                counts["sentence_excluded_order"] += 1
                if len(examples["excluded_order"]) < 5:
                    examples["excluded_order"].append([name, category, orders])
                continue
            counts["sentence_starts_after_zero"] += min(orders) > 0
            counts["sentence_internal_order_gap"] += max(orders) - min(orders) + 1 > len(orders)
            sequence = ["@order_gap"] * (max(orders) + 1)
            for box in boxes:
                sequence[box["OrderNumber"]] = token(box)
            assert len(sequence) <= CONTEXT
            lengths[len(sequence)] += 1
            if all(map(ordinary, sequence)):
                add_plan("complete_regular", sequence, name)
            for valid, run in groupby(sequence, ordinary):
                if valid:
                    add_plan("contiguous_regular", list(run), name)
            add_plan("marked_context", sequence, name)
            if any(map(ordinary, sequence[1:])):
                rows.append(dict(bone=name, group=category, index=index, tokens=sequence))

    for plan, names in groups_by_bone.items():
        plans[plan]["bones"] = len(set(names))
    keys = Counter(tuple(t for t in row["tokens"] if ordinary(t)) for row in rows)
    class_counts = Counter(t for row in rows for t in row["tokens"] if ordinary(t))
    write(OUT / "audit.json", dict(
        counts=counts, group_types=types, statuses=statuses, sentence_statuses=sentence_statuses,
        unique_rubbing_names=len({b["RubbingName"] for b in bones}),
        unique_rubbing_paths=len({b["Rubbing"] for b in bones}),
        unique_facsimile_paths=len({b["Facsimile"] for b in bones}),
        main_uids=len(uids), sub_uids=len(subuids - {None}), cross_parent_subuids=sum(len(p) > 1 for p in parents.values()),
        metadata_entries=len(meta), observed_uids_in_metadata=len(uids & meta.keys()),
        observed_uids_with_reference=sum(bool(meta.get(uid, {}).get("transcription")) for uid in uids),
        observed_uids_multiple_references=sum(len(meta.get(uid, {}).get("transcription", [])) > 1 for uid in uids),
        cleaning=plans, retained_length_histogram=lengths, examples=examples,
        regular_uids_in_model_rows=len(class_counts), regular_uid_singletons=sum(n == 1 for n in class_counts.values()),
        most_common=class_counts.most_common(15), normalized_unique_sequences=len(keys),
        normalized_duplicate_types=sum(n > 1 for n in keys.values()),
        normalized_rows_in_duplicate_types=sum(n for n in keys.values() if n > 1),
    ))
    with (OUT / "rows.jsonl").open("w") as stream:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False), file=stream)


if __name__ == "__main__":
    prepare()

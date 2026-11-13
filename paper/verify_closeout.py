"""Verify the final order-intervention and abstention counts without training."""
from collections import Counter

import numpy as np

from verify_cipher import ROOT, read


def main():
    result = read(ROOT / "整类未见字序与弃答结果.json")
    previous = read(ROOT / "整类未见新骨片结果.json")
    rows = result["rows"]
    assert len(rows) == 90
    assert Counter(r["block"] for r in rows) == Counter(r["block"] for r in previous["rows"])
    assert len({r["query"] for r in rows}) == len({r["uid"] for r in rows}) == 90
    assert sum(r["methods"]["gpt"]["rank"] <= 5 for r in rows) == 40
    for mode, expected in (("near", 40.46875), ("all", 38.609375)):
        ranks = np.array([r[mode]["ranks"] for r in rows])
        assert ranks.shape == (90, 64)
        assert float((ranks <= 5).sum(axis=0).mean()) == expected
    gate = result["equal_size"]["methods"]["gpt"]["score"]
    test = [r["methods"]["gpt"] for r in rows if r["block"] % 3 != 0]
    threshold = gate["threshold"]
    assert len(test) == 60 and sum(r["block"] % 3 == 0 for r in rows) == 30
    assert sum(r["present_score"] > threshold and r["rank"] == 1 for r in test) == 1
    assert sum(r["absent_score"] > threshold for r in test) == 0
    absent = result["absent_rows"]
    assert len({r["query"] for r in absent}) == len(absent) == 38
    assert not {r["query"] for r in absent} & {r["query"] for r in rows}
    assert sum(r["methods"]["gpt"]["score"] > threshold for r in absent) == 4
    print("Verified: 90 paired ranks, 64 shuffles, and 60 paired plus 38 reference-absent trials.")


if __name__ == "__main__":
    main()

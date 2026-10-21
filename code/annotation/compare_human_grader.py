"""
Compare human outcome labels with grader v1 (logged) and grader v2 verdicts.

This is a human-vs-automatic comparison, reported separately from
inter-annotator agreement.

Usage (from the repository root):
    python -m code.annotation.compare_human_grader \
        --labels supporting_files/agreement/round1/per_item_labels.csv \
        --regrade data/processed/regrade_v2.csv \
        --out supporting_files/agreement/round1/human_vs_grader.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import cohen_kappa_score


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", type=Path, required=True, help="per_item_labels.csv from agreement.py")
    p.add_argument("--regrade", type=Path, required=True, help="regrade_v2.csv from grader_v2.py")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ids", nargs=2, default=["A1", "A2"])
    a = p.parse_args()

    regrade = {r["run_id"]: r for r in csv.DictReader(a.regrade.open(encoding="utf-8"))}
    rows = []
    for r in csv.DictReader(a.labels.open(encoding="utf-8")):
        g = regrade[r["run_id"]]
        v1 = "success" if g["answer_correct_v1"] == "True" else "failure"
        v2 = "success" if g["answer_correct_v2"] == "True" else "failure"
        h = {i: r[f"outcome_{i}"] for i in a.ids}
        rows.append({
            "run_id": r["run_id"], "task_id": r["task_id"], "evaluation_rule": g["evaluation_rule"],
            **{f"human_{i}": h[i] for i in a.ids},
            "grader_v1": v1, "grader_v2": v2, "v2_flags": g["flags_v2"],
            "humans_vs_v1_conflict": any(x != v1 for x in h.values()),
            "humans_vs_v2_conflict": any(x != v2 for x in h.values()),
            "gold_answer": g["gold_answer"], "final_answer": g["final_answer"],
        })
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {"n_items": len(rows)}
    for i in a.ids:
        for g in ("grader_v1", "grader_v2"):
            hx = [r[f"human_{i}"] for r in rows]
            gx = [r[g] for r in rows]
            summary[f"{i}_vs_{g}"] = {
                "agree": sum(x == y for x, y in zip(hx, gx)),
                "kappa": round(float(cohen_kappa_score(hx, gx)), 4),
            }
    summary["items_conflicting_with_v1"] = sum(r["humans_vs_v1_conflict"] for r in rows)
    summary["items_conflicting_with_v2"] = sum(r["humans_vs_v2_conflict"] for r in rows)
    a.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

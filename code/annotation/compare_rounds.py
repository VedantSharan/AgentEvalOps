"""
Agreement before (round 1, guidelines v1) vs after (round 2, guidelines v2).

Reads the outputs of code/annotation/agreement.py for both rounds and writes one
comparison table.  Also computes an EXPLORATORY, post-hoc agreement on a coarser
4-class taxonomy (candidate for a v3 revision); it is clearly labelled as such
because it was defined after seeing the round-2 disagreements.

Caveat printed in the output: round 2 was deliberately enriched with failures
(about 60% vs 16-20% in round 1), so kappa is compared at different prevalences;
PABAK and raw agreement are shown alongside.

Usage (from the repository root):
    python -m code.annotation.compare_rounds
Outputs:
    supporting_files/agreement/pre_post_comparison.csv
    supporting_files/agreement/pre_post_comparison.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

COARSE = {
    "no_failure": "no_failure",
    "wrong_tool": "tool_use_error",
    "malformed_arguments": "tool_use_error",
    "hallucinated_result": "reasoning_error",
    "correct_process_with_wrong_answer": "reasoning_error",
    "loop_or_repetition": "control_flow_error",
    "premature_stop": "control_flow_error",
}


def coarse_kappa(per_item: Path) -> dict:
    rows = list(csv.DictReader(per_item.open(encoding="utf-8")))
    a = [COARSE[r["primary_failure_category_A1"]] for r in rows]
    b = [COARSE[r["primary_failure_category_A2"]] for r in rows]
    both_fail = [(x, y) for x, y in zip(a, b) if x != "no_failure" and y != "no_failure"]
    k_bf = cohen_kappa_score([x for x, _ in both_fail], [y for _, y in both_fail]) if len({*a, *b}) > 2 and both_fail else None
    return {"kappa_all": round(float(cohen_kappa_score(a, b)), 4),
            "agree_all": f"{sum(x == y for x, y in zip(a, b))}/{len(a)}",
            "kappa_both_failure": None if k_bf is None else round(float(k_bf), 4),
            "agree_both_failure": f"{sum(x == y for x, y in both_fail)}/{len(both_fail)}"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--round1", type=Path, default=Path("supporting_files/agreement/round1"))
    p.add_argument("--round2", type=Path, default=Path("supporting_files/agreement/round2"))
    p.add_argument("--out", type=Path, default=Path("supporting_files/agreement/pre_post_comparison"))
    a = p.parse_args()

    rows, detail = [], {}
    for name, d in (("round1_v1", a.round1), ("round2_v2", a.round2)):
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        f = s["fields"]
        n = s["n_overlap"]
        fail_a = sum(v for k, v in f["outcome"]["distribution_A"].items() if k != "success")
        fail_b = sum(v for k, v in f["outcome"]["distribution_B"].items() if k != "success")
        cbf = s.get("category_given_both_failure") or {}
        step = f["first_error_step"]["recoded_no_failure_as_minus1"]
        coarse = coarse_kappa(d / "per_item_labels.csv")
        row = {
            "round": name, "n_items": n,
            "failure_prevalence_A1": round(fail_a / n, 3), "failure_prevalence_A2": round(fail_b / n, 3),
            "outcome_raw": f["outcome"]["raw_agreement"], "outcome_kappa": f["outcome"]["cohen_kappa"], "outcome_pabak": f["outcome"]["pabak"],
            "category_raw": f["primary_failure_category"]["raw_agreement"], "category_kappa": f["primary_failure_category"]["cohen_kappa"],
            "category_pabak": f["primary_failure_category"]["pabak"],
            "category_both_failure_n": cbf.get("n_items"), "category_both_failure_raw": cbf.get("raw_agreement"),
            "category_both_failure_kappa": cbf.get("cohen_kappa"),
            "step_exact": step["exact_agreement"], "step_kappa_nominal": step["cohen_kappa_nominal"],
            "step_kappa_quadratic": step["cohen_kappa_quadratic"],
            "coarse4_kappa_EXPLORATORY": coarse["kappa_all"], "coarse4_both_failure_kappa_EXPLORATORY": coarse["kappa_both_failure"],
            "median_seconds_per_item_A1": s["annotation_timing"]["A1"]["median_lead_time_s"],
            "median_seconds_per_item_A2": s["annotation_timing"]["A2"]["median_lead_time_s"],
        }
        rows.append(row)
        detail[name] = {"coarse_taxonomy_exploratory": coarse, "source": (d / "summary.json").as_posix()}

    a.out.with_suffix(".csv").parent.mkdir(parents=True, exist_ok=True)
    with a.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    out = {"rows": rows, "detail": detail, "coarse_mapping": COARSE,
           "caveats": ["Round 2 was enriched with failures (stratified sample), so prevalence differs; compare PABAK and raw agreement too.",
                       "Step agreement uses the same coding in both rounds (no_failure -> -1).",
                       "coarse4_* columns are post-hoc (defined after seeing round-2 disagreements) and are exploratory only."]}
    a.out.with_suffix(".json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()

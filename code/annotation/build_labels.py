"""
Build the final (adjudicated) label table for one annotation round.

Inputs (read-only):
  * two raw Label Studio exports (same format as code/annotation/agreement.py)
  * an adjudication CSV with one row per disagreeing trajectory

Rule: where both annotators agree on all three fields, the agreed label is
final (label_source = "agreed").  Where they disagree, the adjudicated label
is used (label_source = "adjudicated").  Every disagreement must have exactly
one adjudication row.  By default every adjudication must also be confirmed by
a named team member (column confirmed_by); pass --allow-unconfirmed only to
produce a clearly marked DRAFT.

first_error_step is written as an empty cell when the final category is
no_failure (guideline v2, R4): in v1 the value 0 was overloaded.

Usage (from the repository root):
    python -m code.annotation.build_labels \
        --export-a supporting_files/labels/annotator1_raw.json \
        --export-b supporting_files/labels/annotator2_raw.json \
        --adjudication supporting_files/agreement/round1/adjudication_round1.csv \
        --round round1 --guideline v1 \
        --out data/labels/labels_round1.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from code.annotation.agreement import (
    ALL_FIELDS,
    DEFAULT_CONFIG,
    NO_FAILURE,
    ExportError,
    allowed_values_from_config,
    load_export,
)


def build(export_a: Path, export_b: Path, adjudication: Path, round_name: str, guideline: str,
          out: Path, config: Path, allow_unconfirmed: bool) -> dict:
    allowed = allowed_values_from_config(config)
    la, da = load_export(export_a, allowed)
    lb, _ = load_export(export_b, allowed)
    if set(la) != set(lb):
        raise ExportError("Exports do not cover the same run_ids; build labels only on the overlap explicitly.")

    with adjudication.open(encoding="utf-8") as fh:
        adj = {row["run_id"]: row for row in csv.DictReader(fh)}

    disagreeing = {r for r in la if any(la[r][f] != lb[r][f] for f in ALL_FIELDS)}
    missing = disagreeing - set(adj)
    extra = set(adj) - disagreeing
    if missing:
        raise ExportError(f"{len(missing)} disagreements have no adjudication row: {sorted(missing)[:5]}")
    if extra:
        raise ExportError(f"adjudication rows for items without disagreement: {sorted(extra)[:5]}")

    unconfirmed = [r for r in disagreeing if not adj[r].get("confirmed_by", "").strip()]
    if unconfirmed and not allow_unconfirmed:
        raise ExportError(
            f"{len(unconfirmed)} adjudications are not confirmed (empty confirmed_by). "
            "Fill in confirmed_by/confirmed_on, or pass --allow-unconfirmed for a DRAFT."
        )

    rows = []
    for run_id in sorted(la):
        d = da[run_id]
        if run_id in disagreeing:
            a = adj[run_id]
            outcome, category = a["proposed_outcome"], a["proposed_category"]
            step = a["proposed_first_error_step"].strip()
            source = "adjudicated" if a.get("confirmed_by", "").strip() else "adjudicated_DRAFT"
        else:
            outcome, category = la[run_id]["outcome"], la[run_id]["primary_failure_category"]
            step = str(la[run_id]["first_error_step"])
            source = "agreed"
        if outcome not in allowed["outcome"] or category not in allowed["primary_failure_category"]:
            raise ExportError(f"{run_id}: invalid final label {outcome}/{category}")
        if (outcome == "success") != (category == NO_FAILURE):
            raise ExportError(f"{run_id}: outcome {outcome} inconsistent with category {category}")
        if category == NO_FAILURE:
            step = ""
        elif not step.isdigit():
            raise ExportError(f"{run_id}: failure label needs an integer first_error_step, got {step!r}")
        rows.append({
            "run_id": run_id,
            "task_id": d.get("task_id"),
            "template_family": d.get("template_family"),
            "model_version": d.get("model_version"),
            "outcome": outcome,
            "primary_failure_category": category,
            "first_error_step": step,
            "label_source": source,
            "round": round_name,
            "guideline_version": guideline,
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return {
        "n_items": len(rows),
        "n_agreed": sum(r["label_source"] == "agreed" for r in rows),
        "n_adjudicated": sum(r["label_source"].startswith("adjudicated") for r in rows),
        "n_unconfirmed": len(unconfirmed),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Merge two annotators' labels and adjudications into final labels.")
    p.add_argument("--export-a", type=Path, required=True)
    p.add_argument("--export-b", type=Path, required=True)
    p.add_argument("--adjudication", type=Path, required=True)
    p.add_argument("--round", required=True)
    p.add_argument("--guideline", required=True, help="Guideline version the round followed (v1 or v2)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--allow-unconfirmed", action="store_true")
    a = p.parse_args()
    s = build(a.export_a, a.export_b, a.adjudication, a.round, a.guideline, a.out, a.config, a.allow_unconfirmed)
    print(s)
    if s["n_unconfirmed"]:
        print(f"WARNING: {s['n_unconfirmed']} adjudications unconfirmed - output is a DRAFT.")


if __name__ == "__main__":
    main()

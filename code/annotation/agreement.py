"""
Inter-annotator agreement for AgentEvalOps (Assignment 1, Part C).

Reads two raw Label Studio JSON exports (one per annotator), validates them
against the labelling config, matches annotations by the stable trajectory
identifier ``data.run_id`` (never by export row order), and writes:

  <out>/summary.json                 headline numbers + input file checksums
  <out>/per_item_labels.csv          one row per overlapping trajectory
  <out>/disagreements.csv            only rows where any field disagrees
  <out>/human_vs_logged_outcome.csv  each annotator vs. the logged grader outcome
  <out>/confusion_<field>.csv/.png   confusion matrices (rows = A, cols = B)

The raw exports are read-only inputs and are never modified.  Annotator
identities are replaced by the anonymised IDs given on the command line;
no e-mail address or Label Studio user id is written to any output.

Usage (from the repository root):
    python -m code.annotation.agreement \
        --export-a supporting_files/labels/annotator1_raw.json --id-a A1 \
        --export-b supporting_files/labels/annotator2_raw.json --id-b A2 \
        --round round1 --out supporting_files/agreement/round1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

DEFAULT_CONFIG = Path("code/annotation/label_studio_config.xml")
CATEGORICAL_FIELDS = ("outcome", "primary_failure_category")
STEP_FIELD = "first_error_step"
ALL_FIELDS = CATEGORICAL_FIELDS + (STEP_FIELD,)
NO_FAILURE = "no_failure"


class ExportError(ValueError):
    """Raised when an export does not match the expected schema."""


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def allowed_values_from_config(config_path: Path) -> dict[str, list[str]]:
    """Read the permitted Choice values for each Choices control from the XML config."""
    root = ET.parse(config_path).getroot()
    allowed: dict[str, list[str]] = {}
    for choices in root.iter("Choices"):
        allowed[choices.get("name")] = [c.get("value") for c in choices.iter("Choice")]
    for number in root.iter("Number"):
        allowed[number.get("name")] = [number.get("min"), number.get("max")]
    missing = [f for f in ALL_FIELDS if f not in allowed]
    if missing:
        raise ExportError(f"Label config {config_path} has no control named {missing}")
    return allowed


def _extract_result(result: list[dict[str, Any]]) -> dict[str, Any]:
    labels: dict[str, Any] = {}
    for region in result:
        name = region.get("from_name")
        value = region.get("value", {})
        if name in labels:
            raise ExportError(f"field {name!r} appears twice in one annotation")
        if "choices" in value:
            choices = value["choices"]
            if len(choices) != 1:
                raise ExportError(f"field {name!r} must have exactly one choice, got {choices}")
            labels[name] = choices[0]
        elif "number" in value:
            labels[name] = value["number"]
        elif "text" in value:
            text = value["text"]
            labels[name] = " ".join(text) if isinstance(text, list) else text
    return labels


def load_export(path: Path, allowed: dict[str, list[str]]) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    Return ({run_id: labels}, {run_id: task data}) for one annotator's export.

    Raises ExportError on any schema problem so that bad data is never
    silently analysed.
    """
    tasks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not tasks:
        raise ExportError(f"{path}: expected a non-empty JSON list of Label Studio tasks")

    labels_by_run: dict[str, dict] = {}
    data_by_run: dict[str, dict] = {}
    annotator_ids: set[Any] = set()
    lo, hi = (int(x) for x in allowed[STEP_FIELD])

    for i, task in enumerate(tasks):
        data = task.get("data") or {}
        run_id = data.get("run_id")
        if not run_id:
            raise ExportError(f"{path}: task index {i} has no data.run_id")
        if run_id in labels_by_run:
            raise ExportError(f"{path}: run_id {run_id} appears in more than one task")

        anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if len(anns) != 1:
            raise ExportError(f"{path}: run_id {run_id} has {len(anns)} non-cancelled annotations (expected 1)")
        ann = anns[0]
        completed_by = ann.get("completed_by")
        annotator_ids.add(completed_by.get("id") if isinstance(completed_by, dict) else completed_by)

        labels = _extract_result(ann.get("result", []))
        for field in ALL_FIELDS:
            if field not in labels or labels[field] is None:
                raise ExportError(f"{path}: run_id {run_id} is missing {field!r}")
        for field in CATEGORICAL_FIELDS:
            if labels[field] not in allowed[field]:
                raise ExportError(f"{path}: run_id {run_id} has invalid {field}={labels[field]!r}")
        step = labels[STEP_FIELD]
        if not float(step).is_integer() or not lo <= int(step) <= hi:
            raise ExportError(f"{path}: run_id {run_id} has invalid {STEP_FIELD}={step!r}")
        labels[STEP_FIELD] = int(step)

        labels["_created_at"] = ann.get("created_at")
        labels["_lead_time_s"] = ann.get("lead_time")
        labels_by_run[run_id] = labels
        data_by_run[run_id] = data

    if len(annotator_ids) != 1:
        raise ExportError(f"{path}: expected one annotator per export, found {len(annotator_ids)}")
    return labels_by_run, data_by_run


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def kappa(a: list, b: list, labels: list | None = None, weights: str | None = None) -> float | None:
    """Cohen's kappa; None when undefined (both annotators used a single identical label)."""
    if len(set(a) | set(b)) < 2:
        return None
    value = cohen_kappa_score(a, b, labels=labels, weights=weights)
    return None if np.isnan(value) else float(value)


def categorical_stats(a: list, b: list, label_order: list[str]) -> dict[str, Any]:
    n = len(a)
    agree = sum(x == y for x, y in zip(a, b))
    po = agree / n
    used = [lab for lab in label_order if lab in set(a) | set(b)]
    return {
        "n_items": n,
        "n_agree": agree,
        "n_disagree": n - agree,
        "raw_agreement": round(po, 4),
        "cohen_kappa": _r(kappa(a, b, labels=used)),
        # PABAK: prevalence- and bias-adjusted kappa = 2*Po - 1 (for k=2) or
        # (k*Po - 1)/(k - 1) in general, using the number of categories in use.
        "pabak": round((len(used) * po - 1) / (len(used) - 1), 4) if len(used) > 1 else None,
        "categories_used": used,
        "distribution_A": {lab: a.count(lab) for lab in used},
        "distribution_B": {lab: b.count(lab) for lab in used},
    }


def step_stats(a: list[int], b: list[int], cat_a: list[str], cat_b: list[str]) -> dict[str, Any]:
    """
    first_error_step is an ordinal number, and in guideline v1 the value 0 was
    used both for 'no error' and for 'error at step 0'.  We therefore report
    it three ways: on all items, recoded so that no_failure -> -1 (no error),
    and only on items both annotators marked as failures.
    """
    def block(x: list[int], y: list[int]) -> dict[str, Any]:
        if not x:
            return {"n_items": 0}
        diffs = [abs(i - j) for i, j in zip(x, y)]
        return {
            "n_items": len(x),
            "exact_agreement": round(sum(d == 0 for d in diffs) / len(x), 4),
            "within_1_step": round(sum(d <= 1 for d in diffs) / len(x), 4),
            "mean_abs_diff": round(float(np.mean(diffs)), 4),
            "max_abs_diff": int(max(diffs)),
            "cohen_kappa_nominal": _r(kappa(x, y)),
            "cohen_kappa_quadratic": _r(kappa(x, y, weights="quadratic")),
        }

    recoded_a = [-1 if c == NO_FAILURE else s for s, c in zip(a, cat_a)]
    recoded_b = [-1 if c == NO_FAILURE else s for s, c in zip(b, cat_b)]
    both_fail = [(s, t) for s, t, c, d in zip(a, b, cat_a, cat_b) if c != NO_FAILURE and d != NO_FAILURE]
    return {
        "as_labelled": block(a, b),
        "recoded_no_failure_as_minus1": block(recoded_a, recoded_b),
        "both_marked_failure_only": block([s for s, _ in both_fail], [t for _, t in both_fail]),
    }


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["run_id"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion(out: Path, field: str, a: list, b: list, order: list, id_a: str, id_b: str) -> None:
    labels = [lab for lab in order if lab in set(a) | set(b)]
    cm = confusion_matrix(a, b, labels=labels)
    rows = [{f"{id_a} \\ {id_b}": str(lab), **{str(c): int(v) for c, v in zip(labels, row)}} for lab, row in zip(labels, cm)]
    write_csv(out / f"confusion_{field}.csv", rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = max(4.0, 0.9 * len(labels) + 2.5)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), [str(x) for x in labels], rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)), [str(x) for x in labels], fontsize=9)
    ax.set_xlabel(f"Annotator {id_b}")
    ax.set_ylabel(f"Annotator {id_a}")
    ax.set_title(f"{field} (n={len(a)})", fontsize=11)
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out / f"confusion_{field}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(export_a: Path, export_b: Path, id_a: str, id_b: str, round_name: str, out: Path, config: Path) -> dict:
    allowed = allowed_values_from_config(config)
    labels_a, data_a = load_export(export_a, allowed)
    labels_b, data_b = load_export(export_b, allowed)

    overlap = sorted(set(labels_a) & set(labels_b))
    if not overlap:
        raise ExportError("The two exports share no run_id; agreement cannot be computed.")
    mismatched = [r for r in overlap if data_a[r] != data_b[r]]

    out.mkdir(parents=True, exist_ok=True)
    per_item, disagreements, vs_logged = [], [], []
    for run_id in overlap:
        la, lb, d = labels_a[run_id], labels_b[run_id], data_a[run_id]
        diff_fields = [f for f in ALL_FIELDS if la[f] != lb[f]]
        row = {
            "run_id": run_id,
            "task_id": d.get("task_id"),
            "template_family": d.get("template_family"),
            "model_version": d.get("model_version"),
            "logged_outcome": d.get("outcome"),
            "gold_answer": d.get("gold_answer"),
            "final_answer": d.get("final_answer"),
            **{f"{f}_{id_a}": la[f] for f in ALL_FIELDS},
            **{f"{f}_{id_b}": lb[f] for f in ALL_FIELDS},
            "disagreeing_fields": ";".join(diff_fields),
        }
        per_item.append(row)
        if diff_fields:
            disagreements.append(row)
        vs_logged.append({
            "run_id": run_id,
            "task_id": d.get("task_id"),
            "evaluation_rule": d.get("evaluation_rule"),
            "logged_outcome": d.get("outcome"),
            f"outcome_{id_a}": la["outcome"],
            f"outcome_{id_b}": lb["outcome"],
            "both_humans_agree": la["outcome"] == lb["outcome"],
            "humans_agree_with_logged": la["outcome"] == lb["outcome"] == d.get("outcome"),
            "gold_answer": d.get("gold_answer"),
            "final_answer": d.get("final_answer"),
        })

    a = {f: [labels_a[r][f] for r in overlap] for f in ALL_FIELDS}
    b = {f: [labels_b[r][f] for r in overlap] for f in ALL_FIELDS}

    summary: dict[str, Any] = {
        "round": round_name,
        "inputs": {
            id_a: {"file": export_a.as_posix(), "sha256": sha256_of(export_a), "n_items": len(labels_a)},
            id_b: {"file": export_b.as_posix(), "sha256": sha256_of(export_b), "n_items": len(labels_b)},
            "label_config": {"file": config.as_posix(), "sha256": sha256_of(config)},
        },
        "matching_key": "data.run_id",
        "n_overlap": len(overlap),
        "n_only_" + id_a: len(set(labels_a) - set(labels_b)),
        "n_only_" + id_b: len(set(labels_b) - set(labels_a)),
        "n_overlap_with_differing_task_data": len(mismatched),
        "n_items_with_any_disagreement": len(disagreements),
        "fields": {
            "outcome": categorical_stats(a["outcome"], b["outcome"], allowed["outcome"]),
            "primary_failure_category": categorical_stats(
                a["primary_failure_category"], b["primary_failure_category"], allowed["primary_failure_category"]
            ),
            STEP_FIELD: step_stats(a[STEP_FIELD], b[STEP_FIELD], a["primary_failure_category"], b["primary_failure_category"]),
        },
        "category_given_both_failure": None,
        "annotation_timing": {
            ann: {
                "first_created_at": min(lbl["_created_at"] for lbl in labels.values()),
                "last_created_at": max(lbl["_created_at"] for lbl in labels.values()),
                "median_lead_time_s": round(float(np.median([lbl["_lead_time_s"] for lbl in labels.values()])), 2),
                "total_lead_time_min": round(sum(lbl["_lead_time_s"] for lbl in labels.values()) / 60, 2),
            }
            for ann, labels in ((id_a, labels_a), (id_b, labels_b))
        },
        "human_vs_logged_outcome": {
            "both_humans_agree_with_logged": sum(r["humans_agree_with_logged"] for r in vs_logged),
            f"{id_a}_differs_from_logged": sum(r[f"outcome_{id_a}"] != r["logged_outcome"] for r in vs_logged),
            f"{id_b}_differs_from_logged": sum(r[f"outcome_{id_b}"] != r["logged_outcome"] for r in vs_logged),
            "logged_failure_both_humans_success": sum(
                r["logged_outcome"] == "failure" and r[f"outcome_{id_a}"] == r[f"outcome_{id_b}"] == "success" for r in vs_logged
            ),
        },
    }

    both_fail = [i for i, (x, y) in enumerate(zip(a["outcome"], b["outcome"])) if x == y == "failure"]
    if both_fail:
        summary["category_given_both_failure"] = categorical_stats(
            [a["primary_failure_category"][i] for i in both_fail],
            [b["primary_failure_category"][i] for i in both_fail],
            allowed["primary_failure_category"],
        )

    write_csv(out / "per_item_labels.csv", per_item)
    write_csv(out / "disagreements.csv", disagreements)
    write_csv(out / "human_vs_logged_outcome.csv", vs_logged)
    for field in CATEGORICAL_FIELDS:
        write_confusion(out, field, a[field], b[field], allowed[field], id_a, id_b)
    step_order = sorted(set(a[STEP_FIELD]) | set(b[STEP_FIELD]))
    write_confusion(out, STEP_FIELD, a[STEP_FIELD], b[STEP_FIELD], step_order, id_a, id_b)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute inter-annotator agreement from two Label Studio exports.")
    parser.add_argument("--export-a", type=Path, required=True)
    parser.add_argument("--export-b", type=Path, required=True)
    parser.add_argument("--id-a", default="A1", help="Anonymised ID for annotator A")
    parser.add_argument("--id-b", default="A2", help="Anonymised ID for annotator B")
    parser.add_argument("--round", default="round1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    s = run(args.export_a, args.export_b, args.id_a, args.id_b, args.round, args.out, args.config)
    f = s["fields"]
    print(f"Round: {s['round']}   overlapping items: {s['n_overlap']}   items with any disagreement: {s['n_items_with_any_disagreement']}")
    for name in CATEGORICAL_FIELDS:
        st = f[name]
        print(f"  {name:26s} agree {st['n_agree']}/{st['n_items']} ({st['raw_agreement']:.3f})  kappa={st['cohen_kappa']}  PABAK={st['pabak']}")
    st = f[STEP_FIELD]["as_labelled"]
    print(f"  {STEP_FIELD:26s} exact {st['exact_agreement']:.3f}  kappa={st['cohen_kappa_nominal']}  quadratic-weighted kappa={st['cohen_kappa_quadratic']}  mean|diff|={st['mean_abs_diff']}")
    print(f"Outputs written to {args.out.as_posix()}/")


if __name__ == "__main__":
    main()

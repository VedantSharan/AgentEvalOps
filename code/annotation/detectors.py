"""
Simple automatic failure detectors vs. human labels (Assignment 1, Part C).

Detectors (trace-only, no model, no labels):
  loop_strict      some (tool, arguments) pair is called >= 3 times      (v2 R5, first clause)
  loop_extended    loop_strict OR the run hit the step limit without a submitted answer
                   (v2 R5, second clause: "step limit while cycling without progress")
  malformed_any    some tool call was REJECTED because of its arguments: unknown field,
                   expression syntax error, forbidden expression, unknown aggregate,
                   or non-existent file (v2 R3)
  malformed_strict malformed_any without the non-existent-file case

Ground truth: the human PRIMARY failure category (loop_or_repetition /
malformed_arguments).  The detectors fire on any occurrence in the trace,
whereas humans record only the earliest root cause, so a detector "false
positive" can be a real but non-primary event; the error table shows each case.

Comparisons are reported against
  final        adjudicated final labels (data/labels/labels_round*.csv)
  A1, A2       each annotator's own raw labels, per round (no adjudication involved)

Usage (from the repository root):
    python -m code.annotation.detectors
Outputs (supporting_files/detectors/):
    detector_predictions.csv, detector_metrics.csv, detector_errors.csv,
    confusion_<detector>.csv, summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

from sklearn.metrics import cohen_kappa_score, confusion_matrix

from code.annotation.agreement import ALL_FIELDS, allowed_values_from_config, load_export

ARG_REJECTION = re.compile(r"Unknown field|Syntax error in expression|Forbidden AST node|Unknown agg_field|Unsupported aggregate|"
                           r"Malformed arguments|Invocation error|must be a non-empty string|File not found in virtual store")
FILE_NOT_FOUND = re.compile(r"File not found in virtual store")
STEP_LIMIT = "Max steps reached without submitting an answer."

DETECTORS = {
    "loop_strict": "loop_or_repetition",
    "loop_extended": "loop_or_repetition",
    "malformed_any": "malformed_arguments",
    "malformed_strict": "malformed_arguments",
}


def detect(traj: dict) -> dict[str, bool | str]:
    counts: dict[tuple, int] = {}
    rejected, rejected_strict, evidence = [], [], []
    for s in traj["steps"]:
        if s["action_type"] != "tool_call":
            continue
        key = (s.get("tool_name"), json.dumps(s.get("arguments"), sort_keys=True))
        counts[key] = counts.get(key, 0) + 1
        tr = s.get("tool_result") or {}
        err = tr.get("error") or s.get("error") or ""
        if tr and not tr.get("success") and ARG_REJECTION.search(err):
            rejected.append(s["step_index"])
            if not FILE_NOT_FOUND.search(err):
                rejected_strict.append(s["step_index"])
            evidence.append(f"step {s['step_index']}: {err[:60]}")
    max_rep = max(counts.values(), default=0)
    step_limit = STEP_LIMIT in traj.get("errors", []) and not traj.get("final_answer")
    return {
        "loop_strict": max_rep >= 3,
        "loop_extended": max_rep >= 3 or step_limit,
        "malformed_any": bool(rejected),
        "malformed_strict": bool(rejected_strict),
        "max_identical_calls": max_rep,
        "step_limit_no_answer": step_limit,
        "rejected_call_steps": ";".join(map(str, rejected)),
        "evidence": " | ".join(evidence)[:300],
    }


def metrics(pred: list[bool], truth: list[bool]) -> dict[str, float | int | None]:
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[False, True]).ravel()
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
    kap = cohen_kappa_score(truth, pred) if len(set(truth) | set(pred)) > 1 else None
    r = lambda x: None if x is None else round(float(x), 4)
    return {"n": len(pred), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "positives_human": int(tp + fn),
            "precision": r(prec), "recall": r(rec), "f1": r(f1), "kappa": r(kap), "accuracy": r((tp + tn) / len(pred))}


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate loop / malformed-call detectors against human labels.")
    p.add_argument("--trajectories", type=Path, default=Path("data/experiments/multi_provider_run/trajectories.jsonl"))
    p.add_argument("--labels", type=Path, nargs="+", default=[Path("data/labels/labels_round1.csv"), Path("data/labels/labels_round2.csv")])
    p.add_argument("--round-exports", nargs="+", default=[
        "round1:supporting_files/labels/annotator1_raw.json:supporting_files/labels/annotator2_raw.json:code/annotation/label_studio_config.xml",
        "round2:supporting_files/labels/round2/annotator1_raw.json:supporting_files/labels/round2/annotator2_raw.json:code/annotation/label_studio_config_v2.xml"])
    p.add_argument("--out", type=Path, default=Path("supporting_files/detectors"))
    a = p.parse_args()

    trajs = {json.loads(l)["run_id"]: json.loads(l) for l in a.trajectories.read_text(encoding="utf-8").splitlines() if l.strip()}
    final: dict[str, dict] = {}
    for lp in a.labels:
        with lp.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                final[r["run_id"]] = r
    raw: dict[str, dict[str, str]] = {}  # run_id -> {"A1": cat, "A2": cat, "round": ...}
    for spec in a.round_exports:
        rnd, ea, eb, cfg = spec.split(":")
        allowed = allowed_values_from_config(Path(cfg))
        la, _ = load_export(Path(ea), allowed)
        lb, _ = load_export(Path(eb), allowed)
        for rid in la:
            raw[rid] = {"round": rnd, "A1": la[rid]["primary_failure_category"], "A2": lb[rid]["primary_failure_category"]}

    ids = sorted(final)
    preds = {rid: detect(trajs[rid]) for rid in ids}
    a.out.mkdir(parents=True, exist_ok=True)

    pred_rows = [{"run_id": rid, "task_id": final[rid]["task_id"], "round": raw[rid]["round"],
                  "human_final_category": final[rid]["primary_failure_category"],
                  "A1_category": raw[rid]["A1"], "A2_category": raw[rid]["A2"], **preds[rid]} for rid in ids]
    with (a.out / "detector_predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pred_rows[0].keys()))
        w.writeheader()
        w.writerows(pred_rows)

    metric_rows, error_rows = [], []
    for det, cat in DETECTORS.items():
        for ref in ("final", "A1", "A2"):
            for rnd in ("all", "round1", "round2"):
                sub = [rid for rid in ids if rnd == "all" or raw[rid]["round"] == rnd]
                truth = [(final[rid]["primary_failure_category"] if ref == "final" else raw[rid][ref]) == cat for rid in sub]
                pred = [bool(preds[rid][det]) for rid in sub]
                metric_rows.append({"detector": det, "human_category": cat, "reference": ref, "subset": rnd, **metrics(pred, truth)})
        truth = [final[rid]["primary_failure_category"] == cat for rid in ids]
        cm = confusion_matrix(truth, [bool(preds[rid][det]) for rid in ids], labels=[False, True])
        with (a.out / f"confusion_{det}.csv").open("w", newline="", encoding="utf-8") as fh:
            fh.write(f"human_final \\ detector,not_{cat},{cat}\nnot_{cat},{cm[0][0]},{cm[0][1]}\n{cat},{cm[1][0]},{cm[1][1]}\n")
        for rid in ids:
            h, d = final[rid]["primary_failure_category"] == cat, bool(preds[rid][det])
            if h != d:
                error_rows.append({"detector": det, "error_type": "false_positive" if d else "false_negative", "run_id": rid,
                                   "task_id": final[rid]["task_id"], "human_final_category": final[rid]["primary_failure_category"],
                                   "max_identical_calls": preds[rid]["max_identical_calls"],
                                   "step_limit_no_answer": preds[rid]["step_limit_no_answer"],
                                   "detector_evidence": preds[rid]["evidence"]})
    with (a.out / "detector_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(metric_rows[0].keys()))
        w.writeheader()
        w.writerows(metric_rows)
    with (a.out / "detector_errors.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(error_rows[0].keys()) if error_rows else ["detector"])
        w.writeheader()
        w.writerows(error_rows)

    headline = {f"{r['detector']}": {k: r[k] for k in ("n", "positives_human", "tp", "fp", "fn", "precision", "recall", "f1", "kappa")}
                for r in metric_rows if r["reference"] == "final" and r["subset"] == "all"}
    summary = {"label_files": [x.as_posix() for x in a.labels], "n_items": len(ids), "final_labels_vs_detectors": headline,
               "note": "Detectors flag any occurrence; humans label only the primary (earliest) cause."}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()

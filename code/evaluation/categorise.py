"""Automatic assignment of the A1 failure taxonomy to a trajectory (Part C).

The A2 brief asks for Assignment 1 failure categories on every evaluated run.
Humans labelled only 85 A1 trajectories, so A2 runs are categorised by rules
built on the A1 detectors (code/annotation/detectors.py). The rules follow the
A1 guideline order: outcome first, then the earliest root cause.

    correct answer                                    -> no_failure
    no answer submitted
        repeated identical calls / step limit         -> loop_or_repetition
        rejected (malformed) tool calls               -> malformed_arguments
        otherwise                                     -> premature_stop
    wrong answer submitted
        rejected (malformed) tool calls               -> malformed_arguments
        expected tools never used, other tools used   -> wrong_tool
        no tool used at all                           -> premature_stop
        a value given for an unanswerable task        -> hallucinated_result
        otherwise                                     -> correct_process_with_wrong_answer

The rules were fitted on the train-split labels only. A first version also
called "hallucinated" any answer number that no tool returned; A1 annotators
labelled most such runs correct_process_with_wrong_answer, so the rule was
dropped (train kappa 0.58 -> 0.70). Validation labels are the held-out check.

``python -m code.evaluation.categorise`` measures agreement (Cohen's kappa)
between these rules and the human labels on the train (fitting) and validation
(held-out) A1 trajectories. Test-split labels are not used (the gold set is for final
evaluation only), and the three gold labels that mark empty answers as success
(DATA_CHANGES.md) would be excluded anyway.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score

from code.annotation.detectors import detect
from code.shared.models import EvaluationRule, TaskRecord

CATEGORIES = ("no_failure", "wrong_tool", "malformed_arguments", "hallucinated_result", "loop_or_repetition",
              "premature_stop", "correct_process_with_wrong_answer")
def categorise(traj: dict[str, Any], task: TaskRecord, correct: bool) -> str:
    """Return the A1 primary failure category for one trajectory (A1 record format)."""
    if correct:
        return "no_failure"
    det = detect(traj)
    answer = (traj.get("final_answer") or "").strip()
    tool_steps = [s for s in traj.get("steps", []) if s.get("action_type") == "tool_call"]
    if not answer:
        if det["loop_strict"] or (det["step_limit_no_answer"] and det["max_identical_calls"] >= 2):
            return "loop_or_repetition"
        if det["malformed_any"]:
            return "malformed_arguments"
        return "premature_stop"
    if det["malformed_strict"]:
        return "malformed_arguments"
    used = {s.get("tool_name") for s in tool_steps if (s.get("tool_result") or {}).get("success")}
    if task.expected_tools and tool_steps and not used & set(task.expected_tools):
        return "wrong_tool"
    if not tool_steps:
        return "premature_stop"
    if task.evaluation_rule == EvaluationRule.unanswerable:
        return "hallucinated_result"
    return "correct_process_with_wrong_answer"


def agreement_with_humans(trajectories: Path, labels: list[Path], regrade: Path, tasks: Path,
                          splits: list[Path]) -> dict[str, Any]:
    allowed = set()
    for sp in splits:
        with sp.open(encoding="utf-8", newline="") as fh:
            allowed |= {r["item_id"] for r in csv.DictReader(fh) if r["item_type"] == "trajectory"}
    task_map = {}
    with tasks.open(encoding="utf-8") as fh:
        for line in fh:
            t = TaskRecord.model_validate_json(line)
            task_map[t.task_id] = t
    with regrade.open(encoding="utf-8", newline="") as fh:
        correct = {r["run_id"]: r["answer_correct_v2"] == "True" for r in csv.DictReader(fh)}
    trajs = {}
    with trajectories.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            trajs[r["run_id"]] = r
    human, auto, items = [], [], []
    for path in labels:
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                rid = row["run_id"]
                if rid not in allowed or rid not in trajs:
                    continue
                pred = categorise(trajs[rid], task_map[row["task_id"]], correct.get(rid, False))
                human.append(row["primary_failure_category"])
                auto.append(pred)
                items.append({"run_id": rid, "task_id": row["task_id"], "human": human[-1], "auto": pred})
    both_fail = [(h, a) for h, a in zip(human, auto) if h != "no_failure" and a != "no_failure"]
    kappa_all = cohen_kappa_score(human, auto) if len(set(human) | set(auto)) > 1 else None
    kappa_fail = (cohen_kappa_score([h for h, _ in both_fail], [a for _, a in both_fail])
                  if len({x for p in both_fail for x in p}) > 1 else None)
    return {
        "n": len(human),
        "splits": [p.stem for p in splits],
        "accuracy": round(sum(h == a for h, a in zip(human, auto)) / len(human), 4) if human else None,
        "kappa_category": None if kappa_all is None else round(float(kappa_all), 4),
        "n_both_failure": len(both_fail),
        "kappa_category_both_failure": None if kappa_fail is None else round(float(kappa_fail), 4),
        "items": items,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Agreement of the rule-based categoriser with A1 human labels.")
    p.add_argument("--trajectories", type=Path, default=Path("data/experiments/multi_provider_run/trajectories.jsonl"))
    p.add_argument("--labels", type=Path, nargs="+",
                   default=[Path("data/labels/labels_round1.csv"), Path("data/labels/labels_round2.csv")])
    p.add_argument("--regrade", type=Path, default=Path("data/processed/regrade_v2.csv"))
    p.add_argument("--tasks", type=Path, default=Path("data/processed/tasks_revised.jsonl"))
    p.add_argument("--splits", type=Path, nargs="+",
                   default=[Path("data/splits/train.csv"), Path("data/splits/validation.csv")])
    p.add_argument("--out", type=Path, default=Path("supporting_files/a2/categoriser_vs_human.json"))
    args = p.parse_args(argv)
    res = {sp.stem: agreement_with_humans(args.trajectories, args.labels, args.regrade, args.tasks, [sp])
           for sp in args.splits}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8", newline="\n")
    for name, r in res.items():
        print(name, {k: v for k, v in r.items() if k != "items"})


if __name__ == "__main__":
    main()

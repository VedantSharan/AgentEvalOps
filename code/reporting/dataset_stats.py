"""
Compute every statistic quoted in docs/DATASET_CARD.md and report/ from the data files.

Usage (from the repository root):
    python -m code.reporting.dataset_stats
Output:
    supporting_files/dataset_stats.json
"""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

OUT = Path("supporting_files/dataset_stats.json")


def _jsonl(p: str) -> list[dict]:
    return [json.loads(line) for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()]


def _csv(p: str) -> list[dict]:
    return list(csv.DictReader(Path(p).open(encoding="utf-8")))


def main() -> None:
    tasks = {t["task_id"]: t for t in _jsonl("data/processed/tasks_revised.jsonl")}
    trajs = _jsonl("data/experiments/multi_provider_run/trajectories.jsonl")
    regrade = {r["run_id"]: r for r in _csv("data/processed/regrade_v2.csv")}
    aug = _jsonl("data/augmented/tasks_augmented.jsonl")
    labels = _csv("data/labels/labels_round1.csv") + _csv("data/labels/labels_round2.csv")
    fam = lambda tid: tasks[tid]["template_family"]

    steps = [len(t["steps"]) for t in trajs]
    s = {
        "tasks": {
            "n": len(tasks),
            "by_family": Counter(t["template_family"] for t in tasks.values()),
            "by_evaluation_rule": Counter(t["evaluation_rule"] for t in tasks.values()),
            "answerable": sum(t["answerable"] for t in tasks.values()),
            "unanswerable": sum(not t["answerable"] for t in tasks.values()),
            "tasks_with_at_least_one_trajectory": len({t["task_id"] for t in trajs}),
            "tasks_never_run": sorted(set(tasks) - {t["task_id"] for t in trajs}),
        },
        "trajectories": {
            "n": len(trajs),
            "by_family": Counter(fam(t["task_id"]) for t in trajs),
            "by_model": Counter(t["model_version"] for t in trajs),
            "n_models": len({t["model_version"] for t in trajs}),
            "n_providers": len({t["model_version"].split(":")[0] for t in trajs}),
            "logged_outcome_grader_v1": Counter(t["outcome"] for t in trajs),
            "answer_correct_grader_v2": Counter(regrade[t["run_id"]]["answer_correct_v2"] for t in trajs),
            "no_submitted_answer": sum(not t.get("final_answer") for t in trajs),
            "steps_min_median_max": [min(steps), sorted(steps)[len(steps) // 2], max(steps)],
            "runs_per_task_min_max": [min(Counter(t["task_id"] for t in trajs).values()), max(Counter(t["task_id"] for t in trajs).values())],
            "prompt_versions": Counter(t["prompt_version"] for t in trajs),
        },
        "human_labels": {
            "n": len(labels),
            "by_round": Counter(r["round"] for r in labels),
            "label_source": Counter(r["label_source"] for r in labels),
            "outcome": Counter(r["outcome"] for r in labels),
            "primary_failure_category": Counter(r["primary_failure_category"] for r in labels),
            "failure_category_by_family": {f: Counter(r["primary_failure_category"] for r in labels
                                                      if r["template_family"] == f and r["outcome"] != "success")
                                           for f in sorted({r["template_family"] for r in labels})},
            "by_family": Counter(r["template_family"] for r in labels),
            "by_model": Counter(r["model_version"] for r in labels),
        },
        "augmented": {
            "n": len(aug),
            "by_type": Counter(a["augmentation"]["type"] for a in aug),
            "fault_labels": Counter(a["augmentation"]["params"].get("fault_label") for a in aug if a["augmentation"]["type"] == "fault_injection"),
            "label_effect": Counter(a["augmentation"]["label_effect"] for a in aug),
        },
        "splits": json.loads(Path("data/splits/split_definition.json").read_text())["counts"],
        "gold_test_sha256": json.loads(Path("data/splits/split_definition.json").read_text())["gold_test_set"]["sha256"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(s, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(s, indent=1, default=dict)[:4000])


if __name__ == "__main__":
    main()

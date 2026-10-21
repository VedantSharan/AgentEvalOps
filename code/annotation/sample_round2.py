"""
Draw the round-2 annotation sample (guidelines v2) for Assignment 1, Part C.

Design
------
* Candidate pool: every collected trajectory (data/experiments/multi_provider_run/
  trajectories.jsonl) that was NOT in round 1.  Unlike round 1, the 52 runs logged
  as outcome='error' are included: they are agent failures (step-limit exhaustion,
  wrong file/field names, malformed calls), not infrastructure crashes, and they
  carry the loop / malformed-argument cases that round 1 lacked.
* Strata (used only for sampling; never shown to annotators):
    answered_correct   final answer correct under grader v2
    answered_wrong     an answer was submitted, grader v2 says wrong
    run_error          logged outcome 'error' (no valid submission)
  Round 1 was 84% success, so round 2 deliberately over-samples failures to give
  every failure category a chance to occur.  Consequence: round-1 and round-2
  kappas are measured at different prevalences (report PABAK alongside kappa).
* Within a stratum, items are picked round-robin over template families, then
  models, after a seeded shuffle.  The final presentation order is a second
  seeded shuffle and is identical for both annotators.
* The logged grader outcome is removed from what annotators see (guideline v2).

Usage (from the repository root):
    python -m code.annotation.sample_round2
Outputs:
    data/labels/round2_sample.json      Label Studio import file (same for A1 and A2)
    data/labels/round2_manifest.csv     run_id, stratum, family, model, presentation order
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random

from code.annotation.sample_for_labelling import format_steps_for_display

SEED = 2026
DEFAULT_TARGETS = {"answered_correct": 12, "answered_wrong": 10, "run_error": 18}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stratum(traj: dict, correct_v2: bool) -> str:
    if traj["outcome"] == "error":
        return "run_error"
    return "answered_correct" if correct_v2 else "answered_wrong"


def round_robin(items: list[dict], k: int, rng: random.Random, tasks: dict) -> list[dict]:
    """Pick k items, cycling over template families, then models, after a seeded shuffle."""
    items = items[:]
    rng.shuffle(items)
    by_family: dict[str, list[dict]] = defaultdict(list)
    for t in items:
        by_family[tasks[t["task_id"]]["template_family"]].append(t)
    for fam in by_family:  # within a family, alternate models
        seen: dict[str, int] = defaultdict(int)
        ranked = []
        for t in by_family[fam]:
            ranked.append((seen[t["model_version"]], len(ranked), t))
            seen[t["model_version"]] += 1
        by_family[fam] = [t for _, _, t in sorted(ranked, key=lambda x: (x[0], x[1]))]
    families = sorted(by_family)
    rng.shuffle(families)
    chosen: list[dict] = []
    while len(chosen) < k and any(by_family.values()):
        for fam in families:
            if by_family[fam] and len(chosen) < k:
                chosen.append(by_family[fam].pop(0))
    return chosen


def build(trajectories: Path, tasks_path: Path, round1: Path, regrade: Path, targets: dict, seed: int,
          out_json: Path, out_manifest: Path) -> list[dict]:
    rng = random.Random(seed)
    tasks = {t["task_id"]: t for t in _read_jsonl(tasks_path)}
    round1_ids = {s["data"]["run_id"] for s in json.loads(round1.read_text(encoding="utf-8"))}
    with regrade.open(encoding="utf-8") as fh:
        correct_v2 = {r["run_id"]: r["answer_correct_v2"] == "True" for r in csv.DictReader(fh)}

    pool = [t for t in _read_jsonl(trajectories) if t["run_id"] not in round1_ids]
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for t in pool:
        by_stratum[stratum(t, correct_v2[t["run_id"]])].append(t)

    selected: list[tuple[str, dict]] = []
    for name in sorted(targets):
        avail = by_stratum.get(name, [])
        if len(avail) < targets[name]:
            raise ValueError(f"stratum {name}: need {targets[name]}, only {len(avail)} available")
        selected += [(name, t) for t in round_robin(avail, targets[name], rng, tasks)]

    rng.shuffle(selected)  # presentation order
    ls_tasks, manifest = [], []
    for order, (name, t) in enumerate(selected, 1):
        task = tasks[t["task_id"]]
        ls_tasks.append({
            "data": {
                "run_id": t["run_id"],
                "task_id": t["task_id"],
                "template_family": task["template_family"],
                "task_type": task["task_type"],
                "model_version": t["model_version"],
                "prompt": task["prompt"],
                "gold_answer": task["gold_answer"] if task["gold_answer"] is not None else "None (Unanswerable)",
                "evaluation_rule": task["evaluation_rule"],
                "final_answer": t.get("final_answer") if t.get("final_answer") is not None else "None (no answer submitted)",
                "total_steps": t["metrics"]["total_steps"],
                "steps_transcript": format_steps_for_display(t.get("steps", [])),
                "errors": t.get("errors", []),
                "presentation_order": order,
            }
        })
        manifest.append({
            "presentation_order": order,
            "run_id": t["run_id"],
            "task_id": t["task_id"],
            "template_family": task["template_family"],
            "model_version": t["model_version"],
            "stratum": name,
            "logged_outcome": t["outcome"],
            "total_steps": t["metrics"]["total_steps"],
            "seed": seed,
        })

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(ls_tasks, indent=2), encoding="utf-8")
    with out_manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Draw the seeded, stratified round-2 annotation sample.")
    p.add_argument("--trajectories", type=Path, default=Path("data/experiments/multi_provider_run/trajectories.jsonl"))
    p.add_argument("--tasks", type=Path, default=Path("data/processed/tasks_revised.jsonl"))
    p.add_argument("--round1", type=Path, default=Path("data/labels/stratified_sample_45.json"))
    p.add_argument("--regrade", type=Path, default=Path("data/processed/regrade_v2.csv"))
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out-json", type=Path, default=Path("data/labels/round2_sample.json"))
    p.add_argument("--out-manifest", type=Path, default=Path("data/labels/round2_manifest.csv"))
    a = p.parse_args()
    m = build(a.trajectories, a.tasks, a.round1, a.regrade, DEFAULT_TARGETS, a.seed, a.out_json, a.out_manifest)

    from collections import Counter
    print(f"Round-2 sample: {len(m)} trajectories (seed {a.seed})")
    print("  strata:  ", dict(Counter(r["stratum"] for r in m)))
    print("  families:", dict(Counter(r["template_family"] for r in m)))
    print("  models:  ", len({r["model_version"] for r in m}), "distinct")
    print(f"  wrote {a.out_json.as_posix()} and {a.out_manifest.as_posix()}")


if __name__ == "__main__":
    main()

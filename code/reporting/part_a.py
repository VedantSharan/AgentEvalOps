"""Part A summary: baselines vs the proposed agent, and the data comparison, on validation.

Reads supporting_files/a2/eval/<system>__validation.json (written by
code/evaluation/evaluate.py) and writes supporting_files/a2/partA/summary.json
and table.md with success (95% CI), cost, latency, paired tests, the
configurations' hashes, DVC data hashes, seeds, hardware and run time.

    python -m code.reporting.part_a
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from code.agent.config import REPO_ROOT
from code.evaluation.evaluate import RUNS_DIR
from code.evaluation.metrics import paired_comparison
from code.experiments.local_llm import OllamaClient
from code.reporting.hardware import hardware_record

EVAL_DIR = REPO_ROOT / "supporting_files" / "a2" / "eval"
OUT_DIR = REPO_ROOT / "supporting_files" / "a2" / "partA"
SYSTEMS = [
    ("b0_abstain", "B0 trivial: always abstain"),
    ("b1_direct", "B1 unadapted: direct answer, no tools"),
    ("b2_original", "B2 original A1 function-calling loop"),
    ("proposed", "P proposed agent, no exemplars (D0)"),
    ("proposed_train_base", "P + train_base exemplars (D1)"),
    ("proposed_train_aug_al", "P + augmented + active-learning data (D2)"),
]
COMPARISONS = [("b2_original", "proposed"), ("b1_direct", "proposed"), ("proposed", "proposed_train_base"),
               ("proposed", "proposed_train_aug_al"), ("proposed_train_base", "proposed_train_aug_al")]


def _rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-(task, seed) correctness rows from the run cache of an evaluation result."""
    path = REPO_ROOT / result["runs_file"]
    wanted = set(result["per_task"])
    seeds = set(result["seeds"])
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["task_id"] in wanted and r["seed"] in seeds:
            rows.append({"task_id": r["task_id"], "seed": r["seed"], "correct": r["correct"]})
    return rows


def load_results(split: str = "validation", systems: list[tuple[str, str]] = SYSTEMS) -> dict[str, dict[str, Any]]:
    out = {}
    for name, _ in systems:
        path = EVAL_DIR / f"{name}__{split}.json"
        if path.exists():
            out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out


def table_rows(results: dict[str, dict[str, Any]], systems: list[tuple[str, str]] = SYSTEMS) -> list[dict[str, Any]]:
    rows = []
    for name, desc in systems:
        if name not in results:
            continue
        r = results[name]
        m = r["metrics"]
        rows.append({
            "system": name, "description": desc, "config_hash": r["config_hash"][:12], "model": r["model"],
            "success": m["success"], "ci_low": m["success_ci_low"], "ci_high": m["success_ci_high"],
            "n_tasks": m["n_tasks"], "mean_steps": m["mean_steps"], "mean_total_tokens": m["mean_total_tokens"],
            "p50_wall_s": round(m["p50_wall_ms"] / 1000, 1), "p95_wall_s": round(m["p95_wall_ms"] / 1000, 1),
            "submitted": m["rate_submitted"], "loop_rate": m["rate_loop_extended"],
            "malformed_rate": m["rate_malformed_any"],
            "compute_s": round(m["mean_wall_ms"] * m["n_runs"] / 1000, 1),
            "mlflow_run_id": r.get("mlflow_run_id"),
        })
    return rows


def markdown(rows: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> str:
    lines = ["| System | Success (95% CI) | Steps | Tokens | p50 / p95 latency (s) | Submitted | MLflow run |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['description']} | {r['success']:.3f} ({r['ci_low']:.3f}-{r['ci_high']:.3f}) | "
                     f"{r['mean_steps']:.1f} | {r['mean_total_tokens']:.0f} | {r['p50_wall_s']} / {r['p95_wall_s']} | "
                     f"{r['submitted']:.2f} | `{r['mlflow_run_id']}` |")
    lines += ["", "| Comparison (B vs A) | A | B | Diff (95% CI) | McNemar p | Paired bootstrap p |", "|---|---|---|---|---|---|"]
    for c in comparisons:
        if c.get("n_pairs"):
            lines.append(f"| {c['b']} vs {c['a']} | {c['success_a']:.3f} | {c['success_b']:.3f} | "
                         f"{c['diff_mean']:+.3f} ({c['diff_ci_low']:+.3f} to {c['diff_ci_high']:+.3f}) | "
                         f"{c['mcnemar_exact_p']:.3f} | {c['paired_bootstrap_p']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    results = load_results()
    rows = table_rows(results)
    comparisons = []
    for a, b in COMPARISONS:
        if a in results and b in results:
            comparisons.append({"a": a, "b": b, **paired_comparison(_rows(results[a]), _rows(results[b]))})
    summary = {
        "split": "validation",
        "systems": rows,
        "paired_comparisons": comparisons,
        "configurations": {n: results[n]["config"] for n in results},
        "config_hashes": {n: results[n]["config_hash"] for n in results},
        "seeds": {n: results[n]["seeds"] for n in results},
        "data_version": next(iter(results.values()))["data_version"] if results else {},
        "git_commits": {n: results[n]["git_commit"] for n in results},
        "hardware": hardware_record(OllamaClient()),
        "training_note": ("No weights are trained. 'Training' builds the exemplar banks from train-split data "
                          "(python -m code.agent.exemplars, under a second on CPU). compute_s is the summed "
                          "wall time of each system's validation runs on the CPU server."),
        "runs_dir": RUNS_DIR.relative_to(REPO_ROOT).as_posix(),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    (OUT_DIR / "table.md").write_text(markdown(rows, comparisons), encoding="utf-8", newline="\n")
    print(markdown(rows, comparisons))


if __name__ == "__main__":
    main()

"""Part C report: all systems on every split, paired tests, slices, errors, robustness.

    python -m code.reporting.part_c --final production

Reads the evaluation outputs written by code/evaluation/evaluate.py
(supporting_files/a2/eval/<version>__<split>.json) for the registered
baselines and the final version, and writes to supporting_files/a2/partC/:

  comparison.json / comparison.md   success (95% CI), cost and latency per system and split,
                                    with paired McNemar / bootstrap tests of the final
                                    system against each baseline
  slices.json                       final system by template family, answerability,
                                    augmentation type and fault type; weakest groups
  error_analysis.csv                every test-split error of the final system and the
                                    baselines, A1 failure category, evidence, suggested fix
  robustness.json                   base test tasks vs their A1 Part D augmented variants
  ../reported_values.json           the numbers the Docker validation and the UI compare with
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from code.agent.config import REPO_ROOT
from code.evaluation.data import load_split
from code.evaluation.metrics import paired_comparison
from code.registry.registry import Registry

EVAL_DIR = REPO_ROOT / "supporting_files" / "a2" / "eval"
OUT = REPO_ROOT / "supporting_files" / "a2" / "partC"
REPORTED = REPO_ROOT / "supporting_files" / "a2" / "reported_values.json"
SPLITS = ("validation", "test", "gold", "shifted_test", "shifted", "test_augmented")
UI_SUBSET = 5
FIXES = {
    "premature_stop": "Answered before collecting every needed fact: add a check that each quantity in the "
                      "answer came from a successful tool call before submit_answer is accepted.",
    "correct_process_with_wrong_answer": "Right evidence, wrong final value: add a verification turn that "
                                         "recomputes with calculate and checks the unit and rounding asked for.",
    "malformed_arguments": "Tool rejected the arguments: validate arguments against the schema before calling "
                           "and show one correct example call per tool.",
    "loop_or_repetition": "Repeated calls without progress: stop after two blocked repeats and force a submission "
                          "or abstention.",
    "hallucinated_result": "Gave a value where the information does not exist: when a lookup returns no record "
                           "or keeps failing, require 'unanswerable' (add an abstention exemplar).",
    "wrong_tool": "Used a tool that cannot hold the fact: add routing hints (files -> read_file, employees -> "
                  "query_records, policies -> retrieve).",
}


def load(label: str, split: str) -> dict[str, Any] | None:
    path = EVAL_DIR / f"{label}__{split}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def run_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    path = REPO_ROOT / result["runs_file"]
    tasks, seeds = set(result["per_task"]), set(result["seeds"])
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["task_id"] in tasks and r["seed"] in seeds:
            out.append(r)
    return out


def _evidence(r: dict[str, Any]) -> str:
    calls = r.get("tool_call_log") or []
    failed = [c for c in calls if not c["success"]]
    if failed:
        c = failed[0]
        return f"{c['tool']}({json.dumps(c['arguments'])[:80]}) -> {str(c['error'])[:80]}"
    if calls:
        c = calls[-1]
        return f"last call {c['tool']}({json.dumps(c['arguments'])[:80]}) -> {json.dumps(c['output'], default=str)[:80]}"
    return "no tool call"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--final", default="production")
    p.add_argument("--baselines", nargs="+", default=["b0_abstain", "b1_direct", "b2_original"],
                   help="configuration names of the registered baselines")
    p.add_argument("--out", type=Path, default=OUT, help="output folder (default supporting_files/a2/partC)")
    args = p.parse_args(argv)
    out_dir = args.out
    reg = Registry(mlflow_sync=False)
    final_version, final_cfg = reg.resolve(args.final)
    by_name = {reg.entry(v)["config"]["name"]: v for v in reg.versions()}
    systems = [(by_name[b], b) for b in args.baselines if b in by_name] + [(final_version, f"final ({final_cfg.name})")]
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison: dict[str, Any] = {"final_version": final_version, "systems": [v for v, _ in systems], "splits": {}}
    md = ["| Split | System | Success (95% CI) | pass^k | Steps | Tokens | p95 latency (s) | MLflow run |",
          "|---|---|---|---|---|---|---|---|"]
    for split in SPLITS:
        final_res = load(final_version, split)
        entries = {}
        for version, name in systems:
            res = load(version, split)
            if res is None:
                continue
            m = res["metrics"]
            entry = {"name": name, "success": m["success"], "ci": [m["success_ci_low"], m["success_ci_high"]],
                     "pass_k": m["pass_k"], "k": m["k"], "n_tasks": m["n_tasks"], "mean_steps": m["mean_steps"],
                     "mean_total_tokens": m["mean_total_tokens"], "p95_wall_s": round(m["p95_wall_ms"] / 1000, 1),
                     "failure_categories": m["failure_categories"], "mlflow_run_id": res.get("mlflow_run_id")}
            if final_res and version != final_version:
                entry["final_vs_this"] = paired_comparison(run_rows(res), run_rows(final_res))
            entries[version] = entry
            md.append(f"| {split} | {name} | {m['success']:.3f} ({m['success_ci_low']:.3f}-{m['success_ci_high']:.3f}) | "
                      f"{m['pass_k']:.3f} | {m['mean_steps']:.1f} | {m['mean_total_tokens']:.0f} | "
                      f"{m['p95_wall_ms'] / 1000:.1f} | `{res.get('mlflow_run_id')}` |")
        comparison["splits"][split] = entries
    md += ["", "| Split | Final vs | Diff (95% CI) | McNemar p | Paired bootstrap p | n tasks |", "|---|---|---|---|---|---|"]
    for split, entries in comparison["splits"].items():
        for e in entries.values():
            c = e.get("final_vs_this")
            if c and c.get("n_pairs"):
                md.append(f"| {split} | {e['name']} | {c['diff_mean']:+.3f} ({c['diff_ci_low']:+.3f} to "
                          f"{c['diff_ci_high']:+.3f}) | {c['mcnemar_exact_p']:.4f} | {c['paired_bootstrap_p']:.4f} | "
                          f"{c['n_tasks']} |")
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")

    # Slices and weak groups of the final system
    slices = {}
    for split in ("test", "shifted", "test_augmented", "validation"):
        res = load(final_version, split)
        if res:
            slices[split] = res["slices"]
    weak = []
    for split, tables in slices.items():
        for key, table in tables.items():
            for value, s in table.items():
                if s["n_tasks"] >= 3:
                    weak.append({"split": split, "slice": key, "value": value, **s})
    weak.sort(key=lambda w: (w["success"], -w["n_tasks"]))
    (out_dir / "slices.json").write_text(json.dumps({"final_version": final_version, "slices": slices,
                                                 "weakest_groups": weak[:8]}, indent=2) + "\n",
                                     encoding="utf-8", newline="\n")

    # Error sheet: every test-split error, final system first
    items = {i.task_id: i for i in load_split("test")}
    errors = []
    for version, name in [(final_version, f"final ({final_cfg.name})")] + [s for s in systems if s[0] != final_version]:
        res = load(version, "test")
        if not res:
            continue
        for r in run_rows(res):
            if r["correct"]:
                continue
            item = items[r["task_id"]]
            cat = r.get("failure_category", "")
            errors.append({"system": name, "version": version, "task_id": r["task_id"], "family": item.family,
                           "item_type": item.item_type, "answerable": item.task.answerable,
                           "gold": item.task.gold_answer or "unanswerable", "answer": (r["final_answer"] or "")[:120],
                           "failure_category": cat, "steps": r["steps"], "evidence": _evidence(r),
                           "suggested_fix": FIXES.get(cat, ""), "trace_id": r.get("trace_id")})
    with (out_dir / "error_analysis.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(errors[0].keys()) if errors else ["system"])
        w.writeheader()
        w.writerows(errors)

    # Robustness: base test tasks vs their augmented variants
    robustness = {}
    for version, name in systems:
        base, aug = load(version, "test"), load(version, "test_augmented")
        if not base or not aug:
            continue
        base_tasks = {t: v for t, v in base["per_task"].items() if items[t].item_type == "base"}
        base_rate = sum(v["correct"] / v["runs"] for v in base_tasks.values()) / max(len(base_tasks), 1)
        robustness[name] = {"base_success": round(base_rate, 4), "augmented_success": aug["metrics"]["success"],
                            "augmented_ci": [aug["metrics"]["success_ci_low"], aug["metrics"]["success_ci_high"]],
                            "by_augmentation": aug["slices"]["item_type"]}
    (out_dir / "robustness.json").write_text(json.dumps(robustness, indent=2) + "\n", encoding="utf-8", newline="\n")

    # Values the container validation and the UI compare against
    gold = load(final_version, "gold")
    if gold:
        per_task = {t: v["correct"] / v["runs"] for t, v in gold["per_task"].items()}
        subset = sorted(per_task)[:UI_SUBSET]
        reported = {
            "final_version": final_version, "config_hash": gold["config_hash"], "model": gold["model"],
            "gold": {"success": gold["metrics"]["success"], "ci_95": [gold["metrics"]["success_ci_low"],
                                                                      gold["metrics"]["success_ci_high"]],
                     "n_tasks": gold["metrics"]["n_tasks"], "per_task": per_task,
                     "mlflow_run_id": gold.get("mlflow_run_id"), "tolerance": 0.04},
            "headline": {split: entries.get(final_version, {}).get("success")
                         for split, entries in comparison["splits"].items()},
            "ui_validation_subset": {"task_ids": subset, "per_task": {t: bool(per_task[t]) for t in subset},
                                     "success": round(sum(per_task[t] for t in subset) / len(subset), 3)},
        }
        REPORTED.write_text(json.dumps(reported, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(md))
    print(f"\n{len(errors)} test errors written to {(out_dir / 'error_analysis.csv').relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()

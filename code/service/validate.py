"""Offline validation of the deployed version on the full gold test set (Part D).

    docker run --rm --network none agentevalops:a2 validate
    python -m code.service.validate            # same, outside Docker

Verifies the gold checksums, runs the deployed configuration (registry alias
AGENTEVALOPS_ALIAS, default production) on all 25 gold tasks from scratch (no
cached runs), prints the metrics, writes output/validation.json and compares
them with the values in the report (supporting_files/a2/reported_values.json).

Tolerance (stated in the report and DOCKER.md): gold success within
TOLERANCE_SUCCESS (one task of 25). On the machine that produced the report the
answers are identical; another CPU may pick a different llama.cpp CPU kernel,
which can change a greedy decode. Exit code 1 if outside the tolerance.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from code.agent.config import REPO_ROOT
from code.evaluation.evaluate import evaluate
from code.registry.registry import Registry

REPORTED = REPO_ROOT / "supporting_files" / "a2" / "reported_values.json"
TOLERANCE_SUCCESS = 0.04


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--version", default=os.getenv("AGENTEVALOPS_ALIAS", "production"))
    p.add_argument("--out", type=Path, default=REPO_ROOT / "output")
    p.add_argument("--reason", default="offline container validation (docker run --network none ... validate)")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    reg = Registry(mlflow_sync=False)
    with tempfile.TemporaryDirectory() as runs:
        res = evaluate(args.version, "gold", reason=args.reason, registry=reg, runs_dir=Path(runs),
                       out_dir=args.out, label="container_validation")
    m = res["metrics"]
    reported = json.loads(REPORTED.read_text(encoding="utf-8")).get("gold", {}) if REPORTED.exists() else {}
    per_task_live = {t: v["correct"] / v["runs"] for t, v in res["per_task"].items()}
    per_task_rep = reported.get("per_task", {})
    agree = [per_task_live[t] == per_task_rep[t] for t in per_task_live if t in per_task_rep]
    diff = None if "success" not in reported else round(m["success"] - reported["success"], 4)
    within = diff is not None and abs(diff) <= TOLERANCE_SUCCESS + 1e-9
    out = {
        "version": res["registry_version"] or res["label"], "config_hash": res["config_hash"], "model": res["model"],
        "git_commit": res["git_commit"], "n_tasks": m["n_tasks"],
        "success": m["success"], "ci_95": [m["success_ci_low"], m["success_ci_high"]],
        "mean_steps": m["mean_steps"], "mean_total_tokens": m["mean_total_tokens"],
        "p50_latency_s": round(m["p50_wall_ms"] / 1000, 2), "p95_latency_s": round(m["p95_wall_ms"] / 1000, 2),
        "failure_categories": m["failure_categories"],
        "reported_success": reported.get("success"), "difference": diff, "tolerance": TOLERANCE_SUCCESS,
        "within_tolerance": within, "per_task_agreement": round(sum(agree) / len(agree), 4) if agree else None,
        "per_task": per_task_live, "wall_s": res["eval_wall_s"],
    }
    (args.out / "validation.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in out.items() if k != "per_task"}, indent=2))
    print("VALIDATION", "PASSED" if within else "FAILED", f"(gold success {m['success']:.3f}, reported "
          f"{reported.get('success')}, tolerance +/-{TOLERANCE_SUCCESS})")
    return 0 if within else 1


if __name__ == "__main__":
    raise SystemExit(main())

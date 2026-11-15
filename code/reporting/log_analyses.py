"""Log every A2 analysis output as an MLflow run, so each reported result has a run ID.

    python -m code.reporting.log_analyses

Evaluations, search trials and gate checks are MLflow runs already
(code/evaluation/evaluate.py). The analyses built on top of them - the Part A and
Part C reports with paired tests, the search summary, the categoriser agreement,
the drift replay, the confidence model, each retraining decision, the container
validation and the load test - are logged here as runs in the experiment
"AgentEvalOps-A2-analysis". Each run stores the output files as artifacts, the
headline numbers as metrics, the git commit, and the SHA-256 of its inputs.

Re-running is idempotent: a run whose inputs are unchanged is reused, not duplicated.
The run IDs are written to supporting_files/a2/mlflow/analysis_runs.json, which
code/reporting/evidence_index.py reads.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from code.agent.config import REPO_ROOT
from code.evaluation.evaluate import _safe, git_commit
from code.tracing.mlflow_tracer import DEFAULT_TRACKING_URI

A2 = REPO_ROOT / "supporting_files" / "a2"
EXPERIMENT = "AgentEvalOps-A2-analysis"
OUT = A2 / "mlflow" / "analysis_runs.json"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _num(x: Any) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def analyses() -> list[dict[str, Any]]:
    """Name, input files and headline metrics of every analysis that exists on disk."""
    out: list[dict[str, Any]] = []

    def add(name: str, files: list[Path], metrics: dict[str, Any], tags: dict[str, str] | None = None) -> None:
        files = [f for f in files if f.exists()]
        if files:
            out.append({"name": name, "files": files, "tags": tags or {},
                        "metrics": {k: v for k, v in ((k, _num(v)) for k, v in metrics.items()) if v is not None}})

    a = _load(A2 / "partA" / "summary.json")
    if a:
        m = {f"success.{s['system']}": s["success"] for s in a["systems"]}
        m.update({f"diff.{c['b']}_vs_{c['a']}": c.get("diff_mean") for c in a["paired_comparisons"]})
        add("partA_report", [A2 / "partA" / "summary.json", A2 / "partA" / "table.md"], m)
    s = _load(A2 / "search" / "summary.json")
    if s:
        m = {"n_trials": s["n_trials"], "selected_success": s["selected"]["success"]}
        m.update({f"best.{k}": v for k, v in s["best_by_sampler"].items()})
        add("search_summary", [A2 / "search" / f for f in ("summary.json", "trials.csv", "pareto.png",
                                                            "best_so_far.png", "param_effects.png",
                                                            "param_effects.json")], m)
    for folder, label in (("partC", "partC_report_v5"), ("partC_v7", "partC_report_v7")):
        comp = _load(A2 / folder / "comparison.json")
        if comp:
            m = {}
            for split, entries in comp["splits"].items():
                for e in entries.values():
                    m[f"success.{split}.{e['name']}"] = e["success"]
                    c = e.get("final_vs_this") or {}
                    if c.get("n_pairs"):
                        m[f"diff.{split}.final_vs_{e['name']}"] = c["diff_mean"]
                        m[f"mcnemar_p.{split}.final_vs_{e['name']}"] = c["mcnemar_exact_p"]
            add(label, [A2 / folder / f for f in ("comparison.json", "comparison.md", "slices.json",
                                                  "error_analysis.csv", "robustness.json",
                                                  "error_analysis_summary.md")], m,
                {"final_version": comp["final_version"]})
    k = _load(A2 / "categoriser_vs_human.json")
    if k:
        add("categoriser_agreement", [A2 / "categoriser_vs_human.json"],
            {f"kappa.{sp}": r["kappa_category"] for sp, r in k.items()})
    mon = _load(A2 / "monitoring" / "summary.json")
    if mon:
        m = {}
        for name, sc in mon["scenarios"].items():
            for sig, d in sc.items():
                if not isinstance(d, dict):
                    continue
                m[f"rho.{name}.{sig}"] = d.get("spearman_rolling_vs_success")
                for det in ("cusum", "page_hinkley"):
                    if det in d:
                        m[f"delay.{name}.{sig}.{det}"] = d[det].get("delay")
                        m[f"false_alarms.{name}.{sig}.{det}"] = d[det].get("false_alarms")
        files = [A2 / "monitoring" / f.name for f in sorted((A2 / "monitoring").glob("*"))]
        add("drift_replay", files + [REPO_ROOT / "configs" / "monitoring_thresholds.json"], m,
            {"version": mon["version"]})
    conf = _load(REPO_ROOT / "configs" / "confidence_model.json")
    if conf:
        add("confidence_model", [REPO_ROOT / "configs" / "confidence_model.json"],
            {"brier": conf["validation"]["brier"], "ece": conf["validation"]["ece"],
             "base_rate_brier": conf["validation"]["base_rate_brier"]})
    for rep_path in sorted((A2 / "retrain").glob("*/report.json")):
        r = _load(rep_path)
        g = r.get("gate_check", {})
        add(f"retrain_{rep_path.parent.name}", [rep_path, A2 / "retrain" / "gate_log.jsonl"],
            {k: g.get(k) for k in ("production_shifted", "candidate_shifted", "shifted_gain",
                                   "production_gold", "candidate_gold", "gold_drop")},
            {"decision": r.get("decision", ""), **{f"run.{k}": v for k, v in g.get("mlflow_runs", {}).items()}})
    v = _load(A2 / "docker_validation.json")
    if v:
        add("container_validation", [A2 / "docker_validation.json"],
            {"success": v["success"], "difference": v["difference"], "per_task_agreement": v["per_task_agreement"]},
            {"container_mlflow_run": "a1b0e7ee685d41a4b4dfd8d1a493587c", "tolerance": str(v["tolerance"])})
    lt = _load(A2 / "loadtest.json")
    if lt:
        add("load_test", [A2 / "loadtest.json"],
            {k: lt.get(k) for k in ("p50_latency_s", "p95_latency_s", "throughput_req_per_min",
                                    "memory_mib_peak", "memory_mib_mean")},
            {"hardware": lt["client_hardware"]["cpu"]})
    return out


def _sha(files: list[Path]) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(REPO_ROOT).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def log_all() -> dict[str, str]:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    exp = mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    ids: dict[str, str] = {}
    for a in analyses():
        sha = _sha(a["files"])
        found = client.search_runs([exp.experiment_id],
                                   filter_string=f"tags.analysis = '{a['name']}' and tags.inputs_sha256 = '{sha}'",
                                   max_results=1)
        if found:
            ids[a["name"]] = found[0].info.run_id
            continue
        with mlflow.start_run(run_name=a["name"]) as run:
            mlflow.set_tags({"analysis": a["name"], "inputs_sha256": sha, "git_commit": git_commit(),
                             "files": ", ".join(f.relative_to(REPO_ROOT).as_posix() for f in a["files"]),
                             **{k: str(v)[:500] for k, v in a["tags"].items()}})
            if a["metrics"]:
                mlflow.log_metrics({_safe(k): v for k, v in a["metrics"].items()})
            for f in a["files"]:
                mlflow.log_artifact(str(f), artifact_path=f.parent.relative_to(REPO_ROOT).as_posix())
            ids[a["name"]] = run.info.run_id
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ids, indent=2) + "\n", encoding="utf-8", newline="\n")
    return ids


def main() -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    ids = log_all()
    print(json.dumps(ids, indent=2))


if __name__ == "__main__":
    main()
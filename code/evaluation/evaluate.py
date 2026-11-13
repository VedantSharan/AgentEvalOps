"""The single A2 evaluation script (Part C): one registered version on one split.

    python -m code.evaluation.evaluate --version production --split validation
    python -m code.evaluation.evaluate --version configs/b2_original.yaml --split shifted_test --repeats 3
    python -m code.evaluation.evaluate --version v5 --split gold --reason "final evaluation"

``--version`` is a registry alias, a registry version or a configs/*.yaml path.
Every (task, seed) run is cached in ``data/a2/runs/<config hash>.jsonl`` (DVC),
so re-evaluating a version on a split it has already run reuses the same
trajectories. Runs execute in a fixed order (repeat, then task id), with seed
203 + repeat, so the same command gives the same answers.

Outputs
-------
* ``supporting_files/a2/eval/<label>__<split>.json``: headline metrics with 95%
  task-clustered bootstrap CIs, pass^k, cost, latency, detector rates, A1 failure
  categories and slices (template family, answerability, augmentation type,
  fault type), plus the MLflow run id;
* one MLflow run (experiment ``AgentEvalOps-A2-eval``) with parameters, metrics,
  data version, git commit and the summary as an artifact; every agent run is
  also an MLflow trace;
* for ``--split gold``: the gold checksums are verified first and the evaluation
  is appended to TEST_SET_LOG.md (``--reason`` is required).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from code.agent.config import REPO_ROOT, AgentConfig
from code.agent.loop import run_agent
from code.augmentation.fault_injection import FaultInjector
from code.evaluation.categorise import categorise
from code.evaluation.data import EvalItem, load_split
from code.evaluation.gold import log_gold_evaluation
from code.evaluation.metrics import slice_table, summarise
from code.registry.registry import Registry
from code.tracing.agent_tracer import AgentTracer
from code.tracing.mlflow_tracer import DEFAULT_TRACKING_URI

RUNS_DIR = REPO_ROOT / "data" / "a2" / "runs"
OUT_DIR = REPO_ROOT / "supporting_files" / "a2" / "eval"
EVAL_EXPERIMENT = "AgentEvalOps-A2-eval"
SEED_BASE = 203
SLICE_KEYS = ("family", "answerability", "item_type", "fault")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                               capture_output=True, text=True).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.CalledProcessError):
        baked = REPO_ROOT / "GIT_COMMIT"  # written into the Docker image at build time
        if baked.exists():
            return baked.read_text(encoding="utf-8").strip()
        return os.getenv("GIT_COMMIT", "unknown")


def data_version() -> dict[str, str]:
    """DVC md5 hashes of the data the evaluation reads (from dvc.lock and .dvc pointers)."""
    out: dict[str, str] = {}
    lock_path = REPO_ROOT / "dvc.lock"
    if lock_path.exists():
        lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
        for stage in ("splits", "augment"):
            for o in lock.get("stages", {}).get(stage, {}).get("outs", []):
                if o["path"] in ("data/splits", "data/augmented/tasks_augmented.jsonl"):
                    out[o["path"]] = o["md5"]
    ptr = REPO_ROOT / "data" / "processed" / "tasks_revised.jsonl.dvc"
    if ptr.exists():
        out["data/processed/tasks_revised.jsonl"] = yaml.safe_load(ptr.read_text(encoding="utf-8"))["outs"][0]["md5"]
    return out


# ---------------------------------------------------------------------------
# Run cache
# ---------------------------------------------------------------------------


class RunCache:
    """Append-only JSONL of full run records for one configuration hash."""

    def __init__(self, config_hash: str, runs_dir: Path = RUNS_DIR) -> None:
        self.path = runs_dir / f"{config_hash[:16]}.jsonl"
        self.records: dict[tuple[str, int], dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.records[(r["task_id"], r["seed"])] = r

    def get(self, task_id: str, seed: int) -> dict[str, Any] | None:
        return self.records.get((task_id, seed))

    def add(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        self.records[(record["task_id"], record["seed"])] = record


def run_item(item: EvalItem, cfg: AgentConfig, seed: int, tracer: AgentTracer, client: Any = None) -> dict[str, Any]:
    """Run one item (with its injected faults) and return the full record."""
    ctx = FaultInjector(item.faults) if item.faults else contextlib.nullcontext()
    with ctx:
        run = run_agent(item.task, cfg, client=client, tracer=tracer, seed=seed)
    record = run.to_json()
    record["failure_category"] = categorise(record["trajectory"], item.task, run.correct)
    return record


def row_for(record: dict[str, Any], item: EvalItem, split: str) -> dict[str, Any]:
    keep = {k: v for k, v in record.items() if k not in ("turns", "tool_call_log", "trajectory")}
    return {**keep, "split": split, "slices": item.slices(), "template_family": item.family}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(ref: str, split: str, repeats: int = 1, reason: str = "", trace: bool = True,
             limit: int | None = None, out_dir: Path = OUT_DIR, runs_dir: Path = RUNS_DIR, client: Any = None,
             registry: Registry | None = None, log_mlflow: bool = True, label: str | None = None,
             progress: bool = True) -> dict[str, Any]:
    if split == "gold" and not reason.strip():
        raise SystemExit("--reason is required for every gold-test evaluation (TEST_SET_LOG.md)")
    registry = registry or Registry()
    version, cfg = registry.resolve(ref)
    label = label or (version if version.startswith("v") else cfg.name)
    items = sorted(load_split(split), key=lambda i: i.task_id)
    if limit:
        items = items[:limit]
    cache = RunCache(cfg.config_hash(), runs_dir)
    tracer = AgentTracer(enabled=trace)

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    new_runs = 0
    for r in range(repeats):
        seed = SEED_BASE + r
        for item in items:
            record = cache.get(item.task_id, seed)
            if record is None:
                record = run_item(item, cfg, seed, tracer, client=client)
                cache.add(record)
                new_runs += 1
            rows.append(row_for(record, item, split))
            if progress:
                print(f"[{label}/{split}] seed={seed} {item.task_id:45s} correct={record['correct']!s:5s} "
                      f"steps={record['steps']} tokens={record['total_tokens']} wall={record['wall_ms'] / 1000:.1f}s",
                      flush=True)
    tracer.flush()
    elapsed = time.perf_counter() - t0

    summary = summarise(rows)
    result: dict[str, Any] = {
        "label": label, "registry_ref": ref, "registry_version": version if version.startswith("v") else None,
        "config_name": cfg.name, "config_hash": cfg.config_hash(), "model": cfg.model, "split": split,
        "repeats": repeats, "seeds": [SEED_BASE + r for r in range(repeats)], "n_items": len(items),
        "new_runs": new_runs, "eval_wall_s": round(elapsed, 1), "git_commit": git_commit(),
        "data_version": data_version(), "runs_file": _rel(cache.path),
        "config": cfg.model_dump(), "metrics": summary,
        "slices": {k: slice_table(rows, k) for k in SLICE_KEYS},
        "per_task": _per_task(rows),
    }
    if log_mlflow:
        result["mlflow_run_id"] = _log_mlflow(result, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{label}__{split}.json"
    out.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    result["output_file"] = _rel(out)
    if split == "gold":
        log_gold_evaluation(version=label, config_hash=cfg.config_hash(), model=cfg.model, reason=reason,
                            success=summary["success"], run_id=result.get("mlflow_run_id", "-"))
    return result


def _rel(path: Path) -> str:
    """Repository-relative POSIX path (absolute only for paths outside the repository)."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _per_task(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        t = out.setdefault(r["task_id"], {"runs": 0, "correct": 0, "answers": [], "failure_categories": []})
        t["runs"] += 1
        t["correct"] += int(r["correct"])
        t["answers"].append(r["final_answer"])
        t["failure_categories"].append(r.get("failure_category"))
    return out


def _flat_metrics(result: dict[str, Any]) -> dict[str, float]:
    m = result["metrics"]
    flat = {_safe(k): float(v) for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    for key, table in result["slices"].items():
        for value, stats in table.items():
            flat[_safe(f"slice.{key}.{value}.success")] = float(stats["success"])
    return flat


def _safe(name: str) -> str:
    """MLflow metric names allow only alphanumerics and _ - . / space."""
    return "".join(c if c.isalnum() or c in "_-./ " else "_" for c in name)


def _log_mlflow(result: dict[str, Any], cfg: AgentConfig) -> str:
    import mlflow

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(EVAL_EXPERIMENT)
    with mlflow.start_run(run_name=f"{result['label']}-{result['split']}") as run:
        params = {f"cfg.{k}": v for k, v in cfg.model_dump().items() if k != "description"}
        params.update({"split": result["split"], "repeats": result["repeats"], "seeds": result["seeds"],
                       "n_items": result["n_items"]})
        mlflow.log_params(params)
        mlflow.set_tags({
            "git_commit": result["git_commit"], "config_hash": result["config_hash"],
            "registry_version": result["registry_version"] or "", "label": result["label"],
            **{f"data.{k}": v for k, v in result["data_version"].items()},
            "runs_file": result["runs_file"],
        })
        mlflow.log_metrics(_flat_metrics(result))
        mlflow.log_dict({k: v for k, v in result.items() if k != "per_task"}, "summary.json")
        mlflow.log_dict(result["per_task"], "per_task.json")
        return run.info.run_id


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    p = argparse.ArgumentParser(description="Evaluate one registered version on one split.")
    p.add_argument("--version", required=True, help="registry alias, version (v3) or configs/*.yaml")
    p.add_argument("--split", required=True)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--reason", default="", help="required for --split gold")
    p.add_argument("--limit", type=int, default=None, help="first N items only (smoke tests)")
    p.add_argument("--label", default=None, help="name used in output files (default: version or config name)")
    p.add_argument("--no-trace", action="store_true")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args(argv)
    res = evaluate(args.version, args.split, repeats=args.repeats, reason=args.reason, trace=not args.no_trace,
                   limit=args.limit, out_dir=args.out_dir, label=args.label)
    m = res["metrics"]
    print(json.dumps({"label": res["label"], "split": res["split"], "success": m["success"],
                      "ci": [m["success_ci_low"], m["success_ci_high"]], "pass_k": m["pass_k"],
                      "mean_steps": m["mean_steps"], "mean_total_tokens": m["mean_total_tokens"],
                      "p50_wall_ms": m["p50_wall_ms"], "p95_wall_ms": m["p95_wall_ms"],
                      "mlflow_run_id": res.get("mlflow_run_id"), "output": res["output_file"]}, indent=2))



def rows_for(ref: str, split: str, repeats: int = 1, registry: Registry | None = None,
             run_missing: bool = True) -> list[dict[str, Any]]:
    """Run summaries (with slices) of ``ref`` on ``split`` from the run cache.

    Missing runs are executed through ``evaluate`` first (so they are logged like
    any other evaluation) unless ``run_missing`` is False. Gold is not allowed
    here: monitoring and retraining code must never read gold results.
    """
    if split == "gold":
        raise ValueError("rows_for() must not be used on the gold split")
    registry = registry or Registry()
    _, cfg = registry.resolve(ref)
    items = sorted(load_split(split), key=lambda i: i.task_id)
    cache = RunCache(cfg.config_hash())
    seeds = [SEED_BASE + r for r in range(repeats)]
    if run_missing and any(cache.get(i.task_id, s) is None for i in items for s in seeds):
        evaluate(ref, split, repeats=repeats, registry=registry)
        cache = RunCache(cfg.config_hash())
    out = []
    for s in seeds:
        for item in items:
            record = cache.get(item.task_id, s)
            if record is not None:
                out.append(row_for(record, item, split))
    return out


if __name__ == "__main__":
    main()

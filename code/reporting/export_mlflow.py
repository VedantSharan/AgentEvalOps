"""Export the MLflow store for the submission (A2 Part F: run/registry exports).

    python -m code.reporting.export_mlflow --out ../artefacts/mlflow

Writes:
  runs.csv              every MLflow run (experiment, run id, name, parameters, metrics, tags)
  registry.json         registered model versions of agentevalops-agent with their aliases,
                        next to the bundled registry (registry/aliases.json, history.jsonl)
  traces_summary.json   number of traces and spans per experiment
  mlflow.db + artifacts a copy of the SQLite store and the per-experiment artifact folders, viewable with
                        mlflow ui --backend-store-uri sqlite:///<out>/mlflow.db
A copy of runs.csv and registry.json also goes to supporting_files/a2/mlflow/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

from code.agent.config import REPO_ROOT
from code.registry.registry import MLFLOW_MODEL_NAME, Registry
from code.tracing.mlflow_tracer import DEFAULT_TRACKING_URI

EVIDENCE = REPO_ROOT / "supporting_files" / "a2" / "mlflow"


def export(out: Path) -> dict[str, Any]:
    import mlflow
    from mlflow import MlflowClient

    uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    out.mkdir(parents=True, exist_ok=True)
    rows, traces = [], {}
    for exp in client.search_experiments():
        for run in client.search_runs([exp.experiment_id], max_results=50000):
            rows.append({"experiment": exp.name, "run_id": run.info.run_id, "run_name": run.info.run_name,
                         "start_time": run.info.start_time, "status": run.info.status,
                         "params": json.dumps(run.data.params, sort_keys=True),
                         "metrics": json.dumps(run.data.metrics, sort_keys=True),
                         "tags": json.dumps({k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
                                            sort_keys=True)})
        try:
            found = mlflow.search_traces(locations=[exp.experiment_id], max_results=100000, return_type="list")
            traces[exp.name] = {"traces": len(found), "spans": sum(len(t.data.spans) for t in found)}
        except Exception as exc:  # noqa: BLE001 - traces are optional per experiment
            traces[exp.name] = {"error": str(exc)}
    with (out / "runs.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["run_id"])
        w.writeheader()
        w.writerows(rows)
    versions = []
    for mv in client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'"):
        versions.append({"version": mv.version, "source": mv.source, "run_id": mv.run_id, "tags": dict(mv.tags),
                         "aliases": list(getattr(mv, "aliases", []) or [])})
    reg = Registry(mlflow_sync=False)
    registry = {"mlflow_registered_model": MLFLOW_MODEL_NAME, "mlflow_versions": sorted(versions, key=lambda v: int(v["version"])),
                "bundled_aliases": reg.aliases(), "bundled_history": reg.history()}
    (out / "registry.json").write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8", newline="\n")
    (out / "traces_summary.json").write_text(json.dumps(traces, indent=2) + "\n", encoding="utf-8", newline="\n")
    if uri.startswith("sqlite:///"):
        db = Path(uri.replace("sqlite:///", "", 1))
        if db.exists():
            shutil.copyfile(db, out / "mlflow.db")
            # Artifacts live next to the database, one folder per experiment id. Their absolute
            # locations are recorded in the database, so runs.csv/registry.json are the portable
            # export; the folders are copied for completeness.
            for sub in db.parent.iterdir():
                if sub.is_dir():
                    shutil.copytree(sub, out / "artifacts" / sub.name, dirs_exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    for name in ("runs.csv", "registry.json", "traces_summary.json"):
        shutil.copyfile(out / name, EVIDENCE / name)
    return {"runs": len(rows), "model_versions": len(versions), "traces": traces}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=REPO_ROOT / "output" / "mlflow_export")
    args = p.parse_args(argv)
    print(json.dumps(export(args.out), indent=2))


if __name__ == "__main__":
    main()

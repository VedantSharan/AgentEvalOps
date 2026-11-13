"""Evidence index for the A2 report: every reported result -> file -> MLflow run ID.

    python -m code.reporting.evidence_index

Collects the headline numbers from supporting_files/a2/ and writes
supporting_files/a2/evidence_index.csv (id, part, result, value, file, MLflow run).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from code.agent.config import REPO_ROOT

A2 = REPO_ROOT / "supporting_files" / "a2"
OUT = A2 / "evidence_index.csv"


def _load(rel: str) -> Any:
    path = A2 / rel
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(part: str, result: str, value: Any, file: str, run: str | None = None) -> None:
        rows.append({"id": f"E{len(rows) + 1:03d}", "part": part, "result": result, "value": value,
                     "file": f"supporting_files/a2/{file}", "mlflow_run_id": run or ""})

    part_a = _load("partA/summary.json")
    if part_a:
        for s in part_a["systems"]:
            add("A", f"validation success, {s['description']}", f"{s['success']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]",
                "partA/summary.json", s["mlflow_run_id"])
        for c in part_a["paired_comparisons"]:
            if c.get("n_pairs"):
                add("A", f"paired test {c['b']} vs {c['a']} (validation)",
                    f"diff {c['diff_mean']:+.3f}, McNemar p={c['mcnemar_exact_p']}, bootstrap p={c['paired_bootstrap_p']}",
                    "partA/summary.json")
    search = _load("search/summary.json")
    if search:
        sel = search["selected"]
        add("B", "search trials (random + NSGA-II + manual seeds)", search["n_trials"], "search/summary.json")
        add("B", f"selected configuration {sel['name']}: validation success", sel["success"], "search/summary.json",
            sel.get("mlflow_run_id"))
        for sampler, best in search["best_by_sampler"].items():
            add("B", f"best feasible validation success, {sampler}", best, "search/summary.json")
    comp = _load("partC/comparison.json")
    if comp:
        for split, entries in comp["splits"].items():
            for version, e in entries.items():
                add("C", f"{split} success, {e['name']} ({version})", f"{e['success']:.3f} [{e['ci'][0]:.3f}, {e['ci'][1]:.3f}]",
                    "partC/comparison.json", e.get("mlflow_run_id"))
                c = e.get("final_vs_this")
                if c and c.get("n_pairs"):
                    add("C", f"{split}: final vs {e['name']}", f"diff {c['diff_mean']:+.3f}, McNemar p={c['mcnemar_exact_p']}",
                        "partC/comparison.json")
    if (A2 / "partC" / "error_analysis.csv").exists():
        with (A2 / "partC" / "error_analysis.csv").open(encoding="utf-8", newline="") as fh:
            n = sum(1 for _ in csv.DictReader(fh))
        add("C", "categorised test errors", n, "partC/error_analysis.csv")
    cat = _load("categoriser_vs_human.json")
    if cat:
        for split, r in cat.items():
            add("C", f"failure-category kappa vs human labels ({split})", r["kappa_category"], "categoriser_vs_human.json")
    mon = _load("monitoring/summary.json")
    if mon:
        for name, sc in mon["scenarios"].items():
            for signal in mon["monitored_signals"]:
                for det in ("cusum", "page_hinkley"):
                    d = sc.get(signal, {}).get(det)
                    if d:
                        add("E", f"{name}: {signal}/{det} first alarm (delay) / false alarms",
                            f"{d.get('first_alarm')} ({d.get('delay')}) / {d.get('false_alarms')}", "monitoring/summary.json")
                rho = sc.get(signal, {}).get("spearman_rolling_vs_success")
                add("E", f"{name}: Spearman(rolling {signal}, rolling success)", rho, "monitoring/summary.json")
    gate_log = A2 / "retrain" / "gate_log.jsonl"
    if gate_log.exists():
        for line in gate_log.read_text(encoding="utf-8").splitlines():
            g = json.loads(line)
            add("E", f"retraining {g['tag']}: decision", g["decision"], "retrain/gate_log.jsonl")
    for name, rel in (("container gold validation", "docker_validation.json"), ("CPU load test", "loadtest.json")):
        d = _load(rel)
        if d:
            keys = ("success", "within_tolerance", "p50_latency_s", "p95_latency_s", "throughput_req_per_min",
                    "memory_mib_peak")
            add("D", name, json.dumps({k: d[k] for k in keys if k in d}), rel)
    return rows


def main() -> None:
    rows = collect()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "part", "result", "value", "file", "mlflow_run_id"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} entries -> {OUT.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()

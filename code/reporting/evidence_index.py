"""Evidence index for the A2 report: every reported result -> file -> MLflow run ID.

    python -m code.reporting.log_analyses     # log the analysis outputs as MLflow runs (once)
    python -m code.reporting.evidence_index

Writes supporting_files/a2/evidence_index.csv with one row per reported result:
  id, part, result, value, file, mlflow_run_id, source_runs

``mlflow_run_id`` is the run that holds the result: an evaluation run, or the
analysis run (experiment AgentEvalOps-A2-analysis) that stores the report file as an
artifact. ``source_runs`` lists the evaluation runs a derived result was computed
from (for example the two evaluations behind a paired test, or the four gate-check
evaluations behind a promotion decision). Every row has a run ID.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from code.agent.config import REPO_ROOT

A2 = REPO_ROOT / "supporting_files" / "a2"
OUT = A2 / "evidence_index.csv"
RUNS = A2 / "mlflow" / "analysis_runs.json"
FIELDS = ["id", "part", "result", "value", "file", "mlflow_run_id", "source_runs"]


def _load(rel: str) -> Any:
    path = A2 / rel
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def collect() -> list[dict[str, Any]]:
    runs = json.loads(RUNS.read_text(encoding="utf-8")) if RUNS.exists() else {}
    rows: list[dict[str, Any]] = []

    def add(part: str, result: str, value: Any, file: str, run: str | None, sources: list[str] | None = None) -> None:
        rows.append({"id": f"E{len(rows) + 1:03d}", "part": part, "result": result, "value": value,
                     "file": f"supporting_files/a2/{file}" if not file.startswith(("configs/", "TEST_SET_LOG")) else file,
                     "mlflow_run_id": run or "", "source_runs": ";".join(s for s in (sources or []) if s)})

    # Part A
    part_a = _load("partA/summary.json")
    if part_a:
        run_of = {s["system"]: s["mlflow_run_id"] for s in part_a["systems"]}
        for s in part_a["systems"]:
            add("A", f"validation success, {s['description']}",
                f"{s['success']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]", "partA/summary.json", s["mlflow_run_id"])
        for c in part_a["paired_comparisons"]:
            if c.get("n_pairs"):
                add("A", f"paired test {c['b']} vs {c['a']} (validation)",
                    f"diff {c['diff_mean']:+.3f}, McNemar p={c['mcnemar_exact_p']}, bootstrap p={c['paired_bootstrap_p']}",
                    "partA/summary.json", runs.get("partA_report"), [run_of.get(c["a"]), run_of.get(c["b"])])

    # Part B
    search = _load("search/summary.json")
    if search:
        sel = search["selected"]
        trial_runs = [t.get("mlflow_run_id") for t in search.get("pareto_front", [])]
        add("B", "search candidates (random + NSGA-II + stage-2 grid + Part A configurations)", search["n_trials"],
            "search/summary.json", runs.get("search_summary"))
        add("B", f"selected configuration {sel['name']}: validation success", sel["success"], "search/summary.json",
            sel.get("mlflow_run_id"))
        for sampler, best in search["best_by_sampler"].items():
            add("B", f"best feasible validation success, {sampler}", best, "search/summary.json",
                runs.get("search_summary"))
        add("B", "Pareto front (success, tokens, latency) and search plots",
            f"{len(search.get('pareto_front', []))} Pareto-optimal candidates", "search/pareto.png",
            runs.get("search_summary"), trial_runs)

    # Part C: the deployed version (v7) first, then the Part B selection (v5) before retraining
    for folder, label in (("partC_v7", "deployed v7"), ("partC", "Part B selection v5")):
        comp = _load(f"{folder}/comparison.json")
        if not comp:
            continue
        report_run = runs.get("partC_report_v7" if folder == "partC_v7" else "partC_report_v5")
        final = comp["final_version"]
        for split, entries in comp["splits"].items():
            final_run = entries.get(final, {}).get("mlflow_run_id")
            for version, e in entries.items():
                who = f"{label} (final)" if version == final else e["name"]
                add("C", f"[{label}] {split} success, {who} ({version})",
                    f"{e['success']:.3f} [{e['ci'][0]:.3f}, {e['ci'][1]:.3f}]", f"{folder}/comparison.json",
                    e.get("mlflow_run_id"))
                c = e.get("final_vs_this")
                if c and c.get("n_pairs"):
                    add("C", f"[{label}] {split}: final vs {e['name']} (paired)",
                        f"diff {c['diff_mean']:+.3f}, McNemar p={c['mcnemar_exact_p']}, bootstrap p={c['paired_bootstrap_p']}",
                        f"{folder}/comparison.json", report_run, [final_run, e.get("mlflow_run_id")])
        sl = _load(f"{folder}/slices.json")
        if sl:
            for w in sl["weakest_groups"][:4]:
                add("C", f"[{label}] weak group: {w['split']} {w['slice']}={w['value']}",
                    f"{w['success']:.3f} ({w['n_tasks']} tasks)", f"{folder}/slices.json", report_run)
        err = A2 / folder / "error_analysis.csv"
        if err.exists():
            with err.open(encoding="utf-8", newline="") as fh:
                n = sum(1 for _ in csv.DictReader(fh))
            add("C", f"[{label}] categorised test errors with suggested fixes", n, f"{folder}/error_analysis.csv",
                report_run)
        rob = _load(f"{folder}/robustness.json")
        if rob:
            for name, r in rob.items():
                add("C", f"[{label}] robustness, {name}: base test vs augmented test",
                    f"{r['base_success']:.3f} vs {r['augmented_success']:.3f}", f"{folder}/robustness.json", report_run)
    cat = _load("categoriser_vs_human.json")
    if cat:
        for split, r in cat.items():
            add("C", f"failure-category kappa vs human labels ({split}); no LLM judge is used", r["kappa_category"],
                "categoriser_vs_human.json", runs.get("categoriser_agreement"))

    # Part D
    v = _load("docker_validation.json")
    if v:
        add("D", "container validation, gold set offline (--network none)",
            f"success {v['success']} vs reported {v['reported_success']}, within +/-{v['tolerance']}: "
            f"{v['within_tolerance']}, per-task agreement {v['per_task_agreement']}",
            "docker_validation.json", runs.get("container_validation"), ["a1b0e7ee685d41a4b4dfd8d1a493587c"])
    lt = _load("loadtest.json")
    if lt:
        add("D", "CPU load test: p50 / p95 latency (s), throughput (req/min), peak memory (MiB)",
            f"{lt['p50_latency_s']} / {lt['p95_latency_s']}, {lt['throughput_req_per_min']}, {lt['memory_mib_peak']}",
            "loadtest.json", runs.get("load_test"))

    # Part E
    mon = _load("monitoring/summary.json")
    if mon:
        for name, sc in mon["scenarios"].items():
            for signal in mon["monitored_signals"]:
                for det in ("cusum", "page_hinkley"):
                    d = sc.get(signal, {}).get(det)
                    if d:
                        add("E", f"{name}: {signal}/{det} first alarm (delay) / false alarms",
                            f"{d.get('first_alarm')} ({d.get('delay')}) / {d.get('false_alarms')}",
                            "monitoring/summary.json", runs.get("drift_replay"))
            for signal, d in sc.items():
                if isinstance(d, dict) and d.get("spearman_rolling_vs_success") is not None:
                    add("E", f"{name}: Spearman(rolling {signal}, rolling success)",
                        d["spearman_rolling_vs_success"], "monitoring/summary.json", runs.get("drift_replay"))
        add("E", "drift thresholds chosen on validation", ", ".join(mon["monitored_signals"]),
            "configs/monitoring_thresholds.json", runs.get("drift_replay"))
    conf = json.loads((REPO_ROOT / "configs" / "confidence_model.json").read_text(encoding="utf-8")) \
        if (REPO_ROOT / "configs" / "confidence_model.json").exists() else None
    if conf:
        add("E", "answer confidence: validation Brier (base rate) / ECE",
            f"{conf['validation']['brier']} ({conf['validation']['base_rate_brier']}) / {conf['validation']['ece']}",
            "configs/confidence_model.json", runs.get("confidence_model"))
    for rep in sorted((A2 / "retrain").glob("*/report.json")):
        r = json.loads(rep.read_text(encoding="utf-8"))
        g = r.get("gate_check", {})
        add("E", f"retraining {rep.parent.name}: shifted_test and gold, production -> candidate",
            f"shifted {g.get('production_shifted')} -> {g.get('candidate_shifted')}, "
            f"gold {g.get('production_gold')} -> {g.get('candidate_gold')}; {r.get('decision')}",
            f"retrain/{rep.parent.name}/report.json", runs.get(f"retrain_{rep.parent.name}"),
            list(g.get("mlflow_runs", {}).values()))
    return rows


def main() -> None:
    rows = collect()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    missing = [r["id"] for r in rows if not r["mlflow_run_id"]]
    print(f"{len(rows)} entries -> {OUT.relative_to(REPO_ROOT).as_posix()}; without run ID: {missing or 'none'}")


if __name__ == "__main__":
    main()
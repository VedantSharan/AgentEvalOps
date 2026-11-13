"""Part E: one-command retraining after a drift alert.

    python -m code.retrain.pipeline --alert supporting_files/a2/monitoring/alerts.json --budget 16

Steps (all logged to supporting_files/a2/retrain/<run>/report.json and the gate log):

1. Trigger: an alert from the monitor (code/monitoring/replay.py writes alerts.json).
2. Pool: the fault-injected tasks whose parent family is in the TRAIN split
   (configs/gate.yaml: retrain_pool_parent_split), answered by the production
   version. Evaluation data (validation/test families, gold) never enters the pool.
3. Selection: the Assignment 1 active-learning score (code/active_learning/
   al_vs_random.py: steps, repeated calls, tool errors) ranks the pool's runs;
   the top ``budget`` are selected (seeded tie-break).
4. Labels: revealed from a pre-labelled pool (simulated annotator,
   data/a2/retrain/prelabelled_pool.csv): outcome by grader v2 against the gold
   answer, A1 failure category, fault type and the expected behaviour.
5. Validation: the A1 12-check suite on the selected tasks and trajectories, plus
   a leakage check against every evaluation split.
6. Retraining (= retuning prompts/configurations, as the Project 1 brief asks):
   a new exemplar bank adds worked examples from selected successes and pitfall
   notes from selected failures under faults; candidate configurations
   (new bank; new bank with one more retry) are scored on shifted_validation and
   the best is kept.
7. Evaluation: candidate and production on shifted_test (held-out families) and
   on gold (each gold evaluation is logged in TEST_SET_LOG.md).
8. Gate (configs/gate.yaml, fixed in advance): promote only if shifted_test
   success improves by at least min_shifted_gain and gold success drops by no
   more than gold_tolerance. Otherwise the candidate is registered but rejected.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from code.active_learning.al_vs_random import al_order, scores, trace_features
from code.agent.config import CONFIGS_DIR, PROMPTS_DIR, REPO_ROOT, AgentConfig, exemplars, load_config
from code.evaluation.data import load_split
from code.evaluation.evaluate import RunCache, evaluate, rows_for
from code.registry.registry import Registry
from code.shared.models import TaskRecord
from code.tracing.mlflow_tracer import DEFAULT_TRACKING_URI
from code.validation.benchmark_validator import BenchmarkValidator

GATE = CONFIGS_DIR / "gate.yaml"
OUT = REPO_ROOT / "supporting_files" / "a2" / "retrain"
POOL_LABELS = REPO_ROOT / "data" / "a2" / "retrain" / "prelabelled_pool.csv"
SEED = 203
FAULT_TEXT = {
    "transient_failure": "failed once with a 503 error and then worked again",
    "information_outage": "kept failing with 503 errors, so the information could not be obtained",
    "calculator_outage": "the calculate tool kept failing, so arithmetic had to be avoided or written out",
    "slow_response": "answered slowly (2 s) but correctly",
    "format_change": "returned the same data in a different format (CSV instead of JSON records, renamed keys)",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# Pool, labels, selection
# ---------------------------------------------------------------------------


def prelabel_pool(prod_ref: str, split: str, path: Path = POOL_LABELS) -> list[dict[str, Any]]:
    """Simulated annotator: label every pool run (written once, revealed per selection)."""
    items = {i.task_id: i for i in load_split(split)}
    rows = rows_for(prod_ref, split)
    labels = []
    for r in rows:
        item = items[r["task_id"]]
        labels.append({"run_id": r["run_id"], "task_id": r["task_id"], "config_hash": r["config_hash"],
                       "outcome": "success" if r["correct"] else "failure",
                       "failure_category": r.get("failure_category") or "", "fault": item.fault_label,
                       "gold_answer": item.task.gold_answer or "unanswerable",
                       "answerable": item.task.answerable})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(labels[0].keys()))
        w.writeheader()
        w.writerows(labels)
    return labels


def reveal(run_ids: list[str], path: Path = POOL_LABELS) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        pool = {r["run_id"]: r for r in csv.DictReader(fh)}
    missing = [r for r in run_ids if r not in pool]
    if missing:
        raise KeyError(f"runs without a pre-label: {missing}")
    return {r: pool[r] for r in run_ids}


def select(records: list[dict[str, Any]], budget: int, seed: int = SEED) -> list[dict[str, Any]]:
    feats = [trace_features(r["trajectory"]) for r in records]
    order = al_order(scores(feats), np.random.default_rng(seed))
    picked = [records[i] for i in order[:budget]]
    for rank, r in enumerate(picked, 1):
        r["al_rank"] = rank
        r["al_features"] = trace_features(r["trajectory"])
    return picked


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_selection(selected: list[dict[str, Any]], pool_split: str) -> dict[str, Any]:
    items = {i.task_id: i for i in load_split(pool_split)}
    eval_ids: set[str] = set()
    eval_families: set[str] = set()
    for split in ("validation", "test", "shifted_validation", "shifted_test"):
        for i in load_split(split):
            eval_ids.add(i.task_id)
            eval_families.add(i.family)
    with tempfile.TemporaryDirectory() as tmp:
        tasks_path, trajs_path = Path(tmp) / "tasks.jsonl", Path(tmp) / "trajs.jsonl"
        envs = {}
        with tasks_path.open("w", encoding="utf-8") as ft, trajs_path.open("w", encoding="utf-8") as fr:
            for r in selected:
                item = items[r["task_id"]]
                ft.write(item.task.model_dump_json() + "\n")
                fr.write(json.dumps(r["trajectory"]) + "\n")
                envs[item.task_id] = json.dumps([f.to_dict() for f in item.faults], sort_keys=True)
        suite = BenchmarkValidator(tasks_path, trajs_path, environments=envs).run_all_checks()
    leaked_ids = sorted({r["task_id"] for r in selected} & eval_ids)
    leaked_families = sorted({items[r["task_id"]].family for r in selected} & eval_families)
    return {"suite_passed": suite.passed, "checks_passed": suite.checks_passed,
            "failed_checks": sorted(k for k, v in suite.summary_by_check.items() if not v),
            "issues": [i.message for i in suite.issues][:10],
            "leaked_task_ids": leaked_ids, "leaked_families": leaked_families,
            "passed": suite.passed and not leaked_ids and not leaked_families}


# ---------------------------------------------------------------------------
# Retuning
# ---------------------------------------------------------------------------


def _calls(traj: dict[str, Any]) -> list[dict[str, Any]]:
    from code.agent.exemplars import _calls as calls

    return calls(traj)


def build_bank(prod: AgentConfig, selected: list[dict[str, Any]], labels: dict[str, dict[str, str]],
               pool_split: str, name: str) -> Path:
    items = {i.task_id: i for i in load_split(pool_split)}
    bank = list(exemplars(prod))
    used_faults_ok: set[str] = set()
    used_faults_fail: set[str] = set()
    for r in selected:
        lab = labels[r["run_id"]]
        item = items[r["task_id"]]
        fault = lab["fault"]
        if lab["outcome"] == "success" and fault not in used_faults_ok:
            bank.append({"type": "example", "family": item.family, "source_run_id": r["run_id"],
                         "task": item.task.prompt, "calls": _calls(r["trajectory"]),
                         "answer": lab["gold_answer"],
                         "note": f"Here the tool {FAULT_TEXT.get(fault, fault)}."})
            used_faults_ok.add(fault)
        elif lab["outcome"] == "failure" and fault not in used_faults_fail:
            bank.append({"type": "pitfall", "category": lab["failure_category"], "source_run_id": r["run_id"],
                         "text": (f"On the task '{item.task.prompt[:90]}...', a tool {FAULT_TEXT.get(fault, fault)}; "
                                  f"an earlier run answered '{(r.get('final_answer') or 'nothing')[:60]}' "
                                  f"but the correct answer was '{lab['gold_answer']}'.")})
            used_faults_fail.add(fault)
    path = PROMPTS_DIR / "exemplars" / f"{name}.json"
    path.write_text(json.dumps(bank, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def candidate_configs(prod: AgentConfig, bank: str, tag: str) -> list[Path]:
    variants = [
        {"exemplars": bank},
        {"exemplars": bank, "retry_budget": min(prod.retry_budget + 1, 5)},
    ]
    paths = []
    for i, upd in enumerate(variants):
        cfg = prod.model_copy(update={**upd, "name": f"{tag}_c{i}",
                                      "description": f"Retraining candidate {i} from {prod.name}: {json.dumps(upd)}"})
        path = CONFIGS_DIR / "retrain" / f"{cfg.name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {cfg.description}\n{cfg.to_yaml()}", encoding="utf-8", newline="\n")
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run(alert_path: Path | None, budget: int, force: bool = False, tag: str | None = None) -> dict[str, Any]:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    reg = Registry()
    tag = tag or f"retrain_{_now()}"
    report: dict[str, Any] = {"tag": tag, "budget": budget, "gate": gate, "seed": SEED}

    alerts = json.loads(alert_path.read_text(encoding="utf-8")) if alert_path and alert_path.exists() else []
    report["trigger"] = alerts[0] if alerts else None
    if not alerts and not force:
        raise SystemExit("no drift alert: nothing to do (use --force to retrain anyway)")

    prod_version, prod = reg.resolve("production")
    report["production"] = {"version": prod_version, "config_hash": prod.config_hash()}
    pool_split = f"shifted_{gate['retrain_pool_parent_split']}"
    eval_split = f"shifted_{gate['shifted_eval_parent_split']}"
    selection_split = "shifted_validation"

    # 2-4. Pool, selection, labels
    if not POOL_LABELS.exists() or POOL_LABELS.read_text(encoding="utf-8").count(prod.config_hash()) == 0:
        prelabel_pool("production", pool_split)
    rows_for("production", pool_split)
    cache = RunCache(prod.config_hash())
    pool = [dict(cache.get(i.task_id, SEED)) for i in load_split(pool_split)]
    selected = select(pool, budget)
    labels = reveal([r["run_id"] for r in selected])
    report["selection"] = [{"rank": r["al_rank"], "run_id": r["run_id"], "task_id": r["task_id"],
                            **r["al_features"], "label": labels[r["run_id"]]["outcome"],
                            "fault": labels[r["run_id"]]["fault"]} for r in selected]

    # 5. Validation
    report["validation"] = validation = validate_selection(selected, pool_split)
    if not validation["passed"]:
        report["decision"] = "aborted: selected data failed validation"
        return _finish(report, tag)

    # 6. Retune
    bank_path = build_bank(prod, selected, labels, pool_split, f"{tag}_bank")
    report["exemplar_bank"] = bank_path.relative_to(REPO_ROOT).as_posix()
    scored = []
    for path in candidate_configs(prod, bank_path.stem, tag):
        res = evaluate(str(path), selection_split, label=path.stem, out_dir=OUT / tag / "candidates")
        scored.append({"config": path.relative_to(REPO_ROOT).as_posix(), "success": res["metrics"]["success"],
                       "mean_total_tokens": res["metrics"]["mean_total_tokens"], "mlflow_run_id": res.get("mlflow_run_id")})
    prod_sel = evaluate("production", selection_split, label=f"{prod_version}", out_dir=OUT / tag / "candidates")
    report["candidates"] = scored
    report["production_on_selection_split"] = prod_sel["metrics"]["success"]
    best = sorted(scored, key=lambda c: (-c["success"], c["mean_total_tokens"]))[0]
    cand_cfg = load_config(REPO_ROOT / best["config"])
    cand_version = reg.register(cand_cfg, note=f"{tag}: retrained candidate (budget {budget})", source=best["config"],
                                run_id=best["mlflow_run_id"])
    report["candidate"] = {"version": cand_version, **best}

    return gate_and_decide(reg, gate, cand_version, prod_version, eval_split, tag, report)


def gate_and_decide(reg: Registry, gate: dict[str, Any], cand_version: str, prod_version: str, eval_split: str,
                    tag: str, report: dict[str, Any]) -> dict[str, Any]:
    """Steps 7-8: held-out shifted and gold evaluation of candidate vs production, then the gate."""
    # 7. Held-out evaluation
    cand_shift = evaluate(cand_version, eval_split, out_dir=OUT / tag)
    prod_shift = evaluate(prod_version, eval_split, out_dir=OUT / tag)
    cand_gold = evaluate(cand_version, "gold", reason=f"Part E gate check ({tag}): candidate", out_dir=OUT / tag)
    prod_gold = evaluate(prod_version, "gold", reason=f"Part E gate check ({tag}): production", out_dir=OUT / tag)

    # 8. Gate
    shifted_gain = cand_shift["metrics"]["success"] - prod_shift["metrics"]["success"]
    gold_drop = prod_gold["metrics"]["success"] - cand_gold["metrics"]["success"]
    passed_shift = shifted_gain >= gate["min_shifted_gain"]
    passed_gold = gold_drop <= gate["gold_tolerance"]
    report["gate_check"] = {
        "shifted_split": eval_split,
        "production_shifted": prod_shift["metrics"]["success"], "candidate_shifted": cand_shift["metrics"]["success"],
        "shifted_gain": round(shifted_gain, 4), "min_shifted_gain": gate["min_shifted_gain"], "shifted_ok": passed_shift,
        "production_gold": prod_gold["metrics"]["success"], "candidate_gold": cand_gold["metrics"]["success"],
        "gold_drop": round(gold_drop, 4), "gold_tolerance": gate["gold_tolerance"], "gold_ok": passed_gold,
        "mlflow_runs": {"candidate_shifted": cand_shift.get("mlflow_run_id"), "production_shifted": prod_shift.get("mlflow_run_id"),
                        "candidate_gold": cand_gold.get("mlflow_run_id"), "production_gold": prod_gold.get("mlflow_run_id")},
    }
    if passed_shift and passed_gold:
        reg.promote(cand_version, "production", reason=f"{tag}: gate passed (shifted +{shifted_gain:.3f}, "
                                                       f"gold drop {gold_drop:.3f})")
        report["decision"] = f"promoted {cand_version} to production (previous {prod_version})"
    else:
        reasons = [] if passed_shift else [f"shifted gain {shifted_gain:+.3f} < {gate['min_shifted_gain']}"]
        reasons += [] if passed_gold else [f"gold drop {gold_drop:.3f} > {gate['gold_tolerance']}"]
        report["decision"] = f"rejected {cand_version}: " + "; ".join(reasons)
    return _finish(report, tag)


def _finish(report: dict[str, Any], tag: str) -> dict[str, Any]:
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    with (OUT / "gate_log.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"time": _now(), "tag": tag, "production": report.get("production"),
                             "candidate": report.get("candidate", {}).get("version"),
                             "gate_check": {k: v for k, v in report.get("gate_check", {}).items() if k != "mlflow_runs"},
                             "decision": report["decision"]}) + "\n")
    print(report["decision"])
    return report


def gate_candidate(config: Path, tag: str, note: str) -> dict[str, Any]:
    """Send an externally proposed configuration through the same gate (no retraining step).

    Used for changes that do not come from the retraining loop, e.g. a proposal to
    swap the production model for a cheaper one.
    """
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    reg = Registry()
    prod_version, prod = reg.resolve("production")
    cand_version = reg.register(load_config(config), note=f"{tag}: {note}", source=str(config))
    report: dict[str, Any] = {"tag": tag, "mode": "gate-only candidate", "note": note, "gate": gate,
                              "production": {"version": prod_version, "config_hash": prod.config_hash()},
                              "candidate": {"version": cand_version, "config": str(config)}}
    return gate_and_decide(reg, gate, cand_version, prod_version, f"shifted_{gate['shifted_eval_parent_split']}",
                           tag, report)


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    p = argparse.ArgumentParser(description="Drift-triggered retraining with a promotion gate")
    p.add_argument("--alert", type=Path, default=REPO_ROOT / "supporting_files" / "a2" / "monitoring" / "alerts.json")
    p.add_argument("--budget", type=int, default=16, help="labels to reveal (active-learning budget)")
    p.add_argument("--force", action="store_true", help="retrain without an alert")
    p.add_argument("--tag", default=None)
    p.add_argument("--candidate", type=Path, default=None,
                   help="gate an externally proposed configuration instead of retraining")
    p.add_argument("--note", default="externally proposed configuration")
    args = p.parse_args(argv)
    if args.candidate:
        gate_candidate(args.candidate, args.tag or f"gate_{_now()}", args.note)
    else:
        run(args.alert, args.budget, force=args.force, tag=args.tag)


if __name__ == "__main__":
    main()

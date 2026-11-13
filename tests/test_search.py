"""Tests for code/search/run_search.py (selection logic, no model calls)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from code.agent.config import CONFIGS_DIR, load_config
from code.search.run_search import SPACE_FILE, Search, objective_j, trial_config


def rec(name: str, sampler: str, success: float, tokens: float, wall: float, feasible: bool = True) -> dict:
    return {"name": name, "sampler": sampler, "success": success, "mean_total_tokens": tokens,
            "mean_wall_s": wall, "feasible": feasible}


def test_space_is_validation_only_and_has_two_samplers() -> None:
    spec = yaml.safe_load(SPACE_FILE.read_text(encoding="utf-8"))
    assert spec["split"] == "validation"
    assert len(spec["space"]) >= 2 and sum(spec["samplers"].values()) >= 15


def test_trial_config_overrides_only_searched_fields() -> None:
    base = load_config(CONFIGS_DIR / "proposed.yaml")
    cfg = trial_config(base, {"temperature": 0.7, "max_steps": 4}, "t1")
    assert (cfg.temperature, cfg.max_steps, cfg.name) == (0.7, 4, "t1")
    assert cfg.loop_guard == base.loop_guard and cfg.config_hash() != base.config_hash()


def test_selection_prefers_success_then_cost_and_skips_infeasible(tmp_path: Path) -> None:
    s = Search(out_dir=tmp_path)
    s.records = [rec("a", "random", 0.6, 9000, 20), rec("b", "nsga2", 0.6, 5000, 25),
                 rec("c", "nsga2", 0.9, 30000, 90, feasible=False), rec("d", "random", 0.3, 2000, 5)]
    summary = s.finish()
    assert summary["selected"]["name"] == "b"
    # a is not dominated: b costs fewer tokens but is slower.
    assert {p["name"] for p in summary["pareto_front"]} == {"a", "b", "c", "d"}
    assert summary["best_so_far"]["random"] == [0.6, 0.6]
    assert (tmp_path / "trials.csv").exists()


def test_objective_j_penalises_cost_latency_and_errors() -> None:
    w = {"alpha": 1.0, "beta": 0.01, "gamma": 0.002, "delta": 0.2}
    m = {"success": 0.5, "mean_total_tokens": 5000, "mean_wall_ms": 20000, "rate_malformed_any": 0.1,
         "rate_loop_extended": 0.2}
    assert objective_j(m, w) == pytest.approx(0.5 - 0.05 - 0.04 - 0.04)


def test_trials_round_trip_for_reselect(tmp_path: Path) -> None:
    s = Search(out_dir=tmp_path)
    base = {"trial": 0, "prompt_version": "v3", "temperature": 0.0, "max_steps": 6, "retry_budget": 2,
            "model": "qwen2.5:3b", "exemplars": "none", "config_hash": "h", "ci_low": 0.0, "ci_high": 1.0,
            "p95_wall_s": 30.0, "mean_steps": 4.0, "error_rate": 0.1, "J": 0.3, "new_runs": 0, "mlflow_run_id": "r"}
    s.records = [{**rec("a", "random", 0.5, 4000, 10), **base}]
    s.finish()
    again = Search(out_dir=tmp_path)
    again.load_trials()
    r = again.records[0]
    assert r["success"] == 0.5 and r["max_steps"] == 6 and r["feasible"] is True
    assert r["config_path"] == "configs/search/a.yaml"

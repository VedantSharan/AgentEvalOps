"""Tests for code/retrain/pipeline.py building blocks (no model calls)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from code.agent.config import CONFIGS_DIR, load_config
from code.evaluation.data import load_split
from code.retrain import pipeline


def traj(task_id: str, n_steps: int, errors: int, repeat: bool = False) -> dict:
    steps = []
    for i in range(n_steps):
        failed = i < errors
        args = {"query": "x"} if repeat else {"query": f"q{i}"}
        steps.append({"step_index": i, "action_type": "tool_call", "tool_name": "retrieve", "arguments": args,
                      "tool_result": {"tool_name": "retrieve", "success": not failed, "output": None if failed else {"m": []},
                                      "error": "503 Service Unavailable" if failed else None, "latency_ms": 1.0},
                      "llm_output": None, "error": "503 Service Unavailable" if failed else None,
                      "latency_ms": 1.0, "prompt_tokens": 1, "completion_tokens": 1})
    return {"run_id": f"r-{task_id}", "task_id": task_id, "configuration_hash": "h", "model_version": "m",
            "prompt_version": "p", "steps": steps, "final_answer": "x", "outcome": "failure",
            "metrics": {"total_steps": n_steps, "total_latency_ms": 1.0, "total_prompt_tokens": 1,
                        "total_completion_tokens": 1}, "errors": []}


POOL = [i for i in load_split("shifted_train")][:6]


def records() -> list[dict]:
    return [{"run_id": f"r-{it.task_id}", "task_id": it.task_id, "final_answer": "x",
             "trajectory": traj(it.task_id, n_steps=k + 1, errors=k % 3, repeat=k == 5)} for k, it in enumerate(POOL)]


def test_active_learning_selection_prefers_long_failing_runs() -> None:
    picked = pipeline.select(records(), budget=2)
    assert [p["al_rank"] for p in picked] == [1, 2]
    assert picked[0]["al_features"]["steps"] >= picked[1]["al_features"]["steps"] - 1
    assert pipeline.select(records(), budget=2) == picked


def test_selection_from_the_train_pool_passes_validation_and_leakage() -> None:
    res = pipeline.validate_selection(pipeline.select(records(), budget=3), "shifted_train")
    assert res["leaked_task_ids"] == [] and res["leaked_families"] == []


def test_selection_from_an_evaluation_split_is_rejected() -> None:
    item = load_split("shifted_test")[0]
    bad = [{"run_id": "r", "task_id": item.task_id, "trajectory": traj(item.task_id, 2, 0)}]
    res = pipeline.validate_selection(bad, "shifted_test")
    assert not res["passed"] and res["leaked_task_ids"] == [item.task_id]


def test_bank_and_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "PROMPTS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "CONFIGS_DIR", tmp_path)
    (tmp_path / "exemplars").mkdir()
    prod = load_config(CONFIGS_DIR / "proposed.yaml")
    sel = pipeline.select(records(), budget=4)
    labels = {r["run_id"]: {"outcome": "failure" if i % 2 else "success", "failure_category": "premature_stop",
                            "fault": "information_outage" if i % 2 else "transient_failure",
                            "gold_answer": "unanswerable"} for i, r in enumerate(sel)}
    path = pipeline.build_bank(prod, sel, labels, "shifted_train", "test_bank")
    bank = json.loads(path.read_text(encoding="utf-8"))
    assert {e["type"] for e in bank} == {"example", "pitfall"}
    paths = pipeline.candidate_configs(prod, "test_bank", "t")
    assert len(paths) == 2 and all(p.exists() for p in paths)


def test_gate_is_read_from_the_fixed_file() -> None:
    assert pipeline.GATE == CONFIGS_DIR / "gate.yaml"

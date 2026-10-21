"""
Tests for the adversarial benchmark dataset and evaluation results.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from code.agent.trajectory import load_trajectories_jsonl
from code.experiments.adversarial_evaluator import load_results_jsonl
from code.shared.models import TaskRecord, TrajectoryRecord
from code.tasks.adversarial_generator import compute_content_hash, load_tasks_jsonl
from code.tasks.grader import grade

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = REPO_ROOT / "data" / "adversarial" / "tasks.jsonl"
TRAJECTORIES_PATH = REPO_ROOT / "data" / "adversarial" / "trajectories_gemini_3.8_flash.jsonl"
RESULTS_PATH = REPO_ROOT / "data" / "adversarial" / "results.jsonl"

EXPECTED_FAILURE_MODES = {
    "tool_skipping",
    "ignoring_tool_results",
    "output_fabrication",
    "unnecessary_tool_use",
    "multistep_tool_dependency",
    "tool_error_recovery",
}


def test_adversarial_tasks_count_and_distribution():
    assert TASKS_PATH.exists(), f"Missing {TASKS_PATH}"
    tasks = load_tasks_jsonl(TASKS_PATH)
    assert len(tasks) == 30

    family_counts = {}
    for t in tasks:
        family_counts[t.template_family] = family_counts.get(t.template_family, 0) + 1

    assert set(family_counts.keys()) == EXPECTED_FAILURE_MODES
    for fam, count in family_counts.items():
        assert count == 5, f"Expected 5 tasks for {fam}, got {count}"


def test_adversarial_tasks_content_hashes_and_unique_ids():
    tasks = load_tasks_jsonl(TASKS_PATH)
    task_ids = set()

    for t in tasks:
        assert t.task_id not in task_ids, f"Duplicate task_id: {t.task_id}"
        task_ids.add(t.task_id)

        assert t.metadata is not None
        expected_hash = compute_content_hash(t.task_id, t.prompt, t.gold_answer)
        assert t.metadata.content_hash == expected_hash


def test_adversarial_tasks_gold_answers():
    tasks = load_tasks_jsonl(TASKS_PATH)
    for t in tasks:
        target_answer = t.gold_answer if t.answerable else "unanswerable"
        res = grade(t, target_answer)
        assert res.correct is True, f"Task {t.task_id} gold answer failed evaluation: {res.detail}"


def test_adversarial_trajectories_integrity():
    assert TRAJECTORIES_PATH.exists(), f"Missing {TRAJECTORIES_PATH}"
    trajectories = load_trajectories_jsonl(TRAJECTORIES_PATH)
    assert len(trajectories) == 30

    tasks = load_tasks_jsonl(TASKS_PATH)
    task_ids = {t.task_id for t in tasks}

    run_ids = set()
    for traj in trajectories:
        assert traj.run_id not in run_ids, f"Duplicate run_id: {traj.run_id}"
        run_ids.add(traj.run_id)

        assert traj.task_id in task_ids, f"Trajectory references unknown task: {traj.task_id}"
        assert traj.model_version == "gemini-3.8-flash"

        # Verify step contiguousness
        for idx, step in enumerate(traj.steps):
            assert step.step_index == idx


def test_adversarial_results_consistency():
    assert RESULTS_PATH.exists(), f"Missing {RESULTS_PATH}"
    results = load_results_jsonl(RESULTS_PATH)
    assert len(results) == 30

    tasks = load_tasks_jsonl(TASKS_PATH)
    task_map = {t.task_id: t for t in tasks}

    for res in results:
        assert res["task_id"] in task_map
        assert res["target_failure_mode"] in EXPECTED_FAILURE_MODES
        assert "final_answer_correct" in res
        assert "process_correct" in res
        assert "failure_classification" in res
        assert "required_tool_calls_made" in res
        assert "returned_values_used_faithfully" in res
        assert "unsupported_facts_invented" in res
        assert "unnecessary_calls_made" in res
        assert "recovered_from_tool_errors" in res

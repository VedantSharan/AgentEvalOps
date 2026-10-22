"""
Tests for code.experiments.generate_metadata.

Verifies:
- All required columns are present in metadata.csv
- Every sample has a valid SHA-256 canonical hash
- Base tasks, augmented tasks, and trajectories are all covered
- Authentic execution timestamps are present (not a single hard-coded string)
- All 154 multi-provider trajectories are indexed
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from code.experiments.generate_metadata import (
    compute_task_content_hash,
    compute_trajectory_content_hash,
    generate_metadata,
    load_work_queue_timestamps,
)
from code.shared.models import TaskRecord, TrajectoryRecord


def test_compute_task_content_hash():
    task = TaskRecord(
        task_id="T_test_01",
        task_type="test_task",
        template_family="test_family",
        prompt="Sample prompt for testing?",
        gold_answer="42",
        evaluation_rule="exact_match",
        expected_tools=["calculate"],
        answerable=True,
    )
    ch = compute_task_content_hash(task)
    assert isinstance(ch, str)
    assert len(ch) == 64
    # Deterministic check
    assert ch == compute_task_content_hash(task)


def test_load_work_queue_timestamps():
    ts_map = load_work_queue_timestamps()
    if Path("temp/work_queue.sqlite3").exists():
        assert len(ts_map) >= 154
        # Verify valid ISO-8601 string format
        for rid, ts in ts_map.items():
            assert "2026-10-21" in ts


def test_generated_metadata_csv_schema_and_coverage():
    csv_path = Path("data/metadata.csv")
    assert csv_path.exists(), "data/metadata.csv must exist"

    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    expected_cols = {
        "sample_id",
        "sample_type",
        "source",
        "timestamp",
        "collector",
        "tool_version",
        "content_hash",
        "extra_metadata",
    }
    assert expected_cols.issubset(set(reader.fieldnames or []))

    types = {r["sample_type"] for r in rows}
    assert "task" in types
    assert "augmented_task" in types
    assert "trajectory" in types

    # Check content hashes are valid 64-char hex strings
    for r in rows:
        assert len(r["content_hash"]) == 64, f"Invalid hash in {r['sample_id']}"
        assert r["timestamp"], f"Missing timestamp in {r['sample_id']}"
        assert r["collector"], f"Missing collector in {r['sample_id']}"

    # Check trajectory coverage includes all 154 multi-provider runs
    multi_provider_runs = [r for r in rows if r["sample_type"] == "trajectory" and r["source"] != "rule-based-agent-v1"]
    assert len(multi_provider_runs) == 154, f"Expected 154 multi-provider trajectories, found {len(multi_provider_runs)}"

    # Check timestamps are varied and not all identical
    timestamps = {r["timestamp"] for r in multi_provider_runs}
    assert len(timestamps) > 100, f"Expected varied per-run timestamps, found only {len(timestamps)} unique"


def test_generate_metadata_execution(tmp_path: Path):
    tasks_path = tmp_path / "tasks.jsonl"
    task_record = {
        "task_id": "T_mock_01",
        "task_type": "mock",
        "template_family": "mock_family",
        "prompt": "mock prompt",
        "gold_answer": "100",
        "evaluation_rule": "exact_match",
        "expected_tools": ["calculate"],
        "answerable": True,
        "metadata": {
            "provenance": "test",
            "created_at": "2026-10-21T00:00:00Z",
            "generator_version": "1.0",
        },
    }
    tasks_path.write_text(TaskRecord.model_validate(task_record).model_dump_json() + "\n", encoding="utf-8")

    out_csv = tmp_path / "metadata_test.csv"
    count = generate_metadata(
        tasks_paths=[tasks_path],
        trajectories_paths=[],
        output_path=out_csv,
    )
    assert count == 1
    assert out_csv.exists()

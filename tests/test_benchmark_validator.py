"""
Tests for code.validation.benchmark_validator (Assignment 1, Part B).

Verifies that:
1. The 12-check validation suite passes cleanly on the official benchmark datasets.
2. The suite deterministically fails and reports specific errors on the corrupted fixtures in data/corrupted/.
"""

from pathlib import Path
import pytest

from code.validation.benchmark_validator import BenchmarkValidator


def test_clean_benchmark_validation():
    """Verify that the official processed benchmark passes all 12 checks."""
    tasks_path = Path("data/processed/tasks_revised.jsonl")
    trajectories_path = Path("data/processed/trajectories_processed.jsonl")

    validator = BenchmarkValidator(tasks_path, trajectories_path)
    result = validator.run_all_checks()

    assert result.passed is True
    assert result.checks_passed == 12
    assert result.checks_failed == 0
    assert len(result.issues) == 0
    assert result.trajectories_count == 102


def test_processed_dataset_excludes_error_runs():
    """Verify that processed trajectories and human annotation sample contain no error outcomes."""
    import json
    
    # 1. Processed trajectory dataset
    proc_lines = Path("data/processed/trajectories_processed.jsonl").read_text(encoding="utf-8").strip().split("\n")
    proc_records = [json.loads(l) for l in proc_lines if l.strip()]
    assert len(proc_records) == 102
    assert all(r["outcome"] in ("success", "failure") for r in proc_records)
    assert not any(r["outcome"] == "error" for r in proc_records)

    # 2. Human annotation sample
    sample_data = json.loads(Path("data/labels/stratified_sample_45.json").read_text(encoding="utf-8"))
    assert len(sample_data) == 45
    assert all(s["data"]["outcome"] in ("success", "failure") for s in sample_data)
    assert not any(s["data"]["outcome"] == "error" for s in sample_data)


def test_corrupted_benchmark_fixtures():
    """Verify that corrupted data fixtures fail expected validation checks deterministically."""
    tasks_path = Path("data/corrupted/tasks_corrupted.jsonl")
    trajectories_path = Path("data/corrupted/trajectories_corrupted.jsonl")

    validator = BenchmarkValidator(tasks_path, trajectories_path)
    result = validator.run_all_checks()

    assert result.passed is False
    assert result.checks_failed >= 10

    # Verify specific checks failed
    expected_failed_checks = [
        "1_jsonl_parsing",
        "2_pydantic_schemas",
        "3_unique_identifiers",
        "4_reference_integrity",
        "5_content_hashes",
        "6_duplicate_tasks",
        "7_prompt_answer_leakage",
        "9_tool_call_arguments",
        "10_metric_value_ranges",
        "11_missing_values",
        "12_gold_answer_grading_consistency",
    ]

    for check in expected_failed_checks:
        assert result.summary_by_check.get(check) is False, f"Expected {check} to fail on corrupted fixtures"

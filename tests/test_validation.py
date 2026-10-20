"""
Tests for code.validation.dataset_validator.

Covers:
  - Valid current benchmark datasets
  - Invalid JSON and non-object JSONL records
  - Missing required fields and invalid enum values (schema validation)
  - Duplicate task IDs and trajectory run IDs
  - Trajectories referencing nonexistent tasks (reference integrity)
  - Incorrect final answers (gold-answer evaluation)
  - Content hash mismatch and malformed hash format
  - Mismatched aggregate step counts and outcome contradictions (trajectory integrity)
  - Exact duplicate content (distinguishing duplicate IDs from duplicate content)
  - Multiple independent errors reported in a single run
  - Immutability of input source files
  - CLI execution and exit codes
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from code.shared.models import EvaluationRule, Outcome, TaskRecord, TrajectoryRecord
from code.validation.dataset_validator import (
    ALL_CHECKS,
    CHECK_CONTENT_HASHES,
    CHECK_EXACT_DUPLICATES,
    CHECK_GOLD_ANSWER_EVALUATION,
    CHECK_JSONL_PARSING,
    CHECK_REFERENCE_INTEGRITY,
    CHECK_SCHEMA_VALIDATION,
    CHECK_TRAJECTORY_INTEGRITY,
    CHECK_UNIQUE_IDS,
    Severity,
    ValidationReport,
    compute_canonical_content_hash,
    format_report_human,
    main,
    validate_dataset,
    validate_datasets,
)

# ---------------------------------------------------------------------------
# Fixtures and Helpers
# ---------------------------------------------------------------------------

TASKS_PATH = Path("data/raw/tasks.jsonl")
TRAJECTORIES_PATH = Path("data/raw/trajectories.jsonl")


def write_jsonl(path: Path, lines: list[Any]) -> Path:
    """Write items as newline-delimited JSON or raw strings."""
    with path.open("w", encoding="utf-8") as f:
        for item in lines:
            if isinstance(item, str):
                f.write(item + "\n")
            else:
                f.write(json.dumps(item) + "\n")
    return path


@pytest.fixture
def sample_task_dict() -> dict[str, Any]:
    """Return a single valid task dict from the real dataset."""
    with TASKS_PATH.open("r", encoding="utf-8") as f:
        return json.loads(f.readline())


@pytest.fixture
def sample_trajectory_dict() -> dict[str, Any]:
    """Return a single valid trajectory dict from the real dataset."""
    with TRAJECTORIES_PATH.open("r", encoding="utf-8") as f:
        return json.loads(f.readline())


# ---------------------------------------------------------------------------
# 1. Valid Current Datasets
# ---------------------------------------------------------------------------


class TestValidCurrentDatasets:
    """Verify that current raw benchmark datasets pass all checks cleanly."""

    def test_valid_current_datasets(self):
        report = validate_datasets(TASKS_PATH, TRAJECTORIES_PATH)

        assert report.passed is True
        assert report.checks_failed == 0
        assert report.checks_passed == len(ALL_CHECKS)
        assert len(report.findings) == 0
        assert report.has_errors is False
        assert len(report.errors) == 0
        assert len(report.warnings) == 0

        assert report.total_task_records_read == 24
        assert report.total_trajectory_records_read == 24
        assert report.summary.valid_tasks_count == 24
        assert report.summary.valid_trajectories_count == 24

        expected_families = {
            "calculation": 4,
            "file_ops": 4,
            "multistep_qa": 4,
            "retrieval": 4,
            "structured_data": 4,
            "unanswerable": 4,
        }
        assert report.summary.tasks_by_family == expected_families
        assert report.summary.trajectories_by_outcome == {"success": 24}

    def test_convenience_alias_validate_dataset(self):
        report = validate_dataset(TASKS_PATH, TRAJECTORIES_PATH)
        assert report.passed is True

    def test_machine_readable_report_serialization(self):
        report = validate_datasets(TASKS_PATH, TRAJECTORIES_PATH)

        dumped = report.to_dict()
        assert isinstance(dumped, dict)
        assert dumped["passed"] is True
        assert dumped["checks_passed"] == 8

        json_str = report.to_json(indent=2)
        parsed = json.loads(json_str)
        assert parsed["passed"] is True
        assert parsed["summary"]["valid_tasks_count"] == 24

        roundtrip = ValidationReport.model_validate(parsed)
        assert roundtrip.passed == report.passed
        assert roundtrip.checks_passed == report.checks_passed

    def test_human_readable_report_format(self):
        report = validate_datasets(TASKS_PATH, TRAJECTORIES_PATH)
        human_text = format_report_human(report)

        assert "AgentEvalOps Dataset Validation Report" in human_text
        assert "Overall Status:       PASSED" in human_text
        assert "8 passed, 0 failed" in human_text
        assert "Total Findings:       0" in human_text


# ---------------------------------------------------------------------------
# 2. Invalid JSON and Non-Object JSONL Records
# ---------------------------------------------------------------------------


class TestInvalidJSONAndNonObject:
    """Test JSON parsing failures and non-object records."""

    def test_invalid_json_line_in_tasks(self, tmp_path: Path, sample_trajectory_dict: dict):
        tasks_file = tmp_path / "tasks.jsonl"
        trajs_file = tmp_path / "trajs.jsonl"

        write_jsonl(tasks_file, ['{"task_id": "bad", broken_json: true}'])
        write_jsonl(trajs_file, [sample_trajectory_dict])

        report = validate_datasets(tasks_file, trajs_file)
        assert report.passed is False
        assert report.checks_failed >= 1

        json_findings = [f for f in report.findings if f.check_name == CHECK_JSONL_PARSING]
        assert len(json_findings) == 1
        assert json_findings[0].severity == Severity.error
        assert json_findings[0].line_number == 1
        assert "not valid json" in json_findings[0].message.lower()

    def test_non_object_json_in_tasks(self, tmp_path: Path, sample_trajectory_dict: dict):
        tasks_file = tmp_path / "tasks.jsonl"
        trajs_file = tmp_path / "trajs.jsonl"

        write_jsonl(tasks_file, ['["not", "a", "json", "object"]'])
        write_jsonl(trajs_file, [sample_trajectory_dict])

        report = validate_datasets(tasks_file, trajs_file)
        assert report.passed is False

        json_findings = [f for f in report.findings if f.check_name == CHECK_JSONL_PARSING]
        assert len(json_findings) == 1
        assert "does not decode to a JSON object" in json_findings[0].message

    def test_invalid_json_in_trajectories(self, tmp_path: Path, sample_task_dict: dict):
        tasks_file = tmp_path / "tasks.jsonl"
        trajs_file = tmp_path / "trajs.jsonl"

        write_jsonl(tasks_file, [sample_task_dict])
        write_jsonl(trajs_file, ["not json at all"])

        report = validate_datasets(tasks_file, trajs_file)
        assert report.passed is False

        json_findings = [f for f in report.findings if f.check_name == CHECK_JSONL_PARSING]
        assert len(json_findings) == 1
        assert json_findings[0].line_number == 1

    def test_non_object_json_in_trajectories(self, tmp_path: Path, sample_task_dict: dict):
        tasks_file = tmp_path / "tasks.jsonl"
        trajs_file = tmp_path / "trajs.jsonl"

        write_jsonl(tasks_file, [sample_task_dict])
        write_jsonl(trajs_file, ["123456"])

        report = validate_datasets(tasks_file, trajs_file)
        assert report.passed is False

        json_findings = [f for f in report.findings if f.check_name == CHECK_JSONL_PARSING]
        assert len(json_findings) == 1
        assert "does not decode to a JSON object" in json_findings[0].message


# ---------------------------------------------------------------------------
# 3. Missing Required Fields and Invalid Enum Values (Schema Validation)
# ---------------------------------------------------------------------------


class TestMissingRequiredFieldsAndInvalidEnums:
    """Test Pydantic schema validation failures."""

    def test_missing_required_field_in_task(
        self, tmp_path: Path, sample_task_dict: dict, sample_trajectory_dict: dict
    ):
        corrupted = copy.deepcopy(sample_task_dict)
        del corrupted["prompt"]

        tasks_file = write_jsonl(tmp_path / "tasks.jsonl", [corrupted])
        trajs_file = write_jsonl(tmp_path / "trajs.jsonl", [sample_trajectory_dict])

        report = validate_datasets(tasks_file, trajs_file)
        assert report.passed is False

        schema_findings = [f for f in report.findings if f.check_name == CHECK_SCHEMA_VALIDATION]
        assert len(schema_findings) == 1
        assert "prompt" in schema_findings[0].message
        assert schema_findings[0].record_id == corrupted["task_id"]

    def test_invalid_enum_in_task(
        self, tmp_path: Path, sample_task_dict: dict, sample_trajectory_dict: dict
    ):
        corrupted = copy.deepcopy(sample_task_dict)
        corrupted["evaluation_rule"] = "invalid_rule_xyz"

        tasks_file = write_jsonl(tmp_path / "tasks.jsonl", [corrupted])
        trajs_file = write_jsonl(tmp_path / "trajs.jsonl", [sample_trajectory_dict])

        report = validate_datasets(tasks_file, trajs_file)
        assert report.passed is False

        schema_findings = [f for f in report.findings if f.check_name == CHECK_SCHEMA_VALIDATION]
        assert len(schema_findings) == 1
        assert "evaluation_rule" in schema_findings[0].message

    def test_invalid_enum_in_trajectory(
        self, tmp_path: Path, sample_task_dict: dict, sample_trajectory_dict: dict
    ):
        corrupted = copy.deepcopy(sample_trajectory_dict)
        corrupted["outcome"] = "half_success"

        tasks_file = write_jsonl(tmp_path / "tasks.jsonl", [sample_task_dict])
        trajs_file = write_jsonl(tmp_path / "trajs.jsonl", [corrupted])

        report = validate_datasets(tasks_file, trajs_file)
        assert report.passed is False

        schema_findings = [f for f in report.findings if f.check_name == CHECK_SCHEMA_VALIDATION]
        assert len(schema_findings) == 1
        assert "outcome" in schema_findings[0].message

    def test_non_contiguous_step_indices(
        self, tmp_path: Path, sample_task_dict: dict, sample_trajectory_dict: dict
    ):
        corrupted = copy.deepcopy(sample_trajectory_dict)
        # Change second step's index from 1 to 3
        if len(corrupted["steps"]) >= 2:
            corrupted["steps"][1]["step_index"] = 3

        tasks_file = write_jsonl(tmp_path / "tasks.jsonl", [sample_task_dict])
        trajs_file = write_jsonl(tmp_path / "trajs.jsonl", [corrupted])

        report = validate_datasets(tasks_file, trajs_file)
        assert report.passed is False

        schema_findings = [f for f in report.findings if f.check_name == CHECK_SCHEMA_VALIDATION]
        assert len(schema_findings) == 1
        assert "contiguous" in schema_findings[0].message

    def test_validator_continues_after_schema_error(
        self, tmp_path: Path, sample_task_dict: dict, sample_trajectory_dict: dict
    ):
        # 3 tasks: Line 1 valid, Line 2 invalid schema, Line 3 valid
        t1 = copy.deepcopy(sample_task_dict)
        t1["task_id"] = "task-01"
        t1["prompt"] = "What is the capital of France?"
        t1["metadata"]["content_hash"] = compute_canonical_content_hash(
            t1["task_id"], t1["prompt"], t1["gold_answer"]
        )

        t2_bad = copy.deepcopy(sample_task_dict)
        t2_bad["task_id"] = "task-02"
        del t2_bad["prompt"]

        t3 = copy.deepcopy(sample_task_dict)
        t3["task_id"] = "task-03"
        t3["prompt"] = "What is the capital of Japan?"
        t3["gold_answer"] = "Tokyo"
        t3["metadata"]["content_hash"] = compute_canonical_content_hash(
            t3["task_id"], t3["prompt"], t3["gold_answer"]
        )

        tasks_file = write_jsonl(tmp_path / "tasks.jsonl", [t1, t2_bad, t3])
        trajs_file = write_jsonl(tmp_path / "trajs.jsonl", [])

        report = validate_datasets(tasks_file, trajs_file)
        assert report.total_task_records_read == 3
        assert report.summary.valid_tasks_count == 2
        assert len(report.errors) == 1
        assert report.errors[0].check_name == CHECK_SCHEMA_VALIDATION
        assert report.errors[0].line_number == 2


# ---------------------------------------------------------------------------
# 4. Duplicate Task IDs and Trajectory Run IDs
# ---------------------------------------------------------------------------


class TestDuplicateIDs:
    """Test duplicate ID detection."""

    def test_duplicate_task_ids(self, tmp_path: Path, sample_task_dict: dict):
        t1 = copy.deepcopy(sample_task_dict)
        t1["task_id"] = "task-dup"
        t1["prompt"] = "First prompt"

        t2 = copy.deepcopy(sample_task_dict)
        t2["task_id"] = "task-dup"
        t2["prompt"] = "Different prompt with same ID"

        tasks_file = write_jsonl(tmp_path / "tasks.jsonl", [t1, t2])
        trajs_file = write_jsonl(tmp_path / "trajs.jsonl", [])

        report = validate_datasets(tasks_file, trajs_file)
        assert report.passed is False

        dup_findings = [f for f in report.findings if f.check_name == CHECK_UNIQUE_IDS]
        assert len(dup_findings) == 1
        assert dup_findings[0].record_id == "task-dup"
        assert dup_findings[0].line_number == 2
        assert "first seen at line 1" in dup_findings[0].message

    def test_duplicate_trajectory_run_ids(
        self, tmp_path: Path, sample_task_dict: dict, sample_trajectory_dict: dict
    ):
        tr1 = copy.deepcopy(sample_trajectory_dict)
        tr1["run_id"] = "run-same-id"

        tr2 = copy.deepcopy(sample_trajectory_dict)
        tr2["run_id"] = "run-same-id"
        tr2["final_answer"] = "Different answer"

        tasks_file = write_jsonl(tmp_path / "tasks.jsonl", [sample_task_dict])
        trajs_file = write_jsonl(tmp_path / "trajs.jsonl", [tr1, tr2])

        report = validate_datasets(tasks_file, trajs_file)
        assert report.passed is False

        dup_findings = [f for f in report.findings if f.check_name == CHECK_UNIQUE_IDS]
        assert len(dup_findings) == 1
        assert dup_findings[0].record_id == "run-same-id"
        assert dup_findings[0].line_number == 2


# ---------------------------------------------------------------------------
# 5. Reference Integrity
# ---------------------------------------------------------------------------


class TestReferenceIntegrity:
    """Test that trajectory task_id references must exist in tasks.jsonl."""

    def test_trajectory_referencing_nonexistent_task(
        self, tmp_path: Path, sample_task_dict: dict, sample_trajectory_dict: dict
    ):
        corrupted = copy.deepcopy(sample_trajectory_dict)
        corrupted["task_id"] = "nonexistent-task-999"

        tasks_file = write_jsonl(tmp_path / "tasks.jsonl", [sample_task_dict])
        trajs_file = write_jsonl(tmp_path / "trajs.jsonl", [corrupted])

        report = validate_datasets(tasks_file, trajs_file)
        assert report.passed is False

        ref_findings = [f for f in report.findings if f.check_name == CHECK_REFERENCE_INTEGRITY]
        assert len(ref_findings) == 1
        assert "nonexistent-task-999" in ref_findings[0].message
        assert ref_findings[0].record_id == corrupted["run_id"]


# ---------------------------------------------------------------------------
# 6. Content Hashes
# ---------------------------------------------------------------------------


class TestContentHashes:
    """Test verification and recomputation of canonical task content hashes."""

    def test_malformed_content_hash_length(
        self, tmp_path: Path, sample_task_dict: dict, sample_trajectory_dict: dict
    ):
        corrupted = copy.deepcopy(sample_task_dict)
        corrupted["metadata"]["content_hash"] = "abc123short"

        tasks_file = write_jsonl(tmp_path / "tasks.jsonl", [corrupted])
        trajs_file = write_jsonl(tmp_path / "trajs.jsonl", [sample_trajectory_dict])

        report = validate_datasets(tasks_file, trajs_file)
        assert report.passed is False

        hash_findings = [f for f in report.findings if f.check_name == CHECK_CONTENT_HASHES]
        assert len(hash_findings) == 1
        assert "malformed" in hash_findings[0].message.lower()

    def test_malformed_content_hash_non_hex(
        self, tmp_path: Path, sample_task_dict: dict, sample_trajectory_dict: dict
    ):
        corrupted = copy.deepcopy(sample_task_dict)
        corrupted["metadata"]["content_hash"] = "g" * 64

        tasks_file = write_jsonl(tmp_path / "tasks.jsonl", [corrupted])
        trajs_file = write_jsonl(tmp_path / "trajs.jsonl", [sample_trajectory_dict])

        report = validate_datasets(tasks_file, trajs_file)
        assert report.passed is False

        hash_findings = [f for f in report.findings if f.check_name == CHECK_CONTENT_HASHES]
        assert len(hash_findings) == 1
        assert "malformed" in hash_findings[0].message.lower()

    def test_content_hash_mismatch(
        self, tmp_path: Path, sample_task_dict: dict, sample_trajectory_dict: dict
    ):
        corrupted = copy.deepcopy(sample_task_dict)
        # 64 hex characters, but wrong value
        corrupted["metadata"]["content_hash"] = "0" * 64

        tasks_file = write_jsonl(tmp_path / "tasks.jsonl", [corrupted])
        trajs_file = write_jsonl(tmp_path / "trajs.jsonl", [sample_trajectory_dict])

        report = validate_datasets(tasks_file, trajs_file)
        assert report.passed is False

        hash_findings = [f for f in report.findings if f.check_name == CHECK_CONTENT_HASHES]
        assert len(hash_findings) == 1
        assert "mismatch" in hash_findings[0].message.lower()

    def test_task_without_content_hash_passes(
        self, tmp_path: Path, sample_task_dict: dict, sample_trajectory_dict: dict
    ):
        t = copy.deepcopy(sample_task_dict)
        t["metadata"]["content_hash"] = None

        tasks_file = write_jsonl(tmp_path / "tasks.jsonl", [t])
        trajs_file = write_jsonl(tmp_path / "trajs.jsonl", [sample_trajectory_dict])

        report = validate_datasets(tasks_file, trajs_file)
        hash_findings = [f for f in report.findings if f.check_name == CHECK_CONTENT_HASHES]
        assert len(hash_findings) == 0


# ---------------------------------------------------------------------------
# 7. Trajectory Integrity
# ---------------------------------------------------------------------------


class TestTrajectoryIntegrity:
    """Test step count matching and outcome contradictions."""

    def test_mismatched_aggregate_step_count(
        self, tmp_path: Path, sample_task_dict: dict, sample_trajectory_dict: dict
    ):
        corrupted = copy.deepcopy(sample_trajectory_dict)
        actual_steps = len(corrupted["steps"])
        corrupted["metrics"]["total_steps"] = actual_steps + 3

        tasks_file = write_jsonl(tmp_path / "tasks.jsonl", [sample_task_dict])
        trajs_file = write_jsonl(tmp_path / "trajs.jsonl", [corrupted])

        report = validate_datasets(tasks_file, trajs_file)
        assert report.passed is False

        int_findings = [f for f in report.findings if f.check_name == CHECK_TRAJECTORY_INTEGRITY]
        assert any("step count mismatch" in f.message for f in int_findings)

    def test_success_outcome_with_non_empty_errors(
        self, tmp_path: Path, sample_task_dict: dict, sample_trajectory_dict: dict
    ):
        corrupted = copy.deepcopy(sample_trajectory_dict)
        corrupted["outcome"] = "success"
        corrupted["errors"] = ["Fatal tool failure"]

        tasks_file = write_jsonl(tmp_path / "tasks.jsonl", [sample_task_dict])
        trajs_file = write_jsonl(tmp_path / "trajs.jsonl", [corrupted])

        report = validate_datasets(tasks_file, trajs_file)
        assert report.passed is False

        int_findings = [f for f in report.findings if f.check_name == CHECK_TRAJECTORY_INTEGRITY]
        assert any("non-empty errors" in f.message for f in int_findings)

    def test_success_outcome_with_null_final_answer(
        self, tmp_path: Path, sample_task_dict: dict, sample_trajectory_dict: dict
    ):
        corrupted = copy.deepcopy(sample_trajectory_dict)
        corrupted["outcome"] = "success"
        corrupted["final_answer"] = None

        tasks_file = write_jsonl(tmp_path / "tasks.jsonl", [sample_task_dict])
        trajs_file = write_jsonl(tmp_path / "trajs.jsonl", [corrupted])

        report = validate_datasets(tasks_file, trajs_file)
        assert report.passed is False

        int_findings = [f for f in report.findings if f.check_name == CHECK_TRAJECTORY_INTEGRITY]
        assert any("final_answer is null" in f.message for f in int_findings)

    def test_error_outcome_without_any_errors(
        self, tmp_path: Path, sample_task_dict: dict, sample_trajectory_dict: dict
    ):
        corrupted = copy.deepcopy(sample_trajectory_dict)
        corrupted["outcome"] = "error"
        corrupted["errors"] = []
        for s in corrupted["steps"]:
            s["error"] = None
            if s.get("tool_result"):
                s["tool_result"]["error"] = None

        tasks_file = write_jsonl(tmp_path / "tasks.jsonl", [sample_task_dict])
        trajs_file = write_jsonl(tmp_path / "trajs.jsonl", [corrupted])

        report = validate_datasets(tasks_file, trajs_file)
        assert report.passed is False

        int_findings = [f for f in report.findings if f.check_name == CHECK_TRAJECTORY_INTEGRITY]
        assert any("no error recorded" in f.message for f in int_findings)


# ---------------------------------------------------------------------------
# 8. Gold-Answer Evaluation
# ---------------------------------------------------------------------------


class TestGoldAnswerEvaluation:
    """Test independent grading of trajectory answers against tasks."""

    def test_incorrect_final_answer_exact_match(
        self, tmp_path: Path, sample_task_dict: dict, sample_trajectory_dict: dict
    ):
        # sample_task_dict is retrieval-f01 (capital of France -> Paris)
        task = copy.deepcopy(sample_task_dict)
        traj = copy.deepcopy(sample_trajectory_dict)
        traj["final_answer"] = "Berlin"  # Incorrect!

        tasks_file = write_jsonl(tmp_path / "tasks.jsonl", [task])
        trajs_file = write_jsonl(tmp_path / "trajs.jsonl", [traj])

        report = validate_datasets(tasks_file, trajs_file)
        assert report.passed is False

        eval_findings = [f for f in report.findings if f.check_name == CHECK_GOLD_ANSWER_EVALUATION]
        assert len(eval_findings) == 1
        assert "Berlin" in eval_findings[0].message
        assert "failed gold evaluation" in eval_findings[0].message

        # Verify that the trajectory on disk was NOT mutated
        with trajs_file.open("r", encoding="utf-8") as fh:
            line_data = json.loads(fh.readline())
            assert line_data["outcome"] == traj["outcome"]

    def test_incorrect_unanswerable_refusal(
        self, tmp_path: Path, sample_task_dict: dict, sample_trajectory_dict: dict
    ):
        task = copy.deepcopy(sample_task_dict)
        task["task_id"] = "unanswerable-un01"
        task["answerable"] = False
        task["gold_answer"] = None
        task["evaluation_rule"] = "unanswerable"

        traj = copy.deepcopy(sample_trajectory_dict)
        traj["task_id"] = "unanswerable-un01"
        traj["final_answer"] = "I know the answer is 42"  # Missing unanswerable keyword!

        tasks_file = write_jsonl(tmp_path / "tasks.jsonl", [task])
        trajs_file = write_jsonl(tmp_path / "trajs.jsonl", [traj])

        report = validate_datasets(tasks_file, trajs_file)
        assert report.passed is False

        eval_findings = [f for f in report.findings if f.check_name == CHECK_GOLD_ANSWER_EVALUATION]
        assert len(eval_findings) == 1
        assert "unanswerable" in eval_findings[0].message


# ---------------------------------------------------------------------------
# 9. Exact Duplicates
# ---------------------------------------------------------------------------


class TestExactDuplicates:
    """Test duplicate content detection while distinguishing from duplicate IDs."""

    def test_exact_duplicate_task_content_different_ids(
        self, tmp_path: Path, sample_task_dict: dict
    ):
        t1 = copy.deepcopy(sample_task_dict)
        t1["task_id"] = "task-distinct-1"

        t2 = copy.deepcopy(sample_task_dict)
        t2["task_id"] = "task-distinct-2"  # Distinct ID, but identical content

        tasks_file = write_jsonl(tmp_path / "tasks.jsonl", [t1, t2])
        trajs_file = write_jsonl(tmp_path / "trajs.jsonl", [])

        report = validate_datasets(tasks_file, trajs_file)
        assert report.passed is False

        dup_findings = [f for f in report.findings if f.check_name == CHECK_EXACT_DUPLICATES]
        assert len(dup_findings) == 1
        assert "distinct IDs but identical content" in dup_findings[0].message
        assert dup_findings[0].record_id == "task-distinct-2"

        # Unique IDs check should NOT have failed
        unique_findings = [f for f in report.findings if f.check_name == CHECK_UNIQUE_IDS]
        assert len(unique_findings) == 0

    def test_exact_duplicate_task_record_identical_id(
        self, tmp_path: Path, sample_task_dict: dict
    ):
        t1 = copy.deepcopy(sample_task_dict)
        t2 = copy.deepcopy(sample_task_dict)  # Identical ID AND content

        tasks_file = write_jsonl(tmp_path / "tasks.jsonl", [t1, t2])
        trajs_file = write_jsonl(tmp_path / "trajs.jsonl", [])

        report = validate_datasets(tasks_file, trajs_file)
        assert report.passed is False

        # Both unique_ids AND exact_duplicates must be flagged
        unique_findings = [f for f in report.findings if f.check_name == CHECK_UNIQUE_IDS]
        dup_findings = [f for f in report.findings if f.check_name == CHECK_EXACT_DUPLICATES]

        assert len(unique_findings) == 1
        assert len(dup_findings) == 1
        assert "identical ID and content" in dup_findings[0].message

    def test_exact_duplicate_trajectory_records(
        self, tmp_path: Path, sample_task_dict: dict, sample_trajectory_dict: dict
    ):
        tr1 = copy.deepcopy(sample_trajectory_dict)
        tr1["run_id"] = "run-01"

        tr2 = copy.deepcopy(sample_trajectory_dict)
        tr2["run_id"] = "run-02"  # Same execution, different run_id

        tasks_file = write_jsonl(tmp_path / "tasks.jsonl", [sample_task_dict])
        trajs_file = write_jsonl(tmp_path / "trajs.jsonl", [tr1, tr2])

        report = validate_datasets(tasks_file, trajs_file)
        assert report.passed is False

        dup_findings = [f for f in report.findings if f.check_name == CHECK_EXACT_DUPLICATES]
        assert len(dup_findings) == 1
        assert "distinct run_ids but identical execution content" in dup_findings[0].message


# ---------------------------------------------------------------------------
# 10. Multiple Independent Errors in Single Run
# ---------------------------------------------------------------------------


class TestMultipleIndependentErrors:
    """Test that validator catches multiple distinct issues without early exit."""

    def test_multiple_independent_errors_in_single_run(
        self, tmp_path: Path, sample_task_dict: dict, sample_trajectory_dict: dict
    ):
        # 1. Valid task
        t1 = copy.deepcopy(sample_task_dict)
        t1["task_id"] = "task-valid"

        # 2. Corrupted JSON task
        line_bad_json = "not valid json at line 2"

        # 3. Task with malformed content hash
        t3 = copy.deepcopy(sample_task_dict)
        t3["task_id"] = "task-bad-hash"
        t3["metadata"]["content_hash"] = "short"

        # 4. Duplicate task ID
        t4_dup = copy.deepcopy(sample_task_dict)
        t4_dup["task_id"] = "task-valid"
        t4_dup["prompt"] = "Something different"

        # Trajectory 1: references non-existent task
        tr1 = copy.deepcopy(sample_trajectory_dict)
        tr1["run_id"] = "run-ghost"
        tr1["task_id"] = "ghost-task"

        # Trajectory 2: step count mismatch
        tr2 = copy.deepcopy(sample_trajectory_dict)
        tr2["run_id"] = "run-bad-steps"
        tr2["task_id"] = "task-valid"
        tr2["metrics"]["total_steps"] = 99

        # Trajectory 3: wrong final answer
        tr3 = copy.deepcopy(sample_trajectory_dict)
        tr3["run_id"] = "run-wrong-ans"
        tr3["task_id"] = "task-valid"
        tr3["final_answer"] = "Completely wrong answer"

        tasks_file = write_jsonl(tmp_path / "tasks.jsonl", [t1, line_bad_json, t3, t4_dup])
        trajs_file = write_jsonl(tmp_path / "trajs.jsonl", [tr1, tr2, tr3])

        report = validate_datasets(tasks_file, trajs_file)

        assert report.passed is False
        assert report.checks_failed >= 5

        check_names_failed = {f.check_name for f in report.findings}
        assert CHECK_JSONL_PARSING in check_names_failed
        assert CHECK_CONTENT_HASHES in check_names_failed
        assert CHECK_UNIQUE_IDS in check_names_failed
        assert CHECK_REFERENCE_INTEGRITY in check_names_failed
        assert CHECK_TRAJECTORY_INTEGRITY in check_names_failed
        assert CHECK_GOLD_ANSWER_EVALUATION in check_names_failed

        # Verify findings are deterministically sorted
        lines_and_files = [(f.file_path, f.line_number) for f in report.findings]
        assert lines_and_files == sorted(lines_and_files)


# ---------------------------------------------------------------------------
# 11. Immutability of Source Files
# ---------------------------------------------------------------------------


class TestSourceFilesRemainUnchanged:
    """Verify that validator never mutates or modifies input files."""

    def test_source_files_remain_strictly_unchanged(self):
        def file_hash(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        tasks_sha_before = file_hash(TASKS_PATH)
        trajs_sha_before = file_hash(TRAJECTORIES_PATH)
        tasks_mtime_before = TASKS_PATH.stat().st_mtime
        trajs_mtime_before = TRAJECTORIES_PATH.stat().st_mtime

        # Run validation
        report = validate_datasets(TASKS_PATH, TRAJECTORIES_PATH)
        assert report.passed is True

        tasks_sha_after = file_hash(TASKS_PATH)
        trajs_sha_after = file_hash(TRAJECTORIES_PATH)
        tasks_mtime_after = TASKS_PATH.stat().st_mtime
        trajs_mtime_after = TRAJECTORIES_PATH.stat().st_mtime

        assert tasks_sha_before == tasks_sha_after
        assert trajs_sha_before == trajs_sha_after
        assert tasks_mtime_before == tasks_mtime_after
        assert trajs_mtime_before == trajs_mtime_after


# ---------------------------------------------------------------------------
# 12. CLI Execution & Edge Cases
# ---------------------------------------------------------------------------


class TestCLIAndEdgeCases:
    """Test command-line interface execution and missing files."""

    def test_cli_main_success(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "dataset_validator",
                "--tasks",
                str(TASKS_PATH),
                "--trajectories",
                str(TRAJECTORIES_PATH),
            ],
        )
        exit_code = main()
        assert exit_code == 0

    def test_cli_main_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        bad_task_file = write_jsonl(tmp_path / "bad.jsonl", ["{not_json}"])
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "dataset_validator",
                "--tasks",
                str(bad_task_file),
                "--trajectories",
                str(TRAJECTORIES_PATH),
            ],
        )
        exit_code = main()
        assert exit_code == 1

    def test_cli_main_json_flag(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "dataset_validator",
                "--tasks",
                str(TASKS_PATH),
                "--trajectories",
                str(TRAJECTORIES_PATH),
                "--json",
            ],
        )
        exit_code = main()
        assert exit_code == 0

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["passed"] is True
        assert parsed["summary"]["valid_tasks_count"] == 24

    def test_missing_files_reported_as_findings(self, tmp_path: Path):
        fake_tasks = tmp_path / "does_not_exist_tasks.jsonl"
        fake_trajs = tmp_path / "does_not_exist_trajs.jsonl"

        report = validate_datasets(fake_tasks, fake_trajs)
        assert report.passed is False
        assert report.checks_failed >= 1

        missing_findings = [f for f in report.findings if "not found" in f.message.lower()]
        assert len(missing_findings) == 2

"""
Dataset validation module for AgentEvalOps.

Performs comprehensive integrity, schema, reference, cryptographic, and
execution validation across task and trajectory datasets without mutating
the underlying raw data files.
"""

from __future__ import annotations

import argparse
from enum import Enum
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from code.shared.models import Outcome, TaskRecord, TrajectoryRecord
from code.tasks.grader import grade

# ---------------------------------------------------------------------------
# Check Name Constants
# ---------------------------------------------------------------------------

CHECK_JSONL_PARSING = "jsonl_parsing"
CHECK_SCHEMA_VALIDATION = "schema_validation"
CHECK_UNIQUE_IDS = "unique_ids"
CHECK_REFERENCE_INTEGRITY = "reference_integrity"
CHECK_CONTENT_HASHES = "content_hashes"
CHECK_TRAJECTORY_INTEGRITY = "trajectory_integrity"
CHECK_GOLD_ANSWER_EVALUATION = "gold_answer_evaluation"
CHECK_EXACT_DUPLICATES = "exact_duplicates"

ALL_CHECKS: list[str] = [
    CHECK_JSONL_PARSING,
    CHECK_SCHEMA_VALIDATION,
    CHECK_UNIQUE_IDS,
    CHECK_REFERENCE_INTEGRITY,
    CHECK_CONTENT_HASHES,
    CHECK_TRAJECTORY_INTEGRITY,
    CHECK_GOLD_ANSWER_EVALUATION,
    CHECK_EXACT_DUPLICATES,
]


# ---------------------------------------------------------------------------
# Data Models for Findings and Reports
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    """Severity levels for validation findings."""

    error = "error"
    warning = "warning"
    info = "info"


class Finding(BaseModel):
    """A single diagnostic issue discovered during dataset validation."""

    severity: Severity = Severity.error
    check_name: str = Field(..., description="Name of the check that produced this finding.")
    record_id: str | None = Field(None, description="Identifier of the offending record, if known.")
    file_path: str | None = Field(None, description="Path to the file where the issue was found.")
    line_number: int | None = Field(None, description="1-indexed line number in the source file.")
    message: str = Field(..., description="Actionable explanation of the issue.")

    model_config = {"extra": "forbid"}


class DatasetSummary(BaseModel):
    """Aggregate statistics summarizing the validated dataset."""

    total_tasks_read: int = 0
    total_trajectories_read: int = 0
    valid_tasks_count: int = 0
    valid_trajectories_count: int = 0
    tasks_by_family: dict[str, int] = Field(default_factory=dict)
    trajectories_by_outcome: dict[str, int] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class ValidationReport(BaseModel):
    """Structured report produced by dataset validation."""

    task_file: str
    trajectory_file: str
    total_task_records_read: int = 0
    total_trajectory_records_read: int = 0
    checks_passed: int = 0
    checks_failed: int = 0
    passed: bool = True
    findings: list[Finding] = Field(default_factory=list)
    summary: DatasetSummary = Field(default_factory=DatasetSummary)

    model_config = {"extra": "forbid"}

    @property
    def has_errors(self) -> bool:
        """Return True if any findings have error severity."""
        return any(f.severity == Severity.error for f in self.findings)

    @property
    def errors(self) -> list[Finding]:
        """List all error-severity findings."""
        return [f for f in self.findings if f.severity == Severity.error]

    @property
    def warnings(self) -> list[Finding]:
        """List all warning-severity findings."""
        return [f for f in self.findings if f.severity == Severity.warning]

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation of the report."""
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        """Return serialized JSON string of the report."""
        return self.model_dump_json(indent=indent)


# ---------------------------------------------------------------------------
# Cryptographic and Hash Helpers
# ---------------------------------------------------------------------------


def compute_canonical_content_hash(task_id: str, prompt: str, gold_answer: str | None) -> str:
    """
    Return SHA-256 digest of canonical task content matching the generator implementation.
    """
    payload = json.dumps(
        {"task_id": task_id, "prompt": prompt, "gold_answer": gold_answer},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _task_content_key(task: TaskRecord) -> str:
    """Canonical representation of task semantic content (excluding ID and metadata)."""
    payload = {
        "prompt": task.prompt,
        "gold_answer": task.gold_answer,
        "expected_tools": sorted(task.expected_tools),
        "evaluation_rule": task.evaluation_rule.value,
        "template_family": task.template_family,
        "task_type": task.task_type,
        "answerable": task.answerable,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _trajectory_content_key(traj: TrajectoryRecord) -> str:
    """Canonical representation of trajectory execution (excluding run_id and latency jitter)."""
    payload = {
        "task_id": traj.task_id,
        "configuration_hash": traj.configuration_hash,
        "model_version": traj.model_version,
        "prompt_version": traj.prompt_version,
        "final_answer": traj.final_answer,
        "outcome": traj.outcome.value,
        "steps": [
            {
                "step_index": s.step_index,
                "action_type": s.action_type.value,
                "tool_name": s.tool_name,
                "arguments": s.arguments,
                "tool_result": {
                    "tool_name": s.tool_result.tool_name,
                    "success": s.tool_result.success,
                    "output": s.tool_result.output,
                    "error": s.tool_result.error,
                }
                if s.tool_result
                else None,
                "error": s.error,
            }
            for s in traj.steps
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


# ---------------------------------------------------------------------------
# Main Validation Pipeline
# ---------------------------------------------------------------------------


def validate_datasets(
    task_path: str | Path,
    trajectory_path: str | Path,
) -> ValidationReport:
    """
    Validate task and trajectory datasets from disk without mutating them.

    Args:
        task_path:       Path to tasks.jsonl
        trajectory_path: Path to trajectories.jsonl

    Returns:
        ValidationReport containing statistics, findings, and overall passed flag.
    """
    task_file_resolved = Path(task_path).resolve()
    trajectory_file_resolved = Path(trajectory_path).resolve()

    findings: list[Finding] = []

    # -----------------------------------------------------------------------
    # Step 1: Read and parse tasks.jsonl
    # -----------------------------------------------------------------------
    valid_tasks: list[tuple[int, TaskRecord]] = []
    total_task_records_read = 0

    if not task_file_resolved.exists():
        findings.append(
            Finding(
                severity=Severity.error,
                check_name=CHECK_JSONL_PARSING,
                file_path=str(task_file_resolved),
                message=f"Task file not found: {task_file_resolved}",
            )
        )
    else:
        with task_file_resolved.open("r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, 1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                total_task_records_read += 1

                # JSON parsing check
                try:
                    raw_obj = json.loads(stripped)
                except Exception as exc:
                    findings.append(
                        Finding(
                            severity=Severity.error,
                            check_name=CHECK_JSONL_PARSING,
                            file_path=str(task_file_resolved),
                            line_number=line_no,
                            message=f"Line {line_no} is not valid JSON: {exc}",
                        )
                    )
                    continue

                if not isinstance(raw_obj, dict):
                    findings.append(
                        Finding(
                            severity=Severity.error,
                            check_name=CHECK_JSONL_PARSING,
                            file_path=str(task_file_resolved),
                            line_number=line_no,
                            message=f"Line {line_no} does not decode to a JSON object (got {type(raw_obj).__name__})",
                        )
                    )
                    continue

                # Schema validation check
                try:
                    task = TaskRecord.model_validate(raw_obj)
                    valid_tasks.append((line_no, task))
                except ValidationError as exc:
                    rec_id = raw_obj.get("task_id") if isinstance(raw_obj.get("task_id"), str) else None
                    err_msg = "; ".join(
                        f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                        for err in exc.errors()
                    )
                    findings.append(
                        Finding(
                            severity=Severity.error,
                            check_name=CHECK_SCHEMA_VALIDATION,
                            record_id=rec_id,
                            file_path=str(task_file_resolved),
                            line_number=line_no,
                            message=f"Task schema validation failed: {err_msg}",
                        )
                    )

    # -----------------------------------------------------------------------
    # Step 2: Read and parse trajectories.jsonl
    # -----------------------------------------------------------------------
    valid_trajectories: list[tuple[int, TrajectoryRecord]] = []
    total_trajectory_records_read = 0

    if not trajectory_file_resolved.exists():
        findings.append(
            Finding(
                severity=Severity.error,
                check_name=CHECK_JSONL_PARSING,
                file_path=str(trajectory_file_resolved),
                message=f"Trajectory file not found: {trajectory_file_resolved}",
            )
        )
    else:
        with trajectory_file_resolved.open("r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, 1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                total_trajectory_records_read += 1

                # JSON parsing check
                try:
                    raw_obj = json.loads(stripped)
                except Exception as exc:
                    findings.append(
                        Finding(
                            severity=Severity.error,
                            check_name=CHECK_JSONL_PARSING,
                            file_path=str(trajectory_file_resolved),
                            line_number=line_no,
                            message=f"Line {line_no} is not valid JSON: {exc}",
                        )
                    )
                    continue

                if not isinstance(raw_obj, dict):
                    findings.append(
                        Finding(
                            severity=Severity.error,
                            check_name=CHECK_JSONL_PARSING,
                            file_path=str(trajectory_file_resolved),
                            line_number=line_no,
                            message=f"Line {line_no} does not decode to a JSON object (got {type(raw_obj).__name__})",
                        )
                    )
                    continue

                # Schema validation check
                try:
                    traj = TrajectoryRecord.model_validate(raw_obj)
                    valid_trajectories.append((line_no, traj))
                except ValidationError as exc:
                    rec_id = raw_obj.get("run_id") if isinstance(raw_obj.get("run_id"), str) else None
                    err_msg = "; ".join(
                        f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                        for err in exc.errors()
                    )
                    findings.append(
                        Finding(
                            severity=Severity.error,
                            check_name=CHECK_SCHEMA_VALIDATION,
                            record_id=rec_id,
                            file_path=str(trajectory_file_resolved),
                            line_number=line_no,
                            message=f"Trajectory schema validation failed: {err_msg}",
                        )
                    )

    # -----------------------------------------------------------------------
    # Step 3: Unique IDs
    # -----------------------------------------------------------------------
    seen_task_ids: dict[str, int] = {}
    for line_no, task in valid_tasks:
        if task.task_id in seen_task_ids:
            first_line = seen_task_ids[task.task_id]
            findings.append(
                Finding(
                    severity=Severity.error,
                    check_name=CHECK_UNIQUE_IDS,
                    record_id=task.task_id,
                    file_path=str(task_file_resolved),
                    line_number=line_no,
                    message=f"Duplicate task_id '{task.task_id}' (first seen at line {first_line})",
                )
            )
        else:
            seen_task_ids[task.task_id] = line_no

    seen_run_ids: dict[str, int] = {}
    for line_no, traj in valid_trajectories:
        if traj.run_id in seen_run_ids:
            first_line = seen_run_ids[traj.run_id]
            findings.append(
                Finding(
                    severity=Severity.error,
                    check_name=CHECK_UNIQUE_IDS,
                    record_id=traj.run_id,
                    file_path=str(trajectory_file_resolved),
                    line_number=line_no,
                    message=f"Duplicate trajectory run_id '{traj.run_id}' (first seen at line {first_line})",
                )
            )
        else:
            seen_run_ids[traj.run_id] = line_no

    # -----------------------------------------------------------------------
    # Step 4: Reference Integrity
    # -----------------------------------------------------------------------
    tasks_by_id: dict[str, TaskRecord] = {task.task_id: task for _, task in valid_tasks}

    for line_no, traj in valid_trajectories:
        if traj.task_id not in tasks_by_id:
            findings.append(
                Finding(
                    severity=Severity.error,
                    check_name=CHECK_REFERENCE_INTEGRITY,
                    record_id=traj.run_id,
                    file_path=str(trajectory_file_resolved),
                    line_number=line_no,
                    message=f"Trajectory '{traj.run_id}' references non-existent task_id '{traj.task_id}'",
                )
            )

    # -----------------------------------------------------------------------
    # Step 5: Content Hashes
    # -----------------------------------------------------------------------
    for line_no, task in valid_tasks:
        if task.metadata and task.metadata.content_hash is not None:
            ch = task.metadata.content_hash
            is_valid_format = len(ch) == 64 and all(c in "0123456789abcdef" for c in ch)
            if not is_valid_format:
                findings.append(
                    Finding(
                        severity=Severity.error,
                        check_name=CHECK_CONTENT_HASHES,
                        record_id=task.task_id,
                        file_path=str(task_file_resolved),
                        line_number=line_no,
                        message=(
                            f"Task '{task.task_id}' has malformed content_hash '{ch}': "
                            "expected 64-character lowercase hex string"
                        ),
                    )
                )
            else:
                expected_hash = compute_canonical_content_hash(
                    task.task_id, task.prompt, task.gold_answer
                )
                if ch != expected_hash:
                    findings.append(
                        Finding(
                            severity=Severity.error,
                            check_name=CHECK_CONTENT_HASHES,
                            record_id=task.task_id,
                            file_path=str(task_file_resolved),
                            line_number=line_no,
                            message=(
                                f"Task '{task.task_id}' content_hash mismatch: "
                                f"expected '{expected_hash}', found '{ch}'"
                            ),
                        )
                    )

    # -----------------------------------------------------------------------
    # Step 6: Trajectory Integrity
    # -----------------------------------------------------------------------
    for line_no, traj in valid_trajectories:
        actual_step_count = len(traj.steps)
        if traj.metrics.total_steps != actual_step_count:
            findings.append(
                Finding(
                    severity=Severity.error,
                    check_name=CHECK_TRAJECTORY_INTEGRITY,
                    record_id=traj.run_id,
                    file_path=str(trajectory_file_resolved),
                    line_number=line_no,
                    message=(
                        f"Trajectory '{traj.run_id}' step count mismatch: "
                        f"metrics.total_steps={traj.metrics.total_steps} != len(steps)={actual_step_count}"
                    ),
                )
            )

        if traj.outcome == Outcome.success:
            if len(traj.errors) > 0:
                findings.append(
                    Finding(
                        severity=Severity.error,
                        check_name=CHECK_TRAJECTORY_INTEGRITY,
                        record_id=traj.run_id,
                        file_path=str(trajectory_file_resolved),
                        line_number=line_no,
                        message=(
                            f"Trajectory '{traj.run_id}' has outcome='success' but non-empty errors: {traj.errors}"
                        ),
                    )
                )
            if traj.final_answer is None:
                findings.append(
                    Finding(
                        severity=Severity.error,
                        check_name=CHECK_TRAJECTORY_INTEGRITY,
                        record_id=traj.run_id,
                        file_path=str(trajectory_file_resolved),
                        line_number=line_no,
                        message=f"Trajectory '{traj.run_id}' has outcome='success' but final_answer is null",
                    )
                )
        elif traj.outcome == Outcome.error:
            has_error_logged = (
                len(traj.errors) > 0
                or any(s.error for s in traj.steps)
                or any(s.tool_result and s.tool_result.error for s in traj.steps)
            )
            if not has_error_logged:
                findings.append(
                    Finding(
                        severity=Severity.error,
                        check_name=CHECK_TRAJECTORY_INTEGRITY,
                        record_id=traj.run_id,
                        file_path=str(trajectory_file_resolved),
                        line_number=line_no,
                        message=(
                            f"Trajectory '{traj.run_id}' has outcome='error' but no error recorded "
                            "in trajectory errors or step results"
                        ),
                    )
                )

    # -----------------------------------------------------------------------
    # Step 7: Gold-Answer Evaluation
    # -----------------------------------------------------------------------
    for line_no, traj in valid_trajectories:
        if traj.task_id in tasks_by_id:
            referenced_task = tasks_by_id[traj.task_id]
            grading_result = grade(referenced_task, traj.final_answer)
            if not grading_result.correct:
                findings.append(
                    Finding(
                        severity=Severity.error,
                        check_name=CHECK_GOLD_ANSWER_EVALUATION,
                        record_id=traj.run_id,
                        file_path=str(trajectory_file_resolved),
                        line_number=line_no,
                        message=(
                            f"Trajectory '{traj.run_id}' on task '{referenced_task.task_id}' failed gold evaluation: "
                            f"proposed={traj.final_answer!r}, gold={referenced_task.gold_answer!r}, "
                            f"rule='{referenced_task.evaluation_rule.value}'. Detail: {grading_result.detail}"
                        ),
                    )
                )

    # -----------------------------------------------------------------------
    # Step 8: Exact Duplicates
    # -----------------------------------------------------------------------
    seen_task_content: dict[str, tuple[int, str]] = {}
    for line_no, task in valid_tasks:
        key = _task_content_key(task)
        if key in seen_task_content:
            first_line, first_task_id = seen_task_content[key]
            if task.task_id == first_task_id:
                msg = (
                    f"Task '{task.task_id}' at line {line_no} is an exact duplicate of task at line {first_line} "
                    "(identical ID and content)"
                )
            else:
                msg = (
                    f"Task '{task.task_id}' at line {line_no} has identical content to task '{first_task_id}' "
                    f"at line {first_line} (distinct IDs but identical content)"
                )
            findings.append(
                Finding(
                    severity=Severity.error,
                    check_name=CHECK_EXACT_DUPLICATES,
                    record_id=task.task_id,
                    file_path=str(task_file_resolved),
                    line_number=line_no,
                    message=msg,
                )
            )
        else:
            seen_task_content[key] = (line_no, task.task_id)

    seen_traj_content: dict[str, tuple[int, str]] = {}
    for line_no, traj in valid_trajectories:
        key = _trajectory_content_key(traj)
        if key in seen_traj_content:
            first_line, first_run_id = seen_traj_content[key]
            if traj.run_id == first_run_id:
                msg = (
                    f"Trajectory '{traj.run_id}' at line {line_no} is an exact duplicate of trajectory "
                    f"at line {first_line} (identical run_id and execution content)"
                )
            else:
                msg = (
                    f"Trajectory '{traj.run_id}' at line {line_no} has identical execution content to trajectory "
                    f"'{first_run_id}' at line {first_line} (distinct run_ids but identical execution content)"
                )
            findings.append(
                Finding(
                    severity=Severity.error,
                    check_name=CHECK_EXACT_DUPLICATES,
                    record_id=traj.run_id,
                    file_path=str(trajectory_file_resolved),
                    line_number=line_no,
                    message=msg,
                )
            )
        else:
            seen_traj_content[key] = (line_no, traj.run_id)

    # -----------------------------------------------------------------------
    # Step 9: Dataset Summary & Report Assembly
    # -----------------------------------------------------------------------
    tasks_by_family: dict[str, int] = {}
    for _, task in valid_tasks:
        tasks_by_family[task.template_family] = tasks_by_family.get(task.template_family, 0) + 1

    trajectories_by_outcome: dict[str, int] = {}
    for _, traj in valid_trajectories:
        trajectories_by_outcome[traj.outcome.value] = (
            trajectories_by_outcome.get(traj.outcome.value, 0) + 1
        )

    summary = DatasetSummary(
        total_tasks_read=total_task_records_read,
        total_trajectories_read=total_trajectory_records_read,
        valid_tasks_count=len(valid_tasks),
        valid_trajectories_count=len(valid_trajectories),
        tasks_by_family=dict(sorted(tasks_by_family.items())),
        trajectories_by_outcome=dict(sorted(trajectories_by_outcome.items())),
    )

    # Deterministic findings ordering: (file_path, line_number, check_name, record_id, message)
    findings.sort(
        key=lambda f: (
            f.file_path or "",
            f.line_number or 0,
            f.check_name,
            f.record_id or "",
            f.message,
        )
    )

    failed_checks = {f.check_name for f in findings if f.severity == Severity.error}
    checks_failed = len(failed_checks)
    checks_passed = len(ALL_CHECKS) - checks_failed
    passed = checks_failed == 0

    return ValidationReport(
        task_file=str(task_file_resolved),
        trajectory_file=str(trajectory_file_resolved),
        total_task_records_read=total_task_records_read,
        total_trajectory_records_read=total_trajectory_records_read,
        checks_passed=checks_passed,
        checks_failed=checks_failed,
        passed=passed,
        findings=findings,
        summary=summary,
    )


# Convenience alias
validate_dataset = validate_datasets


# ---------------------------------------------------------------------------
# Human-Readable Formatting & CLI
# ---------------------------------------------------------------------------


def format_report_human(report: ValidationReport) -> str:
    """Format a ValidationReport into human-readable text."""
    lines = [
        "=" * 72,
        " AgentEvalOps Dataset Validation Report",
        "=" * 72,
        f"Overall Status:       {'PASSED' if report.passed else 'FAILED'}",
        f"Checks Evaluated:     {report.checks_passed} passed, {report.checks_failed} failed (total {len(ALL_CHECKS)})",
        f"Task File:            {report.task_file}",
        f"  Records Read:       {report.total_task_records_read}",
        f"  Valid TaskRecords:  {report.summary.valid_tasks_count}",
        f"Trajectory File:      {report.trajectory_file}",
        f"  Records Read:       {report.total_trajectory_records_read}",
        f"  Valid Trajectories: {report.summary.valid_trajectories_count}",
        "",
        "--- Dataset Breakdown ---",
        f"Tasks by Family:      {report.summary.tasks_by_family}",
        f"Trajectories Outcome: {report.summary.trajectories_by_outcome}",
        "",
        f"Total Findings:       {len(report.findings)} ({len(report.errors)} errors, {len(report.warnings)} warnings)",
    ]

    if report.findings:
        lines.append("")
        lines.append("--- Findings List ---")
        for idx, f in enumerate(report.findings, 1):
            loc_parts = []
            if f.file_path:
                loc_parts.append(Path(f.file_path).name)
            if f.line_number is not None:
                loc_parts.append(f"line {f.line_number}")
            if f.record_id:
                loc_parts.append(f"id={f.record_id}")
            loc_str = f" ({', '.join(loc_parts)})" if loc_parts else ""
            lines.append(f"  [{idx:02d}] [{f.severity.value.upper()}] [{f.check_name}]{loc_str}: {f.message}")

    lines.append("=" * 72)
    return "\n".join(lines)


def main() -> int:
    """CLI entry point for dataset validation."""
    parser = argparse.ArgumentParser(
        description="Validate AgentEvalOps task and trajectory datasets."
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/raw/tasks.jsonl"),
        help="Path to tasks.jsonl (default: data/raw/tasks.jsonl).",
    )
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=Path("data/raw/trajectories.jsonl"),
        help="Path to trajectories.jsonl (default: data/raw/trajectories.jsonl).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output instead of text summary.",
    )

    args = parser.parse_args()

    report = validate_datasets(task_path=args.tasks, trajectory_path=args.trajectories)

    if args.json:
        print(report.to_json(indent=2))
    else:
        print(format_report_human(report))

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())

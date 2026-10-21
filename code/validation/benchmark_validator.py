"""
Benchmark Validation Suite for AgentEvalOps (Assignment 1, Part B).

Implements 12 automated checks covering:
1. JSONL Parsing
2. Pydantic Schema Integrity (TaskRecord & TrajectoryRecord)
3. Unique Identifiers (task_id, run_id)
4. Reference Integrity (trajectory -> task mapping)
5. Cryptographic Content Hashes (SHA-256 canonical digest)
6. Duplicate Tasks (content-level duplicates)
7. Prompt Answer Leakage (gold answers leaking into prompt text)
8. Tool Layer Determinism (pure tools produce reproducible outputs)
9. Tool Call Argument Types (arguments match tool schemas)
10. Metric Value Ranges (latency >= 0, tokens >= 0, sequential steps)
11. Missing Values (null fields where prohibited)
12. Gold Answer Grading Consistency (grader alignment)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel, Field

from code.agent.tools import TOOL_REGISTRY
from code.shared.models import ActionType, Outcome, TaskRecord, TrajectoryRecord
from code.tasks.grader import grade


class ValidationIssue(BaseModel):
    check_name: str
    severity: str = "error"  # error, warning
    record_id: str | None = None
    line_number: int | None = None
    message: str


class SuiteResult(BaseModel):
    passed: bool
    total_checks: int = 12
    checks_passed: int = 0
    checks_failed: int = 0
    issues: list[ValidationIssue] = Field(default_factory=list)
    tasks_count: int = 0
    trajectories_count: int = 0
    summary_by_check: dict[str, bool] = Field(default_factory=dict)


def compute_task_hash(task: TaskRecord) -> str:
    payload = json.dumps(
        {"task_id": task.task_id, "prompt": task.prompt, "gold_answer": task.gold_answer},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class BenchmarkValidator:
    """Comprehensive validator enforcing 12 automated checks for AgentEvalOps."""

    def __init__(self, tasks_path: Path, trajectories_path: Path):
        self.tasks_path = Path(tasks_path)
        self.trajectories_path = Path(trajectories_path)
        self.issues: list[ValidationIssue] = []
        self.check_status: dict[str, bool] = {}

    def _add_issue(self, check: str, msg: str, rec_id: str | None = None, line: int | None = None, severity: str = "error") -> None:
        self.issues.append(ValidationIssue(
            check_name=check,
            severity=severity,
            record_id=rec_id,
            line_number=line,
            message=msg,
        ))

    def run_all_checks(self) -> SuiteResult:
        tasks: list[tuple[int, TaskRecord]] = []
        trajectories: list[tuple[int, TrajectoryRecord]] = []

        raw_tasks: list[tuple[int, dict]] = []
        raw_trajectories: list[tuple[int, dict]] = []

        # 1. JSONL Parsing Check
        c1 = "1_jsonl_parsing"
        c1_ok = True
        for p, is_task in [(self.tasks_path, True), (self.trajectories_path, False)]:
            if not p.exists():
                self._add_issue(c1, f"File not found: {p}")
                c1_ok = False
                continue
            with p.open("r", encoding="utf-8") as fh:
                for idx, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                        if not isinstance(raw, dict):
                            self._add_issue(c1, f"Line {idx} in {p.name} is not a JSON object", line=idx)
                            c1_ok = False
                        else:
                            if is_task:
                                raw_tasks.append((idx, raw))
                            else:
                                raw_trajectories.append((idx, raw))
                    except Exception as e:
                        self._add_issue(c1, f"Invalid JSON at line {idx} in {p.name}: {e}", line=idx)
                        c1_ok = False
        self.check_status[c1] = c1_ok

        # 2. Schema Validation Check
        c2 = "2_pydantic_schemas"
        c2_ok = True
        if self.tasks_path.exists():
            with self.tasks_path.open("r", encoding="utf-8") as fh:
                for idx, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = TaskRecord.model_validate_json(line)
                        tasks.append((idx, t))
                    except Exception as e:
                        self._add_issue(c2, f"Task schema error at line {idx}: {e}", line=idx)
                        c2_ok = False

        if self.trajectories_path.exists():
            with self.trajectories_path.open("r", encoding="utf-8") as fh:
                for idx, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        tr = TrajectoryRecord.model_validate_json(line)
                        trajectories.append((idx, tr))
                    except Exception as e:
                        self._add_issue(c2, f"Trajectory schema error at line {idx}: {e}", line=idx)
                        c2_ok = False
        self.check_status[c2] = c2_ok

        # 3. Unique Identifiers
        c3 = "3_unique_identifiers"
        c3_ok = True
        seen_t_ids: set[str] = set()
        for idx, t in tasks:
            if t.task_id in seen_t_ids:
                self._add_issue(c3, f"Duplicate task_id '{t.task_id}'", rec_id=t.task_id, line=idx)
                c3_ok = False
            seen_t_ids.add(t.task_id)

        seen_r_ids: set[str] = set()
        for idx, tr in trajectories:
            if tr.run_id in seen_r_ids:
                self._add_issue(c3, f"Duplicate run_id '{tr.run_id}'", rec_id=tr.run_id, line=idx)
                c3_ok = False
            seen_r_ids.add(tr.run_id)
        self.check_status[c3] = c3_ok

        # 4. Reference Integrity
        c4 = "4_reference_integrity"
        c4_ok = True
        task_id_set = {t.task_id for _, t in tasks}
        for idx, tr in trajectories:
            if tr.task_id not in task_id_set:
                self._add_issue(c4, f"Trajectory references non-existent task_id '{tr.task_id}'", rec_id=tr.run_id, line=idx)
                c4_ok = False
        self.check_status[c4] = c4_ok

        # 5. Cryptographic Content Hashes
        c5 = "5_content_hashes"
        c5_ok = True
        for idx, t in tasks:
            if t.metadata and t.metadata.content_hash:
                expected = compute_task_hash(t)
                if t.metadata.content_hash != expected:
                    self._add_issue(c5, f"Content hash mismatch for {t.task_id}", rec_id=t.task_id, line=idx)
                    c5_ok = False
        self.check_status[c5] = c5_ok

        # 6. Duplicate Tasks (Semantic/Exact Duplicates)
        c6 = "6_duplicate_tasks"
        c6_ok = True
        prompts_seen: dict[str, str] = {}
        for idx, t in tasks:
            norm_prompt = re.sub(r"\s+", " ", t.prompt.strip().lower())
            if norm_prompt in prompts_seen:
                prev_id = prompts_seen[norm_prompt]
                self._add_issue(c6, f"Task '{t.task_id}' has duplicate prompt to '{prev_id}'", rec_id=t.task_id, line=idx)
                c6_ok = False
            else:
                prompts_seen[norm_prompt] = t.task_id
        self.check_status[c6] = c6_ok

        # 7. Prompt Answer Leakage
        c7 = "7_prompt_answer_leakage"
        c7_ok = True
        for idx, t in tasks:
            if t.answerable and t.gold_answer and t.expected_tools:
                ans_str = t.gold_answer.strip().lower()
                # Skip trivial tokens like '0', 'true', 'false', 'none'
                if len(ans_str) > 2 and ans_str not in {"true", "false", "none", "null", "pass", "fail"}:
                    # Check if exact answer is embedded in prompt text
                    if ans_str in t.prompt.lower():
                        self._add_issue(c7, f"Answer leakage: prompt contains gold answer '{t.gold_answer}'", rec_id=t.task_id, line=idx)
                        c7_ok = False
        self.check_status[c7] = c7_ok

        # 8. Tool Layer Determinism
        c8 = "8_tool_determinism"
        c8_ok = True
        try:
            calc_fn = TOOL_REGISTRY.get("calculate")
            if calc_fn:
                r1 = calc_fn("45 * 12 + 10")
                r2 = calc_fn("45 * 12 + 10")
                if r1.output != r2.output:
                    self._add_issue(c8, "calculate tool produced non-deterministic output")
                    c8_ok = False
            read_fn = TOOL_REGISTRY.get("read_file")
            if read_fn:
                rf1 = read_fn("config.txt")
                rf2 = read_fn("config.txt")
                if rf1.output != rf2.output:
                    self._add_issue(c8, "read_file tool produced non-deterministic output")
                    c8_ok = False
        except Exception as e:
            self._add_issue(c8, f"Tool determinism test raised exception: {e}")
            c8_ok = False
        self.check_status[c8] = c8_ok

        # 9. Tool Call Argument Types
        c9 = "9_tool_call_arguments"
        c9_ok = True
        for idx, tr in trajectories:
            for s in tr.steps:
                if s.action_type == ActionType.tool_call and s.tool_name:
                    if s.tool_name == "calculate":
                        if not isinstance(s.arguments, dict) or "expression" not in s.arguments:
                            self._add_issue(c9, f"calculate missing 'expression' dict argument in {tr.run_id}", rec_id=tr.run_id, line=idx)
                            c9_ok = False
                    elif s.tool_name == "retrieve":
                        if not isinstance(s.arguments, dict) or "query" not in s.arguments:
                            self._add_issue(c9, f"retrieve missing 'query' dict argument in {tr.run_id}", rec_id=tr.run_id, line=idx)
                            c9_ok = False
                    elif s.tool_name == "read_file":
                        if not isinstance(s.arguments, dict) or "filename" not in s.arguments:
                            self._add_issue(c9, f"read_file missing 'filename' dict argument in {tr.run_id}", rec_id=tr.run_id, line=idx)
                            c9_ok = False
        self.check_status[c9] = c9_ok

        # 10. Metric Value Ranges & Step Integrity
        c10 = "10_metric_value_ranges"
        c10_ok = True
        for idx, raw in raw_trajectories:
            metrics = raw.get("metrics")
            if isinstance(metrics, dict):
                lat = metrics.get("total_latency_ms")
                if isinstance(lat, (int, float)) and lat < 0:
                    self._add_issue(c10, f"Negative latency in trajectory at line {idx}: {lat}", line=idx)
                    c10_ok = False
                p_tok = metrics.get("total_prompt_tokens")
                if isinstance(p_tok, int) and p_tok < 0:
                    self._add_issue(c10, f"Negative prompt tokens in trajectory at line {idx}: {p_tok}", line=idx)
                    c10_ok = False
                tot_steps = metrics.get("total_steps")
                steps = raw.get("steps")
                if isinstance(tot_steps, int) and isinstance(steps, list) and tot_steps != len(steps):
                    self._add_issue(c10, f"Total steps mismatch ({tot_steps} != {len(steps)}) in trajectory at line {idx}", line=idx)
                    c10_ok = False

        for idx, tr in trajectories:
            if tr.metrics.total_latency_ms is not None and tr.metrics.total_latency_ms < 0:
                self._add_issue(c10, f"Negative latency in {tr.run_id}: {tr.metrics.total_latency_ms}", rec_id=tr.run_id, line=idx)
                c10_ok = False
            if tr.metrics.total_prompt_tokens is not None and tr.metrics.total_prompt_tokens < 0:
                self._add_issue(c10, f"Negative prompt tokens in {tr.run_id}", rec_id=tr.run_id, line=idx)
                c10_ok = False
            for step_i, step in enumerate(tr.steps):
                if step.step_index != step_i:
                    self._add_issue(c10, f"Non-sequential step index at step {step_i} in {tr.run_id}", rec_id=tr.run_id, line=idx)
                    c10_ok = False
                    break
        self.check_status[c10] = c10_ok

        # 11. Missing Values
        c11 = "11_missing_values"
        c11_ok = True
        for idx, t in tasks:
            if not t.task_id or not t.prompt or not t.template_family:
                self._add_issue(c11, f"Missing required core field in task {t.task_id}", rec_id=t.task_id, line=idx)
                c11_ok = False
        for idx, tr in trajectories:
            if tr.outcome == Outcome.success and tr.final_answer is None:
                self._add_issue(c11, f"Trajectory {tr.run_id} marked success but has null final_answer", rec_id=tr.run_id, line=idx)
                c11_ok = False
        self.check_status[c11] = c11_ok

        # 12. Gold Answer Grading Consistency
        c12 = "12_gold_answer_grading_consistency"
        c12_ok = True
        task_map = {t.task_id: t for _, t in tasks}
        for idx, tr in trajectories:
            if tr.task_id in task_map:
                t = task_map[tr.task_id]
                res = grade(t, tr.final_answer)
                if tr.outcome == Outcome.success and not res.correct:
                    self._add_issue(c12, f"Trajectory claims success but grader failed: {tr.run_id}", rec_id=tr.run_id, line=idx)
                    c12_ok = False
        self.check_status[c12] = c12_ok

        # Summary computation
        checks_passed = sum(1 for v in self.check_status.values() if v)
        checks_failed = len(self.check_status) - checks_passed
        passed = checks_failed == 0

        return SuiteResult(
            passed=passed,
            total_checks=len(self.check_status),
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            issues=self.issues,
            tasks_count=len(tasks),
            trajectories_count=len(trajectories),
            summary_by_check=self.check_status,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 12 benchmark validation checks.")
    parser.add_argument("--tasks", type=Path, default=Path("data/processed/tasks_revised.jsonl"))
    parser.add_argument("--trajectories", type=Path, default=Path("data/processed/trajectories_processed.jsonl"))
    parser.add_argument("--json", action="store_true", help="Output JSON report")

    args = parser.parse_args()
    validator = BenchmarkValidator(args.tasks, args.trajectories)
    result = validator.run_all_checks()

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print("\n" + "=" * 70)
        print("AGENT EVAL OPS - 12-CHECK BENCHMARK VALIDATION REPORT")
        print("=" * 70)
        print(f"Tasks:        {args.tasks} ({result.tasks_count} records)")
        print(f"Trajectories: {args.trajectories} ({result.trajectories_count} records)")
        print(f"Status:       {'PASSED' if result.passed else 'FAILED'}")
        print(f"Checks:       {result.checks_passed} passed, {result.checks_failed} failed (Total: {result.total_checks})\n")

        print("Check Breakdown:")
        for check_name, ok in result.summary_by_check.items():
            status_tag = "[PASS]" if ok else "[FAIL]"
            print(f"  {status_tag:<8} {check_name}")

        if result.issues:
            print(f"\nIssues Found ({len(result.issues)}):")
            for iss in result.issues[:10]:
                loc = f" (line {iss.line_number})" if iss.line_number else ""
                print(f"  - [{iss.check_name}] {iss.message}{loc}")
            if len(result.issues) > 10:
                print(f"  ... and {len(result.issues) - 10} more issues.")
        print("=" * 70 + "\n")

    exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()

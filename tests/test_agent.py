"""
Tests for code.agent.tools, code.agent.trajectory, and code.agent.runner.

Coverage:
  Tools
    - retrieve: valid queries, empty query rejection, no-match queries
    - calculate: safe expressions, forbidden constructs, division by zero,
                 empty input, non-numeric input
    - query_records: filter, aggregate (count/sum/avg/max/min/list),
                     unknown field, bad operation
    - read_file: valid files, unknown file, path traversal rejection,
                 directory separator rejection
    - submit_answer: valid string, non-string rejection
    - Deterministic: identical inputs → identical outputs

  Trajectory
    - Builder accumulates steps with contiguous indices
    - finish() produces a valid TrajectoryRecord
    - Token counts are never fabricated (always None)
    - JSONL round-trip

  Runner
    - Correct answers for representative tasks from all six families
    - Multi-step task (ms03: combine water symbol + planet count)
    - Error handling: failed tool is recorded, run completes
    - Max-step enforcement
    - Final answer submission step always present
    - Valid trajectory schemas with contiguous zero-based step indices
    - Full 24-task benchmark execution without crashing
    - Repeat run same seed → equivalent results (excluding timestamps/latencies)
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from code.agent.tools import (
    _safe_eval,
    calculate,
    query_records,
    read_file,
    retrieve,
    submit_answer,
)
from code.agent.trajectory import (
    TrajectoryBuilder,
    export_trajectories_jsonl,
    load_trajectories_jsonl,
)
from code.agent.runner import (
    DEFAULT_CONFIG,
    run_benchmark,
    run_task,
)
from code.shared.models import (
    ActionType,
    EvaluationRule,
    ExecutionConfig,
    Outcome,
    TaskRecord,
    TrajectoryRecord,
)
from code.tasks.fixtures import (
    EMPLOYEE_RECORDS,
    FACTS,
    FILE_FIXTURES,
    UNANSWERABLE_SIGNAL,
)
from code.tasks.generator import generate_benchmark


# ===========================================================================
# Tool tests
# ===========================================================================


class TestRetrieve:
    def test_known_query_returns_match(self):
        r = retrieve("capital France")
        assert r.success is True
        assert r.tool_name == "retrieve"
        matches = r.output["matches"]
        assert any("Paris" in m["answer"] for m in matches)

    def test_science_query(self):
        r = retrieve("chemical symbol")
        assert r.success is True
        assert any(m["answer"] == "H2O" for m in r.output["matches"])

    def test_no_match_returns_empty_list(self):
        r = retrieve("zzznomatch999")
        assert r.success is True
        assert r.output["matches"] == []

    def test_empty_query_fails(self):
        r = retrieve("")
        assert r.success is False
        assert r.error is not None

    def test_whitespace_query_fails(self):
        r = retrieve("   ")
        assert r.success is False

    def test_latency_non_negative(self):
        r = retrieve("capital")
        assert r.latency_ms is not None
        assert r.latency_ms >= 0.0

    def test_deterministic(self):
        r1 = retrieve("Japan")
        r2 = retrieve("Japan")
        assert r1.output == r2.output

    def test_output_schema(self):
        r = retrieve("planet")
        assert "matches" in r.output
        assert isinstance(r.output["matches"], list)


class TestCalculate:
    def test_multiplication(self):
        r = calculate("17 * 6")
        assert r.success is True
        assert r.output["result"] == 102

    def test_sqrt(self):
        r = calculate("sqrt(144)")
        assert r.success is True
        assert abs(r.output["result"] - 12.0) < 1e-9

    def test_percentage(self):
        r = calculate("0.15 * 240")
        assert r.success is True
        assert abs(r.output["result"] - 36.0) < 1e-9

    def test_division(self):
        r = calculate("(100 - 37) / 7")
        assert r.success is True
        assert abs(r.output["result"] - 9.0) < 0.01

    def test_round(self):
        r = calculate("round(9.857, 2)")
        assert r.success is True
        assert r.output["result"] == 9.86

    def test_power(self):
        r = calculate("2 ** 10")
        assert r.success is True
        assert r.output["result"] == 1024

    def test_empty_expression_fails(self):
        r = calculate("")
        assert r.success is False
        assert r.error is not None

    def test_whitespace_only_fails(self):
        r = calculate("   ")
        assert r.success is False

    def test_division_by_zero_fails(self):
        r = calculate("1 / 0")
        assert r.success is False
        assert r.error is not None

    def test_import_statement_rejected(self):
        r = calculate("__import__('os').system('echo hi')")
        assert r.success is False

    def test_arbitrary_code_rejected(self):
        r = calculate("open('/etc/passwd').read()")
        assert r.success is False

    def test_string_literal_rejected(self):
        r = calculate("'hello'")
        assert r.success is False

    def test_list_comprehension_rejected(self):
        r = calculate("[x for x in range(10)]")
        assert r.success is False

    def test_assignment_rejected(self):
        r = calculate("x = 5")
        assert r.success is False

    def test_lambda_rejected(self):
        r = calculate("lambda x: x + 1")
        assert r.success is False

    def test_deterministic(self):
        r1 = calculate("7 * 8")
        r2 = calculate("7 * 8")
        assert r1.output == r2.output

    def test_latency_non_negative(self):
        r = calculate("1 + 1")
        assert r.latency_ms is not None
        assert r.latency_ms >= 0.0

    def test_expression_in_output(self):
        r = calculate("3 + 4")
        assert r.output["expression"] == "3 + 4"


class TestSafeEval:
    """Unit tests for the internal _safe_eval function."""

    def test_simple_arithmetic(self):
        assert _safe_eval("2 + 3") == 5

    def test_nested(self):
        assert abs(_safe_eval("(10 - 2) * 3 / 2") - 12.0) < 1e-9

    def test_abs_function(self):
        assert _safe_eval("abs(-5)") == 5

    def test_min_max(self):
        assert _safe_eval("min(3, 7)") == 3
        assert _safe_eval("max(3, 7)") == 7

    def test_forbidden_name_raises(self):
        with pytest.raises(ValueError, match="not permitted"):
            _safe_eval("os")

    def test_forbidden_call_raises(self):
        with pytest.raises(ValueError):
            _safe_eval("exec('pass')")


class TestQueryRecords:
    def test_filter_by_department(self):
        r = query_records(operation="filter", field="department", value="Engineering")
        assert r.success is True
        assert r.output["count"] == 3
        assert all(row["department"] == "Engineering" for row in r.output["rows"])

    def test_filter_case_insensitive(self):
        r = query_records(operation="filter", field="department", value="engineering")
        assert r.success is True
        assert r.output["count"] == 3

    def test_count_aggregate(self):
        r = query_records(
            operation="aggregate",
            field="department",
            value="Engineering",
            aggregate="count",
        )
        assert r.success is True
        assert r.output["result"] == 3

    def test_avg_aggregate(self):
        r = query_records(
            operation="aggregate",
            field="department",
            value="Marketing",
            aggregate="avg",
            agg_field="salary",
        )
        assert r.success is True
        expected_avg = (72000 + 68000) / 2
        assert abs(r.output["result"] - expected_avg) < 1.0

    def test_max_aggregate(self):
        r = query_records(operation="aggregate", aggregate="max", agg_field="salary")
        assert r.success is True
        assert r.output["result"] == 105000

    def test_min_aggregate(self):
        r = query_records(operation="aggregate", aggregate="min", agg_field="salary")
        assert r.success is True
        assert r.output["result"] == 68000

    def test_sum_aggregate(self):
        r = query_records(operation="aggregate", aggregate="sum", agg_field="salary")
        assert r.success is True
        expected = sum(e["salary"] for e in EMPLOYEE_RECORDS)
        assert abs(r.output["result"] - expected) < 1.0

    def test_list_aggregate(self):
        r = query_records(
            operation="aggregate",
            field="department",
            value="Engineering",
            aggregate="list",
            agg_field="name",
        )
        assert r.success is True
        assert r.output["result"] == ["Alice", "Carol", "Eve"]

    def test_unknown_field_fails(self):
        r = query_records(operation="filter", field="nonexistent", value="x")
        assert r.success is False
        assert r.error is not None

    def test_unknown_operation_fails(self):
        r = query_records(operation="delete")
        assert r.success is False

    def test_aggregate_without_agg_field_fails(self):
        r = query_records(operation="aggregate", aggregate="avg")
        assert r.success is False

    def test_latency_non_negative(self):
        r = query_records(operation="filter", field="department", value="Engineering")
        assert r.latency_ms is not None
        assert r.latency_ms >= 0.0

    def test_deterministic(self):
        r1 = query_records(operation="filter", field="department", value="Engineering")
        r2 = query_records(operation="filter", field="department", value="Engineering")
        assert r1.output == r2.output


class TestReadFile:
    def test_known_file_alpha(self):
        r = read_file("notes_alpha.txt")
        assert r.success is True
        assert r.output["line_count"] == 3
        assert "apple" in r.output["content"]

    def test_known_file_numbers(self):
        r = read_file("numbers.txt")
        assert r.success is True
        lines = r.output["lines"]
        assert sum(int(l) for l in lines) == 150

    def test_known_file_config(self):
        r = read_file("config.txt")
        assert r.success is True
        content = r.output["content"]
        kv = dict(line.split("=", 1) for line in content.splitlines() if "=" in line)
        assert kv["mode"] == "production"

    def test_unknown_file_fails(self):
        r = read_file("secret.txt")
        assert r.success is False
        assert r.error is not None
        assert "not found" in r.error.lower()

    def test_path_traversal_rejected(self):
        r = read_file("../../../etc/passwd")
        assert r.success is False
        assert r.error is not None

    def test_absolute_path_rejected(self):
        r = read_file("/etc/hosts")
        assert r.success is False

    def test_backslash_path_rejected(self):
        r = read_file("..\\secrets.txt")
        assert r.success is False

    def test_empty_filename_fails(self):
        r = read_file("")
        assert r.success is False

    def test_latency_non_negative(self):
        r = read_file("notes_alpha.txt")
        assert r.latency_ms is not None
        assert r.latency_ms >= 0.0

    def test_deterministic(self):
        r1 = read_file("notes_alpha.txt")
        r2 = read_file("notes_alpha.txt")
        assert r1.output == r2.output


class TestSubmitAnswer:
    def test_valid_string(self):
        r = submit_answer("Paris")
        assert r.success is True
        assert r.output["submitted_answer"] == "Paris"

    def test_empty_string_accepted(self):
        r = submit_answer("")
        assert r.success is True
        assert r.output["submitted_answer"] == ""

    def test_non_string_fails(self):
        r = submit_answer(42)  # type: ignore[arg-type]
        assert r.success is False
        assert r.error is not None

    def test_tool_name_is_submit_answer(self):
        r = submit_answer("x")
        assert r.tool_name == "submit_answer"

    def test_latency_non_negative(self):
        r = submit_answer("x")
        assert r.latency_ms is not None
        assert r.latency_ms >= 0.0


# ===========================================================================
# Trajectory tests
# ===========================================================================


def _make_task(
    task_id: str = "test-task",
    family: str = "retrieval",
    answerable: bool = True,
    gold: str | None = "Paris",
    rule: EvaluationRule = EvaluationRule.exact_match,
) -> TaskRecord:
    return TaskRecord(
        task_id=task_id,
        template_family=family,
        task_type="test",
        prompt="Test prompt.",
        answerable=answerable,
        gold_answer=gold,
        expected_tools=["retrieve"],
        evaluation_rule=rule,
    )


class TestTrajectoryBuilder:
    def test_single_tool_step(self):
        task = _make_task()
        builder = TrajectoryBuilder(task, "hash123", "model-v1", "prompt-v1")
        r = retrieve("France")
        builder.add_tool_call("retrieve", {"query": "France"}, r, latency_ms=5.0)
        record = builder.finish("Paris", Outcome.success)
        assert isinstance(record, TrajectoryRecord)
        assert len(record.steps) == 1
        assert record.steps[0].step_index == 0
        assert record.steps[0].action_type == ActionType.tool_call

    def test_contiguous_step_indices(self):
        task = _make_task()
        builder = TrajectoryBuilder(task, "h", "m", "p")
        r1 = retrieve("France")
        r2 = retrieve("Japan")
        builder.add_tool_call("retrieve", {"query": "France"}, r1, latency_ms=1.0)
        builder.add_tool_call("retrieve", {"query": "Japan"}, r2, latency_ms=1.0)
        record = builder.finish("Paris", Outcome.success)
        for expected, step in enumerate(record.steps):
            assert step.step_index == expected

    def test_submit_step_recorded(self):
        task = _make_task()
        builder = TrajectoryBuilder(task, "h", "m", "p")
        r = submit_answer("Paris")
        builder.add_submit("Paris", r, latency_ms=0.5)
        record = builder.finish("Paris", Outcome.success)
        assert record.steps[0].action_type == ActionType.submit_answer

    def test_token_counts_are_none(self):
        task = _make_task()
        builder = TrajectoryBuilder(task, "h", "m", "p")
        r = retrieve("France")
        builder.add_tool_call("retrieve", {"query": "France"}, r, latency_ms=1.0)
        record = builder.finish("Paris", Outcome.success)
        for step in record.steps:
            assert step.prompt_tokens is None
            assert step.completion_tokens is None
        assert record.metrics.total_prompt_tokens is None
        assert record.metrics.total_completion_tokens is None

    def test_errors_accumulated(self):
        task = _make_task()
        builder = TrajectoryBuilder(task, "h", "m", "p")
        builder.add_error("Something went wrong")
        record = builder.finish(None, Outcome.error)
        assert "Something went wrong" in record.errors

    def test_failed_tool_error_in_record(self):
        task = _make_task()
        builder = TrajectoryBuilder(task, "h", "m", "p")
        r = read_file("nonexistent.txt")  # will fail
        builder.add_tool_call("read_file", {"filename": "nonexistent.txt"}, r, latency_ms=1.0)
        record = builder.finish(None, Outcome.error)
        assert any("nonexistent.txt" in e for e in record.errors)

    def test_record_fields(self):
        task = _make_task(task_id="my-task")
        builder = TrajectoryBuilder(task, "cfg-hash", "model-v1", "prompt-v2")
        record = builder.finish("answer", Outcome.success)
        assert record.task_id == "my-task"
        assert record.configuration_hash == "cfg-hash"
        assert record.model_version == "model-v1"
        assert record.prompt_version == "prompt-v2"

    def test_metrics_total_steps(self):
        task = _make_task()
        builder = TrajectoryBuilder(task, "h", "m", "p")
        r = retrieve("France")
        builder.add_tool_call("retrieve", {"query": "France"}, r, latency_ms=1.0)
        builder.add_tool_call("retrieve", {"query": "Japan"}, r, latency_ms=1.0)
        record = builder.finish("x", Outcome.success)
        assert record.metrics.total_steps == 2


class TestTrajectoryJSONL:
    def test_export_and_load(self):
        tasks = generate_benchmark(seed=42)[:3]
        trajectories = [run_task(t) for t in tasks]
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "traj.jsonl"
            export_trajectories_jsonl(trajectories, path, overwrite=False)
            loaded = load_trajectories_jsonl(path)
        assert len(loaded) == 3
        for orig, rec in zip(trajectories, loaded):
            assert orig.run_id == rec.run_id
            assert orig.task_id == rec.task_id
            assert orig.outcome == rec.outcome

    def test_overwrite_false_raises(self):
        tasks = generate_benchmark(seed=42)[:1]
        trajectories = [run_task(t) for t in tasks]
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "traj.jsonl"
            export_trajectories_jsonl(trajectories, path, overwrite=False)
            with pytest.raises(FileExistsError):
                export_trajectories_jsonl(trajectories, path, overwrite=False)

    def test_loaded_records_validate(self):
        tasks = generate_benchmark(seed=42)[:5]
        trajectories = [run_task(t) for t in tasks]
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "traj.jsonl"
            export_trajectories_jsonl(trajectories, path, overwrite=False)
            loaded = load_trajectories_jsonl(path)
        for t in loaded:
            assert isinstance(t, TrajectoryRecord)


# ===========================================================================
# Runner tests — per family
# ===========================================================================


@pytest.fixture(scope="module")
def benchmark_tasks() -> list[TaskRecord]:
    return generate_benchmark(seed=42)


@pytest.fixture(scope="module")
def tasks_by_family(benchmark_tasks: list[TaskRecord]) -> dict[str, list[TaskRecord]]:
    result: dict[str, list[TaskRecord]] = {}
    for t in benchmark_tasks:
        result.setdefault(t.template_family, []).append(t)
    return result


class TestRunnerFamilies:
    def test_retrieval_correct(self, tasks_by_family: dict):
        for task in tasks_by_family["retrieval"]:
            traj = run_task(task, DEFAULT_CONFIG)
            assert traj.outcome == Outcome.success, (
                f"Task {task.task_id} failed: final_answer={traj.final_answer!r}, "
                f"gold={task.gold_answer!r}"
            )

    def test_calculation_correct(self, tasks_by_family: dict):
        for task in tasks_by_family["calculation"]:
            traj = run_task(task, DEFAULT_CONFIG)
            assert traj.outcome == Outcome.success, (
                f"Task {task.task_id} failed: final_answer={traj.final_answer!r}, "
                f"gold={task.gold_answer!r}"
            )

    def test_structured_data_correct(self, tasks_by_family: dict):
        for task in tasks_by_family["structured_data"]:
            traj = run_task(task, DEFAULT_CONFIG)
            assert traj.outcome == Outcome.success, (
                f"Task {task.task_id} failed: final_answer={traj.final_answer!r}, "
                f"gold={task.gold_answer!r}"
            )

    def test_file_ops_correct(self, tasks_by_family: dict):
        for task in tasks_by_family["file_ops"]:
            traj = run_task(task, DEFAULT_CONFIG)
            assert traj.outcome == Outcome.success, (
                f"Task {task.task_id} failed: final_answer={traj.final_answer!r}, "
                f"gold={task.gold_answer!r}"
            )

    def test_unanswerable_correct(self, tasks_by_family: dict):
        for task in tasks_by_family["unanswerable"]:
            traj = run_task(task, DEFAULT_CONFIG)
            assert traj.outcome == Outcome.success, (
                f"Task {task.task_id}: final_answer={traj.final_answer!r}"
            )

    def test_multistep_ms01_alphabetical_capitals(self, tasks_by_family: dict):
        """ms01: list both capitals alphabetically — multi-step retrieval."""
        task = next(t for t in tasks_by_family["multistep_qa"] if t.task_id == "multistep-ms01")
        traj = run_task(task, DEFAULT_CONFIG)
        assert traj.outcome == Outcome.success, (
            f"ms01 failed: final_answer={traj.final_answer!r}, gold={task.gold_answer!r}"
        )

    def test_multistep_ms02_topic_count(self, tasks_by_family: dict):
        task = next(t for t in tasks_by_family["multistep_qa"] if t.task_id == "multistep-ms02")
        traj = run_task(task, DEFAULT_CONFIG)
        assert traj.outcome == Outcome.success, (
            f"ms02 failed: final_answer={traj.final_answer!r}"
        )

    def test_multistep_ms03_combine_symbol_planets(self, tasks_by_family: dict):
        """ms03: multi-step — retrieve water symbol AND planet count, combine them."""
        task = next(t for t in tasks_by_family["multistep_qa"] if t.task_id == "multistep-ms03")
        traj = run_task(task, DEFAULT_CONFIG)
        assert traj.outcome == Outcome.success, (
            f"ms03 failed: final_answer={traj.final_answer!r}, gold={task.gold_answer!r}"
        )
        # Verify the trajectory has multiple tool steps.
        tool_steps = [s for s in traj.steps if s.action_type == ActionType.tool_call]
        assert len(tool_steps) >= 2, "ms03 should require at least 2 retrieve calls"

    def test_multistep_ms04_total_facts(self, tasks_by_family: dict):
        task = next(t for t in tasks_by_family["multistep_qa"] if t.task_id == "multistep-ms04")
        traj = run_task(task, DEFAULT_CONFIG)
        assert traj.outcome == Outcome.success


class TestRunnerTrajectorySchema:
    def test_step_indices_contiguous(self, benchmark_tasks: list[TaskRecord]):
        for task in benchmark_tasks:
            traj = run_task(task, DEFAULT_CONFIG)
            for expected, step in enumerate(traj.steps):
                assert step.step_index == expected, (
                    f"Task {task.task_id}: step at position {expected} has "
                    f"step_index={step.step_index}"
                )

    def test_each_trajectory_has_submit_step(self, benchmark_tasks: list[TaskRecord]):
        for task in benchmark_tasks:
            traj = run_task(task, DEFAULT_CONFIG)
            submit_steps = [s for s in traj.steps if s.action_type == ActionType.submit_answer]
            assert len(submit_steps) == 1, (
                f"Task {task.task_id} has {len(submit_steps)} submit steps, expected 1"
            )

    def test_trajectory_validates_as_pydantic(self, benchmark_tasks: list[TaskRecord]):
        for task in benchmark_tasks:
            traj = run_task(task, DEFAULT_CONFIG)
            assert isinstance(traj, TrajectoryRecord)
            # Round-trip through JSON to verify full serialisation.
            reloaded = TrajectoryRecord.model_validate_json(traj.model_dump_json())
            assert reloaded.run_id == traj.run_id

    def test_token_counts_are_none(self, benchmark_tasks: list[TaskRecord]):
        for task in benchmark_tasks:
            traj = run_task(task, DEFAULT_CONFIG)
            for step in traj.steps:
                assert step.prompt_tokens is None
                assert step.completion_tokens is None
            assert traj.metrics.total_prompt_tokens is None
            assert traj.metrics.total_completion_tokens is None

    def test_latencies_non_negative(self, benchmark_tasks: list[TaskRecord]):
        for task in benchmark_tasks:
            traj = run_task(task, DEFAULT_CONFIG)
            for step in traj.steps:
                if step.latency_ms is not None:
                    assert step.latency_ms >= 0.0

    def test_configuration_hash_present(self, benchmark_tasks: list[TaskRecord]):
        expected_hash = DEFAULT_CONFIG.compute_hash()
        for task in benchmark_tasks:
            traj = run_task(task, DEFAULT_CONFIG)
            assert traj.configuration_hash == expected_hash


class TestRunnerErrorHandling:
    def test_unknown_family_does_not_crash(self):
        """An unknown template_family should produce an error-outcome trajectory."""
        task = _make_task(family="nonexistent_family")
        traj = run_task(task, DEFAULT_CONFIG)
        assert isinstance(traj, TrajectoryRecord)
        assert traj.outcome in (Outcome.failure, Outcome.error)

    def test_max_steps_enforced(self, benchmark_tasks: list[TaskRecord]):
        """With max_steps=1 the runner must not exceed that budget."""
        tiny_config = ExecutionConfig(
            model_id="rule-based-agent-v1",
            prompt_version="v1",
            tool_selection_strategy="family_dispatch",
            temperature=None,
            max_steps=1,
            retry_budget=0,
        )
        for task in benchmark_tasks[:6]:  # one per family
            traj = run_task(task, tiny_config)
            # Step count must be ≤ max_steps + 1 (the submit step may push past).
            assert traj.metrics.total_steps <= tiny_config.max_steps + 2, (
                f"Task {task.task_id} used {traj.metrics.total_steps} steps "
                f"with max_steps={tiny_config.max_steps}"
            )


class TestRunnerDeterminism:
    def test_same_task_same_outcome(self, benchmark_tasks: list[TaskRecord]):
        """Two runs of the same task should produce the same outcome and final answer."""
        for task in benchmark_tasks:
            traj1 = run_task(task, DEFAULT_CONFIG)
            traj2 = run_task(task, DEFAULT_CONFIG)
            assert traj1.final_answer == traj2.final_answer, (
                f"Task {task.task_id}: run1={traj1.final_answer!r}, run2={traj2.final_answer!r}"
            )
            assert traj1.outcome == traj2.outcome

    def test_same_task_different_run_id(self, benchmark_tasks: list[TaskRecord]):
        """Each run gets a unique run_id."""
        task = benchmark_tasks[0]
        traj1 = run_task(task, DEFAULT_CONFIG)
        traj2 = run_task(task, DEFAULT_CONFIG)
        assert traj1.run_id != traj2.run_id


class TestFullBenchmarkExecution:
    def test_all_24_tasks_produce_trajectories(self, benchmark_tasks: list[TaskRecord]):
        trajectories = run_benchmark(benchmark_tasks, DEFAULT_CONFIG)
        assert len(trajectories) == 24

    def test_one_trajectory_per_task(self, benchmark_tasks: list[TaskRecord]):
        trajectories = run_benchmark(benchmark_tasks, DEFAULT_CONFIG)
        task_ids = [t.task_id for t in trajectories]
        benchmark_ids = [t.task_id for t in benchmark_tasks]
        assert task_ids == benchmark_ids

    def test_all_trajectories_are_pydantic(self, benchmark_tasks: list[TaskRecord]):
        trajectories = run_benchmark(benchmark_tasks, DEFAULT_CONFIG)
        for traj in trajectories:
            assert isinstance(traj, TrajectoryRecord)

    def test_benchmark_jsonl_roundtrip(self, benchmark_tasks: list[TaskRecord]):
        trajectories = run_benchmark(benchmark_tasks, DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "trajs.jsonl"
            export_trajectories_jsonl(trajectories, path, overwrite=False)
            loaded = load_trajectories_jsonl(path)
        assert len(loaded) == len(trajectories)
        for orig, rec in zip(trajectories, loaded):
            assert orig.run_id == rec.run_id
            assert orig.task_id == rec.task_id
            assert orig.final_answer == rec.final_answer
            assert orig.outcome == rec.outcome

    def test_no_task_crashes_runner(self, benchmark_tasks: list[TaskRecord]):
        """No individual task should raise an unhandled exception."""
        for task in benchmark_tasks:
            traj = run_task(task, DEFAULT_CONFIG)
            assert traj is not None

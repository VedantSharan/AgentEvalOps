"""
Tests for code.tracing.mlflow_tracer and runner tracing integration.

Covers:
  - Trace creation and closure for task executions
  - Task and configuration metadata capture
  - Tool input/output capture across different tools
  - Successful and failed tool call span statuses and attributes
  - Final answer, grading result, and outcome capture
  - Tracing disabled / unavailable behavior (no-op safety)
  - Small benchmark execution and trace verification
  - Trajectory JSONL preservation when tracing is enabled
  - No dependence on external/remote tracking servers
"""

from __future__ import annotations

import pathlib
import tempfile
from typing import Generator

import pytest

from code.agent.runner import run_benchmark, run_task
from code.agent.tools import calculate, read_file, retrieve
from code.agent.trajectory import export_trajectories_jsonl, load_trajectories_jsonl
from code.shared.models import (
    EvaluationRule,
    ExecutionConfig,
    Outcome,
    TaskMetadata,
    TaskRecord,
    TrajectoryRecord,
)
from code.tasks.generator import BENCHMARK_SEED, generate_benchmark
from code.tracing.mlflow_tracer import MLflowTracer, resolve_tracking_uri


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_tracking_uri(tmp_path: pathlib.Path) -> str:
    """Provide an isolated, file-based SQLite tracking URI per test."""
    db_file = tmp_path / "test_mlflow.db"
    return f"sqlite:///{db_file.as_posix()}"


@pytest.fixture
def tracer(tmp_tracking_uri: str) -> MLflowTracer:
    """Provide a fresh MLflowTracer pointing to an isolated SQLite store."""
    return MLflowTracer(
        tracking_uri=tmp_tracking_uri,
        experiment_name="test-experiment",
        enabled=True,
    )


@pytest.fixture
def sample_task() -> TaskRecord:
    """A standard retrieval task for testing."""
    return TaskRecord(
        task_id="retrieval-f01",
        template_family="retrieval",
        task_type="information_retrieval",
        prompt="What is the capital of France? Use the retrieve tool.",
        answerable=True,
        gold_answer="Paris",
        expected_tools=["retrieve"],
        evaluation_rule=EvaluationRule.exact_match,
        metadata=TaskMetadata(
            provenance="test",
            created_at="2026-10-20T00:00:00Z",
            generator_version="0.1.0",
            seed=42,
            content_hash="abc123hash",
        ),
    )


@pytest.fixture
def sample_config() -> ExecutionConfig:
    return ExecutionConfig(
        model_id="test-model-v1",
        prompt_version="v1",
        tool_selection_strategy="family_dispatch",
        temperature=None,
        max_steps=5,
        retry_budget=1,
    )


# ---------------------------------------------------------------------------
# URI resolution tests
# ---------------------------------------------------------------------------


class TestTrackingURIResolution:
    def test_sqlite_uri_preserved(self):
        uri = "sqlite:///c:/data/mlflow.db"
        assert resolve_tracking_uri(uri) == uri

    def test_http_uri_preserved(self):
        uri = "http://localhost:5000"
        assert resolve_tracking_uri(uri) == uri

    def test_file_uri_preserved(self):
        uri = "file:///c:/data/mlruns"
        assert resolve_tracking_uri(uri) == uri

    def test_db_file_path_converted_to_sqlite(self, tmp_path: pathlib.Path):
        db = tmp_path / "custom.db"
        resolved = resolve_tracking_uri(db)
        assert resolved.startswith("sqlite:///")
        assert "custom.db" in resolved

    def test_dir_path_creates_mlflow_db(self, tmp_path: pathlib.Path):
        resolved = resolve_tracking_uri(tmp_path)
        assert resolved.startswith("sqlite:///")
        assert "mlflow.db" in resolved


# ---------------------------------------------------------------------------
# Trace creation, closure, and metadata
# ---------------------------------------------------------------------------


class TestTraceCreationAndMetadata:
    def test_trace_created_and_closed(
        self,
        tracer: MLflowTracer,
        sample_task: TaskRecord,
        sample_config: ExecutionConfig,
    ):
        traj = run_task(sample_task, sample_config, tracer=tracer)
        assert traj.outcome == Outcome.success

        traces = tracer.get_traces()
        assert len(traces) == 1
        trace = traces[0]

        # Root span must exist and be named after the task
        assert len(trace.data.spans) >= 2  # root agent + at least 1 tool call
        root_span = trace.data.spans[0]
        assert "task_retrieval-f01" in root_span.name
        assert root_span.span_type == "AGENT"

    def test_metadata_recorded_on_root_span_and_tags(
        self,
        tracer: MLflowTracer,
        sample_task: TaskRecord,
        sample_config: ExecutionConfig,
    ):
        run_task(sample_task, sample_config, tracer=tracer)
        traces = tracer.get_traces()
        assert len(traces) == 1
        trace = traces[0]

        # Check trace-level tags
        tags = trace.info.tags
        assert tags.get("task_id") == "retrieval-f01"
        assert tags.get("template_family") == "retrieval"
        assert tags.get("task_type") == "information_retrieval"
        assert tags.get("outcome") == "success"
        assert tags.get("correct") == "True"

        # Check root span attributes
        root_span = trace.data.spans[0]
        attrs = root_span.attributes
        assert attrs["task_id"] == "retrieval-f01"
        assert attrs["template_family"] == "retrieval"
        assert attrs["configuration_hash"] == sample_config.compute_hash()
        assert attrs["model_version"] == "test-model-v1"
        assert attrs["prompt_version"] == "v1"
        assert attrs["random_seed"] == 42

    def test_tokens_remain_unavailable_not_invented(
        self,
        tracer: MLflowTracer,
        sample_task: TaskRecord,
        sample_config: ExecutionConfig,
    ):
        traj = run_task(sample_task, sample_config, tracer=tracer)
        # TrajectoryRecord tokens must be None
        assert traj.metrics.total_prompt_tokens is None
        assert traj.metrics.total_completion_tokens is None

        # Trace attributes must not fabricate tokens
        traces = tracer.get_traces()
        root_span = traces[0].data.spans[0]
        assert "prompt_tokens" not in root_span.attributes or root_span.attributes.get("prompt_tokens") is None
        assert "completion_tokens" not in root_span.attributes or root_span.attributes.get("completion_tokens") is None


# ---------------------------------------------------------------------------
# Tool capture (inputs, outputs, latency, errors)
# ---------------------------------------------------------------------------


class TestToolCapture:
    def test_retrieval_tool_span_captured(
        self,
        tracer: MLflowTracer,
        sample_task: TaskRecord,
        sample_config: ExecutionConfig,
    ):
        run_task(sample_task, sample_config, tracer=tracer)
        traces = tracer.get_traces()
        spans = traces[0].data.spans

        retrieve_span = next((s for s in spans if s.name == "retrieve"), None)
        assert retrieve_span is not None
        assert retrieve_span.span_type == "TOOL"
        assert "query" in retrieve_span.inputs
        assert "matches" in retrieve_span.outputs
        assert retrieve_span.attributes["success"] is True
        assert retrieve_span.attributes["latency_ms"] >= 0.0

    def test_submit_answer_tool_span_captured(
        self,
        tracer: MLflowTracer,
        sample_task: TaskRecord,
        sample_config: ExecutionConfig,
    ):
        run_task(sample_task, sample_config, tracer=tracer)
        traces = tracer.get_traces()
        spans = traces[0].data.spans

        submit_span = next((s for s in spans if s.name == "submit_answer"), None)
        assert submit_span is not None
        assert submit_span.span_type == "TOOL"
        assert submit_span.inputs == {"answer": "Paris"}
        assert submit_span.outputs == {"submitted_answer": "Paris"}
        assert submit_span.attributes["success"] is True

    def test_failed_tool_call_captured_as_error(
        self,
        tracer: MLflowTracer,
    ):
        # Manually trace a failing tool call (e.g. invalid arithmetic)
        with tracer.trace_task(
            TaskRecord(
                task_id="calc-err",
                template_family="calculation",
                task_type="numerical_calculation",
                prompt="Divide by zero",
                answerable=True,
                gold_answer="0",
                evaluation_rule=EvaluationRule.exact_match,
            ),
            ExecutionConfig(
                model_id="m", prompt_version="p", tool_selection_strategy="s", max_steps=1, retry_budget=0
            ),
        ) as task_ctx:
            with tracer.trace_tool("calculate", {"expression": "1 / 0"}) as tool_ctx:
                res = calculate("1 / 0")
                tool_ctx.set_result(res)

        traces = tracer.get_traces()
        assert len(traces) == 1
        calc_span = next(s for s in traces[0].data.spans if s.name == "calculate")
        assert calc_span.attributes["success"] is False
        assert "division by zero" in calc_span.attributes["error"].lower()
        assert calc_span.status.status_code.value == "ERROR"

    def test_unhandled_exception_in_tool_closes_trace(
        self,
        tracer: MLflowTracer,
        sample_task: TaskRecord,
        sample_config: ExecutionConfig,
    ):
        with pytest.raises(RuntimeError):
            with tracer.trace_task(sample_task, sample_config):
                with tracer.trace_tool("crashing_tool", {"arg": 1}):
                    raise RuntimeError("Tool crash test")

        traces = tracer.get_traces()
        assert len(traces) == 1
        trace = traces[0]
        assert trace.info.status.value == "ERROR"


# ---------------------------------------------------------------------------
# Grading and outcome capture
# ---------------------------------------------------------------------------


class TestGradingCapture:
    def test_correct_task_grading_captured(
        self,
        tracer: MLflowTracer,
        sample_task: TaskRecord,
        sample_config: ExecutionConfig,
    ):
        run_task(sample_task, sample_config, tracer=tracer)
        trace = tracer.get_traces()[0]
        root_span = trace.data.spans[0]

        outputs = root_span.outputs
        assert outputs["final_answer"] == "Paris"
        assert outputs["outcome"] == "success"
        grading = outputs["grading_result"]
        assert grading["correct"] is True
        assert grading["rule_applied"] == "exact_match"
        assert grading["gold_answer"] == "Paris"

    def test_unanswerable_task_grading_captured(
        self,
        tracer: MLflowTracer,
        sample_config: ExecutionConfig,
    ):
        un_task = TaskRecord(
            task_id="unanswerable-un01",
            template_family="unanswerable",
            task_type="unanswerable",
            prompt="What is the stock price? If you cannot answer, respond with 'unanswerable'.",
            answerable=False,
            gold_answer=None,
            evaluation_rule=EvaluationRule.unanswerable,
        )
        run_task(un_task, sample_config, tracer=tracer)
        trace = tracer.get_traces()[0]
        root_span = trace.data.spans[0]

        outputs = root_span.outputs
        assert outputs["final_answer"] == "unanswerable"
        assert outputs["outcome"] == "success"
        assert outputs["grading_result"]["correct"] is True
        assert outputs["grading_result"]["rule_applied"] == "unanswerable"


# ---------------------------------------------------------------------------
# Tracing disabled / no-op behavior
# ---------------------------------------------------------------------------


class TestTracingDisabled:
    def test_tracer_disabled_produces_no_traces(
        self,
        tmp_tracking_uri: str,
        sample_task: TaskRecord,
        sample_config: ExecutionConfig,
    ):
        tracer = MLflowTracer(
            tracking_uri=tmp_tracking_uri,
            experiment_name="disabled-exp",
            enabled=False,
        )
        assert tracer.enabled is False

        traj = run_task(sample_task, sample_config, tracer=tracer)
        assert traj.outcome == Outcome.success
        assert tracer.get_traces() == []

    def test_runner_runs_without_tracer(
        self,
        sample_task: TaskRecord,
        sample_config: ExecutionConfig,
    ):
        # tracer is None by default
        traj = run_task(sample_task, sample_config, tracer=None)
        assert traj.outcome == Outcome.success
        assert isinstance(traj, TrajectoryRecord)


# ---------------------------------------------------------------------------
# Benchmark integration & JSONL preservation
# ---------------------------------------------------------------------------


class TestBenchmarkTracingIntegration:
    def test_small_benchmark_creates_traces_for_all_tasks(
        self,
        tracer: MLflowTracer,
        sample_config: ExecutionConfig,
    ):
        all_tasks = generate_benchmark(seed=BENCHMARK_SEED)
        subset = all_tasks[:6]  # one per family

        trajectories = run_benchmark(subset, sample_config, tracer=tracer)
        assert len(trajectories) == 6

        traces = tracer.get_traces()
        assert len(traces) == 6

        # Check that each task produced a distinct trace with matching task_id
        recorded_task_ids = {t.info.tags.get("task_id") for t in traces}
        expected_task_ids = {t.task_id for t in subset}
        assert recorded_task_ids == expected_task_ids

    def test_trajectory_jsonl_preserved_with_tracing(
        self,
        tracer: MLflowTracer,
        sample_config: ExecutionConfig,
        tmp_path: pathlib.Path,
    ):
        all_tasks = generate_benchmark(seed=BENCHMARK_SEED)
        subset = all_tasks[:3]

        trajectories = run_benchmark(subset, sample_config, tracer=tracer)

        jsonl_path = tmp_path / "trajectories.jsonl"
        export_trajectories_jsonl(trajectories, jsonl_path, overwrite=True)

        loaded = load_trajectories_jsonl(jsonl_path)
        assert len(loaded) == 3
        for orig, recovered in zip(trajectories, loaded):
            assert orig.task_id == recovered.task_id
            assert orig.final_answer == recovered.final_answer
            assert orig.outcome == recovered.outcome
            assert len(orig.steps) == len(recovered.steps)
            assert orig.configuration_hash == recovered.configuration_hash

    def test_export_trajectories_to_mlflow(
        self,
        tmp_tracking_uri: str,
    ):
        from code.tracing.export_trajectories_to_mlflow import ingest_trajectories_file

        tasks_path = pathlib.Path("data/processed/tasks_revised.jsonl")
        trajs_path = pathlib.Path("data/processed/trajectories_processed.jsonl")

        count = ingest_trajectories_file(
            trajectories_path=trajs_path,
            tasks_path=tasks_path,
            tracking_uri=tmp_tracking_uri,
            experiment_name="test-ingestion-experiment",
            limit=5,
        )
        assert count == 5

        tracer = MLflowTracer(
            tracking_uri=tmp_tracking_uri,
            experiment_name="test-ingestion-experiment",
        )
        traces = tracer.get_traces(max_results=10)
        assert len(traces) == 5
        for t in traces:
            assert t.info.tags.get("task_id") is not None


"""
MLflow Tracing Integration for AgentEvalOps.

Provides instrumentation for agent execution and tool invocations using
MLflow's native tracing API (mlflow.start_span, SpanType, etc.).

Features:
- Records each task execution as an identifiable root agent span.
- Captures task ID, task family, configuration hash, model version,
  prompt version, seed, and provenance metadata.
- Captures every tool invocation (including `submit_answer`) as a typed
  tool span with arguments, results, success/error status, and latency.
- Captures failed tool calls and runtime exceptions without losing the trace.
- Records the final answer and independently computed grading results.
- Preserves token counts as unavailable (None) rather than inventing mock values.
- Supports configurable local backend store URIs (defaulting to a SQLite
  database in mlruns/mlflow.db).
- Provides a clean no-op mode when tracing is disabled or unavailable.

Local UI Usage:
--------------
To inspect traces locally using MLflow's UI, run::

    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

Then open http://127.0.0.1:5000 in your browser to inspect execution traces,
step latencies, tool arguments, and grading diagnostics.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import time
from typing import Any, Generator

from code.shared.models import ExecutionConfig, Outcome, TaskRecord, ToolResult

# ---------------------------------------------------------------------------
# Default paths and configuration
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_TRACKING_DIR: pathlib.Path = _REPO_ROOT / "mlruns"
DEFAULT_DB_FILE: pathlib.Path = DEFAULT_TRACKING_DIR / "mlflow.db"
DEFAULT_TRACKING_URI: str = f"sqlite:///{DEFAULT_DB_FILE.as_posix()}"
DEFAULT_EXPERIMENT_NAME: str = "AgentEvalOps-Benchmark"


def resolve_tracking_uri(uri: str | pathlib.Path | None) -> str:
    """
    Resolve a user-supplied or default tracking URI into a canonical MLflow URI.

    Handles:
    - None: uses MLFLOW_TRACKING_URI environment variable or default SQLite database.
    - Full URIs (sqlite://, http://, https://, databricks://, file://): returns unchanged.
    - Database files (*.db, *.sqlite): converts to absolute sqlite:/// URI.
    - Directory paths: ensures directory exists and creates sqlite:///<dir>/mlflow.db.

    Args:
        uri: String path/URI, Path object, or None.

    Returns:
        Canonical tracking URI string.
    """
    if uri is None:
        env_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if env_uri:
            return env_uri
        DEFAULT_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
        return DEFAULT_TRACKING_URI

    uri_str = str(uri).strip()
    if uri_str.startswith(("sqlite://", "http://", "https://", "databricks://", "file://")):
        return uri_str

    path = pathlib.Path(uri_str).resolve()
    if path.suffix.lower() in (".db", ".sqlite"):
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{path.as_posix()}"

    # Directory path: place mlflow.db inside it
    path.mkdir(parents=True, exist_ok=True)
    db_path = path / "mlflow.db"
    return f"sqlite:///{db_path.as_posix()}"


# ---------------------------------------------------------------------------
# Trace contexts
# ---------------------------------------------------------------------------


class NoOpTaskTraceContext:
    """No-op placeholder when tracing is disabled."""

    def set_outcome(
        self,
        final_answer: str | None,
        grading: Any,
        outcome: Outcome,
        errors: list[str] | None = None,
    ) -> None:
        pass


class NoOpToolTraceContext:
    """No-op placeholder when tool tracing is disabled."""

    def set_result(self, tool_result: ToolResult) -> None:
        pass


class TaskTraceContext:
    """Active task trace context managing the root agent span."""

    def __init__(self, span: Any, task: TaskRecord, config: ExecutionConfig) -> None:
        self.span = span
        self.task = task
        self.config = config

    def set_outcome(
        self,
        final_answer: str | None,
        grading: Any,
        outcome: Outcome,
        errors: list[str] | None = None,
    ) -> None:
        """
        Record the final submitted answer, evaluation grading, and outcome.

        Args:
            final_answer: The string answer submitted by the agent, or None.
            grading: The GradingResult returned by the benchmark grader.
            outcome: Outcome enum (success, failure, error).
            errors: List of top-level errors accumulated during execution.
        """
        try:
            import mlflow
            from mlflow.entities import SpanStatus, SpanStatusCode
        except ImportError:
            return

        grading_dict = {
            "correct": getattr(grading, "correct", False),
            "rule_applied": getattr(getattr(grading, "rule_applied", None), "value", str(getattr(grading, "rule_applied", ""))),
            "detail": getattr(grading, "detail", ""),
            "diagnostics": getattr(grading, "diagnostics", {}),
            "gold_answer": getattr(grading, "gold_answer", None),
            "proposed_answer": getattr(grading, "proposed_answer", None),
        }

        output_data = {
            "final_answer": final_answer,
            "outcome": outcome.value,
            "grading_result": grading_dict,
            "errors": errors or [],
        }

        self.span.set_outputs(output_data)

        # Record structured attributes on the root span
        self.span.set_attributes({
            "task_id": self.task.task_id,
            "template_family": self.task.template_family,
            "task_type": self.task.task_type,
            "configuration_hash": self.config.compute_hash(),
            "model_version": self.config.model_id,
            "prompt_version": self.config.prompt_version,
            "random_seed": self.task.metadata.seed if self.task.metadata else None,
            "outcome": outcome.value,
            "correct": grading_dict["correct"],
            "evaluation_rule": self.task.evaluation_rule.value,
            "errors": errors or [],
        })

        # Update trace-level tags for easy searching and filtering
        try:
            mlflow.update_current_trace(tags={
                "task_id": self.task.task_id,
                "template_family": self.task.template_family,
                "task_type": self.task.task_type,
                "outcome": outcome.value,
                "correct": str(grading_dict["correct"]),
                "evaluation_rule": self.task.evaluation_rule.value,
                "configuration_hash": self.config.compute_hash(),
                "model_version": self.config.model_id,
                "prompt_version": self.config.prompt_version,
            })
        except Exception:
            pass

        # Set span status
        if outcome == Outcome.success and grading_dict["correct"]:
            self.span.set_status(SpanStatus(SpanStatusCode.OK))
        else:
            detail = grading_dict.get("detail") or f"Task outcome: {outcome.value}"
            self.span.set_status(SpanStatus(SpanStatusCode.ERROR, description=detail))


class ToolTraceContext:
    """Active tool trace context managing a child tool span."""

    def __init__(self, span: Any, tool_name: str, kwargs: dict[str, Any]) -> None:
        self.span = span
        self.tool_name = tool_name
        self.kwargs = kwargs

    def set_result(self, tool_result: ToolResult) -> None:
        """
        Record the structured ToolResult on the active tool span.

        Args:
            tool_result: The ToolResult returned by the tool invocation.
        """
        try:
            from mlflow.entities import SpanStatus, SpanStatusCode
        except ImportError:
            return

        self.span.set_inputs(self.kwargs)
        if tool_result.success:
            self.span.set_outputs(tool_result.output)
            self.span.set_attributes({
                "tool_name": self.tool_name,
                "success": True,
                "error": None,
                "latency_ms": tool_result.latency_ms,
            })
            self.span.set_status(SpanStatus(SpanStatusCode.OK))
        else:
            self.span.set_outputs(None)
            self.span.set_attributes({
                "tool_name": self.tool_name,
                "success": False,
                "error": tool_result.error,
                "latency_ms": tool_result.latency_ms,
            })
            self.span.set_status(
                SpanStatus(SpanStatusCode.ERROR, description=tool_result.error or "Tool failure")
            )


# ---------------------------------------------------------------------------
# MLflowTracer
# ---------------------------------------------------------------------------


class MLflowTracer:
    """
    Centralized MLflow tracer for AgentEvalOps benchmark execution.

    Coordinates trace initialization, span creation, metadata logging,
    and trace flushing.
    """

    def __init__(
        self,
        tracking_uri: str | pathlib.Path | None = None,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        enabled: bool = True,
    ) -> None:
        """
        Initialize the tracer.

        Args:
            tracking_uri: Backend store URI or directory (default: mlruns/mlflow.db).
            experiment_name: Name of the MLflow experiment.
            enabled: Set False to disable tracing completely (all methods become no-ops).
        """
        self.enabled = enabled
        self.experiment_name = experiment_name
        self._mlflow_available = False

        if not self.enabled:
            self.tracking_uri = ""
            return

        os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

        try:
            import mlflow
            self.tracking_uri = resolve_tracking_uri(tracking_uri)
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._mlflow_available = True
        except ImportError:
            # Fall back to disabled mode gracefully
            self.enabled = False
            self.tracking_uri = ""

    @contextlib.contextmanager
    def trace_task(
        self,
        task: TaskRecord,
        config: ExecutionConfig,
    ) -> Generator[TaskTraceContext | NoOpTaskTraceContext, None, None]:
        """
        Open a root agent span for the given task execution.

        Args:
            task: The TaskRecord being executed.
            config: ExecutionConfig governing the agent run.

        Yields:
            A TaskTraceContext (or NoOpTaskTraceContext if tracing is disabled).
        """
        if not self.enabled or not self._mlflow_available:
            yield NoOpTaskTraceContext()
            return

        import mlflow
        from mlflow.entities import SpanStatus, SpanStatusCode, SpanType

        trace_name = f"task_{task.task_id}"
        with mlflow.start_span(name=trace_name, span_type=SpanType.AGENT) as root_span:
            root_span.set_inputs({
                "task_id": task.task_id,
                "prompt": task.prompt,
                "template_family": task.template_family,
                "task_type": task.task_type,
                "answerable": task.answerable,
                "expected_tools": task.expected_tools,
                "gold_answer": task.gold_answer,
            })

            # Record initial metadata on root span
            config_hash = config.compute_hash()
            metadata: dict[str, Any] = {
                "task_id": task.task_id,
                "template_family": task.template_family,
                "task_type": task.task_type,
                "configuration_hash": config_hash,
                "model_version": config.model_id,
                "prompt_version": config.prompt_version,
                "tool_selection_strategy": config.tool_selection_strategy,
                "max_steps": config.max_steps,
            }
            if task.metadata:
                metadata["generator_version"] = task.metadata.generator_version
                metadata["random_seed"] = task.metadata.seed
                metadata["content_hash"] = task.metadata.content_hash
                metadata["provenance"] = task.metadata.provenance

            root_span.set_attributes(metadata)

            try:
                mlflow.update_current_trace(
                    tags={
                        "task_id": task.task_id,
                        "template_family": task.template_family,
                        "task_type": task.task_type,
                        "configuration_hash": config_hash,
                    },
                    metadata={
                        k: str(v) for k, v in metadata.items() if v is not None
                    },
                )
            except Exception:
                pass

            ctx = TaskTraceContext(root_span, task, config)
            try:
                yield ctx
            except Exception as exc:
                root_span.set_status(SpanStatus(SpanStatusCode.ERROR, description=str(exc)))
                root_span.set_attributes({"unhandled_exception": str(exc)})
                raise

    @contextlib.contextmanager
    def trace_tool(
        self,
        tool_name: str,
        kwargs: dict[str, Any],
    ) -> Generator[ToolTraceContext | NoOpToolTraceContext, None, None]:
        """
        Open a child tool span for a single tool call.

        Args:
            tool_name: Name of the tool called.
            kwargs: Arguments passed to the tool.

        Yields:
            A ToolTraceContext (or NoOpToolTraceContext if tracing is disabled).
        """
        if not self.enabled or not self._mlflow_available:
            yield NoOpToolTraceContext()
            return

        import mlflow
        from mlflow.entities import SpanStatus, SpanStatusCode, SpanType

        with mlflow.start_span(name=tool_name, span_type=SpanType.TOOL) as tool_span:
            tool_span.set_inputs(kwargs)
            ctx = ToolTraceContext(tool_span, tool_name, kwargs)
            try:
                yield ctx
            except Exception as exc:
                tool_span.set_status(SpanStatus(SpanStatusCode.ERROR, description=str(exc)))
                tool_span.set_attributes({
                    "tool_name": tool_name,
                    "success": False,
                    "error": str(exc),
                })
                raise

    def flush(self) -> None:
        """Flush any pending asynchronous trace logs to the backend store."""
        if self.enabled and self._mlflow_available:
            import mlflow
            try:
                mlflow.flush_trace_async_logging()
            except Exception:
                pass

    def get_traces(self, max_results: int = 100) -> list[Any]:
        """
        Query and return traces recorded in the current experiment.

        Args:
            max_results: Maximum number of traces to return.

        Returns:
            List of MLflow Trace objects.
        """
        if not self.enabled or not self._mlflow_available:
            return []

        self.flush()
        import mlflow
        client = mlflow.tracking.MlflowClient(self.tracking_uri)
        try:
            return client.search_traces(max_results=max_results)
        except Exception:
            return []

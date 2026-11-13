"""Live MLflow tracing for the A2 agent loop.

Each agent run becomes one MLflow trace with this span tree:

    agent  (AGENT)        task, configuration, final answer, grading
    ├── llm_turn_0  (CHAT_MODEL)  messages in, tool calls out, latency, tokens
    ├── <tool name> (TOOL)        arguments, output or error, latency
    ├── llm_turn_1  ...
    └── grader      (EVALUATOR)   grader v2 verdict and explanation

In A1 the real collection was not traced live; trajectories were ingested into
MLflow afterwards, and only root and tool spans existed. Here every span is
written while the agent runs. With ``enabled=False`` (or when MLflow is
missing) all methods are no-ops and ``trace_id`` falls back to the run id.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Iterator

from code.tracing.mlflow_tracer import DEFAULT_TRACKING_URI, resolve_tracking_uri

DEFAULT_TRACE_EXPERIMENT = "AgentEvalOps-A2-traces"


class _NoSpan:
    trace_id = None

    def set_inputs(self, *_: Any) -> None: ...
    def set_outputs(self, *_: Any) -> None: ...
    def set_attributes(self, *_: Any) -> None: ...
    def set_attribute(self, *_: Any) -> None: ...


class AgentTracer:
    """Opens MLflow spans for one agent run at a time."""

    def __init__(self, tracking_uri: str | None = None, experiment: str = DEFAULT_TRACE_EXPERIMENT,
                 enabled: bool = True) -> None:
        self.enabled = enabled
        self._mlflow: Any = None
        if not enabled:
            return
        try:
            import mlflow
        except ImportError:  # pragma: no cover - mlflow is pinned in requirements
            self.enabled = False
            return
        os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
        os.environ.setdefault("DO_NOT_TRACK", "1")
        mlflow.set_tracking_uri(resolve_tracking_uri(tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI))
        mlflow.set_experiment(experiment)
        self._mlflow = mlflow

    @contextlib.contextmanager
    def span(self, name: str, span_type: str, inputs: dict[str, Any] | None = None,
             attributes: dict[str, Any] | None = None) -> Iterator[Any]:
        if not self.enabled:
            yield _NoSpan()
            return
        with self._mlflow.start_span(name=name, span_type=span_type) as span:
            if inputs is not None:
                span.set_inputs(inputs)
            if attributes:
                span.set_attributes({k: v for k, v in attributes.items() if v is not None})
            yield span

    @staticmethod
    def trace_id_of(span: Any) -> str | None:
        for attr in ("trace_id", "request_id"):
            value = getattr(span, attr, None)
            if value:
                return str(value)
        return None

    def flush(self) -> None:
        if self.enabled:
            with contextlib.suppress(Exception):
                self._mlflow.flush_trace_async_logging()

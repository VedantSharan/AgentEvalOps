"""
Trajectory recording helpers for AgentEvalOps.

Provides TrajectoryBuilder — a stateful accumulator that collects steps
during agent execution and produces a validated TrajectoryRecord at the end.

Token counts are never fabricated: this agent has no LLM, so prompt_tokens
and completion_tokens are left as None throughout.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

from code.shared.models import (
    ActionType,
    Outcome,
    TaskRecord,
    ToolResult,
    TrajectoryMetrics,
    TrajectoryRecord,
    TrajectoryStep,
)


class TrajectoryBuilder:
    """
    Accumulates trajectory steps and assembles the final TrajectoryRecord.

    Usage::

        builder = TrajectoryBuilder(task, config_hash, model_version, prompt_version)
        builder.add_tool_call("retrieve", {"query": "capital"}, result, latency)
        record = builder.finish(final_answer="Paris", outcome=Outcome.success)
    """

    def __init__(
        self,
        task: TaskRecord,
        configuration_hash: str,
        model_version: str,
        prompt_version: str,
        tracer: Any = None,
    ) -> None:
        self._task = task
        self._configuration_hash = configuration_hash
        self._model_version = model_version
        self._prompt_version = prompt_version
        self._run_id: str = str(uuid.uuid4())
        self._steps: list[TrajectoryStep] = []
        self._errors: list[str] = []
        self._wall_start: float = time.perf_counter()
        self.tracer = tracer

    # ------------------------------------------------------------------
    # Step accumulation
    # ------------------------------------------------------------------

    def add_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: ToolResult,
        latency_ms: float,
    ) -> None:
        """Record one tool-call step."""
        idx = len(self._steps)
        step = TrajectoryStep(
            step_index=idx,
            action_type=ActionType.tool_call,
            tool_name=tool_name,
            arguments=arguments,
            tool_result=result,
            llm_output=None,
            error=result.error if not result.success else None,
            latency_ms=max(0.0, latency_ms),
            prompt_tokens=None,
            completion_tokens=None,
        )
        self._steps.append(step)
        if not result.success and result.error:
            self._errors.append(f"Step {idx} [{tool_name}]: {result.error}")

    def add_submit(
        self,
        answer: str,
        result: ToolResult,
        latency_ms: float,
    ) -> None:
        """Record the final submit_answer step."""
        idx = len(self._steps)
        step = TrajectoryStep(
            step_index=idx,
            action_type=ActionType.submit_answer,
            tool_name="submit_answer",
            arguments={"answer": answer},
            tool_result=result,
            llm_output=None,
            error=result.error if not result.success else None,
            latency_ms=max(0.0, latency_ms),
            prompt_tokens=None,
            completion_tokens=None,
        )
        self._steps.append(step)

    def add_error(self, description: str) -> None:
        """Record a top-level error (not tied to a specific step)."""
        self._errors.append(description)

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def finish(
        self,
        final_answer: str | None,
        outcome: Outcome,
    ) -> TrajectoryRecord:
        """
        Assemble and return a validated TrajectoryRecord.

        Args:
            final_answer: Answer submitted by the agent (None if none submitted).
            outcome:      Terminal outcome (success / failure / error).

        Returns:
            A fully validated TrajectoryRecord.
        """
        total_latency = round((time.perf_counter() - self._wall_start) * 1000, 3)
        step_latencies = [
            s.latency_ms for s in self._steps if s.latency_ms is not None
        ]
        sum_latency: float | None = round(sum(step_latencies), 3) if step_latencies else None

        metrics = TrajectoryMetrics(
            total_steps=len(self._steps),
            total_latency_ms=sum_latency,
            total_prompt_tokens=None,
            total_completion_tokens=None,
        )

        return TrajectoryRecord(
            run_id=self._run_id,
            task_id=self._task.task_id,
            configuration_hash=self._configuration_hash,
            model_version=self._model_version,
            prompt_version=self._prompt_version,
            steps=self._steps,
            final_answer=final_answer,
            outcome=outcome,
            metrics=metrics,
            errors=self._errors,
        )

    @property
    def step_count(self) -> int:
        return len(self._steps)


# ---------------------------------------------------------------------------
# JSONL I/O for trajectory records
# ---------------------------------------------------------------------------

import pathlib


def export_trajectories_jsonl(
    trajectories: list[TrajectoryRecord],
    output_path: pathlib.Path,
    *,
    overwrite: bool = False,
) -> pathlib.Path:
    """
    Write trajectories to a JSONL file.

    Args:
        trajectories: List of TrajectoryRecord objects.
        output_path:  Destination path.
        overwrite:    If False, raise FileExistsError when path exists.

    Returns:
        Resolved output path.
    """
    output_path = pathlib.Path(output_path).resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass overwrite=True to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for traj in trajectories:
            fh.write(traj.model_dump_json() + "\n")
    return output_path


def load_trajectories_jsonl(path: pathlib.Path) -> list[TrajectoryRecord]:
    """
    Load TrajectoryRecord objects from a JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of validated TrajectoryRecord objects.
    """
    path = pathlib.Path(path).resolve()
    trajectories: list[TrajectoryRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                trajectories.append(TrajectoryRecord.model_validate_json(line))
    return trajectories

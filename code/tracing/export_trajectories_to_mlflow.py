"""
Trace Ingestion Adapter for AgentEvalOps.

Ingests authentic agent execution trajectories from JSONL into MLflow Tracing spans
using MLflowTracer, recording:
  - Root Agent Span: task_id, prompt, model_version, config hash, terminal outcome
  - Child Tool Spans: each tool call, arguments, outputs, real measured latency, tokens
  - Genuine execution metadata matching trajectories.jsonl to the exact millisecond.

Enables interactive trace exploration via `mlflow ui`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from code.shared.models import (
    ActionType,
    ExecutionConfig,
    Outcome,
    TaskRecord,
    ToolResult,
    TrajectoryRecord,
)
from code.tasks.revised_generator import load_tasks_jsonl
from code.tracing.mlflow_tracer import MLflowTracer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MLflowIngestion")


def ingest_trajectory(
    traj: TrajectoryRecord,
    task: TaskRecord,
    tracer: MLflowTracer,
) -> bool:
    """Ingest a single TrajectoryRecord into MLflow Tracing as root and child spans."""
    config = ExecutionConfig(
        model_id=traj.model_version,
        prompt_version=traj.prompt_version or "v1",
        tool_selection_strategy="greedy",
        max_steps=max(traj.metrics.total_steps or 1, 1),
        retry_budget=0,
    )

    with tracer.trace_task(task, config) as task_ctx:
        root_span = getattr(task_ctx, "span", None)
        if root_span is not None:
            root_span.set_attributes({
                "run_id": traj.run_id,
                "configuration_hash": traj.configuration_hash,
                "model_version": traj.model_version,
                "prompt_version": traj.prompt_version or "v1",
                "total_steps": traj.metrics.total_steps,
                "total_latency_ms": traj.metrics.total_latency_ms or 0.0,
                "total_prompt_tokens": traj.metrics.total_prompt_tokens or 0,
                "total_completion_tokens": traj.metrics.total_completion_tokens or 0,
            })

        for step in traj.steps:
            tool_name = step.tool_name or (
                "submit_answer" if step.action_type == ActionType.submit_answer else "unknown_tool"
            )
            args = step.arguments or {}

            with tracer.trace_tool(tool_name, args) as tool_ctx:
                if step.tool_result:
                    tool_ctx.set_result(step.tool_result)
                elif step.error:
                    synthetic_res = ToolResult(
                        tool_name=tool_name,
                        success=False,
                        output=None,
                        error=step.error,
                        latency_ms=step.latency_ms or 0.0,
                    )
                    tool_ctx.set_result(synthetic_res)

                tool_span = getattr(tool_ctx, "span", None)
                if tool_span is not None:
                    tool_span.set_attributes({
                        "step_index": step.step_index,
                        "action_type": step.action_type.value if hasattr(step.action_type, "value") else str(step.action_type),
                        "latency_ms": step.latency_ms or (step.tool_result.latency_ms if step.tool_result else 0.0),
                    })
                    if step.prompt_tokens is not None:
                        tool_span.set_attributes({"prompt_tokens": step.prompt_tokens})
                    if step.completion_tokens is not None:
                        tool_span.set_attributes({"completion_tokens": step.completion_tokens})

        # Set final terminal outcome using grade_v2
        from code.tasks.grader_v2 import grade_v2
        grading = grade_v2(task, traj.final_answer)
        task_ctx.set_outcome(
            final_answer=traj.final_answer,
            grading=grading,
            outcome=traj.outcome,
            errors=traj.errors,
        )

    return True


def ingest_trajectories_file(
    trajectories_path: Path,
    tasks_path: Path,
    tracking_uri: str | Path | None = None,
    experiment_name: str = "AgentEvalOps-MultiProvider",
    limit: int | None = None,
) -> int:
    """
    Ingest trajectories from JSONL into MLflow Tracing store.

    Returns the number of ingested trajectories.
    """
    if not trajectories_path.exists():
        raise FileNotFoundError(f"Trajectories file not found: {trajectories_path}")
    if not tasks_path.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_path}")

    tasks = load_tasks_jsonl(tasks_path)
    task_map = {t.task_id: t for t in tasks}

    tracer = MLflowTracer(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        enabled=True,
    )

    if not tracer._mlflow_available:
        logger.error("MLflow is not installed or available in this environment.")
        return 0

    count = 0
    with trajectories_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            traj = TrajectoryRecord.model_validate_json(line)
            task = task_map.get(traj.task_id)
            if not task:
                # Create a placeholder task record if not in benchmark suite
                task = TaskRecord(
                    task_id=traj.task_id,
                    template_family="unknown",
                    task_type="unknown",
                    prompt="[Original prompt not in revised suite]",
                    gold_answer="N/A",
                )

            ingest_trajectory(traj, task, tracer)
            count += 1
            if limit and count >= limit:
                break

    tracer.flush()
    logger.info(
        f"Successfully ingested {count} trajectories into MLflow experiment '{experiment_name}' at {tracer.tracking_uri}."
    )
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest JSONL agent trajectories into MLflow Tracing.")
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=Path("data/processed/trajectories_processed.jsonl"),
        help="Path to trajectories JSONL file (default: data/processed/trajectories_processed.jsonl)",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/processed/tasks_revised.jsonl"),
        help="Path to tasks JSONL file (default: data/processed/tasks_revised.jsonl)",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="sqlite:///mlruns/mlflow.db",
        help="MLflow tracking URI (default: sqlite:///mlruns/mlflow.db)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="AgentEvalOps-MultiProvider",
        help="MLflow experiment name (default: AgentEvalOps-MultiProvider)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of trajectories to ingest",
    )

    args = parser.parse_args()
    ingest_trajectories_file(
        trajectories_path=args.trajectories,
        tasks_path=args.tasks,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

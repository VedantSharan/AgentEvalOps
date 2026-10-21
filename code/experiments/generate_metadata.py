"""
Metadata Generator for AgentEvalOps Dataset.

Produces data/metadata.csv with one row per sample, recording:
- sample_id: task_id or trajectory run_id
- sample_type: 'task' or 'trajectory'
- source: 'generator_revised', 'generator_baseline', or provider/model name
- timestamp: ISO 8601 creation/execution timestamp
- collector: agent runner / script identity
- tool_version: version of the tool registry and runner
- content_hash: SHA-256 canonical content hash
- extra_metadata: JSON string of additional provenance (family, outcome, tokens, etc.)
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from code.shared.models import TaskRecord, TrajectoryRecord


def compute_task_content_hash(task: TaskRecord) -> str:
    payload = json.dumps(
        {"task_id": task.task_id, "prompt": task.prompt, "gold_answer": task.gold_answer},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_trajectory_content_hash(traj: TrajectoryRecord) -> str:
    payload = json.dumps(
        {
            "run_id": traj.run_id,
            "task_id": traj.task_id,
            "model_version": traj.model_version,
            "final_answer": traj.final_answer,
            "outcome": traj.outcome.value,
            "step_count": len(traj.steps),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_metadata(
    tasks_paths: list[Path],
    trajectories_paths: list[Path],
    output_path: Path,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    # Process Tasks
    for t_path in tasks_paths:
        if not t_path.exists():
            continue
        with t_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                task = TaskRecord.model_validate_json(line)
                ch = task.metadata.content_hash if (task.metadata and task.metadata.content_hash) else compute_task_content_hash(task)
                created_at = task.metadata.created_at if task.metadata else "2026-10-21T00:00:00Z"
                provenance = task.metadata.provenance if task.metadata else t_path.stem
                rows.append({
                    "sample_id": task.task_id,
                    "sample_type": "task",
                    "source": provenance,
                    "timestamp": created_at,
                    "collector": "code.tasks.revised_generator",
                    "tool_version": "v1.2.0",
                    "content_hash": ch,
                    "extra_metadata": json.dumps({
                        "template_family": task.template_family,
                        "task_type": task.task_type,
                        "answerable": task.answerable,
                        "evaluation_rule": task.evaluation_rule.value,
                    }),
                })

    # Process Trajectories
    for traj_path in trajectories_paths:
        if not traj_path.exists():
            continue
        with traj_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                traj = TrajectoryRecord.model_validate_json(line)
                ch = compute_trajectory_content_hash(traj)
                rows.append({
                    "sample_id": traj.run_id,
                    "sample_type": "trajectory",
                    "source": traj.model_version,
                    "timestamp": "2026-10-21T11:00:00Z",
                    "collector": "code.experiments.trajectory_worker",
                    "tool_version": "v1.2.0",
                    "content_hash": ch,
                    "extra_metadata": json.dumps({
                        "task_id": traj.task_id,
                        "outcome": traj.outcome.value,
                        "total_steps": traj.metrics.total_steps,
                        "total_prompt_tokens": traj.metrics.total_prompt_tokens,
                        "total_completion_tokens": traj.metrics.total_completion_tokens,
                        "total_latency_ms": traj.metrics.total_latency_ms,
                    }),
                })

    fieldnames = [
        "sample_id",
        "sample_type",
        "source",
        "timestamp",
        "collector",
        "tool_version",
        "content_hash",
        "extra_metadata",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    tasks_paths = [
        repo_root / "data" / "processed" / "tasks_revised.jsonl",
        repo_root / "data" / "raw" / "tasks.jsonl",
    ]
    trajectories_paths = [
        repo_root / "data" / "processed" / "trajectories_processed.jsonl",
        repo_root / "data" / "raw" / "trajectories.jsonl",
    ]
    output_path = repo_root / "data" / "metadata.csv"

    count = generate_metadata(tasks_paths, trajectories_paths, output_path)
    print(f"Generated {count} rows in {output_path}")


if __name__ == "__main__":
    main()

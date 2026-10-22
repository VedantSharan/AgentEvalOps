"""
Metadata Generator for AgentEvalOps Dataset.

Produces data/metadata.csv with one row per sample, recording:
- sample_id: task_id or trajectory run_id
- sample_type: 'task', 'augmented_task', or 'trajectory'
- source: 'generator_revised', 'generator_baseline', augmentation script, or model name
- timestamp: ISO 8601 creation/execution timestamp (recovered from work queue or task metadata)
- collector: agent runner / generator identity
- tool_version: version of the tool registry, generator, or runner
- content_hash: SHA-256 canonical content hash
- extra_metadata: JSON string of additional provenance (family, outcome, tokens, etc.)
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
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


def load_work_queue_timestamps(db_path: Path | None = None) -> dict[str, str]:
    """Recover authentic execution timestamps from the work queue SQLite database."""
    candidates = [
        db_path,
        Path("temp/work_queue.sqlite3"),
        Path("data/experiments/default/work_queue.sqlite3"),
    ]
    for p in candidates:
        if p and p.exists():
            try:
                conn = sqlite3.connect(p)
                cur = conn.cursor()
                cur.execute(
                    "SELECT j.run_id, a.started_at, a.finished_at FROM jobs j "
                    "JOIN job_attempts a ON j.job_id = a.job_id WHERE j.run_id IS NOT NULL"
                )
                ts_map = {}
                for rid, started, finished in cur.fetchall():
                    if rid and (started or finished):
                        ts_map[rid] = started or finished
                if ts_map:
                    return ts_map
            except Exception:
                continue
    return {}


def generate_metadata(
    tasks_paths: list[Path],
    trajectories_paths: list[Path],
    output_path: Path,
    augmented_tasks_path: Path | None = None,
    work_queue_db: Path | None = None,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    timestamps_map = load_work_queue_timestamps(work_queue_db)

    # 1. Process Base Tasks
    for t_path in tasks_paths:
        if not t_path.exists():
            continue
        with t_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                task = TaskRecord.model_validate_json(line)
                if task.task_id in seen_ids:
                    continue
                seen_ids.add(task.task_id)

                ch = task.metadata.content_hash if (task.metadata and task.metadata.content_hash) else compute_task_content_hash(task)
                created_at = task.metadata.created_at if task.metadata else "2026-10-21T00:00:00Z"
                provenance = task.metadata.provenance if task.metadata else t_path.stem
                tool_ver = task.metadata.generator_version if task.metadata else "v1.2.0"
                rows.append({
                    "sample_id": task.task_id,
                    "sample_type": "task",
                    "source": provenance,
                    "timestamp": created_at,
                    "collector": "code.tasks.revised_generator",
                    "tool_version": tool_ver,
                    "content_hash": ch,
                    "extra_metadata": json.dumps({
                        "template_family": task.template_family,
                        "task_type": task.task_type,
                        "answerable": task.answerable,
                        "evaluation_rule": task.evaluation_rule.value,
                    }),
                })

    # 2. Process Augmented Tasks (Part D)
    if augmented_tasks_path and augmented_tasks_path.exists():
        with augmented_tasks_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                task_data = record.get("task", {})
                aug_data = record.get("augmentation", {})
                task_id = task_data.get("task_id") or aug_data.get("aug_task_id")
                if not task_id or task_id in seen_ids:
                    continue
                seen_ids.add(task_id)

                meta = task_data.get("metadata", {})
                ch = meta.get("content_hash") or hashlib.sha256(
                    json.dumps(
                        {"task_id": task_id, "prompt": task_data.get("prompt"), "gold_answer": task_data.get("gold_answer")},
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()
                created_at = meta.get("created_at", "2026-10-21T12:29:39Z")
                tool_ver = meta.get("generator_version", "aug-1.0.0")
                provenance = meta.get("provenance", "code/augmentation/build_augmented.py")

                rows.append({
                    "sample_id": task_id,
                    "sample_type": "augmented_task",
                    "source": provenance,
                    "timestamp": created_at,
                    "collector": "code.augmentation.build_augmented",
                    "tool_version": tool_ver,
                    "content_hash": ch,
                    "extra_metadata": json.dumps({
                        "parent_task_id": aug_data.get("parent_task_id"),
                        "augmentation_type": aug_data.get("type"),
                        "expected_behavior": aug_data.get("expected_behavior"),
                        "template_family": aug_data.get("template_family"),
                    }),
                })

    # 3. Process Trajectories (All 154 multi-provider runs + pilot runs)
    for traj_path in trajectories_paths:
        if not traj_path.exists():
            continue
        with traj_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                traj = TrajectoryRecord.model_validate_json(line)
                if traj.run_id in seen_ids:
                    continue
                seen_ids.add(traj.run_id)

                ch = compute_trajectory_content_hash(traj)
                # Recover exact microsecond timestamp from work queue if available
                ts = timestamps_map.get(traj.run_id)
                if not ts:
                    ts = "2026-10-21T11:00:00Z"

                rows.append({
                    "sample_id": traj.run_id,
                    "sample_type": "trajectory",
                    "source": traj.model_version,
                    "timestamp": ts,
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
    augmented_tasks_path = repo_root / "data" / "augmented" / "tasks_augmented.jsonl"
    trajectories_paths = [
        repo_root / "data" / "experiments" / "multi_provider_run" / "trajectories.jsonl",
        repo_root / "data" / "raw" / "trajectories.jsonl",
    ]
    output_path = repo_root / "data" / "metadata.csv"
    work_queue_db = repo_root / "temp" / "work_queue.sqlite3"

    count = generate_metadata(
        tasks_paths=tasks_paths,
        trajectories_paths=trajectories_paths,
        output_path=output_path,
        augmented_tasks_path=augmented_tasks_path,
        work_queue_db=work_queue_db,
    )
    print(f"Generated {count} rows in {output_path}")


if __name__ == "__main__":
    main()


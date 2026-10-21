"""
Adversarial task suite generator for AgentEvalOps.

Generates 30 targeted task instances across 6 failure modes:
  1. tool_skipping (5 tasks)
  2. ignoring_tool_results (5 tasks)
  3. output_fabrication (5 tasks)
  4. unnecessary_tool_use (5 tasks)
  5. multistep_tool_dependency (5 tasks)
  6. tool_error_recovery (5 tasks)

Outputs: data/adversarial/tasks.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

from code.shared.models import EvaluationRule, TaskMetadata, TaskRecord
from code.tasks.adversarial_fixtures import ADVERSARIAL_TASK_DEFINITIONS

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = _REPO_ROOT / "data" / "adversarial" / "tasks.jsonl"
GENERATOR_VERSION = "0.2.0-adversarial"
ADVERSARIAL_SEED = 42


def compute_content_hash(task_id: str, prompt: str, gold_answer: str | None) -> str:
    """Compute the SHA-256 canonical hash of the task content."""
    payload = json.dumps(
        {"task_id": task_id, "prompt": prompt, "gold_answer": gold_answer},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_adversarial_tasks(seed: int = ADVERSARIAL_SEED) -> list[TaskRecord]:
    """Generate and return all 30 adversarial TaskRecords with validated schemas."""
    tasks: list[TaskRecord] = []

    for item in ADVERSARIAL_TASK_DEFINITIONS:
        task_id = item["task_id"]
        prompt = item["prompt"]
        gold_answer = item["gold_answer"]
        eval_rule = EvaluationRule(item["evaluation_rule"])

        metadata = TaskMetadata(
            provenance="code/tasks/adversarial_generator.py",
            created_at=datetime.now(timezone.utc).isoformat(),
            generator_version=GENERATOR_VERSION,
            seed=seed,
            content_hash=compute_content_hash(task_id, prompt, gold_answer),
        )

        task = TaskRecord(
            task_id=task_id,
            template_family=item["template_family"],
            task_type=item["task_type"],
            prompt=prompt,
            answerable=item["answerable"],
            gold_answer=gold_answer,
            expected_tools=item["expected_tools"],
            evaluation_rule=eval_rule,
            metadata=metadata,
        )
        tasks.append(task)

    assert len(tasks) == 30, f"Expected 30 tasks, got {len(tasks)}"
    return tasks


def export_tasks_jsonl(
    tasks: list[TaskRecord],
    output_path: pathlib.Path = DEFAULT_OUTPUT_PATH,
    *,
    overwrite: bool = True,
) -> pathlib.Path:
    """Export tasks to JSONL file."""
    output_path = pathlib.Path(output_path).resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass overwrite=True to replace.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(t.model_dump_json() + "\n")
    return output_path


def load_tasks_jsonl(path: pathlib.Path) -> list[TaskRecord]:
    """Load TaskRecord instances from JSONL."""
    path = pathlib.Path(path).resolve()
    tasks: list[TaskRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                tasks.append(TaskRecord.model_validate_json(line))
    return tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 30 adversarial benchmark tasks.")
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=ADVERSARIAL_SEED)
    args = parser.parse_args()

    tasks = generate_adversarial_tasks(seed=args.seed)
    out = export_tasks_jsonl(tasks, args.output, overwrite=True)
    print(f"Generated and wrote {len(tasks)} adversarial tasks to {out}")
    sys.exit(0)

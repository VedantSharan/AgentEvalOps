"""
Deterministic benchmark generator for AgentEvalOps.

Generates exactly 24 TaskRecord objects (4 per family × 6 families) using
only locally defined fixtures.  No internet access, external APIs, or LLM
calls are required.

Task families:
  1. retrieval       — information retrieval from a local fact store
  2. multistep_qa    — multi-step question answering combining multiple facts
  3. calculation     — numerical calculation from a symbolic expression
  4. structured_data — queries over a small in-memory employee table
  5. file_ops        — questions about controlled virtual text files
  6. unanswerable    — tasks without sufficient information to answer

Usage
-----
Generate and return tasks in memory::

    from code.tasks.generator import generate_benchmark
    tasks = generate_benchmark(seed=42)

Generate and write JSONL to the default output path::

    python -m code.tasks.generator

Output path: data/raw/tasks.jsonl  (relative to repository root)
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

from code.shared.models import EvaluationRule, TaskMetadata, TaskRecord
from code.tasks.fixtures import (
    CALC_PROBLEMS,
    EMPLOYEE_RECORDS,
    FACTS,
    FILE_FIXTURES,
    GENERATOR_VERSION,
    MULTISTEP_QUESTIONS,
    STRUCTURED_QUERIES,
    UNANSWERABLE_SIGNAL,
    UNANSWERABLE_TASKS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARK_SEED: int = 42
TASKS_PER_FAMILY: int = 4
TOTAL_TASKS: int = 6 * TASKS_PER_FAMILY  # 24

# Path is relative to the repository root (two levels up from this file).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH: pathlib.Path = _REPO_ROOT / "data" / "raw" / "tasks.jsonl"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _content_hash(task_id: str, prompt: str, gold_answer: str | None) -> str:
    """Return a SHA-256 digest of the canonical task content."""
    payload = json.dumps(
        {"task_id": task_id, "prompt": prompt, "gold_answer": gold_answer},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_metadata(seed: int, task_id: str, prompt: str, gold_answer: str | None) -> TaskMetadata:
    return TaskMetadata(
        provenance="code/tasks/generator.py",
        created_at=datetime.now(timezone.utc).isoformat(),
        generator_version=GENERATOR_VERSION,
        seed=seed,
        content_hash=_content_hash(task_id, prompt, gold_answer),
    )


# ---------------------------------------------------------------------------
# Per-family generators
# ---------------------------------------------------------------------------


def _build_retrieval_tasks(seed: int) -> list[TaskRecord]:
    """Family 1 — retrieve a single fact from the local knowledge base."""
    tasks: list[TaskRecord] = []
    for fact in FACTS:
        task_id = f"retrieval-{fact['fact_id']}"
        prompt = (
            f"{fact['question']} "
            f"Use the retrieve tool to look up the answer in the local fact store."
        )
        gold = fact["answer"]
        tasks.append(
            TaskRecord(
                task_id=task_id,
                template_family="retrieval",
                task_type="information_retrieval",
                prompt=prompt,
                answerable=True,
                gold_answer=gold,
                expected_tools=["retrieve"],
                evaluation_rule=EvaluationRule.exact_match,
                metadata=_make_metadata(seed, task_id, prompt, gold),
            )
        )
    return tasks


def _build_multistep_tasks(seed: int) -> list[TaskRecord]:
    """Family 2 — multi-step QA requiring combinations of facts."""
    tasks: list[TaskRecord] = []
    for ms in MULTISTEP_QUESTIONS:
        task_id = f"multistep-{ms['ms_id']}"
        prompt = ms["question"]
        gold = ms["answer"]
        tasks.append(
            TaskRecord(
                task_id=task_id,
                template_family="multistep_qa",
                task_type="multi_step_question_answering",
                prompt=prompt,
                answerable=True,
                gold_answer=gold,
                expected_tools=["retrieve"],
                evaluation_rule=EvaluationRule.structured_match,
                metadata=_make_metadata(seed, task_id, prompt, gold),
            )
        )
    return tasks


def _build_calculation_tasks(seed: int) -> list[TaskRecord]:
    """Family 3 — numerical calculations with documented expressions."""
    tasks: list[TaskRecord] = []
    for calc in CALC_PROBLEMS:
        task_id = f"calculation-{calc['calc_id']}"
        prompt = (
            f"{calc['description']} "
            f"Use the calculate tool and return the numeric result only."
        )
        gold = calc["answer"]
        tasks.append(
            TaskRecord(
                task_id=task_id,
                template_family="calculation",
                task_type="numerical_calculation",
                prompt=prompt,
                answerable=True,
                gold_answer=gold,
                expected_tools=["calculate"],
                evaluation_rule=EvaluationRule.numeric_tolerance,
                metadata=_make_metadata(seed, task_id, prompt, gold),
            )
        )
    return tasks


def _build_structured_data_tasks(seed: int) -> list[TaskRecord]:
    """Family 4 — queries over the employee records table."""
    tasks: list[TaskRecord] = []
    for sq in STRUCTURED_QUERIES:
        task_id = f"structured-{sq['sq_id']}"
        prompt = (
            f"{sq['question']} "
            f"Use the query_records tool to query the employee dataset."
        )
        gold = sq["answer"]
        # Answers that are pure integers use exact_match; comma-separated
        # lists use structured_match.
        if "," in gold:
            rule = EvaluationRule.structured_match
        else:
            rule = EvaluationRule.exact_match
        tasks.append(
            TaskRecord(
                task_id=task_id,
                template_family="structured_data",
                task_type="structured_data_operation",
                prompt=prompt,
                answerable=True,
                gold_answer=gold,
                expected_tools=["query_records"],
                evaluation_rule=rule,
                metadata=_make_metadata(seed, task_id, prompt, gold),
            )
        )
    return tasks


def _build_file_ops_tasks(seed: int) -> list[TaskRecord]:
    """Family 5 — read and interpret controlled virtual text files."""
    tasks: list[TaskRecord] = []
    for ff in FILE_FIXTURES:
        task_id = f"fileops-{ff['file_id']}"
        prompt = (
            f"{ff['question']} "
            f"Use the read_file tool to access {ff['filename']}."
        )
        gold = ff["answer"]
        # Line count and sum are numbers → numeric_tolerance;
        # last-line and key-value lookups are exact strings → exact_match.
        try:
            float(gold)
            rule = EvaluationRule.numeric_tolerance
        except ValueError:
            rule = EvaluationRule.exact_match
        tasks.append(
            TaskRecord(
                task_id=task_id,
                template_family="file_ops",
                task_type="file_operation",
                prompt=prompt,
                answerable=True,
                gold_answer=gold,
                expected_tools=["read_file"],
                evaluation_rule=rule,
                metadata=_make_metadata(seed, task_id, prompt, gold),
            )
        )
    return tasks


def _build_unanswerable_tasks(seed: int) -> list[TaskRecord]:
    """Family 6 — tasks the agent must refuse to answer."""
    tasks: list[TaskRecord] = []
    for un in UNANSWERABLE_TASKS:
        task_id = f"unanswerable-{un['un_id']}"
        prompt = (
            f"{un['question']} "
            f"If you cannot answer, respond with exactly the word '{UNANSWERABLE_SIGNAL}'."
        )
        tasks.append(
            TaskRecord(
                task_id=task_id,
                template_family="unanswerable",
                task_type="unanswerable",
                prompt=prompt,
                answerable=False,
                gold_answer=None,
                expected_tools=[],
                evaluation_rule=EvaluationRule.unanswerable,
                metadata=_make_metadata(seed, task_id, prompt, None),
            )
        )
    return tasks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_benchmark(seed: int = BENCHMARK_SEED) -> list[TaskRecord]:
    """
    Generate and return all 24 benchmark tasks.

    The output is fully deterministic for a given *seed* value.  The seed
    is recorded in each task's metadata but does not currently affect the
    generated content (the fixture data is static).  It is preserved for
    future use if randomisation is added within a family.

    Args:
        seed: Integer seed recorded in task metadata (default 42).

    Returns:
        A list of 24 validated TaskRecord objects in family order.
    """
    tasks: list[TaskRecord] = []
    tasks.extend(_build_retrieval_tasks(seed))
    tasks.extend(_build_multistep_tasks(seed))
    tasks.extend(_build_calculation_tasks(seed))
    tasks.extend(_build_structured_data_tasks(seed))
    tasks.extend(_build_file_ops_tasks(seed))
    tasks.extend(_build_unanswerable_tasks(seed))

    assert len(tasks) == TOTAL_TASKS, (
        f"Expected {TOTAL_TASKS} tasks, got {len(tasks)}. "
        "Update TOTAL_TASKS or fixture lengths."
    )
    return tasks


def export_jsonl(
    tasks: list[TaskRecord],
    output_path: pathlib.Path = DEFAULT_OUTPUT_PATH,
    *,
    overwrite: bool = False,
) -> pathlib.Path:
    """
    Write *tasks* to *output_path* as newline-delimited JSON (JSONL).

    Each line is the JSON serialisation of one TaskRecord.
    The file is created (including parent directories) if it does not exist.
    If it exists and *overwrite* is False, a FileExistsError is raised to
    prevent silent data loss.

    Args:
        tasks:       List of TaskRecord objects to write.
        output_path: Destination file path (default: data/raw/tasks.jsonl).
        overwrite:   Set True to replace an existing file.

    Returns:
        The resolved output path.
    """
    output_path = pathlib.Path(output_path).resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass overwrite=True to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for task in tasks:
            fh.write(task.model_dump_json() + "\n")
    return output_path


def load_jsonl(path: pathlib.Path) -> list[TaskRecord]:
    """
    Load TaskRecord objects from a JSONL file produced by export_jsonl().

    Args:
        path: Path to the JSONL file.

    Returns:
        List of validated TaskRecord objects.
    """
    path = pathlib.Path(path).resolve()
    tasks: list[TaskRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                tasks.append(TaskRecord.model_validate_json(line))
    return tasks


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate the AgentEvalOps benchmark task suite."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=BENCHMARK_SEED,
        help=f"Random seed recorded in task metadata (default: {BENCHMARK_SEED}).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    args = parser.parse_args()

    tasks = generate_benchmark(seed=args.seed)
    out = export_jsonl(tasks, output_path=args.output, overwrite=args.overwrite)
    print(f"Wrote {len(tasks)} tasks to {out}")
    sys.exit(0)

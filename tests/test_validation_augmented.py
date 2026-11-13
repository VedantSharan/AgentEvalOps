"""Tests for the A2 extensions of the validation suite (fault-injected tasks, CI sample mode)."""

from __future__ import annotations

import json
from pathlib import Path

from code.validation.benchmark_validator import BenchmarkValidator
from code.validation.run_validation_evidence import augmented_tasks_and_environments, main, sample_lines

TASK = {
    "task_id": "T1_multihop_01",
    "template_family": "multi_hop_retrieval",
    "task_type": "x",
    "prompt": "What is the per diem?",
    "expected_tools": ["retrieve"],
    "gold_answer": "240 USD",
    "evaluation_rule": "exact_match",
    "answerable": True,
}


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _duplicate_check(tmp_path: Path, environments: dict[str, str] | None) -> bool:
    tasks = _write(tmp_path / "t.jsonl", [dict(TASK, task_id="a"), dict(TASK, task_id="b")])
    empty = _write(tmp_path / "e.jsonl", [])
    res = BenchmarkValidator(tasks, empty, environments=environments).run_all_checks()
    assert res.tasks_count == 2, "both tasks must pass the schema check"
    return res.summary_by_check["6_duplicate_tasks"]


def test_same_prompt_same_environment_is_a_duplicate(tmp_path: Path) -> None:
    assert _duplicate_check(tmp_path, None) is False


def test_same_prompt_different_fault_is_not_a_duplicate(tmp_path: Path) -> None:
    assert _duplicate_check(tmp_path, {"a": '[{"fault_type": "tool_failure"}]', "b": '[{"fault_type": "format_change"}]'})


def test_environment_map_covers_fault_records_only(tmp_path: Path) -> None:
    records = _write(tmp_path / "aug.jsonl", [
        {"augmentation": {"type": "paraphrase"}, "task": dict(TASK, task_id="p")},
        {"augmentation": {"type": "fault_injection", "fault": [{"tool": "retrieve"}]}, "task": dict(TASK, task_id="f")},
    ])
    out = tmp_path / "tasks.jsonl"
    env = augmented_tasks_and_environments(records, out)
    assert set(env) == {"f"} and len(out.read_text(encoding="utf-8").splitlines()) == 2


def test_sample_is_seeded_and_bounded(tmp_path: Path) -> None:
    src = _write(tmp_path / "src.jsonl", [{"i": i} for i in range(50)])
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    sample_lines(src, a, 10, seed=7)
    sample_lines(src, b, 10, seed=7)
    assert a.read_bytes() == b.read_bytes() and len(a.read_text().splitlines()) == 10


def test_ci_sample_mode_gives_expected_results(tmp_path: Path) -> None:
    main(["--sample", "15", "--out", str(tmp_path)])
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert all(v["as_expected"] for v in summary.values())
    assert summary["augmented_all"]["passed"] and not summary["corrupted"]["passed"]

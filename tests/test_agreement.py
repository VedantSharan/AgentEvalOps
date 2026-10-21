"""Tests for code/annotation/agreement.py (Part C)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from code.annotation.agreement import ExportError, run

CONFIG = Path("code/annotation/label_studio_config.xml")


def _task(run_id: str, outcome: str, category: str, step: int, annotator: int = 1, cancelled: bool = False) -> dict:
    return {
        "data": {"run_id": run_id, "task_id": f"T_{run_id}", "outcome": "failure", "gold_answer": "1", "final_answer": "1"},
        "annotations": [{
            "completed_by": annotator,
            "was_cancelled": cancelled,
            "created_at": "2026-10-21T00:00:00Z",
            "lead_time": 10.0,
            "result": [
                {"from_name": "outcome", "value": {"choices": [outcome]}},
                {"from_name": "primary_failure_category", "value": {"choices": [category]}},
                {"from_name": "first_error_step", "value": {"number": step}},
            ],
        }],
    }


def _write(path: Path, tasks: list[dict]) -> Path:
    path.write_text(json.dumps(tasks), encoding="utf-8")
    return path


def test_matches_by_run_id_not_row_order(tmp_path: Path) -> None:
    a = [_task("r1", "success", "no_failure", 0), _task("r2", "failure", "wrong_tool", 1)]
    b = [_task("r2", "failure", "wrong_tool", 1, 2), _task("r1", "success", "no_failure", 0, 2)]  # reversed order
    s = run(_write(tmp_path / "a.json", a), _write(tmp_path / "b.json", b), "A1", "A2", "t", tmp_path / "out", CONFIG)
    assert s["n_overlap"] == 2
    assert s["fields"]["outcome"]["n_agree"] == 2
    assert s["fields"]["outcome"]["cohen_kappa"] == 1.0


def test_known_kappa_value(tmp_path: Path) -> None:
    # 4 items: A = S,S,F,F ; B = S,F,F,F -> Po=0.75, Pe=0.5*0.25+0.5*0.75=0.5, kappa=0.5
    a = [_task("r1", "success", "no_failure", 0), _task("r2", "success", "no_failure", 0),
         _task("r3", "failure", "wrong_tool", 1), _task("r4", "failure", "wrong_tool", 1)]
    b = [_task("r1", "success", "no_failure", 0, 2), _task("r2", "failure", "wrong_tool", 1, 2),
         _task("r3", "failure", "wrong_tool", 1, 2), _task("r4", "failure", "wrong_tool", 1, 2)]
    s = run(_write(tmp_path / "a.json", a), _write(tmp_path / "b.json", b), "A1", "A2", "t", tmp_path / "out", CONFIG)
    assert s["fields"]["outcome"]["cohen_kappa"] == pytest.approx(0.5)
    assert (tmp_path / "out" / "disagreements.csv").read_text().count("\n") == 2  # header + 1 row


def test_invalid_category_rejected(tmp_path: Path) -> None:
    a = [_task("r1", "failure", "not_a_category", 1)]
    with pytest.raises(ExportError, match="invalid primary_failure_category"):
        run(_write(tmp_path / "a.json", a), _write(tmp_path / "b.json", a), "A1", "A2", "t", tmp_path / "o", CONFIG)


def test_missing_field_rejected(tmp_path: Path) -> None:
    t = _task("r1", "success", "no_failure", 0)
    t["annotations"][0]["result"].pop()  # drop first_error_step
    with pytest.raises(ExportError, match="missing 'first_error_step'"):
        run(_write(tmp_path / "a.json", [t]), _write(tmp_path / "b.json", [t]), "A1", "A2", "t", tmp_path / "o", CONFIG)


def test_cancelled_only_annotation_rejected(tmp_path: Path) -> None:
    a = [_task("r1", "success", "no_failure", 0, cancelled=True)]
    with pytest.raises(ExportError, match="0 non-cancelled"):
        run(_write(tmp_path / "a.json", a), _write(tmp_path / "b.json", a), "A1", "A2", "t", tmp_path / "o", CONFIG)


def test_no_overlap_rejected(tmp_path: Path) -> None:
    a = [_task("r1", "success", "no_failure", 0)]
    b = [_task("r9", "success", "no_failure", 0, 2)]
    with pytest.raises(ExportError, match="share no run_id"):
        run(_write(tmp_path / "a.json", a), _write(tmp_path / "b.json", b), "A1", "A2", "t", tmp_path / "o", CONFIG)

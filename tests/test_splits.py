"""Tests for code/splits/make_splits.py: the checks pass on the real build and FAIL on leaky manifests."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from code.splits import make_splits as ms


@pytest.fixture()
def built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(ms, "OUT", tmp_path / "splits")
    monkeypatch.setattr(ms, "REPORT", tmp_path / "report.json")
    ms.write(ms.build())
    return tmp_path / "splits"


def _status(report: dict) -> dict[str, bool]:
    return {c["check"]: c["passed"] for c in report["checks"]}


def _edit(path: Path, fn) -> None:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows = fn(rows)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_real_build_passes_all_checks(built: Path) -> None:
    assert ms.check()["all_passed"]


def test_moving_a_trajectory_to_another_split_is_caught(built: Path) -> None:
    moved = {}
    def take(rows):
        i = next(i for i, r in enumerate(rows) if r["item_type"] == "trajectory")
        moved["row"] = rows.pop(i)
        return rows
    _edit(built / "test.csv", take)
    _edit(built / "train.csv", lambda rows: rows + [moved["row"]])
    s = _status(ms.check())
    assert not s["1_template_families_disjoint"] and not s["3_trajectories_and_repeated_runs_follow_their_task"]


def test_copying_a_test_task_into_train_is_caught(built: Path) -> None:
    test_task = next(r for r in csv.DictReader((built / "test.csv").open()) if r["item_type"] == "task")
    _edit(built / "train.csv", lambda rows: rows + [test_task])
    s = _status(ms.check())
    assert not s["2_no_item_in_two_splits"] and not s["6_no_identical_prompt_across_train_val_test"]


def test_fault_task_outside_shifted_is_caught(built: Path) -> None:
    fault = next(iter(csv.DictReader((built / "shifted.csv").open())))
    _edit(built / "validation.csv", lambda rows: rows + [fault])
    assert not _status(ms.check())["5_shifted_split_is_exactly_the_fault_injected_tasks"]


def test_tampered_gold_set_is_caught(built: Path) -> None:
    gold = built / "gold_test" / "gold_labels.csv"
    gold.write_text(gold.read_text().replace("success", "failure", 1))
    assert not _status(ms.check())["10_gold_test_checksums_match"]


def test_changed_record_hash_is_caught(built: Path) -> None:
    def corrupt(rows):
        rows[0]["record_sha256"] = "0" * 64
        return rows
    _edit(built / "train.csv", corrupt)
    assert not _status(ms.check())["7_every_manifest_item_exists_with_matching_hash"]

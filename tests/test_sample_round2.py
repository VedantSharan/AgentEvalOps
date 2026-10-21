"""Tests for code/annotation/sample_round2.py on the real repository data."""

from __future__ import annotations

import json
from pathlib import Path

from code.annotation.sample_round2 import DEFAULT_TARGETS, SEED, build

ARGS = dict(
    trajectories=Path("data/experiments/multi_provider_run/trajectories.jsonl"),
    tasks_path=Path("data/processed/tasks_revised.jsonl"),
    round1=Path("data/labels/stratified_sample_45.json"),
    regrade=Path("data/processed/regrade_v2.csv"),
    targets=DEFAULT_TARGETS,
    seed=SEED,
)


def test_sample_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    m1 = build(**ARGS, out_json=tmp_path / "a.json", out_manifest=tmp_path / "a.csv")
    m2 = build(**ARGS, out_json=tmp_path / "b.json", out_manifest=tmp_path / "b.csv")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    ids = [r["run_id"] for r in m1]
    assert len(ids) == len(set(ids)) == sum(DEFAULT_TARGETS.values())
    round1 = {s["data"]["run_id"] for s in json.loads(ARGS["round1"].read_text())}
    assert not round1 & set(ids)
    assert [r["run_id"] for r in m2] == ids


def test_grader_outcome_hidden_from_annotators(tmp_path: Path) -> None:
    build(**ARGS, out_json=tmp_path / "a.json", out_manifest=tmp_path / "a.csv")
    for item in json.loads((tmp_path / "a.json").read_text()):
        assert "outcome" not in item["data"]

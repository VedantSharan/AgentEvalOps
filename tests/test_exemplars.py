"""Tests for code/agent/exemplars.py: banks come only from train-split data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from code.agent.config import CONFIGS_DIR, exemplars, load_config
from code.agent.exemplars import TRAIN_SPLIT, build
from code.agent.loop import _exemplar_block


def _train_ids() -> set[str]:
    with TRAIN_SPLIT.open(encoding="utf-8", newline="") as fh:
        return {r["item_id"] for r in csv.DictReader(fh)}


def test_banks_use_train_split_data_only(tmp_path: Path) -> None:
    paths = build(tmp_path)
    train = _train_ids()
    for path in paths.values():
        for entry in json.loads(path.read_text(encoding="utf-8")):
            assert entry["source_run_id"] in train
            if "augmentation" in entry:
                assert entry["augmentation"] in train


def test_aug_al_bank_extends_the_base_bank(tmp_path: Path) -> None:
    paths = build(tmp_path)
    base = json.loads(paths["train_base"].read_text(encoding="utf-8"))
    full = json.loads(paths["train_aug_al"].read_text(encoding="utf-8"))
    assert full[: len(base)] == base
    assert {e["type"] for e in full} == {"example", "pitfall"}
    assert len({e["family"] for e in base}) == 3


def test_committed_banks_match_a_rebuild(tmp_path: Path) -> None:
    paths = build(tmp_path)
    for name, path in paths.items():
        committed = Path("prompts/exemplars") / f"{name}.json"
        assert committed.read_bytes() == path.read_bytes(), f"rebuild {name} with python -m code.agent.exemplars"


def test_exemplar_block_renders_examples_and_pitfalls() -> None:
    cfg = load_config(CONFIGS_DIR / "variants" / "proposed_train_aug_al.yaml")
    text = _exemplar_block(exemplars(cfg))
    assert "Example 1. Task:" in text and "Mistakes seen in earlier runs" in text
    assert _exemplar_block([]) == ""

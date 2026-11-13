"""Evaluation items for every A2 split, read from the frozen A1 split manifests.

An ``EvalItem`` is one task to run: the ``TaskRecord``, any fault specifications
to inject while it runs (shifted split), and the slice labels used in Part C
(template family, unanswerable, augmentation type, fault type, parent split).

Splits
------
train, validation, test   base tasks + label-preserving variants (entity variants,
                          paraphrases) from data/splits/<split>.csv
test_base                 the 10 base test tasks only
test_augmented            the 15 augmented test tasks (A1 Part D robustness set)
gold                      data/splits/gold_test/gold_tasks.jsonl, after checksum
                          verification (code/evaluation/gold.py)
shifted                   the 96 fault-injected tasks; ``shifted_<parent>`` keeps only
                          those whose parent family is in split <parent>
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from code.augmentation.fault_injection import FaultSpec
from code.shared.models import EvaluationRule, TaskRecord

REPO = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO / "data" / "splits"
BASE_TASKS = REPO / "data" / "processed" / "tasks_revised.jsonl"
AUGMENTED = REPO / "data" / "augmented" / "tasks_augmented.jsonl"
GOLD_TASKS = SPLITS_DIR / "gold_test" / "gold_tasks.jsonl"

SPLITS = ("train", "validation", "test", "test_base", "test_augmented", "gold",
          "shifted", "shifted_train", "shifted_validation", "shifted_test")


@dataclass
class EvalItem:
    task: TaskRecord
    faults: list[FaultSpec] = field(default_factory=list)
    item_type: str = "task"
    parent_task_id: str | None = None
    parent_split: str | None = None
    fault_label: str = "none"

    @property
    def task_id(self) -> str:
        return self.task.task_id

    @property
    def family(self) -> str:
        return self.task.template_family

    @property
    def unanswerable(self) -> bool:
        return self.task.evaluation_rule == EvaluationRule.unanswerable

    def slices(self) -> dict[str, str]:
        return {
            "family": self.family,
            "answerability": "unanswerable" if self.unanswerable else "answerable",
            "item_type": self.item_type,
            "fault": self.fault_label,
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@lru_cache(maxsize=1)
def _base_tasks() -> dict[str, TaskRecord]:
    return {r["task_id"]: TaskRecord.model_validate(r) for r in _read_jsonl(BASE_TASKS)}


@lru_cache(maxsize=1)
def _augmented() -> dict[str, dict[str, Any]]:
    return {r["task"]["task_id"]: r for r in _read_jsonl(AUGMENTED)}


def _manifest(split: str) -> list[dict[str, str]]:
    with (SPLITS_DIR / f"{split}.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _item_from_row(row: dict[str, str]) -> EvalItem:
    if row["item_type"] == "task":
        return EvalItem(task=_base_tasks()[row["item_id"]], item_type="base")
    rec = _augmented()[row["item_id"]]
    aug = rec["augmentation"]
    faults = [FaultSpec(**f) for f in aug.get("fault") or []]
    return EvalItem(task=TaskRecord.model_validate(rec["task"]), faults=faults, item_type=aug["type"],
                    parent_task_id=aug.get("parent_task_id"), parent_split=row.get("parent_split") or None,
                    fault_label=aug.get("params", {}).get("fault_label", "none") if faults else "none")


def load_split(split: str) -> list[EvalItem]:
    """Return the evaluation items of ``split`` (see module docstring)."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; choose from {SPLITS}")
    if split == "gold":
        from code.evaluation.gold import verify_gold

        verify_gold()
        return [EvalItem(task=TaskRecord.model_validate(r), item_type="gold") for r in _read_jsonl(GOLD_TASKS)]
    if split.startswith("shifted"):
        rows = _manifest("shifted")
        if split != "shifted":
            parent = split.split("_", 1)[1]
            rows = [r for r in rows if r["parent_split"] == parent]
        return [_item_from_row(r) for r in rows]
    source = "test" if split.startswith("test") else split
    rows = [r for r in _manifest(source) if r["item_type"] != "trajectory"]
    if split == "test_base":
        rows = [r for r in rows if r["item_type"] == "task"]
    elif split == "test_augmented":
        rows = [r for r in rows if r["item_type"].startswith("augmented:")]
    return [_item_from_row(r) for r in rows]


def item_index() -> dict[str, EvalItem]:
    """Every known item by task_id (used to re-attach slices to cached runs)."""
    out: dict[str, EvalItem] = {}
    for split in ("train", "validation", "test", "shifted"):
        for item in load_split(split):
            out[item.task_id] = item
    return out

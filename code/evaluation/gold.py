"""Gold test set protection (A2 rules).

* ``verify_gold()`` checks both gold files against the SHA-256 values frozen in
  HANDOFF_A2.md and raises if either differs. ``load_split("gold")`` calls it,
  so no code path can evaluate on a modified gold set.
* ``log_gold_evaluation()`` appends one row to TEST_SET_LOG.md for every gold
  evaluation, with the date, registry version, model and reason, as the brief
  requires. The evaluation script calls it automatically.

The gold set is never read by the search, threshold selection or retraining code.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GOLD_DIR = REPO / "data" / "splits" / "gold_test"
TEST_SET_LOG = REPO / "TEST_SET_LOG.md"
FROZEN_SHA256 = {
    "gold_tasks.jsonl": "f4b15d31aafd6ce4eb76ceaaca7c630cc7c4c08c72e93ca1bcec94b53c20d994",
    "gold_labels.csv": "72e8f84fa8363e0f9a0da2dfc408fd3cee454f7befed875e9dba11cbd5ef78bc",
}


class GoldSetModified(RuntimeError):
    """The gold test files do not match the checksums frozen at a1-final."""


def verify_gold(gold_dir: Path = GOLD_DIR) -> None:
    for name, expected in FROZEN_SHA256.items():
        actual = hashlib.sha256((gold_dir / name).read_bytes()).hexdigest()
        if actual != expected:
            raise GoldSetModified(f"{name}: sha256 {actual} != frozen {expected}")


def log_gold_evaluation(version: str, config_hash: str, model: str | None, reason: str,
                        success: float, run_id: str, log_path: Path = TEST_SET_LOG) -> None:
    """Append one gold evaluation to TEST_SET_LOG.md."""
    if not reason.strip():
        raise ValueError("every gold evaluation needs a reason")
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    row = (f"| {date} | {version} | `{config_hash[:12]}` | {model or '-'} | {reason.strip()} | "
           f"{success:.3f} | `{run_id}` |\n")
    with log_path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(row)

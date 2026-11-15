"""Every result in the A2 evidence index must link to an MLflow run ID (A2 brief, Part B)."""

from __future__ import annotations

import csv

from code.reporting.evidence_index import FIELDS, OUT, collect


def test_every_collected_result_has_a_run_id() -> None:
    rows = collect()
    assert rows, "no evidence collected"
    assert [r["id"] for r in rows if not r["mlflow_run_id"]] == []
    assert {"A", "B", "C", "D", "E"} <= {r["part"] for r in rows}


def test_committed_index_matches_a_rebuild() -> None:
    with OUT.open(encoding="utf-8", newline="") as fh:
        committed = list(csv.DictReader(fh))
    rebuilt = [{k: str(v) for k, v in r.items()} for r in collect()]
    assert list(committed[0].keys()) == FIELDS
    assert committed == rebuilt, "rebuild with python -m code.reporting.evidence_index"
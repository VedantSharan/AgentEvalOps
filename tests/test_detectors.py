"""Tests for code/annotation/detectors.py and compare_rounds.py."""

from __future__ import annotations

from code.annotation.compare_rounds import COARSE
from code.annotation.detectors import STEP_LIMIT, detect, metrics


def _step(i: int, tool: str, args: dict, ok: bool = True, err: str | None = None) -> dict:
    return {"step_index": i, "action_type": "tool_call", "tool_name": tool, "arguments": args,
            "tool_result": {"success": ok, "error": err}, "error": None}


def test_loop_strict_needs_three_identical_calls() -> None:
    two = {"steps": [_step(0, "retrieve", {"query": "x"}), _step(1, "retrieve", {"query": "x"})], "errors": []}
    three = {"steps": two["steps"] + [_step(2, "retrieve", {"query": "x"})], "errors": []}
    assert not detect(two)["loop_strict"] and detect(three)["loop_strict"]


def test_loop_extended_fires_on_step_limit_without_answer() -> None:
    t = {"steps": [_step(0, "retrieve", {"query": "a"})], "errors": [STEP_LIMIT], "final_answer": None}
    d = detect(t)
    assert d["loop_extended"] and not d["loop_strict"]


def test_malformed_detects_argument_rejections_only() -> None:
    bad = {"steps": [_step(0, "query_records", {"field": "x"}, ok=False, err="Unknown field 'x'.")], "errors": []}
    missing_file = {"steps": [_step(0, "read_file", {"filename": "y"}, ok=False, err="File not found in virtual store: 'y'.")], "errors": []}
    injected = {"steps": [_step(0, "read_file", {"filename": "y"}, ok=False, err="503 Service Unavailable: upstream read_file backend")], "errors": []}
    assert detect(bad)["malformed_any"] and detect(bad)["malformed_strict"]
    assert detect(missing_file)["malformed_any"] and not detect(missing_file)["malformed_strict"]
    assert not detect(injected)["malformed_any"]  # injected outages are not argument errors


def test_metrics_values() -> None:
    m = metrics([True, True, False, False], [True, False, True, False])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1) and m["precision"] == 0.5 and m["recall"] == 0.5


def test_coarse_mapping_covers_all_categories() -> None:
    from pathlib import Path
    from code.annotation.agreement import allowed_values_from_config
    cats = allowed_values_from_config(Path("code/annotation/label_studio_config_v2.xml"))["primary_failure_category"]
    assert set(cats) == set(COARSE)

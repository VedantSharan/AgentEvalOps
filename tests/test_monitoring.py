"""Tests for code/monitoring (detectors, replay streams, live monitor)."""

from __future__ import annotations

import json
from pathlib import Path

from code.monitoring.detectors import Cusum, PageHinkley, Reference, run_detector
from code.monitoring.monitor import LiveMonitor
from code.monitoring.replay import evaluate_detector, make_stream, select_thresholds, shift_share
from code.monitoring.signals import signals


def row(task: str, tool_errors: int, wall_ms: float, correct: bool, fault: str = "none") -> dict:
    return {"task_id": task, "tool_errors": tool_errors, "retries_used": 0, "malformed_calls": 0,
            "loop_extended": False, "steps": 3, "wall_ms": wall_ms, "total_tokens": 3000, "correct": correct,
            "slices": {"fault": fault}}


BASE = [row(f"b{i}", 0, 20000 + 500 * (i % 5), i % 3 != 0) for i in range(15)]
SHIFTED = [row(f"s{i}", 2 + i % 2, 26000 + 500 * (i % 5), i % 4 == 0, "information_outage") for i in range(10)]


def test_cusum_and_page_hinkley_fire_on_upward_shift_only() -> None:
    ref = Reference(0.0, 1.0)
    calm = [0.1, -0.2, 0.3, 0.0] * 10
    shifted = calm + [3.0] * 10
    for det in (Cusum(ref, k=0.5, h=4.0), PageHinkley(ref, delta=0.5, lam=4.0)):
        for x in calm:
            det.update(x)
        assert det.alarms == []
    assert run_detector("cusum", ref, 0.5, 4.0, shifted)[0] >= len(calm)
    assert run_detector("page_hinkley", ref, 0.5, 4.0, [-3.0] * 30) == []


def test_schedule_ramps_linearly() -> None:
    assert shift_share(99) == 0.0 and shift_share(149) == 0.5 and shift_share(250) == 1.0
    assert shift_share(100, ramp=0) == 1.0


def test_streams_are_seeded_and_follow_the_schedule() -> None:
    a, b = make_stream(BASE, SHIFTED, seed=7), make_stream(BASE, SHIFTED, seed=7)
    assert a == b
    assert all(r["source"] == "base" for r in a[:100])
    assert all(r["source"] == "shifted" for r in a[200:])


def test_detector_reports_delay_and_false_alarms() -> None:
    stream = make_stream(BASE, SHIFTED, seed=1)
    res = evaluate_detector(stream, "tool_failures", "cusum", 0.5, 4.0, drift_start=100)
    assert res["detected"] and res["delay"] >= 0 and res["false_alarms"] == 0


def test_threshold_selection_and_live_monitor(tmp_path: Path) -> None:
    chosen, grid = select_thresholds(BASE, SHIFTED)
    assert "tool_failures" in chosen["monitored_signals"] and len(chosen["monitored_signals"]) == 2
    assert grid and all({"signal", "detector", "false_alarms_per_100"} <= set(g) for g in grid)
    path = tmp_path / "thr.json"
    path.write_text(json.dumps(chosen), encoding="utf-8")
    mon = LiveMonitor(path)
    for r in BASE * 2:
        assert mon.update(r, graded=True) == []
    raised = [a for r in SHIFTED * 2 for a in mon.update(r, graded=True)]
    assert raised and mon.snapshot()["drift"]


def test_signals_count_retries_as_tool_failures() -> None:
    s = signals({"tool_errors": 1, "retries_used": 2, "wall_ms": 1500, "correct": True})
    assert s["tool_failures"] == 3.0 and s["latency_s"] == 1.5 and s["correct"] == 1.0


def test_live_monitor_without_thresholds_tracks_signals_only(tmp_path: Path) -> None:
    mon = LiveMonitor(tmp_path / "missing.json")
    assert mon.update(BASE[0], graded=True) == []
    snap = mon.snapshot()
    assert snap["requests"] == 1 and not snap["thresholds_loaded"] and snap["monitored_signals"] == []


def test_warmup_requests_do_not_raise_alerts(tmp_path: Path) -> None:
    chosen, _ = select_thresholds(BASE, SHIFTED)
    path = tmp_path / "thr.json"
    path.write_text(json.dumps(chosen), encoding="utf-8")
    mon = LiveMonitor(path)
    cold = dict(BASE[0], wall_ms=600000.0, tool_errors=9)
    assert mon.update(cold, graded=False) == [] and mon.update(cold, graded=False) == []

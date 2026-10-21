"""Tests for code/active_learning/al_vs_random.py."""

from __future__ import annotations

import numpy as np

from code.active_learning.al_vs_random import al_order, evaluate, scores, trace_features


def _traj(n_steps: int, repeat: bool = False, fail_tool: bool = False) -> dict:
    steps = []
    for i in range(n_steps):
        ok = not (fail_tool and i == 0)
        args = {"query": "x"} if repeat else {"query": f"q{i}"}
        steps.append({"step_index": i, "action_type": "tool_call", "tool_name": "retrieve", "arguments": args,
                      "tool_result": {"success": ok}, "error": None})
    return {"steps": steps}


def test_trace_features_counts_repeats_and_errors() -> None:
    f = trace_features(_traj(4, repeat=True, fail_tool=True))
    assert f == {"steps": 4, "repeats": 3, "tool_errors": 1}
    assert trace_features(_traj(2)) == {"steps": 2, "repeats": 0, "tool_errors": 0}


def test_scores_are_bounded_and_monotone() -> None:
    feats = [{"steps": 2, "repeats": 0, "tool_errors": 0}, {"steps": 10, "repeats": 5, "tool_errors": 4}]
    s = scores(feats)
    assert 0 <= s[0] < s[1] <= 3


def test_ties_broken_randomly_but_reproducibly() -> None:
    sc = np.array([1.0, 1.0, 1.0, 2.0])
    o1, o2 = al_order(sc, np.random.default_rng(0)), al_order(sc, np.random.default_rng(0))
    assert o1[0] == 3 and list(o1) == list(o2)


def test_informative_signal_beats_random_on_synthetic_pool() -> None:
    pool = [{"features": {"steps": 9, "repeats": 2, "tool_errors": 1}, "failure": True, "category": "loop_or_repetition"} for _ in range(5)]
    pool += [{"features": {"steps": 2, "repeats": 0, "tool_errors": 0}, "failure": False, "category": None} for _ in range(45)]
    rows = evaluate(pool, [5], trials=200, seed=1)
    al = next(r for r in rows if r["strategy"] == "active_learning")
    rnd = next(r for r in rows if r["strategy"] == "random")
    assert al["failure_coverage_mean"] == 1.0 and rnd["failure_coverage_mean"] < 0.3
    assert al["p_random_ge_mean"] < 0.01


def test_uninformative_signal_does_not_beat_random() -> None:
    rng = np.random.default_rng(3)
    pool = [{"features": {"steps": int(rng.integers(1, 6)), "repeats": 0, "tool_errors": 0},
             "failure": bool(rng.random() < 0.2), "category": None} for _ in range(200)]
    rows = evaluate(pool, [50], trials=300, seed=2)
    al = next(r for r in rows if r["strategy"] == "active_learning")
    assert al["p_random_ge_mean"] > 0.05  # no real signal -> AL is not distinguishable from random

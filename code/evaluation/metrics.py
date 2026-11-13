"""Metrics, 95% bootstrap confidence intervals, slices and paired tests (Part C).

Runs are clustered by task: a configuration may run each task several times
(different seeds), and runs of the same task are not independent. The bootstrap
therefore resamples *tasks* with replacement and averages each task's runs, so
the interval reflects task-to-task variation, not just repeated sampling.

Paired comparisons use the tasks both systems ran:
  * exact McNemar test on per-(task, seed) success pairs (binomial test on the
    discordant pairs), and
  * a paired bootstrap of the difference in mean per-task success.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np
from scipy.stats import binomtest

BOOTSTRAP_SEED = 2026
N_BOOTSTRAP = 5000
NUMERIC_FIELDS = ("steps", "llm_calls", "tool_calls", "total_tokens", "prompt_tokens", "completion_tokens",
                  "wall_ms", "llm_latency_ms", "tool_errors", "malformed_calls")
RATE_FIELDS = ("submitted", "loop_strict", "loop_extended", "malformed_any", "malformed_strict", "forced_submit")


def _by_task(rows: Iterable[dict[str, Any]], field: str = "correct") -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        grouped[r["task_id"]].append(float(r[field]))
    return grouped


def bootstrap_ci(rows: list[dict[str, Any]], field: str = "correct", n_boot: int = N_BOOTSTRAP,
                 seed: int = BOOTSTRAP_SEED, alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean of per-task means, with a task-clustered percentile bootstrap CI."""
    grouped = _by_task(rows, field)
    if not grouped:
        return (math.nan, math.nan, math.nan)
    task_means = np.array([np.mean(v) for v in grouped.values()])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(task_means), size=(n_boot, len(task_means)))
    boots = task_means[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(task_means.mean()), float(lo), float(hi)


def pass_at_k(rows: list[dict[str, Any]]) -> tuple[float, int]:
    """pass^k: share of tasks solved in *all* k runs (k = runs per task, the minimum across tasks)."""
    grouped = _by_task(rows)
    if not grouped:
        return (math.nan, 0)
    k = min(len(v) for v in grouped.values())
    return float(np.mean([all(x == 1.0 for x in v[:k]) for v in grouped.values()])), k


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Headline metrics for one (system, split)."""
    if not rows:
        return {"n_runs": 0, "n_tasks": 0}
    success, lo, hi = bootstrap_ci(rows)
    pk, k = pass_at_k(rows)
    out: dict[str, Any] = {
        "n_runs": len(rows),
        "n_tasks": len({r["task_id"] for r in rows}),
        "success": round(success, 4),
        "success_ci_low": round(lo, 4),
        "success_ci_high": round(hi, 4),
        f"pass^{k}": round(pk, 4),
        "pass_k": round(pk, 4),
        "k": k,
    }
    for f in NUMERIC_FIELDS:
        vals = np.array([float(r.get(f) or 0.0) for r in rows])
        out[f"mean_{f}"] = round(float(vals.mean()), 3)
    lat = np.array([float(r.get("wall_ms") or 0.0) for r in rows])
    out["p50_wall_ms"] = round(float(np.percentile(lat, 50)), 1)
    out["p95_wall_ms"] = round(float(np.percentile(lat, 95)), 1)
    for f in RATE_FIELDS:
        out[f"rate_{f}"] = round(float(np.mean([bool(r.get(f)) for r in rows])), 4)
    out["failure_categories"] = dict(Counter(r.get("failure_category", "unknown") for r in rows if not r["correct"]))
    out["process_categories"] = dict(Counter(r.get("process_category", "unknown") for r in rows))
    return out


def slice_table(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """Success with CI and mean cost per value of slice ``key`` (rows carry a ``slices`` dict)."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r["slices"][key]].append(r)
    table = {}
    for value, grp in sorted(groups.items()):
        s, lo, hi = bootstrap_ci(grp)
        table[value] = {"n_tasks": len({r["task_id"] for r in grp}), "n_runs": len(grp), "success": round(s, 4),
                        "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                        "mean_steps": round(float(np.mean([r["steps"] for r in grp])), 2),
                        "mean_total_tokens": round(float(np.mean([r["total_tokens"] for r in grp])), 1)}
    return table


def paired_comparison(a: list[dict[str, Any]], b: list[dict[str, Any]], n_boot: int = N_BOOTSTRAP,
                      seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Compare system B against system A on the tasks (and seeds) both ran."""
    key = lambda r: (r["task_id"], r["seed"])  # noqa: E731
    a_map = {key(r): bool(r["correct"]) for r in a}
    b_map = {key(r): bool(r["correct"]) for r in b}
    common = sorted(set(a_map) & set(b_map))
    if not common:
        return {"n_pairs": 0}
    b_only = sum(1 for k in common if b_map[k] and not a_map[k])
    a_only = sum(1 for k in common if a_map[k] and not b_map[k])
    discordant = a_only + b_only
    p_mcnemar = binomtest(b_only, discordant, 0.5).pvalue if discordant else 1.0

    tasks = sorted({k[0] for k in common})
    diff = np.array([np.mean([b_map[k] for k in common if k[0] == t]) - np.mean([a_map[k] for k in common if k[0] == t])
                     for t in tasks], dtype=float)
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_boot = float(2 * min(np.mean(boots <= 0), np.mean(boots >= 0)))
    return {
        "n_pairs": len(common), "n_tasks": len(tasks),
        "success_a": round(float(np.mean([a_map[k] for k in common])), 4),
        "success_b": round(float(np.mean([b_map[k] for k in common])), 4),
        "b_only_correct": b_only, "a_only_correct": a_only,
        "mcnemar_exact_p": round(float(p_mcnemar), 5),
        "diff_mean": round(float(diff.mean()), 4), "diff_ci_low": round(float(lo), 4), "diff_ci_high": round(float(hi), 4),
        "paired_bootstrap_p": round(min(p_boot, 1.0), 5),
    }

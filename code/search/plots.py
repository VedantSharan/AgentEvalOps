"""Search-analysis plots for Part B (static PNGs for the report).

    python -m code.search.plots

Reads supporting_files/a2/search/{trials.csv,summary.json} and writes:
  pareto.png         validation success vs mean tokens per task, one point per
                     trial (colour = sampler), Pareto-optimal trials ringed,
                     the selected configuration labelled
  best_so_far.png    best feasible validation success after each trial, random vs NSGA-II
  param_effects.png  mean validation success for each value of each searched parameter
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from code.agent.config import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "supporting_files" / "a2" / "search"
COLORS = {"random": "#2a78d6", "nsga2": "#eb6834", "refine": "#1baf7a", "manual": "#eda100"}
LABELS = {"random": "Random search", "nsga2": "NSGA-II", "refine": "Stage 2 grid (3B, T=0)",
          "manual": "Part A hand-designed"}
SAMPLERS = ("random", "nsga2", "refine", "manual")
INK, MUTED, GRID = "#2b2b29", "#6b6a64", "#e4e3dd"


def _style(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def load(out: Path = OUT) -> tuple[list[dict], dict]:
    with (out / "trials.csv").open(encoding="utf-8", newline="") as fh:
        trials = list(csv.DictReader(fh))
    for t in trials:
        for k in ("success", "mean_total_tokens", "mean_wall_s", "p95_wall_s"):
            t[k] = float(t[k])
        t["feasible"] = t["feasible"] == "True"
    return trials, json.loads((out / "summary.json").read_text(encoding="utf-8"))


def pareto_plot(trials: list[dict], summary: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    pareto = {p["name"] for p in summary["pareto_front"]}
    for sampler in SAMPLERS:
        pts = [t for t in trials if t["sampler"] == sampler]
        if not pts:
            continue
        ax.scatter([t["mean_total_tokens"] for t in pts], [t["success"] for t in pts], s=46, color=COLORS[sampler],
                   edgecolor="white", linewidth=1.5, label=LABELS[sampler], zorder=3)
    front = sorted((t for t in trials if t["name"] in pareto), key=lambda t: t["mean_total_tokens"])
    ax.scatter([t["mean_total_tokens"] for t in front], [t["success"] for t in front], s=130, facecolor="none",
               edgecolor=INK, linewidth=1.2, label="Pareto-optimal (success, tokens, latency)", zorder=2)
    best = summary["selected"]
    ax.annotate(f"selected: {best['name']}", (float(best["mean_total_tokens"]), float(best["success"])),
                xytext=(8, -14), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel("Mean tokens per task (cost)")
    ax.set_ylabel("Validation success")
    ax.set_ylim(-0.03, 1.03)
    _style(ax)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=8, loc="best")
    ax.set_title("Configuration search: quality vs cost", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def best_so_far(trials: list[dict], sampler: str) -> list[float]:
    vals, cur = [], 0.0
    for t in sorted((t for t in trials if t["sampler"] == sampler), key=lambda t: int(t["trial"])):
        cur = max(cur, t["success"] if t["feasible"] else 0.0)
        vals.append(cur)
    return vals


def best_so_far_plot(trials: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    for sampler in ("random", "nsga2", "refine"):
        vals = best_so_far(trials, sampler)
        if not vals:
            continue
        ax.plot(range(1, len(vals) + 1), vals, color=COLORS[sampler], linewidth=2, marker="o", markersize=5,
                markeredgecolor="white", label=LABELS[sampler], drawstyle="steps-post")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best feasible validation success")
    ax.set_ylim(-0.03, 1.03)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Search progress (stage 1: 10 trials per sampler; stage 2: 8-point grid)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def param_effects_plot(trials: list[dict], summary: dict, path: Path) -> dict[str, dict[str, float]]:
    params = list(summary["space"])
    effects: dict[str, dict[str, float]] = {}
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 4.8), dpi=150)
    for ax, p in zip(axes.ravel(), params):
        groups: dict[str, list[float]] = defaultdict(list)
        for t in trials:
            groups[str(t[p])].append(t["success"])
        values = [str(v) for v in summary["space"][p]]
        means = [sum(groups[v]) / len(groups[v]) if groups[v] else 0.0 for v in values]
        effects[p] = {v: round(m, 4) for v, m in zip(values, means)}
        ax.bar(range(len(values)), means, color=COLORS["random"], width=0.6, edgecolor="white", linewidth=2)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([f"{v}\n(n={len(groups[v])})" for v in values], fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(p, fontsize=9, color=INK, loc="left")
        _style(ax)
    for ax in axes.ravel()[len(params):]:
        ax.axis("off")
    fig.suptitle("Mean validation success by parameter value (all trials)", fontsize=10, color=INK, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return effects


def main() -> None:
    trials, summary = load()
    pareto_plot(trials, summary, OUT / "pareto.png")
    best_so_far_plot(trials, OUT / "best_so_far.png")
    effects = param_effects_plot(trials, summary, OUT / "param_effects.png")
    (OUT / "param_effects.json").write_text(json.dumps(effects, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("plots written to", OUT.relative_to(REPO_ROOT).as_posix())


if __name__ == "__main__":
    main()

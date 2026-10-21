"""
Active learning vs. random selection for labelling trajectories (Assignment 1, Part D).

Question: with a fixed labelling budget, does choosing trajectories by a
label-free informativeness score find more failures than picking at random?

Signals (computed from the trace only, never from labels or grader output):
    steps        number of trajectory steps
    repeats      tool calls that repeat an earlier (tool, arguments) pair exactly
    tool_errors  tool results with success=False plus steps carrying an error
Score:
    score = steps / max_steps_in_pool + min(repeats, 3) / 3 + min(tool_errors, 3) / 3
    (each signal scaled to [0, 1], equal weights; ties broken by a seeded shuffle)

Pools and ground truth
    human        trajectories with final human labels (data/labels/labels_round*.csv);
                 failure = human outcome != success.  This is the primary result.
    proxy_full   all collected trajectories; failure = grader-v2 says the answer is
                 wrong (data/processed/regrade_v2.csv).  Automatic proxy labels,
                 reported separately and marked as such.

For both strategies the candidate pool, budgets and metric are identical.
Random selection and the AL tie-breaks are each repeated over N_TRIALS seeds;
we report the mean and the 5th-95th percentile band.

Metrics at each budget B:
    failure_coverage   failures selected / failures in pool
    category_coverage  distinct human failure categories found / categories present (human pool only)

Usage (from the repository root):
    python -m code.active_learning.al_vs_random
    python -m code.active_learning.al_vs_random --labels data/labels/labels_round1.csv data/labels/labels_round2.csv
Outputs:
    data/active_learning/<pool>/{al,random}_b<B>.csv     selection manifests (seed 0)
    supporting_files/active_learning/features.csv        per-trajectory signals and score
    supporting_files/active_learning/curve_<pool>.csv     plot data
    supporting_files/active_learning/al_vs_random.png     the plot
    supporting_files/active_learning/summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SEED = 203
N_TRIALS = 1000
SIGNALS = ("steps", "repeats", "tool_errors")
HEADLINE_BUDGETS = (10, 20, 30)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def trace_features(traj: dict) -> dict[str, int]:
    seen, repeats = set(), 0
    for s in traj["steps"]:
        if s["action_type"] == "tool_call":
            key = (s.get("tool_name"), json.dumps(s.get("arguments"), sort_keys=True))
            repeats += key in seen
            seen.add(key)
    errors = sum(1 for s in traj["steps"]
                 if (s.get("tool_result") and not s["tool_result"]["success"]) or s.get("error"))
    return {"steps": len(traj["steps"]), "repeats": repeats, "tool_errors": errors}


def scores(feats: list[dict[str, int]], signals: tuple[str, ...] = SIGNALS) -> np.ndarray:
    max_steps = max(f["steps"] for f in feats) or 1
    parts = {
        "steps": np.array([f["steps"] / max_steps for f in feats]),
        "repeats": np.array([min(f["repeats"], 3) / 3 for f in feats]),
        "tool_errors": np.array([min(f["tool_errors"], 3) / 3 for f in feats]),
    }
    return sum(parts[s] for s in signals)


# ---------------------------------------------------------------------------
# Selection and evaluation
# ---------------------------------------------------------------------------


def al_order(score: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Indices by descending score; ties broken by a random permutation."""
    tiebreak = rng.permutation(len(score))
    return np.lexsort((tiebreak, -score))


def evaluate(pool: list[dict], budgets: list[int], trials: int, seed: int) -> list[dict]:
    is_fail = np.array([p["failure"] for p in pool])
    cats = [p.get("category") for p in pool]
    n_fail = int(is_fail.sum())
    present = {c for c, f in zip(cats, is_fail) if f and c}
    feats = [p["features"] for p in pool]
    strategies = {"active_learning": scores(feats)}
    for sig in SIGNALS:
        strategies[f"al_{sig}_only"] = scores(feats, (sig,))

    rows = []
    random_cov = None
    for name, sc in [("random", None)] + list(strategies.items()):
        rng = np.random.default_rng(seed)
        cov = np.zeros((trials, len(budgets)))
        catcov = np.zeros((trials, len(budgets)))
        for t in range(trials):
            order = rng.permutation(len(pool)) if sc is None else al_order(sc, rng)
            for j, b in enumerate(budgets):
                sel = order[:b]
                cov[t, j] = is_fail[sel].sum() / n_fail if n_fail else np.nan
                found = {cats[i] for i in sel if is_fail[i] and cats[i]}
                catcov[t, j] = len(found) / len(present) if present else np.nan
        if sc is None:
            random_cov = cov
        for j, b in enumerate(budgets):
            rows.append({
                "strategy": name, "budget": b, "pool_size": len(pool), "pool_failures": n_fail,
                "failure_coverage_mean": round(float(np.nanmean(cov[:, j])), 4),
                "failure_coverage_p05": round(float(np.nanpercentile(cov[:, j], 5)), 4),
                "failure_coverage_p95": round(float(np.nanpercentile(cov[:, j], 95)), 4),
                "category_coverage_mean": None if not present else round(float(np.nanmean(catcov[:, j])), 4),
                "oracle_coverage": round(min(b, n_fail) / n_fail, 4) if n_fail else None,
                # share of random trials that do at least as well as this strategy's mean
                "p_random_ge_mean": round(float(np.mean(random_cov[:, j] >= np.nanmean(cov[:, j]))), 4),
            })
    return rows


def manifests(pool: list[dict], budgets: list[int], out: Path, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sc = scores([p["features"] for p in pool])
    al = al_order(sc, np.random.default_rng(seed))
    rnd = np.random.default_rng(seed).permutation(len(pool))
    for name, order in (("al", al), ("random", rnd)):
        for b in budgets:
            with (out / f"{name}_b{b}.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["rank", "run_id", "task_id", "score", "steps", "repeats", "tool_errors", "seed"])
                for rank, i in enumerate(order[:b], 1):
                    p = pool[i]
                    w.writerow([rank, p["run_id"], p["task_id"], round(float(sc[i]), 4), *[p["features"][s] for s in SIGNALS], seed])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot(curves: dict[str, list[dict]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, ink, muted = "#2a78d6", "#eb6834", "#1f1f1e", "#76756e"
    titles = {"human": "Human labels (primary)", "proxy_full": "All trajectories, grader-v2 proxy labels"}
    fig, axes = plt.subplots(1, len(curves), figsize=(5.2 * len(curves), 4.0), squeeze=False)
    for ax, (pool, rows) in zip(axes[0], curves.items()):
        get = lambda s, k: [r[k] for r in rows if r["strategy"] == s]
        b = get("random", "budget")
        ax.fill_between(b, get("random", "failure_coverage_p05"), get("random", "failure_coverage_p95"), color=orange, alpha=0.15, linewidth=0)
        ax.plot(b, get("random", "failure_coverage_mean"), color=orange, lw=2, label="Random (mean, 5-95% band)")
        ax.fill_between(b, get("active_learning", "failure_coverage_p05"), get("active_learning", "failure_coverage_p95"), color=blue, alpha=0.15, linewidth=0)
        ax.plot(b, get("active_learning", "failure_coverage_mean"), color=blue, lw=2, marker="o", ms=4, label="Active learning (score)")
        ax.plot(b, get("random", "oracle_coverage"), color=muted, lw=1.2, ls="--", label="Oracle (perfect ranking)")
        n, f = rows[0]["pool_size"], rows[0]["pool_failures"]
        ax.set_title(f"{titles.get(pool, pool)}\nN={n}, failures={f}", fontsize=10, color=ink)
        ax.set_xlabel("Labelling budget (trajectories)", color=ink)
        ax.set_ylabel("Failure coverage", color=ink)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_pools(traj_path: Path, label_paths: list[Path], regrade_path: Path) -> tuple[dict[str, list[dict]], list[dict]]:
    trajs = [json.loads(line) for line in traj_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    feats = {t["run_id"]: trace_features(t) for t in trajs}
    final = {t["run_id"]: t.get("final_answer") for t in trajs}
    with regrade_path.open(encoding="utf-8") as fh:
        correct_v2 = {r["run_id"]: r["answer_correct_v2"] == "True" for r in csv.DictReader(fh)}

    human: dict[str, dict] = {}
    for lp in label_paths:
        with lp.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["run_id"] in human:
                    raise ValueError(f"{r['run_id']} labelled in more than one label file")
                human[r["run_id"]] = r
    missing = set(human) - set(feats)
    if missing:
        raise ValueError(f"labelled run_ids not in trajectories: {sorted(missing)[:3]}")

    pools = {
        "human": [{"run_id": rid, "task_id": r["task_id"], "features": feats[rid],
                   "failure": r["outcome"] != "success", "answered": bool(final[rid]),
                   "category": r["primary_failure_category"] if r["outcome"] != "success" else None}
                  for rid, r in sorted(human.items())],
        "proxy_full": [{"run_id": t["run_id"], "task_id": t["task_id"], "features": feats[t["run_id"]],
                        "failure": not correct_v2[t["run_id"]], "answered": bool(t.get("final_answer")), "category": None} for t in trajs],
    }
    feature_rows = [{"run_id": t["run_id"], "task_id": t["task_id"], **feats[t["run_id"]],
                     "score": None, "human_label": human.get(t["run_id"], {}).get("outcome", ""),
                     "grader_v2_correct": correct_v2[t["run_id"]]} for t in trajs]
    sc = scores([feats[t["run_id"]] for t in trajs])
    for row, s in zip(feature_rows, sc):
        row["score"] = round(float(s), 4)
    return pools, feature_rows


def main() -> None:
    p = argparse.ArgumentParser(description="Active learning vs random selection (failure coverage).")
    p.add_argument("--trajectories", type=Path, default=Path("data/experiments/multi_provider_run/trajectories.jsonl"))
    p.add_argument("--labels", type=Path, nargs="+", default=[Path("data/labels/labels_round1.csv")])
    p.add_argument("--regrade", type=Path, default=Path("data/processed/regrade_v2.csv"))
    p.add_argument("--trials", type=int, default=N_TRIALS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--evidence-dir", type=Path, default=Path("supporting_files/active_learning"))
    p.add_argument("--manifest-dir", type=Path, default=Path("data/active_learning"))
    a = p.parse_args()

    pools, feature_rows = load_pools(a.trajectories, a.labels, a.regrade)
    a.evidence_dir.mkdir(parents=True, exist_ok=True)
    with (a.evidence_dir / "features.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(feature_rows[0].keys()))
        w.writeheader()
        w.writerows(feature_rows)

    curves, summary = {}, {"seed": a.seed, "trials": a.trials, "label_files": [x.as_posix() for x in a.labels],
                           "score": "steps/max_steps + min(repeats,3)/3 + min(tool_errors,3)/3", "pools": {}}
    for name, pool in pools.items():
        n = len(pool)
        budgets = sorted({b for b in (5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100) if b < n} | {b for b in HEADLINE_BUDGETS if b < n})
        rows = evaluate(pool, budgets, a.trials, a.seed)
        curves[name] = [r for r in rows if r["strategy"] in ("active_learning", "random")]
        with (a.evidence_dir / f"curve_{name}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        manifests(pool, [b for b in HEADLINE_BUDGETS if b < n], a.manifest_dir / name, a.seed)
        head = {}
        for b in HEADLINE_BUDGETS:
            if b >= n:
                continue
            al = next(r for r in rows if r["strategy"] == "active_learning" and r["budget"] == b)
            rd = next(r for r in rows if r["strategy"] == "random" and r["budget"] == b)
            head[str(b)] = {"active_learning": al["failure_coverage_mean"], "random_mean": rd["failure_coverage_mean"],
                            "random_p05_p95": [rd["failure_coverage_p05"], rd["failure_coverage_p95"]],
                            "p_random_ge_al": al["p_random_ge_mean"], "oracle": al["oracle_coverage"],
                            "al_category_coverage": al["category_coverage_mean"], "random_category_coverage": rd["category_coverage_mean"],
                            "ablation_single_signal": {s: next(r["failure_coverage_mean"] for r in rows
                                                               if r["strategy"] == f"al_{s}_only" and r["budget"] == b) for s in SIGNALS}}
        al0 = al_order(scores([x["features"] for x in pool]), np.random.default_rng(a.seed))
        for b in head:
            sel = [pool[i] for i in al0[: int(b)]]
            head[b]["al_seed0_selected_without_submitted_answer"] = sum(not x["answered"] for x in sel)
        summary["pools"][name] = {"size": n, "failures": int(sum(x["failure"] for x in pool)),
                                  "ground_truth": "human final labels" if name == "human" else "grader v2 (automatic proxy)",
                                  "headline": head}
    plot(curves, a.evidence_dir / "al_vs_random.png")
    (a.evidence_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["pools"], indent=2))


if __name__ == "__main__":
    main()

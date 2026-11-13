"""Part E drift study: replayed traffic, threshold selection on validation, detection delay.

    python -m code.monitoring.replay --version production

Traffic is replayed from cached agent runs of the deployed version (every run
is a real agent execution on the CPU server; replay only decides the order).
A stream has N requests; for request t the source is chosen by a fixed
schedule and seed: in-distribution traffic until ``drift_start``, then the
share of shifted traffic rises linearly to 1 over ``ramp`` requests.

1. Thresholds (validation only). Streams from the validation split (in
   distribution) and shifted_validation (fault-injected validation tasks).
   For every label-free signal and both detectors, the (slack, threshold) pair
   with the shortest mean detection delay is chosen among those with at most
   MAX_FALSE_ALARMS_PER_100 false alarms per 100 in-distribution requests and a
   detection rate of at least MIN_DETECTION_RATE over 20 seeded streams. The two
   signals with the shortest delays become the monitored signals. Written to
   configs/monitoring_thresholds.json (bundled with the service).
2. Evaluation (test families, fixed seed 203). One stream test -> shifted_test,
   and one model-swap stream (test traffic answered by the same configuration
   with qwen2.5:1.5b after drift_start). Reported per signal and detector:
   first alarm, detection delay, false alarms before the drift and on a
   drift-free test stream, and the Spearman correlation between each rolling
   signal and rolling task success (the labelled quality metric).

Outputs: supporting_files/a2/monitoring/{thresholds_search.csv, summary.json,
stream_<scenario>.csv, signals_vs_quality_<scenario>.png, alerts.json}.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from code.agent.config import REPO_ROOT
from code.evaluation.evaluate import rows_for
from code.monitoring.detectors import Reference, run_detector
from code.monitoring.signals import LABEL_FREE, signals
from code.registry.registry import Registry
from code.tracing.mlflow_tracer import DEFAULT_TRACKING_URI

OUT = REPO_ROOT / "supporting_files" / "a2" / "monitoring"
THRESHOLDS = REPO_ROOT / "configs" / "monitoring_thresholds.json"
SEED = 203
N_REQUESTS = 300
DRIFT_START = 100
RAMP = 100
N_REFERENCE = 50
N_SELECTION_STREAMS = 20
WINDOW = 25
MAX_FALSE_ALARMS_PER_100 = 0.5
MIN_DETECTION_RATE = 0.9
SLACKS = (0.25, 0.5, 1.0)
THRESHOLD_GRID = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0)
DETECTORS = ("cusum", "page_hinkley")


def shift_share(t: int, drift_start: int = DRIFT_START, ramp: int = RAMP) -> float:
    if t < drift_start:
        return 0.0
    if ramp <= 0:
        return 1.0
    return min(1.0, (t - drift_start + 1) / ramp)


def make_stream(base: list[dict], shifted: list[dict], seed: int, n: int = N_REQUESTS,
                drift_start: int = DRIFT_START, ramp: int = RAMP) -> list[dict[str, Any]]:
    """Seeded replay: request t comes from ``shifted`` with probability shift_share(t)."""
    rng = np.random.default_rng(seed)
    stream = []
    for t in range(n):
        p = shift_share(t, drift_start, ramp) if shifted else 0.0
        source = shifted if rng.random() < p else base
        row = source[int(rng.integers(len(source)))]
        stream.append({"t": t, "shift_share": round(p, 3), "source": "shifted" if source is shifted else "base",
                       "task_id": row["task_id"], "fault": row["slices"]["fault"], **signals(row)})
    return stream


def _reference(stream: list[dict], signal: str) -> Reference:
    return Reference.fit(r[signal] for r in stream[:N_REFERENCE])


def evaluate_detector(stream: list[dict], signal: str, kind: str, slack: float, threshold: float,
                      drift_start: int | None) -> dict[str, Any]:
    ref = _reference(stream, signal)
    monitored = stream[N_REFERENCE:]
    alarms = [a + N_REFERENCE for a in run_detector(kind, ref, slack, threshold, (r[signal] for r in monitored))]
    if drift_start is None:
        return {"alarms": alarms, "false_alarms": len(alarms), "first_alarm": alarms[0] if alarms else None}
    before = [a for a in alarms if a < drift_start]
    after = [a for a in alarms if a >= drift_start]
    return {"alarms": alarms, "false_alarms": len(before), "first_alarm": after[0] if after else None,
            "delay": (after[0] - drift_start) if after else None, "detected": bool(after)}


def select_thresholds(val_base: list[dict], val_shifted: list[dict]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    no_drift = [make_stream(val_base, [], seed=SEED + 1000 + i) for i in range(N_SELECTION_STREAMS)]
    drift = [make_stream(val_base, val_shifted, seed=SEED + 2000 + i) for i in range(N_SELECTION_STREAMS)]
    monitored_len = N_REQUESTS - N_REFERENCE
    grid, best = [], {}
    for signal in LABEL_FREE:
        for kind in DETECTORS:
            for slack in SLACKS:
                for thr in THRESHOLD_GRID:
                    fa = [evaluate_detector(s, signal, kind, slack, thr, None)["false_alarms"] for s in no_drift]
                    dr = [evaluate_detector(s, signal, kind, slack, thr, DRIFT_START) for s in drift]
                    delays = [d["delay"] for d in dr if d["detected"]]
                    row = {"signal": signal, "detector": kind, "slack": slack, "threshold": thr,
                           "false_alarms_per_100": round(100 * float(np.mean(fa)) / monitored_len, 3),
                           "detection_rate": round(float(np.mean([d["detected"] for d in dr])), 3),
                           "mean_delay": round(float(np.mean(delays)), 1) if delays else None}
                    grid.append(row)
                    ok = row["false_alarms_per_100"] <= MAX_FALSE_ALARMS_PER_100 and \
                        row["detection_rate"] >= MIN_DETECTION_RATE
                    key = (signal, kind)
                    if ok and (key not in best or row["mean_delay"] < best[key]["mean_delay"]):
                        best[key] = row
    per_signal: dict[str, dict[str, Any]] = {}
    for (signal, kind), row in best.items():
        per_signal.setdefault(signal, {})[kind] = {k: row[k] for k in
                                                   ("slack", "threshold", "false_alarms_per_100",
                                                    "detection_rate", "mean_delay")}
    ranked = sorted(per_signal, key=lambda s: min(v["mean_delay"] for v in per_signal[s].values()))
    chosen = {
        "selected_on": "validation (cross_tool_chaining) vs shifted_validation; 20 seeded streams each",
        "criteria": {"max_false_alarms_per_100": MAX_FALSE_ALARMS_PER_100, "min_detection_rate": MIN_DETECTION_RATE,
                     "objective": "shortest mean detection delay"},
        "stream": {"n_requests": N_REQUESTS, "n_reference": N_REFERENCE, "drift_start": DRIFT_START, "ramp": RAMP},
        "monitored_signals": ranked[:2],
        "signals": per_signal,
        # Reference mean/spread of every signal on validation traffic, used by the live
        # monitor in the service before it has seen enough of its own traffic.
        "reference": {s: vars(Reference.fit(signals(r)[s] for r in val_base)) for s in LABEL_FREE},
    }
    return chosen, grid


def rolling(values: list[float], window: int = WINDOW) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        out[i] = arr[i - window + 1:i + 1].mean()
    return out


def analyse_stream(stream: list[dict], thresholds: dict[str, Any], drift_start: int | None) -> dict[str, Any]:
    res: dict[str, Any] = {}
    quality = rolling([r["correct"] for r in stream])
    for signal, dets in thresholds["signals"].items():
        res[signal] = {}
        sig_roll = rolling([r[signal] for r in stream])
        mask = ~np.isnan(sig_roll)
        rho = spearmanr(sig_roll[mask], quality[mask]).statistic if np.nanstd(sig_roll) > 0 else None
        res[signal]["spearman_rolling_vs_success"] = None if rho is None or np.isnan(rho) else round(float(rho), 3)
        for kind, p in dets.items():
            res[signal][kind] = evaluate_detector(stream, signal, kind, p["slack"], p["threshold"], drift_start)
    pre = [r["correct"] for r in stream[:drift_start]] if drift_start else [r["correct"] for r in stream]
    post = [r["correct"] for r in stream[drift_start + RAMP:]] if drift_start else []
    res["success_before"] = round(float(np.mean(pre)), 3)
    res["success_after"] = round(float(np.mean(post)), 3) if post else None
    return res


def plot_stream(stream: list[dict], thresholds: dict[str, Any], analysis: dict[str, Any], title: str,
                path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from code.search.plots import GRID, INK, MUTED

    signals_shown = thresholds["monitored_signals"]
    fig, axes = plt.subplots(len(signals_shown) + 1, 1, figsize=(7.2, 1.9 * (len(signals_shown) + 1)), dpi=150,
                             sharex=True)
    t = [r["t"] for r in stream]
    panels = [(s, f"rolling {s}", "#2a78d6") for s in signals_shown] + [("correct", "rolling success (labelled)", "#1baf7a")]
    for ax, (key, label, color) in zip(axes, panels):
        ax.plot(t, rolling([r[key] for r in stream]), color=color, linewidth=2)
        ax.set_ylabel(label, fontsize=8, color=INK)
        ax.axvspan(DRIFT_START, DRIFT_START + RAMP, color="#e4e3dd", alpha=0.6, linewidth=0)
        ax.axvline(N_REFERENCE, color=MUTED, linewidth=0.8, linestyle=":")
        if key in analysis:
            for kind, marker in (("cusum", "v"), ("page_hinkley", "^")):
                if kind in analysis[key]:
                    for a in analysis[key][kind]["alarms"]:
                        ax.plot(a, ax.get_ylim()[1], marker=marker, color="#eb6834", markersize=6, clip_on=False)
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8, colors=MUTED)
    axes[-1].set_xlabel("request", fontsize=8, color=INK)
    axes[0].set_title(f"{title}  (grey: ramp to shifted traffic; markers: alarms, v CUSUM, ^ Page-Hinkley)",
                      fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_stream(stream: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stream[0].keys()))
        w.writeheader()
        w.writerows(stream)


def swap_model_ref(ref: str, model: str, registry: Registry) -> str:
    """Write a copy of ``ref``'s configuration with another model; return its YAML path."""
    _, cfg = registry.resolve(ref)
    swapped = cfg.model_copy(update={"model": model, "name": f"{cfg.name}_swap_{model.replace(':', '_')}",
                                     "description": f"{cfg.name} with the model swapped to {model} (drift scenario)"})
    path = REPO_ROOT / "configs" / "drift" / f"{swapped.name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# Part E model-swap drift scenario\n{swapped.to_yaml()}", encoding="utf-8", newline="\n")
    return str(path)


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    p = argparse.ArgumentParser(description="Part E replay and drift detection")
    p.add_argument("--version", default="production")
    p.add_argument("--swap-model", default="qwen2.5:1.5b")
    args = p.parse_args(argv)
    reg = Registry()
    version, _ = reg.resolve(args.version)
    OUT.mkdir(parents=True, exist_ok=True)

    val_base = rows_for(args.version, "validation")
    val_shifted = rows_for(args.version, "shifted_validation")
    thresholds, grid = select_thresholds(val_base, val_shifted)
    thresholds["version"] = version
    THRESHOLDS.write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8", newline="\n")
    with (OUT / "thresholds_search.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(grid[0].keys()))
        w.writeheader()
        w.writerows(grid)

    test_base = rows_for(args.version, "test")
    test_shifted = rows_for(args.version, "shifted_test")
    swap_rows = rows_for(swap_model_ref(args.version, args.swap_model, reg), "test")
    scenarios = {
        "fault_shift": (make_stream(test_base, test_shifted, seed=SEED), DRIFT_START,
                        "Test traffic shifting to fault-injected test tasks"),
        "model_swap": (make_stream(test_base, swap_rows, seed=SEED, ramp=0), DRIFT_START,
                       f"Test traffic; model swapped to {args.swap_model} at request {DRIFT_START}"),
        "no_drift": (make_stream(test_base, [], seed=SEED), None, "Test traffic, no drift"),
    }
    summary: dict[str, Any] = {"version": version, "seed": SEED, "thresholds_file": "configs/monitoring_thresholds.json",
                               "monitored_signals": thresholds["monitored_signals"], "scenarios": {}}
    alerts = []
    for name, (stream, start, title) in scenarios.items():
        analysis = analyse_stream(stream, thresholds, start)
        summary["scenarios"][name] = {"title": title, "drift_start": start, **analysis}
        write_stream(stream, OUT / f"stream_{name}.csv")
        plot_stream(stream, thresholds, analysis, title, OUT / f"signals_vs_quality_{name}.png")
        if start is not None:
            firsts = [(analysis[s][k]["first_alarm"], s, k) for s in thresholds["monitored_signals"]
                      for k in DETECTORS if k in analysis.get(s, {}) and analysis[s][k]["first_alarm"] is not None]
            if firsts:
                t, s, k = min(firsts)
                alerts.append({"scenario": name, "request": t, "signal": s, "detector": k, "version": version,
                               "delay": t - start})
    summary["alerts"] = alerts
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    (OUT / "alerts.json").write_text(json.dumps(alerts, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "scenarios"}, indent=2))


if __name__ == "__main__":
    main()

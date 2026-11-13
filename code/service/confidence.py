"""Answer confidence for the deployed agent (Part D).

An agent run has no class probability, so confidence is the predicted
probability that the run is correct, from a logistic regression on label-free
features of the run itself:

    steps, LLM calls, tool failures (incl. retries), malformed calls, loop flag,
    blocked repeats, forced submission, abstained ("unanswerable" in the answer),
    submitted at all

It is trained on the production version's runs on the TRAIN split (graded by
grader v2) and its calibration is checked on the VALIDATION split (Brier score,
expected calibration error). Coefficients are stored in
configs/confidence_model.json and bundled with the service.

    python -m code.service.confidence --version production
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from code.agent.config import REPO_ROOT
from code.tracing.mlflow_tracer import DEFAULT_TRACKING_URI

MODEL_PATH = REPO_ROOT / "configs" / "confidence_model.json"
FEATURES = ("steps", "llm_calls", "tool_failures", "malformed_calls", "loop_extended", "blocked_repeats",
            "forced_submit", "abstained", "submitted")


def features(summary: dict[str, Any]) -> list[float]:
    answer = (summary.get("final_answer") or "").lower()
    values = {
        "steps": summary.get("steps", 0),
        "llm_calls": summary.get("llm_calls", 0),
        "tool_failures": summary.get("tool_errors", 0) + summary.get("retries_used", 0),
        "malformed_calls": summary.get("malformed_calls", 0),
        "loop_extended": summary.get("loop_extended", False),
        "blocked_repeats": summary.get("blocked_repeats", 0),
        "forced_submit": summary.get("forced_submit", False),
        "abstained": "unanswerable" in answer,
        "submitted": bool(answer),
    }
    return [float(values[f]) for f in FEATURES]


class ConfidenceModel:
    def __init__(self, path: Path = MODEL_PATH) -> None:
        self.spec = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None

    @property
    def available(self) -> bool:
        return self.spec is not None

    def predict(self, summary: dict[str, Any]) -> float | None:
        if not self.spec:
            return None
        x = np.array(features(summary))
        z = float(np.dot(self.spec["coef"], x) + self.spec["intercept"])
        return round(1.0 / (1.0 + np.exp(-z)), 3)


def calibration(p: np.ndarray, y: np.ndarray, bins: int = 5) -> dict[str, Any]:
    edges = np.linspace(0, 1, bins + 1)
    ece, table = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
            table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "mean_confidence": round(float(p[m].mean()), 3),
                          "accuracy": round(float(y[m].mean()), 3)})
    return {"brier": round(float(np.mean((p - y) ** 2)), 4), "ece": round(float(ece), 4), "bins": table}


def train(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], version: str) -> dict[str, Any]:
    x = np.array([features(r) for r in train_rows])
    y = np.array([float(r["correct"]) for r in train_rows])
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(x, y)
    spec = {"features": list(FEATURES), "coef": [round(float(c), 6) for c in clf.coef_[0]],
            "intercept": round(float(clf.intercept_[0]), 6), "trained_on": f"{version} runs on the train split",
            "n_train": len(train_rows), "train_success_rate": round(float(y.mean()), 4)}
    model = ConfidenceModel.__new__(ConfidenceModel)
    model.spec = spec
    pv = np.array([model.predict(r) for r in val_rows])
    yv = np.array([float(r["correct"]) for r in val_rows])
    spec["validation"] = {"n": len(val_rows), **calibration(pv, yv),
                          "base_rate_brier": round(float(np.mean((yv.mean() - yv) ** 2)), 4)}
    return spec


def main(argv: list[str] | None = None) -> None:
    from code.evaluation.evaluate import rows_for
    from code.registry.registry import Registry

    os.environ.setdefault("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--version", default="production")
    args = p.parse_args(argv)
    version, _ = Registry().resolve(args.version)
    spec = train(rows_for(args.version, "train"), rows_for(args.version, "validation"), version)
    MODEL_PATH.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: spec[k] for k in ("n_train", "train_success_rate", "validation")}, indent=2))


if __name__ == "__main__":
    main()

"""Live monitor used by the deployed service (Part D/E).

Every answered request updates the label-free signals; the two monitored signals
feed CUSUM and Page-Hinkley detectors with the thresholds chosen on validation
(configs/monitoring_thresholds.json). Detectors start from the validation
reference statistics stored in that file, so the service can raise an alert
from its first requests. Labelled quality is tracked only for requests whose
answer can be graded (bundled benchmark tasks).
"""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from code.agent.config import REPO_ROOT
from code.monitoring.detectors import Reference, make_detector
from code.monitoring.signals import LABEL_FREE, signals

THRESHOLDS = REPO_ROOT / "configs" / "monitoring_thresholds.json"
WINDOW = 25
#: The first requests after start-up include loading the model into memory, which
#: would look like a latency drift; they are tracked but not fed to the detectors.
WARMUP = 2


class LiveMonitor:
    def __init__(self, thresholds_path: Path = THRESHOLDS, window: int = WINDOW) -> None:
        path = Path(thresholds_path)
        if path.exists():
            self.thresholds = json.loads(path.read_text(encoding="utf-8"))
        else:  # thresholds not chosen yet: track signals, no drift detection
            self.thresholds = {"monitored_signals": [], "signals": {},
                               "reference": {s: {"mean": 0.0, "std": 1.0} for s in LABEL_FREE},
                               "missing": str(path)}
        self.window = window
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.history: deque[dict[str, float]] = deque(maxlen=500)
        self.quality: deque[float] = deque(maxlen=500)
        self.alerts: list[dict[str, Any]] = []
        self.n = 0
        self.detectors = {}
        for signal in self.thresholds["monitored_signals"]:
            ref = Reference(**self.thresholds["reference"][signal])
            for kind, p in self.thresholds["signals"][signal].items():
                self.detectors[(signal, kind)] = make_detector(kind, ref, p["slack"], p["threshold"])

    def update(self, summary: dict[str, Any], graded: bool) -> list[dict[str, Any]]:
        """Add one request; return any alerts it raised."""
        sig = signals(summary)
        new = []
        with self.lock:
            self.history.append(sig)
            if graded:
                self.quality.append(sig["correct"])
            for (signal, kind), det in self.detectors.items():
                if self.n < WARMUP:
                    continue
                if det.update(sig[signal]):
                    alert = {"request": self.n, "signal": signal, "detector": kind, "value": sig[signal]}
                    self.alerts.append(alert)
                    new.append(alert)
            self.n += 1
        return new

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            recent = list(self.history)[-self.window:]
            q = list(self.quality)[-self.window:]
            return {
                "requests": self.n,
                "window": len(recent),
                "rolling": {s: round(float(np.mean([r[s] for r in recent])), 3) if recent else None for s in LABEL_FREE},
                "reference": {s: round(self.thresholds["reference"][s]["mean"], 3) for s in LABEL_FREE},
                "rolling_success_graded": round(float(np.mean(q)), 3) if q else None,
                "monitored_signals": self.thresholds["monitored_signals"],
                "detector_state": {f"{s}/{k}": round(d.statistic, 2) for (s, k), d in self.detectors.items()},
                "alerts": list(self.alerts[-10:]),
                "drift": bool(self.alerts),
                "thresholds_loaded": "missing" not in self.thresholds,
            }

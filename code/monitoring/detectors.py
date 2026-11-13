"""Sequential change detectors for the label-free monitoring signals (Part E).

Both detectors watch for an *upward* shift (more tool failures, more steps,
higher latency) relative to a reference mean and spread estimated on
in-distribution traffic. Observations are standardised with that reference,
so one threshold scale works for every signal.

CUSUM (one-sided):      S_t = max(0, S_{t-1} + z_t - k);  alarm when S_t > h
Page-Hinkley (upward):  m_t = sum_{i<=t} (z_i - delta);  alarm when m_t - min_{i<=t} m_i > lambda

After an alarm a detector resets, so repeated alarms on persistent drift are
counted separately from the first (the detection delay uses the first alarm).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class Reference:
    mean: float
    std: float

    @classmethod
    def fit(cls, values: Iterable[float], min_std: float = 1e-3) -> "Reference":
        arr = np.asarray(list(values), dtype=float)
        return cls(float(arr.mean()), max(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, min_std))

    def z(self, x: float) -> float:
        return (x - self.mean) / self.std


@dataclass
class Cusum:
    ref: Reference
    k: float = 0.5
    h: float = 5.0
    s: float = 0.0
    alarms: list[int] = field(default_factory=list)
    t: int = 0

    def update(self, x: float) -> bool:
        self.s = max(0.0, self.s + self.ref.z(x) - self.k)
        self.t += 1
        if self.s > self.h:
            self.alarms.append(self.t - 1)
            self.s = 0.0
            return True
        return False

    @property
    def statistic(self) -> float:
        return self.s


@dataclass
class PageHinkley:
    ref: Reference
    delta: float = 0.5
    lam: float = 5.0
    m: float = 0.0
    m_min: float = 0.0
    alarms: list[int] = field(default_factory=list)
    t: int = 0

    def update(self, x: float) -> bool:
        self.m += self.ref.z(x) - self.delta
        self.m_min = min(self.m_min, self.m)
        self.t += 1
        if self.m - self.m_min > self.lam:
            self.alarms.append(self.t - 1)
            self.m = self.m_min = 0.0
            return True
        return False

    @property
    def statistic(self) -> float:
        return self.m - self.m_min


def make_detector(kind: str, ref: Reference, slack: float, threshold: float) -> Cusum | PageHinkley:
    if kind == "cusum":
        return Cusum(ref, k=slack, h=threshold)
    if kind == "page_hinkley":
        return PageHinkley(ref, delta=slack, lam=threshold)
    raise ValueError(kind)


def run_detector(kind: str, ref: Reference, slack: float, threshold: float, values: Iterable[float]) -> list[int]:
    det = make_detector(kind, ref, slack, threshold)
    for x in values:
        det.update(x)
    return det.alarms

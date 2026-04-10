from __future__ import annotations

from typing import Iterable

import numpy as np

from src.monitoring.drift.base import DriftDetector


class ZScoreDriftDetector(DriftDetector):
    def __init__(self, threshold: float):
        self.threshold = float(threshold)
        self._baseline_mean = 0.0
        self._baseline_std = 1.0
        self._fitted = False

    def fit_baseline(self, data: Iterable[float]) -> None:
        arr = np.asarray(list(data), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            self._baseline_mean = 0.0
            self._baseline_std = 1.0
            self._fitted = False
            return
        self._baseline_mean = float(np.mean(arr))
        self._baseline_std = max(float(np.std(arr)), 1e-9)
        self._fitted = True

    def score(self, new_data: Iterable[float]) -> float:
        arr = np.asarray(list(new_data), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.0
        if not self._fitted:
            return 0.0
        shift = float(abs(np.mean(arr) - self._baseline_mean))
        return float(shift / self._baseline_std)

    def status(self, score: float) -> str:
        if score >= self.threshold:
            return "alert"
        if score >= 0.7 * self.threshold:
            return "warning"
        return "ok"

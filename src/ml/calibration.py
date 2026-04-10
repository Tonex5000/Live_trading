from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class CalibrationConfig:
    method: str = "none"  # none | platt | isotonic


class ProbabilityCalibrator:
    def __init__(self, method: str = "none", model=None):
        self.method = method
        self.model = model

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        raw_probs = np.asarray(raw_probs).reshape(-1)
        y_true = np.asarray(y_true).reshape(-1)

        if self.method == "none":
            self.model = None
            return self

        if self.method == "platt":
            lr = LogisticRegression(solver="lbfgs")
            lr.fit(raw_probs.reshape(-1, 1), y_true)
            self.model = lr
            return self

        if self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_probs, y_true)
            self.model = iso
            return self

        raise ValueError(f"Unsupported calibration method: {self.method}")

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        raw_probs = np.asarray(raw_probs).reshape(-1)
        if self.method == "none" or self.model is None:
            return raw_probs
        if self.method == "platt":
            return self.model.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            return self.model.predict(raw_probs)
        raise ValueError(f"Unsupported calibration method: {self.method}")

    def save(self, path: str) -> None:
        payload = {"method": self.method, "model": self.model}
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "ProbabilityCalibrator":
        p = Path(path)
        if not p.exists():
            return cls(method="none", model=None)
        payload = joblib.load(path)
        return cls(method=payload.get("method", "none"), model=payload.get("model"))


def load_calibration_method(config_path: str, default: str = "none") -> str:
    p = Path(config_path)
    if not p.exists():
        return default
    data = json.loads(p.read_text())
    return data.get("calibration_method", default)

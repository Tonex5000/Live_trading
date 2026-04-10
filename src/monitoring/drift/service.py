from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional

from sqlmodel import select

from src.db import session_scope
from src.models import DriftEvent
from src.monitoring.drift.baseline_store import InMemoryBaselineStore
from src.monitoring.drift.detectors import ZScoreDriftDetector


@dataclass
class DriftObservation:
    timestamp: datetime
    symbol: str
    feature_values: List[float]
    signal_values: List[float]
    execution_values: List[float]


class DriftService:
    def __init__(
        self,
        *,
        feature_threshold: float,
        signal_threshold: float,
        execution_threshold: float,
        baseline_window: int = 300,
        min_samples: int = 30,
    ):
        self.store = InMemoryBaselineStore(maxlen=baseline_window)
        self.min_samples = max(10, int(min_samples))
        self.detectors = {
            "feature": ZScoreDriftDetector(feature_threshold),
            "signal": ZScoreDriftDetector(signal_threshold),
            "execution": ZScoreDriftDetector(execution_threshold),
        }
        self._latest: Dict[str, Dict] = {}

    def _dimension_values(self, obs: DriftObservation, dimension: str) -> List[float]:
        if dimension == "feature":
            return obs.feature_values
        if dimension == "signal":
            return obs.signal_values
        return obs.execution_values

    def _persist_event(self, *, obs: DriftObservation, dimension: str, score: float, status: str, baseline_count: int) -> None:
        meta = {
            "baseline_count": baseline_count,
            "feature_points": len(obs.feature_values),
            "signal_points": len(obs.signal_values),
            "execution_points": len(obs.execution_values),
        }
        with session_scope() as session:
            session.add(
                DriftEvent(
                    timestamp=obs.timestamp,
                    symbol=obs.symbol,
                    dimension=dimension,
                    score=float(score),
                    status=status,
                    metadata_json=json.dumps(meta),
                )
            )

    def process(self, observation: DriftObservation) -> Dict[str, Dict]:
        result: Dict[str, Dict] = {}

        for dimension, detector in self.detectors.items():
            vals = self._dimension_values(observation, dimension)
            clean_vals = [float(v) for v in vals if v is not None]
            if not clean_vals:
                continue

            baseline_count = self.store.count(observation.symbol, dimension)
            baseline_vals = self.store.values(observation.symbol, dimension)

            if baseline_count >= self.min_samples:
                detector.fit_baseline(baseline_vals)
                score = detector.score(clean_vals)
                status = detector.status(score)
            else:
                score = 0.0
                status = "warming_up"

            self._persist_event(
                obs=observation,
                dimension=dimension,
                score=score,
                status=status,
                baseline_count=baseline_count,
            )

            self.store.add(observation.symbol, dimension, clean_vals)
            result[dimension] = {"score": float(score), "status": status}

        self._latest[observation.symbol] = {
            "timestamp": observation.timestamp.isoformat(),
            "symbol": observation.symbol,
            "dimensions": result,
        }
        return result

    def latest(self, symbol: Optional[str] = None) -> Dict:
        if symbol:
            return self._latest.get(symbol, {})
        if not self._latest:
            return {}
        _, latest_value = sorted(self._latest.items(), key=lambda kv: kv[1].get("timestamp", ""))[-1]
        return latest_value

    def events(self, *, symbol: Optional[str] = None, dimension: Optional[str] = None, limit: int = 100) -> List[DriftEvent]:
        with session_scope() as session:
            stmt = select(DriftEvent)
            if symbol:
                stmt = stmt.where(DriftEvent.symbol == symbol)
            if dimension:
                stmt = stmt.where(DriftEvent.dimension == dimension)
            stmt = stmt.order_by(DriftEvent.timestamp.desc()).limit(limit)
            return list(session.exec(stmt).all())

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from sqlmodel import select

from src.db import session_scope
from src.models import DecisionEvent, ModelPerformanceSnapshot, Trade
from src.monitoring.performance.metrics import compute_rejection_rate, compute_trade_metrics, detect_degradation


class PerformanceService:
    def __init__(self, *, model_version: str, window_size: int, snapshot_interval_bars: int = 1):
        self.model_version = model_version
        self.window_size = max(20, int(window_size))
        self.snapshot_interval_bars = max(1, int(snapshot_interval_bars))
        self._bar_counter = 0

    def _load_window(self) -> tuple[List[Trade], List[DecisionEvent]]:
        with session_scope() as session:
            trades = list(session.exec(select(Trade).order_by(Trade.closed_at.desc()).limit(self.window_size)).all())
            decisions = list(session.exec(select(DecisionEvent).order_by(DecisionEvent.created_at.desc()).limit(self.window_size)).all())
        return trades, decisions

    def _build_snapshot_payload(self, trades: List[Trade], decisions: List[DecisionEvent]) -> Dict:
        trade_metrics = compute_trade_metrics(trades)
        rejection_rate = compute_rejection_rate(decisions)
        health = detect_degradation(
            expectancy=float(trade_metrics["expectancy"]),
            ev_realization_ratio=float(trade_metrics["ev_realization_ratio"]),
        )

        return {
            "timestamp": datetime.utcnow(),
            "model_version": self.model_version,
            "window_size": self.window_size,
            "total_trades": int(trade_metrics["total_trades"]),
            "win_rate": float(trade_metrics["win_rate"]),
            "avg_win": float(trade_metrics["avg_win"]),
            "avg_loss": float(trade_metrics["avg_loss"]),
            "expectancy": float(trade_metrics["expectancy"]),
            "total_pnl": float(trade_metrics["total_pnl"]),
            "rejection_rate": float(rejection_rate),
            "ev_realization_ratio": float(trade_metrics["ev_realization_ratio"]),
            "status": health,
        }

    def update(self) -> Optional[Dict]:
        self._bar_counter += 1
        if self._bar_counter % self.snapshot_interval_bars != 0:
            return None

        trades, decisions = self._load_window()
        payload = self._build_snapshot_payload(trades, decisions)

        with session_scope() as session:
            session.add(
                ModelPerformanceSnapshot(
                    timestamp=payload["timestamp"],
                    model_version=payload["model_version"],
                    window_size=payload["window_size"],
                    total_trades=payload["total_trades"],
                    win_rate=payload["win_rate"],
                    avg_win=payload["avg_win"],
                    avg_loss=payload["avg_loss"],
                    expectancy=payload["expectancy"],
                    total_pnl=payload["total_pnl"],
                    rejection_rate=payload["rejection_rate"],
                    ev_realization_ratio=payload["ev_realization_ratio"],
                    status=payload["status"],
                )
            )
        return payload

    def latest(self, model_version: Optional[str] = None) -> Optional[ModelPerformanceSnapshot]:
        with session_scope() as session:
            stmt = select(ModelPerformanceSnapshot)
            if model_version:
                stmt = stmt.where(ModelPerformanceSnapshot.model_version == model_version)
            stmt = stmt.order_by(ModelPerformanceSnapshot.timestamp.desc()).limit(1)
            return session.exec(stmt).first()

    def history(self, model_version: Optional[str] = None, limit: int = 100) -> List[ModelPerformanceSnapshot]:
        with session_scope() as session:
            stmt = select(ModelPerformanceSnapshot)
            if model_version:
                stmt = stmt.where(ModelPerformanceSnapshot.model_version == model_version)
            stmt = stmt.order_by(ModelPerformanceSnapshot.timestamp.desc()).limit(limit)
            return list(session.exec(stmt).all())

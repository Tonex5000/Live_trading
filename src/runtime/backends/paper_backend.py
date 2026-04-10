from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import pandas as pd
from sqlmodel import select

from src.db import session_scope
from src.execution import PaperExecutor
from src.features import add_indicators
from src.models import DecisionEvent, EquitySnapshot, Position, Signal, Trade
from src.paper_engine import PaperTradingEngine
from src.risk import RiskDecision, RiskManager
from src.runtime.backends.base import RuntimeBackend
from src.signals import FeatureResolutionError
from src.strategies.signal_schema import SignalAction, StrategySignal


class PaperBackend(RuntimeBackend):
    def __init__(
        self,
        *,
        symbol: str,
        timeframe: str,
        strategy,
        risk_manager: RiskManager,
        paper_engine: PaperTradingEngine,
        executor: PaperExecutor,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.paper_engine = paper_engine
        self.executor = executor
        self._latest_signal: Dict[str, Any] | None = None
        self._last_drift_observation: Dict[str, Any] | None = None

    def startup(self) -> None:
        return None

    @staticmethod
    def _signal_to_int(action: str) -> int:
        if action == "BUY":
            return 1
        if action == "SELL":
            return -1
        return 0

    def _persist_decision_event(self, signal_obj: StrategySignal, risk: RiskDecision) -> None:
        with session_scope() as session:
            session.add(
                DecisionEvent(
                    symbol=signal_obj.symbol,
                    timeframe=signal_obj.timeframe,
                    timestamp=signal_obj.timestamp,
                    action=signal_obj.action.value,
                    approved=risk.approved,
                    reason_code=risk.reason,
                    explanation=risk.explanation,
                    regime=signal_obj.regime,
                    drawdown=risk.drawdown,
                    confidence=risk.confidence,
                    raw_probability=signal_obj.raw_probability,
                    calibrated_probability=signal_obj.calibrated_probability,
                    probability=signal_obj.probability,
                    expected_value=risk.expected_value,
                    expected_rr=risk.expected_rr,
                    effective_risk_pct=risk.effective_risk_pct,
                    dynamic_risk_pct=risk.dynamic_risk_pct,
                    position_size=risk.position_size,
                    stop_loss=risk.stop_loss or 0.0,
                    take_profit=risk.take_profit or 0.0,
                    estimated_cost=risk.estimated_cost,
                )
            )

    def _set_feature_mismatch_signal(self, latest_row: pd.Series, err: Exception, buffer_size: int) -> None:
        signal_obj = StrategySignal(
            symbol=self.symbol,
            timeframe=self.timeframe,
            action=SignalAction.HOLD,
            confidence=0.0,
            probability=0.0,
            reason="feature_mismatch",
            timestamp=latest_row["timestamp"].isoformat(),
            raw_probability=0.0,
            calibrated_probability=0.0,
            regime="unknown",
        )
        risk = RiskDecision(
            approved=False,
            reason="feature_mismatch",
            explanation=str(err),
            confidence=0.0,
            regime="unknown",
        )
        self._persist_decision_event(signal_obj, risk)
        self._latest_signal = {
            "timestamp": signal_obj.timestamp,
            "signal": 0,
            "action": signal_obj.action.value,
            "raw_probability": 0.0,
            "calibrated_probability": 0.0,
            "probability": signal_obj.probability,
            "confidence": signal_obj.confidence,
            "regime": signal_obj.regime,
            "drawdown": risk.drawdown,
            "position_size": 0.0,
            "risk_reason": risk.reason,
            "risk_explanation": risk.explanation,
            "expected_value": 0.0,
            "expected_rr": 0.0,
            "effective_risk_pct": 0.0,
            "dynamic_risk_pct": 0.0,
            "execution_reason": "trade_rejected",
            "buffer_size": buffer_size,
        }
        self._last_drift_observation = {
            "timestamp": datetime.utcnow(),
            "symbol": self.symbol,
            "feature_values": [float(latest_row.get("atr", 0.0)), float(latest_row.get("rsi", 50.0))],
            "signal_values": [0.0],
            "execution_values": [0.0],
        }

    def on_new_bar(self, df_window: pd.DataFrame) -> None:
        df = add_indicators(df_window.copy()).dropna().reset_index(drop=True)
        if len(df) == 0:
            return

        latest_row = df.iloc[-1]
        atr_mean = float(df["atr"].rolling(50, min_periods=1).mean().iloc[-1])

        try:
            signal_obj = self.strategy.generate(df)
        except FeatureResolutionError as err:
            self._set_feature_mismatch_signal(latest_row, err, len(df_window))
            return

        latest_price = float(latest_row["close"])
        atr = float(latest_row["atr"])

        open_positions = self.paper_engine.open_positions()
        account_balance = self.paper_engine.account_balance()
        has_symbol_position = self.paper_engine.has_open_position_for_symbol(self.symbol)
        last_trade_at = self.paper_engine.last_trade_time(self.symbol)
        dd_stats = self.paper_engine.get_drawdown_stats()
        open_risk_pct = self.paper_engine.open_risk_exposure_pct()

        risk = self.risk_manager.evaluate(
            signal=signal_obj,
            latest_price=latest_price,
            atr=atr,
            account_balance=account_balance,
            open_positions_count=len(open_positions),
            has_open_position_for_symbol=has_symbol_position,
            last_trade_at=last_trade_at,
            now=datetime.utcnow(),
            atr_mean=atr_mean,
            signal_timestamp=signal_obj.timestamp,
            current_drawdown=dd_stats.get("current_drawdown", 0.0),
            open_risk_exposure_pct=open_risk_pct,
        )

        execution = self.executor.execute(
            signal=signal_obj,
            risk=risk,
            market_price=latest_price,
            timestamp=signal_obj.timestamp,
        )

        self.paper_engine.process_exits(
            {self.symbol: latest_price},
            current_bars={
                self.symbol: {
                    "high": float(latest_row["high"]),
                    "low": float(latest_row["low"]),
                    "close": latest_price,
                }
            },
        )
        self.paper_engine.mark_to_market({self.symbol: latest_price})

        self._persist_decision_event(signal_obj, risk)

        self._latest_signal = {
            "timestamp": signal_obj.timestamp,
            "signal": self._signal_to_int(signal_obj.action.value),
            "action": signal_obj.action.value,
            "raw_probability": signal_obj.raw_probability,
            "calibrated_probability": signal_obj.calibrated_probability,
            "probability": signal_obj.probability,
            "confidence": risk.confidence,
            "regime": signal_obj.regime,
            "drawdown": risk.drawdown,
            "position_size": risk.position_size if risk.approved else 0.0,
            "risk_reason": risk.reason,
            "risk_explanation": risk.explanation,
            "expected_value": risk.expected_value,
            "expected_rr": risk.expected_rr,
            "effective_risk_pct": risk.effective_risk_pct,
            "dynamic_risk_pct": risk.dynamic_risk_pct,
            "execution_reason": execution.reason,
            "buffer_size": len(df_window),
        }
        self._last_drift_observation = {
            "timestamp": datetime.utcnow(),
            "symbol": self.symbol,
            "feature_values": [
                float(latest_row.get("atr", 0.0)),
                float(latest_row.get("rsi", 50.0)),
                float(latest_row.get("adx", 0.0)),
            ],
            "signal_values": [float(signal_obj.probability), float(signal_obj.confidence)],
            "execution_values": [float(risk.estimated_cost), float(risk.expected_value)],
        }

        with session_scope() as session:
            session.add(
                Signal(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    timestamp=signal_obj.timestamp,
                    signal=self._signal_to_int(signal_obj.action.value),
                    raw_probability=signal_obj.raw_probability,
                    calibrated_probability=signal_obj.calibrated_probability,
                    probability=signal_obj.probability,
                    confidence=risk.confidence,
                    regime=signal_obj.regime,
                    drawdown=risk.drawdown,
                    position_size=risk.position_size if risk.approved else 0.0,
                    buffer_size=len(df_window),
                    reason=f"strategy={signal_obj.reason};risk={risk.reason};exec={execution.reason}",
                    reason_code=risk.reason,
                    explanation=risk.explanation,
                    expected_value=risk.expected_value,
                    expected_rr=risk.expected_rr,
                    effective_risk_pct=risk.effective_risk_pct,
                    dynamic_risk_pct=risk.dynamic_risk_pct,
                    estimated_cost=risk.estimated_cost,
                )
            )

    def latest_signal(self) -> Dict[str, Any] | None:
        return self._latest_signal

    def health(self, buffer_size: int) -> Dict[str, Any]:
        return {"status": "ok", "buffer_size": buffer_size, "mode": "paper"}

    def metrics(self) -> Dict[str, Any]:
        with session_scope() as session:
            open_count = len(session.exec(select(Position).where(Position.status == "OPEN")).all())
            trades = list(session.exec(select(Trade).order_by(Trade.closed_at.desc()).limit(1000)).all())
            rejected = list(session.exec(select(DecisionEvent).where(DecisionEvent.approved == False).limit(1000)).all())

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in trades)
        win_rate = (len(wins) / len(trades)) if trades else 0.0

        return {
            "mode": "paper",
            "open_positions": open_count,
            "total_trades": len(trades),
            "total_rejections": len(rejected),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": (sum(t.pnl for t in wins) / len(wins)) if wins else 0.0,
            "avg_loss": (sum(t.pnl for t in losses) / len(losses)) if losses else 0.0,
        }

    def shutdown(self) -> None:
        return None

    def drift_observation(self) -> Dict[str, Any] | None:
        return self._last_drift_observation

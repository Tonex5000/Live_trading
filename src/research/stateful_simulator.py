from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.risk import RiskConfig, RiskManager
from src.strategies.signal_schema import SignalAction


@dataclass
class SimPosition:
    symbol: str
    side: str
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    stop_distance: float
    take_profit_distance: float
    entry_time: datetime


class StatefulBacktestEngine:
    def __init__(self, risk_manager: RiskManager, risk_config: RiskConfig, initial_capital: float = 10_000.0):
        self.risk_manager = risk_manager
        self.risk_config = risk_config
        self.initial_capital = initial_capital

    @staticmethod
    def _to_dt(value) -> datetime:
        ts = pd.Timestamp(value)
        if ts.tzinfo is not None:
            return ts.tz_convert(None).to_pydatetime()
        return ts.to_pydatetime()

    def _entry_fill(self, action: SignalAction, market_price: float) -> float:
        if action == SignalAction.BUY:
            return market_price * (1 + self.risk_config.slippage_rate)
        return market_price * (1 - self.risk_config.slippage_rate)

    def _exit_fill_for_side(self, side: str, exit_price: float) -> float:
        if side == SignalAction.BUY.value:
            return exit_price * (1 - self.risk_config.slippage_rate)
        return exit_price * (1 + self.risk_config.slippage_rate)

    def _close_position(self, position: SimPosition, exit_price: float, exit_reason: str, when: datetime) -> Dict[str, float]:
        exit_fill = self._exit_fill_for_side(position.side, exit_price)
        entry_notional = abs(position.entry_price * position.size)
        exit_notional = abs(exit_fill * position.size)
        fees = (entry_notional + exit_notional) * self.risk_config.fee_rate

        if position.side == SignalAction.BUY.value:
            gross = (exit_fill - position.entry_price) * position.size
        else:
            gross = (position.entry_price - exit_fill) * position.size

        pnl = gross - fees
        return {
            "pnl": float(pnl),
            "fees": float(fees),
            "entry_price": float(position.entry_price),
            "exit_price": float(exit_fill),
            "size": float(position.size),
            "opened_at": position.entry_time.isoformat(),
            "closed_at": when.isoformat(),
            "holding_minutes": float((when - position.entry_time).total_seconds() / 60.0),
            "exit_reason": exit_reason,
            "side": position.side,
        }

    def _evaluate_exit(self, position: SimPosition, high: float, low: float) -> Optional[Dict[str, float]]:
        if position.side == SignalAction.BUY.value:
            stop_hit = low <= position.stop_loss
            target_hit = high >= position.take_profit
            if stop_hit and target_hit:
                return {"price": position.stop_loss, "reason": "stop_and_target_same_bar_stop_first"}
            if stop_hit:
                return {"price": position.stop_loss, "reason": "stop_loss"}
            if target_hit:
                return {"price": position.take_profit, "reason": "take_profit"}
            return None

        stop_hit = high >= position.stop_loss
        target_hit = low <= position.take_profit
        if stop_hit and target_hit:
            return {"price": position.stop_loss, "reason": "stop_and_target_same_bar_stop_first"}
        if stop_hit:
            return {"price": position.stop_loss, "reason": "stop_loss"}
        if target_hit:
            return {"price": position.take_profit, "reason": "take_profit"}
        return None

    def run(self, df: pd.DataFrame, strategy, symbol: str = "BTC/USDT:USDT") -> Dict:
        equity = float(self.initial_capital)
        peak_equity = float(self.initial_capital)
        equity_curve: List[float] = [equity]

        open_position: Optional[SimPosition] = None
        last_trade_at: Optional[datetime] = None

        trades: List[Dict] = []
        rejection_counts: Dict[str, int] = {}
        regime_counts: Dict[str, int] = {}
        lifecycle_events: List[str] = []

        time_in_market_bars = 0
        max_open_positions_seen = 0

        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0

        for i in range(1, len(df)):
            row = df.iloc[i]
            ts = self._to_dt(row["timestamp"])
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])

            # Step 1: update open position / exits
            if open_position is not None:
                time_in_market_bars += 1
                exit_decision = self._evaluate_exit(open_position, high=high, low=low)
                if exit_decision is not None:
                    closed = self._close_position(open_position, float(exit_decision["price"]), str(exit_decision["reason"]), ts)
                    prev_equity = equity
                    equity += closed["pnl"]
                    peak_equity = max(peak_equity, equity)
                    closed["return"] = (closed["pnl"] / prev_equity) if prev_equity > 0 else 0.0
                    trades.append(closed)
                    lifecycle_events.append("exit")
                    last_trade_at = ts
                    if closed["pnl"] > 0:
                        win_streak += 1
                        loss_streak = 0
                    else:
                        loss_streak += 1
                        win_streak = 0
                    max_win_streak = max(max_win_streak, win_streak)
                    max_loss_streak = max(max_loss_streak, loss_streak)
                    open_position = None

            # Step 2: equity snapshot
            equity_curve.append(equity)

            # Step 3-5: compute signal -> risk -> open trade
            hist = df.iloc[: i + 1]
            signal = strategy.generate(hist)
            regime_counts[signal.regime] = regime_counts.get(signal.regime, 0) + 1

            atr_mean = float(hist["atr"].rolling(50, min_periods=1).mean().iloc[-1]) if "atr" in hist.columns else float(row["atr"])
            current_drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            open_positions_count = 1 if open_position is not None else 0
            open_risk_exposure_pct = (
                (open_position.size * open_position.stop_distance) / equity
                if open_position is not None and equity > 0
                else 0.0
            )

            decision = self.risk_manager.evaluate(
                signal=signal,
                latest_price=close,
                atr=float(row["atr"]),
                account_balance=equity,
                open_positions_count=open_positions_count,
                has_open_position_for_symbol=open_position is not None,
                last_trade_at=last_trade_at,
                now=ts,
                atr_mean=atr_mean,
                signal_timestamp=signal.timestamp,
                current_drawdown=current_drawdown,
                open_risk_exposure_pct=open_risk_exposure_pct,
            )

            if not decision.approved:
                rejection_counts[decision.reason] = rejection_counts.get(decision.reason, 0) + 1
                continue

            if open_position is None:
                entry_fill = self._entry_fill(signal.action, close)
                open_position = SimPosition(
                    symbol=symbol,
                    side=signal.action.value,
                    entry_price=float(entry_fill),
                    size=float(decision.position_size),
                    stop_loss=float(decision.stop_loss),
                    take_profit=float(decision.take_profit),
                    stop_distance=float(decision.stop_distance),
                    take_profit_distance=float(decision.take_profit_distance),
                    entry_time=ts,
                )
                lifecycle_events.append("open")
                max_open_positions_seen = max(max_open_positions_seen, 1)

        # close any remaining position at final close for completeness
        if open_position is not None:
            final_row = df.iloc[-1]
            final_ts = self._to_dt(final_row["timestamp"])
            closed = self._close_position(open_position, float(final_row["close"]), "end_of_data_close", final_ts)
            prev_equity = equity
            equity += closed["pnl"]
            closed["return"] = (closed["pnl"] / prev_equity) if prev_equity > 0 else 0.0
            trades.append(closed)
            lifecycle_events.append("forced_exit")
            open_position = None
            equity_curve[-1] = equity

        avg_hold_minutes = float(np.mean([t["holding_minutes"] for t in trades])) if trades else 0.0
        exposure_pct = float(time_in_market_bars / max(1, len(df) - 1))

        return {
            "equity_curve": equity_curve,
            "trades": trades,
            "rejections": rejection_counts,
            "regimes": regime_counts,
            "position_metrics": {
                "time_in_market_bars": int(time_in_market_bars),
                "exposure_pct": exposure_pct,
                "avg_holding_minutes": avg_hold_minutes,
                "max_open_positions_seen": int(max_open_positions_seen),
                "max_win_streak": int(max_win_streak),
                "max_loss_streak": int(max_loss_streak),
                "lifecycle_events": lifecycle_events,
            },
        }

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from src.strategies.signal_schema import SignalAction, StrategySignal


@dataclass
class RiskConfig:
    max_risk_per_trade: float = 0.01
    max_concurrent_positions: int = 1
    cooldown_minutes: int = 30

    min_confidence: float = 0.30
    confidence_risk_floor: float = 0.50
    confidence_risk_ceiling: float = 1.00
    min_expected_value: float = 0.0

    atr_stop_mult: float = 0.8
    atr_tp_mult: float = 1.2
    min_rr: float = 1.5
    max_atr_vol_mult: float = 1.8
    atr_risk_cut_mult: float = 1.3
    atr_risk_multiplier: float = 0.7

    fee_rate: float = 0.0006
    slippage_rate: float = 0.0004

    min_qty: float = 0.001
    min_notional: float = 5.0
    qty_precision: int = 6
    price_precision: int = 2

    stale_signal_minutes: int = 120

    drawdown_threshold: float = 0.05
    drawdown_risk_multiplier: float = 0.5

    max_total_risk_exposure: float = 0.03
    max_symbol_allocation_pct: float = 1.0


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    explanation: str
    confidence: float = 0.0
    expected_value: float = 0.0
    expected_rr: float = 0.0
    effective_risk_pct: float = 0.0
    dynamic_risk_pct: float = 0.0
    position_size: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    estimated_cost: float = 0.0
    stop_distance: float = 0.0
    take_profit_distance: float = 0.0
    notional: float = 0.0
    drawdown: float = 0.0
    regime: str = "unknown"


class RiskManager:
    def __init__(self, config: RiskConfig):
        self.config = config

    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> float:
        return max(lower, min(value, upper))

    @staticmethod
    def _safe_round(value: float, digits: int) -> float:
        return round(float(value), max(0, int(digits)))

    def _reject(self, reason: str, explanation: str, signal: StrategySignal, drawdown: float = 0.0) -> RiskDecision:
        return RiskDecision(
            approved=False,
            reason=reason,
            explanation=explanation,
            confidence=float(signal.confidence),
            drawdown=float(drawdown),
            regime=signal.regime,
        )

    def evaluate(
        self,
        signal: StrategySignal,
        latest_price: float,
        atr: float,
        account_balance: float,
        open_positions_count: int,
        has_open_position_for_symbol: bool,
        last_trade_at: Optional[datetime],
        now: datetime,
        atr_mean: Optional[float] = None,
        signal_timestamp: Optional[str] = None,
        current_drawdown: float = 0.0,
        open_risk_exposure_pct: float = 0.0,
        available_cash: Optional[float] = None,
        symbol_allocation_pct: float = 0.0,
    ) -> RiskDecision:
        if signal.action == SignalAction.HOLD:
            return self._reject("hold_signal", "Signal action is HOLD; abstaining by strategy.", signal, current_drawdown)

        if open_positions_count >= self.config.max_concurrent_positions:
            return self._reject("max_positions_reached", "Max concurrent positions reached.", signal, current_drawdown)

        if has_open_position_for_symbol:
            return self._reject("duplicate_symbol_open", "An open position already exists for this symbol.", signal, current_drawdown)

        if last_trade_at is not None and now - last_trade_at < timedelta(minutes=self.config.cooldown_minutes):
            return self._reject("cooldown_active", "Cooldown period after previous trade is still active.", signal, current_drawdown)

        if signal_timestamp is not None:
            try:
                signal_ts = datetime.fromisoformat(signal_timestamp.replace("Z", "+00:00"))
                if signal_ts.tzinfo is not None:
                    signal_ts = signal_ts.astimezone(tz=None).replace(tzinfo=None)
                if now - signal_ts > timedelta(minutes=self.config.stale_signal_minutes):
                    return self._reject("stale_signal", "Signal timestamp exceeded stale threshold.", signal, current_drawdown)
            except Exception:
                return self._reject("stale_signal", "Signal timestamp parsing failed.", signal, current_drawdown)

        if latest_price <= 0 or atr <= 0 or account_balance <= 0:
            return self._reject("invalid_market_inputs", "Price/ATR/balance must be positive.", signal, current_drawdown)

        if atr_mean is not None and atr_mean > 0 and atr > atr_mean * self.config.max_atr_vol_mult:
            return self._reject("high_atr_regime", "ATR is above configured volatility safety limit.", signal, current_drawdown)

        confidence = float(signal.confidence)
        if confidence < self.config.min_confidence:
            return self._reject("low_confidence", "Signal confidence is below minimum threshold.", signal, current_drawdown)

        confidence_scale = self._clip(confidence, self.config.confidence_risk_floor, self.config.confidence_risk_ceiling)
        effective_risk_pct = self.config.max_risk_per_trade * confidence_scale

        if current_drawdown > self.config.drawdown_threshold:
            effective_risk_pct *= self.config.drawdown_risk_multiplier

        if atr_mean is not None and atr_mean > 0 and atr > atr_mean * self.config.atr_risk_cut_mult:
            effective_risk_pct *= self.config.atr_risk_multiplier

        if signal.regime == "ranging":
            effective_risk_pct *= 0.8

        stop_distance = atr * self.config.atr_stop_mult
        take_profit_distance = atr * self.config.atr_tp_mult
        if stop_distance <= 0 or take_profit_distance <= 0:
            return self._reject("invalid_stop_distance", "Stop or target distance is non-positive.", signal, current_drawdown)

        expected_rr = take_profit_distance / stop_distance if stop_distance > 0 else 0.0
        if expected_rr < self.config.min_rr:
            return self._reject("rr_below_minimum", "Expected reward-to-risk is below minimum.", signal, current_drawdown)

        risk_capital = account_balance * effective_risk_pct
        raw_size = risk_capital / stop_distance if stop_distance > 0 else 0.0
        rounded_size = self._safe_round(raw_size, self.config.qty_precision)

        if rounded_size <= 0:
            return self._reject("qty_below_minimum", "Rounded quantity is non-positive.", signal, current_drawdown)

        if rounded_size < self.config.min_qty:
            return self._reject("qty_below_minimum", "Rounded quantity is below configured minimum quantity.", signal, current_drawdown)

        notional = rounded_size * latest_price
        if notional < self.config.min_notional:
            return self._reject("notional_below_minimum", "Notional is below configured minimum notional.", signal, current_drawdown)

        if available_cash is not None and notional > available_cash:
            return self._reject("insufficient_cash", "Trade notional exceeds available cash.", signal, current_drawdown)

        if symbol_allocation_pct > self.config.max_symbol_allocation_pct:
            return self._reject("symbol_allocation_limit", "Symbol allocation exceeds configured max symbol allocation.", signal, current_drawdown)

        incremental_risk_pct = (rounded_size * stop_distance) / account_balance
        if open_risk_exposure_pct + incremental_risk_pct > self.config.max_total_risk_exposure:
            return self._reject("portfolio_risk_limit", "Total risk exposure cap would be exceeded.", signal, current_drawdown)

        if signal.action == SignalAction.BUY:
            stop_loss = latest_price - stop_distance
            take_profit = latest_price + take_profit_distance
            win_prob = float(signal.calibrated_probability or signal.probability)
        else:
            stop_loss = latest_price + stop_distance
            take_profit = latest_price - take_profit_distance
            win_prob = float(1 - (signal.calibrated_probability or signal.probability))

        stop_loss = self._safe_round(stop_loss, self.config.price_precision)
        take_profit = self._safe_round(take_profit, self.config.price_precision)

        reward = take_profit_distance
        risk = stop_distance
        p_win = win_prob
        p_loss = 1 - p_win
        estimated_cost = (latest_price * rounded_size * 2) * (self.config.fee_rate + self.config.slippage_rate)
        expected_value = ((p_win * reward) - (p_loss * risk)) * rounded_size - estimated_cost

        if expected_value <= self.config.min_expected_value:
            return self._reject("ev_below_threshold", "Expected value net of costs is below threshold.", signal, current_drawdown)

        return RiskDecision(
            approved=True,
            reason="approved",
            explanation="Trade approved by risk engine.",
            confidence=confidence,
            expected_value=float(expected_value),
            expected_rr=float(expected_rr),
            effective_risk_pct=float(effective_risk_pct),
            dynamic_risk_pct=float(effective_risk_pct),
            position_size=float(rounded_size),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            estimated_cost=float(estimated_cost),
            stop_distance=float(stop_distance),
            take_profit_distance=float(take_profit_distance),
            notional=float(notional),
            drawdown=float(current_drawdown),
            regime=signal.regime,
        )

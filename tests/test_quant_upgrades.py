from datetime import datetime

import numpy as np

from src.ml.calibration import ProbabilityCalibrator
from src.risk import RiskConfig, RiskManager
from src.strategies.regime import classify_regime
from src.strategies.signal_schema import SignalAction, StrategySignal
from src.strategies.thresholds import ThresholdConfig, load_thresholds, optimize_thresholds


def _signal(prob=0.8, confidence=0.8, regime="trending"):
    return StrategySignal(
        symbol="BTC/USDT:USDT",
        timeframe="30m",
        action=SignalAction.BUY,
        confidence=confidence,
        probability=prob,
        raw_probability=prob,
        calibrated_probability=prob,
        regime=regime,
        reason="test",
        timestamp="2026-01-01T00:00:00",
    )


def test_calibration_sanity_platt():
    raw = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0, 0, 1, 1])
    cal = ProbabilityCalibrator(method="platt").fit(raw, y)
    out = cal.predict(raw)
    assert np.all(out >= 0)
    assert np.all(out <= 1)


def test_ev_changes_after_probability_calibration():
    m = RiskManager(RiskConfig(min_rr=1.4))
    s1 = _signal(prob=0.55, confidence=0.8)
    s2 = _signal(prob=0.80, confidence=0.8)

    d1 = m.evaluate(s1, latest_price=100, atr=1, account_balance=10000, open_positions_count=0, has_open_position_for_symbol=False, last_trade_at=None, now=datetime.utcnow())
    d2 = m.evaluate(s2, latest_price=100, atr=1, account_balance=10000, open_positions_count=0, has_open_position_for_symbol=False, last_trade_at=None, now=datetime.utcnow())

    assert d2.expected_value >= d1.expected_value


def test_dynamic_thresholds_applied():
    probs = [0.2, 0.4, 0.6, 0.8]
    y = [-1, 0, 1, 1]
    t = optimize_thresholds(probs, y)
    assert t["p_buy"] > t["p_sell"]

    p_buy, p_sell = load_thresholds(ThresholdConfig(mode="fixed", p_buy=0.63, p_sell=0.37))
    assert p_buy == 0.63 and p_sell == 0.37


def test_drawdown_reduces_risk():
    m = RiskManager(RiskConfig(min_rr=1.4, drawdown_threshold=0.05, drawdown_risk_multiplier=0.5))
    base = m.evaluate(_signal(), latest_price=100, atr=1, account_balance=10000, open_positions_count=0, has_open_position_for_symbol=False, last_trade_at=None, now=datetime.utcnow(), current_drawdown=0.01)
    dd = m.evaluate(_signal(), latest_price=100, atr=1, account_balance=10000, open_positions_count=0, has_open_position_for_symbol=False, last_trade_at=None, now=datetime.utcnow(), current_drawdown=0.10)
    assert dd.effective_risk_pct < base.effective_risk_pct


def test_regime_classification():
    assert classify_regime(adx=30, strong_trend=1, atr=1, atr_mean=1) == "trending"
    assert classify_regime(adx=10, strong_trend=0, atr=1, atr_mean=1) == "ranging"
    assert classify_regime(adx=30, strong_trend=1, atr=3, atr_mean=1, atr_spike_mult=2.0) == "high_volatility"


def test_volatility_risk_adaptation():
    m = RiskManager(RiskConfig(min_rr=1.4, atr_risk_cut_mult=1.2, atr_risk_multiplier=0.5))
    normal = m.evaluate(_signal(), latest_price=100, atr=1.0, atr_mean=1.0, account_balance=10000, open_positions_count=0, has_open_position_for_symbol=False, last_trade_at=None, now=datetime.utcnow())
    high = m.evaluate(_signal(), latest_price=100, atr=1.3, atr_mean=1.0, account_balance=10000, open_positions_count=0, has_open_position_for_symbol=False, last_trade_at=None, now=datetime.utcnow())
    assert high.effective_risk_pct < normal.effective_risk_pct

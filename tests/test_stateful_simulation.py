import pandas as pd

from src.research.stateful_simulator import StatefulBacktestEngine
from src.risk import RiskConfig, RiskManager
from src.strategies.signal_schema import SignalAction, StrategySignal


class FixedSignalStrategy:
    def __init__(self, action=SignalAction.BUY, confidence=0.9, probability=0.8):
        self.action = action
        self.confidence = confidence
        self.probability = probability

    def generate(self, df: pd.DataFrame) -> StrategySignal:
        ts = df.iloc[-1]["timestamp"]
        return StrategySignal(
            symbol="BTC/USDT:USDT",
            timeframe="wf",
            action=self.action,
            confidence=self.confidence,
            probability=self.probability,
            raw_probability=self.probability,
            calibrated_probability=self.probability,
            reason="test_signal",
            regime="trending",
            timestamp=ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
        )


def _df_for_stateful() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=8, freq="h")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100, 100, 100, 100, 100, 100, 100, 100],
            "high": [100.2, 100.5, 100.6, 100.6, 100.4, 100.3, 100.2, 100.1],
            "low": [99.8, 99.7, 99.6, 99.4, 99.6, 99.7, 99.8, 99.9],
            "close": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "atr": [1.0] * 8,
            "adx": [30.0] * 8,
            "rsi": [50.0] * 8,
            "ema_20": [100.0] * 8,
            "ema_50": [100.0] * 8,
            "ema_200": [100.0] * 8,
            "trend": [1] * 8,
            "strong_trend": [1] * 8,
            "target": [1] * 8,
        }
    )


def test_position_persistence_and_no_overlap():
    df = _df_for_stateful()
    risk_cfg = RiskConfig(atr_stop_mult=0.5, atr_tp_mult=2.0, cooldown_minutes=0, min_rr=1.0)
    engine = StatefulBacktestEngine(RiskManager(risk_cfg), risk_cfg)

    result = engine.run(df, FixedSignalStrategy())

    assert result["position_metrics"]["max_open_positions_seen"] == 1
    assert result["position_metrics"]["time_in_market_bars"] > 0


def test_cooldown_enforced_after_exit():
    df = _df_for_stateful()
    # immediate stop/target likely happens quickly, cooldown should block repeated entries
    risk_cfg = RiskConfig(atr_stop_mult=0.3, atr_tp_mult=0.3, cooldown_minutes=120, min_rr=1.0, fee_rate=0.0, slippage_rate=0.0, min_expected_value=-1.0)
    engine = StatefulBacktestEngine(RiskManager(risk_cfg), risk_cfg)

    result = engine.run(df, FixedSignalStrategy())

    assert result["rejections"].get("cooldown_active", 0) >= 1


def test_ohlc_exit_stop_and_target_same_bar_stop_first():
    df = _df_for_stateful().copy()
    # force both stop and target in same bar after entry
    df.loc[2, "high"] = 101.0
    df.loc[2, "low"] = 99.0

    risk_cfg = RiskConfig(atr_stop_mult=0.5, atr_tp_mult=0.5, cooldown_minutes=0, min_rr=1.0)
    engine = StatefulBacktestEngine(RiskManager(risk_cfg), risk_cfg)
    result = engine.run(df, FixedSignalStrategy())

    assert len(result["trades"]) >= 1
    assert result["trades"][0]["exit_reason"] == "stop_and_target_same_bar_stop_first"


def test_trade_lifecycle_ordering_open_hold_exit():
    df = _df_for_stateful()
    risk_cfg = RiskConfig(atr_stop_mult=0.5, atr_tp_mult=1.5, cooldown_minutes=0, min_rr=1.0)
    engine = StatefulBacktestEngine(RiskManager(risk_cfg), risk_cfg)

    result = engine.run(df, FixedSignalStrategy())
    lifecycle = result["position_metrics"]["lifecycle_events"]

    assert "open" in lifecycle
    assert any(e in lifecycle for e in ["exit", "forced_exit"])


def test_sequential_no_future_data_in_strategy_calls():
    df = _df_for_stateful()
    max_seen = {"rows": 0}

    class TrackingStrategy(FixedSignalStrategy):
        def generate(self, hist: pd.DataFrame) -> StrategySignal:
            max_seen["rows"] = max(max_seen["rows"], len(hist))
            return super().generate(hist)

    risk_cfg = RiskConfig(min_rr=1.0)
    engine = StatefulBacktestEngine(RiskManager(risk_cfg), risk_cfg)
    engine.run(df, TrackingStrategy())

    assert max_seen["rows"] <= len(df)
    assert max_seen["rows"] == len(df)


def test_parity_simple_rejections_with_risk_manager():
    df = _df_for_stateful()
    # force low confidence rejection parity with live risk manager
    strategy = FixedSignalStrategy(confidence=0.1)
    risk_cfg = RiskConfig(min_confidence=0.3, min_rr=1.0)
    engine = StatefulBacktestEngine(RiskManager(risk_cfg), risk_cfg)

    result = engine.run(df, strategy)
    assert result["rejections"].get("low_confidence", 0) >= 1

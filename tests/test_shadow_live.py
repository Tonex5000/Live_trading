import pandas as pd

from src.research.portfolio_simulator import PortfolioSimulationConfig
from src.research.shadow_live import ShadowLiveConfig, ShadowLiveEngine
from src.risk import RiskConfig, RiskManager
from src.strategies.signal_schema import SignalAction, StrategySignal


class StepStrategy:
    def __init__(self, activate_at=2, action=SignalAction.BUY, conf=0.9, prob=0.8):
        self.activate_at = activate_at
        self.action = action
        self.conf = conf
        self.prob = prob

    def generate(self, df):
        ts = df.iloc[-1]["timestamp"]
        act = self.action if (len(df) - 1) >= self.activate_at else SignalAction.HOLD
        return StrategySignal(
            symbol="x",
            timeframe="1h",
            action=act,
            confidence=self.conf,
            probability=self.prob,
            raw_probability=self.prob,
            calibrated_probability=self.prob,
            reason="shadow",
            regime="trending",
            timestamp=ts.isoformat(),
        )


def _df():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
            "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "high": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "atr": [1.0] * 10,
            "rsi": [50.0] * 10,
            "volume": [100.0] * 10,
        }
    )


def _engine(**shadow_overrides):
    rcfg = RiskConfig(max_concurrent_positions=5, min_rr=1.0, max_risk_per_trade=0.001, fee_rate=0.0, slippage_rate=0.0, min_qty=0.000001, min_notional=1.0)
    pcfg = PortfolioSimulationConfig(max_fill_cost_bps=500.0, max_spread_bps_for_entry=500.0)
    scfg = ShadowLiveConfig(**shadow_overrides)
    return ShadowLiveEngine(RiskManager(rcfg), rcfg, pcfg, scfg)


def test_shadow_decisions_logged_without_real_orders():
    out = _engine().run({"BTC": _df()}, {"BTC": StepStrategy()})
    assert len(out["decisions"]) > 0


def test_shadow_outcomes_use_future_horizon():
    out = _engine(decision_horizon_bars=2).run({"BTC": _df()}, {"BTC": StepStrategy()})
    assert all(o["horizon_bars"] == 2 for o in out["outcomes"])


def test_incomplete_bar_not_triggered_when_require_closed():
    out = _engine(require_closed_bar=True).run({"BTC": _df()}, {"BTC": StepStrategy()})
    assert out["decisions"][-1]["timestamp"] != _df().iloc[-1]["timestamp"].isoformat()


def test_shadow_portfolio_state_updates():
    out = _engine().run({"BTC": _df()}, {"BTC": StepStrategy()})
    assert len(out["equity_curve"]) > 0


def test_divergence_metrics_populate():
    out = _engine().run({"BTC": _df()}, {"BTC": StepStrategy()})
    assert "execution_divergence" in out["summary"]


def test_drift_summaries_produced(tmp_path):
    out = _engine().run({"BTC": _df()}, {"BTC": StepStrategy()}, output_dir=str(tmp_path))
    assert (tmp_path / "feature_drift_summary.json").exists()
    assert (tmp_path / "signal_drift_summary.json").exists()


def test_readiness_state_changes_when_worse():
    out = _engine(max_allowed_divergence=0.0).run({"BTC": _df()}, {"BTC": StepStrategy()})
    assert out["summary"]["deployment_readiness"] in {"caution", "not_ready"}

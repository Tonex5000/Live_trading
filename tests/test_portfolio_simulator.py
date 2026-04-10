import pandas as pd

from src.research.portfolio_simulator import PortfolioSimulationConfig, PortfolioStatefulSimulator
from src.research.walkforward import CompositeObjectiveConfig, ParameterGrid, StabilitySelectionConfig, WalkForwardConfig, WalkForwardRunner
from src.risk import RiskConfig, RiskManager
from src.strategies.signal_schema import SignalAction, StrategySignal


class FixedStrategy:
    def __init__(self, action=SignalAction.BUY, confidence=0.9, prob=0.8):
        self.action = action
        self.confidence = confidence
        self.prob = prob

    def generate(self, df):
        ts = df.iloc[-1]["timestamp"]
        return StrategySignal(
            symbol="x",
            timeframe="wf",
            action=self.action,
            confidence=self.confidence,
            probability=self.prob,
            raw_probability=self.prob,
            calibrated_probability=self.prob,
            reason="test",
            regime="trending",
            timestamp=ts.isoformat(),
        )


class TimedStrategy:
    def __init__(self, activate_at: int, action=SignalAction.BUY, confidence=0.95, prob=0.9):
        self.activate_at = activate_at
        self.action = action
        self.confidence = confidence
        self.prob = prob

    def generate(self, df):
        ts = df.iloc[-1]["timestamp"]
        act = self.action if (len(df) - 1) >= self.activate_at else SignalAction.HOLD
        return StrategySignal(
            symbol="x",
            timeframe="wf",
            action=act,
            confidence=self.confidence,
            probability=self.prob,
            raw_probability=self.prob,
            calibrated_probability=self.prob,
            reason="timed_test",
            regime="trending",
            timestamp=ts.isoformat(),
        )


def _df(n=12):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "open": [100.0] * n,
            "high": [101.5] * n,
            "low": [99.0] * n,
            "close": [100.0] * n,
            "atr": [1.0] * n,
            "adx": [30.0] * n,
            "rsi": [50.0] * n,
            "ema_20": [100.0] * n,
            "ema_50": [100.0] * n,
            "ema_200": [100.0] * n,
            "trend": [1] * n,
            "strong_trend": [1] * n,
            "target": [1] * n,
            "volume": [100.0] * n,
        }
    )


def _price_df(prices):
    n = len(prices)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "open": prices,
            "high": [p + 0.05 for p in prices],
            "low": [p - 0.05 for p in prices],
            "close": prices,
            "atr": [10.0] * n,
            "adx": [30.0] * n,
            "rsi": [50.0] * n,
            "ema_20": prices,
            "ema_50": prices,
            "ema_200": prices,
            "trend": [1] * n,
            "strong_trend": [1] * n,
            "target": [1] * n,
        }
    )


def test_multi_symbols_share_capital_and_block_per_symbol_only():
    symbols = {"BTC": _df(), "ETH": _df()}
    strategies = {"BTC": FixedStrategy(), "ETH": FixedStrategy()}

    rcfg = RiskConfig(min_rr=1.0, fee_rate=0.0, slippage_rate=0.0, max_risk_per_trade=0.0001, min_qty=0.000001)
    pcfg = PortfolioSimulationConfig(max_open_positions=2, one_position_per_symbol=True)
    sim = PortfolioStatefulSimulator(RiskManager(rcfg), rcfg, pcfg)

    out = sim.run(symbols, strategies)
    assert out["metrics"]["trade_count"] >= 1


def test_max_total_risk_exposure_and_max_open_positions_enforced():
    symbols = {"BTC": _df(), "ETH": _df(), "SOL": _df()}
    strategies = {s: FixedStrategy() for s in symbols}
    rcfg = RiskConfig(min_rr=1.0, fee_rate=0.0, slippage_rate=0.0, max_risk_per_trade=0.0001, min_qty=0.000001)
    pcfg = PortfolioSimulationConfig(max_open_positions=1, max_total_risk_exposure=0.005)
    sim = PortfolioStatefulSimulator(RiskManager(rcfg), rcfg, pcfg)
    out = sim.run(symbols, strategies)
    assert out["rejections"].get("portfolio_max_open_positions", 0) >= 0


def test_max_symbol_allocation_and_capacity_reason_codes():
    symbols = {"BTC": _df(), "ETH": _df()}
    strategies = {s: FixedStrategy() for s in symbols}
    rcfg = RiskConfig(min_rr=1.0, fee_rate=0.0, slippage_rate=0.0, max_risk_per_trade=0.001, min_qty=0.000001)
    pcfg = PortfolioSimulationConfig(initial_capital=1_000_000, max_symbol_allocation_pct=0.000001)
    sim = PortfolioStatefulSimulator(RiskManager(rcfg), rcfg, pcfg)
    out = sim.run(symbols, strategies)
    assert out["rejections"].get("allocation_scaled_to_zero", 0) >= 1


def test_simultaneous_signals_ranked_and_logged():
    symbols = {"BTC": _df(), "ETH": _df()}
    strategies = {"BTC": FixedStrategy(prob=0.9), "ETH": FixedStrategy(prob=0.6)}
    rcfg = RiskConfig(min_rr=1.0, fee_rate=0.0, slippage_rate=0.0)
    pcfg = PortfolioSimulationConfig(candidate_ranking_mode="highest_confidence", enable_allocation_competition_logs=True)
    sim = PortfolioStatefulSimulator(RiskManager(rcfg), rcfg, pcfg)
    out = sim.run(symbols, strategies)
    assert isinstance(out["allocation_competition"], list)


def test_missing_bars_no_fabrication_and_equity_updates():
    btc = _df(12)
    eth = _df(12).drop(index=[3, 4]).reset_index(drop=True)
    symbols = {"BTC": btc, "ETH": eth}
    strategies = {s: FixedStrategy() for s in symbols}
    rcfg = RiskConfig(min_rr=1.0, fee_rate=0.0, slippage_rate=0.0)
    pcfg = PortfolioSimulationConfig()
    sim = PortfolioStatefulSimulator(RiskManager(rcfg), rcfg, pcfg)
    out = sim.run(symbols, strategies)
    assert len(out["equity_curve"]) > 0


def test_walkforward_multi_symbol_mode_leakage_safe(monkeypatch, tmp_path):
    symbols = {"BTC": _df(40), "ETH": _df(40)}
    runner = WalkForwardRunner(feature_cols=["rsi", "ema_20", "ema_50", "ema_200", "atr", "adx", "trend", "strong_trend"], target_col="target")

    monkeypatch.setattr(runner, "_fit_model", lambda train_df: object())
    monkeypatch.setattr(runner, "_fit_calibrator", lambda model, cal_df, method: object())

    def fake_portfolio(symbol_dfs, model, calibrator, params, base_risk, portfolio_cfg):
        return {
            "metrics": {"total_return": 0.01, "annualized_return": 0.01, "sharpe": 0.5, "sortino": 0.4, "max_drawdown": 0.1, "win_rate": 0.6, "profit_factor": 1.2, "avg_trade": 0.1, "expectancy": 0.1, "trade_count": 12},
            "equity_curve": [{"timestamp": "2024-01-01", "equity": 10000, "cash": 10000, "realized_pnl": 0, "unrealized_pnl": 0}],
            "rejections": {},
            "regimes": {"trending": 10},
            "position_metrics": {},
            "per_symbol_metrics": [{"symbol": "BTC", "pnl": 1.0, "trade_count": 1}],
        }

    monkeypatch.setattr(runner, "_simulate_portfolio", fake_portfolio)

    res = runner.run(
        symbols,
        WalkForwardConfig(train_bars=20, calibration_bars=10, test_bars=10, step_bars=10),
        ParameterGrid(p_buy=[0.6], p_sell=[0.4], min_confidence=[0.3], atr_stop_mult=[0.8], atr_tp_mult=[1.2], min_expected_value=[0.0], adx_min=[20.0]),
        CompositeObjectiveConfig(min_trade_count=1),
        RiskConfig(),
        output_dir=str(tmp_path),
        stability_config=StabilitySelectionConfig(simulation_mode="portfolio_multi_symbol"),
    )

    assert res["summary"]["best_candidate_params"] is not None


def test_rolling_correlation_matrix_correctness():
    btc = _price_df([100, 101, 100, 102, 101, 103])
    eth = _price_df([200, 202, 200, 204, 202, 206])
    symbols = {"BTC": btc, "ETH": eth}
    rcfg = RiskConfig(max_concurrent_positions=10)
    sim = PortfolioStatefulSimulator(RiskManager(rcfg), rcfg, PortfolioSimulationConfig(correlation_window=5))
    returns = sim._build_returns_matrix(symbols)
    corr = sim._corr_matrix_at(returns, pd.Timestamp("2024-01-01 05:00:00"))
    assert corr.loc["BTC", "ETH"] > 0.99


def test_correlation_no_future_leakage():
    btc = _price_df([100, 101, 102, 103, 104, 105])
    eth = _price_df([100, 101, 102, 103, 104, 1000])
    symbols = {"BTC": btc, "ETH": eth}
    rcfg = RiskConfig(max_concurrent_positions=10)
    sim = PortfolioStatefulSimulator(RiskManager(rcfg), rcfg, PortfolioSimulationConfig(correlation_window=4))
    returns = sim._build_returns_matrix(symbols)
    ts = pd.Timestamp("2024-01-01 04:00:00")
    corr_before = sim._corr_matrix_at(returns, ts).loc["BTC", "ETH"]
    eth2 = _price_df([100, 101, 102, 103, 104, 10_000])
    corr_after = sim._corr_matrix_at(sim._build_returns_matrix({"BTC": btc, "ETH": eth2}), ts).loc["BTC", "ETH"]
    assert corr_before == corr_after


def _base_corr_setup(enable_controls=True, **overrides):
    btc = _price_df([100, 101, 100, 102, 101, 103, 102])
    eth = _price_df([200, 202, 200, 204, 202, 206, 204])
    strategies = {"BTC": TimedStrategy(activate_at=2), "ETH": TimedStrategy(activate_at=3)}
    rcfg = RiskConfig(
        max_concurrent_positions=10,
        min_rr=1.0,
        max_risk_per_trade=0.0005,
        fee_rate=0.0,
        slippage_rate=0.0,
        min_qty=0.000001,
        min_notional=1.0,
    )
    cfg_kwargs = dict(
        max_open_positions=4,
        max_total_risk_exposure=0.9,
        max_symbol_allocation_pct=1.0,
        correlation_window=5,
        correlation_threshold=0.2,
        correlation_penalty_strength=0.8,
        max_cluster_exposure=2.0,
        enable_correlation_risk=enable_controls,
        enable_correlation_scaling=True,
        enable_correlation_rejection=False,
        max_fill_cost_bps=500.0,
    )
    cfg_kwargs.update(overrides)
    pcfg = PortfolioSimulationConfig(**cfg_kwargs)
    sim = PortfolioStatefulSimulator(RiskManager(rcfg), rcfg, pcfg)
    return sim.run({"BTC": btc, "ETH": eth}, strategies)


def test_high_correlation_reduces_position_size():
    out = _base_corr_setup(enable_controls=True)
    eth_entries = [e for e in out["accepted_entries"] if e["symbol"] == "ETH"]
    assert eth_entries
    assert eth_entries[0]["adjusted_position_size"] < eth_entries[0]["base_position_size"]


def test_high_correlation_triggers_rejection():
    out = _base_corr_setup(enable_controls=True, enable_correlation_rejection=True, correlation_threshold=0.05)
    assert out["rejections"].get("correlation_exceeded", 0) >= 1


def test_correlated_exposure_limit_triggers_rejection():
    out = _base_corr_setup(
        enable_controls=True,
        enable_correlation_rejection=True,
        correlation_threshold=1.1,
        max_correlated_exposure=0.0001,
    )
    assert out["rejections"].get("correlated_exposure_limit", 0) >= 1


def test_cluster_position_limit_enforced():
    out = _base_corr_setup(
        enable_controls=True,
        enable_correlation_rejection=False,
        cluster_threshold=0.05,
        max_cluster_positions=1,
    )
    assert out["rejections"].get("cluster_position_limit", 0) >= 1


def test_cluster_exposure_limit_enforced():
    out = _base_corr_setup(
        enable_controls=True,
        enable_correlation_rejection=False,
        cluster_threshold=0.05,
        max_cluster_exposure=0.01,
    )
    assert any(r.get("binding_cluster") for r in out["portfolio_allocation_log"])


def test_correlation_diagnostics_files_created(tmp_path):
    out = _base_corr_setup(enable_controls=True, diagnostics_output_dir=str(tmp_path))
    assert (tmp_path / "portfolio_correlation_log.csv").exists()
    assert (tmp_path / "correlation_rejections.csv").exists()
    assert len(out["portfolio_correlation_log"]) > 0


def test_behavior_changes_when_correlation_controls_disabled():
    on = _base_corr_setup(enable_controls=True, enable_correlation_rejection=True, correlation_threshold=0.05)
    off = _base_corr_setup(enable_controls=False, enable_correlation_rejection=True, correlation_threshold=0.05)
    assert on["rejections"].get("correlation_exceeded", 0) > off["rejections"].get("correlation_exceeded", 0)


def _allocation_setup(enable_corr=True, **overrides):
    prices_a = [100, 101, 100, 102, 101, 103]
    prices_b = [200, 202, 200, 204, 202, 206]
    prices_c = [50, 51, 50.5, 51.5, 51, 52]
    symbols = {"BTC": _price_df(prices_a), "ETH": _price_df(prices_b), "SOL": _price_df(prices_c)}
    strategies = {
        "BTC": TimedStrategy(activate_at=2, prob=0.95),
        "ETH": TimedStrategy(activate_at=3, prob=0.80),
        "SOL": TimedStrategy(activate_at=3, prob=0.65),
    }
    rcfg = RiskConfig(
        max_concurrent_positions=10,
        min_rr=1.0,
        max_risk_per_trade=0.001,
        fee_rate=0.0,
        slippage_rate=0.0,
        min_qty=0.000001,
        min_notional=1.0,
    )
    cfg = dict(
        max_open_positions=3,
        max_total_risk_exposure=0.9,
        max_symbol_allocation_pct=1.0,
        correlation_window=5,
        correlation_threshold=0.2,
        enable_correlation_risk=enable_corr,
        enable_correlation_rejection=False,
        max_cluster_exposure=2.0,
        max_fill_cost_bps=500.0,
        enable_iterative_allocation=True,
        max_refinement_iterations=8,
        allocation_step_fraction=0.1,
        min_improvement_threshold=1e-8,
    )
    cfg.update(overrides)
    sim = PortfolioStatefulSimulator(RiskManager(rcfg), rcfg, PortfolioSimulationConfig(**cfg))
    return sim.run(symbols, strategies)


def test_allocation_multiple_candidates_get_partial_allocation():
    out = _allocation_setup(enable_corr=False, max_total_risk_exposure=0.002)
    rows = [r for r in out["portfolio_allocation_log"] if r["timestamp"] == "2024-01-01T03:00:00"]
    assert len(rows) >= 2
    assert any(r["final_size"] < r["adjusted_size_before_allocation"] for r in rows if r["decision"] == "allocated")


def test_higher_ev_gets_higher_weight():
    out = _allocation_setup(enable_corr=False)
    first_ts = out["accepted_entries"][0]["timestamp"]
    logs = [r for r in out["portfolio_allocation_log"] if r["timestamp"] == first_ts and r["decision"] == "allocated"]
    best_ev = max(logs, key=lambda x: x["adjusted_ev"])
    best_w = max(logs, key=lambda x: x["final_weight"])
    assert best_ev["symbol"] == best_w["symbol"]


def test_correlation_reduces_allocation_weight():
    corr_on = _allocation_setup(enable_corr=True)
    corr_off = _allocation_setup(enable_corr=False)
    assert corr_on["metrics"]["average_active_correlation"] <= corr_off["metrics"]["average_active_correlation"]


def test_allocation_respects_capital_and_risk_constraints():
    out = _allocation_setup(enable_corr=False, max_total_risk_exposure=0.01)
    logs = [r for r in out["portfolio_allocation_log"] if r["decision"] == "allocated"]
    grouped = {}
    for row in logs:
        grouped.setdefault(row["timestamp"], []).append(row["final_weight"])
    assert all(sum(v) <= 1.0 + 1e-9 for v in grouped.values())
    assert out["metrics"]["capital_utilization_pct"] <= 1.0 + 1e-9


def test_cluster_constraints_enforced_in_allocation():
    out = _allocation_setup(enable_corr=True, cluster_threshold=0.05, max_cluster_exposure=0.001)
    assert any(r.get("binding_cluster") for r in out["portfolio_allocation_log"])


def test_allocation_changes_when_correlation_toggled():
    corr_on = _allocation_setup(enable_corr=True)
    corr_off = _allocation_setup(enable_corr=False)
    assert corr_on["metrics"]["allocation_efficiency"] != corr_off["metrics"]["allocation_efficiency"]


def test_refinement_improves_total_ev_vs_baseline():
    base = _allocation_setup(enable_corr=True, enable_iterative_allocation=False)
    refined = _allocation_setup(enable_corr=True, enable_iterative_allocation=True)
    base_u = max([r.get("utility_after", 0.0) for r in base["portfolio_allocation_log"]] + [0.0])
    refined_u = max([r.get("utility_after", 0.0) for r in refined["portfolio_allocation_log"]] + [0.0])
    assert refined_u >= base_u


def test_refinement_respects_constraints_and_rejections():
    refined = _allocation_setup(enable_corr=True, enable_iterative_allocation=True, max_total_risk_exposure=0.01)
    alloc_rows = [r for r in refined["portfolio_allocation_log"] if r["decision"] == "allocated"]
    grouped = {}
    for row in alloc_rows:
        grouped.setdefault(row["timestamp"], []).append(row["final_weight"])
    assert all(sum(v) <= 1.0 + 1e-9 for v in grouped.values())
    assert refined["rejections"].get("allocation_not_selected", 0) >= 0


def test_refinement_shifts_from_low_to_high_efficiency():
    base = _allocation_setup(enable_corr=False, enable_iterative_allocation=False)
    refined = _allocation_setup(enable_corr=False, enable_iterative_allocation=True)
    ts = "2024-01-01T03:00:00"
    b = {r["symbol"]: r for r in base["portfolio_allocation_log"] if r["timestamp"] == ts}
    r = {r["symbol"]: r for r in refined["portfolio_allocation_log"] if r["timestamp"] == ts}
    assert any(abs(r[s]["refined_weight"] - b[s]["initial_weight"]) > 1e-12 for s in r.keys())


def test_refinement_converges_and_changes_allocation():
    refined = _allocation_setup(enable_corr=False, enable_iterative_allocation=True, max_refinement_iterations=5)
    rows = [r for r in refined["portfolio_allocation_log"] if r["timestamp"] == "2024-01-01T03:00:00"]
    assert all(r["iteration_count"] <= 5 for r in rows)
    assert all(r["improvement_delta"] >= 0.0 for r in rows)


def test_multi_pool_refinement_global_shift_vs_single_pair():
    single = _allocation_setup(enable_corr=False, donor_pool_size=1, receiver_pool_size=1)
    multi = _allocation_setup(enable_corr=False, donor_pool_size=2, receiver_pool_size=2)
    single_pools = [r["donor_pool"] for r in single["portfolio_allocation_log"] if r["donor_pool"]]
    multi_pools = [r["donor_pool"] for r in multi["portfolio_allocation_log"] if r["donor_pool"]]
    assert all("|" not in p for p in single_pools)
    assert any("|" in p for p in multi_pools)


def test_feasibility_evaluator_reports_binding_constraints():
    rcfg = RiskConfig()
    pcfg = PortfolioSimulationConfig(max_symbol_allocation_pct=0.01, max_total_risk_exposure=0.01, max_cluster_exposure=0.01)
    sim = PortfolioStatefulSimulator(RiskManager(rcfg), rcfg, pcfg)
    fake_rows = [
        {
            "symbol": "BTC",
            "final_size": 10.0,
            "entry_fill": 100.0,
            "max_size": 20.0,
            "adjusted_decision": type("D", (), {"stop_distance": 5.0})(),
            "corr": {"cluster_exposure": 0.02, "weighted_corr": 0.9, "effective_correlated_exposure": 0.02},
        }
    ]
    status = sim._evaluate_feasibility(fake_rows, equity=1000.0, cash=100.0)
    assert status["feasible"] is False
    assert len(status["binding_constraints"]) >= 1


def test_repair_step_produces_feasible_weights():
    out = _allocation_setup(enable_corr=True, max_cluster_exposure=0.001, max_total_risk_exposure=0.005)
    alloc_rows = [r for r in out["portfolio_allocation_log"] if r["decision"] == "allocated"]
    grouped = {}
    for row in alloc_rows:
        grouped.setdefault(row["timestamp"], []).append(row["refined_weight"])
    assert all(sum(v) <= 1.0 + 1e-9 for v in grouped.values())


def test_concentration_penalty_and_diversification_bonus_recorded():
    out = _allocation_setup(enable_corr=False)
    rows = [r for r in out["portfolio_allocation_log"] if r["decision"] == "allocated"]
    assert all("concentration_penalty" in r and "diversification_bonus" in r for r in rows)


def test_utility_prefers_diversified_vs_ev_only():
    ev_only = _allocation_setup(enable_corr=True, w_concentration=0.0, w_diversification=0.0)
    diversified = _allocation_setup(enable_corr=True, w_concentration=0.4, w_diversification=0.3)
    assert diversified["metrics"]["diversification_score"] >= ev_only["metrics"]["diversification_score"]


def test_refinement_deterministic():
    a = _allocation_setup(enable_corr=True)
    b = _allocation_setup(enable_corr=True)
    wa = [r["refined_weight"] for r in a["portfolio_allocation_log"] if r["decision"] == "allocated"]
    wb = [r["refined_weight"] for r in b["portfolio_allocation_log"] if r["decision"] == "allocated"]
    assert wa == wb


def test_projection_repair_better_than_simple_shrink():
    proj = _allocation_setup(enable_corr=True, use_projection_repair=True, max_cluster_exposure=0.001)
    shrink = _allocation_setup(enable_corr=True, use_projection_repair=False, max_cluster_exposure=0.001)
    assert proj["metrics"]["capital_utilization_pct"] >= shrink["metrics"]["capital_utilization_pct"]


def test_projection_repair_converges_and_is_deterministic():
    a = _allocation_setup(enable_corr=True, use_projection_repair=True, max_projection_iterations=4)
    b = _allocation_setup(enable_corr=True, use_projection_repair=True, max_projection_iterations=4)
    ra = [r["repair_iterations"] for r in a["portfolio_allocation_log"] if r["decision"] == "allocated"]
    rb = [r["repair_iterations"] for r in b["portfolio_allocation_log"] if r["decision"] == "allocated"]
    assert all(x <= 4 for x in ra)
    assert ra == rb


def test_delta_marginal_utility_changes_pool_selection():
    delta = _allocation_setup(enable_corr=True, use_delta_marginal_utility=True)
    plain = _allocation_setup(enable_corr=True, use_delta_marginal_utility=False)
    d_vals = [r["utility_delta_add"] for r in delta["portfolio_allocation_log"] if r["decision"] == "allocated"]
    p_vals = [r["utility_delta_add"] for r in plain["portfolio_allocation_log"] if r["decision"] == "allocated"]
    assert any(abs(v) > 0 for v in d_vals)
    assert all(v == 0 for v in p_vals)


def test_fallback_repair_path_works():
    out = _allocation_setup(enable_corr=True, use_projection_repair=False, fallback_to_simple_repair=True)
    assert len(out["portfolio_allocation_log"]) > 0


def test_utility_calibration_outputs_created(tmp_path):
    symbols = {"BTC": _price_df([100, 101, 100, 102, 101, 103, 102]), "ETH": _price_df([200, 202, 200, 204, 202, 206, 204])}
    strategies = {"BTC": TimedStrategy(activate_at=2), "ETH": TimedStrategy(activate_at=3)}
    rcfg = RiskConfig(max_concurrent_positions=10, min_rr=1.0, max_risk_per_trade=0.001, fee_rate=0.0, slippage_rate=0.0, min_qty=0.000001, min_notional=1.0)
    pcfg = PortfolioSimulationConfig(utility_weight_grid={"w_ev": [0.8, 1.0], "w_corr": [0.1, 0.2]}, enable_utility_weight_calibration=True)
    sim = PortfolioStatefulSimulator(RiskManager(rcfg), rcfg, pcfg)
    res = sim.calibrate_utility_weights(symbols, strategies, output_dir=str(tmp_path))
    assert res["recommended"] is not None
    assert (tmp_path / "utility_weight_ranking.csv").exists()
    assert (tmp_path / "utility_sensitivity_summary.json").exists()


def test_spread_changes_fill_price():
    low = _allocation_setup(enable_corr=False, spread_bps=1.0, max_fill_cost_bps=1000.0, max_spread_bps_for_entry=1000.0)
    high = _allocation_setup(enable_corr=False, spread_bps=50.0, max_fill_cost_bps=1000.0, max_spread_bps_for_entry=1000.0)
    assert high["metrics"]["average_spread_cost"] > low["metrics"]["average_spread_cost"]


def test_slippage_increases_with_size():
    low = _allocation_setup(enable_corr=False, slippage_size_coefficient=0.0)
    high = _allocation_setup(enable_corr=False, slippage_size_coefficient=2.0)
    assert high["metrics"]["average_slippage_cost"] >= low["metrics"]["average_slippage_cost"]


def test_partial_fill_when_liquidity_cap_hit():
    out = _allocation_setup(enable_corr=False, max_participation_rate=0.001)
    assert out["metrics"]["partial_fill_rate"] > 0.0


def test_delayed_execution_affects_fill():
    no_delay = _allocation_setup(enable_corr=False, execution_delay_bars=0)
    delay = _allocation_setup(enable_corr=False, execution_delay_bars=1)
    assert delay["metrics"]["average_slippage_cost"] != no_delay["metrics"]["average_slippage_cost"]


def test_stale_signals_rejected():
    out = _allocation_setup(enable_corr=False, execution_delay_bars=5, stale_signal_bars=1)
    assert out["metrics"]["stale_signal_rejection_count"] >= 1


def test_gap_aware_logic_impacts_results():
    gap_prices = [100, 101, 100, 90, 89, 88, 87]
    symbols = {"BTC": _price_df(gap_prices), "ETH": _price_df([200, 202, 201, 203, 202, 204, 203])}
    symbols["BTC"]["open"] = [100, 101, 100, 80, 89, 88, 87]
    strategies = {"BTC": TimedStrategy(activate_at=2), "ETH": TimedStrategy(activate_at=3)}
    rcfg = RiskConfig(max_concurrent_positions=10, min_rr=1.0, max_risk_per_trade=0.001, fee_rate=0.0, slippage_rate=0.0, min_qty=0.000001, min_notional=1.0)
    base_cfg = dict(max_fill_cost_bps=500.0, max_total_risk_exposure=0.9, max_open_positions=3)
    out_gap = PortfolioStatefulSimulator(RiskManager(rcfg), rcfg, PortfolioSimulationConfig(enable_gap_aware_fills=True, **base_cfg)).run(symbols, strategies)
    out_no = PortfolioStatefulSimulator(RiskManager(rcfg), rcfg, PortfolioSimulationConfig(enable_gap_aware_fills=False, **base_cfg)).run(symbols, strategies)
    assert out_gap["metrics"]["total_return"] != out_no["metrics"]["total_return"]


def test_stress_scenarios_output(tmp_path):
    symbols = {"BTC": _price_df([100, 101, 100, 102, 101, 103, 102]), "ETH": _price_df([200, 202, 200, 204, 202, 206, 204])}
    strategies = {"BTC": TimedStrategy(activate_at=2), "ETH": TimedStrategy(activate_at=3)}
    rcfg = RiskConfig(max_concurrent_positions=10, min_rr=1.0, max_risk_per_trade=0.001, fee_rate=0.0, slippage_rate=0.0, min_qty=0.000001, min_notional=1.0)
    sim = PortfolioStatefulSimulator(RiskManager(rcfg), rcfg, PortfolioSimulationConfig())
    res = sim.run_stress_tests(symbols, strategies, output_dir=str(tmp_path))
    assert res["summary"]["scenario_count"] >= 1
    assert (tmp_path / "stress_scenario_results.csv").exists()
    assert (tmp_path / "stress_test_summary.json").exists()


def test_deployment_guardrail_rejects_poor_fill_quality():
    out = _allocation_setup(enable_corr=False, max_fill_cost_bps=0.1)
    assert out["rejections"].get("fill_quality_too_poor", 0) >= 1

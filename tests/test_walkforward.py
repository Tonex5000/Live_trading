import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.walkforward import (
    CompositeObjectiveConfig,
    ParameterGrid,
    StabilitySelectionConfig,
    WalkForwardConfig,
    WalkForwardRunner,
    generate_walkforward_splits,
)
from src.risk import RiskConfig


def _make_df(n: int = 240) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    base = np.linspace(100, 120, n)
    noise = np.sin(np.linspace(0, 8, n))
    close = base + noise
    open_ = close - 0.1
    high = close + 0.6
    low = close - 0.6
    atr = np.full(n, 1.0)
    adx = np.where(np.arange(n) % 3 == 0, 25.0, 18.0)
    strong = (adx > 20).astype(int)

    momentum = np.cos(np.linspace(0, 10, n))
    target = (momentum > 0).astype(int)
    target[-1] = target[-2]

    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "rsi": 50 + noise,
            "ema_20": close,
            "ema_50": close - 0.2,
            "ema_200": close - 0.5,
            "atr": atr,
            "adx": adx,
            "trend": strong,
            "strong_trend": strong,
            "target": target,
        }
    )


def _grid() -> ParameterGrid:
    return ParameterGrid(
        p_buy=[0.55, 0.65],
        p_sell=[0.35, 0.45],
        min_confidence=[0.2],
        atr_stop_mult=[0.8],
        atr_tp_mult=[1.2],
        min_expected_value=[0.0],
        adx_min=[18.0, 22.0],
    )


def test_walkforward_split_generation_and_no_leakage():
    cfg = WalkForwardConfig(train_bars=100, calibration_bars=20, test_bars=20, step_bars=20)
    splits = generate_walkforward_splits(200, cfg)

    assert len(splits) > 0
    for s in splits:
        assert s["train_start"] < s["train_end"] <= s["cal_start"] < s["cal_end"] <= s["test_start"] < s["test_end"]


def test_optimizer_uses_validation_not_test(monkeypatch, tmp_path):
    df = _make_df(220)
    runner = WalkForwardRunner(
        feature_cols=["rsi", "ema_20", "ema_50", "ema_200", "atr", "adx", "trend", "strong_trend"],
        target_col="target",
    )

    calls = []

    def fake_fit_model(train_df):
        return object()

    def fake_fit_calibrator(model, cal_df, method):
        return object()

    def fake_sim_stateful(df_slice, model, calibrator, params, base_risk):
        calls.append(len(df_slice))
        # Validation slice has 30 rows, test slice has 30 rows too; separate by first timestamp parity.
        is_validation = int(df_slice["timestamp"].iloc[0].hour) % 2 == 0
        score_anchor = 1.0 if params["p_buy"] == 0.55 else -1.0
        if not is_validation:
            score_anchor *= -1.0
        metrics = {
            "total_return": score_anchor,
            "annualized_return": score_anchor,
            "sharpe": score_anchor,
            "sortino": score_anchor,
            "max_drawdown": 0.05,
            "win_rate": 0.6,
            "profit_factor": 1.3,
            "avg_trade": 0.1,
            "expectancy": 0.1,
            "trade_count": 20,
        }
        return {"metrics": metrics, "equity_curve": [10_000, 10_100], "trades": [], "rejections": {}, "regimes": {"trending": 10}}

    monkeypatch.setattr(runner, "_fit_model", fake_fit_model)
    monkeypatch.setattr(runner, "_fit_calibrator", fake_fit_calibrator)
    monkeypatch.setattr(runner, "_simulate_stateful", fake_sim_stateful)

    result = runner.run(
        df,
        WalkForwardConfig(train_bars=120, calibration_bars=30, test_bars=30, step_bars=30),
        _grid(),
        CompositeObjectiveConfig(min_trade_count=1),
        RiskConfig(),
        output_dir=str(tmp_path),
        include_legacy_comparison=False,
    )

    assert result["folds"][0]["candidate_used"]["p_buy"] == 0.55
    assert len(calls) > 0


def test_aggregate_metrics_and_fold_outputs_persisted(tmp_path):
    df = _make_df(300)
    runner = WalkForwardRunner(
        feature_cols=["rsi", "ema_20", "ema_50", "ema_200", "atr", "adx", "trend", "strong_trend"],
        target_col="target",
    )

    result = runner.run(
        df,
        WalkForwardConfig(train_bars=140, calibration_bars=40, test_bars=40, step_bars=40),
        _grid(),
        CompositeObjectiveConfig(min_trade_count=1),
        RiskConfig(min_rr=1.1),
        output_dir=str(tmp_path),
        calibration_method="none",
    )

    summary_path = Path(tmp_path) / "walkforward_summary.json"
    folds_path = Path(tmp_path) / "per_fold_results.json"
    csv_path = Path(tmp_path) / "fold_metrics.csv"

    assert summary_path.exists()
    assert folds_path.exists()
    assert csv_path.exists()

    summary = json.loads(summary_path.read_text())
    folds = json.loads(folds_path.read_text())

    assert summary["fold_count"] == len(result["folds"])
    assert len(folds) == len(result["folds"])
    assert "aggregate_metrics" in summary


def test_best_params_can_vary_across_folds_with_data_shift(monkeypatch, tmp_path):
    df = _make_df(260)
    runner = WalkForwardRunner(
        feature_cols=["rsi", "ema_20", "ema_50", "ema_200", "atr", "adx", "trend", "strong_trend"],
        target_col="target",
    )

    fold_counter = {"idx": -1}

    def fake_fit_model(train_df):
        fold_counter["idx"] += 1
        return {"fold": fold_counter["idx"]}

    def fake_fit_calibrator(model, cal_df, method):
        return None

    def fake_sim_stateful(df_slice, model, calibrator, params, base_risk):
        fold = model["fold"]
        prefer_low = fold % 2 == 0
        pref = (params["p_buy"] == 0.55) if prefer_low else (params["p_buy"] == 0.65)
        sharpe = 1.0 if pref else -1.0
        metrics = {
            "total_return": sharpe,
            "annualized_return": sharpe,
            "sharpe": sharpe,
            "sortino": sharpe,
            "max_drawdown": 0.02,
            "win_rate": 0.5,
            "profit_factor": 1.2,
            "avg_trade": 0.01,
            "expectancy": 0.01,
            "trade_count": 15,
        }
        return {"metrics": metrics, "equity_curve": [10_000, 10_050], "trades": [], "rejections": {}, "regimes": {"ranging": 5}}

    monkeypatch.setattr(runner, "_fit_model", fake_fit_model)
    monkeypatch.setattr(runner, "_fit_calibrator", fake_fit_calibrator)
    monkeypatch.setattr(runner, "_simulate_stateful", fake_sim_stateful)

    result = runner.run(
        df,
        WalkForwardConfig(train_bars=120, calibration_bars=30, test_bars=30, step_bars=30),
        _grid(),
        CompositeObjectiveConfig(min_trade_count=1),
        RiskConfig(),
        output_dir=str(tmp_path),
        include_legacy_comparison=False,
        stability_config=StabilitySelectionConfig(use_stability_selection=False),
    )

    selected = [fold["candidate_used"]["p_buy"] for fold in result["folds"]]
    assert len(set(selected)) > 1

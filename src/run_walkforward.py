import pandas as pd

from src.features import add_indicators
from src.research import (
    CompositeObjectiveConfig,
    ParameterGrid,
    StabilitySelectionConfig,
    WalkForwardConfig,
    WalkForwardRunner,
)
from src.risk import RiskConfig


def main() -> None:
    df = pd.read_csv("data/raw/BTCUSDT_1h.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = add_indicators(df).dropna().reset_index(drop=True)

    feature_cols = ["rsi", "ema_20", "ema_50", "ema_200", "atr", "adx", "trend", "strong_trend"]

    wf_config = WalkForwardConfig(
        train_bars=2000,
        calibration_bars=500,
        test_bars=500,
        step_bars=500,
        mode="rolling",
    )

    param_grid = ParameterGrid(
        p_buy=[0.55, 0.60, 0.65],
        p_sell=[0.35, 0.40, 0.45],
        min_confidence=[0.25, 0.30],
        atr_stop_mult=[0.8, 1.0],
        atr_tp_mult=[1.2, 1.5],
        min_expected_value=[0.0, 0.25],
        adx_min=[18.0, 20.0, 24.0],
        drawdown_risk_multiplier=[0.5, 0.7],
        atr_risk_multiplier=[0.6, 0.8],
    )

    objective = CompositeObjectiveConfig(
        w_sharpe=1.2,
        w_return=1.0,
        w_profit_factor=0.4,
        w_drawdown=1.1,
        min_trade_count=15,
        penalty_low_trades=2.0,
        penalty_unstable_sharpe=0.5,
        max_drawdown_penalty_threshold=0.20,
        penalty_excessive_drawdown=1.5,
    )

    base_risk = RiskConfig()

    runner = WalkForwardRunner(feature_cols=feature_cols, target_col="target")

    stability = StabilitySelectionConfig(use_stability_selection=True)
    results = runner.run(
        df=df,
        wf_config=wf_config,
        parameter_grid=param_grid,
        objective_config=objective,
        base_risk_config=base_risk,
        output_dir="models/walkforward",
        calibration_method="platt",
        stability_config=stability,
    )

    print("✅ Walk-forward complete")
    print(f"Folds: {results['summary'].get('fold_count', 0)}")
    print(f"Best fold: {results['summary'].get('best_fold')}")


if __name__ == "__main__":
    main()

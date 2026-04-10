import joblib
import pandas as pd

from src.backtest import backtest
from src.features import add_indicators
from src.ml.calibration import ProbabilityCalibrator
from src.strategies import MLStrategy, ThresholdConfig, load_thresholds


def generate_strategy_outputs(df: pd.DataFrame, strategy: MLStrategy):
    signals = []
    probs = []

    for i in range(len(df)):
        row_df = df.iloc[i : i + 1]
        signal = strategy.generate(row_df)
        if signal.action.value == "BUY":
            signals.append(1)
        elif signal.action.value == "SELL":
            signals.append(-1)
        else:
            signals.append(0)
        probs.append(signal.probability)

    return signals, probs


def main() -> None:
    df = pd.read_csv("data/raw/BTCUSDT_1h.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = add_indicators(df)
    df = df.dropna().reset_index(drop=True)

    model = joblib.load("models/xgb_signal_model.pkl")
    calibrator = ProbabilityCalibrator.load("models/prob_calibrator.pkl")
    p_buy, p_sell = load_thresholds(ThresholdConfig(mode="optimized", p_buy=0.60, p_sell=0.40))

    strategy = MLStrategy(
        model=model,
        symbol="BTC/USDT",
        timeframe="1h",
        p_buy=p_buy,
        p_sell=p_sell,
        adx_min=20.0,
        probability_calibrator=calibrator,
    )
    signals, probs = generate_strategy_outputs(df, strategy)

    _, _, metrics, _ = backtest(
        df=df,
        signals=signals,
        probs=probs,
        initial_capital=10_000,
        risk_pct=0.01,
        fee_rate=0.0006,
        slippage_rate=0.0004,
        min_prob=0.60,
        min_rr=1.5,
        min_adx=20.0,
        min_trades_required=500,
    )

    print("Backtest Metrics:")
    for k, v in metrics.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
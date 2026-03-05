
import joblib
import pandas as pd

from src.backtest import backtest
from src.features import add_indicators
from src.signals import generate_signals


def main() -> None:
    df = pd.read_csv("data/raw/BTCUSDT_1h.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = add_indicators(df)
    df = df.dropna().reset_index(drop=True)

    model = joblib.load("models/xgb_signal_model.pkl")

    signals, probs = generate_signals(df, model, p_buy=0.60, p_sell=0.40, adx_min=20.0)

    _, _, metrics = backtest(
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
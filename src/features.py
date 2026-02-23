import pandas as pd
import ta

df = pd.read_csv("data/raw/BTCUSDT_1h.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Indicators
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    df["ema_20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()

    df["strong_trend"] = (
        (df["ema_20"] > df["ema_50"]) &
        (df["ema_50"] > df["ema_200"])
    ).astype(int)

    df["trend"] = (df["ema_20"] > df["ema_50"]).astype(int)

    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"],
        df["low"],
        df["close"]
    ).average_true_range()

    return df

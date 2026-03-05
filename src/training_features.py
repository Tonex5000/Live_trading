import pandas as pd
from src.features import add_indicators


def add_training_features(
    df: pd.DataFrame,
    horizon: int = 2,
    neutral_atr_mult: float = 0.35,
) -> pd.DataFrame:
    """Build supervised labels {-1, 0, 1} from future returns.

    - 1  => expected bullish move beyond a volatility-adjusted neutral zone
    - 0  => expected move too small / noisy (do nothing)
    - -1 => expected bearish move beyond a volatility-adjusted neutral zone
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = add_indicators(df)

    df["future_close"] = df["close"].shift(-horizon)
    df["future_return"] = (df["future_close"] - df["close"]) / df["close"]

    # Volatility-scaled neutral band to avoid overtrading random noise
    atr_ratio = (df["atr"] / df["close"]).clip(lower=0)
    neutral_band = atr_ratio * neutral_atr_mult

    df["target"] = 0
    df.loc[df["future_return"] > neutral_band, "target"] = 1
    df.loc[df["future_return"] < -neutral_band, "target"] = -1

    df = df.dropna().reset_index(drop=True)
    return df

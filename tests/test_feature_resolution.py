import pandas as pd
import pytest

from src.signals import FeatureResolutionError, generate_signals


class DummyModel:
    feature_names_in_ = ["rsi", "ema_20", "ema_50", "ema_200", "atr", "adx", "trend", "strong_trend"]

    def predict_proba(self, X):
        return [[0.2, 0.8] for _ in range(len(X))]


def test_feature_mismatch_fast_fail():
    df = pd.DataFrame(
        [
            {
                "rsi": 50,
                "ema_20": 1,
                "ema_50": 1,
                "ema_200": 1,
                "atr": 1,
                "trend": 1,
                "strong_trend": 1,
            }
        ]
    )

    with pytest.raises(FeatureResolutionError):
        generate_signals(df, DummyModel(), allow_legacy_feature_fallback=False)

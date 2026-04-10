import pandas as pd

from src.signals import _resolve_model_features
from src.strategies.base import BaseStrategy
from src.strategies.regime import classify_regime
from src.strategies.signal_schema import SignalAction, StrategySignal


class MLStrategy(BaseStrategy):
    def __init__(
        self,
        model,
        symbol: str,
        timeframe: str,
        p_buy: float = 0.60,
        p_sell: float = 0.40,
        adx_min: float = 20.0,
        allow_legacy_feature_fallback: bool = False,
        probability_calibrator=None,
    ):
        self.model = model
        self.symbol = symbol
        self.timeframe = timeframe
        self.p_buy = p_buy
        self.p_sell = p_sell
        self.adx_min = adx_min
        self.allow_legacy_feature_fallback = allow_legacy_feature_fallback
        self.probability_calibrator = probability_calibrator

    def generate(self, df: pd.DataFrame) -> StrategySignal:
        latest_row = df.iloc[-1:]
        feature_cols = _resolve_model_features(
            self.model,
            latest_row,
            allow_legacy_feature_fallback=self.allow_legacy_feature_fallback,
        )

        raw_prob = float(self.model.predict_proba(latest_row[feature_cols])[:, 1][0])
        calibrated_prob = raw_prob
        if self.probability_calibrator is not None:
            calibrated_prob = float(self.probability_calibrator.predict([raw_prob])[0])

        confidence = float(max(abs(calibrated_prob - 0.5) * 2, 0.15))

        strong_trend = int(latest_row["strong_trend"].iloc[0])
        adx = float(latest_row["adx"].iloc[0]) if "adx" in latest_row.columns else 0.0
        atr = float(latest_row["atr"].iloc[0]) if "atr" in latest_row.columns else 0.0
        atr_mean = float(df["atr"].rolling(50, min_periods=1).mean().iloc[-1]) if "atr" in df.columns else atr

        regime = classify_regime(adx=adx, strong_trend=strong_trend, atr=atr, atr_mean=atr_mean, adx_trend_threshold=self.adx_min)

        if strong_trend == 0 or adx <= self.adx_min:
            action = SignalAction.HOLD
            reason = "weak_trend"
        elif calibrated_prob >= self.p_buy:
            action = SignalAction.BUY
            reason = f"probability_above_buy_threshold:{self.p_buy}"
        elif calibrated_prob <= self.p_sell:
            action = SignalAction.SELL
            reason = f"probability_below_sell_threshold:{self.p_sell}"
        else:
            action = SignalAction.HOLD
            reason = "ambiguous_probability"

        ts = latest_row["timestamp"].iloc[0]
        return StrategySignal(
            symbol=self.symbol,
            timeframe=self.timeframe,
            action=action,
            confidence=confidence,
            probability=calibrated_prob,
            reason=reason,
            timestamp=ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            raw_probability=raw_prob,
            calibrated_probability=calibrated_prob,
            regime=regime,
        )

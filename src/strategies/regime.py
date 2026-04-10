def classify_regime(adx: float, strong_trend: int, atr: float, atr_mean: float, adx_trend_threshold: float = 20.0, atr_spike_mult: float = 1.8) -> str:
    if atr_mean > 0 and atr > atr_mean * atr_spike_mult:
        return "high_volatility"
    if adx > adx_trend_threshold and strong_trend == 1:
        return "trending"
    return "ranging"

import numpy as np

LEGACY_FEATURES = ["rsi", "ema_20", "ema_50", "ema_200", "atr", "trend", "strong_trend"]
FEATURES = ["rsi", "ema_20", "ema_50", "ema_200", "atr", "adx", "trend", "strong_trend"]


def _resolve_model_features(model, df):
    """Return the most compatible feature list for the loaded model and current dataframe."""
    # Best case: use model feature names if available.
    model_feature_names = getattr(model, "feature_names_in_", None)
    if model_feature_names is not None:
        model_feature_names = list(model_feature_names)
        if all(col in df.columns for col in model_feature_names):
            return model_feature_names

    # Fallback to feature-count compatibility.
    n_features = getattr(model, "n_features_in_", None)
    if n_features == len(FEATURES):
        return FEATURES
    if n_features == len(LEGACY_FEATURES):
        return LEGACY_FEATURES

    # Last fallback: prefer newer set only if present, else legacy.
    if all(col in df.columns for col in FEATURES):
        return FEATURES
    return [col for col in LEGACY_FEATURES if col in df.columns]


def generate_signals(df, model, p_buy=0.60, p_sell=0.40, adx_min=20.0):
    model_features = _resolve_model_features(model, df)
    probs = model.predict_proba(df[model_features])[:, 1]
    signals = []

    for i, p in enumerate(probs):
        strong_trend = df.iloc[i]["strong_trend"]
        adx = df.iloc[i]["adx"] if "adx" in df.columns else 0.0

        # Regime filter: only trade in trending markets.
        if strong_trend == 0 or adx <= adx_min:
            signals.append(0)
            continue

        # Higher confidence threshold for entries.
        if p >= p_buy:
            signals.append(1)
        elif p <= p_sell:
            signals.append(-1)
        else:
            signals.append(0)

    return np.array(signals), probs


def apply_risk_filter(
    signal,
    confidence,
    atr,
    atr_mean,
    expected_rr,
    expected_net_edge=0.0,
    min_conf=0.3,
    atr_mult=1.8,
    min_rr=1.5,
):
    if confidence < min_conf:
        return 0  # NO TRADE
    if atr > atr_mean * atr_mult:
        return 0  # NO TRADE
    if expected_rr < min_rr:
        return 0  # NO TRADE
    if expected_net_edge <= 0:
        return 0  # NO TRADE
    return signal
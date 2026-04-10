import numpy as np

LEGACY_FEATURES = ["rsi", "ema_20", "ema_50", "ema_200", "atr", "trend", "strong_trend"]
FEATURES = ["rsi", "ema_20", "ema_50", "ema_200", "atr", "adx", "trend", "strong_trend"]


class FeatureResolutionError(ValueError):
    pass


def _resolve_model_features(model, df, allow_legacy_feature_fallback: bool = False):
    """Resolve model features with strict mode by default."""
    model_feature_names = getattr(model, "feature_names_in_", None)
    if model_feature_names is not None:
        model_feature_names = list(model_feature_names)
        missing = [col for col in model_feature_names if col not in df.columns]
        if missing:
            raise FeatureResolutionError(f"feature_mismatch: missing model features {missing}")
        return model_feature_names

    n_features = getattr(model, "n_features_in_", None)
    if n_features == len(FEATURES) and all(col in df.columns for col in FEATURES):
        return FEATURES
    if allow_legacy_feature_fallback and n_features == len(LEGACY_FEATURES) and all(col in df.columns for col in LEGACY_FEATURES):
        return LEGACY_FEATURES

    if allow_legacy_feature_fallback and all(col in df.columns for col in FEATURES):
        return FEATURES
    if allow_legacy_feature_fallback:
        legacy_available = [col for col in LEGACY_FEATURES if col in df.columns]
        if len(legacy_available) == len(LEGACY_FEATURES):
            return legacy_available

    raise FeatureResolutionError(
        "feature_mismatch: unable to resolve required feature set; set allow_legacy_feature_fallback=true for compatibility mode"
    )


def generate_signals(df, model, p_buy=0.60, p_sell=0.40, adx_min=20.0, allow_legacy_feature_fallback: bool = False):
    model_features = _resolve_model_features(model, df, allow_legacy_feature_fallback=allow_legacy_feature_fallback)
    probs = model.predict_proba(df[model_features])[:, 1]
    signals = []

    for i, p in enumerate(probs):
        strong_trend = df.iloc[i]["strong_trend"]
        adx = df.iloc[i]["adx"] if "adx" in df.columns else 0.0

        if strong_trend == 0 or adx <= adx_min:
            signals.append(0)
            continue

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
        return 0
    if atr > atr_mean * atr_mult:
        return 0
    if expected_rr < min_rr:
        return 0
    if expected_net_edge <= 0:
        return 0
    return signal
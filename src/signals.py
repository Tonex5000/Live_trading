import numpy as np

FEATURES = ["rsi","ema_20","ema_50","ema_200","atr","trend","strong_trend"]

def generate_signals(df, model, p_buy=0.6, p_sell=0.2):
    probs = model.predict_proba(df[FEATURES])[:, 1]
    signals = []

    for i, p in enumerate(probs):
        if df.iloc[i]["strong_trend"] == 0:
            signals.append(0)
            continue
        if p > p_buy:
            signals.append(1)
        elif p < p_sell:
            signals.append(-1)
        else:
            signals.append(0)
    return np.array(signals), probs

def apply_risk_filter(signal, confidence, atr, atr_mean, min_conf=0.3, atr_mult=1.8):
    if confidence < min_conf:
        return 0  # NO TRADE
    if atr > atr_mean * atr_mult:
        return 0  # NO TRADE
    return signal

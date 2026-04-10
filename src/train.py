import pandas as pd
from xgboost import XGBClassifier
import joblib
from sklearn.model_selection import TimeSeriesSplit
from src.features import add_indicators

# Load & preprocess
df = pd.read_csv("data/processed/BTCUSDT_features.csv")
df = add_indicators(df)
df = df.dropna().reset_index(drop=True)

features = ["rsi", "ema_20", "ema_50", "ema_200", "atr", "adx", "trend", "strong_trend"]
X = df[features]
y = df["target"]

# TimeSeries split (use final split for validation + calibration)
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Train base model
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
)
model.fit(X_train, y_train)

# Calibration (Platt by default for robustness)
raw_test_probs = model.predict_proba(X_test)[:, 1]
# Map labels to long-win target for calibration compatibility
# 1 => bullish success, everything else => not bullish success
cal_y = (y_test == 1).astype(int)
calibrator = ProbabilityCalibrator(method="platt").fit(raw_test_probs, cal_y)
calibrator.save("models/prob_calibrator.pkl")

# Optimize thresholds using calibrated probabilities
cal_probs = calibrator.predict(raw_test_probs)
# For threshold search use directional target {-1,0,1}
thresholds = optimize_thresholds(cal_probs, y_test.values)
save_optimized_thresholds("models/optimized_thresholds.json", thresholds)

# Save model
joblib.dump(model, "models/xgb_signal_model.pkl")
print("✅ Model trained, calibrated, and thresholds optimized/saved!")
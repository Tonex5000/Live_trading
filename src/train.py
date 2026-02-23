import pandas as pd
from xgboost import XGBClassifier
import joblib
from sklearn.model_selection import TimeSeriesSplit
from features import add_features

# Load & preprocess
df = pd.read_csv("data/processed/BTCUSDT_features.csv")
df = add_features(df)

features = ["rsi","ema_20","ema_50","ema_200","atr","trend","strong_trend"]
X = df[features]
y = df["target"]

# TimeSeries split (just for training last split)
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Train model
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/xgb_signal_model.pkl")
print("✅ Model trained and saved!")

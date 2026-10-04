"""
XGBoost DO Predictor
====================
Trains an XGBoost regressor that predicts Dissolved Oxygen (DO)
from the 4 hardware sensor inputs: TEMP, PH, TURBIDITY, TDS.

Dataset: Aquaculture_Master_Dataset_with_TDS.csv (74,758 rows)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PATH  = BASE_DIR / "data" / "Aquaculture_Master_Dataset_with_TDS.csv"
MODEL_OUT  = BASE_DIR / "model" / "aquaculture_xgb_do.pkl"
# ─────────────────────────────────────────────────────────────

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Rename for consistency
df = df.rename(columns={
    "TEMP":      "Temp",
    "PH":        "pH",
    "DO":        "DO",
    "TURBIDITY": "Turbidity",
    "TDS":       "TDS",
})

FEATURES = ["Temp", "pH", "Turbidity", "TDS"]
TARGET   = "DO"

df = df[FEATURES + [TARGET]].dropna()

print(f"Dataset shape after dropna: {df.shape}")

X = df[FEATURES].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training XGBoost regressor...")
model = XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    tree_method="hist",
    device="cpu",
    verbosity=1
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nDO Predictor RMSE: {rmse:.4f} mg/L")
print(f"   (PySR formula DO RMSE for reference: ~1.056 mg/L)")

joblib.dump(model, MODEL_OUT)
print(f"\nModel saved to: {MODEL_OUT}")

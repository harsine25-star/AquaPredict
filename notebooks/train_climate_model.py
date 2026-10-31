
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Project folders
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "steinhagen_rainfall_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "steinhagen_climate_model.joblib"
RESULT_PATH = ROOT / "data" / "processed" / "climate_model_predictions.csv"

# Forecast features
LAGS = [1, 2, 3, 7, 14, 30]
CLIMATE_COLUMNS = ["precipitation", "gross_evap", "net_evap"]

print("Loading climate and reservoir data...")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

# Support the net_evapo spelling used in some source files
if "net_evap" not in df.columns and "net_evapo" in df.columns:
    df = df.rename(columns={"net_evapo": "net_evap"})

required = ["date", "percent_full"] + CLIMATE_COLUMNS
missing = [column for column in required if column not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["date"] = pd.to_datetime(df["date"], errors="coerce")

for column in ["percent_full"] + CLIMATE_COLUMNS:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = (
    df.dropna(subset=required)
    .sort_values("date")
    .drop_duplicates(subset="date", keep="last")
    .reset_index(drop=True)
)

# Create past-storage features
for lag in LAGS:
    df[f"storage_lag_{lag}"] = df["percent_full"].shift(lag)

# Calendar features
df["day_of_year"] = df["date"].dt.dayofyear
df["month"] = df["date"].dt.month

FEATURES = (
    [f"storage_lag_{lag}" for lag in LAGS]
    + ["day_of_year", "month"]
    + CLIMATE_COLUMNS
)

model_data = df.dropna(subset=FEATURES + ["percent_full"]).copy()

if len(model_data) < 100:
    raise ValueError(
        f"Only {len(model_data)} usable rows. Check the dataset and lag features."
    )

X = model_data[FEATURES]
y = model_data["percent_full"]

# Chronological split: earlier data for training, later data for testing
split_index = int(len(model_data) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(f"Usable rows: {len(model_data)}")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Training ends: {model_data.iloc[split_index - 1]['date'].date()}")
print(f"Testing starts: {model_data.iloc[split_index]['date'].date()}")

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest with climate features...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Persistence baseline: predict the next value using the previous day's storage
baseline = model_data["percent_full"].shift(1).iloc[split_index:].to_numpy()

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

baseline_mae = mean_absolute_error(y_test, baseline)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline))

print("\n========== CLIMATE RANDOM FOREST RESULTS ==========")
print(f"MAE:  {mae:.3f} percentage points")
print(f"RMSE: {rmse:.3f} percentage points")
print("\nPersistence baseline:")
print(f"MAE:  {baseline_mae:.3f} percentage points")
print(f"RMSE: {baseline_rmse:.3f} percentage points")

# Save predictions for inspection
results = pd.DataFrame({
    "date": model_data["date"].iloc[split_index:].to_numpy(),
    "actual_percent_full": y_test.to_numpy(),
    "predicted_percent_full": predictions,
    "baseline_percent_full": baseline
})

RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(RESULT_PATH, index=False)

# Fit the final model on all available historical rows
# This saved model is intended for future forecasting.
model.fit(X, y)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump({
    "model": model,
    "features": FEATURES,
    "lags": LAGS,
    "climate_columns": CLIMATE_COLUMNS,
    "model_type": "Random Forest with climate features",
    "training_end_date": str(model_data["date"].max().date()),
    "sklearn_version": sklearn.__version__
}, MODEL_PATH)

print(f"\nSaved evaluation predictions to: {RESULT_PATH}")
print(f"Saved climate model to: {MODEL_PATH}")
print("\nTraining and evaluation completed!")


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "data" / "processed" / "steinhagen_model_data.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_FILE, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Features from past observations and calendar
for lag in [1, 2, 3, 7, 14, 30]:
    df[f"lag_{lag}"] = df["percent_full"].shift(lag)

df["day_of_year"] = df["date"].dt.dayofyear
df["month"] = df["date"].dt.month

# Target: storage percentage on the current row's date
df = df.dropna().reset_index(drop=True)

feature_cols = [
    "lag_1", "lag_2", "lag_3",
    "lag_7", "lag_14", "lag_30",
    "day_of_year", "month"
]

X = df[feature_cols]
y = df["percent_full"]

# Chronological split: earliest 80% for training, latest 20% for testing
split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
dates_test = df["date"].iloc[split:]

# Train model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate Random Forest
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# Baseline: use previous day's observed percentage
baseline_predictions = df["lag_1"].iloc[split:].to_numpy()

baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_predictions)
)

print("\n--- DATA SPLIT ---")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Test period:", dates_test.min().date(), "to", dates_test.max().date())

print("\n--- RANDOM FOREST RESULTS ---")
print(f"MAE:  {mae:.3f} percentage points")
print(f"RMSE: {rmse:.3f} percentage points")

print("\n--- BASELINE RESULTS ---")
print(f"MAE:  {baseline_mae:.3f} percentage points")
print(f"RMSE: {baseline_rmse:.3f} percentage points")

# Save model and feature names
model_path = MODEL_DIR / "steinhagen_model.joblib"

joblib.dump(
    {"model": model, "features": feature_cols},
    model_path
)

print("\nModel saved to:", model_path)

# Plot actual versus predicted values
plt.figure(figsize=(12, 5))
plt.plot(dates_test, y_test.to_numpy(), label="Actual", linewidth=1)
plt.plot(dates_test, predictions, label="Random Forest", linewidth=1)
plt.title("B. A. Steinhagen: Actual vs Predicted Storage")
plt.xlabel("Date")
plt.ylabel("Percent Full (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = MODEL_DIR / "actual_vs_predicted.png"
plt.savefig(plot_path, dpi=150)
plt.close()

print("Prediction graph saved to:", plot_path)

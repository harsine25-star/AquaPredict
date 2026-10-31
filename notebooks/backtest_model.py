
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = (
    PROJECT_DIR / "data" / "processed"
    / "steinhagen_model_data.csv"
)

# Load historical observations
df = pd.read_csv(DATA_FILE, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Create historical lag features
for lag in [1, 2, 3, 7, 14, 30]:
    df[f"lag_{lag}"] = df["percent_full"].shift(lag)

df["day_of_year"] = df["date"].dt.dayofyear
df["month"] = df["date"].dt.month
df = df.dropna().reset_index(drop=True)

features = [
    "lag_1", "lag_2", "lag_3",
    "lag_7", "lag_14", "lag_30",
    "day_of_year", "month"
]

# Chronological 80/20 split
split = int(len(df) * 0.8)

train = df.iloc[:split]
test = df.iloc[split:]

# Train only on the earlier period
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

model.fit(train[features], train["percent_full"])

# Rolling-origin backtest.
# At each origin, only observations up to that date are used.
horizons = [1, 3, 7]
results = {h: {"actual": [], "model": [], "baseline": []}
           for h in horizons}

values = df["percent_full"].to_numpy()

# Origins are spaced 7 days apart to reduce overlap.
for origin in range(split, len(df) - max(horizons) + 1, 7):
    history = list(values[:origin])

    predictions = {}

    for step in range(1, max(horizons) + 1):
        target_date = df.loc[origin + step - 1, "date"]

        row = {
            "lag_1": history[-1],
            "lag_2": history[-2],
            "lag_3": history[-3],
            "lag_7": history[-7],
            "lag_14": history[-14],
            "lag_30": history[-30],
            "day_of_year": target_date.dayofyear,
            "month": target_date.month
        }

        X_next = pd.DataFrame([row])[features]
        prediction = float(model.predict(X_next)[0])
        prediction = np.clip(prediction, 0, 100)

        predictions[step] = prediction

        # Use the prediction, not the future actual value.
        history.append(prediction)

    for h in horizons:
        actual = values[origin + h - 1]

        results[h]["actual"].append(actual)
        results[h]["model"].append(predictions[h])

        # Persistence baseline: latest observed value
        results[h]["baseline"].append(values[origin - 1])

print("\n--- MULTI-DAY BACKTEST RESULTS ---")
print("Training ends:", train["date"].max().date())
print("Testing starts:", test["date"].min().date())

for h in horizons:
    actual = np.array(results[h]["actual"])
    predicted = np.array(results[h]["model"])
    baseline = np.array(results[h]["baseline"])

    model_mae = mean_absolute_error(actual, predicted)
    model_rmse = np.sqrt(mean_squared_error(actual, predicted))

    baseline_mae = mean_absolute_error(actual, baseline)
    baseline_rmse = np.sqrt(mean_squared_error(actual, baseline))

    print(f"\n{h}-day forecast:")
    print(f"  Model MAE:    {model_mae:.3f}")
    print(f"  Model RMSE:   {model_rmse:.3f}")
    print(f"  Baseline MAE: {baseline_mae:.3f}")
    print(f"  Baseline RMSE:{baseline_rmse:.3f}")
    print(f"  Forecast origins evaluated: {len(actual)}")

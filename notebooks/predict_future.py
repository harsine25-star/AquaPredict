
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import joblib

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_DIR / "data" / "processed"
    / "steinhagen_model_data.csv"
)

MODEL_FILE = (
    PROJECT_DIR / "models" / "steinhagen_model.joblib"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load historical data and trained model
df = pd.read_csv(DATA_FILE, parse_dates=["date"])
saved = joblib.load(MODEL_FILE)

model = saved["model"]
features = saved["features"]

df = df.sort_values("date").reset_index(drop=True)

# Use observed history to predict the next day,
# then feed each prediction into the next forecast.
history = df["percent_full"].astype(float).tolist()
last_date = df["date"].iloc[-1]

forecasts = []

for step in range(1, 8):
    next_date = last_date + pd.Timedelta(days=step)

    row = {
        "lag_1": history[-1],
        "lag_2": history[-2],
        "lag_3": history[-3],
        "lag_7": history[-7],
        "lag_14": history[-14],
        "lag_30": history[-30],
        "day_of_year": next_date.dayofyear,
        "month": next_date.month,
    }

    X_next = pd.DataFrame([row])[features]
    prediction = float(model.predict(X_next)[0])

    # Storage percentage should be between 0 and 100.
    prediction = max(0.0, min(100.0, prediction))

    forecasts.append({
        "date": next_date,
        "predicted_percent_full": round(prediction, 2)
    })

    history.append(prediction)

forecast_df = pd.DataFrame(forecasts)

# Simple illustrative threshold, not an official water authority standard.
threshold = 30.0

forecast_df["status"] = forecast_df[
    "predicted_percent_full"
].apply(
    lambda value: "Potential low-storage alert"
    if value < threshold else "No low-storage alert"
)

output_file = OUTPUT_DIR / "steinhagen_7day_forecast.csv"
forecast_df.to_csv(output_file, index=False)

print("\n--- 7-DAY FORECAST ---")
print(forecast_df.to_string(index=False))
print("\nForecast saved to:", output_file)

# Plot recent history and future predictions
recent = df.tail(60)

plt.figure(figsize=(12, 5))
plt.plot(
    recent["date"],
    recent["percent_full"],
    label="Historical observations"
)
plt.plot(
    forecast_df["date"],
    forecast_df["predicted_percent_full"],
    marker="o",
    label="7-day forecast"
)

plt.axhline(
    threshold,
    linestyle="--",
    label="Illustrative 30% threshold"
)

plt.title("B. A. Steinhagen — Historical Storage and 7-Day Forecast")
plt.xlabel("Date")
plt.ylabel("Percent Full (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_file = OUTPUT_DIR / "steinhagen_7day_forecast.png"
plt.savefig(plot_file, dpi=150)
plt.close()

print("Forecast graph saved to:", plot_file)

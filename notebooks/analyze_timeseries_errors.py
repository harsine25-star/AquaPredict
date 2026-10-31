
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "timeseries_predictions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load predictions
df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

actual_col = "actual_percent_full"

prediction_cols = {
    "Persistence": "persistence_prediction",
    "SARIMA": "sarima_prediction",
    "SARIMAX": "sarimax_prediction",
}

# Check columns
required_cols = [actual_col] + list(prediction_cols.values())

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# Remove rows without actual observations
df = df.dropna(subset=[actual_col]).copy()

# Calculate errors
for model_name, pred_col in prediction_cols.items():
    df[f"{model_name}_error"] = df[pred_col] - df[actual_col]
    df[f"{model_name}_absolute_error"] = (
        df[f"{model_name}_error"].abs()
    )

# --------------------------------------------------
# 1. Actual vs predicted graph
# --------------------------------------------------

plt.figure(figsize=(15, 7))

plt.plot(
    df["date"],
    df[actual_col],
    label="Actual reservoir fullness",
    linewidth=2.5,
)

for model_name, pred_col in prediction_cols.items():
    plt.plot(
        df["date"],
        df[pred_col],
        label=f"{model_name} prediction",
        linewidth=1.5,
        alpha=0.85,
    )

plt.title("AquaPredict: Actual vs Predicted Monthly Reservoir Fullness")
plt.xlabel("Date")
plt.ylabel("Reservoir Fullness (%)")
plt.ylim(0, 100)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

graph1 = OUTPUT_DIR / "actual_vs_predicted_monthly.png"
plt.savefig(graph1, dpi=200, bbox_inches="tight")
plt.close()

# --------------------------------------------------
# 2. Absolute prediction error graph
# --------------------------------------------------

plt.figure(figsize=(15, 6))

for model_name in prediction_cols:
    plt.plot(
        df["date"],
        df[f"{model_name}_absolute_error"],
        label=f"{model_name} absolute error",
        linewidth=1.5,
    )

plt.title("AquaPredict: Monthly Absolute Prediction Errors")
plt.xlabel("Date")
plt.ylabel("Absolute Error (percentage points)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

graph2 = OUTPUT_DIR / "monthly_absolute_errors.png"
plt.savefig(graph2, dpi=200, bbox_inches="tight")
plt.close()

# --------------------------------------------------
# 3. Calculate metrics on the same months
# --------------------------------------------------

print("\n========== MODEL PERFORMANCE ==========")

metrics = []

for model_name, pred_col in prediction_cols.items():
    valid = df[[actual_col, pred_col]].dropna()

    mae = mean_absolute_error(
        valid[actual_col], valid[pred_col]
    )

    rmse = np.sqrt(
        mean_squared_error(valid[actual_col], valid[pred_col])
    )

    metrics.append({
        "Model": model_name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "Months evaluated": len(valid),
    })

    print(
        f"{model_name}: "
        f"MAE = {mae:.3f}, RMSE = {rmse:.3f}"
    )

metrics_df = pd.DataFrame(metrics).sort_values("MAE")

print("\n", metrics_df.to_string(index=False))

# --------------------------------------------------
# 4. Find the 10 largest errors for each model
# --------------------------------------------------

error_rows = []

for model_name, pred_col in prediction_cols.items():
    error_col = f"{model_name}_absolute_error"

    worst = df.nlargest(10, error_col)

    for _, row in worst.iterrows():
        error_rows.append({
            "Model": model_name,
            "Date": row["date"].strftime("%Y-%m-%d"),
            "Actual fullness (%)": round(
                row[actual_col], 3
            ),
            "Predicted fullness (%)": round(
                row[pred_col], 3
            ),
            "Absolute error (percentage points)": round(
                row[error_col], 3
            ),
        })

error_df = pd.DataFrame(error_rows)

error_file = OUTPUT_DIR / "largest_monthly_prediction_errors.csv"
error_df.to_csv(error_file, index=False)

# Save metrics too
metrics_file = OUTPUT_DIR / "monthly_model_performance.csv"
metrics_df.to_csv(metrics_file, index=False)

print("\nFiles created successfully:")
print(graph1)
print(graph2)
print(error_file)
print(metrics_file)

print("\nAnalysis completed!")

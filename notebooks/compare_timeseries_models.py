
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import statsmodels
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STORAGE_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "steinhagen_rainfall_data.csv"
)

CLIMATE_FILE = PROJECT_ROOT / "data" / "raw" / "rainfall.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load and clean the data
# --------------------------------------------------

print("Loading AquaPredict data...")
print("Statsmodels version:", statsmodels.__version__)

storage_df = pd.read_csv(STORAGE_FILE)
climate_df = pd.read_csv(CLIMATE_FILE, comment="#")

# Normalize column names.
storage_df.columns = (
    storage_df.columns.str.strip().str.lower().str.replace(" ", "_")
)
climate_df.columns = (
    climate_df.columns.str.strip().str.lower().str.replace(" ", "_")
)

# Check required columns.
if "date" not in storage_df.columns or "percent_full" not in storage_df.columns:
    raise ValueError(
        "Storage CSV must contain 'date' and 'percent_full' columns."
    )

if "period" not in climate_df.columns:
    raise ValueError(
        "Rainfall CSV must contain a 'period' column."
    )

# Support both net_evapo and net_evap spellings.
if "net_evapo" in climate_df.columns and "net_evap" not in climate_df.columns:
    climate_df = climate_df.rename(columns={"net_evapo": "net_evap"})

required_climate = ["precipitation", "gross_evap", "net_evap"]

missing_columns = [
    col for col in required_climate if col not in climate_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing climate columns in rainfall.csv: {missing_columns}. "
        f"Available columns: {list(climate_df.columns)}"
    )

# Convert storage dates and values.
storage_df["date"] = pd.to_datetime(
    storage_df["date"], errors="coerce"
)
storage_df["percent_full"] = pd.to_numeric(
    storage_df["percent_full"], errors="coerce"
)

storage_df = storage_df.dropna(
    subset=["date", "percent_full"]
).copy()

# Convert the climate period into a monthly date.
climate_df["month"] = pd.to_datetime(
    climate_df["period"].astype(str), errors="coerce"
)

# If period values are formatted like YYYY-MM, pandas parses them.
# Normalize every climate date to the first day of its month.
climate_df = climate_df.dropna(subset=["month"]).copy()
climate_df["month"] = (
    climate_df["month"].dt.to_period("M").dt.to_timestamp()
)

for col in required_climate:
    climate_df[col] = pd.to_numeric(
        climate_df[col], errors="coerce"
    )

climate_df = climate_df.dropna(subset=required_climate)

# One observation per month:
# - Reservoir fullness: monthly mean
# - Climate measurements: monthly value
monthly_storage = (
    storage_df.set_index("date")["percent_full"]
    .resample("MS")
    .mean()
)

monthly_climate = (
    climate_df.groupby("month")[required_climate]
    .mean()
    .sort_index()
)

# Use only the period covered by both datasets.
start_date = max(monthly_storage.index.min(), monthly_climate.index.min())
end_date = min(monthly_storage.index.max(), monthly_climate.index.max())

if start_date >= end_date:
    raise ValueError(
        "Storage and climate datasets do not have an overlapping date range."
    )

full_index = pd.date_range(
    start=start_date.to_period("M").to_timestamp(),
    end=end_date.to_period("M").to_timestamp(),
    freq="MS",
)

# Reindex to the actual calendar months. Missing storage observations
# remain NaN; we do NOT invent reservoir measurements.
monthly = pd.DataFrame(index=full_index)
monthly.index.name = "date"

monthly["percent_full"] = monthly_storage.reindex(full_index)

# Climate data is sourced independently from rainfall.csv, so it does
# not disappear merely because reservoir measurements are missing.
monthly = monthly.join(monthly_climate.reindex(full_index))

print("\nMonthly data summary")
print("Calendar months:", len(monthly))
print("First month:", monthly.index.min().date())
print("Last month:", monthly.index.max().date())

# Correct gap calculation: count absent calendar months between
# the first and last observed reservoir months.
observed_dates = monthly.index[monthly["percent_full"].notna()]
if len(observed_dates) > 1:
    observed_periods = observed_dates.to_period("M").asi8
    gaps_between_observations = int(
        np.maximum(np.diff(observed_periods) - 1, 0).sum()
    )
else:
    gaps_between_observations = 0

print("Observed reservoir months:", len(observed_dates))
print("Missing reservoir months:", int(monthly["percent_full"].isna().sum()))
print("Gaps between observed reservoir months:", gaps_between_observations)

# Climate variables must be available for all months used by the models.
if monthly[required_climate].isna().any().any():
    missing_climate_months = monthly.index[
        monthly[required_climate].isna().any(axis=1)
    ]
    raise ValueError(
        "Climate data has missing months. First missing months: "
        f"{list(missing_climate_months[:10].strftime('%Y-%m'))}. "
        "Check rainfall.csv before fitting SARIMAX."
    )

# Use climate values from the PREVIOUS month, not the target month.
exog = monthly[required_climate].shift(1)

# The first month has no preceding month's climate measurements.
monthly = monthly.iloc[1:].copy()
exog = exog.iloc[1:].copy()

if len(monthly) < 120:
    raise ValueError(
        "Not enough monthly observations for a seasonal model."
    )

if exog.isna().any().any():
    raise ValueError(
        "Lagged climate features contain missing values."
    )


# --------------------------------------------------
# 3. Chronological train/test split
# --------------------------------------------------

split_position = int(len(monthly) * 0.80)

train_index = monthly.index[:split_position]
test_index = monthly.index[split_position:]

y_train = monthly.loc[train_index, "percent_full"]
y_test = monthly.loc[test_index, "percent_full"]

X_train = exog.loc[train_index]
X_test = exog.loc[test_index]

print("\nTrain months:", len(train_index))
print("Test months:", len(test_index))
print("Training ends:", train_index[-1].date())
print("Testing starts:", test_index[0].date())

# Missing target months are deliberately kept as NaN in training.
# Statsmodels state-space models can handle missing endogenous values.
if y_train.notna().sum() < 60:
    raise ValueError(
        "Too few observed reservoir months in the training period."
    )


# --------------------------------------------------
# 4. Evaluation helper
# --------------------------------------------------

def calculate_metrics(actual, predicted, model_name):
    """Evaluate only dates where both actual and prediction exist."""
    actual = pd.Series(actual, dtype=float)
    predicted = pd.Series(predicted, dtype=float)

    aligned = pd.concat(
        [actual.rename("actual"), predicted.rename("predicted")],
        axis=1,
    ).dropna()

    if aligned.empty:
        raise ValueError(
            f"No valid evaluation observations for {model_name}."
        )

    mae = mean_absolute_error(
        aligned["actual"], aligned["predicted"]
    )
    rmse = np.sqrt(
        mean_squared_error(
            aligned["actual"], aligned["predicted"]
        )
    )

    print(f"\n{model_name}")
    print(f"MAE:  {mae:.3f} percentage points")
    print(f"RMSE: {rmse:.3f} percentage points")
    print(f"Evaluated months: {len(aligned)}")

    return {
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "test_months": len(aligned),
        "test_start": aligned.index.min().strftime("%Y-%m-%d"),
        "test_end": aligned.index.max().strftime("%Y-%m-%d"),
    }


# --------------------------------------------------
# 5. Monthly persistence baseline
# --------------------------------------------------

print("\nEvaluating monthly persistence baseline...")

# Predict each month using the previous calendar month's observed fullness.
# This can use the final observed training month for the first test month.
persistence_all = monthly["percent_full"].shift(1)
persistence_test = persistence_all.loc[test_index]


# --------------------------------------------------
# 6. Train SARIMA
# --------------------------------------------------

print("\nTraining SARIMA...")

sarima_model = SARIMAX(
    y_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False,
)

sarima_fit = sarima_model.fit(disp=False, maxiter=200)

sarima_forecast = sarima_fit.get_forecast(
    steps=len(test_index)
).predicted_mean

# Explicitly align predictions with the test calendar.
sarima_predictions = pd.Series(
    np.asarray(sarima_forecast),
    index=test_index,
    name="sarima_prediction",
)


# --------------------------------------------------
# 7. Train SARIMAX with lagged climate variables
# --------------------------------------------------

print("\nTraining SARIMAX with lagged climate features...")

sarimax_model = SARIMAX(
    y_train,
    exog=X_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False,
)

sarimax_fit = sarimax_model.fit(disp=False, maxiter=200)

sarimax_forecast = sarimax_fit.get_forecast(
    steps=len(test_index),
    exog=X_test,
).predicted_mean

sarimax_predictions = pd.Series(
    np.asarray(sarimax_forecast),
    index=test_index,
    name="sarimax_prediction",
)


# --------------------------------------------------
# 8. Compare models on EXACTLY the same test months
# --------------------------------------------------

predictions = pd.DataFrame(index=test_index)
predictions.index.name = "date"

predictions["actual_percent_full"] = y_test
predictions["persistence_prediction"] = persistence_test
predictions["sarima_prediction"] = sarima_predictions
predictions["sarimax_prediction"] = sarimax_predictions

# All three models must be scored on the same months.
common_mask = (
    predictions["actual_percent_full"].notna()
    & predictions["persistence_prediction"].notna()
    & predictions["sarima_prediction"].notna()
    & predictions["sarimax_prediction"].notna()
)

common = predictions.loc[common_mask]

if len(common) == 0:
    raise ValueError(
        "No common observed test months are available for comparison."
    )

print("\nCommon evaluation months:", len(common))
print("Common test start:", common.index.min().date())
print("Common test end:", common.index.max().date())

results = [
    calculate_metrics(
        common["actual_percent_full"],
        common["persistence_prediction"],
        "Monthly persistence baseline",
    ),
    calculate_metrics(
        common["actual_percent_full"],
        common["sarima_prediction"],
        "SARIMA",
    ),
    calculate_metrics(
        common["actual_percent_full"],
        common["sarimax_prediction"],
        "SARIMAX with lagged climate",
    ),
]

metrics_df = pd.DataFrame(results).sort_values("MAE")

metrics_path = PROCESSED_DIR / "timeseries_metrics.csv"
predictions_path = PROCESSED_DIR / "timeseries_predictions.csv"

metrics_df.to_csv(metrics_path, index=False)
predictions.reset_index().to_csv(predictions_path, index=False)

# Save fitted models for future use.
joblib.dump(sarima_fit, MODELS_DIR / "sarima_monthly.joblib")
joblib.dump(sarimax_fit, MODELS_DIR / "sarimax_monthly.joblib")

print("\n========== MODEL COMPARISON ==========")
print(metrics_df.to_string(index=False))

print("\nMetrics saved to:", metrics_path)
print("Predictions saved to:", predictions_path)
print("SARIMA saved to:", MODELS_DIR / "sarima_monthly.joblib")
print("SARIMAX saved to:", MODELS_DIR / "sarimax_monthly.joblib")

print("\nComparison completed!")

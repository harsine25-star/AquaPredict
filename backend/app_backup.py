from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "steinhagen_model.joblib"
DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "steinhagen_model_data.csv"
)

app = Flask(__name__)
CORS(app)

# Load the trained model and historical data
saved_model = joblib.load(MODEL_PATH)

if isinstance(saved_model, dict):
    model = saved_model.get("model")
    if model is None:
        raise ValueError(
            "No 'model' key found in the saved model dictionary."
        )
else:
    model = saved_model

history_df = pd.read_csv(DATA_PATH, parse_dates=["date"])
history_df = history_df.sort_values("date").reset_index(drop=True)

# These features must match the training script exactly.
LAGS = [1, 2, 3, 7, 14, 30]
FEATURES = (
    [f"lag_{lag}" for lag in LAGS]
    + ["day_of_year", "month"]
)

# Illustrative threshold, not an official warning standard.
LOW_STORAGE_THRESHOLD = 30.0


def create_features(values, forecast_date):
    """Create features in the same order used during training."""
    features = {
        f"lag_{lag}": values[-lag]
        for lag in LAGS
    }

    features["day_of_year"] = forecast_date.dayofyear
    features["month"] = forecast_date.month

    return pd.DataFrame([features], columns=FEATURES)


def generate_forecast(days=7):
    """Predict water availability for the next calendar days."""

    # Get the historical water availability values.
    values = history_df["percent_full"].astype(float).tolist()

    # Find today's date on the computer running Flask.
    today = pd.Timestamp.now().normalize()

    # The forecast starts tomorrow.
    forecast_start = today + pd.Timedelta(days=1)

    predictions = []

    for step in range(days):

        # Calculate the date for this prediction.
        forecast_date = forecast_start + pd.Timedelta(days=step)

        # Prepare the features required by the ML model.
        features = create_features(values, forecast_date)

        # Predict water availability.
        prediction = float(model.predict(features)[0])

        # Keep the percentage between 0 and 100.
        prediction = max(0.0, min(100.0, prediction))

        # Save the date and predicted percentage.
        predictions.append({
            "date": forecast_date.strftime("%Y-%m-%d"),
            "predicted_percent_full": round(prediction, 2),
        })

        # Use today's prediction to help predict the following day.
        values.append(prediction)

    return predictions


@app.route("/")
def home():
    return jsonify({
        "message": "Welcome to AquaPredict API",
        "status": "running",
    })


@app.route("/api/history")
def get_history():
    """Return historical reservoir fullness."""
    limit = request.args.get("limit", default=365, type=int)
    limit = max(1, min(limit, len(history_df)))

    recent = history_df.tail(limit)

    records = [
        {
            "date": row.date.strftime("%Y-%m-%d"),
            "percent_full": round(float(row.percent_full), 2),
        }
        for row in recent.itertuples(index=False)
    ]

    return jsonify({
        "reservoir": "B. A. Steinhagen",
        "count": len(records),
        "history": records,
    })


@app.route("/api/forecast")
def get_forecast():
    """Return predictions for the requested number of days."""
    days = request.args.get("days", default=7, type=int)

    if not 1 <= days <= 30:
        return jsonify({
            "error": "days must be between 1 and 30"
        }), 400

    predictions = generate_forecast(days)

    return jsonify({
        "reservoir": "B. A. Steinhagen",
        "forecast_days": days,
        "last_observed_date": (
            history_df["date"].iloc[-1].strftime("%Y-%m-%d")
        ),
        "forecast": predictions,
        "note": (
            "Predictions are model estimates, not guaranteed outcomes."
        ),
    })


@app.route("/api/metrics")
def get_metrics():
    """Return results from the chronological multi-day backtest."""
    return jsonify({
        "model": "Random Forest Regressor",
        "metric_unit": "percentage points",
        "backtest": {
            "1_day": {
                "model_mae": 1.427,
                "model_rmse": 2.077,
                "baseline_mae": 1.621,
                "baseline_rmse": 2.410,
            },
            "3_day": {
                "model_mae": 3.317,
                "model_rmse": 4.831,
                "baseline_mae": 3.650,
                "baseline_rmse": 5.429,
            },
            "7_day": {
                "model_mae": 3.893,
                "model_rmse": 6.755,
                "baseline_mae": 4.124,
                "baseline_rmse": 7.175,
            },
        },
        "test_period": {
            "start": "2011-10-05",
            "end": "2026-10-13",
        },
    })


@app.route("/api/alerts")
def get_alerts():
    """Generate alerts using the latest observation and forecast."""
    latest = float(history_df["percent_full"].iloc[-1])
    predictions = generate_forecast(7)

    alerts = []

    if latest < LOW_STORAGE_THRESHOLD:
        alerts.append({
            "type": "low_storage",
            "severity": "high",
            "message": (
                f"Latest observed fullness is {latest:.2f}%."
            ),
        })

    low_predictions = [
        item for item in predictions
        if item["predicted_percent_full"] < LOW_STORAGE_THRESHOLD
    ]

    if low_predictions:
        alerts.append({
            "type": "forecast_low_storage",
            "severity": "warning",
            "message": (
                "A forecast falls below the illustrative "
                f"{LOW_STORAGE_THRESHOLD:.0f}% threshold."
            ),
            "dates": [item["date"] for item in low_predictions],
        })

    return jsonify({
        "threshold_percent": LOW_STORAGE_THRESHOLD,
        "threshold_is_illustrative": True,
        "alerts": alerts,
        "alert_count": len(alerts),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
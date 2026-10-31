
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "steinhagen_model.joblib"

CLIMATE_MODEL_PATH = (
    BASE_DIR / "models" / "steinhagen_climate_model.joblib"
)

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "steinhagen_model_data.csv"
)

app = Flask(__name__)
CORS(app)


# ==========================================
# 2. LOAD ORIGINAL RANDOM FOREST MODEL
# ==========================================

saved_model = joblib.load(MODEL_PATH)

if isinstance(saved_model, dict):
    model = saved_model.get("model")

    if model is None:
        raise ValueError("Original model not found in saved file.")
else:
    model = saved_model


# ==========================================
# 3. LOAD CLIMATE RANDOM FOREST MODEL
# ==========================================

climate_saved_model = joblib.load(CLIMATE_MODEL_PATH)

climate_model = climate_saved_model["model"]

CLIMATE_FEATURES = climate_saved_model["features"]

CLIMATE_LAGS = climate_saved_model["lags"]

print("Climate model loaded successfully!")
print("Climate features:", CLIMATE_FEATURES)


# ==========================================
# 4. LOAD HISTORICAL RESERVOIR DATA
# ==========================================

history_df = pd.read_csv(DATA_PATH, parse_dates=["date"])

history_df = (
    history_df
    .sort_values("date")
    .reset_index(drop=True)
)

history_df["percent_full"] = pd.to_numeric(
    history_df["percent_full"],
    errors="coerce"
)

history_df = history_df.dropna(
    subset=["date", "percent_full"]
).reset_index(drop=True)


# ==========================================
# 5. ORIGINAL MODEL FEATURES
# ==========================================

LAGS = [1, 2, 3, 7, 14, 30]

FEATURES = (
    [f"lag_{lag}" for lag in LAGS]
    + ["day_of_year", "month"]
)

LOW_STORAGE_THRESHOLD = 30.0


# ==========================================
# 6. ORIGINAL FORECAST FEATURE FUNCTION
# ==========================================

def create_features(values, forecast_date):

    features = {
        f"lag_{lag}": values[-lag]
        for lag in LAGS
    }

    features["day_of_year"] = forecast_date.dayofyear
    features["month"] = forecast_date.month

    return pd.DataFrame(
        [features],
        columns=FEATURES
    )


# ==========================================
# 7. ORIGINAL FORECAST FUNCTION
# ==========================================

def generate_forecast(days=7):

    values = history_df["percent_full"].astype(float).tolist()

    today = pd.Timestamp.now().normalize()

    forecast_start = today + pd.Timedelta(days=1)

    predictions = []

    for step in range(days):

        forecast_date = (
            forecast_start + pd.Timedelta(days=step)
        )

        features = create_features(
            values,
            forecast_date
        )

        prediction = float(
            model.predict(features)[0]
        )

        prediction = max(
            0.0,
            min(100.0, prediction)
        )

        predictions.append({
            "date": forecast_date.strftime("%Y-%m-%d"),
            "predicted_percent_full": round(prediction, 2)
        })

        values.append(prediction)

    return predictions


# ==========================================
# 8. HOME API
# ==========================================

@app.route("/")
def home():

    return jsonify({
        "message": "Welcome to AquaPredict API",
        "status": "running"
    })


# ==========================================
# 9. HISTORICAL DATA API
# ==========================================

@app.route("/api/history")
def get_history():

    limit = request.args.get(
        "limit",
        default=365,
        type=int
    )

    limit = max(
        1,
        min(limit, len(history_df))
    )

    recent = history_df.tail(limit)

    records = [
        {
            "date": row.date.strftime("%Y-%m-%d"),
            "percent_full": round(
                float(row.percent_full), 2
            )
        }
        for row in recent.itertuples(index=False)
    ]

    return jsonify({
        "reservoir": "B. A. Steinhagen",
        "count": len(records),
        "history": records
    })


# ==========================================
# 10. ORIGINAL FORECAST API
# ==========================================

@app.route("/api/forecast")
def get_forecast():

    days = request.args.get(
        "days",
        default=7,
        type=int
    )

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
        "note": "Predictions are model estimates, not guaranteed outcomes."
    })


# ==========================================
# 11. MODEL PERFORMANCE API
# ==========================================

@app.route("/api/metrics")
def get_metrics():

    return jsonify({
        "model": "Random Forest Regressor",
        "metric_unit": "percentage points",

        "backtest": {
            "1_day": {
                "model_mae": 1.427,
                "model_rmse": 2.077,
                "baseline_mae": 1.621,
                "baseline_rmse": 2.410
            },

            "3_day": {
                "model_mae": 3.317,
                "model_rmse": 4.831,
                "baseline_mae": 3.650,
                "baseline_rmse": 5.429
            },

            "7_day": {
                "model_mae": 3.893,
                "model_rmse": 6.755,
                "baseline_mae": 4.124,
                "baseline_rmse": 7.175
            }
        },

        "test_period": {
            "start": "2011-10-05",
            "end": "2026-10-13"
        }
    })


# ==========================================
# 12. WATER SCARCITY ALERT API
# ==========================================

@app.route("/api/alerts")
def get_alerts():

    latest = float(
        history_df["percent_full"].iloc[-1]
    )

    predictions = generate_forecast(7)

    alerts = []

    if latest < LOW_STORAGE_THRESHOLD:

        alerts.append({
            "type": "low_storage",
            "severity": "high",
            "message": (
                f"Latest observed fullness is {latest:.2f}%."
            )
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
            "dates": [
                item["date"] for item in low_predictions
            ]
        })

    return jsonify({
        "threshold_percent": LOW_STORAGE_THRESHOLD,
        "threshold_is_illustrative": True,
        "alerts": alerts,
        "alert_count": len(alerts)
    })


# ==========================================
# 13. NEW RAINFALL SCENARIO PREDICTION API
# ==========================================

@app.route("/api/scenario-predict", methods=["POST"])
def scenario_predict():

    data = request.get_json(silent=True) or {}

    try:

        # Get user inputs
        rainfall = float(data["rainfall"])

        gross_evap = float(data["gross_evap"])

        current_full = float(data["current_fullness"])


        # Validate rainfall
        if not 0 <= rainfall <= 20:

            return jsonify({
                "error": "Rainfall must be between 0 and 20 inches."
            }), 400


        # Validate evaporation
        if not 0 <= gross_evap <= 20:

            return jsonify({
                "error": "Evaporation must be between 0 and 20 inches."
            }), 400


        # Validate reservoir fullness
        if not 0 <= current_full <= 100:

            return jsonify({
                "error": "Current fullness must be between 0 and 100."
            }), 400


        # Calculate net evaporation
        net_evap = gross_evap - rainfall


        # Get historical fullness values
        values = (
            history_df["percent_full"]
            .astype(float)
            .tolist()
        )

        if len(values) < max(CLIMATE_LAGS):

            return jsonify({
                "error": "Not enough historical data."
            }), 400


        # Replace latest historical value with user input
        values[-1] = current_full


        # Prediction date: next day after latest observation
        forecast_date = (
            history_df["date"].iloc[-1]
            + pd.Timedelta(days=1)
        )


        # Prepare historical lag features
        features = {
            f"storage_lag_{lag}": values[-lag]
            for lag in CLIMATE_LAGS
        }


        # Add calendar features
        features["day_of_year"] = forecast_date.dayofyear

        features["month"] = forecast_date.month


        # Add climate features
        features["precipitation"] = rainfall

        features["gross_evap"] = gross_evap

        features["net_evap"] = net_evap


        # Arrange features in exact training order
        input_df = pd.DataFrame(
            [[features[name] for name in CLIMATE_FEATURES]],
            columns=CLIMATE_FEATURES
        )


        # Generate prediction
        prediction = float(
            climate_model.predict(input_df)[0]
        )


        # Keep prediction between 0 and 100
        prediction = max(
            0.0,
            min(100.0, prediction)
        )


        # Return prediction
        return jsonify({

            "reservoir": "B. A. Steinhagen",

            "current_fullness": current_full,

            "rainfall_inches": rainfall,

            "gross_evaporation_inches": gross_evap,

            "net_evaporation_inches": round(
                net_evap, 3
            ),

            "predicted_fullness": round(
                prediction, 2
            ),

            "prediction_date": forecast_date.strftime(
                "%Y-%m-%d"
            ),

            "model": "Climate Random Forest",

            "note": (
                "This is a model estimate, "
                "not a guaranteed reservoir level."
            )

        })


    except KeyError as error:

        return jsonify({
            "error": f"Missing input: {error}"
        }), 400


    except (ValueError, TypeError):

        return jsonify({
            "error": "Please enter valid numerical values."
        }), 400


    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500


# ==========================================
# 14. RUN FLASK SERVER
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )

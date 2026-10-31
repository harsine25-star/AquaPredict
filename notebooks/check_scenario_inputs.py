
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "steinhagen_climate_model.joblib"

DATA_PATH = (
    ROOT / "data" / "processed" /
    "steinhagen_rainfall_data.csv"
)

# Load climate model
saved_model = joblib.load(MODEL_PATH)

print("========== MODEL INFORMATION ==========")

print("Model type:", saved_model.get("model_type"))
print("Training end date:", saved_model.get("training_end_date"))
print("Expected features:", saved_model.get("features"))
print("Lags:", saved_model.get("lags"))
print("Climate columns:", saved_model.get("climate_columns"))

# Load dataset
df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip().str.lower()

print("\n========== DATASET INFORMATION ==========")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

print("\nLast 5 records:")
print(df.tail(5).to_string(index=False))

print("\n========== CLIMATE DATA ==========")

for col in ["precipitation", "gross_evap", "net_evap"]:
    if col in df.columns:
        print("\n", col)
        print("Minimum:", df[col].min())
        print("Maximum:", df[col].max())
        print("Recent values:", df[col].tail(5).tolist())

print("\n========== DATE INFORMATION ==========")

df["date"] = pd.to_datetime(df["date"], errors="coerce")

print("First date:", df["date"].min())
print("Last date:", df["date"].max())


print("\n========== NET EVAPORATION CHECK ==========")

difference = (
    df["gross_evap"] - df["precipitation"] - df["net_evap"]
).abs()

print("Maximum difference:", difference.max())

if difference.max() < 0.001:
    print("Confirmed: net_evap = gross_evap - precipitation")
else:
    print("Formula needs further checking")


print("\nCHECK COMPLETED")

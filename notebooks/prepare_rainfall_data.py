
import pandas as pd
from pathlib import Path

# ==========================================
# 1. SET PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

reservoir_path = (
    ROOT / "data" / "raw" / "steinhagen_75years.csv"
)

rainfall_path = (
    ROOT / "data" / "raw" / "rainfall.csv"
)

output_path = (
    ROOT / "data" / "processed" / "steinhagen_rainfall_data.csv"
)

# ==========================================
# 2. LOAD RESERVOIR DATA
# ==========================================

print("Loading reservoir data...")

reservoir = pd.read_csv(reservoir_path)

reservoir["date"] = pd.to_datetime(
    reservoir["date"], errors="coerce"
)

reservoir["percent_full"] = pd.to_numeric(
    reservoir["percent_full"], errors="coerce"
)

reservoir = reservoir.dropna(
    subset=["date", "percent_full"]
).copy()

reservoir["year_month"] = reservoir["date"].dt.to_period("M")

print("Reservoir rows:", len(reservoir))
print("Reservoir start:", reservoir["date"].min())
print("Reservoir end:", reservoir["date"].max())

# ==========================================
# 3. LOAD RAINFALL DATA
# ==========================================

print("\nLoading rainfall data...")

rainfall = pd.read_csv(
    rainfall_path,
    comment="#"
)

# Standardize column names
rainfall.columns = (
    rainfall.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

print("Detected rainfall columns:", rainfall.columns.tolist())

# Check required columns
required_columns = [
    "period",
    "precipitation",
    "gross_evap",
    "net_evap"
]

missing_columns = [
    col for col in required_columns
    if col not in rainfall.columns
]

if missing_columns:
    raise ValueError(
        f"Missing rainfall columns: {missing_columns}. "
        f"Available columns: {rainfall.columns.tolist()}"
    )

# Convert monthly period to year-month
rainfall["period"] = rainfall["period"].astype(str).str.strip()

rainfall["year_month"] = pd.PeriodIndex(
    rainfall["period"], freq="M"
)

# Convert climate measurements to numeric
for col in ["precipitation", "gross_evap", "net_evap"]:
    rainfall[col] = pd.to_numeric(
        rainfall[col], errors="coerce"
    )

# Keep required columns and remove duplicate months
rainfall = rainfall[
    ["year_month", "precipitation", "gross_evap", "net_evap"]
].drop_duplicates(
    subset=["year_month"]
).copy()

print("Rainfall months:", len(rainfall))
print("Rainfall start:", rainfall["year_month"].min())
print("Rainfall end:", rainfall["year_month"].max())

# ==========================================
# 4. MERGE RESERVOIR AND RAINFALL DATA
# ==========================================

print("\nMerging datasets...")

merged = reservoir.merge(
    rainfall,
    on="year_month",
    how="left",
    validate="many_to_one"
)

print("Merged rows before cleaning:", len(merged))

print("\nMissing climate values before cleaning:")
print(
    merged[
        ["precipitation", "gross_evap", "net_evap"]
    ].isna().sum()
)

# ==========================================
# 5. REMOVE ROWS WITHOUT CLIMATE DATA
# ==========================================

merged = merged.dropna(
    subset=[
        "date",
        "percent_full",
        "precipitation",
        "gross_evap",
        "net_evap"
    ]
).copy()

# Sort chronologically
merged = merged.sort_values(
    "date"
).reset_index(drop=True)

# ==========================================
# 6. SAVE CLEAN DATASET
# ==========================================

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

merged.drop(
    columns=["year_month"],
    errors="ignore"
).to_csv(
    output_path,
    index=False
)

# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("\n==========================================")
print("DATA PREPARATION COMPLETED")
print("==========================================")

print("\nSaved file:")
print(output_path)

print("\nClean dataset shape:", merged.shape)

print("\nStart date:", merged["date"].min())
print("End date:", merged["date"].max())

print("\nMissing values in important columns:")
print(
    merged[
        ["percent_full", "precipitation",
         "gross_evap", "net_evap"]
    ].isna().sum()
)

print("\nLast 10 observations:")
print(
    merged[
        ["date", "percent_full", "precipitation",
         "gross_evap", "net_evap"]
    ].tail(10).to_string(index=False)
)

print("\nDataset saved successfully!")

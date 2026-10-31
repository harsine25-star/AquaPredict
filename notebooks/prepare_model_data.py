
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
project_dir = Path(__file__).resolve().parents[1]
raw_file = project_dir / "data" / "raw" / "steinhagen_75years.csv"
processed_dir = project_dir / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

# Load historical data
df = pd.read_csv(raw_file, comment="#")

# Prepare dates and sort chronologically
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "percent_full"])
df = df.sort_values("date").reset_index(drop=True)

# Keep only the fields needed for our first forecasting model
df = df[["date", "percent_full"]].copy()

# Check data quality
print("Rows:", len(df))
print("Date range:", df["date"].min(), "to", df["date"].max())
print("Duplicate dates:", df["date"].duplicated().sum())
print("Missing values:")
print(df.isnull().sum())

# Save prepared data
output_file = processed_dir / "steinhagen_model_data.csv"
df.to_csv(output_file, index=False)

# Plot historical percentage full
plt.figure(figsize=(12, 5))
plt.plot(df["date"], df["percent_full"], linewidth=0.7)
plt.title("B. A. Steinhagen Reservoir — Historical Storage")
plt.xlabel("Date")
plt.ylabel("Percent Full (%)")
plt.grid(True)
plt.tight_layout()

plot_file = processed_dir / "steinhagen_history.png"
plt.savefig(plot_file, dpi=150)
plt.close()

print("\nPrepared dataset saved:", output_file)
print("Historical graph saved:", plot_file)
print("\nLast 5 observations:")
print(df.tail().to_string(index=False))


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
project_dir = Path(__file__).resolve().parents[1]
raw_path = project_dir / "data" / "raw" / "statewide.csv"
processed_path = project_dir / "data" / "processed"
processed_path.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(raw_path)

# Convert date to datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Sort records by date
df = df.sort_values("date").reset_index(drop=True)

# Remove duplicate dates, if any
df = df.drop_duplicates(subset=["date"])

# Remove rows with invalid dates
df = df.dropna(subset=["date"])

# Save cleaned dataset
output_path = processed_path / "statewide_cleaned.csv"
df.to_csv(output_path, index=False)

print("Data preparation completed!")
print("Dataset shape:", df.shape)
print("Date range:", df["date"].min(), "to", df["date"].max())
print("Duplicate dates:", df["date"].duplicated().sum())
print("Saved to:", output_path)

# Plot percentage full over time
plt.figure(figsize=(12, 5))
plt.plot(df["date"], df["percent_full"], linewidth=0.8)
plt.title("Texas Reservoir Storage Percentage Over Time")
plt.xlabel("Date")
plt.ylabel("Percent Full")
plt.grid(True)
plt.tight_layout()

plot_path = processed_path / "storage_trend.png"
plt.savefig(plot_path)
plt.show()

print("Graph saved to:", plot_path)

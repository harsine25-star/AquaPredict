
from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parents[1]
csv_path = project_dir / "data" / "raw" / "statewide.csv"

df = pd.read_csv(csv_path)

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- FIRST FIVE ROWS ---")
print(df.head().to_string())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- NUMERIC SUMMARY ---")
print(df.describe().to_string())

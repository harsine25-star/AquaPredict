
from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parents[1]
raw_dir = project_dir / "data" / "raw"

files = [
    "statewide.csv",
    "steinhagen_1year.csv",
    "steinhagen_30day.csv",
    "steinhagen_75years.csv"
]

for filename in files:
    path = raw_dir / filename

    print("\n" + "=" * 60)
    print("FILE:", filename)

    if not path.exists():
        print("File not found!")
        continue

    try:
        df = pd.read_csv(path, comment="#")

        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())

        if "date" in df.columns:
            dates = pd.to_datetime(df["date"], errors="coerce")
            print("Start date:", dates.min())
            print("End date:", dates.max())
            print("Duplicate dates:", dates.duplicated().sum())

        if "percent_full" in df.columns:
            print("Missing percent_full:", df["percent_full"].isna().sum())

        print("\nFirst 3 rows:")
        print(df.head(3).to_string(index=False))

    except Exception as e:
        print("Error:", e)

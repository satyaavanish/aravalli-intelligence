import pandas as pd
import numpy as np
import os

# ----------------------------------
# 1. Setup & Paths
# ----------------------------------
INPUT_PATH = "data/raw/aravalli_dataset.csv"
OUTPUT_PATH = "data/processed/aravalli_cleaned.csv"
os.makedirs("data/processed", exist_ok=True)

print("🚀 Aravalli Intelligence: Starting Preprocessing & Time-Grid Alignment...")

# ----------------------------------
# 2. Load & Validate
# ----------------------------------
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"❌ Raw dataset not found at {INPUT_PATH}. Run Script 1 first.")

df = pd.read_csv(INPUT_PATH)
print(f"📊 Raw dataset shape: {df.shape}")

# Ensure all 14 Mission features from Script 1 are present
required_columns = [
    "grid_id", "year", "month", "lat", "lon", "elevation", "slope", 
    "NDVI", "BSI", "NDWI", "EVI", "LST", "nightlight", "LULC"
]
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"❌ Missing columns from GEE extraction: {missing_cols}")

# ----------------------------------
# 3. Rebuild the Continuous Time Grid
# ----------------------------------
# CRITICAL FOR SCRIPT 3: We must ensure every grid has exactly 60 continuous months.
# If GEE skipped a cloudy month, .diff(12) in Script 3 will align the wrong years.
print("📅 Rebuilding continuous time grid to protect rolling windows...")

df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))

start_date = df["date"].min()
end_date = df["date"].max()
all_grids = df["grid_id"].unique()
all_dates = pd.date_range(start=start_date, end=end_date, freq='MS') 

full_index = pd.MultiIndex.from_product([all_grids, all_dates], names=["grid_id", "date"])
df = df.set_index(["grid_id", "date"]).reindex(full_index).reset_index()

# Re-fill the year and month columns for the newly created empty rows
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# ----------------------------------
# 4. Interpolation & Imputation
# ----------------------------------
print("🔧 Interpolating Gaps & Formatting Categories...")

sensor_cols = ["NDVI", "NDWI", "BSI", "EVI", "LST", "nightlight"]
static_cols = ["lat", "lon", "elevation", "slope"]

# Kill "Ghost Grids" (Grids that are 100% NaN for the entire 5 years)
valid_grids = df.groupby('grid_id')['NDVI'].apply(lambda x: x.notna().any())
df = df[df['grid_id'].isin(valid_grids[valid_grids].index)]

# 1. Interpolate Continuous Sensors linearly across time
df[sensor_cols] = df.groupby("grid_id")[sensor_cols].transform(
    lambda x: x.infer_objects(copy=False).interpolate(method="linear").ffill().bfill()
)

# 2. Fill Static Terrain Data
df[static_cols] = df.groupby("grid_id")[static_cols].transform(lambda x: x.ffill().bfill())

# 3. Handle LULC (Categorical)
# DO NOT interpolate categories. Forward-fill the last known land cover.
df["LULC"] = df.groupby("grid_id")["LULC"].transform(lambda x: x.ffill().bfill())
# Force LULC to integer so Script 3's `isin([6, 7])` logic works perfectly
df["LULC"] = df["LULC"].round().astype(int)

# ----------------------------------
# 5. Scientific Bounds Clipping
# ----------------------------------
print("🧪 Applying Physical Limits to remove GEE anomalies...")
limits = {
    "NDVI": (-1, 1),
    "NDWI": (-1, 1),
    "BSI": (-1, 1),
    "EVI": (-1, 2.5),
    "LST": (-10, 65),  # Buffer for extreme Aravalli temperatures
    "nightlight": (0, 500)
}

for col, (low, high) in limits.items():
    df[col] = df[col].clip(lower=low, upper=high)

# ----------------------------------
# 6. Final Export for Script 3
# ----------------------------------
# We leave 'date' intact and do NOT scale here, exactly as Script 3 expects.
print(f"📊 Final Cleaned Dataset Shape: {df.shape}")
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Preprocessing Complete! Saved to: {OUTPUT_PATH}")
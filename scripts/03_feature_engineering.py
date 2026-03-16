import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import RobustScaler



print("🛠️ Aravalli Intelligence: Starting Next-Level Feature Engineering...")

# --------------------------------
# 0. Paths & Initialization
# --------------------------------
INPUT_PATH = "data/processed/aravalli_cleaned.csv"
OUTPUT_PATH = "data/features/aravalli_features.csv"

os.makedirs("data/features", exist_ok=True)

# --------------------------------
# 1. Load Dataset
# --------------------------------
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"❌ Preprocessed dataset not found at {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)

# Ensure time-series is perfectly ordered for rolling/diff math
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["grid_id", "date"]).reset_index(drop=True)

print(f"📊 Dataset loaded: {df.shape}")

# --------------------------------
# 2. Safety Bounds
# --------------------------------
if "NDVI" in df.columns:
    df["NDVI"] = df["NDVI"].clip(-1, 1)

# --------------------------------
# 3. Temporal Drift Features & LULC Transitions
# --------------------------------
print("⏱️ Calculating Temporal Drift, Volatility, and Slopes...")

if {"NDVI", "BSI"}.issubset(df.columns):
    # Year-over-Year change (bypasses seasonal leaf-drop)
    df["NDVI_yoy_diff"] = df.groupby("grid_id")["NDVI"].diff(12)
    df["BSI_yoy_diff"] = df.groupby("grid_id")["BSI"].diff(12)

    # MANDATORY FEATURE: Vegetation Trend Slopes (Velocity of Degradation)
    def calculate_slope(y):
        if len(y.dropna()) < 6: return 0.0
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0]
    
    df["ndvi_trend_slope"] = df.groupby("grid_id")["NDVI"].transform(
        lambda x: x.rolling(window=12, min_periods=6).apply(calculate_slope, raw=False)
    )

if "nightlight" in df.columns:
    # MANDATORY FEATURE: Nightlight Volatility (detects bursts of industrial/mining activity)
    df["light_volatility"] = df.groupby("grid_id")["nightlight"].transform(
        lambda x: x.rolling(window=6).std() / (x.rolling(window=6).mean() + 0.1)
    )

# Categorical Land Cover Transitions
if "LULC" in df.columns:
    print("🌍 Calculating Land Cover Transitions...")
    # Get the LULC from 12 months ago
    df["LULC_last_year"] = df.groupby("grid_id")["LULC"].shift(12)
    
    # Flag 1: Did the land change at all? (1 = yes, 0 = no)
    df["LULC_changed"] = (df["LULC"] != df["LULC_last_year"]).astype(int)
    
    # Flag 2: Did it turn into Bare Ground (7) or Built Area (6)?
    df["LULC_degraded"] = (
        (df["LULC"].isin([6, 7])) & 
        (~df["LULC_last_year"].isin([6, 7])) & 
        (df["LULC_last_year"].notna())
    ).astype(int)

# --------------------------------
# 4. Complex Index Ratios
# --------------------------------
print("🧮 Calculating Ratio Indices...")

if {"BSI", "NDVI"}.issubset(df.columns):
    # High Bare Soil + Low Vegetation = Deforestation/Mining
    df["inversion_index"] = df["BSI"] / (df["NDVI"].abs() + 0.05)

if {"LST", "NDVI"}.issubset(df.columns):
    # High Temp + Low Vegetation = Heat Island / Barren Land
    df["thermal_stress"] = df["LST"] / (df["NDVI"].abs() + 0.05)

# --------------------------------
# 5. Statistical Baselines (Hackathon Requirement)
# --------------------------------
print("📈 Calculating Regional & Historical Baselines...")

if {"NDVI", "BSI"}.issubset(df.columns):
    # Spatial Deviation: How abnormal is this grid compared to the rest of the map THIS month?
    regional_avg = df.groupby("date")[["NDVI", "BSI"]].transform("mean")
    df["ndvi_rel_to_region"] = df["NDVI"] - regional_avg["NDVI"]
    df["bsi_rel_to_region"] = df["BSI"] - regional_avg["BSI"]

    # Historical Z-Score: How abnormal is this month compared to this grid's OWN history?
    df["ndvi_historical_zscore"] = df.groupby("grid_id")["NDVI"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )

# --------------------------------
# 6. Combined Drift Score
# --------------------------------
components = []
for col in ["NDVI_yoy_diff", "BSI_yoy_diff", "inversion_index"]:
    if col in df.columns:
        components.append(abs(df[col]))

if components:
    df["drift_score"] = (
        abs(df.get("NDVI_yoy_diff", 0)) * 0.4 +
        abs(df.get("BSI_yoy_diff", 0)) * 0.4 +
        abs(df.get("inversion_index", 0)) * 0.2
    )

# --------------------------------
# 7. Impute Engineered NaNs
# --------------------------------
# .diff(12) and .rolling(12) create NaNs at the start of the series. 
# We fill these with 0 (implying "no drift/volatility yet")
drift_cols = [
    "NDVI_yoy_diff", "BSI_yoy_diff", "ndvi_trend_slope", "light_volatility", 
    "ndvi_historical_zscore", "drift_score", "LULC_last_year"
]
existing_drift_cols = [c for c in drift_cols if c in df.columns]
df[existing_drift_cols] = df[existing_drift_cols].fillna(0)

# Fill any remaining edge-case NaNs in ratios safely
df = df.fillna(0)

# --------------------------------
# 8. Outlier-Robust Scaling for ML
# --------------------------------
print("📏 Applying Robust Scaling for Unsupervised ML...")
scaler = RobustScaler()

# EXCLUDE spatial IDs and Categorical features from the math
cols_to_exclude = [
    "grid_id", "date", "year", "month", "lat", "lon", "elevation", "slope", 
    "LULC", "LULC_last_year", "LULC_changed", "LULC_degraded"
]
features_to_scale = [col for col in df.columns if col not in cols_to_exclude]

# Fit and transform the features
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# --------------------------------
# 9. Final Export
# --------------------------------
df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "="*30)
print("✅ Feature Engineering Complete!")
print(f"📁 Saved to: {OUTPUT_PATH}")
print(f"🛠️ Total Features: {df.shape[1]}")
print("="*30)
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest
import joblib

print("🚀 Aravalli Intelligence: Starting Forensic Drift Engine...")

# --------------------------------
# 1. Paths & Load Dataset
# --------------------------------
INPUT_PATH = "data/features/aravalli_features.csv"
OUTPUT_PATH = "outputs/drift_results.csv"
MODEL_PATH = "models/drift_model.pkl"

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError("❌ Feature dataset not found. Run Script 3 first.")

df = pd.read_csv(INPUT_PATH)
df["date"] = pd.to_datetime(df["date"])

# IMPORTANT: Sort for time-series persistence math
df = df.sort_values(["grid_id", "date"]).reset_index(drop=True)

print(f"📊 Dataset Loaded: {df.shape}")

# --------------------------------
# 2. Feature Selection (Aligned with Script 3)
# --------------------------------
# We feed the model exactly what the problem statement demanded: 
# Regional baselines, YoY differences, trend slopes, and volatility.
candidate_features = [
    "NDVI_yoy_diff",
    "BSI_yoy_diff",
    "ndvi_trend_slope",       # MANDATORY FEATURE ADDED
    "light_volatility",          # MANDATORY FEATURE ALIGNED
    "inversion_index",
    "thermal_stress",
    "ndvi_rel_to_region",
    "bsi_rel_to_region",
    "ndvi_historical_zscore",
    "drift_score",
    "LULC_changed",           
    "LULC_degraded"           
]

# Safety check to only use features that successfully generated
features = [f for f in candidate_features if f in df.columns]

if len(features) == 0:
    raise ValueError("❌ No valid features found for drift detection")

X = df[features].fillna(0)

print("⚙️ Brain loaded with Features:", features)

# --------------------------------
# 3. Train Unsupervised Isolation Forest
# --------------------------------
print("🌳 Training Unsupervised Isolation Forest (Detect Phase)...")


model = IsolationForest(
    n_estimators=300,
    contamination=0.04,  # Assumes ~4% of the Aravalli grid is experiencing severe illegal drift
    random_state=42,
    n_jobs=-1            # Uses all CPU cores
)

model.fit(X)

# --------------------------------
# 4. Predict Drift & Severity
# --------------------------------
df["drift_flag"] = model.predict(X)

# Convert labels: Isolation Forest uses 1=normal, -1=anomaly. We want 0=normal, 1=anomaly.
df["drift_flag"] = df["drift_flag"].map({1: 0, -1: 1})

# Decision function: negative implies outlier.
# We flip it and scale it 0 to 1 so '1.0' is the most extreme ecological destruction.
score = -model.decision_function(X)

df["drift_severity"] = (
    (score - score.min()) /
    (score.max() - score.min() + 1e-9)
)

# --------------------------------
# 5. The Persistence Filter (Differentiation Phase)
# --------------------------------
print("⏳ Applying Time-Series Persistence Filter...")
# Differentiation Requirement: To prove it isn't seasonal, the AI flag must persist for at least 2 out of 3 months.
df["persistence_count"] = df.groupby("grid_id")["drift_flag"].transform(
    lambda x: x.rolling(window=3, min_periods=1).sum()
)

df["is_confirmed"] = np.where(df["persistence_count"] >= 2, 1, 0)

# --------------------------------
# 6. Explainability Logic (The Dashboard Requirement)
# --------------------------------
print("🗣️ Generating Explainability Flags...")

# Define "extreme" behavior based on the statistical distribution of the mountains
thresholds = {
    "inv": df["inversion_index"].quantile(0.90) if "inversion_index" in df.columns else None,
    "ndvi_drop": df["NDVI_yoy_diff"].quantile(0.10) if "NDVI_yoy_diff" in df.columns else None,
    "trend": df["ndvi_trend_slope"].quantile(0.10) if "ndvi_trend_slope" in df.columns else None,
    "light": df["light_volatility"].quantile(0.95) if "light_volatility" in df.columns else None
}

def get_drift_reason(row):
    if row["is_confirmed"] == 0:
        return "Stable"

    reasons = []

    # 1. Physical land cover destruction
    if row.get("LULC_degraded", 0) == 1:
        reasons.append("Land Cover Transitioned to Built/Barren")

    # 2. Mathematical anomalies
    if thresholds["inv"] is not None and row.get("inversion_index", 0) > thresholds["inv"]:
        reasons.append("Bare Soil Inversion (Possible Mining)")

    if thresholds["ndvi_drop"] is not None and row.get("NDVI_yoy_diff", 0) < thresholds["ndvi_drop"]:
        reasons.append("Severe YoY Vegetation Loss")

    # 3. The Mandatory Features explicitly requested by the prompt
    if thresholds["trend"] is not None and row.get("ndvi_trend_slope", 0) < thresholds["trend"]:
        reasons.append("Vegetation Trend Slope heavily negative")

    if thresholds["light"] is not None and row.get("light_volatility", 0) > thresholds["light"]:
        reasons.append("Nightlight intensity volatility spiked drastically")

    if len(reasons) == 0:
        return "Complex Statistical Drift from Regional Baseline"

    return " | ".join(reasons)

# Apply explainability logic
df["drift_reason"] = [get_drift_reason(row) for _, row in df.iterrows()]

# --------------------------------
# 7. Final Save
# --------------------------------
df.to_csv(OUTPUT_PATH, index=False)
joblib.dump(model, MODEL_PATH)

print("\n✅ Drift Detection Complete")
print(f"📁 Results saved: {OUTPUT_PATH}")
print(f"🧠 Model saved: {MODEL_PATH}")

print("\n📊 Mission Summary")
print("-" * 20)
print(f"Total grid-months analyzed: {len(df)}")
print(f"Raw anomalies detected: {df['drift_flag'].sum()}")
print(f"Confirmed permanent drifts: {df['is_confirmed'].sum()}")
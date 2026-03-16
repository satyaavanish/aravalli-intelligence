import pandas as pd
import numpy as np
import os
import geopandas as gpd

print("🔮 Initializing 2026 Ecological Risk Forecaster...")

# ------------------------------------------------
# 1. Paths & Setup
# ------------------------------------------------
INPUT_PATH = "outputs/drift_results.csv"
OUTPUT_CSV_PATH = "outputs/alerts.csv"
OUTPUT_GEOJSON_PATH = "outputs/alerts.geojson" # NEW: GeoJSON for the dashboard

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError("❌ Run 04_drift_detection.py first!")

df = pd.read_csv(INPUT_PATH)
df['date'] = pd.to_datetime(df['date'])

# ------------------------------------------------
# 2. Identify At-Risk Grids
# ------------------------------------------------
# Find any grid that has an active confirmed threat in the most recent month
latest_date = df['date'].max()
current_threats = df[(df['date'] == latest_date) & (df['is_confirmed'] == 1)]

dangerous_grids = current_threats['grid_id'].unique()
alerts = []

if len(dangerous_grids) == 0:
    print("✅ No active, confirmed drift hotspots in the latest data.")
else:
    print(f"⚠️ Analyzing trajectories for {len(dangerous_grids)} high-risk regions...")
    
    for grid_id in dangerous_grids:
        # GET THE FULL TIMELINE for this grid
        grid_data = df[df['grid_id'] == grid_id].sort_values('date')
        
        recent_data = grid_data.tail(12)
        if len(recent_data) < 3:
            continue 
        
        # ------------------------------------------------
        # 3. Calculate Velocity (The "Speed" of Destruction)
        # ------------------------------------------------
        y = recent_data['NDVI'].values
        x = np.arange(len(y))
        
        slope, _ = np.polyfit(x, y, 1)
        
        current_ndvi = y[-1]
        current_sev = recent_data['drift_severity'].iloc[-1]
        
        # Grab current LULC to pass to the dashboard (if it exists)
        current_lulc = recent_data.get('LULC', pd.Series([0])).iloc[-1]
        
        # ------------------------------------------------
        # 4. 2026 Prediction: Points until collapse
        # ------------------------------------------------
        STATISTICAL_COLLAPSE_FLOOR = -2.5 
        
        if slope < -0.01: # Only project if it's actively degrading
            months_to_collapse = abs((current_ndvi - STATISTICAL_COLLAPSE_FLOOR) / slope)
        else:
            months_to_collapse = 99  # Stabilized or recovering

        # Risk Categorization
        if months_to_collapse <= 6:
            risk_level = "🚨 CRITICAL COLLAPSE"
        elif months_to_collapse <= 12:
            risk_level = "⚠️ HIGH RISK"
        else:
            risk_level = "🟡 ONGOING MONITORING"

        # ------------------------------------------------
        # 5. Build Dashboard Payload
        # ------------------------------------------------
        alerts.append({
            "grid_id": grid_id,
            "lat": recent_data['lat'].iloc[-1],
            "lon": recent_data['lon'].iloc[-1],
            "Severity_Score": round(current_sev * 100, 1),
            "Drift_Reason": recent_data['drift_reason'].iloc[-1],
            "Degradation_Velocity": round(slope, 4),
            "Months_Until_Barren": round(months_to_collapse, 1) if months_to_collapse != 99 else "Stable",
            "Forecast_2026": risk_level,
            "Current_LULC_Class": current_lulc
        })

# ------------------------------------------------
# 6. Save Final Alert Inventory & GeoJSON
# ------------------------------------------------
if alerts:
    alert_df = pd.DataFrame(alerts)
    alert_df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    # --- GEOPANDAS INTEGRATION (Hackathon Rubric Requirement) ---
    print("🌍 Generating Spatial GeoJSON using GeoPandas...")
    gdf = gpd.GeoDataFrame(
        alert_df, 
        geometry=gpd.points_from_xy(alert_df.lon, alert_df.lat),
        crs="EPSG:4326"
    )
    gdf.to_file(OUTPUT_GEOJSON_PATH, driver="GeoJSON")
    
    print(f"\n✅ Alert System Synced!")
    print(f"Total Active Threats Profiled: {len(alert_df)}")
    print(f"📁 CSV ready at: {OUTPUT_CSV_PATH}")
    print(f"🗺️ GeoJSON ready at: {OUTPUT_GEOJSON_PATH}")
else:
    headers = ["grid_id", "lat", "lon", "Severity_Score", "Drift_Reason", "Degradation_Velocity", "Months_Until_Barren", "Forecast_2026", "Current_LULC_Class"]
    empty_df = pd.DataFrame(columns=headers)
    empty_df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    # Create empty GeoJSON safely
    gpd.GeoDataFrame(empty_df, geometry=[], crs="EPSG:4326").to_file(OUTPUT_GEOJSON_PATH, driver="GeoJSON")
    print("ℹ️ Empty alert templates generated.")
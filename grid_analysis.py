import ee
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

import os
SERVICE_ACCOUNT = os.getenv("EE_SERVICE_ACCOUNT")
PRIVATE_KEY = os.getenv("EE_PRIVATE_KEY")

credentials = ee.ServiceAccountCredentials(
    SERVICE_ACCOUNT,
    key_data=PRIVATE_KEY
)

ee.Initialize(credentials)
# Aravalli bounding box
region = ee.Geometry.Rectangle([72.5, 26.5, 77.5, 29.5])


# -------------------------------------------------
# GRID
# -------------------------------------------------
def create_grid(region, rows=10, cols=10):

    bounds = region.bounds().coordinates().getInfo()[0]

    min_lon = bounds[0][0]
    min_lat = bounds[0][1]
    max_lon = bounds[2][0]
    max_lat = bounds[2][1]

    lon_step = (max_lon - min_lon) / cols
    lat_step = (max_lat - min_lat) / rows

    grid_cells = []

    for i in range(rows):
        for j in range(cols):
            cell = ee.Geometry.Rectangle([
                min_lon + j * lon_step,
                min_lat + i * lat_step,
                min_lon + (j + 1) * lon_step,
                min_lat + (i + 1) * lat_step
            ])
            grid_cells.append(cell)

    return grid_cells


# -------------------------------------------------
# DATE HANDLER
# -------------------------------------------------
def get_date_ranges():

    today = datetime.utcnow()

    latest_end = today
    latest_start = today - timedelta(days=90)

    prev_end = latest_start
    prev_start = latest_start - timedelta(days=90)

    comparison_days = (latest_end - latest_start).days

    return (
        latest_start.strftime("%Y-%m-%d"),
        latest_end.strftime("%Y-%m-%d"),
        prev_start.strftime("%Y-%m-%d"),
        prev_end.strftime("%Y-%m-%d"),
        comparison_days
    )


# -------------------------------------------------
# NDVI
# -------------------------------------------------
def get_ndvi_mean(geometry, start_date, end_date):

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .map(lambda img: img.normalizedDifference(['B8', 'B4']).rename("NDVI"))
    )

    if collection.size().getInfo() == 0:
        return None

    mean_img = collection.mean()

    stats = mean_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=500,
        maxPixels=1e9
    ).getInfo()

    return stats.get("NDVI")


# -------------------------------------------------
# NIGHTLIGHT
# -------------------------------------------------
def get_nightlight_mean(geometry, start_date, end_date):

    collection = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
    )

    if collection.size().getInfo() == 0:
        return None

    mean_img = collection.mean()

    stats = mean_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=1000,
        maxPixels=1e9
    ).getInfo()

    return stats.get("avg_rad")


# -------------------------------------------------
# SINGLE POINT ANALYSIS (Map Click)
# -------------------------------------------------
def analyze_single_point(lat, lon):

    geometry = ee.Geometry.Point([lon, lat]).buffer(500)

    latest_start, latest_end, prev_start, prev_end, comparison_days = get_date_ranges()

    ndvi_latest = get_ndvi_mean(geometry, latest_start, latest_end)
    ndvi_prev = get_ndvi_mean(geometry, prev_start, prev_end)

    nl_latest = get_nightlight_mean(geometry, latest_start, latest_end)
    nl_prev = get_nightlight_mean(geometry, prev_start, prev_end)

    if None in [ndvi_latest, ndvi_prev, nl_latest, nl_prev]:
        return {"error": "No satellite data available"}

    return {
    "lat": lat,
    "lon": lon,
    "delta_ndvi": float(ndvi_latest - ndvi_prev),
    "delta_nightlight": float(nl_latest - nl_prev),
    "ndvi_latest": float(ndvi_latest),
    "ndvi_previous": float(ndvi_prev),
    "nightlight_latest": float(nl_latest),
    "nightlight_previous": float(nl_prev),
    "latest_start": latest_start,
    "latest_end": latest_end,
    "previous_start": prev_start,
    "previous_end": prev_end,
    "comparison_days": comparison_days
    
}


# -------------------------------------------------
# REGIONAL ANALYSIS
# -------------------------------------------------
def run_analysis():

    grid = create_grid(region, rows=8, cols=8)

    latest_start, latest_end, prev_start, prev_end, comparison_days = get_date_ranges()

    features = []
    centers = []
    raw_values = []   # ✅ store original values per cell

    for cell in grid:

        ndvi_latest = get_ndvi_mean(cell, latest_start, latest_end)
        ndvi_prev = get_ndvi_mean(cell, prev_start, prev_end)

        nl_latest = get_nightlight_mean(cell, latest_start, latest_end)
        nl_prev = get_nightlight_mean(cell, prev_start, prev_end)

        if None in [ndvi_latest, ndvi_prev, nl_latest, nl_prev]:
            continue

        delta_ndvi = ndvi_latest - ndvi_prev
        delta_nightlight = nl_latest - nl_prev

        bounds = cell.bounds().coordinates().getInfo()[0]
        center_lat = (bounds[0][1] + bounds[2][1]) / 2
        center_lon = (bounds[0][0] + bounds[2][0]) / 2

        features.append([delta_ndvi, delta_nightlight])
        centers.append((center_lat, center_lon))

        # ✅ store full data per cell
        raw_values.append({
            "ndvi_latest": ndvi_latest,
            "ndvi_previous": ndvi_prev,
            "nightlight_latest": nl_latest,
            "nightlight_previous": nl_prev,
            "delta_ndvi": delta_ndvi,
            "delta_nightlight": delta_nightlight
        })

    if len(features) == 0:
        return []

    features = np.array(features)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    model = IsolationForest(contamination=0.15, random_state=42)
    model.fit(features_scaled)
    predictions = model.predict(features_scaled)

    results = []

    for raw, pred, center in zip(raw_values, predictions, centers):

        if pred == -1:

            delta_ndvi = raw["delta_ndvi"]
            delta_nightlight = raw["delta_nightlight"]

            center_lat, center_lon = center

            if delta_ndvi < -0.05 and delta_nightlight > 0.2:
                anomaly_type = "MAN_MADE_ACTIVITY"
            elif delta_ndvi < -0.05:
                anomaly_type = "VEGETATION_LOSS"
            elif delta_nightlight > 0.3:
                anomaly_type = "URBAN_GROWTH"
            else:
                anomaly_type = "STATISTICAL_ANOMALY"

            results.append({
                "lat": center_lat,
                "lon": center_lon,
                "delta_ndvi": float(delta_ndvi),
                "delta_nightlight": float(delta_nightlight),

                "ndvi_latest": float(raw["ndvi_latest"]),
                "ndvi_previous": float(raw["ndvi_previous"]),
                "nightlight_latest": float(raw["nightlight_latest"]),
                "nightlight_previous": float(raw["nightlight_previous"]),

                "latest_start": latest_start,
                "latest_end": latest_end,
                "previous_start": prev_start,
                "previous_end": prev_end,
                "comparison_days": comparison_days,

                "type": anomaly_type
            })

    return results
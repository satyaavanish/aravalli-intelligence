import ee
import os
import json
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
import ruptures as rpt
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta


# =========================================
# Google Earth Engine Authentication
# =========================================

SERVICE_ACCOUNT = os.getenv("EE_SERVICE_ACCOUNT")
PRIVATE_KEY = os.getenv("EE_PRIVATE_KEY")

if SERVICE_ACCOUNT and PRIVATE_KEY:
    credentials = ee.ServiceAccountCredentials(
        SERVICE_ACCOUNT,
        key_data=json.loads(PRIVATE_KEY)
    )
    ee.Initialize(credentials, project="aravalli-488205")
else:
    # fallback for local development
    ee.Initialize(project="aravalli-488205")

 

region = ee.Geometry.Rectangle([72.5, 26.5, 77.5, 29.5])

# ── LULC datasets 
LULC_CURRENT = ee.ImageCollection("ESA/WorldCover/v200").first()
LULC_PREVIOUS = ee.ImageCollection("ESA/WorldCover/v100").first()

# ── LULC class labels 
LULC_NAMES = {
    10:  "Tree cover",
    20:  "Shrubland",
    30:  "Grassland",
    40:  "Cropland",
    50:  "Built-up",
    60:  "Bare land",
    70:  "Snow/Ice",
    80:  "Water",
    90:  "Wetland",
    95:  "Mangroves",
    100: "Moss/Lichen",
}

# ── Adaptive grid resolution per LULC class  
LULC_RESOLUTION = {
    50:  (3, 3),
    60:  (3, 3),
    40:  (2, 2),
    10:  (2, 2),
    20:  (1, 2),
    30:  (1, 2),
    90:  (1, 1),
    80:  (1, 1),
    95:  (1, 1),
    70:  (1, 1),
    100: (1, 1),
}
DEFAULT_RESOLUTION = (1, 1)

# ── LULC transition risk table 
LULC_TRANSITION_SCORE = {
    (10, 50): 4,
    (10, 60): 4,
    (20, 60): 3,
    (30, 60): 3,
    (40, 50): 3,
    (10, 40): 2,
    (20, 50): 2,
    (30, 50): 2,
    (60, 50): 1,
    (90, 60): 3,
}



def compute_trend(values):
    x = np.arange(len(values))
    return float(np.polyfit(x, values, 1)[0])


def compute_volatility(values):
    return float(np.std(values))


def growth_rate_pct(current, previous):
    if not previous:
        return 0.0
    return ((current - previous) / abs(previous)) * 100.0


def _normalise_scores(scores):
    arr = np.asarray(scores, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.full(len(arr), 0.5)
    return 0.05 + (arr - mn) / (mx - mn) * 0.90


def _dbscan_proxy_scores(features_scaled, labels):
    core_pts = features_scaled[labels != -1]
    if len(core_pts) == 0:
        return np.zeros(len(features_scaled))
    return np.array([
        np.min(np.linalg.norm(core_pts - pt, axis=1))
        for pt in features_scaled
    ])



def _coarse_grid(region, rows=8, cols=8):
    bounds = region.bounds().coordinates().getInfo()[0]
    min_lon, min_lat = bounds[0][0], bounds[0][1]
    max_lon, max_lat = bounds[2][0], bounds[2][1]
    lon_step = (max_lon - min_lon) / cols
    lat_step = (max_lat - min_lat) / rows
    cells = []
    for i in range(rows):
        for j in range(cols):
            x0 = min_lon + j * lon_step
            y0 = min_lat + i * lat_step
            cells.append({
                "geom": ee.Geometry.Rectangle([x0, y0, x0 + lon_step, y0 + lat_step]),
                "min_lon": x0, "min_lat": y0,
                "lon_step": lon_step, "lat_step": lat_step,
            })
    return cells


def create_adaptive_grid(region, coarse_rows=8, coarse_cols=8):
    coarse_cells = _coarse_grid(region, coarse_rows, coarse_cols)
    coarse_geoms = [c["geom"] for c in coarse_cells]
    print("Adaptive grid — fetching coarse LULC ...")
    lulc_classes = _batch_lulc_mode(LULC_CURRENT, coarse_geoms, scale=100)
    grid_cells, centers = [], []
    for coarse, lulc_cls in zip(coarse_cells, lulc_classes):
        sub_r, sub_c = LULC_RESOLUTION.get(lulc_cls, DEFAULT_RESOLUTION)
        dx = coarse["lon_step"] / sub_c
        dy = coarse["lat_step"] / sub_r
        for si in range(sub_r):
            for sj in range(sub_c):
                x0 = coarse["min_lon"] + sj * dx
                y0 = coarse["min_lat"] + si * dy
                grid_cells.append(ee.Geometry.Rectangle([x0, y0, x0 + dx, y0 + dy]))
                centers.append((y0 + dy / 2, x0 + dx / 2))
    print(f"Adaptive grid: {len(grid_cells)} cells "
          f"(base {coarse_rows}x{coarse_cols}={coarse_rows * coarse_cols}, "
          f"expanded by LULC-driven subdivision)")
    return grid_cells, centers



def get_date_ranges():
    """
    Compute all three comparison windows and return as a dict.
    The dict is passed through to every anomaly result so the frontend
    can display exactly which dates are being compared.
    """
    today        = datetime.utcnow()
    latest_end   = today
    latest_start = today - timedelta(days=90)
    prev_end     = latest_start
    prev_start   = latest_start - timedelta(days=90)
    fmt = "%Y-%m-%d"

    latest_start_str = latest_start.strftime(fmt)
    latest_end_str   = latest_end.strftime(fmt)
    prev_start_str   = prev_start.strftime(fmt)
    prev_end_str     = prev_end.strftime(fmt)
    seas_start, seas_end = same_month_last_year(latest_start_str, latest_end_str)

    return {
        "latest_start":   latest_start_str,
        "latest_end":     latest_end_str,
        "prev_start":     prev_start_str,
        "prev_end":       prev_end_str,
        "seasonal_start": seas_start,
        "seasonal_end":   seas_end,
    }


def same_month_last_year(start_str, end_str):
    fmt   = "%Y-%m-%d"
    start = datetime.strptime(start_str, fmt) - timedelta(days=365)
    end   = datetime.strptime(end_str,   fmt) - timedelta(days=365)
    return start.strftime(fmt), end.strftime(fmt)




def _ndvi_image(start_date, end_date):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .map(lambda img: img.normalizedDifference(["B8", "B4"]).rename("NDVI"))
        .mean()
    )


def _nightlight_image(start_date, end_date):
    return (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterDate(start_date, end_date)
        .select("avg_rad")
        .mean()
    )



def _batch_reduce(image, grid_cells, band_name, scale):
    fc = ee.FeatureCollection([
        ee.Feature(cell, {"cell_idx": idx})
        for idx, cell in enumerate(grid_cells)
    ])

    def _reduce(f):
        return f.set(image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=f.geometry(),
            scale=scale,
            bestEffort=True,
            maxPixels=1e9,
        ))

    info    = fc.map(_reduce).getInfo()
    results = [None] * len(grid_cells)
    for feat in info["features"]:
        idx = feat["properties"]["cell_idx"]
        val = feat["properties"].get(band_name)
        results[idx] = float(val) if val is not None else None
    return results


def _batch_lulc_mode(image, geom_list, scale=100):
    fc = ee.FeatureCollection([
        ee.Feature(g, {"cell_idx": idx})
        for idx, g in enumerate(geom_list)
    ])

    def _reduce(f):
        return f.set(image.reduceRegion(
            reducer=ee.Reducer.mode(),
            geometry=f.geometry(),
            scale=scale,
            bestEffort=True,
            maxPixels=1e8,
        ))

    info    = fc.map(_reduce).getInfo()
    results = [None] * len(geom_list)
    for feat in info["features"]:
        idx = feat["properties"]["cell_idx"]
        val = feat["properties"].get("Map")
        results[idx] = int(val) if val is not None else None
    return results


def _batch_ndvi_timeseries(grid_cells, months=12):
    end_date = datetime.utcnow()
    windows = [
        (
            (end_date - timedelta(days=45 * (i + 1))).strftime("%Y-%m-%d"),
            (end_date - timedelta(days=45 * i)).strftime("%Y-%m-%d"),
        )
        for i in range(months - 1, -1, -1)
    ]
    series = [[] for _ in grid_cells]
    for w_start, w_end in windows:
        vals = _batch_reduce(_ndvi_image(w_start, w_end), grid_cells, "NDVI", scale=500)
        for idx, val in enumerate(vals):
            series[idx].append(val)
    return series


def _batch_nightlight_timeseries(grid_cells, months=12):
    end_date = datetime.utcnow()
    windows = [
        (
            (end_date - timedelta(days=30 * (i + 1))).strftime("%Y-%m-%d"),
            (end_date - timedelta(days=30 * i)).strftime("%Y-%m-%d"),
        )
        for i in range(months - 1, -1, -1)
    ]
    series = [[] for _ in grid_cells]
    for w_start, w_end in windows:
        vals = _batch_reduce(_nightlight_image(w_start, w_end), grid_cells, "avg_rad", scale=1000)
        for idx, val in enumerate(vals):
            series[idx].append(val)
    return series



# LULC TRANSITION


def encode_lulc_transition(prev, curr):
    if prev is None or curr is None:
        return 0
    return LULC_TRANSITION_SCORE.get((prev, curr), 0)



# ANOMALY DETECTION


def detect_anomalies(features_scaled, method="lof"):
    if method == "isolation_forest":
        mdl   = IsolationForest(contamination=0.12, n_estimators=200, random_state=42)
        mdl.fit(features_scaled)
        preds = mdl.predict(features_scaled)
        raw   = mdl.decision_function(features_scaled)
        conf  = _normalise_scores(-raw)
        return preds, raw, conf
    elif method == "lof":
        mdl   = LocalOutlierFactor(n_neighbors=10, contamination=0.12, metric="euclidean")
        preds = mdl.fit_predict(features_scaled)
        raw   = mdl.negative_outlier_factor_
        conf  = _normalise_scores(-raw)
        return preds, raw, conf
    elif method == "dbscan":
        mdl    = DBSCAN(eps=1.5, min_samples=5)
        labels = mdl.fit_predict(features_scaled)
        preds  = np.where(labels == -1, -1, 1)
        raw    = _dbscan_proxy_scores(features_scaled, labels)
        conf   = _normalise_scores(raw)
        return preds, raw, conf
    else:
        n = len(features_scaled)
        return np.ones(n), np.zeros(n), np.full(n, 0.5)



# CHANGE POINT DETECTION


def detect_change_point(series):
    if len(series) < 6:
        return False
    algo = rpt.Pelt(model="rbf").fit(np.array(series, dtype=float))
    return len(algo.predict(pen=2)) > 1



# EXPLANATION ENGINE


def generate_explanation(raw):
    reasons = []
    score   = 0.0

    if raw["delta_ndvi"] < -0.05:
        reasons.append(f"NDVI dropped by {abs(raw['delta_ndvi']):.3f} vs previous period")
        score += 0.15

    if raw["seasonal_delta_ndvi"] < -0.10:
        reasons.append(
            f"NDVI is {abs(raw['seasonal_delta_ndvi']):.3f} below same month last year "
            "(beyond seasonal variation)"
        )
        score += 0.20

    if raw["ndvi_trend"] < -0.005:
        reasons.append(
            f"Vegetation trend slope: {raw['ndvi_trend']:.4f}/step "
            "(long-term decline over 12 months)"
        )
        score += 0.15

    if raw["ndvi_mean"] < 0.25:
        reasons.append(
            f"Absolute NDVI mean {raw['ndvi_mean']:.3f} — bare/degraded land level"
        )
        score += 0.10

    if raw.get("change_point"):
        reasons.append("Structural break detected in NDVI time series (ruptures PELT)")
        score += 0.15

    if raw["delta_nightlight"] > 0.2:
        reasons.append(f"Nightlight radiance increased by {raw['delta_nightlight']:.3f} nW/cm2/sr")
        score += 0.15

    if raw["nightlight_growth_pct"] > 50:
        reasons.append(
            f"Nightlight growth rate: {raw['nightlight_growth_pct']:.1f}% "
            "vs previous 90-day period"
        )
        score += 0.15

    if raw["night_volatility_change_pct"] > 200:
        reasons.append(
            f"Nightlight volatility increased {raw['night_volatility_change_pct']:.1f}% "
            "(intermittent construction / extraction activity)"
        )
        score += 0.15

    if raw["lulc_change"]:
        severity = raw["lulc_transition_score"]
        reasons.append(
            f"Land cover changed: {raw['lulc_change']} "
            f"(transition severity {severity}/4)"
        )
        score += 0.05 * severity

    anomaly_type = "Environmental Change"
    if raw["delta_nightlight"] > 0.3 and raw["delta_ndvi"] < -0.05:
        anomaly_type = "Urban Expansion"
    elif raw["nightlight_growth_pct"] > 100 and raw["ndvi_mean"] < 0.2:
        anomaly_type = "Active Mining / Quarrying"
    elif raw["seasonal_delta_ndvi"] < -0.15 and raw["ndvi_trend"] < -0.01:
        anomaly_type = "Persistent Vegetation Loss"
    elif raw["delta_ndvi"] < -0.15:
        anomaly_type = "Vegetation Loss"
    elif raw["lulc_change"] and "Built-up" in raw["lulc_change"]:
        anomaly_type = "Urban Land Conversion"
    elif raw["lulc_change"] and "Bare land" in raw["lulc_change"]:
        anomaly_type = "Possible Mining Activity"
    elif raw.get("change_point") and raw["ndvi_trend"] < 0:
        anomaly_type = "Sudden Land Disturbance"

    return anomaly_type, min(0.95, score), reasons



# POST-PROCESSING

def apply_spatial_boost(results, radius_deg=0.3):
    """
    Boost confidence by 10% (cap 0.95) for anomalies that have at least one
    neighbour within radius_deg. Sets spatial_cluster=True on those results.

    Rationale: spatially-clustered anomalies are much more likely to represent
    real land-use change events than isolated statistical outliers.
    """
    coords = np.array([[r["lat"], r["lon"]] for r in results])
    for i, r in enumerate(results):
        neighbours = sum(
            1 for j in range(len(results))
            if i != j and np.sqrt(
                (coords[i][0] - coords[j][0]) ** 2 +
                (coords[i][1] - coords[j][1]) ** 2
            ) <= radius_deg
        )
        r["spatial_cluster"] = neighbours >= 1
        if r["spatial_cluster"]:
            r["confidence"] = round(min(0.95, r["confidence"] + 0.10), 4)
            r["intensity"]  = r["confidence"]
    return results


def assign_priority(confidence, anomaly_type):
    """
    HIGH  — confidence >= 0.65  OR  anomaly is mining/urban
    MEDIUM — confidence >= 0.40
    LOW    — otherwise
    """
    high_types = {"Active Mining / Quarrying", "Urban Expansion", "Urban Land Conversion"}
    if confidence >= 0.65 or anomaly_type in high_types:
        return "HIGH"
    if confidence >= 0.40:
        return "MEDIUM"
    return "LOW"



# MAIN ANALYSIS


def run_analysis(method="lof", ts_months=12):
    """
    Full pipeline. Returns:
    {
      "status":       "ok" | "no_data",
      "date_windows": { latest_start, latest_end, prev_start, prev_end,
                        seasonal_start, seasonal_end },
      "data":         [ anomaly_dict, ... ],   # sorted by confidence desc
      "reason":       str  # only present if status == "no_data"
    }

    Each anomaly_dict shape:
    {
      lat, lon, type, priority,
      anomaly_score, model_confidence, rule_confidence, confidence, intensity,
      spatial_cluster,
      date_windows: {
        current_period:     "YYYY-MM-DD -> YYYY-MM-DD",
        previous_period:    "YYYY-MM-DD -> YYYY-MM-DD",
        seasonal_baseline:  "YYYY-MM-DD -> YYYY-MM-DD"
      },
      indicators: { ndvi_mean, delta_ndvi, seasonal_delta_ndvi, ndvi_trend,
                    nightlight_mean, nightlight_growth_pct, night_vol_change_pct,
                    lulc_change, lulc_severity, change_point },
      explanation: [ str, ... ]
    }
    """

    # Grid
    grid_cells, centers = create_adaptive_grid(region)

    # Date windows
    dw = get_date_ranges()
    latest_start   = dw["latest_start"]
    latest_end     = dw["latest_end"]
    prev_start     = dw["prev_start"]
    prev_end       = dw["prev_end"]
    seasonal_start = dw["seasonal_start"]
    seasonal_end   = dw["seasonal_end"]

    print(f"\nDate windows:")
    print(f"  Current period   : {latest_start}  to  {latest_end}")
    print(f"  Previous period  : {prev_start}  to  {prev_end}")
    print(f"  Seasonal baseline: {seasonal_start}  to  {seasonal_end}  (same window, 1 year ago)")
    print()

    # Scalar indicators
    print("Fetching NDVI — latest period ...")
    ndvi_latest   = _batch_reduce(_ndvi_image(latest_start, latest_end),       grid_cells, "NDVI",    500)
    print("Fetching NDVI — previous period ...")
    ndvi_prev     = _batch_reduce(_ndvi_image(prev_start,   prev_end),         grid_cells, "NDVI",    500)
    print("Fetching NDVI — same month last year ...")
    ndvi_seasonal = _batch_reduce(_ndvi_image(seasonal_start, seasonal_end),   grid_cells, "NDVI",    500)
    print("Fetching nightlight — latest period ...")
    nl_latest     = _batch_reduce(_nightlight_image(latest_start, latest_end), grid_cells, "avg_rad", 1000)
    print("Fetching nightlight — previous period ...")
    nl_prev       = _batch_reduce(_nightlight_image(prev_start,   prev_end),   grid_cells, "avg_rad", 1000)

    # LULC
    print("Fetching LULC — current (ESA v200) ...")
    lulc_curr = _batch_lulc_mode(LULC_CURRENT,  grid_cells)
    print("Fetching LULC — previous (ESA v100) ...")
    lulc_prev = _batch_lulc_mode(LULC_PREVIOUS, grid_cells)

    # Time series
    print(f"Fetching NDVI time series ({ts_months} windows x 45 days) ...")
    ndvi_ts  = _batch_ndvi_timeseries(grid_cells, months=ts_months)
    print(f"Fetching nightlight time series ({ts_months} windows x 30 days) ...")
    night_ts = _batch_nightlight_timeseries(grid_cells, months=ts_months)

    # Feature matrix
    features      = []
    raw_values    = []
    valid_centers = []

    for idx in range(len(grid_cells)):
        nv_lat = ndvi_latest[idx]
        nv_prv = ndvi_prev[idx]
        nv_sea = ndvi_seasonal[idx]
        nl_lat = nl_latest[idx]
        nl_prv = nl_prev[idx]

        if None in [nv_lat, nv_prv, nl_lat, nl_prv]:
            continue
        if nv_sea is None:
            nv_sea = nv_prv

        ndvi_series  = [v for v in ndvi_ts[idx]  if v is not None]
        night_series = [v for v in night_ts[idx] if v is not None]
        if len(ndvi_series) < 6 or len(night_series) < 6:
            continue

        delta_ndvi          = nv_lat - nv_prv
        seasonal_delta_ndvi = nv_lat - nv_sea
        ndvi_mean           = float(np.mean(ndvi_series))
        ndvi_trend          = compute_trend(ndvi_series)
        ndvi_volatility     = compute_volatility(ndvi_series)
        change_pt           = detect_change_point(ndvi_series)

        delta_nightlight         = nl_lat - nl_prv
        nightlight_mean          = float(np.mean(night_series))
        nightlight_growth_pct    = growth_rate_pct(nl_lat, nl_prv)
        night_trend              = compute_trend(night_series)
        night_volatility         = compute_volatility(night_series)
        half                     = max(1, len(night_series) // 2)
        prev_night_vol           = compute_volatility(night_series[:half])
        night_vol_change_pct     = growth_rate_pct(night_volatility, prev_night_vol)

        lc  = lulc_curr[idx]
        lp  = lulc_prev[idx]
        lulc_score  = encode_lulc_transition(lp, lc)
        lulc_change = None
        if lp is not None and lc is not None and lp != lc:
            lulc_change = f"{LULC_NAMES.get(lp, 'Unknown')} -> {LULC_NAMES.get(lc, 'Unknown')}"

        ndvi_night_interaction = delta_ndvi * nightlight_growth_pct

        features.append([
            delta_ndvi,
            seasonal_delta_ndvi,
            ndvi_mean,
            ndvi_trend,
            ndvi_volatility,
            delta_nightlight,
            nightlight_mean,
            nightlight_growth_pct,
            night_trend,
            night_volatility,
            night_vol_change_pct,
            lulc_score,
            float(change_pt),
            ndvi_night_interaction,
        ])

        raw_values.append({
            "delta_ndvi":                  delta_ndvi,
            "seasonal_delta_ndvi":         seasonal_delta_ndvi,
            "ndvi_mean":                   ndvi_mean,
            "ndvi_trend":                  ndvi_trend,
            "ndvi_volatility":             ndvi_volatility,
            "delta_nightlight":            delta_nightlight,
            "nightlight_mean":             nightlight_mean,
            "nightlight_growth_pct":       nightlight_growth_pct,
            "night_trend":                 night_trend,
            "night_volatility":            night_volatility,
            "night_volatility_change_pct": night_vol_change_pct,
            "lulc_current":                lc,
            "lulc_previous":               lp,
            "lulc_change":                 lulc_change,
            "lulc_transition_score":       lulc_score,
            "change_point":                change_pt,
        })

        valid_centers.append(centers[idx])

    # Structured empty result
    if not features:
        print("No valid cells found.")
        return {
            "status": "no_data",
            "reason": (
                "No valid satellite cells were found for the requested date range. "
                "This is usually caused by high cloud cover or missing VIIRS data. "
                "Try adjusting the analysis window."
            ),
            "date_windows": dw,
            "data": [],
        }

    # ML detection
    features_scaled= StandardScaler().fit_transform(np.array(features))
    predictions, raw_scores, model_conf = detect_anomalies(features_scaled, method=method)

 
    results = []
    for raw, pred, center, raw_score, m_conf in zip(
        raw_values, predictions, valid_centers, raw_scores, model_conf
    ):
        if pred != -1:
            continue

        anomaly_type, rule_conf, explanation = generate_explanation(raw)
        confidence = float(np.clip(0.5 * m_conf + 0.5 * rule_conf, 0.05, 0.95))
        priority   = assign_priority(confidence, anomaly_type)

        results.append({
            "lat": round(center[0], 6),
            "lon": round(center[1], 6),
            "type":     anomaly_type,
            "priority": priority,

            "anomaly_score":    round(float(raw_score), 4),
            "model_confidence": round(float(m_conf),    4),
            "rule_confidence":  round(float(rule_conf), 4),
            "confidence":       round(confidence,       4),
            "intensity":        round(confidence,       4),

           
            "date_windows": {
                "current_period":    f"{latest_start} to {latest_end}",
                "previous_period":   f"{prev_start} to {prev_end}",
                "seasonal_baseline": f"{seasonal_start} to {seasonal_end}",
            },

            "indicators": {
                "ndvi_mean":             round(raw["ndvi_mean"], 4),
                "delta_ndvi":            round(raw["delta_ndvi"], 4),
                "seasonal_delta_ndvi":   round(raw["seasonal_delta_ndvi"], 4),
                "ndvi_trend":            round(raw["ndvi_trend"], 6),
                "nightlight_mean":       round(raw["nightlight_mean"], 4),
                "nightlight_growth_pct": round(raw["nightlight_growth_pct"], 2),
                "night_vol_change_pct":  round(raw["night_volatility_change_pct"], 2),
                "lulc_change":           raw["lulc_change"],
                "lulc_severity":         raw["lulc_transition_score"],
                "change_point":          raw["change_point"],
            },
            "explanation": explanation,
        })

    # Spatial boost
    if results:
        results = apply_spatial_boost(results, radius_deg=0.3)

    results.sort(key=lambda r: r["confidence"], reverse=True)

    print(f"\nAnalysis complete: {len(results)} anomalies detected "
          f"from {len(valid_centers)} valid cells.")

    return {
        "status": "ok",
        "date_windows": dw,
        "data": results,
    }


# SINGLE-POINT ANALYSIS


def analyze_single_point(lat: float, lon: float, buffer_m: int = 5000, ts_months: int = 12):
    """
    Analyse a single coordinate. Returns one anomaly dict (same schema as
    run_analysis()["data"] items), or None if data is insufficient.
    """
    point = ee.Geometry.Point([lon, lat])
    cell  = point.buffer(buffer_m).bounds()

    dw = get_date_ranges()
    latest_start   = dw["latest_start"]
    latest_end     = dw["latest_end"]
    prev_start     = dw["prev_start"]
    prev_end       = dw["prev_end"]
    seasonal_start = dw["seasonal_start"]
    seasonal_end   = dw["seasonal_end"]

    def _get(image, band, scale):
        return _batch_reduce(image, [cell], band, scale)[0]

    nv_lat = _get(_ndvi_image(latest_start, latest_end),     "NDVI",    500)
    nv_prv = _get(_ndvi_image(prev_start,   prev_end),       "NDVI",    500)
    nv_sea = _get(_ndvi_image(seasonal_start, seasonal_end), "NDVI",    500)
    nl_lat = _get(_nightlight_image(latest_start, latest_end), "avg_rad", 1000)
    nl_prv = _get(_nightlight_image(prev_start,   prev_end),   "avg_rad", 1000)

    if None in [nv_lat, nv_prv, nl_lat, nl_prv]:
        return None
    if nv_sea is None:
        nv_sea = nv_prv

    lc = _batch_lulc_mode(LULC_CURRENT,  [cell])[0]
    lp = _batch_lulc_mode(LULC_PREVIOUS, [cell])[0]

    ndvi_series  = [v for v in _batch_ndvi_timeseries([cell], months=ts_months)[0]  if v is not None]
    night_series = [v for v in _batch_nightlight_timeseries([cell], months=ts_months)[0] if v is not None]

    if len(ndvi_series) < 6 or len(night_series) < 6:
        return None

    delta_ndvi          = nv_lat - nv_prv
    seasonal_delta_ndvi = nv_lat - nv_sea
    ndvi_mean           = float(np.mean(ndvi_series))
    ndvi_trend          = compute_trend(ndvi_series)
    ndvi_volatility     = compute_volatility(ndvi_series)
    change_pt           = detect_change_point(ndvi_series)

    delta_nightlight         = nl_lat - nl_prv
    nightlight_mean          = float(np.mean(night_series))
    nightlight_growth_pct    = growth_rate_pct(nl_lat, nl_prv)
    night_trend              = compute_trend(night_series)
    night_volatility         = compute_volatility(night_series)
    half                     = max(1, len(night_series) // 2)
    prev_night_vol           = compute_volatility(night_series[:half])
    night_vol_change_pct     = growth_rate_pct(night_volatility, prev_night_vol)

    lulc_score  = encode_lulc_transition(lp, lc)
    lulc_change = None
    if lp is not None and lc is not None and lp != lc:
        lulc_change = f"{LULC_NAMES.get(lp, 'Unknown')} -> {LULC_NAMES.get(lc, 'Unknown')}"

    ndvi_night_interaction = delta_ndvi * nightlight_growth_pct

    raw = {
        "delta_ndvi": delta_ndvi, "seasonal_delta_ndvi": seasonal_delta_ndvi,
        "ndvi_mean": ndvi_mean, "ndvi_trend": ndvi_trend,
        "ndvi_volatility": ndvi_volatility,
        "delta_nightlight": delta_nightlight, "nightlight_mean": nightlight_mean,
        "nightlight_growth_pct": nightlight_growth_pct, "night_trend": night_trend,
        "night_volatility": night_volatility,
        "night_volatility_change_pct": night_vol_change_pct,
        "lulc_current": lc, "lulc_previous": lp,
        "lulc_change": lulc_change, "lulc_transition_score": lulc_score,
        "change_point": change_pt,
    }

    fv = np.array([[
        delta_ndvi, seasonal_delta_ndvi, ndvi_mean, ndvi_trend, ndvi_volatility,
        delta_nightlight, nightlight_mean, nightlight_growth_pct,
        night_trend, night_volatility, night_vol_change_pct,
        lulc_score, float(change_pt), ndvi_night_interaction,
    ]])

    raw_score  = float(np.linalg.norm(fv))
    model_conf = float(np.clip(1 / (1 + np.exp(-0.3 * (raw_score - 3))), 0.05, 0.95))

    anomaly_type, rule_conf, explanation = generate_explanation(raw)
    confidence = float(np.clip(0.5 * model_conf + 0.5 * rule_conf, 0.05, 0.95))
    priority   = assign_priority(confidence, anomaly_type)

    return {
        "lat": round(lat, 6), "lon": round(lon, 6),
        "type": anomaly_type, "priority": priority,
        "anomaly_score":    round(raw_score,   4),
        "model_confidence": round(model_conf,  4),
        "rule_confidence":  round(rule_conf,   4),
        "confidence":       round(confidence,  4),
        "intensity":        round(confidence,  4),
        "spatial_cluster":  False,
        "date_windows": {
            "current_period":    f"{latest_start} to {latest_end}",
            "previous_period":   f"{prev_start} to {prev_end}",
            "seasonal_baseline": f"{seasonal_start} to {seasonal_end}",
        },
        "indicators": {
            "ndvi_mean":             round(ndvi_mean, 4),
            "delta_ndvi":            round(delta_ndvi, 4),
            "seasonal_delta_ndvi":   round(seasonal_delta_ndvi, 4),
            "ndvi_trend":            round(ndvi_trend, 6),
            "nightlight_mean":       round(nightlight_mean, 4),
            "nightlight_growth_pct": round(nightlight_growth_pct, 2),
            "night_vol_change_pct":  round(night_vol_change_pct, 2),
            "lulc_change":           lulc_change,
            "lulc_severity":         lulc_score,
            "change_point":          change_pt,
        },
        "explanation": explanation,
    }
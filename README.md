# ⛰️ Aravalli Intelligence
### Ecological Drift Detection System for the Aravalli Hill Range

 

---

## 🌐 The Problem

The **Aravalli Hills**, one of the world's oldest mountain ranges and a critical green barrier for northern India, face escalating threats from illegal mining, urban encroachment, and deforestation. The challenge isn't just the damage — it's that early-stage degradation is nearly invisible until it's too late.

Conventional monitoring approaches either rely on manual surveys (slow and resource-intensive) or require pre-labeled datasets (unavailable for novel degradation patterns). Meanwhile, ecosystems continue to silently erode.

**Aravalli Intelligence** was built to close that gap — and to put ecological protection in the hands of everyone, from satellite analysts to citizens on the ground.

---

## 💡 What This System Does

Aravalli Intelligence is an **end-to-end geospatial ML pipeline** that ingests multi-source satellite data, engineers ecologically meaningful features, and applies unsupervised anomaly detection to flag regions showing abnormal environmental behavior — all without needing a single labeled training example.

The system distinguishes **natural seasonal variation** from **human-induced ecological drift** using a combination of vegetation indices, thermal signals, nightlight data, and land-cover transitions.

It then surfaces these findings through a **role-based dashboard** accessible to Forest Officers, Researchers, and Citizens alike.

---

## 👥 Who Is This System For?

Aravalli Intelligence is designed for three distinct user roles, each with a dedicated interface in the dashboard:

---

### 🛡️ Forest Officers
**Goal:** Receive targeted alerts and dispatch ground verification teams efficiently.

Forest Officers interact with the **Geospatial Alert Dispatcher** module, which:
- Displays AI-confirmed threat hotspots ranked by severity
- Automatically identifies the **nearest Forest Officer** to any flagged grid using Haversine distance calculation
- Generates a pre-filled **Enforcement Brief** with coordinates, drift trigger, and required action checklist
- Supports **CC dispatch** to District Collectors or Divisional Forest Officers
- Shows all **citizen-filed pending reports** for review and follow-up
- Lists all officers ranked by proximity to a given hotspot

**Typical workflow:**
```
View Active Alerts → Select Grid → Review AI Trigger → Dispatch Nearest Officer → Monitor Citizen Reports
```

---

### 🌍 Researchers & Analysts
**Goal:** Explore anomaly data, validate model outputs, and derive conservation insights.

Researchers access a full **Analytics Dashboard** with:
- Summary metric cards — total anomalies, Critical / High / Monitoring counts
- Interactive data table of all flagged grids with satellite feature values
- Spatial map visualization of alert distribution across the Aravalli range
- Feature-level analysis with histograms and descriptive statistics
- **CSV export** of the full alerts dataset for external GIS or notebook workflows

**Typical workflow:**
```
Review Metrics → Explore Spatial Map → Analyse Feature Distributions → Export for Research
```

---

### 📢 Citizen Reporters
**Goal:** Report suspicious ecological activity in real time, directly from the field.

Citizens can submit structured field reports through a dedicated portal that:
- Accepts **anonymous or named submissions** — no account required
- Captures category (illegal mining, burning, construction, dumping, tree felling, etc.), location, GPS coordinates, urgency level, and a free-text description
- **Auto-routes the report** to the nearest Forest Range Officer based on submitted coordinates
- Stores all reports persistently and surfaces them to officers for review
- Provides **emergency contact numbers** for Forest Helpline, Wildlife Crime Bureau, and state forest departments

**Typical workflow:**
```
Observe Threat → Open Citizen Portal → Fill Report → System Routes to Nearest Officer → Officer Reviews
```

> 💬 **Why citizen reports matter:** AI satellite detection operates on monthly update cycles. Human observers on the ground can detect and report threats within hours. These two layers together create a near-real-time ecological monitoring network.

---

## 🛰️ Satellite Data Sources

| Satellite | Data Type | Ecological Use |
|-----------|-----------|----------------|
| **Sentinel-2** | Multispectral Optical | Vegetation health, bare soil exposure |
| **VIIRS (Day/Night Band)** | Nighttime Radiance | Industrial activity, illegal excavation detection |
| **MODIS** | Thermal Infrared | Surface temperature, heat island mapping |
| **Dynamic World** | Land Cover (ML-based) | LULC transitions and degradation tracking |

> The Aravalli region is divided into **2,880 spatial grids** (5 km² each) at an elevation threshold of **300 meters**, ensuring only the ecologically relevant hill terrain is analyzed.

---

## 🧬 Feature Engineering

### Raw Satellite Features

| Feature | Source | Description |
|---------|--------|-------------|
| `NDVI` | Sentinel-2 | Vegetation density and health (range: -1 to 1) |
| `EVI` | Sentinel-2 | Enhanced vegetation index for high-biomass areas |
| `BSI` | Sentinel-2 | Bare Soil Index — detects exposed rock/mining surfaces |
| `LST` | MODIS | Land Surface Temperature in Kelvin/Celsius |
| `nightlight` | VIIRS | Artificial radiance — proxy for industrial activity |
| `LULC` | Dynamic World | Categorical land-cover class |

### Engineered Temporal & Drift Features

| Feature | Category | Purpose |
|---------|----------|---------|
| `NDVI_yoy_diff` | Temporal | Year-over-year vegetation change |
| `BSI_yoy_diff` | Temporal | Annual increase in exposed soil |
| `ndvi_trend_slope` | Temporal | Long-term vegetation trajectory |
| `light_volatility` | Industrial | Erratic night-light patterns (mining signature) |
| `thermal_stress` | Industrial | Temperature spike vs. vegetation decline |
| `inversion_index` | Industrial | Atmospheric thermal inversion indicator |
| `ndvi_rel_to_region` | Statistical | Local NDVI vs. regional baseline |
| `bsi_rel_to_region` | Statistical | Local BSI vs. regional baseline |
| `ndvi_historical_zscore` | Statistical | Deviation from historical mean |
| `drift_score` | Drift Logic | Composite anomaly signal |
| `LULC_changed` | Drift Logic | Boolean: land cover transition detected |
| `LULC_degraded` | Drift Logic | Boolean: transition towards degraded category |

---

## 🔍 Ecological Signatures Decoded

The feature engineering layer encodes expert domain knowledge into machine-readable signals:

### 🏗️ The Mining Signature
```
Night Light ↑  +  BSI ↑  +  NDVI ↓
```
Elevated artificial light at night combined with rising bare soil and falling vegetation health strongly indicates **continuous excavation or illegal stone quarrying**, even before LULC maps update.

### 🌡️ The Thermal Stress Signal
```
LST ↑  +  NDVI Trend Slope ↓
```
A ground temperature spike paired with a declining vegetation trend indicates **forest canopy removal** — the cooling effect of trees is measurably lost before visual deforestation is confirmed.

### 🌿 The False Season Check
```
Regional Baseline (High)  +  Local Z-Score (Low)
```
When the surrounding region is healthy but a single grid is underperforming statistically, it rules out seasonal drought and points to a **localized, human-caused ecological anomaly**.

---

## 🤖 Drift Detection Model

The core ML engine uses **Isolation Forest**, an unsupervised anomaly detection algorithm ideally suited for this problem:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Algorithm | Isolation Forest | No labeled data required |
| `n_estimators` | 300 | High ensemble size for stability |
| `contamination` | 4% | Assumed ~4% of grids show illegal drift |
| `n_jobs` | -1 | Full CPU parallelization |

**Why Isolation Forest?** Unlike supervised models, it learns the structure of "normal" environmental behavior from historical data, then isolates observations that deviate from that norm — precisely what ecological drift detection requires.

---

## 🚨 Risk Classification System

Detected anomalies are graded into three actionable tiers:

| Risk Level | Score Threshold | Interpretation | Recommended Response |
|------------|-----------------|----------------|----------------------|
| 🔴 **Critical Collapse** | Severity ≥ 75% | Severe, fast-moving degradation | Immediate ground intervention |
| 🟡 **High Risk** | Severity 45–74% | Strong degradation signal | Urgent field survey within 24 hrs |
| 🟢 **Monitoring** | Severity < 45% | Early-stage anomaly | Periodic satellite observation |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      ARAVALLI INTELLIGENCE                        │
│                    Ecological Drift Pipeline                      │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│  Google Earth   │────▶│  01_gee_extract  │  Sentinel-2, VIIRS,
│  Engine API     │     │  ion.py          │  MODIS, DynamicWorld
└─────────────────┘     └────────┬─────────┘
                                  │
                                  ▼
                        ┌──────────────────┐
                        │  02_preprocess   │  Cloud masking,
                        │  ing.py          │  interpolation,
                        └────────┬─────────┘  atm. correction
                                  │
                                  ▼
                        ┌──────────────────┐
                        │  03_feature_eng  │  NDVI, BSI, EVI,
                        │  ineering.py     │  temporal features,
                        └────────┬─────────┘  drift indicators
                                  │
                                  ▼
                        ┌──────────────────┐
                        │  04_drift_detec  │  Isolation Forest
                        │  tion.py         │  anomaly scoring
                        └────────┬─────────┘
                                  │
                                  ▼
                        ┌──────────────────┐
                        │  05_alert_syste  │  Risk classification,
                        │  m.py            │  alert generation
                        └────────┬─────────┘
                                  │
                                  ▼
                        ┌──────────────────┐
                        │  06_raster_expo  │  GeoJSON + GeoTIFF
                        │  rt.py           │  export
                        └────────┬─────────┘
                                  │
                                  ▼
              ┌───────────────────────────────────────┐
              │        Streamlit Dashboard            │
              │    Role-Based Multi-User Interface    │
              ├─────────────┬──────────┬──────────────┤
              │ 🛡️ Officer  │🌍 Analyst│ 📢 Citizen   │
              │  Dispatch   │Analytics │  Reporter    │
              └─────────────┴──────────┴──────────────┘
```

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.9+
- A [Google Earth Engine](https://earthengine.google.com/) account (free for research)
- Conda (strongly recommended for GIS dependency management)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/aravalli-intelligence.git
cd aravalli-intelligence
```

### 2. Create and Activate Environment

```bash
conda create -n aravalli python=3.9 -y
conda activate aravalli
```

### 3. Install GIS Dependencies (via Conda)

```bash
conda install -c conda-forge rasterio geopandas -y
```

### 4. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

### 5. Authenticate Google Earth Engine

```bash
earthengine authenticate
```

Follow the browser prompt to authorize your GEE account.

---

## ▶️ Running the Pipeline

Execute scripts in sequence for a full analysis run:

```bash
# Step 1: Extract satellite data from GEE
python scripts/01_gee_extraction.py

# Step 2: Clean and preprocess raw data
python scripts/02_preprocessing.py

# Step 3: Engineer ecological features
python scripts/03_feature_engineering.py

# Step 4: Run Isolation Forest drift detection
python scripts/04_drift_detection.py

# Step 5: Generate and classify risk alerts
python scripts/05_alert_system.py

# Step 6: Export GIS-ready outputs
python scripts/06_raster_export.py
```

---

## 📊 Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

On launch, select your role from the sidebar:

| Role | What You'll See |
|------|-----------------|
| 🛡️ Forest Officer | Alert dispatcher, proximity routing, citizen report review panel |
| 🌍 Researcher / Analyst | Full analytics dashboard, feature explorer, CSV export |
| 📢 Citizen Reporter | Field report submission form, auto-routing, emergency contacts |

---

## 📂 Project Structure

```
aravalli-intelligence/
│
├── scripts/
│   ├── 01_gee_extraction.py       # GEE satellite data retrieval
│   ├── 02_preprocessing.py        # Cloud masking, interpolation, correction
│   ├── 03_feature_engineering.py  # Ecological feature computation
│   ├── 04_drift_detection.py      # Isolation Forest model
│   ├── 05_alert_system.py         # Risk classification & alert generation
│   └── 06_raster_export.py        # GeoJSON / GeoTIFF export
│
├── data/                          # Raw and processed satellite data
├── models/                        # Trained Isolation Forest model artifacts
├── outputs/
│   ├── alerts.geojson             # AI-generated spatial alert file
│   ├── severity_map.tif           # Raster severity heatmap (GeoTIFF)
│   └── user_reports.json          # Citizen field reports (persistent store)
│
├── dashboard/
│   └── app.py                     # Streamlit multi-role dashboard
│
├── requirements.txt
└── README.md
```
## 📁 Dataset and Generated Outputs

Some files generated by this project can be very large (for example feature datasets and model outputs).  
GitHub has file size limits, so these large datasets are **not included in this repository**.

Examples of excluded files:

- `data/features/aravalli_features.csv`
- `outputs/drift_results.csv`

These files are automatically created when you run the pipeline.

To regenerate them, execute the pipeline scripts:

```bash
python scripts/01_gee_extraction.py
python scripts/02_preprocessing.py
python scripts/03_feature_engineering.py
python scripts/04_drift_detection.py
python scripts/05_alert_system.py
python scripts/06_raster_export.py
```

After running the pipeline, the required datasets and outputs will be generated inside the `data/` and `outputs/` folders.
---

## 🔭 Future Roadmap

- [ ] **Automated Monthly Runs** — scheduled pipeline execution via GitHub Actions or Apache Airflow
- [ ] **SAR Integration** — add Sentinel-1 Synthetic Aperture Radar for all-weather, cloud-penetrating monitoring
- [ ] **Change Point Detection** — CUSUM / PELT algorithms for precise temporal breakpoint identification
- [ ] **Multi-Region Scaling** — extend the framework to Western Ghats, Vindhya Range, and Deccan Plateau
- [ ] **RESTful API Layer** — real-time alert delivery endpoint for conservation organizations and government portals
- [ ] **Ground Truth Validation** — integration with field survey data for model calibration and accuracy benchmarking
- [ ] **Mobile Citizen App** — lightweight Android/iOS app for easier field reporting with GPS auto-fill
- [ ] **Officer Authentication** — role-based login so sensitive dispatch data is access-controlled

---

## 👥 Team

**Team Name:** Theppas Gang

| Name  
|------
| Kartheek  
| Avanish  
| Sai Baba  
| Sai Ram  

---
 

## 📜 License

This project is developed for **educational and research purposes only**. Not intended for commercial use. Satellite data accessed via Google Earth Engine is subject to [GEE Terms of Service](https://earthengine.google.com/terms/).

---

 
---

<div align="center">

*Three roles. One mission. Protect India's oldest mountain range — one satellite grid at a time.*

⛰️ **Aravalli Intelligence** | Theppas Gang

</div>

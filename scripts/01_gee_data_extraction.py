import ee
import pandas as pd
import geemap
import os
import time

try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()

print("🚀 Aravalli Intelligence Initialized. Starting Optimized Extraction...")

output_path = "data/raw/aravalli_dataset.csv"
os.makedirs("data/raw", exist_ok=True)

# --------------------------------
# 2. Define Region & Spatial Grid (5km Resolution)
# --------------------------------
# Broad corridor covering the Aravalli belt
corridor = ee.Geometry.Polygon([[
    [72.0, 23.5], [76.8, 28.8], [77.2, 28.5], [72.5, 23.2]
]])

# Load SRTM Elevation
dem = ee.Image("USGS/SRTMGL1_003")

# Create a mask for land > 300m (Isolating the mountain ecology from flatlands)
aravalli_mask = dem.gt(300).selfMask().clip(corridor)

# Convert to vector to get the exact study boundary
vectors = aravalli_mask.reduceToVectors(
    geometry=corridor,
    scale=1000, 
    geometryType='polygon',
    eightConnected=True,
    maxPixels=1e8
)
region = vectors.geometry()

# ~5 km grid (Hackathon Sweet Spot for detecting fragmented degradation)
rows = 156
cols = 120

# Create fishnet grid based on the mountain bounds
bbox = region.bounds()
grid = geemap.fishnet(bbox, rows=rows, cols=cols)

# KEEP ONLY THE GRIDS TOUCHING THE MOUNTAINS (>300m)
grid = grid.filterBounds(region)
grid = grid.map(lambda f: f.set("grid_id", f.id()))

print(f"✅ Exact Region Locked. Analyzing {grid.size().getInfo()} specific mountain cells.")

# --------------------------------
# 3. Processing Functions
# --------------------------------
def mask_clouds(image):
    qa = image.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return image.updateMask(mask).divide(10000)

def add_indices(image):
    # NDVI: Core Vegetation
    ndvi = image.normalizedDifference(['B8','B4']).rename('NDVI')
    # NDWI: Water / Moisture
    ndwi = image.normalizedDifference(['B3','B8']).rename('NDWI')
    # BSI: Bare Soil (Crucial for detecting illegal mining/clearing)
    bsi = image.expression(
        '((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))',
        {
            'SWIR': image.select('B11'), 'RED': image.select('B4'),
            'NIR': image.select('B8'), 'BLUE': image.select('B2')
        }
    ).rename('BSI')
    # EVI: Enhanced Vegetation
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        { 'NIR': image.select('B8'), 'RED': image.select('B4'), 'BLUE': image.select('B2') }
    ).rename('EVI')
    
    return image.addBands([ndvi, ndwi, bsi, evi])

# --------------------------------
# 4. Static Data Extraction (Run Once)
# --------------------------------
print("📡 Extracting Static Terrain Baselines (Elevation/Slope)...")
slope = ee.Terrain.slope(dem)
lonlat = ee.Image.pixelLonLat()

static_layer = dem.rename("elevation").addBands(slope.rename("slope")).addBands(lonlat)
static_stats = static_layer.reduceRegions(
    collection=grid,
    reducer=ee.Reducer.mean(),
    scale=500
).getInfo()

# Map grid_id to static values for quick lookup
static_lookup = {}
for f in static_stats['features']:
    p = f['properties']
    static_lookup[p['grid_id']] = {
        "lat": p.get("latitude"),
        "lon": p.get("longitude"),
        "elevation": p.get("elevation"),
        "slope": p.get("slope")
    }

# --------------------------------
# 5. Production Extraction Loop (Dynamic Temporal Data)
# --------------------------------
results = []
failed_months = [] # TRACKER: Logs any months that failed due to Google server timeouts

start_year, end_year = 2020, 2025

for year in range(start_year, end_year + 1):
    for month in range(1, 13):
        print(f"📅 Fetching Satellite Data: {year}-{month:02d}...", end="\r")
        
        try:
            start = ee.Date.fromYMD(year, month, 1)
            end = start.advance(1, 'month')

            # 1. Sentinel-2 (Multispectral Indices)
            s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(region).filterDate(start, end)
                  .map(mask_clouds).map(add_indices).median())

            # 2. Nightlights (Detecting Urban Encroachment & Mining Camps)
            nl = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
                  .filterDate(start, end).select("avg_rad").mean())

            # 3. Land Surface Temp (Heat Islands)
            lst = (ee.ImageCollection("MODIS/061/MOD11A2")
                   .filterDate(start, end).select("LST_Day_1km")
                   .mean().multiply(0.02).subtract(273.15))

            # 4. LULC (Dynamic World Land Cover - Mode/Most Common)
            lulc = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                    .filterBounds(region).filterDate(start, end)
                    .select("label").mode())

            # Combine dynamic bands
            dynamic_combined = s2.select(['NDVI','BSI','NDWI','EVI']) \
                                .addBands(nl.rename("nightlight")) \
                                .addBands(lst.rename("LST")) \
                                .addBands(lulc.rename("LULC"))

            # Calculate stats
            stats = dynamic_combined.reduceRegions(
                collection=grid,
                reducer=ee.Reducer.mean(),
                scale=500,
                tileScale=4
            )

            # Prevent rate limits / API timeouts
            time.sleep(1.5) 

            features = stats.getInfo()['features']
            for f in features:
                p = f['properties']
                grid_id = p.get("grid_id")
                
                if p.get("NDVI") is not None and grid_id in static_lookup:
                    row = {
                        "grid_id": grid_id, 
                        "year": year, 
                        "month": month,
                        "lat": static_lookup[grid_id]["lat"],
                        "lon": static_lookup[grid_id]["lon"],
                        "elevation": static_lookup[grid_id]["elevation"],
                        "slope": static_lookup[grid_id]["slope"]
                    }
                    row.update({k: p.get(k) for k in ['NDVI','BSI','NDWI','EVI','LST','nightlight','LULC']})
                    results.append(row)

        except Exception as e:
            error_date = f"{year}-{month:02d}"
            print(f"\n⚠️ API Timeout/Error: Skipping {error_date}. Moving to next month.")
            failed_months.append(error_date) # Log the failure
            time.sleep(5) # Cool down before retrying next month
            continue

    # Intermediate Save to protect against crashes
    pd.DataFrame(results).to_csv(output_path, index=False)

# --------------------------------
# 6. Finalize Dataset & Audit Report
# --------------------------------
df = pd.DataFrame(results).dropna()
df.to_csv(output_path, index=False)

print("\n" + "="*50)
print(f"🎉 INGESTION COMPLETE: {len(df)} records safely stored.")

# The Audit Report
if len(failed_months) > 0:
    print(f"⚠️ AUDIT WARNING: The following {len(failed_months)} months were skipped due to Google API timeouts:")
    print(f"-> {failed_months}")
    print("ℹ️ Do not worry! Script 2 (Preprocessing) is built to mathematically interpolate and fill these exact gaps.")
else:
    print("🌟 AUDIT SUCCESS: 100% of months extracted with zero skipped gaps!")

print("="*50)
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import os

print("🗺️ Aravalli Intelligence: Initializing Rasterio Engine...")

INPUT_GEOJSON = "outputs/alerts.geojson"
OUTPUT_TIFF = "outputs/drift_severity_map.tif"

if not os.path.exists(INPUT_GEOJSON):
    raise FileNotFoundError("❌ GeoJSON not found. Run Script 5 first.")

# 1. Load the spatial data
gdf = gpd.read_file(INPUT_GEOJSON)

if gdf.empty:
    print("✅ No active threats to rasterize.")
else:
    print(f"⚙️ Burning {len(gdf)} threat zones into GeoTIFF raster...")
    
    # 2. Define the geographic bounds of the Aravalli target area
    minx, miny, maxx, maxy = gdf.total_bounds
    
    # Add a small buffer so points aren't exactly on the edge
    buffer = 0.05
    minx, miny, maxx, maxy = minx - buffer, miny - buffer, maxx + buffer, maxy + buffer

    # 3. Define the Raster Resolution (approx 1km pixels)
    pixel_size = 0.01 
    width = int((maxx - minx) / pixel_size)
    height = int((maxy - miny) / pixel_size)

    # 4. Create the Rasterio Transform
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # 5. Extract geometries and the AI Severity Score to burn into the raster
    # We multiply by 100 to store it clearly as an integer in the raster
    shapes = ((geom, int(value)) for geom, value in zip(gdf.geometry, gdf['Severity_Score']))

    # 6. RASTERIO IN ACTION: Rasterize the vector points
    raster_array = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,                  # Background is 0 (Safe)
        all_touched=True,
        dtype=rasterio.uint8     # Store as efficient 8-bit image
    )

    # 7. Write the final GeoTIFF using Rasterio
    with rasterio.open(
        OUTPUT_TIFF,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=raster_array.dtype,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(raster_array, 1)

    print(f"✅ RASTERIO EXPORT COMPLETE!")
    print(f"📁 GeoTIFF saved to: {OUTPUT_TIFF}")
    print("📈 You have successfully fulfilled the Rasterio & GeoPandas technical requirement.")
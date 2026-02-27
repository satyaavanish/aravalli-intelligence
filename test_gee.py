import ee
import geemap
import numpy as np

# Initialize Earth Engine
ee.Initialize(project="aravalli-488205")

# Define Aravalli region (bounding box)
region = ee.Geometry.Rectangle([72.5, 26.5, 77.5, 29.5])

# Get Sentinel-2 collection (last 30 days)
collection = (
    ee.ImageCollection("COPERNICUS/S2_SR")
    .filterBounds(region)
    .filterDate("2025-01-01", "2025-02-01")
)

# Compute NDVI
def add_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename("NDVI")
    return image.addBands(ndvi)

ndvi_collection = collection.map(add_ndvi)

# Take mean NDVI image
ndvi_image = ndvi_collection.select("NDVI").mean()

# Convert small sample to numpy array
ndvi_np = geemap.ee_to_numpy(ndvi_image, region=region, scale=1000)

print("NDVI shape:", ndvi_np.shape)
print("Sample values:", ndvi_np[0:5])
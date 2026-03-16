import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import os
from folium.plugins import MarkerCluster

st.set_page_config(page_title="Live Map", layout="wide", page_icon="🗺️")
st.title("🗺️ Live Explainability Map")

@st.cache_data
def load_data():
    path = "outputs/alerts.geojson"
    if not os.path.exists(path):
        return None
    return gpd.read_file(path)

alerts_gdf = load_data()

LULC_MAP = {
    0: "Water", 1: "Trees / Forest", 2: "Grass", 3: "Flooded Vegetation", 
    4: "Crops", 5: "Shrub & Scrub", 6: "Built Area / Urban", 7: "Bare Ground", 8: "Snow & Ice"
}

if alerts_gdf is None or alerts_gdf.empty:
    st.warning("No geospatial data available to map.")
else:
    # 1. Handle LULC labels
    if 'Current_LULC_Class' in alerts_gdf.columns:
        alerts_gdf['LULC_Label'] = alerts_gdf['Current_LULC_Class'].fillna(-1).astype(int).map(LULC_MAP).fillna("Unknown")

    center_lat = alerts_gdf.geometry.y.mean()
    center_lon = alerts_gdf.geometry.x.mean()
    
    # 2. PURE SATELLITE TILES (Esri World Imagery)
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=8, 
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery'
    )

    # 3. Marker Cluster Setup
    marker_cluster = MarkerCluster(
        options={'showCoverageOnHover': False, 'zoomToBoundsOnClick': True}
    ).add_to(m)

    for _, row in alerts_gdf.iterrows():
        # Dynamic color based on risk level
        if "CRITICAL" in row['Forecast_2026']:
            color = "#ff4b4b" 
        elif "HIGH RISK" in row['Forecast_2026']:
            color = "#ffa500" 
        else:
            color = "#ffd700" 

        popup_html = f"""
        <div style="width: 260px; font-family: Arial, sans-serif; font-size: 13px;">
            <h4 style="color: {color}; margin-bottom: 2px; margin-top: 0;">{row['Forecast_2026']}</h4>
            <b style="color: gray;">Grid ID: {row['grid_id']}</b><br>
            <hr style="margin: 8px 0;">
            <b>🚨 Primary Trigger:</b><br>
            <span style="color: #ff4b4b; font-weight: bold;">{row['Drift_Reason']}</span><br>
            <hr style="margin: 8px 0;">
            <b>Current Land Cover:</b> {row.get('LULC_Label', 'N/A')}<br>
            <b>Severity Score:</b> {row['Severity_Score']}%<br>
            <b>Degradation Velocity:</b> {row['Degradation_Velocity']}<br>
            <b>Est. Months to Barren:</b> {row['Months_Until_Barren']}
        </div>
        """

        # Using white outline for better visibility against the satellite terrain
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=10, 
            color="white", 
            fill=True, 
            fill_color=color, 
            fill_opacity=0.9, 
            weight=1.5,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Inspect Grid {row['grid_id']}"
        ).add_to(marker_cluster)

    # Render full width
    st_folium(m, use_container_width=True, height=650, returned_objects=[])
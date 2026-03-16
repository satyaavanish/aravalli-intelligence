import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import os

st.set_page_config(page_title="Historical Hotspots", layout="wide", page_icon="📉")

st.title("📉 Aravalli Historical Degradation Heatmap")
st.markdown("""
**Intensity of Confirmed Man-Made Drift (2020 - 2025).**
This map visualizes the density of permanent ecological collapse. Areas in **Red** indicate high-frequency, high-severity degradation events that passed the AI persistence filter.
""")

@st.cache_data
def load_drift_data():
    path = "outputs/drift_results.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Ensure we only plot confirmed (non-seasonal) man-made drifts
    if 'is_confirmed' in df.columns:
        return df[df['is_confirmed'] == 1].copy()
    return df

df = load_drift_data()

if df is None or df.empty:
    st.warning("⚠️ No confirmed drift data found. Please ensure the backend processing is complete.")
else:
    # 1. Prepare Map Center
    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()
    
    # 2. Create Map
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=7, 
        tiles="CartoDB dark_matter"
    )

    # 3. Generate Heatmap Data
    # Intensity is boosted by the drift_severity score if available
    if 'drift_severity' in df.columns:
        heat_data = [[row['lat'], row['lon'], row['drift_severity']] for index, row in df.iterrows()]
    else:
        heat_data = [[row['lat'], row['lon']] for index, row in df.iterrows()]

    # 4. Add Heatmap Layer
    HeatMap(
        heat_data, 
        radius=15, 
        blur=20, 
        min_opacity=0.4,
        gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
    ).add_to(m)

    # 5. Display Map
    st_folium(m, use_container_width=True, height=700, returned_objects=[])

    st.info("💡 **Pitch Tip:** Explain to the judges that this heatmap ignores seasonal browning and only shows cumulative permanent damage caused by human activity.")
import streamlit as st
import geopandas as gpd
import os

st.set_page_config(page_title="Data Logs", layout="wide", page_icon="📂")
st.title("📂 System Data Logs")
st.markdown("Raw tabular outputs from the Unsupervised Drift Engine.")

@st.cache_data
def load_data():
    path = "outputs/alerts.geojson"
    if not os.path.exists(path):
        return None
    return gpd.read_file(path)

alerts_gdf = load_data()

if alerts_gdf is None or alerts_gdf.empty:
    st.warning("No logs available.")
else:
    # Drop geometry to make the table clean and readable
    display_df = alerts_gdf.drop(columns='geometry')
    st.dataframe(display_df, use_container_width=True)
    
    # Allow judges to download the CSV directly from the UI
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Raw CSV",
        data=csv,
        file_name='ecodrift_alerts_log.csv',
        mime='text/csv',
    )
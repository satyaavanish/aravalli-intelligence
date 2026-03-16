import streamlit as st
import streamlit.components.v1 as components
import geopandas as gpd
import pandas as pd
import os
import base64

# 1. Page Configuration
st.set_page_config(page_title="Aravalli Intelligence", layout="wide", page_icon="🏔️")

# 2. Initialize Session State
if 'started' not in st.session_state:
    st.session_state.started = False

# 3. Video Encoding (for local offline playback)
def get_video_base64(video_path):
    try:
        with open(video_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return ""

# --- VIEW LOGIC ---

if not st.session_state.started:
    # 4. Hide Sidebar and Streamlit UI elements for a clean Landing Page
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] {display: none;}
            [data-testid="stHeader"] {display: none;}
            .main .block-container {padding: 0;}
            iframe {border: none;}
        </style>
    """, unsafe_allow_html=True)

    video_base64 = get_video_base64("video.mp4")

    landing_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
        <style>
            * {{ margin:0; padding:0; box-sizing:border-box; font-family:'Inter',sans-serif; }}
            body {{ overflow:hidden; background-color: black; }}
            .background-video {{ position:fixed; top:0; left:0; width:100%; height:100%; object-fit:cover; z-index:-2; }}
            .overlay {{ position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.45); z-index:-1; }}
            .content {{ position:absolute; bottom:60px; right:120px; }}
            .get-started {{
                padding:18px 50px; font-size:22px; font-weight:600; letter-spacing:1px;
                border:none; border-radius:40px; color:white; cursor:pointer;
                background:linear-gradient(135deg,#22c55e,#16a34a);
                transition:all 0.35s ease;
                box-shadow: 0 10px 30px rgba(0,0,0,0.5), 0 0 15px rgba(34,197,94,0.4);
                animation:pulse 2.5s infinite;
            }}
            .get-started:hover {{ transform:translateY(-4px) scale(1.05); box-shadow: 0 20px 60px rgba(0,0,0,0.7); }}
            @keyframes pulse {{
                0% {{box-shadow:0 0 10px rgba(34,197,94,0.4);}}
                50% {{box-shadow:0 0 25px rgba(34,197,94,0.8);}}
                100% {{box-shadow:0 0 10px rgba(34,197,94,0.4);}}
            }}
        </style>
    </head>
    <body>
        <video autoplay muted loop playsinline class="background-video">
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        </video>
        <div class="overlay"></div>
        <div class="content">
            <button class="get-started" id="start-btn">Get Started</button>
        </div>
        <script>
            const btn = document.getElementById('start-btn');
            btn.addEventListener('click', () => {{
                window.parent.postMessage({{type: 'streamlit:setComponentValue', value: true}}, '*');
            }});
        </script>
    </body>
    </html>
    """
    
    # Use components.html to listen for the click event
    clicked = components.html(landing_html, height=1200)
    
    if clicked:
        st.session_state.started = True
        st.rerun()

else:
    # --- MAIN DASHBOARD VIEW (app.py content) ---
    
    # Optional: Sidebar button to go back
    if st.sidebar.button("🏠 Exit to Landing"):
        st.session_state.started = False
        st.rerun()

    st.title("🏔️ Aravalli Intelligence: Ecological Risk Forecaster")
    st.markdown("""
    **Automated anomaly detection for land degradation, immune to seasonal phenology.**
    This system ingests temporal satellite data (Sentinel-2, VIIRS Nightlights, MODIS Thermal, and Dynamic World LULC) to flag regions experiencing irreversible ecological collapse. 
    """)

    @st.cache_data
    def load_alert_data():
        path = "outputs/alerts.geojson"
        return gpd.read_file(path) if os.path.exists(path) else None

    @st.cache_data
    def load_raw_data():
        path = "outputs/drift_results.csv"
        return pd.read_csv(path) if os.path.exists(path) else None

    alerts_gdf = load_alert_data()
    raw_df = load_raw_data()

    st.markdown("---")
    st.markdown("### 📡 Real-Time Threat Assessment")

    if alerts_gdf is not None and raw_df is not None:
        st.markdown("#### 1. The Persistence Filter (Noise Reduction)")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_grids = raw_df['grid_id'].nunique()
        total_grid_months = len(raw_df)
        raw_anomalies = len(raw_df[raw_df['drift_flag'] == 1])
        unique_anomalous_grids = raw_df[raw_df['drift_flag'] == 1]['grid_id'].nunique()
        active_threats_count = len(alerts_gdf)
        
        col1.metric("Total Grids", f"{total_grids:,}")
        col2.metric("Grid-Months Analyzed", f"{total_grid_months:,}", delta="5 Years", delta_color="off")
        col3.metric("Raw Anomalies", f"{raw_anomalies:,}", delta="Monthly Events", delta_color="off")
        col4.metric("Unique Flags", f"{unique_anomalous_grids:,}", delta="Pre-Filter", delta_color="off")
        col5.metric("🔥 Active Man-Made Threats", f"{active_threats_count}", delta="Verified Signal")

        st.markdown("---")
        st.info("👈 **Use the sidebar to navigate to the Live Map and Data Explorer.**")
    else:
        st.error("Backend data missing in /outputs/")
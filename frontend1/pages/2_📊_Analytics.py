import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Forensic Analytics", layout="wide", page_icon="📊")

# Custom CSS to make Plotly charts feel more integrated
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔬 Forensic Analytics: Signal vs. Noise")

@st.cache_data
def load_data():
    alert_path = "outputs/alerts.geojson"
    raw_path = "outputs/drift_results.csv"
    
    alerts = gpd.read_file(alert_path) if os.path.exists(alert_path) else None
    raw = pd.read_csv(raw_path) if os.path.exists(raw_path) else None
    
    return alerts, raw

alerts_gdf, raw_df = load_data()

if alerts_gdf is None or raw_df is None:
    st.warning("Waiting for backend data files...")
else:
    # --- SECTION 1: SIGNAL EXTRACTION (Unique Grids vs Man-Made) ---
    st.markdown("### 📡 Unique Grid Filtering: Eliminating Seasonal Noise")
    
    # 1. Total unique grids that the AI flagged at least once
    unique_suspected_grids = raw_df[raw_df['drift_flag'] == 1]['grid_id'].nunique()
    
    # 2. Confirmed Man-Made (Permanent) Threats from your alerts file
    man_made_threats = alerts_gdf['grid_id'].nunique()
    
    # 3. Seasonal / False Positives
    seasonal_noise = unique_suspected_grids - man_made_threats

    comparison_df = pd.DataFrame({
        "Category": ["Man-Made (Confirmed)", "Seasonal Noise (Filtered)"],
        "Unique Grids": [man_made_threats, seasonal_noise],
        "Status": ["Permanent Threat", "Temporary/Seasonal"]
    })

    fig_funnel = px.bar(
        comparison_df, 
        x="Unique Grids", y="Category", 
        orientation='h',
        text="Unique Grids",
        color="Category",
        color_discrete_map={
            "Man-Made (Confirmed)": "#ff4b4b", 
            "Seasonal Noise (Filtered)": "#333333"
        },
        title="Intelligence Funnel: From 8000+ Flags to 40 Real Threats"
    )
    
    fig_funnel.update_layout(
        height=350, 
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white")
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

    st.markdown("---")

    # --- SECTION 2: EXPANDED PIE CHART (Primary Triggers) ---
    st.markdown("### 🚨 Root Cause Analysis of Man-Made Threats")
    
    reason_counts = alerts_gdf['Drift_Reason'].value_counts().reset_index()
    reason_counts.columns = ['Drift_Reason', 'Count']
    
    # Large Pie Chart
    fig_pie = px.pie(
        reason_counts, 
        values='Count', 
        names='Drift_Reason', 
        hole=0.5,
        color_discrete_sequence=px.colors.sequential.Reds_r,
        template="plotly_dark"
    )
    
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(
        height=700,  # Increased height significantly
        margin=dict(t=50, b=50, l=0, r=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=14)
        )
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)

    # --- SECTION 3: QUICK METRICS SUMMARY ---
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Anomalous Grids Found", unique_suspected_grids)
    m2.metric("Seasonal/Cloud Noise Removed", seasonal_noise)
    m3.metric("Confirmed Man-Made Hotspots", man_made_threats, delta="Actionable")
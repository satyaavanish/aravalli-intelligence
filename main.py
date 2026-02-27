from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import traceback

from grid_analysis import run_analysis, analyze_single_point

app = FastAPI(
    title="Aravalli Intelligence API",
    description="Real-Time Land Drift Detection System",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# GLOBAL CACHE (Lazy Loaded)
# ------------------------------

cached_results = None


# ------------------------------
# ROOT
# ------------------------------

@app.get("/")
def root():
    return {"status": "API running"}


# ------------------------------
# REGIONAL ANALYSIS (Lazy Load)
# ------------------------------

@app.get("/analyze")
def analyze():
    global cached_results

    try:
        # First time → compute
        if cached_results is None:
            print("Running regional analysis...")
            cached_results = run_analysis()
            print("Analysis complete.")

        return {
            "status": "success",
            "total_anomalies": len(cached_results),
            "data": cached_results
        }

    except Exception:
        return {
            "status": "error",
            "message": "Regional analysis failed",
            "trace": traceback.format_exc()
        }


# ------------------------------
# FORCE REFRESH
# ------------------------------

@app.get("/refresh")
def refresh_analysis():
    global cached_results

    try:
        print("Refreshing regional analysis...")
        cached_results = run_analysis()

        return {
            "status": "success",
            "message": "Analysis refreshed",
            "total_anomalies": len(cached_results)
        }

    except Exception:
        return {
            "status": "error",
            "message": "Refresh failed",
            "trace": traceback.format_exc()
        }


# ------------------------------
# SINGLE POINT ANALYSIS
# ------------------------------

@app.get("/analyze_point")
def analyze_point(lat: float = Query(...), lon: float = Query(...)):
    try:
        result = analyze_single_point(lat, lon)

        return {
            "status": "success",
            "data": result
        }

    except Exception:
        return {
            "status": "error",
            "message": "Point analysis failed",
            "trace": traceback.format_exc()
        }
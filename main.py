from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback
import math
import threading
import time
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from grid_analysis import run_analysis

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

# ============================================
# Request/Response Models
# ============================================

class PointRequest(BaseModel):
    lat: float
    lon: float

class IndicatorModel(BaseModel):
    ndvi_mean: Optional[float] = None
    delta_ndvi: Optional[float] = None
    seasonal_delta_ndvi: Optional[float] = None
    ndvi_trend: Optional[float] = None
    nightlight_mean: Optional[float] = None
    nightlight_growth_pct: Optional[float] = None
    night_vol_change_pct: Optional[float] = None
    lulc_change: Optional[str] = None
    lulc_severity: Optional[int] = None
    change_point: Optional[bool] = None

class DateWindowsModel(BaseModel):
    current_period: Optional[str] = None
    previous_period: Optional[str] = None
    seasonal_baseline: Optional[str] = None

class AnomalyResponse(BaseModel):
    lat: float
    lon: float
    type: str
    priority: str
    anomaly_score: Optional[float] = None
    model_confidence: Optional[float] = None
    rule_confidence: Optional[float] = None
    confidence: float
    intensity: Optional[float] = None
    date_windows: Optional[DateWindowsModel] = None
    indicators: IndicatorModel
    explanation: List[str] = []
    spatial_cluster: bool = False

# ============================================
# Global Cache with Update Tracking
# ============================================

cached_results: List[Dict[str, Any]] = []
analysis_complete = False
last_updated: Optional[datetime] = None
update_in_progress = False
update_stats = {
    "total_updates": 0,
    "successful_updates": 0,
    "failed_updates": 0,
    "last_error": None,
    "last_update_time": None
}

# ============================================
# Background Update Function
# ============================================

def perform_update():
    """Run analysis and update cache"""
    global cached_results, analysis_complete, last_updated, update_in_progress, update_stats
    
    if update_in_progress:
        print(f"⏳ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Update already in progress, skipping...")
        return False
    
    update_in_progress = True
    update_stats["total_updates"] += 1
    start_time = datetime.now()
    
    try:
        print(f"\n{'='*60}")
        print(f"🔄 AUTO-UPDATE STARTED at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # Run the analysis
        analysis_result = run_analysis()
        
        # Handle different return types
        if isinstance(analysis_result, dict):
            new_data = analysis_result.get("data", [])
        elif isinstance(analysis_result, list):
            new_data = analysis_result
        else:
            new_data = []
        
        # Filter to ensure we have dictionaries
        new_data = [item for item in new_data if isinstance(item, dict)]
        
        if new_data:
            cached_results = new_data
            analysis_complete = True
            last_updated = datetime.now()
            update_stats["successful_updates"] += 1
            update_stats["last_update_time"] = last_updated.isoformat()
            
            print(f"\n✅ AUTO-UPDATE COMPLETE!")
            print(f"   • Anomalies detected: {len(new_data)}")
            print(f"   • Update time: {(datetime.now() - start_time).total_seconds():.1f} seconds")
        else:
            print(f"\n⚠️ AUTO-UPDATE RETURNED NO DATA")
            update_stats["failed_updates"] += 1
            update_stats["last_error"] = "No data returned from analysis"
        
        update_in_progress = False
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ AUTO-UPDATE FAILED: {error_msg}")
        print(traceback.format_exc())
        update_stats["failed_updates"] += 1
        update_stats["last_error"] = error_msg
        update_in_progress = False
        return False

# ============================================
# Background Thread Scheduler
# ============================================

def background_updater():
    """Background thread that runs every 6 hours"""
    # Wait 2 minutes after startup before first update
    time.sleep(120)
    
    while True:
        try:
            print(f"\n⏰ Scheduled auto-update check at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            perform_update()
            
            # Sleep for 6 hours (21600 seconds)
            print(f"💤 Next auto-update scheduled in 6 hours...")
            time.sleep(6 * 60 * 60)
            
        except Exception as e:
            print(f"❌ Background updater error: {e}")
            # If error, wait 30 minutes and retry
            time.sleep(30 * 60)

# ============================================
# Startup Event
# ============================================

@app.on_event("startup")
def startup_event():
    """Run when API starts"""
    global cached_results, analysis_complete, last_updated
    
    print("\n" + "="*60)
    print("🚀 STARTING ARAVALLI INTELLIGENCE API")
    print("="*60)
    
    # Load initial data
    print("\n📥 Loading initial data...")
    try:
        analysis_result = run_analysis()
        
        if isinstance(analysis_result, dict):
            cached_results = analysis_result.get("data", [])
        elif isinstance(analysis_result, list):
            cached_results = analysis_result
        else:
            cached_results = []
        
        cached_results = [item for item in cached_results if isinstance(item, dict)]
        analysis_complete = True
        last_updated = datetime.now()
        
        print(f"✅ Initial data loaded: {len(cached_results)} anomalies")
        print(f"   • Time: {last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Failed to load initial data: {e}")
        print(traceback.format_exc())
        cached_results = []
        analysis_complete = False
    
    # Start background updater thread
    print("\n⏰ Starting background auto-updater...")
    updater_thread = threading.Thread(target=background_updater, daemon=True)
    updater_thread.start()
    print("✅ Auto-updater running (updates every 6 hours)")
    
    print("\n" + "="*60)
    print("✨ API READY - Data updates automatically every 6 hours")
    print("="*60 + "\n")

# ============================================
# API Endpoints
# ============================================

@app.get("/")
def root():
    return {
        "status": "API running",
        "version": "2.1.0",
        "last_updated": last_updated.isoformat() if last_updated else None,
        "auto_update": "Every 6 hours"
    }

# ============================================
# Get Analysis Results
# ============================================

@app.get("/analyze")
def analyze():
    """Get cached analysis results"""
    global cached_results, analysis_complete, last_updated

    try:
        if not analysis_complete or not cached_results:
            return {
                "status": "pending",
                "message": "Analysis still running or no data available",
                "total_anomalies": 0,
                "last_updated": None,
                "data": {
                    "data": [],
                    "date_windows": {
                        "latest_start": "2025-12-09",
                        "latest_end": "2026-03-09",
                        "prev_start": "2025-09-10",
                        "prev_end": "2025-12-09",
                        "seasonal_start": "2024-12-09",
                        "seasonal_end": "2025-03-09"
                    }
                }
            }

        return {
            "status": "success",
            "total_anomalies": len(cached_results),
            "last_updated": last_updated.isoformat() if last_updated else None,
            "data": {
                "data": cached_results,
                "date_windows": {
                    "latest_start": "2025-12-09",
                    "latest_end": "2026-03-09",
                    "prev_start": "2025-09-10",
                    "prev_end": "2025-12-09",
                    "seasonal_start": "2024-12-09",
                    "seasonal_end": "2025-03-09"
                }
            }
        }

    except Exception as e:
        print(f"❌ Analysis error: {str(e)}")
        print(traceback.format_exc())
        return {
            "status": "error",
            "message": "Failed to retrieve analysis",
            "trace": traceback.format_exc()
        }

# ============================================
# Manual Refresh Endpoint (Optional)
# ============================================

@app.post("/refresh")
def refresh_analysis(background_tasks: BackgroundTasks):
    """Manually trigger a data refresh (optional)"""
    if update_in_progress:
        return {
            "status": "busy",
            "message": "Auto-update already in progress",
            "last_updated": last_updated.isoformat() if last_updated else None
        }
    
    # Run in background
    background_tasks.add_task(perform_update)
    
    return {
        "status": "refresh_started",
        "message": "Manual refresh started in background",
        "last_updated": last_updated.isoformat() if last_updated else None
    }

# ============================================
# Update Status Endpoint
# ============================================

@app.get("/status")
def get_status():
    """Get cache and update status"""
    return {
        "status": "healthy" if analysis_complete else "starting",
        "total_anomalies": len(cached_results),
        "last_updated": last_updated.isoformat() if last_updated else None,
        "update_in_progress": update_in_progress,
        "cache_age_minutes": round((datetime.now() - last_updated).total_seconds() / 60, 1) if last_updated else None,
        "stats": update_stats,
        "next_update": "in 6 hours",
        "auto_update": True
    }

# ============================================
# Single Point Analysis
# ============================================

@app.post("/analyze-point")
def analyze_point(request: PointRequest):
    """Analyze a single location for environmental changes"""
    global cached_results, analysis_complete, last_updated
    
    try:
        if not analysis_complete or not cached_results:
            return {
                "lat": request.lat,
                "lon": request.lon,
                "type": "System Initializing",
                "priority": "LOW",
                "confidence": 0,
                "anomaly_score": None,
                "model_confidence": 0,
                "rule_confidence": 0,
                "intensity": 0,
                "spatial_cluster": False,
                "indicators": {
                    "ndvi_mean": None,
                    "delta_ndvi": None,
                    "seasonal_delta_ndvi": None,
                    "ndvi_trend": None,
                    "nightlight_mean": None,
                    "nightlight_growth_pct": None,
                    "night_vol_change_pct": None,
                    "lulc_change": None,
                    "lulc_severity": 0,
                    "change_point": False
                },
                "explanation": ["System initializing - please try again in a few minutes"],
                "date_windows": {
                    "current_period": "2025-12-09 to 2026-03-09",
                    "previous_period": "2025-09-10 to 2025-12-09",
                    "seasonal_baseline": "2024-12-09 to 2025-03-09"
                }
            }
        
        print(f"🔍 Searching for point near {request.lat}, {request.lon}")
        
        # Find the closest point
        closest_point = None
        min_distance = float('inf')
        
        for point in cached_results:
            if not isinstance(point, dict) or 'lat' not in point or 'lon' not in point:
                continue
                
            try:
                distance = math.sqrt(
                    (float(point['lat']) - request.lat) ** 2 + 
                    (float(point['lon']) - request.lon) ** 2
                )
                
                if distance < min_distance:
                    min_distance = distance
                    closest_point = point
            except:
                continue
        
        if closest_point and min_distance < 0.1:
            print(f"✅ Found point at distance {min_distance:.6f}°")
            return closest_point
        else:
            print(f"❌ No close point found. Closest: {min_distance:.6f}°")
            return {
                "lat": request.lat,
                "lon": request.lon,
                "type": "No Data Available",
                "priority": "LOW",
                "confidence": 0,
                "anomaly_score": None,
                "model_confidence": 0,
                "rule_confidence": 0,
                "intensity": 0,
                "spatial_cluster": False,
                "indicators": {
                    "ndvi_mean": None,
                    "delta_ndvi": None,
                    "seasonal_delta_ndvi": None,
                    "ndvi_trend": None,
                    "nightlight_mean": None,
                    "nightlight_growth_pct": None,
                    "night_vol_change_pct": None,
                    "lulc_change": None,
                    "lulc_severity": 0,
                    "change_point": False
                },
                "explanation": [f"Location not in analyzed grid (closest point was {min_distance:.4f}° away)"],
                "date_windows": {
                    "current_period": "2025-12-09 to 2026-03-09",
                    "previous_period": "2025-09-10 to 2025-12-09",
                    "seasonal_baseline": "2024-12-09 to 2025-03-09"
                }
            }
            
    except Exception as e:
        print(f"❌ Error in analyze_point: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# Debug Endpoint
# ============================================

@app.get("/debug/cache")
def debug_cache():
    """Debug endpoint to check cache contents"""
    if not cached_results:
        return {"message": "Cache is empty", "count": 0}
    
    sample = []
    for i, item in enumerate(cached_results[:5]):
        if isinstance(item, dict):
            sample.append({
                "index": i,
                "lat": item.get('lat'),
                "lon": item.get('lon'),
                "type": item.get('type'),
                "priority": item.get('priority'),
                "confidence": item.get('confidence')
            })
    
    return {
        "total_count": len(cached_results),
        "last_updated": last_updated.isoformat() if last_updated else None,
        "update_stats": update_stats,
        "sample": sample
    }
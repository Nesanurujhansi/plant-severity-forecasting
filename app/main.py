"""
Phase 10: FastAPI Backend Application
Provides high-performance, safe REST API endpoints for single-photo crop disease analysis,
model bundle metadata inspection, health monitoring, and microclimate weather lookups.
"""

import json
import logging
import os
import time
import uuid

logger = logging.getLogger("app.main")

# Disable telemetry threads that block sandbox shutdown
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.inference.loader import get_model_container
from app.inference.pipeline import InferencePipeline
from app.inference.upload_safety import UploadSafetyError
from app.inference.weather import resolve_weather
from app.schemas import (
    AnalyzeResponse, ErrorResponse, HealthResponse, ModelInfoResponse, WeatherQueryResponse
)

# Global pipeline instance initialized during lifespan startup
pipeline: Optional[InferencePipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager loading model container singleton at startup."""
    global pipeline
    dev_stub = settings.dev_stub_models or (os.getenv("DEV_STUB_MODELS", "0").lower() in ("1", "true", "yes"))
    container = get_model_container(bundle_dir=settings.bundle_dir, dev_stub=dev_stub, force_reload=True)
    pipeline = InferencePipeline(model_container=container)
    yield


app = FastAPI(
    title="crop-prognosis API",
    description=(
        "Decision support API for crop disease severity assessment, weather-aware epidemiological progression forecasting, "
        "and treatment scenario simulation. Provides non-prescription decision support only."
    ),
    version=settings.app_version,
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Simple In-Memory Rate Limiting Middleware
request_timestamps: Dict[str, List[float]] = {}

@app.middleware("http")
async def rate_limit_and_request_id_middleware(request: Request, call_next):
    # 1. Generate Request ID
    req_id = str(uuid.uuid4())
    request.state.request_id = req_id

    # 2. Rate Limiting Check (Exempt Gradio UI assets under /ui)
    if not request.url.path.startswith("/ui"):
        client_ip = request.client.host if request.client else "127.0.0.1"
        now = time.time()
        timestamps = request_timestamps.get(client_ip, [])
        # Keep timestamps within the last 60 seconds
        timestamps = [t for t in timestamps if now - t < 60.0]
        if len(timestamps) >= settings.rate_limit_per_minute:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded ({settings.rate_limit_per_minute} requests/minute).",
                    "hint": "Please wait a moment before sending more requests."
                },
                headers={"X-Request-ID": req_id}
            )
        timestamps.append(now)
        request_timestamps[client_ip] = timestamps

    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


# Exception Handler for Upload Safety Validation Errors
@app.exception_handler(UploadSafetyError)
async def upload_safety_exception_handler(request: Request, exc: UploadSafetyError):
    status_code = 413 if exc.error_code in ("FILE_TOO_LARGE", "DECOMPRESSION_BOMB") else status.HTTP_400_BAD_REQUEST
    return JSONResponse(
        status_code=status_code,
        content={
            "error_code": exc.error_code,
            "message": exc.message,
            "hint": exc.hint
        }
    )


# Standard HTTP Exception Handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": f"HTTP_{exc.status_code}",
            "message": exc.detail,
            "hint": "Refer to the API documentation at /docs for correct payload formats."
        }
    )


# GET /health
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Liveness check endpoint returning app version, stub status, and bundle version."""
    container = get_model_container()
    info = container.get_bundle_info()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        is_stub=container.is_stub,
        bundle_version=info.get("version", "unknown")
    )


# GET /model-info
@app.get("/model-info", response_model=ModelInfoResponse, tags=["Metadata"])
async def model_info():
    """Inspect active model bundle metadata, evaluation metrics, and scientific disclaimers."""
    container = get_model_container()
    info = container.get_bundle_info()
    
    if container.is_stub or container.manifest is None:
        return ModelInfoResponse(
            bundle_version="DEV-STUB",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            git_commit="dev-stub",
            is_stub=True,
            stub_banner=container.stub_banner,
            metrics_summary={},
            dataset_identifiers=[],
            disclaimer="[STUB MODE] Non-prescription decision support stub.",
            known_limitations=["Development stub mode active - not real predictions."]
        )

    manifest = container.manifest
    return ModelInfoResponse(
        bundle_version=manifest.bundle_version,
        created_at=manifest.created_at,
        git_commit=manifest.git_commit,
        is_stub=False,
        stub_banner=None,
        metrics_summary=manifest.metrics_summary,
        dataset_identifiers=[d.model_dump() for d in manifest.dataset_identifiers],
        disclaimer=manifest.disclaimer,
        known_limitations=manifest.known_limitations
    )


# POST /analyze and /api/analyze
@app.post("/analyze", response_model=AnalyzeResponse, tags=["Inference"])
@app.post("/api/analyze", response_model=AnalyzeResponse, tags=["Inference"])
async def analyze_leaf_photo(
    image: UploadFile = File(..., description="Single crop leaf photo (JPEG, PNG, or WebP format)"),
    crop: str = Form("wheat", description="Crop category (default: wheat)"),
    cultivar: Optional[str] = Form(None, description="Optional cultivar / genotype name"),
    days_since_heading: Optional[float] = Form(None, description="Optional growth stage days since heading"),
    latitude: Optional[float] = Form(None, description="Optional field latitude (-90 to 90)"),
    longitude: Optional[float] = Form(None, description="Optional field longitude (-180 to 180)"),
    weather_override: Optional[str] = Form(None, description="Optional JSON string with custom daily weather parameters"),
    weather_scenario: str = Form("live_nwp", description="Forward weather mode (default: live_nwp via Open-Meteo 9km API)"),
    horizons: str = Form("3,7", description="Comma-separated forecast horizons in days"),
    include_treatment_scenario: bool = Form(False, description="Legacy parameter retained for backward compatibility (default: false)")
):
    """
    Main disease prognosis endpoint: validates leaf upload, estimates continuous severity %,
    forecasts 3-day and 7-day progression, and calculates counterfactual treatment scenarios.
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Inference pipeline failed to initialize.")

    # Validate Geolocation Coordinates
    if latitude is not None and not (-90.0 <= latitude <= 90.0):
        raise UploadSafetyError(
            error_code="INVALID_LATITUDE",
            message=f"Latitude value '{latitude}' is out of valid range [-90.0, 90.0].",
            hint="Provide a valid latitude between -90.0 and 90.0 degrees."
        )
    if longitude is not None and not (-180.0 <= longitude <= 180.0):
        raise UploadSafetyError(
            error_code="INVALID_LONGITUDE",
            message=f"Longitude value '{longitude}' is out of valid range [-180.0, 180.0].",
            hint="Provide a valid longitude between -180.0 and 180.0 degrees."
        )

    # Parse Weather Override JSON if provided
    parsed_override: Optional[Dict[str, Any]] = None
    if weather_override and weather_override.strip():
        try:
            parsed_override = json.loads(weather_override)
        except Exception as e:
            raise UploadSafetyError(
                error_code="INVALID_WEATHER_JSON",
                message=f"Failed to parse weather_override JSON: {e}",
                hint="Ensure weather_override is a valid JSON string (e.g. {\"temp_c\": 20.0, \"rh_pct\": 80.0})."
            )

    # Parse horizons list
    try:
        horizon_list = [int(h.strip()) for h in horizons.split(",") if h.strip()]
    except Exception:
        horizon_list = [3, 7]

    image_bytes = await image.read()

    response_payload = pipeline.process(
        image_bytes=image_bytes,
        filename=image.filename or "upload.jpg",
        crop=crop,
        cultivar=cultivar,
        days_since_heading=days_since_heading,
        latitude=latitude,
        longitude=longitude,
        weather_override=parsed_override,
        weather_scenario=weather_scenario,
        horizons=horizon_list,
        include_treatment_scenario=include_treatment_scenario
    )

    return response_payload


# GET /weather
@app.get("/weather", response_model=WeatherQueryResponse, tags=["Weather"])
async def query_weather(
    latitude: float = Query(..., description="Latitude coordinate"),
    longitude: float = Query(..., description="Longitude coordinate"),
    scenario: str = Query("as_recent", description="Scenario preset")
):
    """Query recent daily weather data for a given geolocation coordinate."""
    res, warnings = resolve_weather(latitude=latitude, longitude=longitude, weather_scenario=scenario)
    return WeatherQueryResponse(
        latitude=latitude,
        longitude=longitude,
        source=res["source"],
        days_count=len(res["summary_table"]),
        records=res["summary_table"]
    )


# GET /api/resolve-maps-url & /resolve-maps-url
@app.get("/api/resolve-maps-url", tags=["Utilities"])
@app.get("/resolve-maps-url", tags=["Utilities"])
async def resolve_maps_url(url: str = Query(..., description="Google Maps URL to resolve")):
    """Follow Google Maps shortened redirects and extract exact latitude and longitude."""
    import re
    import requests
    try:
        resp = requests.get(
            url,
            allow_redirects=True,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        final_url = resp.url or url
        patterns = [
            r"search/(-?\d+\.\d+),\+?(-?\d+\.\d+)",
            r"@(-?\d+\.\d+),(-?\d+\.\d+)",
            r"[?&]q=(-?\d+\.\d+),(-?\d+\.\d+)",
            r"place/(-?\d+\.\d+),(-?\d+\.\d+)",
            r"ll=(-?\d+\.\d+),(-?\d+\.\d+)"
        ]
        for p in patterns:
            m = re.search(p, final_url)
            if m:
                return {
                    "status": "ok",
                    "latitude": float(m.group(1)),
                    "longitude": float(m.group(2)),
                    "destination_url": final_url
                }
        raise HTTPException(status_code=422, detail=f"Coordinates not found in resolved destination URL: {final_url}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to follow Google Maps redirect: {e}")


# POST /api/export-health-report-pdf & /export-health-report-pdf
@app.post("/api/export-health-report-pdf", tags=["Reports"])
@app.post("/export-health-report-pdf", tags=["Reports"])
async def export_health_report_pdf(payload: Dict[str, Any]):
    """Generate and return a formal 2-page A4 crop health advisory PDF report with all 4 overlays."""
    from app.inference.pdf_report import generate_crop_health_report_pdf
    try:
        pdf_bytes = generate_crop_health_report_pdf(payload)
        crop_name = str(payload.get("crop", "crop")).lower()
        diag_id = str(payload.get("diagnostic_id", "report")).replace(" ", "_")
        filename = f"Crop_Health_Guide_{crop_name}_{diag_id}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except Exception as e:
        logger.error(f"Failed to generate health report PDF: {e}")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")


# GET /
@app.get("/", include_in_schema=False)
async def root():
    """Root status endpoint pointing to Next.js frontend."""
    return {"status": "ok", "service": "crop-prognosis-api", "frontend": "http://localhost:3000", "docs": "/docs"}


# ==============================================================================
# OPTIONAL GRADIO UI MOUNTING AT /ui (Only if ENABLE_GRADIO=1)
# ==============================================================================
if os.getenv("ENABLE_GRADIO", "0").lower() in ("1", "true", "yes"):
    try:
        import gradio as gr
        from app.ui.gradio_ui import create_gradio_ui
        gradio_blocks_ui = create_gradio_ui()
        app = gr.mount_gradio_app(app, gradio_blocks_ui, path="/ui")
    except Exception as e:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

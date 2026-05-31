"""
FastAPI application entry-point for the Digit Prediction API.

Endpoints:
  POST /predict          — predict digit from image + metadata
  GET  /health           — liveness/readiness check
  GET  /models           — list available model versions
  GET  /metrics          — request count, error count, avg latency
  GET  /monitoring/stats — rolling prediction window statistics
  GET  /monitoring/drift — PSI + chi-squared drift report
  POST /retrain/trigger  — trigger a background retraining job (webhook)
  GET  /retrain/status   — status of the last retraining job
  GET  /retrain/log      — last N entries from the retrain audit log
"""

import logging
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from PIL import Image

from app.inference import DigitPredictor
from app.monitoring import DriftMonitor
from app.retrain_webhook import router as retrain_router
from app.schemas import HealthResponse, MetadataInput, PredictionResponse
from app.validation import DataValidationError, validate_image_bytes, validate_metadata

DEFAULT_MODEL_VERSION = "v1"  # fallback version when none is supplied by the caller

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Module-level singletons — loaded once at startup via the lifespan hook
predictor = DigitPredictor()
drift_monitor = DriftMonitor()

# --- In-memory request metrics ---
# A threading.Lock protects counter updates because the ASGI middleware
# can be invoked concurrently when uvicorn runs with multiple threads.
_metrics_lock = threading.Lock()
request_count = 0
error_count = 0
total_latency = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load default model artifacts before the server starts accepting traffic."""
    predictor.load()
    yield


app = FastAPI(title="Digit Prediction API", version="1.0.0", lifespan=lifespan)
app.include_router(retrain_router)  # mounts /retrain/* endpoints


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Track per-request latency, total request count, and error count."""
    global request_count, error_count, total_latency
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    with _metrics_lock:
        request_count += 1
        total_latency += duration
        if response.status_code >= 400:
            error_count += 1
    logger.info(
        "%s %s %d %.3fs",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness probe — returns 200 when the API is up and the model is loaded."""
    return HealthResponse(status="healthy", model_loaded=predictor.is_loaded)


@app.get("/models")
async def list_models():
    """Return all model version directories that contain a valid image_model.pth."""
    return {"versions": predictor.available_versions()}


@app.get("/metrics")
async def metrics():
    """Lightweight in-process metrics: total requests, errors, and average latency."""
    return {
        "request_count": request_count,
        "error_count": error_count,
        "avg_latency_s": round(total_latency / max(request_count, 1), 4),
    }


@app.get("/monitoring/stats")
async def monitoring_stats():
    """Raw rolling-window prediction counts for the drift monitor."""
    return drift_monitor.stats()


@app.get("/monitoring/drift")
async def monitoring_drift():
    """PSI and chi-squared drift report against the MNIST reference distribution."""
    return drift_monitor.drift_report()


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(..., description="Grayscale image of a handwritten digit"),
    pen_pressure: float = Form(...),
    writer_age: int = Form(...),
    handedness: str = Form(...),
    model_version: str = Form(
        DEFAULT_MODEL_VERSION, description="Model version to use"
    ),
):
    """
    Predict the handwritten digit in the uploaded image.

    - Validates metadata (pen_pressure, writer_age, handedness) → 422 on failure
    - Validates image bytes (size, dimensions, blank-image check) → 400 on failure
    - Returns predicted digit (0-9) and confidence score
    - Records the prediction in the drift monitor
    """
    # Validate metadata fields before touching the image
    try:
        clean_meta = validate_metadata(pen_pressure, writer_age, handedness)
    except DataValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Read image bytes and run validation + preprocessing (resize to 28x28, normalise)
    try:
        contents = await image.read()
        img_array = validate_image_bytes(contents)
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Run inference; raises FileNotFoundError if the requested version doesn't exist
    try:
        result = predictor.predict(img_array, clean_meta, version=model_version)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    drift_monitor.record(result["predicted_digit"])
    return PredictionResponse(**result)

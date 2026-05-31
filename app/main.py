"""Digit Prediction API — FastAPI application entry-point."""

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

DEFAULT_MODEL_VERSION = "v1"  # used when no model_version form field is sent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Singletons loaded once at startup
predictor = DigitPredictor()
drift_monitor = DriftMonitor()

# Lock protects metric counters under concurrent requests
_metrics_lock = threading.Lock()
request_count = 0
error_count = 0
total_latency = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts before accepting traffic."""
    predictor.load()
    yield


app = FastAPI(title="Digit Prediction API", version="1.0.0", lifespan=lifespan)
app.include_router(retrain_router)  # /retrain/* endpoints


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Record latency, request count and error count for every request."""
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
    """Liveness check."""
    return HealthResponse(status="healthy", model_loaded=predictor.is_loaded)


@app.get("/models")
async def list_models():
    """List available model versions."""
    return {"versions": predictor.available_versions()}


@app.get("/metrics")
async def metrics():
    """Return request count, error count and average latency."""
    return {
        "request_count": request_count,
        "error_count": error_count,
        "avg_latency_s": round(total_latency / max(request_count, 1), 4),
    }


@app.get("/monitoring/stats")
async def monitoring_stats():
    """Rolling prediction window stats."""
    return drift_monitor.stats()


@app.get("/monitoring/drift")
async def monitoring_drift():
    """PSI + chi-squared drift report vs MNIST reference distribution."""
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
    """Predict handwritten digit from image + metadata."""
    # Validate metadata → 422 on failure
    try:
        clean_meta = validate_metadata(pen_pressure, writer_age, handedness)
    except DataValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Validate image bytes, resize to 28×28, normalise → 400 on failure
    try:
        contents = await image.read()
        img_array = validate_image_bytes(contents)
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Run inference → 404 if requested version doesn't exist
    try:
        result = predictor.predict(img_array, clean_meta, version=model_version)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    drift_monitor.record(result["predicted_digit"])
    return PredictionResponse(**result)

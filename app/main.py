import logging
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from PIL import Image

from app.inference import DigitPredictor
from app.schemas import HealthResponse, MetadataInput, PredictionResponse

DEFAULT_MODEL_VERSION = "v1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

predictor = DigitPredictor()

# --- Metrics ---
request_count = 0
error_count = 0
total_latency = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()
    yield


app = FastAPI(title="Digit Prediction API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    global request_count, error_count, total_latency
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
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
    return HealthResponse(status="healthy", model_loaded=predictor.is_loaded)


@app.get("/models")
async def list_models():
    return {"versions": predictor.available_versions()}


@app.get("/metrics")
async def metrics():
    return {
        "request_count": request_count,
        "error_count": error_count,
        "avg_latency_s": round(total_latency / max(request_count, 1), 4),
    }


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
    # Validate metadata via schema
    try:
        metadata = MetadataInput(
            pen_pressure=pen_pressure,
            writer_age=writer_age,
            handedness=handedness,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Read and validate image
    try:
        contents = await image.read()
        img = Image.open(__import__("io").BytesIO(contents)).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img = img.resize((28, 28))
    img_array = np.array(img).astype("float32") / 255.0

    try:
        result = predictor.predict(
            img_array, metadata.model_dump(), version=model_version
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return PredictionResponse(**result)

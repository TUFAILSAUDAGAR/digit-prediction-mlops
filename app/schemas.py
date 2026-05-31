"""
Pydantic request / response schemas for the Digit Prediction API.

MetadataInput   — optional body model used internally; primary validation is
                  performed at the form-field level in main.py via validate_metadata().
PredictionResponse — structured response returned by POST /predict.
HealthResponse     — structured response returned by GET /health.
"""

from pydantic import BaseModel, Field


class MetadataInput(BaseModel):
    """Writer metadata accompanying a digit image."""

    pen_pressure: float = Field(..., ge=0.0, le=5.0, description="Pen pressure value")
    writer_age: int = Field(..., ge=1, le=150, description="Age of the writer")
    handedness: str = Field(
        ..., pattern="^(left|right)$", description="Handedness: left or right"
    )


class PredictionResponse(BaseModel):
    """Inference result returned by POST /predict."""

    predicted_digit: int   # Predicted MNIST class (0–9)
    confidence: float      # Softmax probability of the predicted class (0.0–1.0)


class HealthResponse(BaseModel):
    """Liveness check response returned by GET /health."""

    status: str            # Always "healthy" when the endpoint is reachable
    model_loaded: bool     # True when at least one model version is resident in memory

"""Pydantic request/response schemas."""

from pydantic import BaseModel, Field


class MetadataInput(BaseModel):
    """Writer metadata for a digit image."""

    pen_pressure: float = Field(..., ge=0.0, le=5.0, description="Pen pressure value")
    writer_age: int = Field(..., ge=1, le=150, description="Age of the writer")
    handedness: str = Field(
        ..., pattern="^(left|right)$", description="Handedness: left or right"
    )


class PredictionResponse(BaseModel):
    """Response from POST /predict."""

    predicted_digit: int   # 0–9
    confidence: float      # softmax probability of the predicted class


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: str
    model_loaded: bool

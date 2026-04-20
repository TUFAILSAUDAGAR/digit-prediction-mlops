from pydantic import BaseModel, Field


class MetadataInput(BaseModel):
    pen_pressure: float = Field(..., ge=0.0, le=5.0, description="Pen pressure value")
    writer_age: int = Field(..., ge=1, le=150, description="Age of the writer")
    handedness: str = Field(
        ..., pattern="^(left|right)$", description="Handedness: left or right"
    )


class PredictionResponse(BaseModel):
    predicted_digit: int
    confidence: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

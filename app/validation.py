"""
Input data validation for the Digit Prediction API.

Validates image and metadata at the API boundary before inference:
  - Image: dimensions, pixel range, channel count, file size
  - Metadata: type constraints (already handled by Pydantic, but with
    richer logging and structured error reporting here)

Raises DataValidationError with a list of violations on failure.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB
MIN_IMAGE_DIM = 8                   # pixels
MAX_IMAGE_DIM = 4096                # pixels
EXPECTED_MODEL_INPUT = (28, 28)     # after resize

VALID_HANDEDNESS = {"left", "right"}
MIN_PEN_PRESSURE = 0.0
MAX_PEN_PRESSURE = 5.0
MIN_WRITER_AGE = 1
MAX_WRITER_AGE = 150


# ── Errors ────────────────────────────────────────────────────────────────────

@dataclass
class DataValidationError(Exception):
    violations: list[str] = field(default_factory=list)

    def __str__(self):
        return "Data validation failed:\n" + "\n".join(f"  - {v}" for v in self.violations)


# ── Validators ────────────────────────────────────────────────────────────────

def validate_image_bytes(raw_bytes: bytes) -> np.ndarray:
    """
    Validate raw image bytes and return a normalised (28, 28) float32 array.
    Raises DataValidationError on any violation.
    """
    violations: list[str] = []

    # File size
    if len(raw_bytes) == 0:
        violations.append("Image file is empty")
        raise DataValidationError(violations)

    if len(raw_bytes) > MAX_IMAGE_BYTES:
        violations.append(
            f"Image size {len(raw_bytes)} bytes exceeds maximum {MAX_IMAGE_BYTES} bytes"
        )

    # Open with PIL
    try:
        img = Image.open(io.BytesIO(raw_bytes))
    except Exception as exc:
        violations.append(f"Cannot decode image: {exc}")
        raise DataValidationError(violations) from exc

    # Dimensions
    w, h = img.size
    if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
        violations.append(
            f"Image dimensions {w}x{h} are below minimum {MIN_IMAGE_DIM}x{MIN_IMAGE_DIM}"
        )
    if w > MAX_IMAGE_DIM or h > MAX_IMAGE_DIM:
        violations.append(
            f"Image dimensions {w}x{h} exceed maximum {MAX_IMAGE_DIM}x{MAX_IMAGE_DIM}"
        )

    if violations:
        raise DataValidationError(violations)

    # Convert to grayscale, resize, normalise
    img = img.convert("L").resize(EXPECTED_MODEL_INPUT)
    arr = np.array(img, dtype="float32") / 255.0

    # Pixel range sanity
    if arr.min() < 0.0 or arr.max() > 1.0:
        violations.append("Pixel values out of [0, 1] range after normalisation")

    # Blank image check (nearly uniform)
    if arr.std() < 1e-3:
        violations.append(
            "Image appears blank or nearly uniform — no digit content detected"
        )

    if violations:
        raise DataValidationError(violations)

    logger.debug("Image validated: shape=%s std=%.4f", arr.shape, arr.std())
    return arr


def validate_metadata(pen_pressure: float, writer_age: int, handedness: str) -> dict:
    """
    Validate metadata fields.
    Returns cleaned metadata dict or raises DataValidationError.
    """
    violations: list[str] = []

    if not isinstance(pen_pressure, (int, float)):
        violations.append(f"pen_pressure must be a number, got {type(pen_pressure).__name__}")
    elif not (MIN_PEN_PRESSURE <= pen_pressure <= MAX_PEN_PRESSURE):
        violations.append(
            f"pen_pressure {pen_pressure} out of range [{MIN_PEN_PRESSURE}, {MAX_PEN_PRESSURE}]"
        )

    if not isinstance(writer_age, int):
        violations.append(f"writer_age must be an integer, got {type(writer_age).__name__}")
    elif not (MIN_WRITER_AGE <= writer_age <= MAX_WRITER_AGE):
        violations.append(
            f"writer_age {writer_age} out of range [{MIN_WRITER_AGE}, {MAX_WRITER_AGE}]"
        )

    if handedness not in VALID_HANDEDNESS:
        violations.append(
            f"handedness '{handedness}' is invalid. Must be one of {VALID_HANDEDNESS}"
        )

    if violations:
        raise DataValidationError(violations)

    return {
        "pen_pressure": float(pen_pressure),
        "writer_age": int(writer_age),
        "handedness": handedness,
    }

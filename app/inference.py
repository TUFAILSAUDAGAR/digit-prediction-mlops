"""
Model loading and inference for the Digit Prediction API.

DigitPredictor manages one or more versioned model sets in memory.
Each version consists of three artifacts loaded from models/<version>/:
  - image_model.pth        : CNNEncoder weights
  - final_classifier.pth   : FinalClassifier weights
  - metadata_encoder.joblib: sklearn ColumnTransformer for metadata features

Versions are loaded lazily on first use and cached for subsequent requests.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from app.models import CNNEncoder, FinalClassifier

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_VERSION = "v1"


class DigitPredictor:
    """Loads versioned model artifacts and runs digit prediction inference."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self._versions: dict[str, dict] = {}  # cache: version_name → {image_model, final_model, metadata_encoder}

    def _load_version(self, version: str):
        """Load all three artifacts for a given version and cache them."""
        version_dir = self.model_dir / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Model version '{version}' not found")

        logger.info("Loading model version '%s' from %s", version, version_dir)

        # Load the sklearn metadata encoder and infer its output dimension
        metadata_encoder = joblib.load(version_dir / "metadata_encoder.joblib")
        sample_dim = metadata_encoder.transform(
            pd.DataFrame(
                [{"pen_pressure": 1.0, "writer_age": 25, "handedness": "right"}]
            )
        ).shape[1]

        # Load CNN image encoder
        image_model = CNNEncoder()
        image_model.load_state_dict(
            torch.load(
                version_dir / "image_model.pth", map_location="cpu", weights_only=True
            )
        )
        image_model.eval()

        # Load fusion classifier (image features + metadata features → digit logits)
        final_model = FinalClassifier(metadata_dim=sample_dim)
        final_model.load_state_dict(
            torch.load(
                version_dir / "final_classifier.pth",
                map_location="cpu",
                weights_only=True,
            )
        )
        final_model.eval()

        self._versions[version] = {
            "image_model": image_model,
            "final_model": final_model,
            "metadata_encoder": metadata_encoder,
        }
        logger.info("Model version '%s' loaded successfully", version)

    def load(self, version: str = DEFAULT_VERSION):
        """Eagerly load a specific version (called at application startup)."""
        self._load_version(version)

    @property
    def is_loaded(self) -> bool:
        """True when at least one model version is resident in memory."""
        return len(self._versions) > 0

    def available_versions(self) -> list[str]:
        """Return sorted list of version directories that contain image_model.pth."""
        return sorted(
            d.name
            for d in self.model_dir.iterdir()
            if d.is_dir() and (d / "image_model.pth").exists()
        )

    def predict(
        self, image_array: np.ndarray, metadata: dict, version: str = DEFAULT_VERSION
    ) -> dict:
        """
        Run inference on a pre-processed 28×28 float32 image array.

        Args:
            image_array: Normalised (0–1) grayscale image, shape (28, 28).
            metadata:    Dict with keys pen_pressure, writer_age, handedness.
            version:     Model version to use; loaded lazily if not yet cached.

        Returns:
            Dict with predicted_digit (int 0-9) and confidence (float 0-1).
        """
        # Lazy-load the requested version if it hasn't been loaded yet
        if version not in self._versions:
            self._load_version(version)

        v = self._versions[version]

        # Add batch and channel dimensions: (28,28) → (1,1,28,28)
        img_tensor = (
            torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )

        # Encode metadata using the fitted sklearn ColumnTransformer
        meta_df = pd.DataFrame([metadata])
        meta_encoded = v["metadata_encoder"].transform(meta_df).astype("float32")
        meta_tensor = torch.tensor(meta_encoded)

        with torch.no_grad():
            img_feat = v["image_model"](img_tensor)               # CNN feature vector
            logits = v["final_model"](img_feat, meta_tensor)       # fused classification logits
            probs = torch.softmax(logits, dim=1)                   # convert to probabilities
            pred = torch.argmax(probs, dim=1).item()               # argmax → predicted class
            confidence = probs[0, pred].item()

        return {"predicted_digit": pred, "confidence": round(confidence, 4)}

"""
Utilities for listing available model versions from the models directory.
"""

from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def list_model_versions(models_dir: Path = MODELS_DIR) -> list[str]:
    """Return sorted list of available model version directories."""
    if not models_dir.exists():
        return []
    return sorted(
        d.name
        for d in models_dir.iterdir()
        if d.is_dir() and (d / "image_model.pth").exists()
    )


"""
Model evaluation script.

Evaluates a trained model version on the MNIST test set and:
  - Prints per-class accuracy
  - Computes precision / recall / F1 via sklearn
  - Writes metrics to models/<version>/eval_metrics.json
  - Compares against the currently promoted version and fails if accuracy drops
    below the configured threshold.

Usage:
    python -m train.evaluate --version v2 --baseline v1 --min-accuracy 0.98
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.models import CNNEncoder, FinalClassifier
from train.registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / ".data"


def _generate_metadata(n: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "pen_pressure": rng.uniform(0.5, 4.5, n).round(2),
            "writer_age": rng.integers(6, 80, n),
            "handedness": rng.choice(["left", "right"], n),
        }
    )


def evaluate(version: str, baseline: str | None = None, min_accuracy: float = 0.95) -> dict:
    version_dir = MODELS_DIR / version
    if not version_dir.exists():
        raise FileNotFoundError(f"Model version '{version}' not found at {version_dir}")

    logger.info("Evaluating version '%s'", version)

    metadata_encoder = joblib.load(version_dir / "metadata_encoder.joblib")

    image_model = CNNEncoder()
    image_model.load_state_dict(
        torch.load(version_dir / "image_model.pth", map_location="cpu", weights_only=True)
    )
    image_model.eval()

    sample_dim = metadata_encoder.transform(
        pd.DataFrame([{"pen_pressure": 1.0, "writer_age": 25, "handedness": "right"}])
    ).shape[1]
    final_model = FinalClassifier(metadata_dim=sample_dim)
    final_model.load_state_dict(
        torch.load(version_dir / "final_classifier.pth", map_location="cpu", weights_only=True)
    )
    final_model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    test_mnist = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    test_meta_raw = _generate_metadata(len(test_mnist), seed=1)
    test_meta = metadata_encoder.transform(test_meta_raw).astype("float32")

    loader = DataLoader(
        list(zip([img for img, _ in test_mnist], test_meta, [lbl for _, lbl in test_mnist])),
        batch_size=256,
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, metas, labels in loader:
            imgs = torch.stack(list(imgs)) if not isinstance(imgs, torch.Tensor) else imgs
            metas = torch.tensor(np.array([m.numpy() for m in metas]), dtype=torch.float32) if not isinstance(metas, torch.Tensor) else metas
            img_feat = image_model(imgs)
            logits = final_model(img_feat, metas)
            preds = torch.argmax(logits, dim=1).tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist() if hasattr(labels, "tolist") else list(labels))

    report = classification_report(all_labels, all_preds, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds).tolist()
    accuracy = report["accuracy"]

    metrics = {
        "version": version,
        "accuracy": round(accuracy, 4),
        "per_class": {
            str(i): {
                "precision": round(report[str(i)]["precision"], 4),
                "recall": round(report[str(i)]["recall"], 4),
                "f1": round(report[str(i)]["f1-score"], 4),
            }
            for i in range(10)
        },
        "confusion_matrix": cm,
    }

    out_path = version_dir / "eval_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics written to %s", out_path)
    logger.info("Accuracy: %.4f", accuracy)

    # Gate: compare against baseline
    if baseline:
        registry = ModelRegistry(MODELS_DIR)
        baseline_info = registry.get(baseline)
        if baseline_info:
            baseline_acc = baseline_info.get("metrics", {}).get("accuracy", 0.0)
            logger.info("Baseline '%s' accuracy: %.4f", baseline, baseline_acc)
            if accuracy < baseline_acc - 0.005:
                raise ValueError(
                    f"New version accuracy {accuracy:.4f} is more than 0.5% below "
                    f"baseline {baseline} accuracy {baseline_acc:.4f}. Failing."
                )

    if accuracy < min_accuracy:
        raise ValueError(
            f"Accuracy {accuracy:.4f} is below minimum threshold {min_accuracy}. Failing."
        )

    logger.info("Evaluation passed for version '%s'", version)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model version")
    parser.add_argument("--version", required=True, help="Version to evaluate (e.g. v2)")
    parser.add_argument("--baseline", default=None, help="Baseline version to compare against")
    parser.add_argument("--min-accuracy", type=float, default=0.95, help="Minimum accuracy gate")
    args = parser.parse_args()

    evaluate(
        version=args.version,
        baseline=args.baseline,
        min_accuracy=args.min_accuracy,
    )

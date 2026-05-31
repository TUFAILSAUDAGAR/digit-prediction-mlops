"""Training pipeline: MNIST + synthetic metadata → CNNEncoder + FinalClassifier."""

import argparse
import logging
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from app.models import CNNEncoder, FinalClassifier
from train.registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / ".data"

# ── Synthetic metadata helpers ────────────────────────────────────────────────

def _generate_metadata(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "pen_pressure": rng.uniform(0.5, 4.5, n).round(2),
            "writer_age": rng.integers(6, 80, n),
            "handedness": rng.choice(["left", "right"], n),
        }
    )


def _build_metadata_encoder(df: pd.DataFrame) -> ColumnTransformer:
    encoder = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["pen_pressure", "writer_age"]),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["handedness"]),
        ]
    )
    encoder.fit(df)
    return encoder


# ── Dataset ───────────────────────────────────────────────────────────────────

class MNISTWithMeta(Dataset):
    def __init__(self, mnist_dataset, metadata_array: np.ndarray):
        self.mnist = mnist_dataset
        self.meta = metadata_array

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        meta = torch.tensor(self.meta[idx], dtype=torch.float32)
        return img, meta, label


# ── Training ──────────────────────────────────────────────────────────────────

def train(version: str, epochs: int = 5, batch_size: int = 128, lr: float = 1e-3):
    """Train, evaluate, save artifacts, and register the model version."""
    logger.info("Starting training for version '%s'", version)

    # MNIST transform: PIL → float tensor [0,1]
    transform = transforms.Compose([transforms.ToTensor()])
    train_mnist = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_mnist = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    # Synthetic metadata with fixed seeds for reproducibility
    train_meta_raw = _generate_metadata(len(train_mnist), seed=0)
    test_meta_raw = _generate_metadata(len(test_mnist), seed=1)

    # Fit encoder on train only, then transform both splits
    metadata_encoder = _build_metadata_encoder(train_meta_raw)
    train_meta = metadata_encoder.transform(train_meta_raw).astype("float32")
    test_meta = metadata_encoder.transform(test_meta_raw).astype("float32")

    train_loader = DataLoader(
        MNISTWithMeta(train_mnist, train_meta), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        MNISTWithMeta(test_mnist, test_meta), batch_size=batch_size
    )

    metadata_dim = train_meta.shape[1]  # output size of ColumnTransformer
    image_model = CNNEncoder()
    final_model = FinalClassifier(metadata_dim=metadata_dim)

    # Single Adam optimiser for both sub-networks
    optimizer = torch.optim.Adam(
        list(image_model.parameters()) + list(final_model.parameters()), lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        image_model.train()
        final_model.train()
        running_loss = 0.0
        for imgs, metas, labels in train_loader:
            optimizer.zero_grad()
            img_feat = image_model(imgs)          # (B, 128)
            logits = final_model(img_feat, metas) # (B, 10)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
        avg_loss = running_loss / len(train_mnist)
        logger.info("Epoch %d/%d  loss=%.4f", epoch, epochs, avg_loss)

    # ── Evaluation ─────────────────────────────────────────────────────────────
    image_model.eval()
    final_model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, metas, labels in test_loader:
            img_feat = image_model(imgs)
            logits = final_model(img_feat, metas)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    accuracy = correct / total
    logger.info("Test accuracy: %.4f", accuracy)

    # ── Save artifacts ─────────────────────────────────────────────────────────
    out_dir = MODELS_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(image_model.state_dict(), out_dir / "image_model.pth")
    torch.save(final_model.state_dict(), out_dir / "final_classifier.pth")
    joblib.dump(metadata_encoder, out_dir / "metadata_encoder.joblib")
    logger.info("Artifacts saved to %s", out_dir)

    # ── Register version ───────────────────────────────────────────────────────
    registry = ModelRegistry(MODELS_DIR)
    registry.register(
        version=version,
        metrics={"accuracy": round(accuracy, 4), "epochs": epochs, "lr": lr},
    )
    logger.info("Version '%s' registered in model registry", version)
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train digit prediction model")
    parser.add_argument("--version", default="v1", help="Model version tag (e.g. v2)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(version=args.version, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

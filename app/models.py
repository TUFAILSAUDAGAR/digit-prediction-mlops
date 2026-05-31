"""
PyTorch model definitions for the Digit Prediction API.

Architecture overview:
  CNNEncoder      — extracts a 128-dim feature vector from a (1, 28, 28) grayscale image
                    using two Conv2d → ReLU → MaxPool2d blocks followed by a fully-connected layer.
  FinalClassifier — fuses the image feature vector with encoded metadata features
                    (pen_pressure, writer_age, handedness) and outputs 10-class digit logits.
"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    Convolutional encoder that maps a single-channel 28×28 image to a 128-dim vector.

    Layer sizes:
      Input  : (B, 1, 28, 28)
      Conv1  : (B, 16, 28, 28) → MaxPool → (B, 16, 14, 14)
      Conv2  : (B, 32, 14, 14) → MaxPool → (B, 32, 7, 7)
      Flatten: (B, 32*7*7 = 1568)
      FC     : (B, 128)
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 1 input channel, 16 filters, 3×3 kernel
            nn.ReLU(),
            nn.MaxPool2d(2),                  # halve spatial dims: 28 → 14
            nn.Conv2d(16, 32, 3, padding=1), # 16 → 32 feature maps
            nn.ReLU(),
            nn.MaxPool2d(2),                  # halve again: 14 → 7
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 7 * 7, 128)  # project to 128-dim embedding

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class FinalClassifier(nn.Module):
    """
    Fusion classifier that concatenates image and metadata features and produces digit logits.

    Args:
        image_feat_dim: Dimensionality of the CNN encoder output (default 128).
        metadata_dim:   Dimensionality of the encoded metadata vector (determined by the
                        ColumnTransformer fitted during training, typically 6).
        num_classes:    Number of output classes (10 for MNIST digits 0–9).
    """

    def __init__(self, image_feat_dim=128, metadata_dim=6, num_classes=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(image_feat_dim + metadata_dim, 64),  # fused input → hidden
            nn.ReLU(),
            nn.Linear(64, num_classes),                    # hidden → logits
        )

    def forward(self, img_feat, meta_feat):
        # Concatenate along feature dimension before classification head
        return self.fc(torch.cat([img_feat, meta_feat], dim=1))

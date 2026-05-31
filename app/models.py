"""PyTorch model definitions: CNNEncoder and FinalClassifier."""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """CNN that maps a (1,28,28) image to a 128-dim feature vector."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 1→16 filters
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 28→14
            nn.Conv2d(16, 32, 3, padding=1), # 16→32 filters
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 14→7
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 7 * 7, 128)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class FinalClassifier(nn.Module):
    """
    Fuses image features and metadata features to produce 10-class digit logits.
    """

    def __init__(self, image_feat_dim=128, metadata_dim=6, num_classes=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(image_feat_dim + metadata_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, img_feat, meta_feat):
        # Concat image and metadata vectors then classify
        return self.fc(torch.cat([img_feat, meta_feat], dim=1))

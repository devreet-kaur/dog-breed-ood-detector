"""Binary CNN for dog-versus-OOD classification."""

from __future__ import annotations

import torch
from torch import nn


class BinaryCNN(nn.Module):
    """Small CNN for binary dog-versus-OOD classification.

    Input:
        RGB images with shape ``(batch_size, 3, 224, 224)``.

    Output:
        Two logits per image:
        - class 0: dog / in-distribution
        - class 1: non-dog / out-of-distribution
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be between 0 and 1")

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return binary classification logits."""
        features = self.features(inputs)
        return self.classifier(features)
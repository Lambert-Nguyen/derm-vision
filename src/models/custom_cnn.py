"""
custom_cnn.py
-------------
Baseline 4-layer Convolutional Neural Network for skin lesion classification.

Architecture
------------
    Conv2d(3→32, 3×3) → BN → ReLU → MaxPool
    Conv2d(32→64, 3×3) → BN → ReLU → MaxPool
    Conv2d(64→128, 3×3) → BN → ReLU → MaxPool
    Conv2d(128→256, 3×3) → BN → ReLU → MaxPool
    AdaptiveAvgPool → Flatten → FC(256→128) → Dropout → FC(128→num_classes)

Intended as a lightweight baseline to compare against the EfficientNet-B3
transfer learning backbone.
"""

import torch
import torch.nn as nn
from typing import Optional


class ConvBlock(nn.Module):
    """A single convolution block: Conv2d → BatchNorm → ReLU → MaxPool.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output feature channels.
    kernel_size : int
        Convolution kernel size.  Default is 3.
    pool_size : int
        Max-pooling kernel size.  Default is 2.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the conv block."""
        return self.block(x)


class CustomCNN(nn.Module):
    """Baseline 4-layer CNN for ISIC 2019 skin lesion classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes.  Default is 8.
    dropout_rate : float
        Dropout probability applied before the final FC layer.  Default 0.5.
    """

    def __init__(self, num_classes: int = 8, dropout_rate: float = 0.5) -> None:
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32),    # → B×32×H/2×W/2
            ConvBlock(32, 64),   # → B×64×H/4×W/4
            ConvBlock(64, 128),  # → B×128×H/8×W/8
            ConvBlock(128, 256), # → B×256×H/16×W/16
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)``.
        """
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

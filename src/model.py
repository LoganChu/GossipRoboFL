"""
model.py — Neural network architectures for GossipRoboFL.

Uses a compact CNN (~500k params) rather than ResNet-18 to keep the gossip
simulation tractable. With 100 robots each exchanging 4 messages/round, a
500k-param model requires ~2MB per message vs ~44MB for ResNet-18.

Swarm robotics framing: each robot runs identical architecture but learns
from its own sensor view (non-IID local data partition).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CIFAR-10 small CNN  (input: 3 x 32 x 32)
# ---------------------------------------------------------------------------

class SmallCNNCifar(nn.Module):
    """Compact 3-block CNN for CIFAR-10 / CIFAR-100.

    Architecture:
        Block 1: Conv(3→32, 3x3) → BN → ReLU → Conv(32→32, 3x3) → BN → ReLU → MaxPool(2)
        Block 2: Conv(32→64, 3x3) → BN → ReLU → Conv(64→64, 3x3) → BN → ReLU → MaxPool(2)
        Block 3: Conv(64→128, 3x3, pad=1) → BN → ReLU → AdaptiveAvgPool → Flatten
        FC:      128 → 256 → num_classes
    Params: ~487k for num_classes=10.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # 32x32 → 16x16
            nn.Dropout2d(0.1),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # 16x16 → 8x8
            nn.Dropout2d(0.1),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)), # 8x8 → 2x2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# FashionMNIST small CNN  (input: 1 x 28 x 28)
# ---------------------------------------------------------------------------

class SmallCNNFashion(nn.Module):
    """Compact 2-block CNN for FashionMNIST / MNIST.

    Architecture:
        Block 1: Conv(1→32, 3x3) → BN → ReLU → MaxPool(2)
        Block 2: Conv(32→64, 3x3) → BN → ReLU → MaxPool(2)
        FC:      64*5*5 → 128 → num_classes
    Params: ~107k for num_classes=10.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # 28x28 → 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # 14x14 → 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Factory + utilities
# ---------------------------------------------------------------------------

def get_model(name: str, num_classes: int = 10) -> nn.Module:
    """Instantiate a model by name.

    Args:
        name: One of "cnn_cifar" (CIFAR-10/100) or "cnn_fashion" (FashionMNIST).
        num_classes: Output dimension.

    Returns:
        Randomly-initialized nn.Module.
    """
    registry: dict[str, type[nn.Module]] = {
        "cnn_cifar": SmallCNNCifar,
        "cnn_fashion": SmallCNNFashion,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(registry)}")
    return registry[name](num_classes=num_classes)


def model_for_dataset(dataset_name: str, num_classes: int = 10) -> nn.Module:
    """Convenience: pick the right model for a dataset name."""
    mapping = {
        "cifar10": "cnn_cifar",
        "fashion_mnist": "cnn_fashion",
    }
    if dataset_name not in mapping:
        raise ValueError(f"No default model for dataset '{dataset_name}'.")
    return get_model(mapping[dataset_name], num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_bytes(model: nn.Module) -> int:
    """Return approximate size of model parameters in bytes (float32 assumed)."""
    return sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )

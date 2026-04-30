"""
model_registry.py — Defines your actual model zoo.

Add entries here as you train new models. Each entry specifies:
  - constructor : callable that returns an nn.Module (unloaded)
  - saved_dict  : path to the .pth file with the pruned state_dict
  - dataset     : dataset name string (used to build the data loader)
  - input_shape : (C, H, W) — WITHOUT batch dimension
  - num_classes : int

The registry key format is  "<ModelName>/<Dataset>"  e.g. "ResNet32/CIFAR10".
For models with a single dataset, just "<ModelName>".

HOW TO ADD YOUR OWN MODEL
--------------------------
1. Import or define the constructor function at the top of this file.
2. Add an entry to MODEL_REGISTRY below.
3. That's it — benchmark.py picks it up automatically.

LOADING BEHAVIOUR
-----------------
load_pruned_model() calls constructor() → loads state_dict → eval().
Weights that were zeroed by your pruning method remain zero.
No further modification is applied — sparsity is exactly as trained.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Placeholder constructors for models defined in your codebase.
# Replace these imports with the real ones from your project.
# ---------------------------------------------------------------------------

def _import_or_stub(name: str):
    """Returns a stub constructor that raises a clear error if not found."""
    def stub(*args, **kwargs):
        raise ImportError(
            f"Constructor '{name}' not found. "
            f"Make sure your project's model definitions are on sys.path."
        )
    stub.__name__ = name
    return stub


# ---- LeNet5 ---------------------------------------------------------------
try:
    from models.lenet import create_lenet5          # adjust import path
except ImportError:
    create_lenet5 = _import_or_stub("create_lenet5")

# ---- DenseNet40 -----------------------------------------------------------
try:
    from models.densenet import create_densenet40   # adjust import path
except ImportError:
    create_densenet40 = _import_or_stub("create_densenet40")

# ---- ResNet32 -------------------------------------------------------------
try:
    from models.resnet import resnet32              # adjust import path
except ImportError:
    resnet32 = _import_or_stub("resnet32")

# ---- VGG19 ----------------------------------------------------------------
try:
    from models.vgg import vgg19                    # adjust import path
except ImportError:
    vgg19 = _import_or_stub("vgg19")


# ---------------------------------------------------------------------------
# Inline definitions for models with no external dependency
# ---------------------------------------------------------------------------

def _lenet300():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 300),
        nn.ReLU(),
        nn.Linear(300, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )


# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------

@dataclass
class ModelEntry:
    constructor: Callable[[], nn.Module]
    saved_dict: str                    # relative to project root or absolute
    dataset: str
    input_shape: tuple                 # (C, H, W) — no batch dim
    num_classes: int
    notes: str = ""

    def input_tensor(self, batch_size: int = 1) -> torch.Tensor:
        return torch.randn(batch_size, *self.input_shape)


# ---------------------------------------------------------------------------
# YOUR MODEL ZOO
# Edit saved_dict paths to match your directory layout.
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelEntry] = {

    # --- LeNet5 / MNIST ---------------------------------------------------
    "LeNet5": ModelEntry(
        constructor=create_lenet5,
        saved_dict="Best_Results_HPO/LeNet5/best_2025-12-08_15-03-41_MNIST_LeNet5.pth",
        dataset="MNIST",
        input_shape=(1, 28, 28),
        num_classes=10,
    ),

    # --- LeNet300 / MNIST -------------------------------------------------
    "LeNet300": ModelEntry(
        constructor=_lenet300,
        saved_dict="Best_Results_HPO/LeNet300/best_2025-12-09_18-58-19_MNIST_LeNet300.pth",
        dataset="MNIST",
        input_shape=(1, 28, 28),
        num_classes=10,
    ),

    # --- DenseNet40 / CIFAR10 ---------------------------------------------
    "DenseNet40": ModelEntry(
        constructor=create_densenet40,
        saved_dict="Best_Results_HPO/DenseNet40/best_2025-11-16_18-44-04_CIFAR10_DenseNet40.pth",
        dataset="cifar10",
        input_shape=(3, 32, 32),
        num_classes=10,
    ),

    # --- ResNet56 / CIFAR10 (pretrained from hub — no saved_dict needed) --
    "ResNet56": ModelEntry(
        constructor=lambda: torch.hub.load(
            "chenyaofo/pytorch-cifar-models",
            "cifar10_resnet56",
            pretrained=True,
        ),
        saved_dict="",           # empty = skip state_dict load
        dataset="cifar10",
        input_shape=(3, 32, 32),
        num_classes=10,
        notes="Loaded from torch.hub — no local .pth required",
    ),

    # --- ResNet32 / CIFAR10 -----------------------------------------------
    "ResNet32/CIFAR10": ModelEntry(
        constructor=lambda: resnet32(num_classes=10),
        saved_dict="Best_Results_HPO/ResNet32/CIFAR-10/best_2025-12-05_00-10-54_cifar10_ResNet_32.pth",
        dataset="cifar10",
        input_shape=(3, 32, 32),
        num_classes=10,
    ),

    # --- ResNet32 / CIFAR100 ----------------------------------------------
    "ResNet32/CIFAR100": ModelEntry(
        constructor=lambda: resnet32(num_classes=100),
        saved_dict="Best_Results_HPO/ResNet32/CIFAR-100/86_Sparsity/best_2025-12-03_10-14-41_cifar100_ResNet_32.pth",
        dataset="cifar100",
        input_shape=(3, 32, 32),
        num_classes=100,
    ),

    # --- ResNet32 / TinyImageNet ------------------------------------------
    "ResNet32/TinyImageNet": ModelEntry(
        constructor=lambda: resnet32(num_classes=200),
        saved_dict="Best_Results_HPO/ResNet32/TinyImageNet/best_2025-12-15_10-04-01_tiny_imagenet_ResNet_32.pth",
        dataset="tiny_imagenet",
        input_shape=(3, 64, 64),
        num_classes=200,
    ),

    # --- VGG19 / CIFAR10 --------------------------------------------------
    "VGG19/CIFAR10": ModelEntry(
        constructor=lambda: vgg19(dataset="cifar10"),
        saved_dict="Best_Results_HPO/VGG-19/CIFAR-10/best_2025-11-25_06-06-15_cifar10_vgg19.pth",
        dataset="cifar10",
        input_shape=(3, 32, 32),
        num_classes=10,
    ),

    # --- VGG19 / CIFAR100 -------------------------------------------------
    "VGG19/CIFAR100": ModelEntry(
        constructor=lambda: vgg19(dataset="cifar100"),
        saved_dict="Best_Results_HPO/VGG-19/CIFAR-100/98_sparsity/best_2025-11-23_20-24-12_cifar100_vgg19.pth",
        dataset="cifar100",
        input_shape=(3, 32, 32),
        num_classes=100,
    ),

    # --- VGG19 / TinyImageNet ---------------------------------------------
    "VGG19/TinyImageNet": ModelEntry(
        constructor=lambda: vgg19(dataset="tiny_imagenet"),
        saved_dict="Best_Results_HPO/VGG-19/TinyImageNet/best_2025-12-31_11-17-06_tiny_imagenet_vgg19.pth",
        dataset="tiny_imagenet",
        input_shape=(3, 64, 64),
        num_classes=200,
    ),
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_pruned_model(
    key: str,
    project_root: str | Path = ".",
    device: torch.device | None = None,
) -> tuple[nn.Module, ModelEntry, dict]:
    """
    Load a pruned model from the registry.

    Returns
    -------
    model      : nn.Module in eval mode, weights loaded
    entry      : ModelEntry (input_shape, dataset, etc.)
    meta       : dict for the benchmarking runner
    """
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Model '{key}' not in registry. "
            f"Available: {sorted(MODEL_REGISTRY.keys())}"
        )

    entry = MODEL_REGISTRY[key]
    device = device or torch.device("cpu")

    model = entry.constructor()

    if entry.saved_dict:
        path = Path(project_root) / entry.saved_dict
        if not path.exists():
            raise FileNotFoundError(
                f"Saved dict not found: {path}\n"
                f"Check 'project_root' argument or update MODEL_REGISTRY."
            )
        state = torch.load(path, map_location=device, weights_only=True)

        # Handle common save patterns:
        #   (a) raw state_dict
        #   (b) {"model": state_dict, ...}
        #   (c) {"state_dict": state_dict, ...}
        if isinstance(state, dict):
            if "model" in state:
                state = state["model"]
            elif "state_dict" in state:
                state = state["state_dict"]
        model.load_state_dict(state)

    model.to(device).eval()

    # Compute sparsity
    total, zeros = 0, 0
    for p in model.parameters():
        total += p.numel()
        zeros += (p == 0).sum().item()
    sparsity = zeros / total if total > 0 else 0.0

    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())

    meta = {
        "format": f"Pruned ({sparsity*100:.1f}% sparse)",
        "param_bytes": param_bytes,
        "model_size_mb": param_bytes / (1024 ** 2),
        "notes": (
            f"Loaded from {entry.saved_dict or 'torch.hub'}. "
            f"Measured sparsity: {sparsity*100:.1f}%. "
            f"Dense tensor storage — sparsity is structural (weights=0)."
        ),
    }

    return model, entry, meta


def list_models() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())
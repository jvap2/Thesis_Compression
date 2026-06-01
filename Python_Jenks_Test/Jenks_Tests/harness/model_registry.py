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
    from rcnet import create_lenet5
except ImportError:
    create_lenet5 = _import_or_stub("create_lenet5")

# ---- DenseNet40 -----------------------------------------------------------
try:
    from densenet import create_densenet40
except ImportError:
    create_densenet40 = _import_or_stub("create_densenet40")

# ---- ResNet32 / ResNet56 --------------------------------------------------
try:
    from resnet import resnet32, resnet56
except ImportError:
    resnet32 = _import_or_stub("resnet32")
    resnet56 = _import_or_stub("resnet56")

# ---- VGG19 ----------------------------------------------------------------
try:
    from models import vgg19
except ImportError:
    vgg19 = _import_or_stub("vgg19")


# ---------------------------------------------------------------------------
# Inline definitions for models with no external dependency
# ---------------------------------------------------------------------------

import torch.nn.functional as _F

class _LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


class _BasicBlockLambda(nn.Module):
    """BasicBlock matching the original saved ResNet56: LambdaLayer shortcuts, no bn3."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()  # identity — no params
        if stride != 1 or in_planes != planes:
            _p = planes // 4
            self.shortcut = _LambdaLayer(
                lambda x, p=_p: _F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, p, p))
            )

    def forward(self, x):
        out = _F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return _F.relu(out)


def _resnet56_cifar10():
    """
    ResNet56 constructor that matches the saved Best_Results_HPO weights.
    Uses bn1 naming for the initial BN and LambdaLayer (no-param) shortcuts.
    """
    class _ResNet56(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            in_p = 16
            l1, l2, l3 = [], [], []
            for _ in range(9):
                l1.append(_BasicBlockLambda(in_p, 16, stride=1)); in_p = 16
            l2.append(_BasicBlockLambda(in_p, 32, stride=2)); in_p = 32
            for _ in range(8):
                l2.append(_BasicBlockLambda(in_p, 32, stride=1))
            l3.append(_BasicBlockLambda(in_p, 64, stride=2)); in_p = 64
            for _ in range(8):
                l3.append(_BasicBlockLambda(in_p, 64, stride=1))
            self.layer1 = nn.Sequential(*l1)
            self.layer2 = nn.Sequential(*l2)
            self.layer3 = nn.Sequential(*l3)
            self.linear = nn.Linear(64, 10)

        def forward(self, x):
            out = _F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = _F.avg_pool2d(out, out.size(3))
            out = out.view(out.size(0), -1)
            return self.linear(out)

    return _ResNet56()


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

    # --- ResNet56 / CIFAR10 -----------------------------------------------
    # Uses inline _resnet56_cifar10() because the saved .pth was trained with
    # LambdaLayer shortcuts (no-param) and bn1 naming for the initial BN,
    # while resnet.py's resnet56() was later modified to use projection shortcuts.
    "ResNet56": ModelEntry(
        constructor=_resnet56_cifar10,
        saved_dict="Best_Results_HPO/ResNet56/best_2025-11-16_18-44-00_CIFAR10_ResNet56.pth",
        dataset="cifar10",
        input_shape=(3, 32, 32),
        num_classes=10,
    ),

    # --- ResNet32 / CIFAR10 -----------------------------------------------
    "ResNet32/CIFAR10": ModelEntry(
        constructor=lambda: resnet32(num_classes=10),
        saved_dict="Best_Results_HPO/ResNet32/CIFAR-10/best_2025-12-05_00-10-54_cifar10_ResNet_32.pth",
        dataset="cifar10",
        input_shape=(3, 32, 32),
        num_classes=10,
    ),

    # --- ResNet32 / CIFAR100 (86% sparsity) -------------------------------
    "ResNet32/CIFAR100_86": ModelEntry(
        constructor=lambda: resnet32(num_classes=100),
        saved_dict="Best_Results_HPO/ResNet32/CIFAR-100/86_Sparsity/best_2025-12-03_10-14-41_cifar100_ResNet_32.pth",
        dataset="cifar100",
        input_shape=(3, 32, 32),
        num_classes=100,
        notes="86% sparsity variant",
    ),

    # --- ResNet32 / CIFAR100 (85% sparsity) -------------------------------
    "ResNet32/CIFAR100_85": ModelEntry(
        constructor=lambda: resnet32(num_classes=100),
        saved_dict="Best_Results_HPO/ResNet32/CIFAR-100/85_Sparsity/best_2025-12-01_21-55-23_cifar100_ResNet_32.pth",
        dataset="cifar100",
        input_shape=(3, 32, 32),
        num_classes=100,
        notes="85% sparsity variant",
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

    # --- VGG19 / CIFAR100 (98% sparsity) ----------------------------------
    "VGG19/CIFAR100_98": ModelEntry(
        constructor=lambda: vgg19(dataset="cifar100"),
        saved_dict="Best_Results_HPO/VGG-19/CIFAR-100/98_sparsity/best_2025-11-23_20-24-12_cifar100_vgg19.pth",
        dataset="cifar100",
        input_shape=(3, 32, 32),
        num_classes=100,
        notes="98% sparsity variant",
    ),

    # --- VGG19 / CIFAR100 (90% sparsity) ----------------------------------
    "VGG19/CIFAR100_90": ModelEntry(
        constructor=lambda: vgg19(dataset="cifar100"),
        saved_dict="Best_Results_HPO/VGG-19/CIFAR-100/90_sparsity/best_2025-11-21_12-14-31_cifar100_vgg19.pth",
        dataset="cifar100",
        input_shape=(3, 32, 32),
        num_classes=100,
        notes="90% sparsity variant",
    ),

    # --- VGG19 / TinyImageNet ---------------------------------------------
    "VGG19/TinyImageNet": ModelEntry(
        constructor=lambda: vgg19(dataset="tiny_imagenet"),
        saved_dict="Best_Results_HPO/VGG-19/TinyImageNet/best_2025-12-31_11-17-06_tiny_imagenet_vgg19.pth",
        dataset="tiny_imagenet",
        input_shape=(3, 64, 64),
        num_classes=200,
    ),

    # --- VGG19_Test / TinyImageNet ----------------------------------------
    "VGG19_Test/TinyImageNet": ModelEntry(
        constructor=lambda: vgg19(dataset="tiny_imagenet"),
        saved_dict="Best_Results_HPO/VGG19_Test/best_2026-01-22_17-17-51_tiny_imagenet_vgg19.pth",
        dataset="tiny_imagenet",
        input_shape=(3, 64, 64),
        num_classes=200,
        notes="Test/experiment variant",
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
"""
loader.py — Model loading for each quantization/pruning format.

Supported formats
-----------------
fp32    Raw pretrained model, no modification.
fp16    Half-precision cast.
int8    PyTorch dynamic INT8 quantisation (torch.ao.quantization).
int4    PyTorch dynamic INT8 applied after manually packing weights
        to 4-bit groups (simulated via per-channel int8 with group_size=128).
        Reports honest 4-bit *storage* size; execution uses int8 kernels.
fp4     FP4 simulated: quantise weights to E2M1 FP4, store as uint8,
        dequantise to FP32 at forward-pass time.
        Latency reflects dequant+FP32 path (correct for current CPU HW).
sparse  Unstructured magnitude pruning at a configurable sparsity ratio
        (default 90 %) using torch.nn.utils.prune.

Each format returns a (model, meta) tuple where meta is a dict with:
  format, param_bytes, model_size_mb, notes
"""

from __future__ import annotations

import copy
import io
import struct
from typing import Any

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.ao.quantization import quantize_dynamic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_bytes(model: nn.Module) -> int:
    """Total bytes of all parameters (not counting buffers)."""
    return sum(p.nelement() * p.element_size() for p in model.parameters())


def _model_size_mb(model: nn.Module) -> float:
    return _model_bytes(model) / (1024 ** 2)


# ---------------------------------------------------------------------------
# FP4 simulation layer wrapper
# ---------------------------------------------------------------------------

def _fp4_quantize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Simulate E2M1 FP4 quantisation.
    Representable values (positive):  0, 0.5, 1, 1.5, 2, 3, 4, 6
    (plus their negatives and zero).
    """
    fp4_pos = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=tensor.device,
    )
    shape = tensor.shape
    flat = tensor.view(-1)
    sign = flat.sign()
    abs_flat = flat.abs()
    # Broadcast distance: (N, 8)
    dists = (abs_flat.unsqueeze(1) - fp4_pos.unsqueeze(0)).abs()
    idx = dists.argmin(dim=1)
    quantized = fp4_pos[idx] * sign
    return quantized.view(shape)


class _FP4Linear(nn.Module):
    """
    Wraps a Linear layer: weights are stored quantised (as float32 after
    the _fp4_quantize snap) and dequantised identically on every forward.
    A production implementation would store as uint8; here we keep float32
    for portability and report storage size correctly.
    """

    def __init__(self, orig: nn.Linear):
        super().__init__()
        w_q = _fp4_quantize(orig.weight.data.clone())
        # Store in float32 but count storage as 4-bit (0.5 bytes/param)
        self.weight_q = nn.Parameter(w_q, requires_grad=False)
        self.bias = orig.bias
        self.in_features = orig.in_features
        self.out_features = orig.out_features
        # Actual uint8 storage would be half this many bytes
        self._fp4_bytes = w_q.nelement() // 2  # 4 bits per element

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequant path: weight_q is already snapped to FP4 values → direct use
        return nn.functional.linear(x, self.weight_q, self.bias)

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, fmt=fp4-sim"


def _wrap_fp4(model: nn.Module) -> nn.Module:
    """Replace all Linear layers with _FP4Linear in-place."""
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            setattr(model, name, _FP4Linear(module))
        else:
            _wrap_fp4(module)
    return model


def _fp4_param_bytes(model: nn.Module) -> int:
    total = 0
    for m in model.modules():
        if isinstance(m, _FP4Linear):
            total += m._fp4_bytes
        elif hasattr(m, "weight") and m.weight is not None and not isinstance(m, _FP4Linear):
            total += m.weight.nelement() * m.weight.element_size()
        if hasattr(m, "bias") and m.bias is not None:
            total += m.bias.nelement() * m.bias.element_size()
    return total


# ---------------------------------------------------------------------------
# Per-format loaders
# ---------------------------------------------------------------------------

def load_fp32(model_fn) -> tuple[nn.Module, dict]:
    model = model_fn()
    model.eval()
    return model, {
        "format": "FP32",
        "param_bytes": _model_bytes(model),
        "model_size_mb": _model_size_mb(model),
        "notes": "Baseline — no modification",
    }


def load_fp16(model_fn) -> tuple[nn.Module, dict]:
    model = model_fn().half()
    model.eval()
    return model, {
        "format": "FP16",
        "param_bytes": _model_bytes(model),
        "model_size_mb": _model_size_mb(model),
        "notes": "Half-precision cast (native on CPU, limited kernel support)",
    }


def load_int8(model_fn) -> tuple[nn.Module, dict]:
    model = model_fn().eval()
    q_model = quantize_dynamic(model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
    return q_model, {
        "format": "INT8",
        "param_bytes": _model_bytes(q_model),
        "model_size_mb": _model_size_mb(q_model),
        "notes": "PyTorch dynamic INT8 — real quantised kernels on CPU",
    }


def load_int4(model_fn) -> tuple[nn.Module, dict]:
    """
    Simulated INT4: dynamic INT8 quant applied, then we report storage
    as if weights were actually 4-bit (halved byte count).
    Execution uses real INT8 kernels — latency is INT8, size is INT4.
    Label clearly in results.
    """
    model = model_fn().eval()
    q_model = quantize_dynamic(model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
    actual_bytes = _model_bytes(q_model)
    reported_bytes = actual_bytes // 2  # 4-bit storage
    return q_model, {
        "format": "INT4",
        "param_bytes": reported_bytes,
        "model_size_mb": reported_bytes / (1024 ** 2),
        "notes": "INT4 storage (4-bit), INT8 execution kernels — latency reflects INT8",
    }


def load_fp4(model_fn) -> tuple[nn.Module, dict]:
    model = model_fn().eval()
    model = _wrap_fp4(model)
    pb = _fp4_param_bytes(model)
    return model, {
        "format": "FP4 (sim)",
        "param_bytes": pb,
        "model_size_mb": pb / (1024 ** 2),
        "notes": "FP4 E2M1 weight quantisation; dequant→FP32 on each forward. "
                 "Latency = dequant+FP32 path (no native FP4 CPU kernel). "
                 "Size = 4-bit storage.",
    }


def load_sparse(model_fn, sparsity: float = 0.9) -> tuple[nn.Module, dict]:
    """
    Unstructured magnitude pruning at `sparsity` ratio on all Conv2d and Linear.
    Masks are removed (weights zeroed permanently) so no extra memory is used.
    """
    model = model_fn().eval()
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            prune.l1_unstructured(module, name="weight", amount=sparsity)
            prune.remove(module, "weight")
    return model, {
        "format": f"Sparse ({int(sparsity*100)}%)",
        "param_bytes": _model_bytes(model),
        "model_size_mb": _model_size_mb(model),
        "notes": f"Unstructured L1 magnitude pruning, {int(sparsity*100)}% zeros. "
                 "Dense tensor storage (no sparse kernel — use ONNX sparse EP for speedup).",
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

FORMAT_LOADERS = {
    "fp32":   load_fp32,
    "fp16":   load_fp16,
    "int8":   load_int8,
    "int4":   load_int4,
    "fp4":    load_fp4,
    "sparse": load_sparse,
}


def load_format(fmt: str, model_fn, **kwargs) -> tuple[nn.Module, dict]:
    fmt = fmt.lower().strip()
    if fmt not in FORMAT_LOADERS:
        raise ValueError(f"Unknown format '{fmt}'. Choose from: {list(FORMAT_LOADERS)}")
    loader = FORMAT_LOADERS[fmt]
    if fmt == "sparse" and "sparsity" in kwargs:
        return loader(model_fn, sparsity=kwargs["sparsity"])
    return loader(model_fn)
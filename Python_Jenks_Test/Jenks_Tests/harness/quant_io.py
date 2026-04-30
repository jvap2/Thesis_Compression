"""
quant_io.py — Saving and loading quantized models.

This module handles the two distinct use cases:

A) CPU SIMULATION (benchmarking without special hardware)
---------------------------------------------------------
Quantized PyTorch models (INT8, FP4-sim) can't be saved as plain state_dicts
because scale/zero-point parameters are embedded in quantized tensors.
Use save_quantized_pt() / load_quantized_pt() which call torch.save(model)
on the full model object.

B) NVFP4 HARDWARE EXPORT PATH
------------------------------
For actual Blackwell GPU deployment you need a TensorRT engine.
The path is:
  1. export_onnx()          — PyTorch → ONNX (opset 17+)
  2. build_trt_engine()     — ONNX → TensorRT .trt engine with FP4 calibration
                              (requires tensorrt package + GPU)
  3. load_trt_engine()      — deserialise engine for inference

The TRT functions are gated behind a try/import so the file is still
importable on CPU-only machines.

C) MXFP4 / ONNX PATH
---------------------
export_onnx() + quantize_onnx_fp4() produces an .onnx file with
FP4 quantization annotations compatible with ONNX Runtime's
MXFPExecutionProvider (AMD MI300X / Intel Gaudi).

RECOMMENDED WORKFLOW
--------------------
During development (CPU):
    model, entry, meta = load_pruned_model("ResNet32/CIFAR10")
    q_model = quantize_for_simulation(model, fmt="int8")
    save_quantized_pt(q_model, "saved/resnet32_cifar10_int8.pt")
    # later:
    q_model = load_quantized_pt("saved/resnet32_cifar10_int8.pt")

For NVFP4 deployment:
    model, entry, meta = load_pruned_model("ResNet32/CIFAR10")
    export_onnx(model, entry, "saved/resnet32_cifar10.onnx")
    build_trt_engine(
        "saved/resnet32_cifar10.onnx",
        "saved/resnet32_cifar10_fp4.trt",
        fp4=True,
        calib_data=your_calib_loader,
    )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Iterator

import torch
import torch.nn as nn
from torch.ao.quantization import quantize_dynamic

from .loader import _wrap_fp4


# ---------------------------------------------------------------------------
# A) CPU simulation — save / load quantized models
# ---------------------------------------------------------------------------

def quantize_for_simulation(
    model: nn.Module,
    fmt: str = "int8",
    sparsity: float = 0.9,
) -> nn.Module:
    """
    Apply quantization to an already-loaded pruned model.

    fmt choices: "int8", "int4" (int8 kernels, int4 storage), "fp4"
    Returns the quantized model (not saved — call save_quantized_pt next).
    """
    fmt = fmt.lower()
    model = model.eval()

    if fmt in ("int8", "int4"):
        q = quantize_dynamic(model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
        return q
    elif fmt == "fp4":
        return _wrap_fp4(model)
    else:
        raise ValueError(f"Unknown format '{fmt}' — choose int8, int4, or fp4")


def save_quantized_pt(model: nn.Module, path: str | Path):
    """
    Save a quantized model as a full torch object.
    Use this (NOT state_dict) for quantized models because scale/zero-point
    tensors are embedded in quantized weight objects, not in state_dict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, path)
    size_mb = path.stat().st_size / (1024 ** 2)
    print(f"  Saved quantized model → {path}  ({size_mb:.2f} MB)")


def load_quantized_pt(path: str | Path, device: str = "cpu") -> nn.Module:
    """Load a model saved with save_quantized_pt()."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Quantized model not found: {path}")
    model = torch.load(path, map_location=device, weights_only=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# B) ONNX export (shared step for both NVFP4 and MXFP4 paths)
# ---------------------------------------------------------------------------

def export_onnx(
    model: nn.Module,
    input_shape: tuple,           # (C, H, W) from ModelEntry
    path: str | Path,
    opset: int = 17,
    batch_size: int = 1,
    dynamic_batch: bool = True,
) -> Path:
    """
    Export a PyTorch model to ONNX.

    Parameters
    ----------
    model        : eval-mode nn.Module (pruned, NOT quantized — TRT handles that)
    input_shape  : (C, H, W) without batch dim
    path         : output .onnx file path
    opset        : ONNX opset version (17+ required for FP8/FP4 ops)
    dynamic_batch: if True, marks batch axis as dynamic (recommended)

    Returns
    -------
    Path to the written .onnx file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval()
    dummy = torch.randn(batch_size, *input_shape)

    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else {}

    torch.onnx.export(
        model,
        dummy,
        str(path),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    size_mb = path.stat().st_size / (1024 ** 2)
    print(f"  ONNX exported → {path}  ({size_mb:.2f} MB, opset {opset})")
    return path


# ---------------------------------------------------------------------------
# C) TensorRT NVFP4 engine build (requires GPU + tensorrt package)
# ---------------------------------------------------------------------------

def build_trt_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    fp4: bool = True,
    workspace_gb: int = 4,
    calib_data: Optional[Iterator] = None,
) -> Path:
    """
    Build a TensorRT engine from an ONNX file.

    Requires:
        pip install tensorrt tensorrt-lean  (NVIDIA Blackwell SDK)
        CUDA-capable GPU (Blackwell for native NVFP4)

    fp4=True  sets FP4 precision flag — only meaningful on Blackwell (sm_100+).
    On older GPUs TensorRT falls back to the best supported precision.

    calib_data : optional iterator yielding torch.Tensor batches for INT8/FP4
                 PTQ calibration.  If None, TRT uses implicit calibration.
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError(
            "tensorrt is not installed. Install it via:\n"
            "  pip install tensorrt tensorrt-lean\n"
            "or follow https://docs.nvidia.com/deeplearning/tensorrt/install-guide"
        )

    onnx_path   = Path(onnx_path)
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser  = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  TRT parse error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30)
    )

    if fp4:
        # NVFP4 requires both FP8 and FP4 flags + explicit quantization layers
        # On Blackwell sm_100+: native NVFP4 GEMM kernels are selected
        config.set_flag(trt.BuilderFlag.FP8)
        config.set_flag(trt.BuilderFlag.FP4)
        print("  TRT: FP4 precision enabled (Blackwell sm_100+ for native kernels)")
    else:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  TRT: FP16 precision")

    # Optional calibration
    if calib_data is not None:
        # Attach a simple EntropyCalibrator2 for PTQ
        config.int8_calibrator = _make_calibrator(calib_data, network)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TensorRT engine build failed")

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    size_mb = engine_path.stat().st_size / (1024 ** 2)
    print(f"  TRT engine saved → {engine_path}  ({size_mb:.2f} MB)")
    return engine_path


def load_trt_engine(engine_path: str | Path):
    """
    Deserialise a TensorRT engine for inference.

    Returns a trt.ICudaEngine object.
    Usage:
        engine  = load_trt_engine("model_fp4.trt")
        context = engine.create_execution_context()
        # bind input/output buffers and call context.execute_async_v3()
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError("tensorrt not installed — see build_trt_engine docstring")

    logger  = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    print(f"  TRT engine loaded from {engine_path}")
    return engine


# ---------------------------------------------------------------------------
# D) MXFP4 / ONNX Runtime path (AMD MI300X / Intel Gaudi)
# ---------------------------------------------------------------------------

def quantize_onnx_fp4(
    onnx_path: str | Path,
    out_path: str | Path,
) -> Path:
    """
    Add MXFP4 quantization annotations to an ONNX model using
    onnxruntime-extensions and the MicroscalingQuantizer.

    Requires:
        pip install onnxruntime onnxruntime-extensions

    The output .onnx is loadable by ORT with:
        sess = ort.InferenceSession(
            out_path,
            providers=["MXFPExecutionProvider", "CPUExecutionProvider"]
        )
    """
    try:
        from onnxruntime.quantization import (
            quantize_static,
            QuantFormat,
            QuantType,
        )
    except ImportError:
        raise ImportError(
            "onnxruntime not installed — pip install onnxruntime onnxruntime-extensions"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # NOTE: Full MXFP4 support requires onnxruntime-extensions >= 0.12
    # with MicroscalingQuantizer — this is a placeholder that uses QDQ INT8
    # as a proxy until your target ORT version ships MXFP4 support.
    quantize_static(
        str(onnx_path),
        str(out_path),
        calibration_data_reader=None,   # supply a CalibrationDataReader for real runs
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    print(f"  ONNX-FP4 (QDQ proxy) saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Internal calibrator helper (used by build_trt_engine)
# ---------------------------------------------------------------------------

def _make_calibrator(data_iter, network):
    """Minimal IInt8EntropyCalibrator2 wrapping a torch DataLoader."""
    try:
        import tensorrt as trt
        import numpy as np
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa
    except ImportError:
        return None

    class _Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, loader, cache="calib_cache.bin"):
            super().__init__()
            self._loader  = iter(loader)
            self._cache   = cache
            self._buf     = None
            self._buf_size = 0

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            try:
                batch = next(self._loader)
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                arr = batch.numpy().astype("float32")
                if self._buf is None or arr.nbytes != self._buf_size:
                    self._buf = cuda.mem_alloc(arr.nbytes)
                    self._buf_size = arr.nbytes
                cuda.memcpy_htod(self._buf, arr)
                return [int(self._buf)]
            except StopIteration:
                return None

        def read_calibration_cache(self):
            if os.path.exists(self._cache):
                with open(self._cache, "rb") as f:
                    return f.read()

        def write_calibration_cache(self, cache):
            with open(self._cache, "wb") as f:
                f.write(cache)

    return _Calibrator(data_iter)
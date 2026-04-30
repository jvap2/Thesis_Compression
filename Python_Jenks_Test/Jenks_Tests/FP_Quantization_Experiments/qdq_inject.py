"""
qdq_inject.py — Inject custom per-block QDQ nodes into an ONNX graph.

Overview
--------
Takes the three artifacts from save_for_trt_export():
  - weights_fp16.pt      (absorbed-bias FP16 weight tensors)
  - scales.pt            (per-block alpha_eff, one value per block)
  - metadata.json        (layer names, block_size, shapes)

And produces a QDQ-annotated ONNX model where every quantized Conv/Gemm
weight is wrapped as:

  weight_fp16
      |
  QuantizeLinear  (scale=alpha_eff, zero_point=0, block_size=B, dtype=FLOAT8E4M3FN)
      |
  DequantizeLinear (scale=alpha_eff, zero_point=0, block_size=B)
      |
  [original Conv / Gemm node]

TensorRT reads these QDQ pairs and on Blackwell (sm_100+) lowers them to
NVFP4 kernels, using your alpha_eff values as the calibration scales.

Why FP8 as the ONNX interchange dtype
--------------------------------------
ONNX opset 21 does not yet have a native FP4 dtype constant.
We use FLOAT8E4M3FN as the interchange type — TRT recognises QDQ-wrapped
FLOAT8 weights and on Blackwell will lower to NVFP4 if the --fp4 flag is
set in trtexec / BuilderFlag.FP4 in the API. The scale values (your
alpha_eff) are preserved exactly — only the interchange container changes.

Usage
-----
from harness.qdq_inject import build_qdq_onnx

build_qdq_onnx(
    checkpoint_dir = "saved/resnet32_cifar10",
    base_onnx      = "onnx/resnet32_cifar10.onnx",     # FP32 export from PyTorch
    out_onnx       = "onnx/resnet32_cifar10_qdq.onnx",
)

Then on a Blackwell node:
  trtexec --onnx=resnet32_cifar10_qdq.onnx \
          --fp4 --fp8 \
          --saveEngine=resnet32_cifar10_fp4.trt \
          --workspace=4096
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ONNX opset for per-block QDQ (block_size attribute added in opset 21)
TARGET_OPSET = 21

# ONNX dtype for FP8 interchange (TRT lowers to FP4 on Blackwell)
# FLOAT8E4M3FN = 17  in onnx TensorProto
FLOAT8_DTYPE = TensorProto.FLOAT8E4M3FN  # = 17

# Zero-point value for FP4/FP8 (always 0 — symmetric quantization)
ZERO_POINT_VALUE = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _np_from_tensor(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _onnx_tensor(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array, name=name)


def _find_weight_initializers(graph: onnx.GraphProto) -> dict[str, int]:
    """Return {initializer_name: index} for all initializers."""
    return {init.name: i for i, init in enumerate(graph.initializer)}


def _nodes_consuming(graph: onnx.GraphProto, input_name: str) -> list[onnx.NodeProto]:
    return [n for n in graph.node if input_name in n.input]


def _param_name_to_init_name(param_name: str) -> str:
    """
    PyTorch ONNX export names initializers like 'layer1.0.conv1.weight'.
    The param_name from named_parameters() matches directly.
    """
    return param_name


# ---------------------------------------------------------------------------
# Scale tensor reshaping for per-block QDQ
# ---------------------------------------------------------------------------

def _make_block_scale(
    scales_1d: torch.Tensor,   # [num_blocks] — one scale per block
    weight_shape: list[int],   # original weight shape e.g. [64, 3, 3, 3]
    block_size: int,
) -> tuple[np.ndarray, int]:
    """
    Produce scale array and quantization axis for ONNX per-block QDQ.

    Strategy: quantize along axis=1 (inner dimension after output channels).
    For weight [out, in*kH*kW]:
      scale shape = [out, ceil(inner / block_size)]

    This maps naturally to your per-block quantization where blocks run
    along the inner (input) dimension for each output channel.

    Returns (scale_array, axis) — axis is always 1 here.
    """
    import math
    out_channels = weight_shape[0]
    inner = 1
    for d in weight_shape[1:]:
        inner *= d
    n_blocks_inner = math.ceil(inner / block_size)
    total_blocks   = out_channels * n_blocks_inner

    s = scales_1d.float()
    if s.numel() < total_blocks:
        pad = total_blocks - s.numel()
        s = torch.nn.functional.pad(s, (0, pad), value=float(s[-1]))
    s = s[:total_blocks].view(out_channels, n_blocks_inner)
    return _np_from_tensor(s), 1  # axis=1


# ---------------------------------------------------------------------------
# Core injector
# ---------------------------------------------------------------------------

def _inject_qdq_for_weight(
    graph: onnx.GraphProto,
    init_name: str,
    init_index: int,
    weight_fp16: torch.Tensor,
    scale_tensor: np.ndarray,   # shape [out, n_blocks_inner]
    block_size: int,
    qdq_axis: int = 1,
    zero_point_dtype: int = FLOAT8_DTYPE,
):
    """
    Replace a single weight initializer with a QDQ-wrapped version:

    BEFORE:
        [weight initializer] ──► [Conv/Gemm node]

    AFTER:
        [weight_dq initializer, fp16]
              │
        QuantizeLinear(scale, zp, block_size=B) ──► weight_q
              │
        DequantizeLinear(scale, zp, block_size=B) ──► weight_dq_out
              │
        [Conv/Gemm node]  (input renamed to weight_dq_out)
    """
    w_np = _np_from_tensor(weight_fp16)

    # Names for new nodes/tensors
    w_fp16_name  = f"{init_name}_fp16"
    scale_name   = f"{init_name}_qdq_scale"
    zp_name      = f"{init_name}_qdq_zp"
    q_out_name   = f"{init_name}_quantized"
    dq_out_name  = f"{init_name}_dequantized"

    # 1. Replace weight initializer with FP16 version
    w_init = numpy_helper.from_array(w_np.astype(np.float16), name=w_fp16_name)
    graph.initializer[init_index].CopyFrom(w_init)
    # Rename the original slot so downstream find works
    graph.initializer[init_index].name = w_fp16_name

    # 2. Scale initializer [out, n_blocks] — must match weight dtype (fp16)
    #    ORT opset21 QuantizeLinear requires scale dtype == input dtype.
    scale_init = numpy_helper.from_array(scale_tensor.astype(np.float16), name=scale_name)
    graph.initializer.append(scale_init)

    # 3. QuantizeLinear — output_dtype=FLOAT8E4M3FN (17), no explicit zero-point
    #    Omitting zp avoids the dtype-binding conflict across opset versions.
    #    TRT reads output_dtype to select the quantized storage type.
    q_node = helper.make_node(
        "QuantizeLinear",
        inputs=[w_fp16_name, scale_name],
        outputs=[q_out_name],
        name=f"QuantizeLinear_{init_name}",
        axis=qdq_axis,
        block_size=block_size,
        output_dtype=int(TensorProto.FLOAT8E4M3FN),
    )

    # 4. DequantizeLinear — reads the FP8 quantized tensor back to FP16
    dq_node = helper.make_node(
        "DequantizeLinear",
        inputs=[q_out_name, scale_name],
        outputs=[dq_out_name],
        name=f"DequantizeLinear_{init_name}",
        axis=qdq_axis,
        block_size=block_size,
    )

    # 5b. Cast DQ output back to FP32 so downstream Gemm/Conv (FP32 activations) 
    #     stays type-consistent. TRT fuses this cast away during engine build.
    cast_out_name = f"{init_name}_cast_fp32"
    cast_node = helper.make_node(
        "Cast",
        inputs=[dq_out_name],
        outputs=[cast_out_name],
        name=f"Cast_{init_name}",
        to=int(TensorProto.FLOAT),
    )

    # 6. Rewrite any node that consumed the original weight name
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp == init_name:
                node.input[i] = cast_out_name

    # 7. Insert QDQ + Cast nodes at the front of the node list
    graph.node.insert(0, cast_node)
    graph.node.insert(0, dq_node)
    graph.node.insert(0, q_node)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_qdq_onnx(
    checkpoint_dir: str | Path,
    base_onnx: str | Path,
    out_onnx: str | Path,
    opset: int = TARGET_OPSET,
    verbose: bool = True,
) -> Path:
    """
    Inject per-block QDQ nodes into a base ONNX model using the absorbed
    scales from save_for_trt_export().

    Parameters
    ----------
    checkpoint_dir : directory produced by save_for_trt_export()
    base_onnx      : FP32 ONNX export from PyTorch (torch.onnx.export)
    out_onnx       : where to write the QDQ-annotated model
    opset          : ONNX opset (21+ required for block_size attribute)

    Returns
    -------
    Path to the written .onnx file.
    """
    checkpoint_dir = Path(checkpoint_dir)
    base_onnx      = Path(base_onnx)
    out_onnx       = Path(out_onnx)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)

    # Load checkpoint artifacts
    weights_fp16 = torch.load(checkpoint_dir / "weights_fp16.pt",
                               map_location="cpu", weights_only=True)
    scales       = torch.load(checkpoint_dir / "scales.pt",
                               map_location="cpu", weights_only=True)
    with open(checkpoint_dir / "metadata.json") as f:
        meta = json.load(f)

    block_size = meta["block_size"]

    # Load base ONNX model and upgrade opset if needed
    model_proto = onnx.load(str(base_onnx))
    model_proto = _ensure_opset(model_proto, opset)
    graph       = model_proto.graph

    init_index_map = _find_weight_initializers(graph)

    injected = 0
    skipped  = []

    for param_name, scale_1d in scales.items():
        init_name = _param_name_to_init_name(param_name)

        if init_name not in init_index_map:
            skipped.append(param_name)
            continue

        if param_name not in weights_fp16:
            skipped.append(param_name)
            continue

        weight_shape = meta["layers"][param_name]["shape"]
        w_fp16       = weights_fp16[param_name]
        scale_array, qdq_axis = _make_block_scale(scale_1d, weight_shape, block_size)

        _inject_qdq_for_weight(
            graph,
            init_name  = init_name,
            init_index = init_index_map[init_name],
            weight_fp16 = w_fp16,
            scale_tensor = scale_array,
            block_size   = block_size,
            qdq_axis     = qdq_axis,
        )

        injected += 1
        if verbose:
            print(f"  QDQ injected : {param_name}  "
                  f"scale_shape={scale_array.shape}  "
                  f"sparsity={meta['layers'][param_name]['sparsity']*100:.1f}%")

    if skipped and verbose:
        print(f"\n  Skipped (not in ONNX graph): {skipped}")

    # Validate and save
    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, str(out_onnx))

    size_mb = out_onnx.stat().st_size / (1024**2)
    print(f"\n  QDQ ONNX saved → {out_onnx}  ({size_mb:.2f} MB)")
    print(f"  Layers injected : {injected}")
    print(f"  Block size      : {block_size}")
    print(f"  Opset           : {opset}")
    print()
    print("  Ready for TensorRT engine build:")
    print(f"    trtexec --onnx={out_onnx.name} \\")
    print( "            --fp4 --fp8 \\")
    print(f"            --saveEngine={out_onnx.stem}_fp4.trt \\")
    print( "            --workspace=4096")

    return out_onnx


def verify_qdq_onnx(onnx_path: str | Path, input_shape: tuple, verbose: bool = True):
    """
    Run the QDQ model through ONNX Runtime on CPU to verify it executes
    correctly before sending to TRT. Reports output shape and any errors.
    """
    import onnxruntime as ort

    onnx_path = Path(onnx_path)
    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    inp_name = sess.get_inputs()[0].name
    dummy    = np.random.randn(*input_shape).astype(np.float32)
    out      = sess.run(None, {inp_name: dummy})

    if verbose:
        print(f"  ORT verification passed")
        print(f"  Input  shape : {dummy.shape}")
        print(f"  Output shape : {out[0].shape}")

    return out


# ---------------------------------------------------------------------------
# Opset upgrade helper
# ---------------------------------------------------------------------------

def _ensure_opset(model: onnx.ModelProto, target: int) -> onnx.ModelProto:
    """Bump the opset version if the model was exported at a lower opset."""
    current = next(
        (op.version for op in model.opset_import if op.domain == ""),
        1,
    )
    if current < target:
        from onnx import version_converter
        model = version_converter.convert_version(model, target)
    return model
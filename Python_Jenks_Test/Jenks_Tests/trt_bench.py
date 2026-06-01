#!/usr/bin/env python3
"""
trt_bench.py — Build TensorRT FP32/FP16 engines from pre-exported ONNX files
               and benchmark vs PyTorch CPU baseline from prior CSV results.

Prerequisites:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    (TensorRT 10.x already installed)

Usage:
    python trt_bench.py                   # all models, FP16 + FP32
    python trt_bench.py --model LeNet300  # single model
    python trt_bench.py --fp32-only       # skip FP16
    python trt_bench.py --runs 500        # more timed runs
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

PROJECT_ROOT = Path(__file__).parent
ONNX_DIR     = PROJECT_ROOT / "onnx_exports"
RESULTS_DIR  = PROJECT_ROOT / "onnx_bench_results"

# Maps registry key → (onnx filename stem, input shape C×H×W)
MODELS: dict[str, tuple[str, tuple[int, int, int]]] = {
    "DenseNet40":              ("DenseNet40",              (3, 32, 32)),
    "LeNet300":                ("LeNet300",                (1, 28, 28)),
    "LeNet5":                  ("LeNet5",                  (1, 32, 32)),
    "ResNet32/CIFAR10":        ("ResNet32_CIFAR10",        (3, 32, 32)),
    "ResNet32/CIFAR100_85":    ("ResNet32_CIFAR100_85",    (3, 32, 32)),
    "ResNet32/CIFAR100_86":    ("ResNet32_CIFAR100_86",    (3, 32, 32)),
    "ResNet32/TinyImageNet":   ("ResNet32_TinyImageNet",   (3, 64, 64)),
    "ResNet56":                ("ResNet56",                (3, 32, 32)),
    "VGG19/CIFAR10":           ("VGG19_CIFAR10",           (3, 32, 32)),
    "VGG19/CIFAR100_90":       ("VGG19_CIFAR100_90",       (3, 32, 32)),
    "VGG19/CIFAR100_98":       ("VGG19_CIFAR100_98",       (3, 32, 32)),
    "VGG19/TinyImageNet":      ("VGG19_TinyImageNet",      (3, 64, 64)),
    "VGG19_Test/TinyImageNet": ("VGG19_Test_TinyImageNet", (3, 64, 64)),
}


# ---------------------------------------------------------------------------
# Engine build
# ---------------------------------------------------------------------------

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(
    onnx_path: Path,
    fp16: bool = True,
    workspace_mb: int = 1024,
) -> trt.ICudaEngine:
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser, \
         builder.create_builder_config() as config:

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"  TRT parse error {i}: {parser.get_error(i)}")
                raise RuntimeError(f"Failed to parse {onnx_path}")

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(f"build_serialized_network returned None for {onnx_path}")

    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(serialized)


# ---------------------------------------------------------------------------
# TRT inference benchmark
# ---------------------------------------------------------------------------

def _percentile(data: list[float], p: float) -> float:
    if not data:
        return float("nan")
    s = sorted(data)
    idx = (len(s) - 1) * p / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def bench_engine(
    engine: trt.ICudaEngine,
    input_shape: tuple,        # (batch, C, H, W)
    warmup: int = 20,
    timed: int = 200,
) -> dict:
    context = engine.create_execution_context()
    stream  = torch.cuda.Stream()

    dummy = torch.randn(*input_shape, dtype=torch.float32, device="cuda")

    # Allocate output buffer
    out_shape = tuple(engine.get_tensor_shape(engine.get_tensor_name(1)))
    out_shape = (input_shape[0],) + out_shape[1:]  # replace batch dim
    out_buf = torch.empty(out_shape, dtype=torch.float32, device="cuda")

    in_name  = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)
    context.set_tensor_address(in_name,  dummy.data_ptr())
    context.set_tensor_address(out_name, out_buf.data_ptr())

    for _ in range(warmup):
        context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

    latencies: list[float] = []
    t_wall_start = time.perf_counter()
    for _ in range(timed):
        t0 = time.perf_counter()
        context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    total_wall = time.perf_counter() - t_wall_start

    return {
        "p50_ms":   _percentile(latencies, 50),
        "p95_ms":   _percentile(latencies, 95),
        "thr_i_s":  timed / total_wall,
        "min_ms":   min(latencies),
        "max_ms":   max(latencies),
    }


# ---------------------------------------------------------------------------
# Load PT CPU baseline from existing comparison CSV
# ---------------------------------------------------------------------------

def load_pt_baselines() -> dict[str, dict]:
    """Return {model_name: row} from the latest comparison CSV."""
    baselines: dict[str, dict] = {}
    for path in sorted(RESULTS_DIR.glob("*_comparison.csv")):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                baselines[row["Model"]] = row
    return baselines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--runs",  type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--fp32-only", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    keys = [args.model] if args.model else list(MODELS.keys())
    modes = ["fp32"] if args.fp32_only else ["fp32", "fp16"]

    print(f"TensorRT {trt.__version__}  |  GPU: {torch.cuda.get_device_name(0)}")
    print(f"Modes: {modes}  |  Runs: {args.runs}  |  Warmup: {args.warmup}\n")

    pt_baselines = load_pt_baselines()
    rows: list[dict] = []

    for key in keys:
        stem, chw = MODELS[key]
        onnx_path = ONNX_DIR / f"{stem}.onnx"
        if not onnx_path.exists():
            print(f"  [SKIP] {key}: {onnx_path} not found")
            continue

        input_shape = (args.batch_size, *chw)
        pt_p50 = pt_baselines.get(key, {}).get("PT p50 (ms)", "—")
        pt_thr  = pt_baselines.get(key, {}).get("PT thr (i/s)", "—")
        sparsity = pt_baselines.get(key, {}).get("Sparsity", "—")

        print(f"{'='*60}")
        print(f"  {key}  (sparsity={sparsity}, input={input_shape})")
        print(f"  PT CPU baseline: p50={pt_p50} ms  thr={pt_thr} i/s")

        for mode in modes:
            fp16 = (mode == "fp16")
            label = "TRT FP16" if fp16 else "TRT FP32"
            print(f"  Building {label} engine ...", end="", flush=True)
            try:
                t0 = time.perf_counter()
                engine = build_engine(onnx_path, fp16=fp16)
                build_s = time.perf_counter() - t0
                print(f" done ({build_s:.1f}s)")
            except Exception as e:
                print(f" FAILED: {e}")
                continue

            print(f"  Benchmarking {label} ({args.runs} runs) ...", end="", flush=True)
            try:
                r = bench_engine(engine, input_shape, warmup=args.warmup, timed=args.runs)
            except Exception as e:
                print(f" FAILED: {e}")
                del engine
                continue

            speedup_vs_cpu = (
                f"{float(pt_p50) / r['p50_ms']:.2f}×"
                if pt_p50 != "—" else "—"
            )
            print(f" p50={r['p50_ms']:.3f}ms  thr={r['thr_i_s']:.0f}i/s  "
                  f"speedup_vs_PT_CPU={speedup_vs_cpu}")

            rows.append({
                "Model":            key,
                "Sparsity":         sparsity,
                "Mode":             label,
                "TRT p50 (ms)":     f"{r['p50_ms']:.3f}",
                "TRT p95 (ms)":     f"{r['p95_ms']:.3f}",
                "TRT thr (i/s)":    f"{r['thr_i_s']:.0f}",
                "PT CPU p50 (ms)":  pt_p50,
                "Speedup vs PT CPU": speedup_vs_cpu,
            })
            del engine

    if not rows:
        print("No results.")
        return

    # ---- Speedup summary table ------------------------------------------
    print(f"\n{'='*60}")
    print("  TensorRT Speedup Summary")
    print(f"{'='*60}")
    headers = list(rows[0].keys())
    col_w = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers}
    sep = "+-" + "-+-".join("-" * col_w[h] for h in headers) + "-+"
    print(sep)
    print("| " + " | ".join(h.ljust(col_w[h]) for h in headers) + " |")
    print(sep)
    for r in rows:
        print("| " + " | ".join(str(r.get(h, "")).ljust(col_w[h]) for h in headers) + " |")
    print(sep)

    # ---- Save CSV --------------------------------------------------------
    stem = f"trt_bench_{int(time.time())}"
    csv_path = RESULTS_DIR / f"{stem}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Results → {csv_path}")


if __name__ == "__main__":
    main()

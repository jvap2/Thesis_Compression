#!/usr/bin/env python3
"""
benchmark.py — CLI entry point for the benchmarking harness.

Usage examples
--------------
# Run all formats on MobileNetV2 (default)
python benchmark.py

# Specific formats only
python benchmark.py --formats fp32,int8,sparse

# ResNet-50, batch size 4, more runs
python benchmark.py --model resnet50 --batch-size 4 --runs 500

# Custom sparsity
python benchmark.py --formats fp32,sparse --sparsity 0.95

# Skip chart (faster, no matplotlib needed)
python benchmark.py --no-chart

Available models: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large,
                  resnet18, resnet34, resnet50, shufflenet_v2_x0_5,
                  squeezenet1_0, efficientnet_b0
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torchvision.models as tvm

# Make harness importable from the script's directory
sys.path.insert(0, str(Path(__file__).parent))
from harness import load_format, run_benchmark, report_all
from harness.power import get_meter


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "mobilenet_v2":        tvm.mobilenet_v2,
    "mobilenet_v3_small":  tvm.mobilenet_v3_small,
    "mobilenet_v3_large":  tvm.mobilenet_v3_large,
    "resnet18":            tvm.resnet18,
    "resnet34":            tvm.resnet34,
    "resnet50":            tvm.resnet50,
    "shufflenet_v2_x0_5":  tvm.shufflenet_v2_x0_5,
    "squeezenet1_0":       tvm.squeezenet1_0,
    "efficientnet_b0":     tvm.efficientnet_b0,
}


def get_model_fn(name: str):
    name = name.lower().strip()
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}"
        )
    fn = MODEL_REGISTRY[name]
    # Use weights=None to avoid network download in CI; swap to
    # weights="IMAGENET1K_V1" locally for realistic weight distributions.
    return lambda: fn(weights=None)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="CPU inference benchmarking harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="mobilenet_v2",
                   help="Model name (default: mobilenet_v2)")
    p.add_argument("--formats", default="fp32,int8,int4,fp4,sparse",
                   help="Comma-separated list of formats to benchmark")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=20,
                   help="Warmup runs (discarded)")
    p.add_argument("--runs", type=int, default=200,
                   help="Timed runs per format")
    p.add_argument("--sparsity", type=float, default=0.90,
                   help="Pruning sparsity for 'sparse' format (default 0.90)")
    p.add_argument("--output-dir", default="results",
                   help="Directory for JSON/CSV/chart output")
    p.add_argument("--stem", default=None,
                   help="Filename stem (default: {model}_{timestamp})")
    p.add_argument("--no-chart", action="store_true",
                   help="Skip matplotlib chart generation")
    p.add_argument("--list-models", action="store_true",
                   help="Print available model names and exit")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_models:
        print("Available models:")
        for name in sorted(MODEL_REGISTRY):
            print(f"  {name}")
        sys.exit(0)

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    stem = args.stem or f"{args.model}_{int(time.time())}"

    print(f"\n  Benchmarking harness")
    print(f"  Model      : {args.model}")
    print(f"  Formats    : {formats}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Warmup     : {args.warmup}  |  Timed runs: {args.runs}")
    print(f"  Power      : {get_meter().info()}")
    print()

    model_fn = get_model_fn(args.model)
    input_shape = (args.batch_size, 3, 224, 224)

    results = []
    for fmt in formats:
        print(f"  [{fmt.upper():12s}] loading ...", end="", flush=True)
        try:
            model, meta = load_format(fmt, model_fn, sparsity=args.sparsity)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        print(f" running {args.runs} inferences ...", end="", flush=True)
        try:
            result = run_benchmark(
                model, meta,
                input_shape=input_shape,
                warmup_runs=args.warmup,
                timed_runs=args.runs,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        results.append(result)
        print(f" p50={result.latency_p50_ms:.2f}ms  thr={result.throughput_inf_per_s:.0f}i/s  "
              f"size={result.model_size_mb:.1f}MB  E={result.energy_per_inf_mj:.2f}mJ")

        # Free memory before next format
        del model

    if results:
        report_all(
            results,
            output_dir=args.output_dir,
            stem=stem,
            chart=not args.no_chart,
        )
    else:
        print("  No results to report.")
        sys.exit(1)


if __name__ == "__main__":
    main()
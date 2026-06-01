#!/usr/bin/env python3
"""
onnx_bench.py — Export all Best_Results_HPO pruned networks to ONNX
                 and benchmark PyTorch vs ONNX Runtime side-by-side.

Usage:
    python onnx_bench.py                   # all models, 200 timed runs
    python onnx_bench.py --runs 50         # faster, fewer runs
    python onnx_bench.py --model ResNet56  # single model
    python onnx_bench.py --list            # print available keys and exit
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import psutil
import torch

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import onnxruntime as ort


# ---------------------------------------------------------------------------
# ONNX compatibility wrappers
# ---------------------------------------------------------------------------

class _ResNetONNXWrapper(torch.nn.Module):
    """
    Thin wrapper for ResNet-style models that replaces
      F.avg_pool2d(out, out.size()[3])   (dynamic — ONNX-incompatible)
    with
      F.adaptive_avg_pool2d(out, 1)      (ONNX opset 9+ supported)
    Works for both ResNet32 (initial BN named 'bn') and
    ResNet56 legacy constructor (initial BN named 'bn1').
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self._m = model

    def forward(self, x):
        initial_bn = getattr(self._m, 'bn', None) or self._m.bn1
        out = torch.nn.functional.relu(initial_bn(self._m.conv1(x)))
        out = self._m.layer1(out)
        out = self._m.layer2(out)
        out = self._m.layer3(out)
        out = torch.nn.functional.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self._m.linear(out)


def _make_export_model(model: torch.nn.Module) -> torch.nn.Module:
    """Wrap models that need ONNX-compatibility fixes."""
    if hasattr(model, 'layer1') and hasattr(model, 'layer2') and hasattr(model, 'layer3'):
        return _ResNetONNXWrapper(model)
    return model

from harness.model_registry import load_pruned_model, list_models
from harness.quant_io import export_onnx
from harness.runner import BenchResult, _percentile
from harness.reporter import print_table, save_json, save_csv, save_chart

ONNX_DIR    = PROJECT_ROOT / "onnx_exports"
RESULTS_DIR = PROJECT_ROOT / "onnx_bench_results"


# ---------------------------------------------------------------------------
# ONNX Runtime benchmark (mirrors harness/runner.py logic)
# ---------------------------------------------------------------------------

def run_onnx_benchmark(
    onnx_path: Path,
    input_shape: tuple,          # (batch, C, H, W)
    warmup: int = 20,
    timed: int = 200,
) -> BenchResult:
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name
    dummy = np.random.randn(*input_shape).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {input_name: dummy})

    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

    latencies: list[float] = []
    t_start = time.perf_counter()
    for _ in range(timed):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000.0)
    total_wall = time.perf_counter() - t_start

    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    rss_after = proc.memory_info().rss

    heap_delta = sum(
        s.size_diff for s in snap_after.compare_to(snap_before, "lineno")
    )
    size_mb = onnx_path.stat().st_size / (1024 ** 2)

    return BenchResult(
        format="ONNX (ORT CPU)",
        model_size_mb=size_mb,
        notes=f"opset 17 · OnnxRuntime {ort.__version__} · CPUExecutionProvider",
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        latency_p99_ms=_percentile(latencies, 99),
        latency_mean_ms=sum(latencies) / len(latencies),
        latency_min_ms=min(latencies),
        latency_max_ms=max(latencies),
        throughput_inf_per_s=timed / total_wall,
        rss_delta_mb=(rss_after - rss_before) / (1024 ** 2),
        heap_delta_mb=heap_delta / (1024 ** 2),
        power_backend="n/a",
        warmup_runs=warmup,
        timed_runs=timed,
        batch_size=input_shape[0],
    )


# ---------------------------------------------------------------------------
# PyTorch benchmark (thin wrapper around harness runner)
# ---------------------------------------------------------------------------

def run_pt_benchmark(
    model: torch.nn.Module,
    meta: dict,
    input_shape: tuple,
    warmup: int = 20,
    timed: int = 200,
) -> BenchResult:
    from harness.runner import run_benchmark
    return run_benchmark(model, meta, input_shape=input_shape,
                         warmup_runs=warmup, timed_runs=timed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ONNX export + benchmark for pruned models")
    p.add_argument("--model", default=None,
                   help="Single registry key to run (default: all models)")
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--no-chart", action="store_true")
    p.add_argument("--list", action="store_true", help="Print registry keys and exit")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("Available registry keys:")
        for k in list_models():
            print(f"  {k}")
        return

    keys = [args.model] if args.model else list_models()
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[tuple[str, BenchResult, BenchResult]] = []

    for key in keys:
        print(f"\n{'='*60}")
        print(f"  Model: {key}")
        print(f"{'='*60}")

        # ---- Load pruned model ------------------------------------------
        try:
            model, entry, meta = load_pruned_model(
                key, project_root=PROJECT_ROOT, device=torch.device("cpu")
            )
        except Exception as e:
            print(f"  [SKIP] load failed: {e}")
            continue

        input_shape = (args.batch_size, *entry.input_shape)
        sparsity_pct = float(meta["format"].split("(")[1].rstrip("% sparse)"))
        print(f"  Sparsity : {sparsity_pct:.1f}%")
        print(f"  Size     : {meta['model_size_mb']:.2f} MB")
        print(f"  Input    : {input_shape}")

        # ---- PyTorch benchmark ------------------------------------------
        print(f"  Benchmarking PyTorch ({args.runs} runs) ...", end="", flush=True)
        try:
            pt_result = run_pt_benchmark(
                model, meta, input_shape,
                warmup=args.warmup, timed=args.runs,
            )
            print(f" p50={pt_result.latency_p50_ms:.2f}ms  "
                  f"thr={pt_result.throughput_inf_per_s:.0f}i/s")
        except Exception as e:
            print(f"  FAILED: {e}")
            pt_result = None

        # ---- ONNX export ------------------------------------------------
        safe_key = key.replace("/", "_")
        onnx_path = ONNX_DIR / f"{safe_key}.onnx"
        print(f"  Exporting to ONNX → {onnx_path.name} ...", end="", flush=True)
        try:
            export_model = _make_export_model(model)
            export_onnx(export_model, entry.input_shape, onnx_path, opset=17,
                        batch_size=args.batch_size)
        except Exception as e:
            print(f"  FAILED: {e}")
            del model
            continue

        # ---- ONNX Runtime benchmark -------------------------------------
        print(f"  Benchmarking ONNX Runtime ({args.runs} runs) ...", end="", flush=True)
        try:
            ort_result = run_onnx_benchmark(
                onnx_path, input_shape,
                warmup=args.warmup, timed=args.runs,
            )
            print(f" p50={ort_result.latency_p50_ms:.2f}ms  "
                  f"thr={ort_result.throughput_inf_per_s:.0f}i/s")
        except Exception as e:
            print(f"  FAILED: {e}")
            ort_result = None

        if pt_result and ort_result:
            speedup = pt_result.latency_p50_ms / ort_result.latency_p50_ms
            print(f"  ORT speedup vs PyTorch: {speedup:.2f}×")
            all_results.append((key, pt_result, ort_result))

        del model

    # ---- Per-model side-by-side tables ----------------------------------
    print(f"\n\n{'='*60}")
    print("  FULL RESULTS")
    print(f"{'='*60}")

    flat_results: list[BenchResult] = []
    speedup_rows: list[dict] = []

    for key, pt_r, ort_r in all_results:
        # Tag format with model name for the table
        pt_r_tagged  = BenchResult(**{**pt_r.to_dict(),  "format": f"[PT]  {key}"})
        ort_r_tagged = BenchResult(**{**ort_r.to_dict(), "format": f"[ORT] {key}"})
        flat_results.extend([pt_r_tagged, ort_r_tagged])

        speedup = pt_r.latency_p50_ms / ort_r.latency_p50_ms if ort_r.latency_p50_ms > 0 else float("nan")
        speedup_rows.append({
            "Model":           key,
            "Sparsity":        pt_r.notes.split("Measured sparsity:")[1].split(".")[0].strip() + "%"
                               if "Measured sparsity:" in pt_r.notes else "—",
            "PT p50 (ms)":     f"{pt_r.latency_p50_ms:.2f}",
            "ORT p50 (ms)":    f"{ort_r.latency_p50_ms:.2f}",
            "ORT speedup":     f"{speedup:.2f}×",
            "PT size (MB)":    f"{pt_r.model_size_mb:.2f}",
            "ONNX size (MB)":  f"{ort_r.model_size_mb:.2f}",
            "PT thr (i/s)":    f"{pt_r.throughput_inf_per_s:.0f}",
            "ORT thr (i/s)":   f"{ort_r.throughput_inf_per_s:.0f}",
        })

    print_table(flat_results, title="PyTorch (pruned) vs ONNX Runtime — all models")

    # ---- Speedup summary table ------------------------------------------
    if speedup_rows:
        headers = list(speedup_rows[0].keys())
        col_w = {h: max(len(h), max(len(r[h]) for r in speedup_rows)) for h in headers}
        sep = "+-" + "-+-".join("-" * col_w[h] for h in headers) + "-+"
        print("\n  Speedup summary")
        print(sep)
        print("| " + " | ".join(h.ljust(col_w[h]) for h in headers) + " |")
        print(sep)
        for r in speedup_rows:
            print("| " + " | ".join(r[h].ljust(col_w[h]) for h in headers) + " |")
        print(sep)

    # ---- Save outputs ---------------------------------------------------
    stem = f"onnx_bench_{int(time.time())}"
    save_json(flat_results, RESULTS_DIR / f"{stem}.json")
    save_csv(flat_results,  RESULTS_DIR / f"{stem}.csv")
    if not args.no_chart:
        save_chart(flat_results, RESULTS_DIR / f"{stem}_chart.png")

    # ---- Save comparison CSV (PT vs ORT side-by-side with speedup) -------
    if speedup_rows:
        import csv as _csv
        cmp_path = RESULTS_DIR / f"{stem}_comparison.csv"
        with open(cmp_path, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=list(speedup_rows[0].keys()))
            writer.writeheader()
            writer.writerows(speedup_rows)
        print(f"  Comparison CSV → {cmp_path}")

    print(f"\n  Results written to {RESULTS_DIR}/")
    print(f"  ONNX files written to {ONNX_DIR}/")


if __name__ == "__main__":
    main()

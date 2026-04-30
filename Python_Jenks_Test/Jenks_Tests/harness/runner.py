"""
runner.py — Inference benchmarking loop.

Returns a BenchResult dataclass with all metrics for one (model, format) run.

Timing methodology
------------------
- time.perf_counter() wraps each individual forward pass.
- Warmup runs (default 20) are discarded — they prime caches and JIT.
- Timed runs (default 200) collect per-run latency in ms.
- Percentiles computed from the timed array: p50, p95, p99.
- Throughput = timed_runs / total_timed_wall_time  (inferences/sec).

Memory methodology
------------------
- tracemalloc captures Python heap delta during the forward pass.
- psutil.Process.memory_info().rss before and after the timed block
  gives OS-level RSS delta (includes PyTorch C++ allocator).
- Both are reported; RSS is the more meaningful number.

Energy methodology
------------------
- PowerMeter.start() before the timed block.
- PowerMeter.delta() after the block returns total Joules.
- Energy-per-inference = total_joules / timed_runs.
- Average power = total_joules / total_timed_wall_time (Watts).
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, List, Optional

import psutil
import torch
import torch.nn as nn

from .power import get_meter


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    format: str
    model_size_mb: float
    notes: str

    # Latency (ms)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_mean_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0

    # Throughput
    throughput_inf_per_s: float = 0.0

    # Memory
    rss_delta_mb: float = 0.0
    heap_delta_mb: float = 0.0

    # Energy
    energy_total_j: float = 0.0
    energy_per_inf_mj: float = 0.0
    avg_power_w: float = 0.0
    power_backend: str = ""

    # Run config
    warmup_runs: int = 0
    timed_runs: int = 0
    batch_size: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_row(self) -> dict:
        """Compact dict for the console table."""
        return {
            "Format":          self.format,
            "Size (MB)":       f"{self.model_size_mb:.2f}",
            "p50 (ms)":        f"{self.latency_p50_ms:.2f}",
            "p95 (ms)":        f"{self.latency_p95_ms:.2f}",
            "Throughput (i/s)":f"{self.throughput_inf_per_s:.1f}",
            "RSS Δ (MB)":      f"{self.rss_delta_mb:.1f}",
            "E/inf (mJ)":      f"{self.energy_per_inf_mj:.2f}",
            "Avg power (W)":   f"{self.avg_power_w:.2f}",
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _percentile(data: List[float], p: float) -> float:
    sorted_d = sorted(data)
    idx = int(len(sorted_d) * p / 100)
    idx = min(idx, len(sorted_d) - 1)
    return sorted_d[idx]


def run_benchmark(
    model: nn.Module,
    meta: dict,
    input_shape: tuple = (1, 3, 224, 224),
    warmup_runs: int = 20,
    timed_runs: int = 200,
    dtype: torch.dtype = torch.float32,
) -> BenchResult:
    """
    Benchmark a single model variant.

    Parameters
    ----------
    model       : already-loaded model (from loader.load_format)
    meta        : metadata dict returned by the loader
    input_shape : (batch, C, H, W) for CNN models
    warmup_runs : discarded warm-up passes
    timed_runs  : measured passes
    dtype       : input tensor dtype (match model precision)
    """
    model.eval()
    proc = psutil.Process()
    meter = get_meter()

    # Build a random input (on CPU)
    dummy = torch.randn(*input_shape, dtype=dtype)

    # --- Warmup ---
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy)

    # --- Timed block ---
    rss_before = proc.memory_info().rss

    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

    meter.start()
    t_block_start = time.perf_counter()

    latencies_ms: List[float] = []
    with torch.no_grad():
        for _ in range(timed_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    t_block_end = time.perf_counter()
    total_joules = meter.delta()

    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    rss_after = proc.memory_info().rss

    # --- Compute metrics ---
    total_wall = t_block_end - t_block_start

    # Memory
    heap_delta = sum(
        stat.size_diff
        for stat in snap_after.compare_to(snap_before, "lineno")
    )
    rss_delta_mb = (rss_after - rss_before) / (1024 ** 2)
    heap_delta_mb = heap_delta / (1024 ** 2)

    # Energy
    energy_per_inf_mj = (total_joules / timed_runs) * 1000.0
    avg_power_w = total_joules / total_wall if total_wall > 0 else 0.0

    return BenchResult(
        format=meta["format"],
        model_size_mb=meta["model_size_mb"],
        notes=meta["notes"],
        latency_p50_ms=_percentile(latencies_ms, 50),
        latency_p95_ms=_percentile(latencies_ms, 95),
        latency_p99_ms=_percentile(latencies_ms, 99),
        latency_mean_ms=sum(latencies_ms) / len(latencies_ms),
        latency_min_ms=min(latencies_ms),
        latency_max_ms=max(latencies_ms),
        throughput_inf_per_s=timed_runs / total_wall,
        rss_delta_mb=rss_delta_mb,
        heap_delta_mb=heap_delta_mb,
        energy_total_j=total_joules,
        energy_per_inf_mj=energy_per_inf_mj,
        avg_power_w=avg_power_w,
        power_backend=meter.backend,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        batch_size=input_shape[0],
    )
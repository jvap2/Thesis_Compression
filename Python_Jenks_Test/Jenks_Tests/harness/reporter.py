"""
reporter.py — Formats and persists BenchResult lists.

Outputs
-------
console  Pretty-printed comparison table (always).
json     Full results with all fields, one object per format.
csv      Summary rows for easy import into Excel / pandas.
chart    Latency + throughput + energy bar charts (optional, requires matplotlib).
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import List, Optional

from .runner import BenchResult


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

def _col_widths(rows: List[dict], headers: List[str]) -> dict:
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))
    return widths


def print_table(results: List[BenchResult], title: str = "Benchmark results"):
    rows = [r.summary_row() for r in results]
    if not rows:
        print("No results.")
        return

    headers = list(rows[0].keys())
    widths = _col_widths(rows, headers)

    sep = "+-" + "-+-".join("-" * widths[h] for h in headers) + "-+"
    header_row = "| " + " | ".join(h.ljust(widths[h]) for h in headers) + " |"

    print()
    print(f"  {title}")
    print(sep)
    print(header_row)
    print(sep)
    for row in rows:
        line = "| " + " | ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers) + " |"
        print(line)
    print(sep)

    # Print power backend notice
    backends = {r.power_backend for r in results}
    backend_str = ", ".join(sorted(backends))
    print(f"\n  Power backend: {backend_str}")
    if "psutil" in backend_str:
        print("  Note: psutil power is estimated from CPU% × TDP.")
        print("        Set TDP_WATTS env var to match your CPU (default 15 W).")
    print()


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def save_json(results: List[BenchResult], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [r.to_dict() for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  JSON saved → {path}")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def save_csv(results: List[BenchResult], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.summary_row() for r in results]
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV  saved → {path}")


# ---------------------------------------------------------------------------
# Chart (matplotlib — optional)
# ---------------------------------------------------------------------------

def save_chart(results: List[BenchResult], path: str | Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  matplotlib not installed — skipping chart. pip install matplotlib")
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = [r.format for r in results]
    n = len(labels)
    x = range(n)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Benchmarking harness — CPU inference comparison", fontsize=13)

    palette = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD"][:n]

    def _bar(ax, values, title, ylabel, fmt=".2f"):
        bars = ax.bar(x, values, color=palette, width=0.5, edgecolor="none")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:{fmt}}",
                ha="center", va="bottom", fontsize=8,
            )

    _bar(axes[0], [r.latency_p50_ms for r in results], "p50 latency", "ms")
    _bar(axes[1], [r.throughput_inf_per_s for r in results], "Throughput", "inf/s", ".1f")
    _bar(axes[2], [r.model_size_mb for r in results], "Model size", "MB")
    _bar(axes[3], [r.energy_per_inf_mj for r in results], "Energy / inference", "mJ")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {path}")


# ---------------------------------------------------------------------------
# All-in-one
# ---------------------------------------------------------------------------

def report_all(
    results: List[BenchResult],
    output_dir: str | Path = "results",
    stem: str = "bench",
    chart: bool = True,
):
    output_dir = Path(output_dir)
    print_table(results)
    save_json(results, output_dir / f"{stem}.json")
    save_csv(results, output_dir / f"{stem}.csv")
    if chart:
        save_chart(results, output_dir / f"{stem}_chart.png")
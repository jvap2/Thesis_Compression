#!/usr/bin/env python3
"""
make_onnx_report.py — Consolidate per-model onnx_bench_results CSVs into
                       a single summary with PT vs ORT side-by-side + speedup.

Run after onnx_bench.py has produced results for all models.
Writes  onnx_bench_results/full_report.csv  and  full_report.txt
"""

from __future__ import annotations
import csv
import json
import re
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "onnx_bench_results"


# ---------------------------------------------------------------------------
# Load and deduplicate: keep the latest entry per (model, format)
# ---------------------------------------------------------------------------

def _model_key(fmt: str) -> tuple[str, str]:
    """Split '[PT]  ModelName' → ('ModelName', 'PT')."""
    fmt = fmt.strip()
    if fmt.startswith("[PT]"):
        return fmt.replace("[PT]", "").strip(), "PT"
    if fmt.startswith("[ORT]"):
        return fmt.replace("[ORT]", "").strip(), "ORT"
    return fmt, "?"


def _load_sparsity_from_json(results_dir: Path) -> dict[str, str]:
    """Read sparsity % from PT notes in JSON benchmark files."""
    sparsity: dict[str, str] = {}
    for jpath in sorted(results_dir.glob("onnx_bench_*.json")):
        with open(jpath) as f:
            try:
                entries = json.load(f)
            except json.JSONDecodeError:
                continue
        for entry in entries:
            fmt   = entry.get("format", "")
            notes = entry.get("notes", "")
            if fmt.startswith("[PT]"):
                model = fmt.replace("[PT]", "").strip()
                m = re.search(r"Measured sparsity: ([\d.]+)%", notes)
                if m:
                    sparsity[model] = m.group(1) + "%"
    return sparsity


def load_latest_results(results_dir: Path):
    """
    Read all *_comparison.csv files first (newer, more explicit), then fall
    back to the raw per-format CSVs.  Also reads trt_bench_*.csv if present.
    Returns (cmp_rows, raw, trt_rows).
    """
    # --- Prefer comparison CSVs (have speedup already) ---
    cmp_rows: dict[str, dict] = {}  # model_key → row
    for path in sorted(results_dir.glob("*_comparison.csv")):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                cmp_rows[row["Model"]] = row   # last write wins (latest file)

    # --- Fall back to raw format CSVs for models not in comparison files ---
    raw: dict[tuple[str, str], dict] = {}
    for path in sorted(results_dir.glob("onnx_bench_*.csv")):
        if "_comparison" in path.name:
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                model_name, side = _model_key(row["Format"])
                raw[(model_name, side)] = row  # last write wins (latest file)

    # --- TRT results (keyed by (model, mode)) ---
    trt_rows: dict[tuple[str, str], dict] = {}  # (model, "TRT FP16"|"TRT FP32") → row
    for path in sorted(results_dir.glob("trt_bench_*.csv")):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                trt_rows[(row["Model"], row["Mode"])] = row  # last write wins

    return cmp_rows, raw, trt_rows


# ---------------------------------------------------------------------------
# Build unified table
# ---------------------------------------------------------------------------

def build_table(cmp_rows, raw, trt_rows) -> list[dict]:
    rows = []

    all_models = sorted(
        set(cmp_rows.keys()) | {m for m, _ in raw.keys()}
    )

    for model in all_models:
        r = cmp_rows.get(model, {})
        if r:
            pt_p50  = r.get("PT p50 (ms)", "—")
            ort_p50 = r.get("ORT p50 (ms)", "—")
            speedup_ort = r.get("ORT speedup", "—")
            pt_size  = r.get("PT size (MB)", "—")
            onnx_size = r.get("ONNX size (MB)", "—")
            pt_thr   = r.get("PT thr (i/s)", "—")
            ort_thr  = r.get("ORT thr (i/s)", "—")
            sparsity = r.get("Sparsity", "—")
        else:
            pt  = raw.get((model, "PT"),  {})
            ort = raw.get((model, "ORT"), {})
            pt_p50_f  = float(pt.get("p50 (ms)", "nan"))
            ort_p50_f = float(ort.get("p50 (ms)", "nan"))
            pt_p50   = f"{pt_p50_f:.2f}" if pt else "—"
            ort_p50  = f"{ort_p50_f:.2f}" if ort else "—"
            speedup_ort = f"{pt_p50_f / ort_p50_f:.2f}×" if ort_p50_f > 0 else "—"
            pt_size  = pt.get("Size (MB)", "—")
            onnx_size = ort.get("Size (MB)", "—")
            pt_thr   = pt.get("Throughput (i/s)", "—")
            ort_thr  = ort.get("Throughput (i/s)", "—")
            sparsity = "—"

        fp16 = trt_rows.get((model, "TRT FP16"), {})
        fp32 = trt_rows.get((model, "TRT FP32"), {})

        rows.append({
            "Model":              model,
            "Sparsity":           sparsity,
            "PT p50 (ms)":        pt_p50,
            "ORT p50 (ms)":       ort_p50,
            "ORT speedup":        speedup_ort,
            "TRT FP32 p50 (ms)":  fp32.get("TRT p50 (ms)", "—"),
            "TRT FP32 speedup":   fp32.get("Speedup vs PT CPU", "—"),
            "TRT FP16 p50 (ms)":  fp16.get("TRT p50 (ms)", "—"),
            "TRT FP16 speedup":   fp16.get("Speedup vs PT CPU", "—"),
            "PT size (MB)":       pt_size,
            "ONNX size (MB)":     onnx_size,
            "PT thr (i/s)":       pt_thr,
            "ORT thr (i/s)":      ort_thr,
            "TRT FP16 thr (i/s)": fp16.get("TRT thr (i/s)", "—"),
        })

    return rows


# ---------------------------------------------------------------------------
# Text table formatter
# ---------------------------------------------------------------------------

def _txt_table(rows: list[dict], title: str) -> str:
    if not rows:
        return "(no data)\n"
    headers = list(rows[0].keys())
    widths = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers}
    sep = "+-" + "-+-".join("-" * widths[h] for h in headers) + "-+"
    hdr = "| " + " | ".join(h.ljust(widths[h]) for h in headers) + " |"
    lines = [f"\n{title}", sep, hdr, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(h, "")).ljust(widths[h]) for h in headers) + " |")
    lines.append(sep)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cmp_rows, raw, trt_rows = load_latest_results(RESULTS_DIR)
    sparsity_map = _load_sparsity_from_json(RESULTS_DIR)
    table = build_table(cmp_rows, raw, trt_rows)

    # Inject sparsity for rows that have "—"
    for row in table:
        if row["Sparsity"] == "—":
            row["Sparsity"] = sparsity_map.get(row["Model"], "—")

    if not table:
        print("No results found in", RESULTS_DIR)
        return

    # ---- CSV ----
    csv_path = RESULTS_DIR / "full_report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)

    # ---- Text report ----
    txt_path = RESULTS_DIR / "full_report.txt"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = (
        f"ONNX Export & Inference Benchmark Report\n"
        f"Generated: {ts}\n"
        f"Platform : CPU-only  (GPU driver unavailable — nvidia-smi fails)\n"
        f"Runtime  : OnnxRuntime CPUExecutionProvider\n"
        f"Runs     : 50–100 timed inferences, batch=1\n"
        f"Models   : Best_Results_HPO pruned networks\n"
        f"Note     : VGG19/CIFAR100_90 ORT slowdown is genuine (confirmed 2 runs);\n"
        f"           likely an ORT graph-fusion miss for this weight distribution.\n"
    )

    txt_body = _txt_table(table, "PyTorch (pruned) vs ONNX Runtime — full summary")

    with open(txt_path, "w") as f:
        f.write(header + txt_body)

    print(f"  Full report CSV  → {csv_path}")
    print(f"  Full report text → {txt_path}")
    print(txt_body)


if __name__ == "__main__":
    main()

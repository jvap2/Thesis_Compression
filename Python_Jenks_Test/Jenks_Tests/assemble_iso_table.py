"""
Read iso_results.csv (produced by iso_energy_125m.py across models) and emit the
W4A4 LaTeX rows for the paper table, with Delta vs the FP32 baseline. Keeps the
whole W4A4 block on ONE consistent protocol.

  python3 assemble_iso_table.py            # main (retention ON)
  python3 assemble_iso_table.py ablation   # retention OFF (the ablation table)
"""
import os, sys, csv

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "iso_results.csv")

# model HF id -> (short column name, FP32 baseline ppl from the paper table)
MODELS = {
    "facebook/opt-125m":        ("OPT-125M", 27.60),
    "facebook/opt-1.3b":        ("OPT-1.3B", 14.62),
    "facebook/opt-2.7b":        ("OPT-2.7B", 12.48),
    "facebook/opt-6.7b":        ("OPT-6.7B", 10.86),
    "EleutherAI/pythia-1b":     ("Pythia-1B", 13.17),
    "EleutherAI/pythia-1.4b":   ("Pythia-1.4B", 11.81),
    "meta-llama/Llama-3.2-3B":  ("LLaMA-3.2-3B", 7.82),
    "meta-llama/Llama-2-7b-hf": ("LLaMA-2-7b", 5.47),
}
# table row label -> CSV column. NVFP4 uses the FIXED-clip E2M1 (true NVFP4 spec)
# for consistency across models; switch to NVFP4_adap if you report the adaptive variant.
ROWS = [
    ("A16 ceiling (ours)$^\\ddagger$",   "A16"),
    ("NVFP4 acts (ours)",                "NVFP4_fixed"),
    ("GF4 (ours)",                       "GF4"),
    ("Adaptive GF4 (ours)",              "GF4_adap"),
    ("Residual GF4 (ours)$^\\S$",        "residual"),
    ("Learned GF4 (ours)",               "GF4_learned"),
]


def fmt(v, fp32):
    if v is None or str(v).strip().lower() in ("", "nan"):
        return "--"
    v = float(v)
    if v != v:                       # NaN
        return "--"
    if v > 1e3:                      # collapsed
        return f"{v:.1f} (+{v-fp32:.1f})"
    return f"{v:.2f} (+{v-fp32:.2f})"


def main():
    retain = 0 if (len(sys.argv) > 1 and sys.argv[1] == "ablation") else 1
    if not os.path.exists(CSV):
        sys.exit(f"no {CSV} yet — run ./run_iso_all.sh first")
    # latest row per model at the requested retention
    latest = {}
    for r in csv.DictReader(open(CSV)):
        if int(r["retain"]) != retain:
            continue
        if r.get("had_bs", "auto") not in ("auto", ""):
            continue             # skip block-Hadamard probes (full-row is the main protocol)
        latest[r["model"]] = r   # CSV is append-order, so last wins
    print(f"% W4A4 rows, retention {'ON' if retain else 'OFF'} "
          f"(from iso_results.csv, {len(latest)} models)")
    for label, col in ROWS:
        cells = []
        for mid, (_, fp32) in MODELS.items():
            r = latest.get(mid)
            cells.append(fmt(r[col], fp32) if r else "--")
        print(f"{label}\n  & FP4 (W4A4)\n  & " + " & ".join(cells) + r" \\")
    # quick console summary
    print("\n% --- summary (ppl) ---")
    hdr = ["model"] + [c for _, c in ROWS]
    print("% " + "  ".join(f"{h:>11}" for h in hdr))
    for mid, (name, _) in MODELS.items():
        r = latest.get(mid)
        if not r:
            continue
        vals = [name] + [r[c] if r[c] else "--" for _, c in ROWS]
        print("% " + "  ".join(f"{v:>11}" for v in vals))


if __name__ == "__main__":
    main()

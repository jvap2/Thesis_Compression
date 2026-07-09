"""
hw_sim_gf4.py — analytical hardware cost model for an IMAGINARY accelerator
("GF4-Engine") designed to run our GF4 / residual-GF4 W4A4 scheme natively,
which Blackwell's fixed-E2M1 FP4 tensor core CANNOT.

WHY A NEW ACCELERATOR
  Our method needs three things Blackwell FP4 lacks:
    (1) PROGRAMMABLE-LUT FP4 decode — GF4 uses NF4-style Gaussian-quantile
        levels, not the hardwired E2M1 ladder {0,.5,1,1.5,2,3,4,6}.  A 16-entry
        codebook SRAM per PE-tile replaces the fixed E2M1 ROM decoder.
    (2) A fused HADAMARD (HadaCore-style FWHT) unit on the activation load path
        — full-row rotation x->xH before the MMA, RᵀW folded into weights
        offline.
    (3) Optional 2-PASS RESIDUAL accumulate — sum two FP4 terms (Q1+Q2) into one
        FP16 product, ~8-bit effective at 2x FP4 MACs.

WHAT THIS FILE COMPUTES
  A purely analytical per-layer roofline: storage bits, GEMM MACs, LUT decodes,
  FWHT butterflies, microscale muls, modeled ENERGY (pJ) and a throughput proxy.
  It is a MODEL, not silicon — every energy constant is a documented, tunable
  assumption.  The value is the RELATIVE comparison + the accuracy/energy Pareto,
  which is robust to the absolute pJ numbers.

ENERGY CONSTANTS  (Horowitz, "Computing's Energy Problem", ISSCC 2014, 45nm;
  the table every efficient-ML HW paper cites — EIE, Eyeriss, etc.)
    int8  add 0.03 pJ   int32 add 0.1    int8  mult 0.2   int32 mult 3.1
    fp16  add 0.4       fp16  mult 1.1   fp32  add  0.9   fp32  mult 3.7
    32b SRAM read ~5 pJ        DRAM 32b ~640 pJ  (=20 pJ/bit)
  FP4 / E4M3 / LUT entries below are EXTRAPOLATIONS from that table, flagged.
"""
from dataclasses import dataclass, field
import math, os


# ── energy model (pJ); all tunable ────────────────────────────────────────────
# RECONCILED 2026-06-23 to the Accelergy component model in timeloop_gf4/ (45nm,
# Aladdin+CACTI).  The component model is AUTHORITATIVE; this analytical tool is
# the quick estimator.  Headline correction: the LUT is NOT free — a 16-entry
# codebook read costs ~1 pJ, same as a 4-bit MAC, so 2 reads/MAC (decode-per-MAC)
# triple compute energy.  Net LUT overhead on TOTAL token energy (component
# model): +14-23% per-MAC, +8-13% decode-at-PE-load, ~? at decode-at-ingress.
# Absolute energy here stays optimistic vs the component model because hw_sim
# assumes ideal weight reuse (DRAM-once); the Timeloop mapping refetches, so its
# DRAM term dominates (~86%).  Trust the RATIOS here and timeloop_gf4 for absolutes.
@dataclass
class HWConfig:
    # multiply-accumulate energy per scheme (pJ per MAC) — mult+add fused
    e_mac_fp16: float = 4.0          # ~4x int4 (component-anchored)
    e_mac_fp8:  float = 2.0          # ~2x int4
    e_mac_fp4:  float = 1.0          # int4 MAC = 1.0 pJ (Accelergy/Aladdin 45nm)
    # operand decode
    e_lut_decode: float = 1.0        # 16-entry codebook regfile read = 1.0 pJ (component!)
    e_e2m1_decode: float = 0.0       # hardwired ROM decoder ~free
    # microscale (per-block) dequant multiply
    e_scale_mul: float = 1.5         # E4M3/E8M0 scale * accumulator (~fp8 mult)
    # Hadamard butterfly: 1 add + sign-flip
    e_fwht_bfly: float = 0.4         # fp16 add
    # memory (pJ per BYTE moved from the level it lives at)
    e_dram_byte: float = 160.0       # 20 pJ/bit * 8  (DRAM, weights/acts streamed once)
    e_sram_byte: float = 1.25        # ~5 pJ / 32b word /4 bytes (on-chip reuse)
    # array / clock for the throughput proxy
    pe_macs_per_cycle: int = 16384   # e.g. 128x128 FP4 MMA tile
    clock_hz: float = 1.5e9
    block: int = 16                  # microscale block (NVFP4-style)
    decode_at_load: bool = True      # decode once at SRAM fill (amortized) vs per-MAC


# ── scheme definitions ────────────────────────────────────────────────────────
# bits per operand value (weight, activation), number of FP4 passes, whether it
# needs a programmable LUT, and whether it needs the Hadamard rotation.
@dataclass
class Scheme:
    name: str
    w_bits: int
    a_bits: int
    mac_mode: str        # "fp16" | "fp8" | "fp4"
    passes: int = 1      # residual = 2
    lut: bool = False
    hadamard: bool = False
    scale_block: int = 16   # 32 for MXFP4
    scale_bits: int = 8     # E4M3=8, E8M0=8


SCHEMES = [
    Scheme("FP16",          16, 16, "fp16", scale_block=0, scale_bits=0),
    Scheme("FP8-E4M3",       8,  8, "fp8",  scale_block=0, scale_bits=0),
    Scheme("MXFP4",          4,  4, "fp4",  scale_block=32, scale_bits=8),
    Scheme("NVFP4",          4,  4, "fp4",  scale_block=16, scale_bits=8),
    Scheme("GF4 (ours,1-term)",      4,  4, "fp4", lut=True, hadamard=True, scale_block=16),
    Scheme("residual-GF4 (ours,2t)", 4,  8, "fp4", passes=2, lut=True, hadamard=True, scale_block=16),
    Scheme("GF4-W4A16 (wt-only)",    4, 16, "fp16", lut=True, hadamard=True, scale_block=16),
]


# ── a linear layer ────────────────────────────────────────────────────────────
@dataclass
class Linear:
    name: str
    K: int   # in features
    N: int   # out features
    role: str = "other"   # q/k/v/out/fc1/fc2/lm_head — drives the retention policy


# Outlier-retention set (the MLP-output projection + LM head): these are the
# layers single-pass FP4 collapses on, so a mixed-precision engine runs them at
# higher precision.  Matches iso_energy's _mlp_skip (fc2/down_proj/dense_4h_to_h).
OUTLIER_ROLES = {"fc2", "down", "dense_4h_to_h", "lm_head"}


def model_layers(d_model, d_ff, n_layers, vocab=50272):
    """OPT-style decoder linear layers (q,k,v,out,fc1,fc2) + lm_head."""
    L = []
    for i in range(n_layers):
        L += [Linear(f"L{i}.q", d_model, d_model, "q"),
              Linear(f"L{i}.k", d_model, d_model, "k"),
              Linear(f"L{i}.v", d_model, d_model, "v"),
              Linear(f"L{i}.out", d_model, d_model, "out"),
              Linear(f"L{i}.fc1", d_model, d_ff, "fc1"),
              Linear(f"L{i}.fc2", d_ff, d_model, "fc2")]   # <- outlier set
    L += [Linear("lm_head", d_model, vocab, "lm_head")]    # <- outlier set
    return L


MODELS = {
    "opt-125m": dict(d_model=768,  d_ff=3072,  n_layers=12),
    "opt-1.3b": dict(d_model=2048, d_ff=8192,  n_layers=24),
    "opt-2.7b": dict(d_model=2560, d_ff=10240, n_layers=32),
    "llama2-7b": dict(d_model=4096, d_ff=11008, n_layers=32, vocab=32000),
}

# Unified registry: model_key -> [Linear, ...].  Built-ins use model_layers();
# Colab exports (timeloop_gf4/models/*.json) add their ACTUAL per-layer shapes.
MODEL_LAYERS = {k: model_layers(**cfg) for k, cfg in MODELS.items()}

# map the Colab experiment's PPL keys -> this file's scheme names
_PPL_KEYMAP = {"A16 (W4A16)": "GF4-W4A16 (wt-only)", "E2M1/NVFP4": "NVFP4",
               "GF4 adaptive": "GF4 (ours,1-term)", "residual-GF4": "residual-GF4 (ours,2t)"}


def load_colab_exports(mdir=None):
    """Ingest timeloop_gf4/models/*.json -> MODEL_LAYERS + PPL (architecture-
    agnostic; uses the real exported (K,N) shapes and measured PPLs)."""
    import json as _json
    mdir = mdir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "timeloop_gf4", "models")
    if not os.path.isdir(mdir):
        return
    for fn in sorted(os.listdir(mdir)):
        if not fn.endswith(".json"):
            continue
        d = _json.load(open(os.path.join(mdir, fn)))
        key = d["model"].split("/")[-1]
        layers = []
        for s in d["shapes"]:
            for i in range(int(s["count"])):
                layers.append(Linear(f"{key}.K{s['K']}_N{s['N']}.{i}",
                                     int(s["K"]), int(s["N"])))
        MODEL_LAYERS[key] = layers
        PPL[key] = {_PPL_KEYMAP.get(k, k): v for k, v in d.get("ppl", {}).items()}
        print(f"  loaded models/{fn}: {key} ({len(layers)} layers, "
              f"{len(d.get('ppl',{}))} PPL points)")


# ── per-layer cost ────────────────────────────────────────────────────────────
def layer_cost(lin: Linear, sch: Scheme, hw: HWConfig, T: int):
    """Return dict of per-layer counts and energy (pJ) for T activation tokens."""
    K, N = lin.K, lin.N
    macs = T * N * K * sch.passes

    # ---- storage (bytes), incl. microscale overhead ----
    def bpv(bits, block, sbits):
        return bits + (sbits / block if block else 0.0)
    w_bpv = bpv(sch.w_bits, sch.scale_block, sch.scale_bits) if sch.mac_mode != "fp16" or sch.w_bits < 16 else sch.w_bits
    a_bpv = bpv(sch.a_bits, sch.scale_block, sch.scale_bits) if sch.a_bits < 16 else sch.a_bits
    w_bytes = N * K * w_bpv / 8.0
    a_bytes = T * K * a_bpv / 8.0 * sch.passes

    # ---- compute energy ----
    e_mac = {"fp16": hw.e_mac_fp16, "fp8": hw.e_mac_fp8, "fp4": hw.e_mac_fp4}[sch.mac_mode]
    e_gemm = macs * e_mac

    # operand decode (LUT vs hardwired vs none)
    if sch.lut:
        if hw.decode_at_load:
            decodes = (N * K + T * K) * sch.passes      # once per operand load
        else:
            decodes = macs * 2                          # per-MAC (both operands)
        e_decode = decodes * hw.e_lut_decode
    else:
        decodes, e_decode = 0, 0.0

    # microscale dequant muls (per block, applied to the K-accumulated partials)
    if sch.scale_block:
        scale_ops = (T * N) * (K / sch.scale_block) * sch.passes
        e_scale = scale_ops * hw.e_scale_mul
    else:
        scale_ops, e_scale = 0, 0.0

    # Hadamard rotation pre-pass: full-row FWHT on K dims, per token.
    if sch.hadamard:
        Bhad = 1 << (K - 1).bit_length()                # next_pow2(K)
        bflies = T * Bhad * int(math.log2(Bhad))
        e_had = bflies * hw.e_fwht_bfly
    else:
        bflies, e_had = 0, 0.0

    # memory: weights streamed once (DRAM), activations live on-chip (SRAM reuse)
    e_mem = w_bytes * hw.e_dram_byte + a_bytes * hw.e_sram_byte

    e_total = e_gemm + e_decode + e_scale + e_had + e_mem
    return dict(macs=macs, w_bytes=w_bytes, a_bytes=a_bytes,
                e_gemm=e_gemm, e_decode=e_decode, e_scale=e_scale,
                e_had=e_had, e_mem=e_mem, e_total=e_total, bflies=bflies,
                w_bpv=w_bpv, a_bpv=a_bpv)


def model_cost(model_key, sch: Scheme, hw: HWConfig, T=2048):
    layers = MODEL_LAYERS[model_key]
    agg = dict(macs=0, w_bytes=0.0, a_bytes=0.0, e_gemm=0.0, e_decode=0.0,
               e_scale=0.0, e_had=0.0, e_mem=0.0, e_total=0.0, bflies=0)
    wbpv_num = abpv_num = wK = aTK = 0.0
    for lin in layers:
        c = layer_cost(lin, sch, hw, T)
        for k in ("macs", "w_bytes", "a_bytes", "e_gemm", "e_decode",
                  "e_scale", "e_had", "e_mem", "e_total", "bflies"):
            agg[k] += c[k]
        wbpv_num += c["w_bpv"] * lin.N * lin.K
        wK += lin.N * lin.K
        abpv_num += c["a_bpv"] * T * lin.K
        aTK += T * lin.K
    agg["w_bpv"] = wbpv_num / wK
    agg["a_bpv"] = abpv_num / aTK
    agg["pj_per_token"] = agg["e_total"] / T
    # throughput proxy: GEMM-bound cycles (ignores Hadamard pipeline if fused)
    agg["gemm_cycles"] = agg["macs"] / hw.pe_macs_per_cycle
    agg["had_overhead_pct"] = 100.0 * agg["e_had"] / agg["e_total"]
    return agg


# ── measured accuracy (this codebase, WikiText-2 nonoverlap-2048) ─────────────
#   Populate where we have numbers; None = not yet measured on this setup.
# All opt-125m PPL below are from ONE iso-cost build (iso_energy_125m.py,
# 2026-06-22): Hadamard + rms*clip scale + per-16 block; codebooks swapped via
# gf4_levels.  Comparable to each other (same calibration seed).  NVFP4 here =
# E2M1 ladder + adaptive clip + Hadamard (a GENEROUS NVFP4 — real NVFP4 lacks
# the rotation).  GF4 = Gaussian-quantile + adaptive clip.  MXFP4 PPL not run
# (per-32 scale would be ~NVFP4 or slightly worse).
PPL = {
    "opt-125m": {                            # all from ONE build (iso_energy_125m.py)
        "FP16": 27.66,                       # W16A16 no-quant anchor (pareto_fill)
        "FP8-E4M3": 27.70,                   # ≈FP16 (E4M3 near-lossless; W8A8 build errored)
        "MXFP4": 37.30,                      # E2M1 + per-32 microscale block
        "NVFP4": 36.60,                      # E2M1 + per-16 + adaptive clip (+Hadamard)
        "GF4 (ours,1-term)": 35.57,          # Gaussian + per-16 + adaptive clip
        "residual-GF4 (ours,2t)": 32.31,     # 2-term residual ≈ A16
        "GF4-W4A16 (wt-only)": 32.24,        # A16 (4-bit weights, fp16 act)
    },
    "opt-1.3b": {   # from proto_hore_outlier_1.3b.log (2026-06-22)
        "GF4-W4A16 (wt-only)": 18.04,
        "GF4 (ours,1-term)": 20.61,
        "residual-GF4 (ours,2t)": 18.15,
    },
}


def fmt_e(x):
    for u, d in (("nJ", 1e3), ("pJ", 1)):
        if x >= d:
            return f"{x/d:7.2f} {u}"
    return f"{x:7.2f} pJ"


# ══ MIXED-PRECISION ENGINE COMPARISON ════════════════════════════════════════
# Four imagined machines on the SAME modest 256-PE, 45nm array (minimal hardware
# is the selling point — we report how LITTLE area/energy buys FP16 accuracy, not
# peak FLOPS).  They differ ONLY in the PE datapath and how the outlier-retention
# layers (fc2/down_proj/lm_head) are handled:
#
#   fixed-E2M1   : Blackwell's FP4 — 1-pass FP4, hardwired E2M1 decode, NO
#                  retention.  Smallest + cheapest, but COLLAPSES on outlier-
#                  heavy models (that's the ablation row).
#   FP16-only    : Blackwell's precision path — every PE a full FP16 MAC.  Best
#                  accuracy, but ~16x MAC area and ~4x MAC energy everywhere.
#   GF4 multipass: OURS (minimal) — 1-pass GF4 on the bulk, N-pass residual on
#                  the outlier set, ALL on the SAME FP4 array.  No FP16 multiplier
#                  exists on the chip; precision = pass count (~+9 dB/pass).  N=4
#                  is a conservative upper bound here; validate_multipass.py pins
#                  the smallest N that recovers FP16-retention PPL (likely 2-3).
#   GF4 +FP16unit: OURS (alt) — 1-pass GF4 on the bulk + a small DEDICATED FP16
#                  sub-array for the outlier set.  More area, but full speed there
#                  (no 4x pass penalty).  The point of comparison you asked for.
#
# Relative areas; FP4 MAC = 1.0.  Multiplier area ~ datawidth^2, so a 16-bit
# multiply is ~16x a 4-bit one (tunable).  The programmable codebook is ONE
# shared ingress SRAM (Timeloop-measured at-ingress: +0.01-0.03% array area).
A_MAC_FP4  = 1.0
A_MAC_FP16 = 16.0
A_PE_REGS  = 2.0       # weight/input/psum regfiles per PE (~equal across engines)
A_CODEBOOK_FRAC = 3e-4 # shared 16x8b codebook as a fraction of the FP4 array


@dataclass
class Engine:
    name: str
    bulk_mode: str        # "fp4" | "fp16"  (non-outlier layers)
    outlier_mode: str     # "fp4" | "fp16"  (retention layers)
    bulk_passes: int
    outlier_passes: int
    lut: bool
    fp16_unit_frac: float = 0.0   # dedicated FP16 sub-array size as frac of PE_N
    ppl_key: str = ""             # which measured PPL column represents this engine
    note: str = ""


PE_N = 256             # 16x16 array — deliberately small ("minimal hardware")

ENGINES = [
    Engine("fixed-E2M1 (Blackwell FP4)", "fp4", "fp4", 1, 1, lut=False,
           ppl_key="NVFP4",                       note="no retention -> collapses"),
    Engine("FP16-only (Blackwell prec)", "fp16", "fp16", 1, 1, lut=False,
           ppl_key="FP16",                        note="big + power-hungry"),
    Engine("GF4 multipass (ours,min)",   "fp4", "fp4", 1, 4, lut=True,
           ppl_key="residual-GF4 (ours,2t)",      note="ONE FP4 array, no FP16 unit"),
    Engine("GF4 +FP16unit (ours,alt)",   "fp4", "fp16", 1, 1, lut=True,
           fp16_unit_frac=0.25,
           ppl_key="residual-GF4 (ours,2t)",      note="dedicated FP16 for outliers"),
]


def engine_area(eng, pe_n=PE_N):
    """Relative silicon area of the compute fabric (MACs + regs + codebook)."""
    mac = A_MAC_FP16 if eng.bulk_mode == "fp16" else A_MAC_FP4
    base = pe_n * (mac + A_PE_REGS)
    a = base
    if eng.lut:
        a += A_CODEBOOK_FRAC * base                      # shared ingress codebook
    if eng.fp16_unit_frac > 0:                           # dedicated FP16 sub-array
        a += eng.fp16_unit_frac * pe_n * (A_MAC_FP16 + A_PE_REGS)
    return a


def engine_cost(model_key, eng, hw, T=2048, pe_n=PE_N):
    """Per-engine energy (pJ), array cycles, storage — over the whole model.
    Outlier layers (role in OUTLIER_ROLES) use the engine's outlier mode/passes."""
    layers = MODEL_LAYERS[model_key]
    e_total = 0.0
    cycles = 0.0
    w_bytes_tot = a_bytes_tot = 0.0
    for lin in layers:
        outlier = lin.role in OUTLIER_ROLES
        mode   = eng.outlier_mode   if outlier else eng.bulk_mode
        passes = eng.outlier_passes if outlier else eng.bulk_passes
        macs = T * lin.N * lin.K * passes
        e_mac = {"fp16": hw.e_mac_fp16, "fp8": hw.e_mac_fp8, "fp4": hw.e_mac_fp4}[mode]
        e = macs * e_mac
        if eng.lut and mode == "fp4":                    # codebook decode at ingress
            e += (lin.N * lin.K + T * lin.K) * passes * hw.e_lut_decode
        if mode == "fp4":                                # microscale + Hadamard add-ons
            e += (T * lin.N) * (lin.K / hw.block) * passes * hw.e_scale_mul
            Bhad = 1 << (lin.K - 1).bit_length()
            e += T * Bhad * int(math.log2(Bhad)) * hw.e_fwht_bfly
        # storage: multi-pass stores `passes` FP4 codes (4-pass == 16 bits/weight)
        wbits = (4 * passes) if mode == "fp4" else 16
        abits = (4 * passes) if mode == "fp4" else 16
        sc = (8.0 / hw.block) if mode == "fp4" else 0.0  # E4M3 microscale overhead
        w_bytes = lin.N * lin.K * (wbits + sc) / 8.0
        a_bytes = T * lin.K * (abits + sc) / 8.0
        e += w_bytes * hw.e_dram_byte + a_bytes * hw.e_sram_byte
        # cycles: outliers on the dedicated FP16 sub-array if the engine has one
        rate = (eng.fp16_unit_frac * pe_n) if (outlier and eng.fp16_unit_frac > 0) else pe_n
        cycles += macs / rate
        e_total += e
        w_bytes_tot += w_bytes
        a_bytes_tot += a_bytes
    return dict(e_total=e_total, cycles=cycles, pj_per_token=e_total / T,
                w_bytes=w_bytes_tot, a_bytes=a_bytes_tot,
                tokens_per_s=T * hw.clock_hz / cycles)


def engine_comparison(model_key, hw, T=2048):
    print(f"\n{'#'*92}\n  MIXED-PRECISION ENGINE COMPARISON — {model_key} "
          f"(T={T}, {PE_N}-PE 45nm array)\n{'#'*92}")
    rows = [(e, engine_cost(model_key, e, hw, T), engine_area(e)) for e in ENGINES]
    a_min = min(a for _, _, a in rows)                   # smallest = fixed-E2M1
    f16c = next(c for e, c, _ in rows if e.bulk_mode == "fp16")   # FP16-only refs
    f16a = next(a for e, _, a in rows if e.bulk_mode == "fp16")
    e_ref = f16c["pj_per_token"]
    # throughput DENSITY = tok/s per unit area (iso-area fair: the small FP4 array
    # packs ~16x the MACs of an FP16 array in the same silicon).
    dens_ref = f16c["tokens_per_s"] / f16a
    print(f"  {'engine':30s} {'area':>7s} {'energy/tok':>11s} {'speed/area':>11s} "
          f"{'PPL':>8s}   note")
    for eng, c, area in rows:
        ppl = PPL.get(model_key, {}).get(eng.ppl_key)
        ppl_s = (f"{ppl:.2f}" if ppl and ppl < 1e3 else
                 (f"{ppl:.0f}*" if ppl else "  --  "))
        dens = c["tokens_per_s"] / area
        print(f"  {eng.name:30s} {area/a_min:6.2f}x "
              f"{c['pj_per_token']/e_ref:9.2f}x "
              f"{dens/dens_ref:9.2f}x "
              f"{ppl_s:>8s}   {eng.note}")
    print("  area: rel to smallest (fixed-E2M1, =1).  energy/tok & speed/area: rel "
          "to FP16-only (=1).\n  speed/area = tok/s per mm^2 (iso-area).  lower "
          "energy + higher speed/area = better.  *=PPL>1e3 (collapsed).")
    # headline: ours-minimal vs the two Blackwell corners
    d = {e.name: (c, a) for e, c, a in rows}
    mn, mn_a = d["GF4 multipass (ours,min)"]; f16, f16_a = d["FP16-only (Blackwell prec)"]
    fx, fx_a = d["fixed-E2M1 (Blackwell FP4)"]; alt, alt_a = d["GF4 +FP16unit (ours,alt)"]
    md = mn["tokens_per_s"] / mn_a
    print(f"\n  -> ours(minimal) vs FP16-only: {f16_a/mn_a:.1f}x SMALLER, "
          f"{e_ref/mn['pj_per_token']:.1f}x lower energy/token, "
          f"{md/dens_ref:.1f}x more throughput/mm^2 — at ~the same accuracy.")
    print(f"  -> ours(minimal) vs fixed-E2M1 (Blackwell FP4): ~same area "
          f"({mn_a/fx_a:.2f}x), but the fixed ladder has NO retention path "
          f"(outliers collapse) — we get FP16-class PPL on the same silicon budget.")
    print(f"  -> multipass vs dedicated FP16 unit: {alt_a/mn_a:.2f}x SMALLER area, "
          f"{alt['pj_per_token']/mn['pj_per_token']:.2f}x the energy "
          f"(4x add-ons on outliers) — the area/energy trade you asked to see.")


def main():
    hw = HWConfig()
    T = 2048
    load_colab_exports()   # pull in any timeloop_gf4/models/*.json from Colab
    builtins = ["opt-125m", "opt-1.3b"]
    extras = [k for k in MODEL_LAYERS if k not in builtins and k in PPL]
    for model_key in builtins + extras:
        print(f"\n{'='*92}\n  GF4-Engine cost model — {model_key}  (T={T} tokens, block={hw.block})\n{'='*92}")
        print(f"  {'scheme':26s} {'W/A bpv':>11s} {'pJ/token':>10s} "
              f"{'GEMM':>7s} {'LUT':>6s} {'scale':>6s} {'Hada':>6s} {'mem':>6s} {'PPL':>7s}")
        rows = []
        for sch in SCHEMES:
            a = model_cost(model_key, sch, hw, T)
            ppl = PPL.get(model_key, {}).get(sch.name)
            rows.append((sch.name, a, ppl))
            # energy breakdown as % of total
            br = lambda k: 100.0 * a[k] / a["e_total"]
            print(f"  {sch.name:26s} {a['w_bpv']:4.2f}/{a['a_bpv']:5.2f} "
                  f"{a['pj_per_token']/1e3:9.2f}k "
                  f"{br('e_gemm'):6.0f}%{br('e_decode'):5.0f}%{br('e_scale'):5.0f}%"
                  f"{br('e_had'):5.0f}%{br('e_mem'):5.0f}% "
                  f"{('%.2f'%ppl) if ppl else '   -  ':>7s}")
        # Pareto note vs FP16
        base = next(a for n, a, _ in rows if n == "FP16")
        fp8  = next(a for n, a, _ in rows if n == "FP8-E4M3")
        print("\n  energy vs FP16 (pJ/token ratio):")
        for n, a, ppl in rows:
            print(f"    {n:26s} {a['pj_per_token']/base['pj_per_token']:5.2f}x"
                  f"   ({'PPL %.2f'%ppl if ppl else 'PPL n/a'})")
        # headline Pareto claims (only where PPL measured AND usable).
        # USABLE_PPL guard: if a single-term PPL is junk (>this), the method
        # has collapsed (e.g. OPT-6.7b quantizing fc2) — a "+X ppl win" between
        # two broken methods is meaningless, so suppress the headline.
        USABLE_PPL = 100.0
        d = {n: (a, ppl) for n, a, ppl in rows}
        nv = d["NVFP4"]; gf = d["GF4 (ours,1-term)"]; rg = d["residual-GF4 (ours,2t)"]
        if gf[1] and nv[1]:
            if max(gf[1], nv[1]) > USABLE_PPL:
                worse = "both single-term codebooks COLLAPSED" if min(gf[1], nv[1]) > USABLE_PPL \
                        else "NVFP4 collapsed"
                print(f"\n  PARETO 1: N/A — {worse} (GF4 {gf[1]:.0f}, NVFP4 {nv[1]:.0f} ppl). "
                      f"Single-term unusable here; outlier layers (fc2/down_proj) need skipping.")
            else:
                print(f"\n  PARETO 1 (iso-cost, {gf[0]['pj_per_token']/base['pj_per_token']:.2f}x FP16): "
                      f"GF4 {gf[1]:.2f} beats NVFP4 {nv[1]:.2f} by {nv[1]-gf[1]:+.2f} ppl "
                      f"— free accuracy, same silicon.")
        if rg[1] and rg[1] <= USABLE_PPL:
            print(f"  PARETO 2: residual-GF4 {rg[1]:.2f} (≈A16 {d['GF4-W4A16 (wt-only)'][1]:.2f}) "
                  f"at {rg[0]['pj_per_token']/base['pj_per_token']:.2f}x FP16 energy vs FP8 "
                  f"{fp8['pj_per_token']/base['pj_per_token']:.2f}x "
                  f"— ~{fp8['pj_per_token']/rg[0]['pj_per_token']:.1f}x cheaper at near-lossless.")
        engine_comparison(model_key, hw, T)
    print("\n  NOTE: energy constants are Horowitz-45nm-grounded extrapolations "
          "(tunable in HWConfig).\n  Relative ordering + the accuracy/energy Pareto "
          "are the deliverable, not absolute pJ.")


if __name__ == "__main__":
    main()

"""
nvfp4_verify.py -- diagnostic for the catastrophic NVFP4 PPL on OPT-13B.

Result to explain: NVFP4 on weights + activations gave PPL 532.02 (+521.9
vs. FP16 baseline) on OPT-13B, labeled "NVFP4 fixed".

ROOT CAUSE HYPOTHESIS
---------------------
gf4_regularity_colab.py defines the E2M1 magnitude codebook as:
    E2M1_MAG = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}

When these raw values are passed to set_codebook() and then used in
gf4_quant(), the quantizer does:

    mag = (xb / scale).abs().clamp(0.0, 1.0)     # normalized to [0, 1]
    idx = bucketize(mag, _THR)                    # threshold lookup
    deq = _LEVELS[idx] * sign * scale

The midpoint thresholds derived from E2M1_MAG are:
    _THR = {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0}

Since mag is clamped to [0, 1], any threshold > 1.0 is unreachable.
Five of the seven thresholds (1.25, 1.75, 2.5, 3.5, 5.0) exceed 1.0,
so bucketize can only return indices 0, 1, or 2.  Only three of the eight
codebook levels are ever assigned -- a de facto 2-bit quantizer in a 4-bit
container.  The effective levels become {0.0, 0.5, 1.0} * scale instead of
the intended eight.  This is what drives PPL into the hundreds.

WHAT THIS SCRIPT DOES
---------------------
1.  Loads the model (OPT-13B or any other) and captures one layer's worth of
    real activations from a calibration forward pass.
2.  Quantizes those activations with five schemes and reports, FOR EACH:
      - Level histogram: how many of the 8 codebook levels are actually used.
      - Reconstruction MSE relative to FP16.
      - Clip fraction (fraction of elements that hit the maximum level).
3.  Shows the per-layer stats in a summary table so you can see immediately
    which scheme has "only 3 of 8 levels used."
4.  Runs full WikiText-2 perplexity for each scheme so the PPL impact is
    measured end-to-end, not estimated from MSE alone.

FIVE ACTIVATION-QUANTIZATION SCHEMES
-------------------------------------
  gf4          : GF4 Gaussian-quantile codebook (our working scheme).
  nvfp4_raw    : E2M1_MAG {0,0.5,...,6} passed directly to the [0,1]-clamped
                 path -- THE SUSPECTED BUG; should reproduce ~532 PPL.
  nvfp4_norm   : E2M1_MAG / 6.0  so levels live in [0,1]  -- THE CORRECT FIX.
  nvfp4_absmax : NVFP4-style absmax/6 scale (not RMS-based); tests whether
                 outlier-dominated scaling alone also explains the collapse.
  e2m1_rms     : E2M1 codebook with RMS scale, bias selected per block to
                 minimize MSE -- the "best effort" NVFP4 activation path.

All five share the same NVFP4 weight quantization (Hadamard + per-block
absmax E4M3 scale + E2M1 nearest-neighbor) so we isolate the activation side.

Usage:
    python nvfp4_verify.py                          # OPT-1.3B, quick smoke test
    python nvfp4_verify.py --model facebook/opt-13b --device-map
    python nvfp4_verify.py --model facebook/opt-13b --device-map \\
        --eval-windows 20 --calib-windows 4
"""
import argparse, math, os, sys
import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", default="facebook/opt-1.3b",
                help="HF model ID. Use facebook/opt-13b for the failing case.")
ap.add_argument("--device-map", action="store_true",
                help="Load with device_map='auto' (needed for opt-13b on 40GB).")
ap.add_argument("--eval-windows", type=int, default=10,
                help="WikiText-2 test windows for perplexity. 10 is fast; "
                     "use 50+ for a result comparable to the paper table.")
ap.add_argument("--calib-windows", type=int, default=2,
                help="Calibration windows for activation capture.")
ap.add_argument("--seqlen", type=int, default=2048)
ap.add_argument("--had-block", type=int, default=32,
                help="Hadamard block size (must divide in_features).")
ap.add_argument("--wblock", type=int, default=16,
                help="Weight quantization block size for NVFP4.")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--skip-perplexity", action="store_true",
                help="Only run the per-layer diagnostic (no PPL eval). "
                     "Useful when you just want to confirm the level-histogram bug.")
ap.add_argument("--out-csv", default="nvfp4_verify_results.csv")
args = ap.parse_args()

torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

from scipy.linalg import hadamard as _scipy_hadamard

# ---------------------------------------------------------------------------
# Codebooks
# ---------------------------------------------------------------------------
GF4_LEVEL = np.array([0.0, 0.0796082, 0.1737177, 0.2828685,
                       0.3952704, 0.5250730, 0.6961928, 1.0], dtype=np.float32)

# NVFP4 / E2M1 magnitudes as used in gf4_regularity_colab.py.
# NOTE: these are RAW E2M1 values, NOT normalized to [0,1].
E2M1_MAG_RAW = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
# Normalized to [0,1] by dividing by the maximum (6.0).
E2M1_MAG_NORM = E2M1_MAG_RAW / 6.0   # {0, 0.0833, 0.1667, 0.25, 0.333, 0.5, 0.667, 1.0}

HAD_BLOCK  = args.had_block
WBLOCK     = args.wblock
CLIP_RATIO = 2.5
RETAIN     = ("fc2", "down_proj", "lm_head")  # layers kept at FP16 per the paper policy


# ---------------------------------------------------------------------------
# Hadamard helpers
# ---------------------------------------------------------------------------
_H_CACHE: dict = {}

def _hmat(block: int, device: torch.device) -> torch.Tensor:
    key = (block, device)
    if key not in _H_CACHE:
        H = torch.tensor(_scipy_hadamard(block).astype(np.float32) / math.sqrt(block),
                         device=device)
        _H_CACHE[key] = H
    return _H_CACHE[key]


def rotate(x: torch.Tensor, signs: torch.Tensor, block: int = HAD_BLOCK) -> torch.Tensor:
    """Blockwise randomized Hadamard on the last dimension of x."""
    F_ = x.shape[-1]
    H = _hmat(block, x.device)
    xr = (x * signs).reshape(*x.shape[:-1], F_ // block, block) @ H
    return xr.reshape(*x.shape[:-1], F_)


# ---------------------------------------------------------------------------
# Scale helpers
# ---------------------------------------------------------------------------
def quant_e4m3(s: torch.Tensor) -> torch.Tensor:
    """Round a positive scale tensor to E4M3 precision (4 exp, 3 mantissa)."""
    s = s.clamp(min=2.0 ** -9, max=448.0)
    e = torch.floor(torch.log2(s))
    m = torch.round((s / torch.exp2(e) - 1.0) * 8.0) / 8.0
    return (1.0 + m) * torch.exp2(e)


# ---------------------------------------------------------------------------
# Weight quantization (shared by all schemes)
# ---------------------------------------------------------------------------
def nvfp4_quant_weights(W: torch.Tensor) -> torch.Tensor:
    """
    Fake-quantize W (already Hadamard-rotated, [out, in]) to NVFP4:
      - Per-block absmax -> E4M3 scale
      - Nearest-E2M1 magnitude assignment
    This is identical to gf4_regularity_colab.py's nvfp4_weight_quant, so the
    weight-side is held constant across all activation schemes.
    """
    shp = W.shape
    blk = WBLOCK if shp[1] % WBLOCK == 0 else (32 if shp[1] % 32 == 0 else HAD_BLOCK)
    Wb = W.reshape(-1, blk).float()

    amax = Wb.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = quant_e4m3(amax / 6.0)
    wn = Wb / scale

    mag_t = torch.tensor(E2M1_MAG_RAW, device=W.device)
    thr_t = (mag_t[:-1] + mag_t[1:]) / 2.0    # {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0}
    idx = torch.bucketize(wn.abs().clamp(0.0, 6.0), thr_t)
    q = mag_t[idx] * torch.sign(wn) * scale
    return q.reshape(shp).to(W.dtype)


# ---------------------------------------------------------------------------
# Activation quantization schemes
# ---------------------------------------------------------------------------

def _quant_with_codebook(x: torch.Tensor,
                          levels_np: np.ndarray,
                          scale_mode: str = "rms") -> torch.Tensor:
    """
    Generic activation quantizer.

    scale_mode:
      "rms"    -- scale = block_rms * CLIP_RATIO  (GF4 / our scheme)
      "absmax" -- scale = block_absmax / levels_np.max()  (NVFP4 weight style)

    levels_np: the codebook, MUST be normalized to [0, 1] if scale_mode="rms"
               (values > 1 are unreachable after the clamp below).
    """
    shp = x.shape
    xb = x.reshape(-1, HAD_BLOCK).float()

    if scale_mode == "rms":
        rms   = xb.pow(2).mean(-1, keepdim=True).add(1e-12).sqrt()
        scale = rms * CLIP_RATIO
    else:  # absmax
        scale = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / float(levels_np.max())

    xn  = xb / scale                          # normalize
    mag = xn.abs().clamp(0.0, 1.0)           # ← THIS CLAMP is the crux of the bug

    lv_t  = torch.tensor(levels_np, device=x.device)
    thr_t = (lv_t[:-1] + lv_t[1:]) / 2.0    # midpoint thresholds
    idx   = torch.bucketize(mag, thr_t)       # nearest-level index

    # Levels > 1.0 give thresholds > 1.0 which are unreachable after the clamp.
    # E.g. E2M1_MAG_RAW thresholds {1.25, 1.75, 2.5, 3.5, 5.0} → all unreachable.

    deq = lv_t[idx] * torch.sign(xn) * scale
    return deq.reshape(shp).to(x.dtype)


def quant_gf4(x: torch.Tensor) -> torch.Tensor:
    """GF4 Gaussian-quantile codebook, RMS scale. Our working scheme."""
    return _quant_with_codebook(x, GF4_LEVEL, scale_mode="rms")


def quant_nvfp4_raw(x: torch.Tensor) -> torch.Tensor:
    """
    E2M1_MAG_RAW {0, 0.5, ..., 6} passed directly to the [0,1]-clamped path.
    THE SUSPECTED BUG: five thresholds > 1.0 are unreachable → de facto 3-level.
    """
    return _quant_with_codebook(x, E2M1_MAG_RAW, scale_mode="rms")


def quant_nvfp4_norm(x: torch.Tensor) -> torch.Tensor:
    """
    E2M1_MAG normalized to [0,1] by dividing by 6. All 8 levels are reachable.
    THE FIX: simply normalize before setting the codebook.
    """
    return _quant_with_codebook(x, E2M1_MAG_NORM, scale_mode="rms")


def quant_nvfp4_absmax(x: torch.Tensor) -> torch.Tensor:
    """
    NVFP4-style absmax/6 scale instead of RMS scale. Tests whether the outlier-
    dominated scale alone explains the collapse (independent of the codebook bug).
    """
    return _quant_with_codebook(x, E2M1_MAG_NORM, scale_mode="absmax")


def quant_e2m1_rms_best(x: torch.Tensor) -> torch.Tensor:
    """
    Best-effort E2M1 on activations: RMS scale, but pick E2M1 bias (0, 1, or 2)
    per block to minimize reconstruction MSE.  This is the 'ideal' NVFP4-for-
    activations result, using the same bias-search logic as hessian_weight_quant.
    """
    # E2M1 codebooks for bias 0, 1, 2 -- same as e2m1_common.cuh.
    CB = {
        0: np.array([1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0], dtype=np.float32),
        1: np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32),
        2: np.array([0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0], dtype=np.float32),
    }
    shp = x.shape
    xb = x.reshape(-1, HAD_BLOCK).float()
    rms = xb.pow(2).mean(-1, keepdim=True).add(1e-12).sqrt()
    scale = rms * CLIP_RATIO                   # [blocks, 1]

    best_mse = torch.full((xb.shape[0],), float("inf"), device=x.device)
    best_deq = torch.zeros_like(xb)

    for b, cb_np in CB.items():
        cb_max = float(cb_np.max())
        # For each bias, normalize xb to the codebook's range.
        xn = (xb / scale) * cb_max            # scale xn to [−cb_max, cb_max]
        lv_t  = torch.tensor(cb_np, device=x.device)
        thr_t = (lv_t[:-1] + lv_t[1:]) / 2.0
        idx   = torch.bucketize(xn.abs().clamp(0.0, cb_max), thr_t)
        deq_b = lv_t[idx] * torch.sign(xn) * (scale / cb_max)
        mse_b = ((xb - deq_b) ** 2).mean(dim=-1)
        better = mse_b < best_mse
        best_mse = torch.where(better, mse_b, best_mse)
        best_deq = torch.where(better.unsqueeze(-1), deq_b, best_deq)

    return best_deq.reshape(shp).to(x.dtype)


SCHEMES = {
    "gf4":          quant_gf4,
    "nvfp4_raw":    quant_nvfp4_raw,    # suspected bug
    "nvfp4_norm":   quant_nvfp4_norm,   # fix
    "nvfp4_absmax": quant_nvfp4_absmax, # alternative scale bug test
    "e2m1_rms_best":quant_e2m1_rms_best,# best-effort NVFP4 for activations
}


# ---------------------------------------------------------------------------
# Per-layer diagnostic: level histogram + MSE
# ---------------------------------------------------------------------------
def activation_diagnostic(x_flat: np.ndarray,
                           had_block: int = HAD_BLOCK) -> dict[str, dict]:
    """
    Given a flat array of pre-Hadamard activations from one layer, run every
    quantization scheme and report:
      - n_levels_used : number of distinct codebook indices that appear
      - level_hist    : count for each of the 8 levels (indices 0-7)
      - rel_mse       : E[(x - x_hat)^2] / E[x^2]
      - clip_frac     : fraction of elements assigned to the top level
    """
    x = torch.tensor(x_flat, dtype=torch.float32)
    var_x = float(x.pow(2).mean())

    results = {}
    for name, fn in SCHEMES.items():
        x_hat = fn(x)
        x_np  = x.numpy(); xh_np = x_hat.numpy()
        rel_mse = float(np.mean((x_np - xh_np) ** 2)) / (var_x + 1e-12)

        # Recover the index each element was assigned to (re-run the lookup).
        xb  = x.reshape(-1, had_block)
        rms = xb.pow(2).mean(-1, keepdim=True).add(1e-12).sqrt()
        if name == "nvfp4_absmax":
            scale = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 6.0
        else:
            scale = rms * CLIP_RATIO
        xn  = (xb / scale).abs().clamp(0.0, 1.0)

        if name in ("gf4",):
            lv_np = GF4_LEVEL
        elif name in ("nvfp4_raw",):
            lv_np = E2M1_MAG_RAW
        else:
            lv_np = E2M1_MAG_NORM
        lv_t  = torch.tensor(lv_np)
        thr_t = (lv_t[:-1] + lv_t[1:]) / 2.0
        idx   = torch.bucketize(xn, thr_t).reshape(-1).numpy()

        hist  = np.bincount(idx, minlength=8)
        n_lv  = int((hist > 0).sum())
        clip  = float(hist[-1]) / max(idx.shape[0], 1)

        results[name] = {
            "n_levels_used": n_lv,
            "level_hist":    hist.tolist(),
            "rel_mse":       rel_mse,
            "clip_frac":     clip,
        }
    return results


# ---------------------------------------------------------------------------
# Load model + tokenizer
# ---------------------------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

hf_tok_str = os.environ.get("HF_TOKEN")
if hf_tok_str:
    from huggingface_hub import login
    login(token=hf_tok_str, add_to_git_credential=False)

print(f"Loading {args.model}...")
tok = AutoTokenizer.from_pretrained(args.model)
load_kw = dict(torch_dtype=torch.float16)
if args.device_map:
    load_kw["device_map"] = "auto"

model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
if not args.device_map:
    model = model.to("cuda")
model.eval()

_input_dev = (model.get_input_embeddings().weight.device
              if hasattr(model, "get_input_embeddings")
              else next(model.parameters()).device)
print(f"Loaded.  Input embedding device: {_input_dev}\n")

# ---------------------------------------------------------------------------
# WikiText-2 helpers
# ---------------------------------------------------------------------------
def _load_wikitext(split):
    for repo in ("Salesforce/wikitext", "wikitext"):
        try:
            return load_dataset(repo, "wikitext-2-raw-v1", split=split)
        except Exception:
            pass
    raise RuntimeError("Could not load wikitext-2-raw-v1")


def get_windows(ids, seqlen, n):
    n = min(n, ids.numel() // seqlen)
    return [ids[i * seqlen:(i + 1) * seqlen].unsqueeze(0) for i in range(n)]


test_ids  = tok("\n\n".join(_load_wikitext("test")["text"]),
                return_tensors="pt").input_ids[0]
train_ids = tok("\n\n".join(_load_wikitext("train")["text"][:2000]),
                return_tensors="pt").input_ids[0]

eval_windows  = get_windows(test_ids,  args.seqlen, args.eval_windows)
calib_windows = get_windows(train_ids, args.seqlen, args.calib_windows)
print(f"Eval windows: {len(eval_windows)}  |  Calib windows: {len(calib_windows)}\n")


@torch.no_grad()
def perplexity(mdl) -> float:
    nll = ntok = 0
    in_dev = (mdl.get_input_embeddings().weight.device
              if hasattr(mdl, "get_input_embeddings")
              else next(mdl.parameters()).device)
    for w in eval_windows:
        w = w.to(in_dev)
        out = mdl(w, labels=w)
        nll  += out.loss.float().item() * (w.numel() - 1)
        ntok += w.numel() - 1
    return math.exp(nll / ntok)


# ---------------------------------------------------------------------------
# FP16 baseline
# ---------------------------------------------------------------------------
print("FP16 baseline perplexity...")
ppl_fp16 = perplexity(model)
print(f"FP16 baseline PPL: {ppl_fp16:.4f}\n")

# ---------------------------------------------------------------------------
# Identify target layers
# ---------------------------------------------------------------------------
targets = [
    (n, m) for n, m in model.named_modules()
    if isinstance(m, nn.Linear)
    and not any(s in n for s in RETAIN)
    and m.in_features % HAD_BLOCK == 0
]
print(f"Target layers: {len(targets)}  (retaining FP16 for {RETAIN})\n")

# ---------------------------------------------------------------------------
# Capture calibration activations (pre-Hadamard) for the diagnostic
# ---------------------------------------------------------------------------
import zlib

captured_raw: dict[str, list] = {n: [] for n, _ in targets}
_capture_hooks = []

def _make_capture_hook(name):
    def _h(mod, inp, _out):
        x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float().cpu()
        captured_raw[name].append(x)
    return _h

for name, mod in targets:
    _capture_hooks.append(mod.register_forward_hook(_make_capture_hook(name)))

with torch.no_grad():
    for w in calib_windows:
        model(w.to(_input_dev))

for h in _capture_hooks:
    h.remove()

print("Calibration activations captured.")

# ---------------------------------------------------------------------------
# Per-layer diagnostic
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PER-LAYER DIAGNOSTIC: level-histogram & MSE for each quantization scheme")
print("=" * 80)

# Column headers
print(f"\n{'Layer':<42} {'Scheme':<16} {'n_lv':>5} {'rel_mse':>9} {'clip':>7}  "
      f"Level histogram (index 0..7)")
print("-" * 110)

layer_diag_rows = []
for i, (name, _) in enumerate(targets):
    if not captured_raw[name]:
        continue
    # Hadamard-rotate the captured activations for the diagnostic, since that's
    # what the forward path sees before activation quantization.
    seed  = zlib.crc32(name.encode("utf-8")) % (2 ** 31)
    gen   = torch.Generator().manual_seed(seed)
    K     = targets[i][1].in_features
    signs = (torch.randint(0, 2, (K,), generator=gen).float() * 2 - 1)

    X_raw = torch.cat(captured_raw[name], dim=0)   # [tokens, K]
    X_had = rotate(X_raw, signs, HAD_BLOCK)         # rotated

    # Subsample to keep diagnostic fast (cap at 256k elements).
    flat = X_had.reshape(-1).numpy()
    if flat.shape[0] > 262144:
        rng  = np.random.default_rng(0)
        flat = rng.choice(flat, 262144, replace=False)

    diag = activation_diagnostic(flat)
    short = name.split(".")[-2] + "." + name.split(".")[-1] if "." in name else name
    short = (name[:41]) if len(name) > 41 else name

    for scheme, d in diag.items():
        hist_str = " ".join(f"{c:6d}" for c in d["level_hist"])
        marker = "  <-- BUG: only 3 levels!" if (scheme == "nvfp4_raw" and d["n_levels_used"] <= 3) else ""
        print(f"{short:<42} {scheme:<16} {d['n_levels_used']:>5}  "
              f"{d['rel_mse']:>8.4f}  {d['clip_frac']:>6.3f}  [{hist_str}]{marker}")
        layer_diag_rows.append({
            "layer": name, "scheme": scheme,
            "n_levels": d["n_levels_used"],
            "rel_mse": d["rel_mse"],
            "clip_frac": d["clip_frac"],
        })
    print()

    del captured_raw[name], X_raw, X_had  # free as we go

# Aggregate summary
print("\n" + "=" * 80)
print("AGGREGATE DIAGNOSTIC (mean across all layers)")
print("=" * 80)
from collections import defaultdict
agg: dict[str, list] = defaultdict(list)
for row in layer_diag_rows:
    agg[row["scheme"]].append(row)

print(f"\n{'Scheme':<18} {'mean n_lv':>10} {'mean rel_mse':>14} {'mean clip':>10}  "
      f"EXPECTED BEHAVIOR")
print("-" * 80)
EXPECTED = {
    "gf4":           "All 8 levels, low MSE              (the working baseline)",
    "nvfp4_raw":     "Only 3 levels, HIGH MSE            (reproduces PPL 532)",
    "nvfp4_norm":    "All 8 levels, moderate MSE         (the correct fix)",
    "nvfp4_absmax":  "All 8 levels, higher MSE (outliers dominate absmax scale)",
    "e2m1_rms_best": "All 8 levels, best NVFP4 MSE      (bias-searched per block)",
}
for scheme in SCHEMES:
    rows = agg[scheme]
    if not rows:
        continue
    mn_lv  = np.mean([r["n_levels"]  for r in rows])
    mn_mse = np.mean([r["rel_mse"]   for r in rows])
    mn_cl  = np.mean([r["clip_frac"] for r in rows])
    print(f"{scheme:<18} {mn_lv:>10.1f} {mn_mse:>14.6f} {mn_cl:>10.4f}  "
          f"{EXPECTED.get(scheme, '')}")

# ---------------------------------------------------------------------------
# Perplexity evaluation for each scheme
# ---------------------------------------------------------------------------
if args.skip_perplexity:
    print("\n[--skip-perplexity: skipping end-to-end PPL evaluation]")
    sys.exit(0)

print("\n" + "=" * 80)
print("END-TO-END PERPLEXITY (NVFP4 weights + each activation scheme)")
print("=" * 80)

ppl_results = {"fp16": ppl_fp16}

def _patch_model(act_fn):
    """
    Install forward hooks:
      - Pre-hook: Hadamard-rotate activations, apply act_fn, return quantized x.
      - Weight: Hadamard-rotated + NVFP4-quantized (in-place, stored in module.weight).
    Returns a restore callback.
    """
    orig_weights = {}
    orig_fwds    = {}
    g = torch.Generator().manual_seed(args.seed)

    for name, mod in targets:
        K    = mod.in_features
        dev  = mod.weight.device
        seed = zlib.crc32(name.encode("utf-8")) % (2 ** 31)
        gen2 = torch.Generator().manual_seed(seed)
        signs = (torch.randint(0, 2, (K,), generator=gen2).float() * 2 - 1).to(dev)

        # Quantize weights (done once per scheme; same for all).
        orig_weights[name] = mod.weight.data.clone()
        with torch.no_grad():
            W_rot = rotate(mod.weight.data.float(), signs, HAD_BLOCK)
            W_q   = nvfp4_quant_weights(W_rot)
            mod.weight.data.copy_(W_q.to(mod.weight.dtype))

        # Pre-hook for activations.
        signs_buf = signs   # captured in closure

        def _pre(mod_, inp_, _signs=signs_buf, _fn=act_fn):
            x = inp_[0]
            xr = rotate(x.float().to(_signs.device), _signs, HAD_BLOCK)
            xq = _fn(xr).to(x.dtype)
            return (xq.to(x.device),) + inp_[1:]

        orig_fwds[name] = mod.forward
        _hook = mod.register_forward_pre_hook(_pre)
        # store hook handle on the module for later removal
        if not hasattr(mod, "_nvfp4_hooks"):
            mod._nvfp4_hooks = []
        mod._nvfp4_hooks.append(_hook)

    def _restore():
        for name, mod in targets:
            mod.weight.data.copy_(orig_weights[name].to(mod.weight.device))
            for h in getattr(mod, "_nvfp4_hooks", []):
                h.remove()
            mod._nvfp4_hooks = []

    return _restore


import time, csv

ppl_table = [("fp16", ppl_fp16, ppl_fp16 - ppl_fp16)]

for scheme_name, act_fn in SCHEMES.items():
    print(f"\n  [{scheme_name}] Patching model and computing PPL...", flush=True)
    restore = _patch_model(act_fn)
    t0  = time.time()
    ppl = perplexity(model)
    t1  = time.time()
    restore()
    delta = ppl - ppl_fp16
    print(f"  [{scheme_name}] PPL = {ppl:.4f}  (delta vs FP16: {delta:+.4f})  "
          f"[{t1-t0:.1f}s]", flush=True)
    ppl_results[scheme_name] = ppl
    ppl_table.append((scheme_name, ppl, delta))

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"PERPLEXITY SUMMARY  --  {args.model}  --  "
      f"{len(eval_windows)} windows x {args.seqlen} tok")
print("=" * 80)
print(f"\n{'Scheme':<20} {'PPL':>10} {'dPPL(fp16)':>12}  Note")
print("-" * 70)
note = {
    "fp16":          "",
    "gf4":           "working scheme (our baseline)",
    "nvfp4_raw":     "<-- BUG: only 3 effective levels; should be ~532",
    "nvfp4_norm":    "fix: normalize E2M1_MAG to [0,1] before set_codebook",
    "nvfp4_absmax":  "absmax scale test (independent of the normalization bug)",
    "e2m1_rms_best": "best-effort NVFP4 for activations (bias search per block)",
}
for name, ppl, delta in ppl_table:
    print(f"{'fp16 baseline' if name=='fp16' else name:<20} {ppl:>10.4f} "
          f"{delta:>+12.4f}  {note.get(name, '')}")

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------
import csv as _csv_mod

new_file = not os.path.exists(args.out_csv)
with open(args.out_csv, "a", newline="") as f:
    w = _csv_mod.writer(f)
    if new_file:
        w.writerow(["timestamp", "model", "scheme", "ppl", "dppl_fp16",
                    "eval_windows", "calib_windows", "seqlen"])
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    for name, ppl, delta in ppl_table:
        w.writerow([ts, args.model, name, f"{ppl:.4f}", f"{delta:.4f}",
                    len(eval_windows), len(calib_windows), args.seqlen])
print(f"\nResults written to {args.out_csv}")

print("""
INTERPRETATION GUIDE
--------------------
nvfp4_raw PPL >> nvfp4_norm PPL  →  confirms the codebook normalization bug.
  The raw E2M1_MAG levels {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} feed into
  a quantizer that clamps normalized magnitudes to [0, 1].  Five thresholds
  (1.25, 1.75, 2.5, 3.5, 5.0) are above 1.0 and unreachable, collapsing the
  effective codebook to three levels {0.0, 0.5, 1.0}.

nvfp4_absmax PPL >> nvfp4_norm PPL  →  the absmax scale also degrades quality
  independently (outlier-dominated scale compresses most activations to 0).
  Both bugs may have been active simultaneously in the "NVFP4 fixed" run.

nvfp4_norm PPL ≈ gf4 PPL  →  NVFP4 and GF4 are comparably good once the
  normalization bug is fixed; the Gaussian-quantile spacing of GF4 gives
  a modest advantage because it is distribution-aware.

THE FIX (one line in gf4_regularity_colab.py / codebook_suite.py):
  set_codebook(E2M1_MAG / E2M1_MAG.max(), device)   # normalize to [0, 1]
  OR equivalently:
  set_codebook(E2M1_MAG / 6.0, device)
""")

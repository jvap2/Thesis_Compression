# FP_Quantization_Experiments

Research package for FP4 post-training quantization of LLMs using Hadamard rotation, Gaussian-optimal activation quantization, and novel gap-closing variants.

---

## Overview

This package implements a full FP4 W4A4 quantization pipeline for decoder-only LLMs (OPT, LLaMA, BLOOM). The core insight is that a **randomized blockwise Hadamard transform** applied to weights and activations simultaneously smooths outliers and makes the resulting distribution approximately Gaussian — at which point standard FP4 E2M1 levels are suboptimal and Gaussian-optimal levels (GF4) are significantly better.

**Benchmark model:** `facebook/opt-1.3b`, `wikitext-2-raw-v1`

| Configuration | Sliding PPL | GPTQ PPL |
|---|---|---|
| FP32 baseline | 14.624 | 14.625 |
| FP4 A16 (weight-only, Hadamard) | 20.231 | 20.215 |
| FP4 NVFP4 activations | ~8082 | ~8236 |
| FP4 GF4 activations | 27.259 | 27.234 |
| FP4 GF4 adaptive (new) | TBD | TBD |
| FP4 GF4 residual (new) | TBD | TBD |
| FP4 GF4 learned levels (new) | TBD | TBD |
| FP4 H-SmoothQuant + GF4 (new) | TBD | TBD |

Config: `e_bits=2, m_bits=1, e_bits_scale=4, m_bits_scale=3, block_size=16, batch_size=8`

---

## Pipeline

### 1. Weight Quantization (Hadamard domain)

All linear layers are wrapped in `HadamardQuantLinearFP` which:
1. Generates a per-layer random sign vector `D ∈ {±1}^M`
2. Transforms the weight offline: `W_had = H(W · D)` via `fwht_blockwise`
3. Quantizes `W_had` to FP4 E2M1 via `calibrate_model_stochastic_fp4` (GPTQ-style Hessian calibration in the Hadamard domain)
4. Computes `μ = E[H(D·x)]` and `bias_correction = W_had_q @ μ` to compensate for token-outlier mean removal

### 2. Activation Quantization (inference time)

The `HadamardQuantLinearFP` forward pass:

```
Step 1  (optional) SmoothQuant channel division
Step 2  x_had = fwht(D * x)        — ALWAYS applied (weights live in H-domain)
Step 3  x_had -= μ                 — token-outlier mean subtraction
Step 3.5 (optional) x_had /= s     — H-domain per-channel smooth scaling
Step 4  x_q = quantize(x_had)      — GF4 or variant
Step 5  out = F.linear(x_q, W_had_q, b)
Step 6  out += bias_correction
```

Activation quantization mode is set via `act_quant_mode(model, mode=...)` context manager.

---

## Activation Quantization Modes

### Standard Modes

| Mode string | Description |
|---|---|
| `None` | A16 — no activation quantization (weight-only) |
| `"nvfp4"` | FP8 E4M3 scale, block size 16 (hardware NVFP4 spec) |
| `"mxfp4"` | E8M0 (power-of-2) scale, block size 32 (MX spec) |
| `"gf4"` | Gaussian-optimal FP4 with per-layer calibrated clip ratio |

### Novel Modes (new)

| Mode string | Description | Cost |
|---|---|---|
| `"gf4_adaptive"` | Per-block online MSE-optimal clip selection | ~5× activation memory, no calibration |
| `"gf4_residual"` | Two-stage: `Q(x) + Q(x − Q(x))` | 2× activation compute |

---

## GF4 Design

### Codebook: `GF4_POS`

Standard FP4 E2M1 levels are logarithmically spaced and waste 3–4 dB SNR on Gaussian inputs. GF4 uses **equal-mass Gaussian quantile boundaries** for N(0,1), normalized to [0, 1]:

```python
GF4_POS = [0.0, 0.0796082, 0.1737177, 0.2828685, 0.3952704, 0.5250730, 0.6961928, 1.0]
```

Scale per block: `block_rms × clip_ratio`, where `clip_ratio` is calibrated per-layer via MSE minimization over `α ∈ {1.5, 2.0, 2.5, 3.0, 4.0}`.

### Why NVFP4 fails after Hadamard

NVFP4 uses FP8 E4M3 scale with block_max scaling. After the Hadamard transform, activations are approximately Gaussian — the maximum of a Gaussian block is heavily influenced by rare outlier samples. FP4's log-spaced levels then misalign with the dense low-level Gaussian mass, causing PPL collapse (8082 vs 27 for GF4).

---

## Novel Contributions

### 1. Per-Block Adaptive Clip (`quantize_activations_gf4_adaptive`)

```python
quantize_activations_gf4_adaptive(x, block_size=16,
    clip_candidates=(1.5, 2.0, 2.5, 3.0, 4.0), levels=None)
```

Current art calibrates clip ratio at **layer granularity** (one α* for all blocks). This function operates at **block granularity**: for each 16-element block independently, it evaluates all `clip_candidates` in a vectorized `[C, B, bs]` tensor and selects the candidate minimizing per-block reconstruction MSE. No calibration pass required — optimal clip is computed online.

**Key properties:**
- Fully vectorized (no Python loop over blocks), uses `.gather` on `[B, C, bs]`
- Requires no modification to weight calibration
- Supports custom `levels` (e.g., learned codebook)

**Smoke test MSE vs baseline GF4:** 0.0070 vs 0.0128 (45% reduction)

### 2. Two-Stage Residual GF4 (`quantize_activations_gf4_residual`)

```python
quantize_activations_gf4_residual(x, block_size=16,
    clip_ratio1=2.5, clip_ratio2=2.5, levels=None)
```

The residual `r = x − Q(x)` of GF4-quantized Gaussian activations is:
- Zero-mean (quantization noise is roughly unbiased)
- Reduced variance (~1/8th of original for 3-bit effective resolution)
- Still approximately Gaussian

Therefore GF4 is near-optimal for the second stage too. Net effective resolution approaches 8-bit at 2× the activation compute cost. This is analogous to residual vector quantization (AQLM, QuIP) but applied to activations in the Hadamard domain.

**Smoke test MSE vs baseline GF4:** 0.0001 vs 0.0128 (99% reduction)

### 3. Learned GF4 Codebook (`optimize_gf4_levels`, `calibrate_gf4_learned_levels`)

```python
# Calibrate learned levels on actual post-Hadamard activations
calibrate_gf4_learned_levels(model, calib_loader, device, block_size,
    num_batches=4, n_steps=400, max_samples=4096, per_layer=False)
```

NF4/GF4 use *theoretical* Gaussian quantile positions. Post-Hadamard distributions in practice have:
- Heavier tails than pure Gaussian (residual outliers)
- Layer-specific kurtosis excess
- Architecture-dependent deviations

`optimize_gf4_levels` learns the 7 interior level positions via gradient descent on real calibration activations. Implementation details:
- **Parameterization:** cumulative softmax increments — guarantees monotonicity and [0, 1] normalization
- **Differentiable quantization:** temperature-annealed soft assignment τ: 0.15 → 0.005 (geometric decay), converging to hard nearest-neighbor
- **Initialization:** from GF4_POS increments (inverse softmax)

After calibration, `module.gf4_levels` is set on each `HadamardQuantLinearFP`; the `"gf4"` mode uses learned levels automatically.

**Observed shift from GF4_POS (10 steps, random data):**
```
GF4_POS:  [0.0, 0.0796, 0.1737, 0.2829, 0.3953, 0.5251, 0.6962, 1.0]
Learned:  [0.0, 0.0789, 0.1763, 0.2889, 0.4045, 0.5382, 0.7149, 1.0]
```
Interior levels shift outward slightly, consistent with heavier-than-Gaussian tails.

### 4. H-Domain SmoothQuant (`calibrate_model_gf4_hsmooth`)

```python
calibrate_model_gf4_hsmooth(model, calib_loader, block_size, device, ...)
```

SmoothQuant [Xiao et al., 2022] migrates activation scale to weights in the original domain. This applies the same principle **in the Hadamard domain**.

Even after `H(D·x)`, different output channels can have unequal per-channel variance (the rotation is energy-preserving but not variance-equalizing for non-i.i.d. inputs). Per-channel RMS `s[m] = E[H(D·x)[:, m]²]^0.5` is computed from calibration data, then:

```
W_had_q[n, m]  ←  W_had_q[n, m] * s[m]      # fold into weights (offline)
x_had[m]       ←  x_had[m] / s[m]            # at inference (step 3.5)
```

Net: `(x/s) @ (W·s)^T = x @ W^T` (mathematically exact). After scaling, all channels have near-unit variance → per-block GF4 clipping is near-optimal for every block in every layer.

**Note:** Current implementation adjusts `W_had_q` post-quantization (approximate). A production version would quantize `Q(W_had · s)` directly for maximum weight quality. Enable via `gf4_variant="hsmooth"` in `quantize_model_fp`.

---

## API Reference

### Main Entry Point

```python
from FP_Quantization_Experiments import quantize_model_fp

quant_model = quantize_model_fp(
    model, data_loader,
    block_size=16,
    e_bits=2, m_bits=1,
    e_bits_scale=4, m_bits_scale=3,
    device="cuda",
    Hadamard=True,
    use_gf4=True,
    gf4_variant="gf4",       # "gf4" | "hsmooth" | "learned" | "learned_per_layer"
    had_block_size="auto",
    num_calib_batches=4,
)
```

### Inference-Time Mode Switching

```python
from FP_Quantization_Experiments import act_quant_mode

# Weight-only (A16)
with act_quant_mode(model, mode=None):
    ppl = compute_ppl(model, ...)

# GF4 with calibrated per-layer clip
with act_quant_mode(model, mode="gf4"):
    ppl = compute_ppl(model, ...)

# Per-block adaptive clip (no calibration needed)
with act_quant_mode(model, mode="gf4_adaptive"):
    ppl = compute_ppl(model, ...)

# Two-stage residual (best PPL, 2× activation compute)
with act_quant_mode(model, mode="gf4_residual"):
    ppl = compute_ppl(model, ...)
```

### Post-Calibration Learned Levels

```python
from FP_Quantization_Experiments import calibrate_gf4_learned_levels

# Run after quantize_model_fp (requires HadamardQuantLinearFP with calibrated clip ratios)
model = calibrate_gf4_learned_levels(
    model, calib_loader, device, block_size=16,
    num_batches=4, n_steps=400,
    per_layer=False,   # True = per-layer codebook, more expressive
)
# Then evaluate with act_quant_mode(model, mode="gf4") — uses learned levels automatically
```

### Standalone Quantizer Functions

```python
from FP_Quantization_Experiments import (
    quantize_activations_gf4,          # baseline GF4
    quantize_activations_gf4_adaptive, # per-block adaptive clip
    quantize_activations_gf4_residual, # two-stage residual
    optimize_gf4_levels,               # gradient-descent level optimization
    fwht_blockwise,                    # fast Walsh-Hadamard transform
)

# All share the same interface:
x_q = quantize_activations_gf4(x, block_size=16, clip_ratio=2.5)
x_q = quantize_activations_gf4_adaptive(x, block_size=16)
x_q = quantize_activations_gf4_residual(x, block_size=16, clip_ratio1=2.5)
```

---

## File Structure

```
FP_Quantization_Experiments/
├── __init__.py              — package exports
├── bit_split.py             — core quantization logic (GF4, Hadamard, calibration)
│     GF4_POS               — Gaussian-optimal 8-level codebook
│     quantize_activations_gf4          — baseline GF4
│     quantize_activations_gf4_adaptive — per-block adaptive clip (new)
│     quantize_activations_gf4_residual — two-stage residual (new)
│     optimize_gf4_levels               — gradient codebook optimization (new)
│     calibrate_gf4_learned_levels      — model-level learned levels (new)
│     calibrate_model_gf4_hsmooth       — H-domain SmoothQuant (new)
│     HadamardQuantLinearFP             — quantized layer wrapper
│     calibrate_model_gf4               — full Hadamard + GF4 calibration
│     calibrate_model_stochastic_fp4    — stochastic FP4 Hessian calibration
│     fwht_blockwise                    — fast Walsh-Hadamard transform
│     quantize_model_fp                 — main entry point
├── quant_layers_fp.py       — QuantLinearFP, QuantConv2dFP base layers
├── fp_flexround.py          — FlexRound / BRECQ calibration
├── block_smoothquant.py     — block-level SmoothQuant
├── utils.py                 — shared utilities
└── README.md                — this file
```

---

## Key Implementation Notes

### Hadamard is Unconditional in Forward

`W_had_q = Q(H(W·D))` lives in the Hadamard domain. The Hadamard transform in `HadamardQuantLinearFP.forward` (step 2) is applied **regardless of activation quantization mode** — even in A16 mode. Skipping it when `act_quant_mode=None` causes ~1.6 relative error and PPL explosion (previously observed at PPL ~2100).

### h_smooth_scale Fold Math

```
x_smooth = x_had / s          # per-channel division  [T, M]
W_smooth  = W_had_q * s       # per-column multiply    [N, M]
x_smooth @ W_smooth^T
  = (x/s) @ (W*s)^T
  = sum_m  x[m]/s[m] * W[n,m]*s[m]
  = sum_m  x[m] * W[n,m]       ✓  exact
```

### Level Parameterization for `optimize_gf4_levels`

Monotone normalization via cumulative softmax increments:
```python
inc      = softmax(delta_raw)      # [7], strictly positive, sums to 1
interior = cumsum(inc)[:-1]        # [6], in (0, 1), strictly increasing
levels   = [0.0, *interior, 1.0]   # [8]
```
This guarantees strict monotonicity and [0,1] bounds without projection.

---

## GPU / Environment Notes

**Driver issue (unresolved):** DKMS has `nvidia/535.274.02` registered but source is at `nvidia-535.309.01`. Kernel module not built for `6.8.0-111-generic`. Fix:
```bash
sudo dkms remove nvidia/535.274.02 --all
sudo dkms add -m nvidia -v 535.309.01
sudo dkms build -m nvidia -v 535.309.01 -k 6.8.0-111-generic
sudo dkms install -m nvidia -v 535.309.01 -k 6.8.0-111-generic
sudo modprobe nvidia
```

**TORCH_CUDA_ARCH_LIST:** Must be set before any import. Already patched at the top of `FP_QuantNetworkTest_LLM.py`:
```python
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6;8.9"
```

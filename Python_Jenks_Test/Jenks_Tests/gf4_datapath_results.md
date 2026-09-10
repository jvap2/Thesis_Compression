# GF4 single-4-bit-datapath results (faithful harness)

Method: NVFP4 (E2M1) weights via Hadamard + Hessian reconstruction (`quantize_model_fp`,
`Hadamard=True, use_gf4=True, block_size=16, e_bits=2/m_bits=1, e_scale=4/m_scale=3, had_block_size="auto"`);
GF4 activations. Outlier layer = `fc2` (OPT) / `down_proj` (Llama); `lm_head` kept FP16.

Profiles (WikiText-2 PPL, non-overlapping 2048):
- **retain** — outliers FP16 (weights + acts), bulk acts 2-pass GF4  → **two datapaths** (the baseline to beat)
- **A16-all** — every layer NVFP4, all acts FP16  → weight-quant ceiling for a single datapath
- **out-Np** — every layer NVFP4, bulk acts 2-pass GF4, outlier acts N-pass GF4  → **single 4-bit datapath**

`retain` keeps outlier *weights* FP16; `A16-all`/`out-*` quantize them. So:
`A16-all − retain` = cost of quantizing the outlier **weights**; `out-Np − A16-all` = cost of the outlier **activation** datapath.

### FAITHFUL harness (gf4_datapath_colab.ipynb / real quantize_model_fp) — TRUSTWORTHY
| model | source | fp16 | retain (2dp) | A16-all | out-1p | out-2p | out-4p | out-8p | min |
|---|---|---|---|---|---|---|---|---|---|
| OPT-125m | local (3-sample, noisy) | 29.37 | 33.93 | 34.52 | 35.26 | 34.54 | 34.50 | 34.49 | — |
| OPT-1.3b | local (40-sample) | 14.469 | 16.614 | 16.601 | 16.876 | 16.677 | 16.689 | 16.701 | ~130 |

**Faithful OPT-1.3b confirms the thesis:** single 4-bit datapath (out-2p 16.677) == two-datapath retain
(16.614) within +0.06; outlier WEIGHT quant free (A16-all 16.601 ~= retain); outlier ACTS need 2 passes
(1-pass +0.26, 2/4/8 flat). vs the SELF-CONTAINED notebook's 1.3b COLLAPSE (retain 23.69) => that collapse
was a broken reimpl, NOT a real property of OPT-1.3b. Use the faithful notebook.

### SELF-CONTAINED notebook (gf4_multipass_colab.ipynb) — reimplementation, NOT faithful on activations
GPTQ weights are fine (A16 ≈ fp16), but the reimplemented GF4 activation quant lacks the harness's outlier
handling (post-Hadamard mean subtraction, adaptive clip) → it CATASTROPHICALLY COLLAPSES on outlier-heavy
OPT-1.3b (retain +9.2 vs the faithful harness's +2.1). Survives on 2.7b/6.7b by luck. Do not cite these.
Column order here matches the self-contained notebook: fp16 | A16 | retain | all-1p | out-2p | out-4p | out-8p.
| model | source | fp16 | A16 | retain | all-1p | out-2p | out-4p | out-8p | min |
|---|---|---|---|---|---|---|---|---|---|
| OPT-1.3b | Colab (self-contained) | 14.469 | 14.605 | **23.685 (COLLAPSE)** | 25.042 | 23.713 | 23.712 | 23.751 | 5.7 |
| OPT-2.7b | Colab (self-contained) | 12.357 | 12.445 | 12.448 | 12.739 | 12.457 | 12.456 | 12.455 | 9.0 |
| OPT-6.7b | Colab (self-contained) | 10.674 | 10.689 | 10.693 | 10.967 | 10.697 | 10.701 | 10.698 | 15.3 |

## Reading

- **Activation datapath is replaceable:** `out-1p` is always worse (2.7b +0.28, 6.7b +0.27 over out-2p);
  `out-2p ≈ out-4p ≈ out-8p ≈ A16-all`. So **2-pass GF4 activations on the outlier layers match FP16
  activations** — the FP16 *activation* path is unnecessary. 1-pass is not enough; >2 passes add nothing.
- **Weight-quant cost (A16-all − retain):** 2.7b +0.003, 6.7b +0.004 — i.e. quantizing the outlier weights
  to NVFP4 is essentially free at ≥2.7B. Combined with (1), the full single 4-bit datapath (`out-2p`) lands
  within ~+0.01 of retain at 2.7B/6.7B → **the second (FP16) datapath is eliminable at scale.**
- **Scale:** the whole W4A4 gap vs fp16 shrinks fast — retain−fp16 = +2.14 (1.3b local) → +0.09 (2.7b) →
  +0.015 (6.7b). The 1.3b→2.7b jump is large; OPT-1.3b Colab cross-check pending to confirm the pipeline
  matches the local +2.1 (vs a possible clone/local code divergence).

## Caveats

- OPT-125m row used only 3 eval windows (±0.3 noise) — ordering solid, absolutes rough.
- To confirm the Hessian reconstruction ran (vs a rounding fallback), the per-layer log should show
  `=== Stochastic FP4 calibration ===` / `Standard v5 quantization` / `Computing post-Hadamard mean` /
  `sanity rel_err ... A4≈0.5`.

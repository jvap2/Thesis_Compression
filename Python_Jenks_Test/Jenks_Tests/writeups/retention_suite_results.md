# Retention-elimination suite results

Notebook: `multipass_vs_retention_llama3_opt67.ipynb` (full 8-model suite).
Setting: randomized-Hadamard rotation, NVFP4 weights (E2M1 + per-16 E4M3, RTN),
GF4 activations; W4A4 on all transformer linears; `lm_head`/embeddings FP16 both
conditions. Only the MLP down-projection switches: FP16 retention vs. K-pass
residual (activation = GF4 `gf4_npass`; weight residual = NVFP4 `quant_nvfp4`).
Protocol: WikiText-2, 2048-token non-overlapping, `NSAMP=40` windows unless noted.
`dPPL` = increase over that model's FP16-retention baseline (negative = beats it).
`~Wbits` = 4·(1+ksel_eff) naive down-proj weight cost.

Reading: **flat concentration curve** (top-k% rows hold ≈k% of the 1st-pass
weight error) ⇒ rotation already equalized the weight, so a *full* weight
residual (not selective) is what ties retention — confirms the Corollary in
`retention_unnecessary.tex`.

---

## OPT-1.3B  (24 down-proj layers; NSAMP=40)

concentration (top-k% rows hold X% of 1st-pass weight error):
`5%→5%, 10%→11%, 25%→26%, 50%→51%, 100%→100%`  — **FLAT (error spread, no outlier channel)**

| down-proj treatment            | PPL    | dPPL    | ~Wbits |
|--------------------------------|--------|---------|--------|
| FP16 retention (baseline)      | 17.557 | --      | 16     |
| 1-pass FP4 (plain W4A4)        | 17.951 | +0.394  | 4.0    |
| act-only 2-pass (W 1-pass)     | 17.811 | +0.254  | 4.0    |
| act-only 4-pass (W 1-pass)     | 17.771 | +0.214  | 4.0    | ← act floor
| act2 + W2 top-5% rows          | 17.778 | +0.221  | 4.20   |
| act2 + W2 top-10% rows         | 17.774 | +0.217  | 4.40   |
| act2 + W2 top-25% rows         | 17.730 | +0.173  | 5.00   |
| act2 + W2 top-50% rows         | 17.545 | **-0.012** | 6.00 | ← ties retention
| act2 + W2 top-100% rows        | 17.528 | **-0.029** | 8.00 | ← **beats** retention, no FP16

**Verdict:** retention ELIMINATED — full act-2 + weight-2 (top-100%) reaches
−0.029 PPL vs retention (i.e. slightly better) at ~W8 on the down-proj, on the
FP4 array, no FP16 datapath. Flat concentration ⇒ selective gives no shortcut
(top-50% already ties at ~W6; below that the weight floor dominates).

---

## OPT-2.7B  (32 down-proj layers; NSAMP=40)

concentration: `5%→5%, 10%→10%, 25%→26%, 50%→51%, 100%→100%` — **FLAT**

| down-proj treatment            | PPL    | dPPL    | ~Wbits |
|--------------------------------|--------|---------|--------|
| FP16 retention (baseline)      | 13.597 | --      | 16     |
| 1-pass FP4 (plain W4A4)        | 13.674 | +0.077  | 4.0    | ← tiny collapse (rotation already tames it)
| act-only 2-pass (W 1-pass)     | 13.593 | **-0.004** | 4.0 | ← **ties retention, act-only, W4A4**
| act-only 4-pass (W 1-pass)     | 13.613 | +0.016  | 4.0    |
| act2 + W2 top-5% rows          | 13.636 | +0.039  | 4.20   |
| act2 + W2 top-10% rows         | 13.642 | +0.045  | 4.40   |
| act2 + W2 top-25% rows         | 13.613 | +0.016  | 5.00   |
| act2 + W2 top-50% rows         | 13.587 | -0.010  | 6.00   |
| act2 + W2 top-100% rows        | 13.560 | **-0.037** | 8.00 | ← beats retention

**Verdict:** retention ELIMINATED with **activation-only** residual (act-2 ties at −0.004, still W4A4);
full act+weight beats it (−0.037). The 1-pass collapse is only +0.077 here — the rotation already nearly
handles this layer, so opt-2.7b needs no weight residual at all. Flat concentration as always.

---

## OPT-6.7B  (32 down-proj layers; NSAMP=40)

concentration: `5%→5%, 10%→10%, 25%→26%, 50%→51%, 100%→100%` — **FLAT**

| down-proj treatment            | PPL    | dPPL    | ~Wbits |
|--------------------------------|--------|---------|--------|
| FP16 retention (baseline)      | 12.799 | --      | 16     |
| 1-pass FP4 (plain W4A4)        | 12.943 | +0.145  | 4.0    |
| act-only 2-pass (W 1-pass)     | 12.853 | +0.055  | 4.0    | ← near-parity, act-only
| act-only 4-pass (W 1-pass)     | 12.857 | +0.059  | 4.0    |
| act2 + W2 top-5% rows          | 12.821 | +0.023  | 4.20   |
| act2 + W2 top-25% rows         | 12.819 | +0.020  | 5.00   |
| act2 + W2 top-50% rows         | 12.817 | +0.018  | 6.00   |
| act2 + W2 top-100% rows        | 12.786 | **-0.012** | 8.00 | ← beats retention

**Verdict:** retention ELIMINATED — act-only 2-pass within +0.055 (eval noise, W4A4); full act+weight
beats (−0.012). Matches the earlier v3 run exactly. Flat concentration.

---

## OPT-125M  (12 down-proj layers; NSAMP=8 SMOKE — noisy, local 4080 validation only)

concentration: `5%→6%, 10%→11%, 25%→27%, 50%→52%, 100%→100%` — flat

| down-proj treatment            | PPL    | dPPL    | ~Wbits |
|--------------------------------|--------|---------|--------|
| FP16 retention (baseline)      | 33.836 | --      | 16     |
| 1-pass FP4 (plain W4A4)        | 34.975 | +1.139  | 4.0    |
| act-only 2-pass                | 34.496 | +0.660  | 4.0    |
| act2 + W2 top-100%             | 33.929 | +0.093  | 8.00   |

(NSAMP=8 smoke; rerun at NSAMP=40 on Colab for the paper number.)

---

## LLAMA-3-8B  (32 down-proj layers; NSAMP=40) — the HARD case / conditional-claim validator

concentration: `5%→5%, 10%→10%, 25%→26%, 50%→51%, 100%→100%` — **FLAT**

| down-proj treatment            | PPL    | dPPL    | ~Wbits |
|--------------------------------|--------|---------|--------|
| FP16 retention (baseline)      | 6.622  | --      | 16     |
| 1-pass FP4 (plain W4A4)        | 7.367  | +0.745  | 4.0    | ← BIG collapse (Llama-3 hard)
| act-only 2-pass (W 1-pass)     | 7.139  | +0.517  | 4.0    |
| act-only 4-pass (W 1-pass)     | 7.120  | +0.499  | 4.0    | ← act FLOOR = weight term
| act2 + W2 top-5% rows          | 7.118  | +0.496  | 4.20   |
| act2 + W2 top-10% rows         | 7.098  | +0.476  | 4.40   |
| act2 + W2 top-25% rows         | 7.008  | +0.386  | 5.00   |
| act2 + W2 top-50% rows         | 6.907  | +0.285  | 6.00   |
| act2 + W2 top-100% rows        | 6.663  | **+0.041** | 8.00 | ← near-parity ONLY at full weight-2

**Verdict:** retention eliminable but requires the **FULL** act-2 + weight-2 (top-100%, ~W8) → +0.041.
Weight-gap recovery is ~LINEAR in row fraction (25%→27%, 50%→49%, 100%→100% of the 0.476 gap): no
outlier-channel concentration (flat curve) → selective residual gives ~0 advantage over uniform. This is
the model that makes the retention proof CONDITIONAL: the activation half alone (+0.499 floor) cannot tie
retention here; the weight residual is necessary. Matches the earlier v3-notebook Llama-3-8B run exactly.

---

## STATUS / OOMs (2026-08-29, Colab A100-40GB)

Done (NSAMP=40): **opt-1.3b, opt-2.7b, opt-6.7b, Llama-3-8B.** Local smoke only: opt-125m (NSAMP=8).
FAILED — CUDA OOM on 40GB: **opt-13b** (26GB fp16, but `list(named_modules())` in wrap_model pins the
original fp16 weights so fp16+Wq coexist ≈52GB), **Llama-2-13b** (started with a dirty GPU after 13b's
failure). Not reached: **opt-30b** (60GB fp16 — needs A100-80GB regardless).
Fix: wrap_model now frees each original weight during wrapping (peak ≈26GB → 13b/Llama-2-13b fit 40GB);
30b still needs 80GB.

---

## Residual / distribution diagnostic (OPT-125m fc2, layer 6) — `residual_distribution_plot.py`

Figure: `writeups/residual_distributions.png` / `.pdf`.

**Excess kurtosis (Gaussian = 0; higher = heavier tails):**
| tensor | raw | rotated | 1-pass residual |
|--------|-----|---------|-----------------|
| weight (fc2)      | 3.65  | 0.07 | 1.41 (NVFP4) |
| activation (→fc2) | **65.61** | 0.16 | 2.67 (GF4) / **1.82 (NVFP4)** |

**Post-rotation activation quantized with NVFP4 (counterfactual — the weight format applied to acts):**
residual excess kurtosis **1.82** (LIGHTER than GF4's 2.67 — E2M1+absmax gives a more uniform residual),
and it **SATURATES at ~42.3 dB** (20→38→42.3→42.3→42.3) after 3 passes, just like NVFP4-on-weight (~39.4).
=> **saturation is a property of the NVFP4 FORMAT (E4M3-quantized block scale), not the operand.** GF4's
float RMS scale is the only reason the activation residual keeps decaying (→70 dB). Corollary: a GF4
(float-scale) *weight* residual would let the weight half keep decaying past the ~39 dB NVFP4 floor.

- At the DOWN-PROJ layer the **activation is far heavier-tailed than the weight** (65.6 vs 3.6) — the
  opposite of the generic intuition, because fc2/down_proj is the *massive-activation* layer (its input is
  post-nonlinearity, right-skewed with a 0-spike). This is exactly why it is the retention layer.
- **Randomized Hadamard Gaussianizes BOTH** (kurt → 0.07 / 0.16) — direct confirmation of the incoherence
  mechanism (`retention_unnecessary.tex` Lemma 1) and the Gaussian-residual premise (Lemma 2).

**Residual SNR (dB) by pass — decay asymmetry:**
- weight (NVFP4):    0 → 20.1 → 37.7 → **39.4 → 39.4 → 39.4**  (SATURATES ~39 dB after 3 passes)
- activation (GF4):  0 → 19.9 → 38.5 → 51.8 → 61.3 → **70.1**  (keeps decaying)
- Cause: NVFP4's block scale is E4M3-**quantized** (finite-precision scale → residual floors), while GF4's
  RMS scale is full-precision float → keeps shrinking. This is the mechanistic origin of the down-proj
  **weight floor** seen in the retention sweeps, and argues for a GF4 (float-scale) weight-residual if we
  want the weight half to keep decaying on hard models.

---

## GF4 vs NVFP4 granularity — why GF4 wins (`gf4_vs_nvfp4_granularity.py/.tex`)

Figure: `writeups/gf4_vs_nvfp4_granularity.png`. Proof: `writeups/gf4_vs_nvfp4_granularity.tex`.

**Mechanism (measured, honest):** the formats differ mainly in the BLOCK SCALE — GF4 uses RMS (scale=2.5·RMS,
outlier-robust), NVFP4 uses ABSMAX (scale=absmax/6). On a block of dynamic range κ=absmax/RMS, NVFP4 spends
its 15 levels over [0, κ·RMS], so only O(1/κ) land in the Gaussian BULK [0,~RMS] (5 levels at κ=2 → 2 at
κ=6 → bulk rounds to ZERO at κ>12); GF4 keeps all levels matched to the bulk. So NVFP4 bulk error grows with
κ, GF4's is κ-independent. This IS "heavy tail → loss of granularity," precisely.

Synthetic bulk-RMSE ratio (NVFP4/GF4) by κ: 1.00(κ2) 1.04(κ2.5) 1.16(κ6) 1.30(κ10) 1.52(κ20) — GF4 edge
grows monotonically with κ. Real rotated fc2 blocks: κ median 2.06 (rotation keeps κ small!), ratio 1.07
([2,3)) → 1.13 ([3,4)). Codebook shape is a SECOND, minor lever (~4% at clean κ; GF4 quantiles are
companding-matched to Gaussian).

**Honest effect size:** post-rotation κ is small, so the per-block granularity edge is MODEST (~7-13%);
it's large only at high κ (pre-rotation / outlier layers). Two SEPARATE compounding GF4 wins: (i) float RMS
scale → multipass keeps decaying (NVFP4 E4M3 scale saturates ~42 dB); (ii) codebook match. End-to-end
GF4<NVFP4 PPL = sum of the three, each small at post-rotation κ but consistently signed. Rotation is what
makes NVFP4 usable at all (collapses κ); GF4 keeps the residual edge on heavy blocks.

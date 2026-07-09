# Detailed Description of the Invention — GF4 (Gaussian-Optimal 4-Bit Datatype)

> Draft detailed description for a patent / invention disclosure. Placeholders in
> ANGLE BRACKETS (e.g. `<FIG. 1>`) mark figures to be prepared. Numeric results are
> from the project's own experiments and should be re-verified against the final
> results table before filing.

---

## 1. Field of the Invention

The invention relates to numeric datatypes and arithmetic hardware for machine
learning, and more specifically to a **4-bit non-uniform numeric datatype defined
by a lookup table** whose values are placed according to the statistics of a
normal (Gaussian) distribution, together with the quantization procedure and
hardware structures that use it to represent the **weights and activations** of
neural networks — in particular large language models (LLMs) — at 4 bits.

## 2. Background

Deploying LLMs is dominated by the cost of storing and moving their weights and
activations and by the energy of the multiply-accumulate (MAC) operations that
consume them. Reducing the numeric precision of these operands directly reduces
memory footprint, memory bandwidth, and MAC energy. Prior 4-bit representations
fall into two families, each with limitations that the present invention
addresses:

**(a) Uniform / floating-point 4-bit formats.** INT4 places 16 evenly spaced
levels; FP4 formats such as **E2M1** (and its packaged forms **NVFP4** and the
per-32 micro-scaled **MXFP4**) place levels on a coarse exponent–mantissa grid
that is logarithmically spaced. Both are a poor match to the distribution of
neural-network operands, which — especially after a decorrelating transform — are
approximately **Gaussian (bell-shaped)** and concentrate their probability mass
near zero. Evenly or logarithmically spaced levels "waste" codes in low-density
regions and under-resolve the dense region near zero, costing several decibels of
signal-to-noise ratio (SNR) and, at 4 bits, unacceptable model-accuracy loss —
particularly when quantizing **activations** (as opposed to weights only).

**(b) Quantile-based codebooks (NF4).** The NormalFloat-4 ("NF4") format used in
QLoRA improves on (a) by placing its 16 levels at quantiles of a normal
distribution. However, NF4 (i) is **asymmetric** (seven negative levels, an exact
zero, and eight positive levels), (ii) is defined and used as a **weight-only
storage format** for parameter-efficient fine-tuning, in which the 4-bit codes are
**dequantized to a wider floating-point type (e.g., bf16) before any arithmetic**,
and (iii) uses an **absolute-maximum (abs-max)** per-block scale in which the
single largest-magnitude element defines full scale. NF4 is therefore not a
hardware-native compute datatype, not applied to activations at inference, and its
asymmetric, abs-max design is sub-optimal for the symmetric, outlier-heavy
distributions seen in LLM inference.

There is accordingly a need for a 4-bit datatype that (1) matches the Gaussian
statistics of transformed neural-network operands, (2) is structured for efficient
**native hardware decode and arithmetic** rather than software dequantization,
(3) can represent **both weights and activations** at 4 bits (a "W4A4" regime),
and (4) provides a controllable precision/energy trade-off. The present invention
provides such a datatype and its supporting method and hardware.

## 3. Summary

In one aspect, the invention is a **4-bit numeric datatype** defined by a lookup
table (LUT) of **eight non-negative magnitude levels** whose positions are the
**equal-probability-mass quantiles of a standard normal distribution N(0,1)**,
normalized so that the largest level equals 1.0. The datatype is applied in
**symmetric sign-magnitude** form: each 4-bit code comprises one sign bit and a
3-bit index selecting one of the eight magnitudes, yielding a symmetric set of
levels about zero.

In a further aspect, operands are quantized to the datatype **per block** using a
scale derived from the **root-mean-square (RMS)** of the block multiplied by a
**clip ratio**, such that the bulk of the (approximately Gaussian) distribution
maps into the dense low-magnitude region of the table and tail values saturate to
full scale.

In a further aspect, the datatype is used in combination with an **incoherence
(e.g., Hadamard) rotation** applied to the operands prior to quantization, which
renders the per-block distribution approximately Gaussian and thereby makes the
Gaussian-optimal table near-ideal.

In a further aspect, the invention includes **hardware** comprising the LUT
embedded in a compute datapath — for example an eight-entry read-only memory (ROM)
plus a sign bit — that expands a 4-bit code to its represented value on the fly
and fuses it with a shared per-block scale (a block-floating-point arrangement),
enabling native 4-bit storage, bandwidth, and MAC operation.

In further aspects, the invention includes three variants that extend the base
datatype: (A) **per-block adaptive clip selection**; (B) a **learned codebook** in
which the eight level positions are optimized on calibration data; and (C) a
**residual / multi-pass** mode in which several 4-bit codes are accumulated to
reach a higher effective precision on a wide accumulator, providing a
precision-versus-energy dial and allowing sensitive ("outlier") layers to be run
without any dedicated wider-precision unit.

## 4. Brief Description of the Drawings

- `<FIG. 1>` A number line showing the eight GF4 magnitude levels (mirrored about
  zero) versus E2M1 and NF4 levels, illustrating the denser near-zero spacing.
- `<FIG. 2>` A histogram of post-rotation activations overlaid with the GF4 level
  positions, illustrating equal-mass placement.
- `<FIG. 3>` A flowchart of the per-block quantization procedure (rotate → RMS
  scale × clip → normalize/clamp → nearest-level LUT → sign·scale·level).
- `<FIG. 4>` A block diagram of the hardware embodiment: 4-bit operand storage →
  sign/index split → 8-entry magnitude ROM → multiplier by per-block scale → MAC.
- `<FIG. 5>` A block diagram of the multi-pass (residual) accumulator embodiment.
- `<FIG. 6>` A schematic of the W4A4 inference pipeline with optional outlier-layer
  handling.

## 5. Detailed Description

### 5.1 The GF4 datatype and its structure

The datatype, denoted **GF4** (Gaussian-optimal FP4), is defined by a table of
eight non-negative magnitude levels:

```
GF4_POS = { 0.0,        0.0796082,  0.1737177,  0.2828685,
            0.3952704,  0.5250730,  0.6961928,  1.0 }
```

These eight values are stored (e.g., in hardware as an 8-entry ROM, or in software
as a constant vector). A GF4 operand is a **4-bit sign-magnitude code**:

- **bit 3 (MSB):** sign `s ∈ {+, −}`;
- **bits 2–0:** a 3-bit index `i ∈ {0..7}` selecting `GF4_POS[i]`.

The represented value (before applying the block scale, Section 5.3) is
`s · GF4_POS[i]`. Because the negative levels mirror the positive levels, the
datatype is **symmetric about zero** and represents 15 distinct values across 16
codes (the index `i = 0` yields zero for either sign; the redundant ±0 code may be
reserved, e.g., for a special value, without affecting the numeric range). This
symmetric sign-magnitude structure is deliberately chosen for hardware simplicity:
sign handling reduces to a single bit, magnitude decode is a small fixed table,
and the representable range is exactly symmetric, which suits the symmetric
distributions of transformed weights and activations.

The levels are **non-uniformly spaced**: closely spaced near zero (successive gaps
of ≈0.080, 0.094, 0.109, …) and widely spaced in the tail (≈0.171, 0.304 up to
1.0). This concentrates representational resolution where a Gaussian operand
places most of its probability mass. `<FIG. 1>`, `<FIG. 2>`.

### 5.2 Construction of the table (equal-mass normal quantiles)

The level positions are obtained by a construction rule, so that the invention
covers both the specific table of Section 5.1 and tables produced by the same
method for other parameterizations:

1. Take the positive half of a standard normal distribution N(0,1) and divide it
   into a number of bins of **equal probability mass** (here, eight bins over the
   half-normal).
2. Take the bin boundaries (equivalently, representative quantiles) as candidate
   level positions.
3. **Normalize** so that the largest level equals 1.0.
4. Prepend an exact **0.0** level and apply the levels symmetrically via the sign
   bit (Section 5.1).

Because each level then corresponds to a region of approximately equal
probability mass under a Gaussian, the expected quantization error is minimized
for Gaussian-distributed inputs (an equal-mass, or "companding", quantizer). The
construction generalizes: the same rule with a different bin count yields tables
for other bit-widths, and with a different base distribution (e.g., Student-t or a
generalized normal) yields tables tuned to heavier- or lighter-tailed operands.

### 5.3 Per-block quantization procedure

Operands (a weight matrix or an activation tensor) are quantized in **blocks** of
`bs` contiguous elements (e.g., `bs = 16`), each block sharing a single scale.
For a block `x` (see `<FIG. 3>`):

1. **Scale.** Compute the block root-mean-square and multiply by a **clip ratio**
   `α` (default `α = 2.5`):
   `scale = RMS(x) · α`, with `RMS(x) = sqrt(mean(x²))`.
   Using RMS·α rather than the block abs-max sets full-scale at approximately
   `α` standard deviations, so ordinary values occupy the dense low-magnitude
   levels and only rare tail values reach or saturate full scale. This differs
   from abs-max scaling (as in NF4), in which a single outlier dilates the scale
   and de-resolves the bulk of the block.
2. **Normalize and clamp.** `x_norm = clamp(|x| / scale, 0, 1)`. Values beyond
   `α·σ` **saturate** to 1.0 (mapping to the top level), trading a small amount of
   tail fidelity for greatly improved resolution of the bulk.
3. **Assign.** For each element, select the nearest table level:
   `i* = argmin_i |x_norm − GF4_POS[i]|`, and record the sign `s = sign(x)`.
   The stored 4-bit code is `(s, i*)`.
4. **Reconstruct (dequantize).** `x_hat = s · scale · GF4_POS[i*]`.

The per-block `scale` is stored alongside the block (optionally itself quantized
to a compact floating-point form, e.g., an E4M3-like scale). The pair
(4-bit codes, per-block scale) constitutes a **block-floating-point** representation
in which the shared scale acts as a common exponent and the LUT code as a
non-uniform mantissa.

### 5.4 Enabling context — incoherence rotation

Raw neural-network activations are not Gaussian and contain large per-channel
outliers. In a preferred embodiment, an **incoherence rotation** — for example a
(randomized) **Hadamard transform** applied within each block, or a fused
block-butterfly rotation — is applied to the operands before quantization
(Section 5.3, step 0). Such an energy-preserving rotation spreads outlier energy
across the block and renders the per-block distribution **approximately Gaussian**,
which is precisely the condition under which the GF4 table (Section 5.2) is
optimal. The rotation is inverted implicitly by folding its inverse into an
adjacent linear operation, so it adds no run-time cost to the matrix product. In
this regime GF4 realizes several decibels of SNR advantage over logarithmically
spaced FP4 (E2M1) on the same operands. `<FIG. 6>`.

### 5.5 Hardware embodiment

In a hardware aspect (`<FIG. 4>`), the datatype is realized natively in a compute
datapath rather than dequantized in software:

- **Storage / bandwidth.** Operands are stored and transferred as 4-bit codes plus
  one shared scale per block, roughly quartering memory and bandwidth versus 16-bit
  operands.
- **Decode.** Each 4-bit code is split into its sign bit and 3-bit index. The
  index addresses an **eight-entry magnitude ROM/LUT** holding `GF4_POS`; the
  output is optionally negated per the sign bit. The LUT is small, fixed, and
  shared across the lanes of a systolic/tensor array.
- **Scale fusion.** The decoded magnitude is multiplied by the block's shared
  scale (a block-floating-point multiply); alternatively the scale is applied once
  to the accumulated partial sum for a block, amortizing it across the block's
  MACs.
- **MAC.** The decoded, scaled operands feed the multiply-accumulate array as in a
  conventional low-precision engine. Because decode is a table lookup plus a sign,
  the datatype imposes negligible area/energy overhead over INT4/FP4 decode while
  delivering Gaussian-matched accuracy.

The LUT may be **hardwired** (mask ROM) for the fixed table of Section 5.1, or
**writable** (register file / SRAM) to support the learned tables of Section 5.7,
allowing the same silicon to load a per-model or per-layer codebook.

### 5.6 Variant A — per-block adaptive clip

Rather than a fixed clip ratio `α`, an embodiment evaluates a small set of
candidate ratios (e.g., `{1.5, 2.0, 2.5, 3.0, 4.0}`) **independently per block**
and selects, for each block, the ratio minimizing that block's reconstruction
mean-squared error. The selected ratio (a few bits) is stored per block. This
adapts full-scale to each block's tail heaviness at negligible storage cost and is
computed once, at calibration time for weights or on-the-fly for activations.

### 5.7 Variant B — learned codebook

In another embodiment the eight level positions are **optimized on calibration
data** rather than fixed to the analytic quantiles. The levels are parameterized to
guarantee monotonicity and `[0,1]` normalization — for example as the cumulative
sum of a softmax over seven learnable increments, yielding six interior levels
bracketed by fixed endpoints `0` and `1`. A differentiable, **temperature-annealed
soft nearest-level assignment** (the assignment temperature `τ` decays from a large
to a small value, converging to hard selection) permits gradient descent to move
the levels to positions that minimize reconstruction error on representative
activations. The optimization is initialized from the analytic table of
Section 5.1. The resulting per-model (or per-layer) table is loaded into the
writable LUT of Section 5.5.

### 5.8 Variant C — residual / multi-pass mode

In another embodiment, higher effective precision is obtained by representing an
operand as the **sum of several GF4 codes**, each quantizing the residual left by
the previous passes (`<FIG. 5>`):

```
Q ← 0
for p in 1..P:  Q ← Q + GF4(x − Q)      # accumulated in a wide accumulator
```

Because the residual of GF4-quantized Gaussian data is itself approximately
Gaussian (zero-mean, reduced variance), GF4 remains near-optimal on **every**
pass. Each additional pass adds on the order of 9 dB SNR (≈1.5 effective bits),
providing a **precision dial**:

- `P = 1`: base GF4 (a true W4A4 operation);
- `P = 2`: "residual-GF4," reaching effective precision comparable to 16-bit
  activations (an "A16"-equivalent) while remaining a sum of 4-bit codes;
- `P = 3–4`: sufficient, in the inventors' experiments, to recover the accuracy of
  full-precision retention on the most sensitive layers.

Crucially, the multi-pass output is exactly what the **wide accumulator** of a
4-bit MAC array produces when it accumulates several 4-bit partial products; thus a
single multi-pass GF4 engine can run both ordinary layers (one pass) and sensitive
layers (several passes) **without any dedicated 16-bit unit**, the pass count being
a per-layer control. This unifies the compute for the whole network on one 4-bit
datapath.

### 5.9 Application to W4A4 LLM inference and outlier layers

In a system embodiment (`<FIG. 6>`), a transformer's linear projections have their
**weights** quantized to GF4 offline (optionally with the learned table of
Section 5.7 and per-block adaptive clip of Section 5.6) and their **activations**
quantized to GF4 at inference, both preceded by the incoherence rotation of
Section 5.4. A small number of empirically identified **outlier layers**
(for example the second feed-forward projection, the attention output projection,
or the output/head projection) may be treated with additional precision — either
retained at higher precision or, per Section 5.8, run at a higher pass count on the
same GF4 engine — to protect overall model accuracy at minimal cost.

### 5.10 Advantages and distinctions over the prior art

- **Versus E2M1 / NVFP4 / MXFP4:** GF4 places levels by Gaussian probability mass
  rather than on a logarithmic exponent–mantissa grid, recovering several dB of SNR
  on transformed operands and enabling 4-bit **activation** quantization that the
  logarithmic formats cannot sustain at comparable accuracy.
- **Versus NF4:** GF4 is (i) **symmetric sign-magnitude** (simpler, exactly
  symmetric hardware decode) versus NF4's asymmetric 16-value table; (ii) an
  **RMS·clip** companding scale with tail saturation versus NF4's abs-max scale;
  (iii) a **hardware-native compute datatype** decoded in the datapath versus NF4's
  software dequantization to bf16; and (iv) applied to **weights and activations
  (W4A4)** at inference versus NF4's weight-only storage for fine-tuning. GF4 is
  further extended by the adaptive-clip, learned-codebook, and multi-pass aspects,
  which NF4 does not contemplate.

### 5.11 Experimental support

On standard language-modeling benchmarks (e.g., WikiText-2 perplexity), operands
quantized with GF4 under the rotation-plus-block-clip scheme achieve accuracy at
4 bits close to full-precision baselines. For example, in the inventors'
experiments the two-pass residual-GF4 mode reaches perplexity essentially equal to
a 16-bit-activation configuration on a 7-billion-parameter model (within run-to-run
noise), and single-pass GF4 substantially outperforms logarithmically spaced FP4
of equal bit-width under matched fixed-clip conditions. These results should be
tabulated from the final results file and inserted here as `<TABLE 1>` prior to
filing.

## 6. Illustrative Claims (non-limiting sketch)

1. **(Datatype)** A numeric datatype for representing a value in four bits,
   comprising a sign bit and a three-bit index into a table of eight non-negative
   magnitude levels whose positions correspond to equal-probability-mass quantiles
   of a normal distribution normalized to a unit maximum.
2. The datatype of claim 1 wherein the eight levels are substantially
   `{0, 0.0796, 0.1737, 0.2829, 0.3953, 0.5251, 0.6962, 1.0}`.
3. **(Method)** A method of quantizing a block of neural-network operands to the
   datatype of claim 1, comprising computing a block scale from the root-mean-square
   of the block multiplied by a clip ratio, normalizing and clamping operand
   magnitudes by the scale, and selecting for each operand a nearest level of the
   table.
4. The method of claim 3, further comprising applying an incoherence (e.g.,
   Hadamard) rotation to the operands before said quantizing.
5. The method of claim 3, applied to both the weights and the activations of a
   transformer language model.
6. **(Adaptive clip)** The method of claim 3 wherein the clip ratio is selected
   per block from a plurality of candidate ratios to minimize a per-block
   reconstruction error.
7. **(Learned table)** The method of claim 3 wherein the eight level positions are
   obtained by gradient-based optimization on calibration data using a
   temperature-annealed differentiable assignment.
8. **(Multi-pass)** A method of representing an operand as a sum of a plurality of
   codes of the datatype of claim 1, each code quantizing a residual of the
   preceding codes accumulated in a wide accumulator, wherein the number of codes
   is selectable per layer.
9. **(Hardware)** An arithmetic circuit comprising storage for four-bit codes of
   the datatype of claim 1, a decode unit having an eight-entry magnitude lookup
   table addressed by the three-bit index and a sign inversion controlled by the
   sign bit, a multiplier applying a shared per-block scale, and a
   multiply-accumulate array consuming the decoded scaled operands.
10. The circuit of claim 9 wherein the lookup table is writable to load a per-model
    or per-layer table of level positions.

## 7. Alternative Embodiments

The base distribution may be other than standard normal (e.g., generalized normal
or Student-t) to match heavier- or lighter-tailed operands; the bin count may be
changed to define tables for other bit-widths; the rotation may be any
energy-preserving incoherence transform; the per-block scale may be shared over
other granularities (per-channel, per-tensor, micro-block); the LUT may be
hardwired or writable; and the multi-pass accumulation may share or vary the scale
and clip ratio across passes. Such variations are within the scope of the
invention.

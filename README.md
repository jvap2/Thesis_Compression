# TIME_ML — Jenks-Driven Compression for Deep Networks

Research code exploring a single unifying idea: **Jenks Natural Breaks** — a 1-D
clustering method that finds the partition of a sorted array minimizing
within-group variance — is a surprisingly powerful primitive for neural-network
compression. The same statistical break that separates "signal" from "noise" in a
weight tensor turns out to drive three otherwise-separate problems:

1. **Pruning** — the natural break in sorted weight magnitudes *is* the pruning threshold.
2. **Hyperparameter tuning** — Jenks statistics of the weights/saliency auto-schedule learning rate, weight decay, and momentum.
3. **FP4 quantization** — a Gaussian-optimal 4-bit datatype (**GF4**) built from equal-probability-mass quantiles, paired with Hadamard incoherence processing.

Most active work lives under [`Python_Jenks_Test/Jenks_Tests/`](Python_Jenks_Test/Jenks_Tests).

---

## 1. Jenks Natural-Breaks Pruning

Magnitude pruning normally needs a hand-picked sparsity target per layer. Instead we
sort each layer's weights and ask Jenks where the *natural* break between the
near-zero cluster and the load-bearing cluster falls — then prune everything below it.
The quality of that split is reported as the **Goodness of Variance Fit (GVF)**,
`GVF = (SSD_total − SSD_within) / SSD_total`, which doubles as a per-layer diagnostic
of how cleanly separable the weights are.

- **CUDA kernels** (`cuda_helpers.py`): `Jenks_Optimization`, `Jenks_Optimization_Biases`,
  and a 2-D conv variant compute the optimal break in parallel over rows / filters,
  JIT-compiled via `torch.utils.cpp_extension`.
- **Mask builders**: `Linear_Mask`, `Conv_Mask`, `Bias_Mask` return the binary keep-mask
  plus the GVF for a weight/bias tensor.
- **Iterative over-pruning**: the `OVER_PRUNE` dial (0.0 = pure Jenks) removes an
  additional `alpha`-fraction of the *survivors* beyond the natural break, so repeated
  passes compound sparsity — this is how the higher-sparsity points on the Pareto front
  are reached.

**Representative result:** VGG-style network on CIFAR-10 sustains **93.0% accuracy at
90% sparsity**.

---

## 2. Jenks Hyperparameter Tuning

Rather than manually schedule optimization during the prune → recover cycle, the
optimizer and LR scheduler read Jenks statistics of the current weights/saliency and
adjust themselves.

- **Optimizers** (`custom_optimizer.py`): `JenksSGD` / `JenksSGD_Test` fold the Jenks
  break into the update (per-element momentum handled inside the optimizer); also
  includes `ElementwiseMomentumSGD`, a `SAM` implementation, and noise variants.
- **Schedulers** (`custom_schedulers.py`): `WarmupAutoJenks` (and `WarmupAutoSGDJenks`)
  **automate both learning rate and weight decay** from the pruning saliency
  distribution — warmup, milestone decay, optional cosine, and rewind-to-init are all
  supported. This automated LR+WD path is the intended default; it is not a drop-in for
  a fixed manual schedule.

---

## 3. GF4 — Gaussian-Optimal FP4 Quantization

The [`FP_Quantization_Experiments/`](Python_Jenks_Test/Jenks_Tests/FP_Quantization_Experiments)
package implements a full **W4A4** post-training quantization pipeline for decoder-only
LLMs (OPT, LLaMA). See that package's [README](Python_Jenks_Test/Jenks_Tests/FP_Quantization_Experiments/README.md)
for the detailed method and ablations.

**The datatype.** A randomized blockwise **Hadamard rotation** applied to weights and
activations simultaneously suppresses outliers and drives the distribution toward
Gaussian. At that point the log-spaced E2M1/NVFP4 levels are suboptimal, and
**GF4** — 8 positive-half magnitudes placed at equal-probability-mass normal quantiles —
fits far better. GF4 is symmetric sign-magnitude (16 codes):

```
positive half (×scale): {0, 0.0796, 0.1737, 0.2829, 0.3953, 0.5251, 0.6962, 1.0}
```

Contrast with **NF4** (asymmetric 7neg/0/8pos, abs-max scale, weight-only) and
**E2M1/MXFP4** (log-spaced). GF4's equal-mass spacing minimizes expected quantization
MSE for a normal source.

**Pipeline highlights.**
- Per-block RMS × clip-ratio scaling; μ (Hadamard-domain mean) subtraction + bias correction.
- Hessian-weighted MSE weight reconstruction (block-diagonal Hessian).
- **CPU-offload calibration & block-major streaming eval** — the model stays in CPU RAM
  and one decoder block streams to the GPU at a time, so **13B/70B calibrate and evaluate
  on a single 24 GB GPU**.
- Fast fused-butterfly Hadamard (`fwht_hadacore`, Triton) for inference; pure-torch FWHT
  for calibration.

**Representative results** (WikiText-2, non-overlapping 2048):

| Model | Setting | PPL |
|---|---|---|
| Llama-2-7B | A16 (weight-only) | 5.65 |
| Llama-2-7B | residual GF4 **W4A4** | 5.66 |
| OPT-1.3B (FP32 baseline) | — | 14.62 |
| OPT-1.3B | residual GF4 W4A4 | 17.70 |

---

## Repository Layout

```
Python_Jenks_Test/Jenks_Tests/
├── cuda_helpers.py            # Jenks CUDA kernels + prune mask builders (GVF)
├── custom_optimizer.py        # JenksSGD, ElementwiseMomentumSGD, SAM
├── custom_schedulers.py       # WarmupAutoJenks (auto LR + weight decay)
├── training_loop.py           # prune → recover training harness
├── FP_QuantNetworkTest_CV.py  # CV (LeNet/ResNet/VGG/DenseNet) entry point
├── FP_QuantNetworkTest_LLM.py # LLM entry point
├── iso_energy_125m.py         # iso-energy FP4 quant harness (OFFLOAD/EVAL_OFFLOAD)
├── FP_Quantization_Experiments/   # GF4 / Hadamard / offload calibration (see its README)
└── <Model>_<Dataset>_output/  # experiment logs (git-ignored)
```

Other top-level directories (`CUDA_C_Code/`, `Python_Code/`, `Golang_Jenks_Test/`,
`Plane_Tests/`) hold earlier prototypes and language ports.

---

## Getting Started

Requires PyTorch with CUDA (the Jenks kernels JIT-compile on first import via
`torch.utils.cpp_extension`), plus `jenkspy`, `transformers`, `datasets`, and — for the
fast Hadamard path — `triton`.

```bash
# CV pruning + Jenks-scheduled training
python Python_Jenks_Test/Jenks_Tests/FP_QuantNetworkTest_CV.py

# FP4 GF4 LLM quantization (small model, in-GPU)
python Python_Jenks_Test/Jenks_Tests/iso_energy_125m.py facebook/opt-1.3b

# Large model via CPU-offload calibration + streaming eval (fits 13B on 24 GB)
RETAIN=1 OFFLOAD=1 EVAL_OFFLOAD=auto \
  python Python_Jenks_Test/Jenks_Tests/iso_energy_125m.py facebook/opt-13b
```

---

## Notes

- Experiment logs, checkpoints, and bytecode are git-ignored; results tables are checked
  in as CSV (`iso_results.csv`).
- GPU-side details for the large-model runs are in
  [`GCP_RUN.md`](Python_Jenks_Test/Jenks_Tests/GCP_RUN.md).

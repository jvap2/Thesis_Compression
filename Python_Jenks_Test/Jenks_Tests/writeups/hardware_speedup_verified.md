# GF4 datapath CUDA-kernel speedups — verified (2026-08-29, RTX 4080)

Source: `CUDA_FP4_Test/bench.py` (raw run: `CUDA_FP4_Test/gf4_datapath_bench_results_20260829_214701.txt`)
plus a large-N Hadamard re-run (bench.py N-list extended to 8192/16384). All are **measured GPU
wall-clock** on an RTX 4080 (Ada), fused kernel vs a naive/unfused GPU baseline — NOT ASIC numbers.

## 1. Fused randomized Hadamard (fused O(N log N) vs naive O(N^2))

| N | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|-----|-----|-----|------|------|------|------|-------|
| speedup | 0.56× | 0.57× | 0.69× | 1.29× | 1.84× | 3.36× | **5.26×** | **6.37×** |

- **The advantage GROWS with N** (O(N log N) vs O(N^2)); it is NOT a fixed number at a threshold.
- **Crossover (fused overtakes naive) is ~N=1024**, NOT N=512 (at N=512 the naive still wins, 0.69×).
- max_err(fused) ~1.4e-6 (numerically exact).
- **DISSERTATION CORRECTION:** the draft's "5.96× once N>=512" is wrong on both the number and the
  threshold. There is no N giving exactly 5.96× (it falls between the 8192 and 16384 power-of-2 points).
  Report the growth instead: 5.3× at N=8192, 6.4× at N=16384; crossover ~N=1024; 0.56× at N=128.

## 2. GF4 activation encoder (fused RMS+quantize+pack vs naive 3-kernel)  — CONFIRMS DRAFT

| n (elems) | 32768 | 2.1M | 8.4M |
|---|---|---|---|
| speedup | 1.30× | 1.50× | 1.46× |
=> **1.3–1.5×** as claimed. ✓ (code mismatch vs ref ~0)

## 3. Fused E2M1-dequant + GEMV (weight-only W4A16, single-token)  — CONFIRMS DRAFT

| M×K | 4096×4096 | 11008×4096 | 11008×11008 |
|---|---|---|---|
| wall-clock speedup | 1.49× | **2.96×** | 1.51× |
=> **1.5–2.9× wall-clock** as claimed. ✓

- HBM traffic: bench.py prints a *calculated* 6.8× (looser bound). The **measured** number (Nsight,
  `profile_hbm.py`, `dram__bytes_read/write.sum`) is **4.0× at M=K=4096 → 6.3× at M=K=11008** (README:
  3.96×/6.29×). The draft's "4.0–6.3×" is correct — cite the measured, not the 6.8× calculated. ✓

## 4. Hessian-damped weight solve kernel — CONFIRMS DRAFT
Block-diagonal Hessian accumulation, condition-number damping via shifted power iteration (no cuSOLVER),
per-(row,block) α*/bias/code solve. CUDA vs exact-eigvalsh rel_err tiny; clip-ratio search CUDA==ref
(α*=2.5). ✓ (matches `reference.py`)

## UNVERIFIED — the 30× / 2.5 GB transient-memory claim
`bench.py` does NOT measure this (no `torch.cuda.max_memory_allocated` in it, nothing in its output). The
draft's "≈30× the packed weight, 2.5 GB for an 83 MB matrix at M=K=11008" is unsourced in the committed
code. ACTION: locate the script that produced it, or re-measure with a `max_memory_allocated` harness.

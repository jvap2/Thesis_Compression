# JORTsE-on-ImageNet: performance optimizations (what actually worked)

Getting the pretrained ResNet-50 + JORTsE run from **~3.5 s/it → 3.11 it/s (~10×)** on a Colab
A100, **without changing the method** (automated LR/WD/momentum + Jenks-driven pruning). Masks are
bit-identical to the original except where noted (bf16). ~27 min/epoch; 40 epochs ≈ 18 h (one session).

Repo: `jvap2/Thesis_Compression`, files in `Python_Jenks_Test/Jenks_Tests/`.

## The changes that worked (in order), with measured impact

| # | Commit | Change | Why / impact |
|---|--------|--------|--------------|
| 1 | `f8733ee9` | `cuda_helpers.load_cuda`: build the inline CUDA ext for the **actual GPU capability** (autodetect + PTX fallback) instead of hardcoded `sm_86` | **Enabler, not speed.** Fixed `cudaErrorNoKernelImageForDevice` on the A100 (sm_80). Without it nothing ran. |
| 2 | `185c86f4` | **O(N²) → O(N) prefix-sum Jenks** (`torch_jenks_var`): replace the per-break CUDA kernel in `Linear_Mask`/`Bias_Mask` with `SSD = Σx² − (Σx)²/n` via `cumsum` | **The first big win.** The FC layer's O(N²) kernel over ~2M weights was ~3 s/step → killed the original 3.5 s/it. Verified: max diff ~1e-12, 0 argmin mismatches. |
| 3 | `bcdedc08` | Drop per-step `empty_cache()` in `train_one_step_prune_HPO` | Minor. Removed a forced per-step sync. |
| 4 | `6b820f4c` | Build masks **on-device** in `Linear_Mask`/`Bias_Mask` (were `torch.zeros(...)` on CPU + H2D copy per layer) | 44.5 → 3.1 ms for all 54 layers. Minor–moderate. |
| 5 | `2974a4ca` | Remove per-layer `synchronize()` + `empty_cache()` in `ElementwiseMomentumSGD.step` | Correct (a full device barrier per layer per step), but turned out **not** to be the dominant cost. |
| 6 | `0cf5fb14` | **Sync-free Jenks masks** — the decisive win | See below. |
| 7 | notebook cell 5b (not a repo commit) | **bf16 autocast on the forward** | Buys back the fwd/bwd (the remaining floor). See caveat. |

### #6 in detail — the decisive win (`0cf5fb14`)
Profiling the masked `opt.step` (batch 16, ResNet-50) showed it was **93% mask computation and CPU-bound**:
- GPU did only ~22 ms of work; the step took ~262 ms.
- ~4,500 tiny kernel launches/step, and **one `.item()` per layer** (`var.argmin().item()` + GVF readback)
  forced the CPU to *wait* for each layer's sort/kernel before starting the next → **no pipelining**, 54 serial stalls.

Fix: make the mask **sync-free** so the CPU launches all layers async and the GPU pipelines them:
- `Linear_Mask`/`Bias_Mask`: keep `var_min` on the GPU (`argmin`, no `.item()`); build the mask by
  `scatter` of a threshold-range over the sort indices — provably identical to `indices[var_min:]`.
- `Conv_Mask`: return `GVF.detach()` instead of `GVF.cpu().numpy().tolist()` (the per-conv-layer sync).
- GVF is diagnostic-only (logged / file-written, never used for control flow); the two `.4f` log sites are
  guarded for multi-element tensors.

**Result:** `opt.step` **262 → 40 ms locally (6.5×)**, **262 → 110 ms on the Colab A100 (2.4×)** — the A100
number is lower because the step is CPU-dispatch-bound and Colab's shared CPU dispatches slower. Masks
**verified bit-identical** vs the old logic at OVER_PRUNE 0.0 **and** 0.3 (0 mismatches). Also folded in a
dead-`decay_mask` removal and an arithmetic `beta_tensor` (`unsal + mask*(sal−unsal)`, no boolean-index).

### #7 in detail — bf16 forward (notebook only)
Wrap `model.forward` in `torch.autocast(bfloat16)`: conv/matmul on the A100 tensor cores; weights, the
elementwise optimizer, BatchNorm, and Jenks stay fp32. **bf16 (not fp16) ⇒ no `GradScaler`**, which is what
makes it safe with the custom momentum optimizer. Took the end-to-end rate to **3.11 it/s**.
**Caveat:** bf16 forward slightly perturbs the gradients → `WB = |W·∇|` and therefore the Jenks masks are
**not** bit-identical to fp32. Standard AMP practice, fine for accuracy, but keep **one fp32 run** (5b's bf16
off) as the method-fidelity anchor for the thesis.

## What we tried that did NOT work (rejected by measurement, not assumption)
- **Global segmented-batch Jenks** (compute every layer's break in one batched pass): implemented and
  **verified bit-identical** (0 mismatches, incl OVER_PRUNE), but **20× slower** — two full argsorts over all
  25.5M weights lose badly to 54 native per-tensor sorts. Correct ≠ faster.
- **`torch._foreach_` batching** of the per-layer optimizer arithmetic: the arithmetic was only ~18.7 ms of the
  step, so batching it was pointless.
- **Removing only the `std().item()` diagnostic sync:** 3% — confirmed the syncs weren't individually the cost;
  it was the *argmin* `.item()` gating the whole per-layer pipeline.

## Invariants preserved (verified)
- **Automated LR & weight decay** (`WarmupAutoJenks`, `custom_schedulers.py`): untouched.
- **Automated momentum** (`sal_beta = (1 − √(lr·‖∇⊙mask‖²))²`): untouched; `beta_tensor` arithmetic is
  identical in value.
- **Jenks masks:** bit-identical (verified), so gradient masking, per-weight β, `agg_score`, and the
  eventual `Prune_Score` behavior are unchanged. Only GVF changed type (GPU tensor, diagnostic-only).

## Meta-lesson
**Profile, don't guess.** Several "obvious" fixes (`empty_cache`, `synchronize`) gave ~0; the real cause
(per-layer `.item()` sync = 93% of the step, CPU-bound) only showed up under the profiler, and the rejected
batched idea was killed by a measurement, not intuition.

# JORTs / JORTsE CV pruning results (sparsity)

All accuracies are **honestly measured**: best val tracked ONLY from the frozen mask (after sparsity hits
the target), so the number is a true at-target-sparsity result, not a mid-churn lower-sparsity peak. EMA is
evaluated every post-freeze epoch against its own max. CIFAR-10 unless noted.

## The winning recipe (2026-08-29)
Iterative Jenks → target sparsity (`one_shot=False`) + **GC OFF** (`custom_optimizer.GC_ENABLED=False`) +
**gradient clipping** (`custom_optimizer.GRAD_CLIP_NORM=5.0`, `clip_grad_norm_` before `optimizer.step`) +
**late-EMA** (freeze-gated, `ema_decay=0.95`) + **NO rewind** (`reset=False`). MIXUP off.
- GC-off + grad-clip together kill the frozen-tail NaN explosion (GC starves 95%-empty filters; clip caps
  gradient magnitude without distorting direction).
- Late-EMA averages only the stable frozen-mask tail → real variance-reduction lift.

## Results

| Model | Target | Final sparsity | Plain (best) | **EMA (best)** | EMA lift | NaN | Prune/freeze |
|-------|--------|----------------|--------------|----------------|----------|-----|--------------|
| **ResNet-32** | 95% | 0.95334 | 0.91480 | **0.92951** | +1.47% | 0 | iter@150, froze ~195 |
| **DenseNet-40** | 90% | 0.90055 | 0.93028 | **0.93859** | +0.83% | 0 | iter@160, froze 255 |
| **VGG-19** | 99% | **plateaued 0.9674** (run 1) | — | — | — | 0 | see note; RERUN 2026-08-30 |

**VGG-19 run 1 (2026-08-29) — plateau, no usable result (fixed):** `prune_between=90` gave only 3 iterative
cuts (ep280/370/460), so sparsity plateaued at **0.9674** and never reached the 0.99 target. Because the
late-EMA freeze fires only at `sparsity >= prune_ratio`, it NEVER froze → the honest tracker reported
**plain 0.0 / EMA 0.0** (its failsafe: it will not report a below-target checkpoint; no mirage). Training was
healthy (0 NaN; plain hit 0.9288 at ~94% mid-run). FIX: `prune_between` 90 -> 5 (matches ResNet-32 win, which
cut every 5 epochs); rerun 2026-08-30 (log `vgg19_win2_*`). LESSON: for the iterative path, `prune_between`
must be small (~5) so there are enough cuts to reach a high target; a large value starves the ramp.

Scripts: `ResNet32_CIFAR10_rewind_ema_iterative.py`, `DenseNet40_CIFAR10_rewind_ema.py`,
`VGG19_CIFAR10_rewind_ema.py`. Logs under `*_output/` and `run_logs/`.

## Honest-baseline context (the "mirage")
The old HPO headline numbers were inflated: best-val was tracked from the prune epoch, so it locked onto the
FIRST-cut (~90%-sparsity) peak, not the target sparsity. Example — ResNet-32 "95%" HPO baseline
(`Best_Results_HPO/ResNet32/CIFAR-10/95_sparsity/`, 2026-06-11): headline **92.88%** was recorded at
**epoch 154, ~89.8% sparsity**; the TRUE frozen-95.345% best is **91.11%** (ep341), tail mean ~89.9%. So the
honest ResNet-32 @95% baseline is ~91%, and the new **EMA 92.95%** genuinely beats it by ~+1.85 at true 95%.
ACTION ITEM: audit every `@X%` headline in the results table — recompute as max val AFTER the sparsity log
shows the frozen (final) sparsity.

## Competition (CIFAR-10, from the user's results table)
- **ProbMask** (Zhou CVPR2021), ResNet-32: 94.96%@90, **94.16%@95**, 93.30%@98 (58800 iters, learned mask).
- **GSM**: 93.80%@90 ResNet-56 (but 859100 iters incl. 390500 pretrain).
- **SNIP** 91.01%@95 / **GRaSP** 91.39%@95 (prune-at-init) — JORTsE beats both.
- Old JORTsE data points: ResNet-32 92.88%@95.3 (MIRAGE), 93.55%@86.6; VGG 93.24%@90.9 & 90.79%@99.5;
  DenseNet 93.15%@84.5.

Standing: ResNet-32 EMA 92.95%@95 (honest) trails ProbMask 94.16%@95 by ~1.2% (was ~3% before the fixes);
DenseNet-40 EMA 93.86%@90 beats the old 93.15%@84.5 on both axes. VGG-19@99 pending.

---

## FULL AUDIT of Best_Results_HPO (2026-08-30) — honest vs mirage

Method: for each run, HONEST = max val acc over epochs AFTER the mask froze (sparsity reached its final
value); MIRAGE = max over all post-prune epochs. Iterations = epochs × ⌈train/batch⌉ (train=50k CIFAR,
100k TinyImageNet). **Only ResNet-32/CIFAR-10/95 is a genuine mirage** (iterative → prune@150 but froze@195,
so the 92.88% was a ~90%-sparsity checkpoint). Every other run was one-shot (freeze = prune epoch) so its
reported number was already honest.

| Model | Dataset | Final sparsity | Honest acc | EMA | Iterations | Mirage |
|-------|---------|----------------|-----------|-----|-----------|--------|
| ResNet-32 | CIFAR-10 | 0.8657 | 0.9355 | — | 75,000 | no |
| ResNet-32 | CIFAR-10 | 0.9535 | **0.9111** | — | 58,450 | **YES (was 0.9288)** |
| ResNet-32 | CIFAR-100 | 0.8519 | 0.7139 | — | 58,800 | no |
| ResNet-32 | CIFAR-100 | 0.8627 | 0.7028 | — | 75,000 | no |
| ResNet-32 | TinyImageNet | 0.8939 | 0.5112 | — | 78,200 | no |
| VGG-19 | CIFAR-10 | 0.9089 | 0.9324 | — | 50,100 | no |
| VGG-19 | CIFAR-10 | 0.9951 | 0.9079 | — | 98,000 | no |
| VGG-19 | CIFAR-100 | 0.8940 | 0.7354 | — | 58,450 | no |
| VGG-19 | CIFAR-100 | 0.9892 | 0.6434 | — | 195,500 | no |
| VGG-19 | TinyImageNet | 0.9792 | 0.4965 | — | 150,000 | no |
| DenseNet-40 | CIFAR-10 | 0.8450 | 0.9316 | — | 156,400 | no |
| ResNet-56 | CIFAR-10 | 0.8538 | 0.9245 | — | 156,400 | no |
| ResNet-56 (EMA) | CIFAR-10 | 0.8915 | 0.9154 | **0.9310** | 250,000 | no |
| **ResNet-32 NEW** | CIFAR-10 | **0.9533** | 0.9148 | **0.9295** | 58,450 | (honest) |
| **DenseNet-40 NEW** | CIFAR-10 | **0.9006** | 0.9303 | **0.9386** | 156,400 | (honest) |

ACTION: in the paper's CV table, correct ONLY the ResNet-32@95 CIFAR-10 row (92.88 → 91.11 for the old HPO
run, or replace it with the new EMA 92.95@95.33). All other rows are already honest.

### Prune epochs + pre-prune iterations (per HPO run)

| Run | batch | epochs | prune@ epoch | pre-prune iters |
|-----|-------|--------|--------------|-----------------|
| ResNet-32 CIFAR-10 86% | 200 | 300 | 200 | 50,000 |
| ResNet-32 CIFAR-10 95% | 300 | 350 | 150 | 25,050 |
| ResNet-32 CIFAR-100 85% | 256 | 300 | 250 | 49,000 |
| ResNet-32 CIFAR-100 86% | 200 | 300 | 250 | 62,500 |
| ResNet-32 TinyImageNet | 256 | 200 | 50 | 19,550 |
| VGG-19 CIFAR-10 90% | 300 | 300 | 260 | 43,420 |
| VGG-19 CIFAR-10 99% | 256 | 500 | 450 | 88,200 |
| VGG-19 CIFAR-100 90% | 300 | 350 | 280 | 46,760 |
| VGG-19 CIFAR-100 98% | 128 | 500 | 450 | 175,950 |
| VGG-19 TinyImageNet | 200 | 300 | 100 | 50,000 |
| DenseNet-40 CIFAR-10 84.5% | 128 | 400 | 350 | 136,850 |
| ResNet-56 CIFAR-10 85.4% | 128 | 400 | 350 | 136,850 |
| ResNet-56 (EMA) CIFAR-10 89% | 128 | 500 | 200 | 78,200 |
| NEW ResNet-32 CIFAR-10 95% | 300 | 350 | 150 | 25,050 |
| NEW DenseNet-40 CIFAR-10 90% | 128 | 400 | 160 | 62,560 |

Note: pre-prune iters = prune_epoch × ⌈train/batch⌉. For iterative runs the mask keeps growing AFTER prune@
until the freeze epoch; for one-shot runs prune@ = freeze. These are NOT a separate dense pretrain (unlike
GSM's ~390,500) — they are the early segment of a single continuous run.

---

## CIFAR-100 runs (2026-08-30) — two knob lessons

**ResNet-32 CIFAR-100 @95%:**
- v1 (LR 0.1, inherited from CIFAR-10): COLLAPSED to random (EMA 0.017) — LR too hot for CIFAR-100 at high
  sparsity. HPO CIFAR-100 used LR 0.035.
- v2 (LR 0.035): NO collapse — plain reached ~0.70 at ~94% sparsity (ep268). BUT the lower LR slowed the
  Jenks ramp, so it froze at ep320/95.47% leaving only 30 tail epochs -> undercooked final (plain 0.556,
  EMA 0.073, barely started). Freeze timing is LR-dependent: OP=0 froze @195 at LR 0.1 (CIFAR-10) but @320
  at LR 0.035.
- v3 (LR 0.035 + OVER_PRUNE 0.30): queued — the OP bump speeds the ramp so it freezes early enough for a
  long stable tail (recovery + EMA). LESSON: at a lower LR, raise OVER_PRUNE to keep the freeze early.

**VGG-19 CIFAR-100 @98%:** LR 0.02 already matches HPO (no LR issue); min_epochs bug fixed (=500). Running.

**VGG-19 CIFAR-10 @99%:** froze at 99.2% (recipe reaches the target!) but min_epochs=300<EPOCHS=500 made the
loop exit at the freeze (ep315) with no tail -> 0.0. min_epochs=500 fix applied; rerun queued.
LESSON (applies to all): min_epochs MUST equal EPOCHS, else the loop exits the moment sparsity hits target
after min_epochs, skipping the frozen tail + EMA.

---

## CIFAR-100 COMPETITION TABLE (user-provided 2026-08-30) — the targets

Accuracy | Sparsity | Pretrain iters | Pruning iters | Total iters. All methods 0 pretrain here.

**VGG-19:**
| Method | Acc | Sparsity | Total iters |
|--------|-----|----------|-------------|
| SNIP | 72.84% | 90% | 62,560 |
| SNIP | 71.83% | 95% | 62,560 |
| SNIP | 58.46% | 98% | 62,560 |
| GRaSP | 71.95% | 90% | 62,560 |
| GRaSP | 71.23% | 95% | 62,560 |
| GRaSP | 68.90% | 98% | 62,560 |
| GRaSP | 60.21% | 99.5% | 62,560 |
| **ProbMask** | **74.48%** | 90% | 58,800 |
| **ProbMask** | **73.94%** | 95% | 58,800 |
| **ProbMask** | **72.22%** | 98% | 58,800 |
| **ProbMask** | **70.10%** | 99.5% | 58,800 |
| JORTsE (ours) | 73.54% | 89.40% | 58,450 |
| JORTsE (ours) | 64.34% | 98.92% | 195,500 |

**ResNet-32:**
| Method | Acc | Sparsity | Total iters |
|--------|-----|----------|-------------|
| SNIP | 68.89% | 90% | 62,560 |
| SNIP | 65.22% | 95% | 62,560 |
| SNIP | 54.81% | 98% | 62,560 |
| GRaSP | 69.24% | 90% | 62,560 |
| GRaSP | 66.50% | 95% | 62,560 |
| GRaSP | 58.43% | 98% | 62,560 |
| **ProbMask** | **74.09%** | 90% | 58,800 |
| **ProbMask** | **73.06%** | 95% | 58,800 |
| **ProbMask** | **70.35%** | 98% | 58,800 |
| JORTsE (ours) | 72.08% | 87.24% | 120,000 |
| JORTsE (ours) | 67.70% | 95.2% | 58,450 |

TARGETS to beat with the winning recipe: ProbMask @95 = 73.06% (ResNet-32), 73.94% (VGG-19). Our current
best @95 CIFAR-100: ResNet-32 67.70% (old) — v3 (LR0.035 + OP0.30 + late-EMA) aims to match/beat this and,
if the tail is long enough, add an EMA lift. The honest ResNet-32 @95 gap to ProbMask is ~5% (vs ~1.2% on
CIFAR-10) — the regime where learned-mask quality (loss-aware saliency direction) matters most.

**VGG-19 CIFAR-100 @98% (2026-08-30, min_epochs fixed):** froze ep310/0.98294, ran full tail to 500, 0 NaN.
Plain 0.6284, **EMA 0.6545 @ 98.3% => EMA lift +2.6%.** Confirms the EMA thesis: 190-epoch tail let EMA work
(vs ResNet-32 C100 v2's 30-epoch tail where EMA=0.073). vs competition @98: beats SNIP 58.46 + old JORTsE
64.34, trails GRaSP 68.90 / ProbMask 72.22.

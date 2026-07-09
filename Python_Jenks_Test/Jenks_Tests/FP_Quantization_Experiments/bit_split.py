import functools
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F


from .utils import *


from contextlib import contextmanager



def set_act_quant(model, mode):
    for name, module in model.named_modules():
        if type(module).__name__ == "HadamardQuantLinearFP":
            module.act_quant_mode = mode
            # act_block_size is set during calibration; don't recalculate here
        elif isinstance(module, (QuantLinearFP, QuantConv2dFP, QuantConv1dFP,
                                 QuantLinearFP_Decomposed)):
            module.act_quant_mode = mode
            if not isinstance(module, QuantLinearFP_Decomposed):
                _, module.act_block_size = infer_act_quant_mode(
                    module.e_bits_scale, module.m_bits_scale, module.block_size
                )

@contextmanager
def act_quant_mode(model, mode="generic"):
    saved = {}
    for name, module in model.named_modules():
        if type(module).__name__ == "HadamardQuantLinearFP":
            saved[name] = module.act_quant_mode          # scalar, no act_block_size tuple
        elif isinstance(module, (QuantLinearFP, QuantConv2dFP, QuantConv1dFP,
                                 QuantLinearFP_Decomposed)):
            saved[name] = (module.act_quant_mode, module.act_block_size)

    set_act_quant(model, mode)

    try:
        yield
    finally:
        for name, module in model.named_modules():
            if type(module).__name__ == "HadamardQuantLinearFP":
                module.act_quant_mode = saved[name]
            elif isinstance(module, (QuantLinearFP, QuantConv2dFP, QuantConv1dFP,
                                     QuantLinearFP_Decomposed)):
                module.act_quant_mode, module.act_block_size = saved[name]


def infer_act_quant_mode(e_bits_scale, m_bits_scale, block_size):
    """
    Infer activation quantization mode from scale format and block size.
    Returns (mode_str, act_block_size) where mode_str matches the
    quantize_activations_* family of functions.
    
    MXFP4: E8M0 scale, block size 32
    NVFP4: FP8 E4M3 scale, block size 16
    Generic: anything else — uses the same e_bits_scale/m_bits_scale as weights
    """
    if e_bits_scale == 8 and m_bits_scale == 0:
        return "mxfp4", 32
    elif e_bits_scale == 4 and m_bits_scale == 3:
        return "nvfp4", 16
    else:
        return "generic", block_size


# Precompute and cache at module level
FP4_CODEBOOK_E2M1 = None

# Gaussian-optimal FP4 levels (NF4-style, positive half).
# Positive quantile boundaries for N(0,1) split into 8 equal-mass bins,
# normalized so the largest level = 1.0.
GF4_POS = torch.tensor([
    0.0, 0.0796082, 0.1737177, 0.2828685,
    0.3952704, 0.5250730, 0.6961928, 1.0,
], dtype=torch.float32)

def get_fp4_codebook(e_bits, m_bits, bias, device):
    """Build and cache the FP4 codebook — only 4 entries for E2M1."""
    global FP4_CODEBOOK_E2M1
    if FP4_CODEBOOK_E2M1 is None or FP4_CODEBOOK_E2M1.device != device:
        e_levels = torch.arange(0, 2**e_bits, device=device)
        m_levels = torch.arange(0, 2**m_bits, device=device)
        base     = 2.0 ** (e_levels.float() - bias)
        mant     = 1.0 + m_levels.float() / (2**m_bits)
        FP4_CODEBOOK_E2M1 = (base.unsqueeze(1) * mant.unsqueeze(0)).reshape(-1)
    return FP4_CODEBOOK_E2M1


def quantize_activations_fast(x, block_size, e_bits=2, m_bits=1,
                               e_bits_scale=4, m_bits_scale=3):
    """
    Fast path for inference-time activation quantization.
    Avoids Python loops, codebook reconstruction, and unnecessary ops.
    Assumes x.shape[-1] is divisible by block_size (true for OPT/LLaMA).
    """
    orig_dtype  = x.dtype
    orig_shape  = x.shape
    device      = x.device

    x_2d = x.reshape(-1, orig_shape[-1]).float()  # [N, K]
    N, K = x_2d.shape

    n_blocks = K // block_size
    # [N, n_blocks, block_size]
    x_blocks  = x_2d.view(N, n_blocks, block_size)

    # Per-block max for scaling — fully vectorized
    block_max  = x_blocks.abs().amax(dim=-1)       # [N, n_blocks]
    block_max  = block_max.clamp(min=1e-30)

    # Quantize scale to E4M3 (NVFP4) — vectorized
    scale = quantize_scale_tensor(block_max, e_bits=e_bits_scale, m_bits=m_bits_scale)  # [N, n_blocks]

    # Zero block guard
    zero_mask  = block_max < 1e-30                 # [N, n_blocks]
    scale_safe = scale.clone()
    scale_safe[zero_mask] = 1.0

    # Normalize into FP4 range — [N*n_blocks, block_size]
    scale_exp  = scale_safe.unsqueeze(-1)           # [N, n_blocks, 1]
    x_norm     = x_blocks.abs() / scale_exp        # [N, n_blocks, block_size]
    x_flat     = x_norm.reshape(N * n_blocks, block_size)  # [N*n_blocks, block_size]

    # Codebook lookup — fully vectorized, no Python loop
    # FP4 E2M1 codebook has only 4 entries
    codebook = get_fp4_codebook(e_bits, m_bits, bias=1, device=device)
    # [N*n_blocks, block_size, 1] vs [1, 1, 4]
    dist     = (x_flat.unsqueeze(-1) - codebook.view(1, 1, -1)).abs()
    basis    = codebook[dist.argmin(dim=-1)]        # [N*n_blocks, block_size]

    # Reshape and reconstruct
    basis    = basis.reshape(N, n_blocks, block_size)
    x_hat    = torch.sign(x_blocks) * scale_exp * basis  # [N, n_blocks, block_size]
    x_hat[zero_mask] = 0.0

    return x_hat.reshape(orig_shape).to(orig_dtype)

def quantize_activations(x, block_size, e_bits=2, m_bits=1,
                         e_bits_scale=8, m_bits_scale=0):
    """
    Generalized activation fake-quantization.
    Mode is inferred from e_bits_scale and m_bits_scale:
      e_bits_scale=8, m_bits_scale=0  -> MXFP4 (E8M0 scale, block 32)
      e_bits_scale=4, m_bits_scale=3  -> NVFP4 (FP8 E4M3 scale, block 16)
      anything else                   -> generic (uses given scale format)

    x: (..., K) — any leading dims treated as batch
    Returns x_hat of same shape.
    """
    mode, act_block_size = infer_act_quant_mode(e_bits_scale, m_bits_scale, block_size)
    orig_dtype = x.dtype
    orig_shape = x.shape
    x= x.float()
    x_2d = x.reshape(-1, orig_shape[-1])
    N, K = x_2d.shape

    pad   = (act_block_size - K % act_block_size) % act_block_size
    x_pad = F.pad(x_2d, (0, pad))
    K_pad = x_pad.shape[1]
    n_blocks = K_pad // act_block_size

    x_blocks  = x_pad.view(N, n_blocks, act_block_size)
    block_max = x_blocks.abs().amax(dim=-1).clamp(min=1e-30)  # [N, n_blocks]

    # ── Compute per-block scale ──────────────────────────────────────────
    if mode == "mxfp4":
        # E8M0: pure power-of-2, floor of log2
        log2_floor = torch.floor(torch.log2(block_max)).clamp(-127, 127)
        scale      = 2.0 ** log2_floor                         # [N, n_blocks]

    elif mode == "nvfp4":
        # FP8 E4M3 scale
        scale = quantize_scale_tensor(block_max, e_bits=4, m_bits=3)  # [N, n_blocks]

    else:
        # Generic: use whatever scale format the weights use
        scale = quantize_scale_tensor(block_max, e_bits_scale, m_bits_scale)

    # Handle dead blocks — zero block_max should stay zero after dequant
    zero_mask = block_max < 1e-30                              # [N, n_blocks]

    # ── Reshape to 2D for assign_fp4_dynamic ────────────────────────────
    # x_blocks: [N, n_blocks, act_block_size] -> [N*n_blocks, act_block_size]
    # scale:    [N, n_blocks]                 -> [N*n_blocks]
    x_flat    = x_blocks.abs().reshape(N * n_blocks, act_block_size)
    scale_flat = scale.reshape(N * n_blocks)

    # Zero-block guard — assign_fp4_dynamic divides by scale,
    # replace zero scales with 1.0 temporarily to avoid div-by-zero,
    # we will zero out the result afterwards via zero_mask
    zero_flat        = zero_mask.reshape(N * n_blocks)
    scale_flat_safe  = scale_flat.clone()
    scale_flat_safe[zero_flat] = 1.0

    _, _, basis = assign_fp4_dynamic(
        x_flat, scale_flat_safe, e_bits, m_bits
    )                                                          # [N*n_blocks, act_block_size]

    # Reshape back
    basis     = basis.reshape(N, n_blocks, act_block_size)    # [N, n_blocks, block_size]
    scale_exp = scale.unsqueeze(-1)                            # [N, n_blocks, 1]

    # Reconstruct: sign * scale * fp4_code
    x_hat = torch.sign(x_blocks) * scale_exp * basis          # [N, n_blocks, block_size]

    # Zero out dead blocks
    x_hat[zero_mask] = 0.0

    # ── Unpad and restore original shape ────────────────────────────────
    x_hat = x_hat.reshape(N, K_pad)
    if pad > 0:
        x_hat = x_hat[:, :K]
    return x_hat.reshape(orig_shape).to(orig_dtype)


def quantize_activations_gf4(x, block_size, clip_ratio=2.5, levels=None):
    """
    Gaussian-optimal FP4 activation quantization (GF4).

    After a full-row Hadamard transform the activation distribution is
    approximately Gaussian N(0, σ²).  Standard FP4 E2M1 levels are
    logarithmically spaced and waste 3–4 dB SNR on Gaussian inputs.
    GF4 uses NF4-style levels (equal-mass Gaussian quantile boundaries)
    with an RMS-based scale so that the bulk of the distribution maps
    to the dense low-level region.

    Scale per block = block_rms * clip_ratio.
    levels: [L] positive-half codebook (default: GF4_POS). Pass learned
            levels from calibrate_gf4_learned_levels for improved accuracy.
    """
    orig_dtype = x.dtype
    orig_shape = x.shape
    device     = x.device

    x_2d = x.reshape(-1, orig_shape[-1]).float()
    N, K = x_2d.shape

    pad   = (block_size - K % block_size) % block_size
    x_pad = F.pad(x_2d, (0, pad))
    K_pad = x_pad.shape[1]
    n_blk = K_pad // block_size

    x_blk = x_pad.reshape(N * n_blk, block_size)  # [B, bs]

    rms   = x_blk.pow(2).mean(dim=-1).sqrt().clamp(min=1e-8)  # [B]
    scale = (rms * clip_ratio).unsqueeze(-1)                   # [B, 1]

    sign      = torch.sign(x_blk)
    x_abs_n   = (x_blk.abs() / scale).clamp(0.0, 1.0)         # [B, bs]

    if levels is None:
        levels = GF4_POS.to(device=device)
    else:
        levels = levels.to(device=device)
    dist   = (x_abs_n.unsqueeze(-1) - levels.view(1, 1, -1)).abs()
    q_lvl  = levels[dist.argmin(dim=-1)]                       # [B, bs]

    x_hat = (sign * scale * q_lvl).reshape(N, K_pad)
    if pad > 0:
        x_hat = x_hat[:, :K]
    return x_hat.reshape(orig_shape).to(orig_dtype)


def quantize_activations_gf4_adaptive(
    x, block_size,
    clip_candidates=(1.5, 2.0, 2.5, 3.0, 4.0),
    levels=None,
    chunk_size=32768,
):
    """
    Per-block online clip-ratio selection for GF4 (novel).

    For each quantization block independently, evaluates all clip_candidates
    and picks the one minimizing reconstruction MSE. Blocks are processed in
    chunks of `chunk_size` to bound peak memory regardless of sequence length
    — the [B, bs, 8] dist tensor is the dominant allocation and would OOM on
    long GPTQ sequences (2048 tokens × batch 4 × 128 blocks = ~1M blocks).
    chunk_size=32768 keeps dist at ~16MB per candidate pass.
    """
    orig_dtype = x.dtype
    orig_shape = x.shape
    device     = x.device

    if levels is None:
        levels = GF4_POS.to(device=device)
    else:
        levels = levels.to(device=device)

    x_2d = x.reshape(-1, orig_shape[-1]).float()
    N, K = x_2d.shape

    pad   = (block_size - K % block_size) % block_size
    x_pad = F.pad(x_2d, (0, pad))
    K_pad = x_pad.shape[1]
    n_blk = K_pad // block_size
    B     = N * n_blk

    x_blk = x_pad.reshape(B, block_size)   # [B, bs]
    sign  = torch.sign(x_blk)
    x_abs = x_blk.abs()                    # [B, bs]
    rms   = x_abs.pow(2).mean(dim=-1).sqrt().clamp(min=1e-8)  # [B]

    best_mse  = torch.full((B,), float('inf'), device=device)  # [B]
    best_hat  = torch.zeros_like(x_blk)                        # [B, bs]

    for start in range(0, B, chunk_size):
        end      = min(start + chunk_size, B)
        s_abs    = x_abs[start:end]                            # [C, bs]
        s_sign   = sign[start:end]
        s_rms    = rms[start:end]                              # [C]

        c_best_mse = best_mse[start:end].clone()
        c_best_hat = best_hat[start:end].clone()

        for alpha in clip_candidates:
            scale  = (s_rms * alpha).unsqueeze(-1)             # [C, 1]
            x_norm = (s_abs / scale).clamp(0.0, 1.0)          # [C, bs]
            dist   = (x_norm.unsqueeze(-1) - levels.view(1, 1, -1)).abs()
            q_lvl  = levels[dist.argmin(dim=-1)]               # [C, bs]
            x_hat  = s_sign * scale * q_lvl                    # [C, bs]
            mse    = (s_abs - x_hat.abs()).pow(2).mean(dim=-1) # [C]

            better     = mse < c_best_mse
            c_best_mse = torch.where(better, mse, c_best_mse)
            c_best_hat = torch.where(better.unsqueeze(-1), x_hat, c_best_hat)

        best_mse[start:end] = c_best_mse
        best_hat[start:end] = c_best_hat

    x_hat_out = best_hat.reshape(N, K_pad)
    if pad > 0:
        x_hat_out = x_hat_out[:, :K]
    return x_hat_out.reshape(orig_shape).to(orig_dtype)


def quantize_activations_gf4_residual(
    x, block_size,
    clip_ratio1=2.5, clip_ratio2=2.5,
    levels=None,
):
    """
    Two-stage residual GF4 quantization (novel).

    Stage 1: Q1 = GF4(x,  clip_ratio1)
    Stage 2: Q2 = GF4(x - Q1, clip_ratio2)
    Output:  Q1 + Q2

    The residual of GF4-quantized Gaussian activations is also approximately
    Gaussian (zero-mean, reduced variance), so GF4 is near-optimal for the
    second stage too. Net effective resolution ≈ 2× at 2× activation compute.

    Useful as an ablation: compare against a single GF4 stage to measure
    the quantization-noise floor after one pass.
    """
    x_f  = x.float()
    x_q1 = quantize_activations_gf4(x_f, block_size, clip_ratio=clip_ratio1, levels=levels)
    residual = x_f - x_q1
    x_q2 = quantize_activations_gf4(residual, block_size, clip_ratio=clip_ratio2, levels=levels)
    return (x_q1 + x_q2).to(x.dtype)


def quantize_gf4_residual_npass(x, block_size, n_pass=2, clip_ratios=None, levels=None):
    """
    N-stage residual GF4 — the hardware "multi-pass" precision model.

    Generalizes quantize_activations_gf4_residual to an arbitrary pass count.
    Each pass GF4-quantizes the residual the previous passes left behind and
    accumulates in FP (one shared per-block scale across passes):

        Q = 0;  for p in range(n_pass):  Q += GF4(x - Q)

    Precision dial (each pass adds ~9 dB SNR ≈ 1.5 effective bits; the residual
    of GF4-quantized Gaussian data is itself ~Gaussian, so GF4 stays near-optimal
    on every pass):
        n_pass=1 -> plain GF4 (W4A4, ~3 eff. bits)
        n_pass=2 -> residual-GF4 (~6 eff. bits, ≈A16; == the 2-stage version)
        n_pass=4 -> ~6-10 eff. bits depending on tail heaviness — empirically
                    enough to recover FP16-RETENTION accuracy on the outlier
                    layers (down_proj/fc2/lm_head), so a multi-pass FP4 engine
                    runs them on its accumulator with NO dedicated FP16 unit.
                    (Validate the exact pass count with validate_multipass.py.)

    Distribution-agnostic, so it applies to weights or activations.  The output
    is a sum of n_pass FP4 codes — exactly what the wide accumulator on a
    multi-pass FP4 array produces, so its reconstruction error is the accuracy
    that hardware would actually see.
    """
    if clip_ratios is None:
        clip_ratios = [2.5] * n_pass
    x_f = x.float()
    q_total = torch.zeros_like(x_f)
    for p in range(n_pass):
        q = quantize_activations_gf4(x_f - q_total, block_size,
                                     clip_ratio=clip_ratios[p], levels=levels)
        q_total = q_total + q
    return q_total.to(x.dtype)


# ── Learned GF4 codebook optimization ────────────────────────────────────────

def optimize_gf4_levels(
    act_tensor,
    n_steps=400,
    lr=5e-3,
    tau_init=0.15,
    tau_final=0.005,
):
    """
    Gradient-descent optimization of GF4 codebook level positions (novel).

    Parameterizes 8 levels as cumulative softmax increments to enforce
    strict monotonicity and [0, 1] normalization:

        increments = softmax(Δ_raw)              # [7], sum to 1
        interior   = cumsum(increments)[:-1]     # [6] interior levels
        levels     = [0, interior..., 1]         # [8]

    Quantization uses temperature-annealed soft assignment (τ decays from
    tau_init → tau_final), which is differentiable and converges to hard
    nearest-neighbor. Initialized from GF4_POS.

    act_tensor: [N, bs] — block-normalized |activation| samples in [0, 1].
                Collect post-H, post-μ activations, divide by (block_rms * clip).
    Returns:    [8] float32 CPU tensor of learned levels.
    """
    device = act_tensor.device

    init_levels = GF4_POS.to(device)
    increments  = torch.diff(init_levels).clamp(min=1e-6)      # [7]
    # Inverse softmax (up to additive const; softmax is shift-invariant)
    delta_raw = torch.log(increments).detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([delta_raw], lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()

        inc      = F.softmax(delta_raw, dim=0)                  # [7], sums to 1
        interior = torch.cumsum(inc, dim=0)[:-1]                # [6]
        levels   = torch.cat([
            torch.zeros(1, device=device),
            interior,
            torch.ones(1, device=device),
        ])                                                       # [8]

        # Geometric temperature decay: high τ → smooth, low τ → hard
        t   = step / max(n_steps - 1, 1)
        tau = tau_init * (tau_final / tau_init) ** t

        dist   = (act_tensor.unsqueeze(-1) - levels.view(1, 1, -1)).abs()  # [N, bs, 8]
        soft_w = F.softmax(-dist / tau, dim=-1)
        q_soft = (soft_w * levels.view(1, 1, -1)).sum(dim=-1)              # [N, bs]

        loss = (act_tensor - q_soft).pow(2).mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        inc      = F.softmax(delta_raw, dim=0)
        interior = torch.cumsum(inc, dim=0)[:-1]
        levels   = torch.cat([
            torch.zeros(1, device=device),
            interior,
            torch.ones(1, device=device),
        ])
    return levels.detach().cpu()


def calibrate_gf4_learned_levels(
    model,
    calib_loader,
    device,
    block_size,
    num_batches=4,
    n_steps=400,
    max_samples=4096,
    per_layer=False,
):
    """
    Collect block-normalized post-H activations and optimize GF4 level
    positions via optimize_gf4_levels. Must be called after calibrate_model_gf4
    so that each HadamardQuantLinearFP has had_block_size, D, mu, and
    act_clip_ratio already set.

    per_layer=False: one shared codebook for all layers (stored on model as
                     model.gf4_learned_levels; each module gets a reference).
    per_layer=True:  separate codebook per layer (larger calibration budget).

    After this call, module.gf4_levels is set → quantize_activations_gf4
    uses the learned levels instead of GF4_POS when act_quant_mode=="gf4".
    """
    SKIP = {"lm_head", "embed_tokens", "embed_positions"}
    all_samples   = []
    layer_samples = {}

    hooks = []

    def make_hook(name, module):
        def _hook(mod, inp, out):
            x_raw = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            D   = module.D
            mu  = module.mu
            bs  = module.had_block_size
            clip = module.act_clip_ratio if module.act_clip_ratio is not None else 2.5
            if bs is None:
                return
            if D is not None:
                x_h = _rotate(x_raw, D, bs)
            else:
                if x_raw.shape[-1] < bs:
                    x_raw = F.pad(x_raw, (0, bs - x_raw.shape[-1]))
                x_h = fwht_blockwise(x_raw, bs)
            if mu is not None:
                x_h = x_h - mu.to(x_h.device)
            # Normalize blocks to [0, 1] for level optimization
            N2, K = x_h.shape
            pad   = (block_size - K % block_size) % block_size
            x_p   = F.pad(x_h, (0, pad))
            x_b   = x_p.reshape(-1, block_size)
            rms   = x_b.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
            x_n   = (x_b.abs() / (rms * clip)).clamp(0.0, 1.0)  # [B, bs]
            sample = x_n.cpu()
            all_samples.append(sample)
            if per_layer:
                layer_samples.setdefault(name, []).append(sample)
        return _hook

    for name, module in model.named_modules():
        if type(module).__name__ != "HadamardQuantLinearFP":
            continue
        if any(s in name for s in SKIP) or module.had_block_size is None:
            continue
        hooks.append(module.register_forward_hook(make_hook(name, module)))

    with torch.no_grad():
        n = 0
        for batch in calib_loader:
            if batch is None: continue
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if x is None: continue
            model(x.to(device))
            n += 1
            if n >= num_batches: break

    for h in hooks:
        h.remove()

    if not all_samples:
        print("calibrate_gf4_learned_levels: no activations collected")
        return model

    if per_layer:
        for name, module in model.named_modules():
            if type(module).__name__ != "HadamardQuantLinearFP":
                continue
            if name not in layer_samples or not layer_samples[name]:
                continue
            samples = torch.cat(layer_samples[name], dim=0).float()
            if samples.shape[0] > max_samples:
                idx = torch.randperm(samples.shape[0])[:max_samples]
                samples = samples[idx]
            print(f"  {name}: optimizing levels ({samples.shape[0]} blocks)")
            learned = optimize_gf4_levels(samples.to(device), n_steps=n_steps)
            module.gf4_levels = learned.to(device)
            print(f"    {learned.tolist()}")
    else:
        samples = torch.cat(all_samples, dim=0).float()
        if samples.shape[0] > max_samples:
            idx = torch.randperm(samples.shape[0])[:max_samples]
            samples = samples[idx]
        print(f"Optimizing global GF4 levels "
              f"({samples.shape[0]} blocks, {n_steps} steps)...")
        learned = optimize_gf4_levels(samples.to(device), n_steps=n_steps)
        print(f"  GF4_POS (theoretical): {GF4_POS.tolist()}")
        print(f"  Learned (empirical):   {learned.tolist()}")
        for name, module in model.named_modules():
            if type(module).__name__ == "HadamardQuantLinearFP":
                module.gf4_levels = learned.to(device)
        model.gf4_learned_levels = learned

    return model


def get_block_size(layer, default_block_size, override_conv=True):
    if override_conv and isinstance(layer, nn.Conv2d):
        kH, kW = layer.kernel_size
        return kH * kW  # spatial block per input channel
    return default_block_size

def solve_sign(W):
    """Extract sign: +1 or -1 per element."""
    return torch.sign(W).float()  # 0 stays 0
# =========================================================
# 🔹 SCALE QUANTIZATION (FP FORMAT)
# =========================================================
def quantize_scale(alpha, e_bits, m_bits):
    alpha = torch.tensor(alpha)
    e_min = -(2 ** (e_bits - 1))
    e_max = (2 ** (e_bits - 1)) - 1
    e = torch.floor(torch.log2(alpha))
    e = torch.clamp(e, e_min, e_max)
    base = 2.0 ** e
    if m_bits > 0:
        levels = 2 ** m_bits
        frac = alpha / base - 1.0
        frac_q = torch.round(frac * levels) / levels
        return base * (1.0 + frac_q)
    else:
        return base

def quantize_scale_batched(alpha, e_bits, m_bits):
    """
    Batched version of quantize_scale.
    Accepts alpha as [N] tensor — no torch.tensor() wrapping needed.
    """
    e_min = -(2 ** (e_bits - 1))
    e_max = (2 ** (e_bits - 1)) - 1

    e = torch.floor(torch.log2(alpha.clamp(min=1e-8)))
    e = torch.clamp(e, e_min, e_max)
    base = 2.0 ** e

    if m_bits > 0:
        levels = 2 ** m_bits
        frac = alpha / base - 1.0
        frac_q = torch.round(frac * levels) / levels
        return base * (1.0 + frac_q)
    else:
        return base

@functools.lru_cache(maxsize=16384)
def _quantize_scale_cached(alpha_val: float, e_bits: int, m_bits: int) -> float:
    """LRU-cached scalar version of quantize_scale."""
    alpha = torch.tensor(alpha_val, dtype=torch.float32)
    e_min = -(2 ** (e_bits - 1))
    e_max =  (2 ** (e_bits - 1)) - 1
    e     = torch.floor(torch.log2(alpha.clamp(min=1e-30))).clamp(e_min, e_max)
    base  = 2.0 ** e
    if m_bits > 0:
        levels = 2 ** m_bits
        frac_q = torch.round((alpha / base - 1.0) * levels) / levels
        return float(base * (1.0 + frac_q))
    return float(base)


def quantize_scale_tensor(alpha: torch.Tensor,
                          e_bits: int,
                          m_bits: int) -> torch.Tensor:
    """
    Fully vectorised quantize_scale that operates on an arbitrary-shaped
    tensor of positive alpha values.  Avoids any Python loop and is
    compatible with autograd (though gradients are not needed here).
    """
    e_min = -(2 ** (e_bits - 1))
    e_max =  (2 ** (e_bits - 1)) - 1
    e     = torch.floor(torch.log2(alpha.clamp(min=1e-30))).clamp(e_min, e_max)
    base  = 2.0 ** e
    if m_bits > 0:
        levels = 2 ** m_bits
        frac_q = torch.round((alpha / base - 1.0) * levels) / levels
        return base * (1.0 + frac_q)
    return base

# =========================================================
# 🔹 BLOCKWISE ALPHA SOLVE (mask-aware)
# =========================================================
def solve_alpha_blockwise(W, W_tilde, mask, block_size):
    if W.dim() == 4:
        W_mat = W.view(W.shape[0], -1)
        Wt_mat = W_tilde.view(W.shape[0], -1)
        M_mat = mask.view(W.shape[0], -1)
    else:
        W_mat = W
        Wt_mat = W_tilde
        M_mat = mask

    alpha = torch.zeros_like(W_mat)

    for row in range(W_mat.shape[0]):
        w_row = W_mat[row]
        wt_row = Wt_mat[row]
        m_row = M_mat[row]

        for i in range(0, w_row.numel(), block_size):
            end = min(i + block_size, w_row.numel())

            mask_idx = (m_row[i:end] > 1e-8)

            w_block = w_row[i:end]
            wt_block = wt_row[i:end]

            num = (w_block[mask_idx] * wt_block[mask_idx]).sum()
            den = (wt_block[mask_idx] ** 2).sum() + 1e-8

            alpha[row, i:end] = num / den

    return alpha.view_as(W)


def solve_alpha_blockwise_Hessian_correct(
    W, basis, H_diag, mask, block_size, eps=1e-8
):
    """
    Hessian-aware blockwise alpha solve (per-block scalar).

    W, basis, H_diag, mask: [N, M]
    Returns alpha: [N, M] where each block has a single scalar
    """
    N, M = W.shape
    alpha = torch.zeros_like(W)

    for row in range(N):
        w_row = W[row]
        b_row = basis[row]
        h_row = H_diag[row]
        m_row = mask[row]

        for i in range(0, M, block_size):
            end = min(i + block_size, M)

            w_block = w_row[i:end]
            b_block = b_row[i:end]
            h_block = h_row[i:end]
            m_block = m_row[i:end]

            # apply mask
            w_block = w_block * m_block
            b_block = b_block * m_block
            h_block = h_block * m_block

            # normalize Hessian to avoid domination
            mean_h = h_block.mean()
            if mean_h < eps:
                h_block = torch.ones_like(h_block)
            else:
                h_block = h_block / mean_h

            # blockwise scalar alpha
            num = (h_block * w_block * b_block).sum()
            den = (h_block * b_block * b_block).sum() + eps
            alpha_block = (num / den).clamp(min=eps)

            # broadcast alpha to entire block
            alpha[row, i:end] = alpha_block

    return alpha


def solve_alpha_blockwise_Hessian_full(
    W, basis, H_blocks, mask, block_size, eps=1e-8
):
    """
    Full block-diagonal Hessian solve (correct geometry).

    W, basis, mask: [N, M]
    H_blocks: list of [k, k]
    """
    N, M = W.shape
    alpha = torch.zeros((N, M), device=W.device)

    for row in range(N):
        w_row = W[row]
        b_row = basis[row]
        m_row = mask[row]

        block_idx = 0

        for i in range(0, M, block_size):
            end = min(i + block_size, M)

            H = H_blocks[block_idx].to(W.device)

            w_block = w_row[i:end]
            b_block = b_row[i:end]
            m_block = m_row[i:end]

            # --- handle fully pruned block ---
            if m_block.sum() < 1e-8:
                alpha_block = torch.tensor(1.0, device=W.device)
                alpha[row, i:end] = alpha_block
                block_idx += 1
                continue

            # --- apply mask (DO NOT drop indices) ---
            w_block = w_block * m_block
            b_block = b_block * m_block

            # --- ensure H matches size ---
            k = w_block.numel()
            if H.shape[0] != k:
                H = H[:k, :k]

            # --- quadratic form ---
            Hb = H @ b_block
            Hw = H @ w_block

            num = (b_block * Hw).sum()
            den = (b_block * Hb).sum() + eps

            alpha_block = num / den

            # --- prevent collapse ---
            alpha_block = torch.clamp(alpha_block, min=1e-6)

            alpha[row, i:end] = alpha_block

            block_idx += 1

    return alpha


# =========================================================
# 🔹 EXPONENT (COARSE, mask-aware)
# =========================================================
def solve_exponent(W_abs_scaled, e_bits, mask):
    """
    W_abs_scaled = |W| / alpha  (positive, per-element)
    Returns integer exponent e in [0, 2^e_bits - 1]
    bias = 2^(e_bits-1) - 1
    """
    bias = 2 ** (e_bits - 1) - 1
    # guard against log(0)
    log_val = torch.log2(W_abs_scaled.clamp(min=1e-8))
    e = torch.round(log_val) + bias
    e = torch.clamp(e, 0, 2 ** e_bits - 1)
    return (e * mask).long()


def compute_hessian_blocks(x, layer, block_size):
    if isinstance(layer, nn.Conv2d):
        unfold = nn.Unfold(
            kernel_size=layer.kernel_size,
            dilation=layer.dilation,
            padding=layer.padding,
            stride=layer.stride
        )
        x = unfold(x)
        x = x.permute(0, 2, 1).reshape(-1, x.shape[1])
    elif isinstance(layer, nn.Linear):
        x = x.reshape(-1, x.shape[-1])
    elif type(layer).__name__ == "Conv1D":
        # GPT-2 Conv1D is a linear projection: x @ W + b
        # x shape is (B, seq_len, in_features) — flatten to (N, D)
        x = x.reshape(-1, x.shape[-1])
    else:
        print(f"  compute_hessian_blocks: unrecognised layer type {type(layer).__name__}, returning None")
        return None

    N, D = x.shape
    H = (x.T @ x) / N

    H_blocks = []
    for i in range(0, D, block_size):
        end = min(i + block_size, D)
        # .clone() is REQUIRED: a bare slice is a view that keeps the full
        # D×D dense H alive. Stored across all layers in H_data, those views
        # retain a dense Hessian per layer (419 MB for a 10240-wide fc2) and
        # OOM large models. Cloning detaches the k×k block so H can be freed.
        H_blocks.append(H[i:end, i:end].clone())
    del H
    return H_blocks

def solve_alpha_blockwise_Hessian_blockdiag(W, basis, H_blocks, mask, block_size):
    """
    Block-diagonal Hessian solve.

    W, basis, mask: [N, M]
    H_blocks: list of [k, k]
    """
    N, M = W.shape
    alpha = torch.zeros_like(W)

    for row in range(N):
        w_row = W[row]
        b_row = basis[row]
        m_row = mask[row]

        block_idx = 0

        for i in range(0, M, block_size):
            end = min(i + block_size, M)

            H = H_blocks[block_idx].to(W.device)

            w_block = w_row[i:end]
            b_block = b_row[i:end]
            m_block = m_row[i:end]

            # apply mask
            w_block = w_block * m_block
            b_block = b_block * m_block

            # compute quadratic form
            Hb = H @ b_block
            Hw = H @ w_block

            num = (b_block * Hw).sum()
            den = (b_block * Hb).sum() + 1e-8

            alpha_block = num / den

            alpha[row, i:end] = alpha_block

            block_idx += 1

    return alpha

# def compute_hessian_blockdiag_model(model, data_loader, device, block_size, num_batches=4):
#     H_data = {}
#     hook_map = {}

#     def make_hook(name, inner_mod):
#         def hook(mod, inp, out):
#             x = inp[0].detach()
#             H_blocks = compute_hessian_blocks(x, inner_mod, block_size)
#             if H_blocks is None:
#                 return
#             if name not in H_data:
#                 H_data[name] = H_blocks
#             else:
#                 for i in range(len(H_blocks)):
#                     H_data[name][i] += H_blocks[i]
#         return hook

#     handles = []
#     for name, module in model.named_modules():
#         if isinstance(module, QuantConv2dFP):
#             handles.append(module.register_forward_hook(make_hook(name, module.conv)))
#             hook_map[name] = module.conv
#         elif isinstance(module, QuantLinearFP):
#             handles.append(module.register_forward_hook(make_hook(name, module.linear)))
#             hook_map[name] = module.linear
#         elif isinstance(module, QuantConv1dFP):
#             print(f"  Registering hook for Conv1D: {name}")
#             handles.append(module.register_forward_hook(make_hook(name, module.conv1d)))
#             hook_map[name] = module.conv1d

#     print(f"Total hooks registered: {len(handles)}")
#     model.eval()
#     with torch.no_grad():
#         for i, (x, _) in enumerate(data_loader):
#             x = x.to(device)
#             model(x)
#             if i + 1 >= num_batches:
#                 break

#     for h in handles:
#         h.remove()

#     H_final = {}
#     for name, blocks in H_data.items():
#         H_final[name] = [b / num_batches for b in blocks]

#     return H_final

def compute_hessian_blockdiag_model(model, data_loader, device, block_size, num_batches=4):
    H_data   = {}
    hook_map = {}
    handles  = []

    def make_hook(name, inner_mod):
        def hook(mod, inp, out):
            x = inp[0].detach()
            H_blocks = compute_hessian_blocks(x, inner_mod, block_size)
            if H_blocks is None:
                return
            if name not in H_data:
                H_data[name] = H_blocks
            else:
                for i in range(len(H_blocks)):
                    H_data[name][i] += H_blocks[i]
        return hook

    for name, module in model.named_modules():
        if isinstance(module, QuantConv2dFP):
            handles.append(module.register_forward_hook(
                make_hook(name, module.conv)))
            hook_map[name] = module.conv

        elif isinstance(module, QuantLinearFP):
            # Skip inner modules that live inside a HadamardQuantLinearFP —
            # the wrapper is registered separately below under the correct name
            if name.endswith(".inner"):
                continue
            handles.append(module.register_forward_hook(
                make_hook(name, module.linear)))
            hook_map[name] = module.linear

        elif isinstance(module, HadamardQuantLinearFP):
            # Register on the wrapper but collect activations from the inner
            # linear — inp[0] is the pre-Hadamard input, which is what we want
            # for computing H_original (identical to v5, orthogonality argument)
            handles.append(module.register_forward_hook(
                make_hook(name, module.inner.linear)))
            hook_map[name] = module.inner.linear

        elif isinstance(module, QuantConv1dFP):
            print(f"  Registering hook for Conv1D: {name}")
            handles.append(module.register_forward_hook(
                make_hook(name, module.conv1d)))
            hook_map[name] = module.conv1d

    print(f"Total hooks registered: {len(handles)}")

    model.eval()
    batches_run = 0
    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                continue
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            if x is None:
                continue
            model(x.to(device))
            batches_run += 1
            if batches_run >= num_batches:
                break

    for h in handles:
        h.remove()

    H_final = {}
    for name, blocks in H_data.items():
        H_final[name] = [b.float() / batches_run for b in blocks]

    print(f"  Collected H for {len(H_final)} layers "
          f"over {batches_run} batches")
    return H_final

def compute_hessian_hadamard_domain(model, data_loader, device, block_size, num_batches=4):
    """
    Compute blockwise Hessian H_had = E[x_had x_had^T] in the Hadamard domain.

    Must be called AFTER wrap_layers_with_hadamard() and after D and had_block_size
    are set on every HadamardQuantLinearFP module (Phase 2 of calibration).
    Each hook captures inp[0] (pre-Hadamard raw activations), applies H(D*x),
    and accumulates the block-diagonal of the resulting outer product.

    This is the correct Hessian for quantizing W_had under the objective
        argmin_{Q(W_had)} ||W_had - Q(W_had)||_{H_had}
    rather than the pre-rotation H_orig which is in the wrong domain.
    """
    H_data  = {}
    handles = []

    for name, module in model.named_modules():
        if type(module).__name__ != "HadamardQuantLinearFP":
            continue
        if module.D is None or module.had_block_size is None:
            continue

        D_local      = module.D
        had_bs_local = module.had_block_size
        inner_mod    = module.inner.linear

        def make_hook(n, D_ref, had_bs, inner):
            def hook(mod, inp, out):
                x = inp[0].detach().float()
                x_2d  = x.reshape(-1, x.shape[-1])
                x_had = _rotate(x_2d, D_ref, had_bs)
                H_blocks = compute_hessian_blocks(x_had, inner, block_size)
                if H_blocks is None:
                    return
                if n not in H_data:
                    H_data[n] = H_blocks
                else:
                    for i in range(len(H_blocks)):
                        H_data[n][i] += H_blocks[i]
            return hook

        handles.append(module.register_forward_hook(
            make_hook(name, D_local, had_bs_local, inner_mod)
        ))

    print(f"Hadamard-domain Hessian: {len(handles)} hooks registered")

    model.eval()
    batches_run = 0
    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                continue
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if x is None:
                continue
            model(x.to(device))
            batches_run += 1
            if batches_run >= num_batches:
                break

    for h in handles:
        h.remove()

    H_final = {}
    for name, blocks in H_data.items():
        H_final[name] = [b.float() / batches_run for b in blocks]

    print(f"  Collected H_had for {len(H_final)} layers over {batches_run} batches")
    return H_final


def compute_hessian_blockdiag_model_joint(
    model, data_loader, device, block_size, num_batches=4,
    e_bits=2, m_bits=1, e_bits_scale=8, m_bits_scale=0,
):
    """
    Computes H = x_bar^T x_bar / N using fake-quantized activations,
    giving the correct Hessian for the joint objective ||Wx - W_bar x_bar||^2.
    Mode (MXFP4/NVFP4/generic) is inferred from e_bits_scale and m_bits_scale.
    """
    mode, act_block_size = infer_act_quant_mode(e_bits_scale, m_bits_scale, block_size)
    print(f"Joint Hessian mode: {mode}, act_block_size: {act_block_size}")

    H_data   = {}
    hook_map = {}

    def make_hook(name, inner_mod):
        def hook(mod, inp, out):
            x = inp[0].detach()
            # Quantize activations before computing Hessian
            x_q = quantize_activations(
                x, block_size, e_bits, m_bits, e_bits_scale, m_bits_scale
            )
            H_blocks = compute_hessian_blocks(x_q, inner_mod, block_size)
            if H_blocks is None:
                return
            if name not in H_data:
                H_data[name] = H_blocks
            else:
                for i in range(len(H_blocks)):
                    H_data[name][i] += H_blocks[i]
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, QuantLinearFP):
            handles.append(module.register_forward_hook(make_hook(name, module.linear)))
            hook_map[name] = module.linear
        elif isinstance(module, QuantConv2dFP):
            handles.append(module.register_forward_hook(make_hook(name, module.conv)))
            hook_map[name] = module.conv
        elif isinstance(module, QuantConv1dFP):
            handles.append(module.register_forward_hook(make_hook(name, module.conv1d)))
            hook_map[name] = module.conv1d

    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            x = x.to(device)
            model(x)
            if i + 1 >= num_batches:
                break

    for h in handles:
        h.remove()

    return {name: [b / num_batches for b in blocks]
            for name, blocks in H_data.items()}

def conv_input_hessian_diag(x, conv):
    """
    Memory-efficient equivalent of unfold-based Hessian diag.

    Returns:
        diag: [out_channels, in_channels * kh * kw]
    """
    B, C, H, W = x.shape
    kh, kw = conv.kernel_size

    x2 = x ** 2

    weight = torch.ones(
        (C, 1, kh, kw),
        device=x.device,
        dtype=x.dtype
    )

    out = torch.nn.functional.conv2d(
        x2,
        weight,
        bias=None,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=C
    )
    # [B, C, H_out, W_out]

    # Sum exactly like unfold version
    diag_per_channel = out.sum(dim=(0, 2, 3))  # [C]

    # Expand to kernel elements
    diag = diag_per_channel.repeat_interleave(kh * kw)  # [C * kh * kw]

    diag = diag.unsqueeze(0).repeat(conv.out_channels, 1)

    return diag



def compute_hessian_diag_model(model, data_loader, device, num_batches=50):
    H_data = {}
    hook_name_to_submodule = {}

    def make_hook(name, inner_mod):
        def hook(mod, inp, out):
            x = inp[0].detach()

            if isinstance(inner_mod, nn.Conv2d):
                diag = conv_input_hessian_diag(x, inner_mod)
                count = x.shape[0] * out.shape[2] * out.shape[3]

            elif isinstance(inner_mod, nn.Linear):
                x = x.reshape(-1, x.shape[-1])
                diag = (x ** 2).sum(dim=0, keepdim=True)
                diag = diag.repeat(inner_mod.out_features, 1)
                count = x.shape[0]

            else:
                return

            if name not in H_data:
                H_data[name] = [diag, count]
            else:
                H_data[name][0] += diag
                H_data[name][1] += count

        return hook

    handles = []

    for name, module in model.named_modules():
        if isinstance(module, QuantConv2dFP):
            handles.append(module.register_forward_hook(make_hook(name, module.conv)))
            hook_name_to_submodule[name] = module.conv

        elif isinstance(module, QuantLinearFP):
            handles.append(module.register_forward_hook(make_hook(name, module.linear)))
            hook_name_to_submodule[name] = module.linear

    print(f"=== Hooks registered: {len(handles)} ===")

    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            x = x.to(device)
            _ = model(x)

            if i + 1 >= num_batches:
                break

    for h in handles:
        h.remove()

    H_diag_dict = {}
    for name, (diag_sum, count) in H_data.items():
        actual_mod = hook_name_to_submodule[name]
        H_diag_dict[actual_mod] = diag_sum / count

    print(f"Collected Hessians for {len(H_diag_dict)} modules")

    return H_diag_dict

def compute_hessian_blockdiag_direct(x, layer, block_size):
    """
    Compute only the block-diagonal of H = X^T X / N directly.
    Never builds the full [D, D] matrix.
    Returns list of [block_size, block_size] tensors.
    """
    if isinstance(layer, nn.Linear):
        x_flat = x.reshape(-1, x.shape[-1])  # [N, D]
    elif isinstance(layer, nn.Conv2d):
        unfold = nn.Unfold(kernel_size=layer.kernel_size,
                           dilation=layer.dilation,
                           padding=layer.padding,
                           stride=layer.stride)
        x_flat = unfold(x).permute(0, 2, 1).reshape(-1, x.shape[1] *
                         layer.kernel_size[0] * layer.kernel_size[1])
    else:
        return None

    N, D = x_flat.shape
    H_blocks = []

    for i in range(0, D, block_size):
        end = min(i + block_size, D)
        x_block = x_flat[:, i:end]             # [N, bs]
        H_block  = (x_block.T @ x_block) / N  # [bs, bs] — small!
        H_blocks.append(H_block)

    return H_blocks


def solve_alpha_blockwise_Hessian(W, basis, H_diag, mask, block_size):
    """
    Hessian-weighted least squares for alpha, blockwise.
    W, basis, H_diag, mask: [N, M]
    Returns alpha: [N, M]
    """
    N, M = W.shape
    alpha = torch.zeros_like(W)

    for row in range(N):
        W_row = W[row]
        H_row = H_diag[row]
        B_row = basis[row]
        mask_row = mask[row]

        for i in range(0, M, block_size):
            end = min(i + block_size, M)
            w_block = W_row[i:end]
            h_block = H_row[i:end]
            b_block = B_row[i:end]
            m_block = mask_row[i:end]

            num = (h_block * w_block * b_block * m_block).sum()
            den = (h_block * (b_block**2) * m_block).sum() + 1e-8
            alpha_block = num / den
            alpha[row, i:end] = alpha_block

    return alpha

def reconstruct_layer_fp_Hessian(layer, H_diag_layer, block_size, e_bits, m_bits, e_bits_scale, m_bits_scale, device):
    W = layer.weight.data.to(device)
    mask = (W != 0).float()

    # Flatten conv layers
    if W.dim() == 4:
        W_mat = W.view(W.shape[0], -1)
        mask_mat = mask.view(W.shape[0], -1)
        H_mat = H_diag_layer.view(W.shape[0], -1)
    else:
        W_mat = W
        mask_mat = mask
        H_mat = H_diag_layer

    # … rest of FP4 reconstruction …
    sign = torch.sign(W_mat)
    W_abs = W_mat.abs()
    alpha = initialize_alpha(W_abs, mask_mat, block_size)

    for _ in range(5):
        e, m, basis = assign_fp4(W_abs, alpha, e_bits, m_bits)
        alpha = solve_alpha_blockwise_Hessian(W_abs, basis, H_mat, mask_mat, block_size)
        alpha = quantize_scale(alpha, e_bits_scale, m_bits_scale)

    return alpha.view_as(W), e.view_as(W), m.view_as(W), sign.view_as(W)


def reconstruct_layer_fp_blockdiag(
    layer, 
    H_blocks_layer,  # list of [block_size, block_size] Hessians
    block_size, 
    e_bits, 
    m_bits, 
    e_bits_scale, 
    m_bits_scale, 
    device
):
    W = layer.weight.data.to(device)
    mask = (W != 0).float()

    # Flatten conv layers
    if W.dim() == 4:
        W_mat = W.view(W.shape[0], -1)
        mask_mat = mask.view(W.shape[0], -1)
    else:
        W_mat = W
        mask_mat = mask

    # --- magnitude/sign split ---
    sign = torch.sign(W_mat)
    W_abs = W_mat.abs()

    # --- initialize alpha ---
    alpha = initialize_alpha(W_abs, mask_mat, block_size)

    for _ in range(5):
        # --- assign FP4 (exponent + mantissa) ---
        e, m, basis = assign_fp4(W_abs, alpha, e_bits, m_bits)

        # --- block-diagonal alpha update ---
        alpha = solve_alpha_blockwise_Hessian_blockdiag(
            W_abs, basis, H_blocks_layer, mask_mat, block_size
        )

        # --- quantize the scale ---
        alpha = quantize_scale(alpha, e_bits_scale, m_bits_scale)

    # reshape to original weight shape
    alpha = alpha.view_as(W)
    e = e.view_as(W)
    m = m.view_as(W)
    sign = sign.view_as(W)

    return alpha, e, m, sign

def initialize_alpha_safe(w_block, mask_block, k=None):
    """
    Initialize alpha scale for a block of weights.

    Args:
        w_block: 1D tensor of weights (already block-selected)
        mask_block: same shape, 1s for active weights
        k: optional, number of elements to use
    Returns:
        alpha: scalar or tensor per block
    """
    if k is None:
        k = w_block.numel()

    # Only consider nonzero entries
    w_nz = w_block[mask_block > 0]

    if w_nz.numel() == 0:
        return torch.tensor(1.0, device=w_block.device)  # default alpha

    # Simple L2 scale initialization (can replace with your preferred method)
    alpha = torch.sqrt((w_nz ** 2).mean().clamp_min(1e-8))
    return alpha

def initialize_alpha(W_abs, mask, block_size, eps=1e-6,
                     e_bits=4, m_bits=3, mode="percentile"):

    orig_shape = W_abs.shape

    if W_abs.dim() == 4:
        W_mat = W_abs.view(W_abs.shape[0], -1)
        M_mat = mask.view(mask.shape[0], -1)
    else:
        W_mat = W_abs
        M_mat = mask

    alpha = torch.zeros_like(W_mat)

    for row in range(W_mat.shape[0]):  # per output channel
        w_row = W_mat[row]
        m_row = M_mat[row]

        for i in range(0, w_row.numel(), block_size):
            end = min(i + block_size, w_row.numel())

            w_block = w_row[i:end]
            m_block = m_row[i:end]

            vals = w_block[m_block.bool()]

            if vals.numel() == 0:
                alpha_block = eps
            else:
                if mode == "l2":
                    alpha_block = vals.mean()
                elif mode == "l1":
                    alpha_block = vals.abs().mean()
                elif mode == "percentile":
                    alpha_block = torch.quantile(vals, 0.9)
                else:
                    raise ValueError

            alpha_block = quantize_scale(alpha_block, e_bits, m_bits)

            alpha[row, i:end] = alpha_block

    return alpha.view_as(W_abs)

def assign_fp4(W_abs, alpha, E_bits=2, M_bits=1):
    """
    Bit-accurate FP assignment based on exponent and mantissa bits.
    """

    x = W_abs / alpha.clamp_min(1e-8)

    device = W_abs.device

    # ----- Build exponent space -----
    e_levels = torch.arange(0, 2**E_bits, device=device)  # [E]

    # ----- Build mantissa space -----
    m_levels = torch.arange(0, 2**M_bits, device=device)  # [M]

    # ----- Build full codebook -----
    # base = 0.5 * 2^e
    bias = 2**(E_bits - 1) - 1
    base = 2.0 ** (e_levels - bias)
    # base = 0.5 * (2.0 ** e_levels)  # [E]
    # base = 2.0 ** (e_levels - bias)
    # mantissa factor = (1 + m / 2^M)
    mantissa_factor = 1.0 + (m_levels / (2**M_bits))  # [M]

    # Combine → all possible values
    # shape: [E, M]
    codebook = base.unsqueeze(1) * mantissa_factor.unsqueeze(0)

    # Flatten → [K]
    codebook = codebook.view(-1)

    # ----- Assign nearest value -----
    x_expanded = x.unsqueeze(-1)              # [..., 1]
    codebook_expanded = codebook.view(*([1]*x.dim()), -1)

    dist = (x_expanded - codebook_expanded).abs()

    indices = dist.argmin(dim=-1)             # [...]

    # ----- Recover exponent + mantissa -----
    M = 2**M_bits

    exponent = (indices // M)
    mantissa = (indices % M)

    # ----- Reconstruct basis -----
    base_selected = 0.5 * (2.0 ** exponent)
    basis = base_selected * (1.0 + mantissa / (2**M_bits))

    return exponent, mantissa, basis


def assign_fp4_dynamic(W_abs, alpha, E_bits=2, M_bits=1, bias=None):
    """
    Handles both single-row (1D) and batched (2D) input natively.
    alpha can be scalar, [N] or [N,1] — all handled correctly.
    """
    device = W_abs.device
    if bias is None:
        bias = 2**(E_bits - 1) - 1

    # Fix alpha shape — if W_abs is 2D [N, k] and alpha is 1D [N], unsqueeze
    if W_abs.dim() == 2 and alpha.dim() == 1:
        alpha = alpha.unsqueeze(1)  # [N, 1] — broadcasts over k

    # Build codebook
    e_levels = torch.arange(0, 2**E_bits, device=device)
    m_levels = torch.arange(0, 2**M_bits, device=device)
    base     = 2.0 ** (e_levels.float() - bias)
    mant     = 1.0 + m_levels.float() / (2**M_bits)
    codebook = (base.unsqueeze(1) * mant.unsqueeze(0)).reshape(-1)  # [C]
    C        = codebook.shape[0]

    original_shape = W_abs.shape
    x      = (W_abs / alpha.clamp_min(1e-8)).reshape(-1)  # [N*k] flat
    N_flat = x.shape[0]

    # Process in small chunks to keep peak allocation tiny
    chunk_size = 512
    indices    = torch.empty(N_flat, dtype=torch.long, device=device)

    for start in range(0, N_flat, chunk_size):
        end              = min(start + chunk_size, N_flat)
        x_chunk          = x[start:end].unsqueeze(1)   # [chunk, 1]
        cb               = codebook.unsqueeze(0)        # [1, C]
        dist             = (x_chunk - cb).abs()         # [chunk, C]
        indices[start:end] = dist.argmin(dim=-1)
        del x_chunk, cb, dist

    del x, codebook, e_levels, m_levels, base, mant
    torch.cuda.empty_cache()

    M_size   = 2**M_bits
    exponent = (indices // M_size).reshape(original_shape)
    mantissa = (indices %  M_size).reshape(original_shape)
    del indices

    base_sel = 2.0 ** (exponent.float() - bias)
    basis    = base_sel * (1.0 + mantissa.float() / M_size)
    del base_sel

    return exponent, mantissa, basis

def assign_fp4_dynamic_vectorized(W_abs, alpha, E_bits=2, M_bits=1, bias=None):
    """
    Fully shape-consistent FP4 assignment.
    FORCES block format: [N, B, bs]
    """
    device = W_abs.device

    # ============================================================
    # FORCE BLOCK FORMAT
    # ============================================================
    if W_abs.dim() == 2:
        N, M = W_abs.shape
        # Handle cases where alpha might be [N, 1] or [N, B]
        B = alpha.shape[1] if alpha.dim() >= 2 else 1
        bs = M // B
        W_blocks = W_abs.view(N, B, bs)
    else:
        W_blocks = W_abs
        N, B, bs = W_blocks.shape

    # Ensure alpha is [N, B, 1]
    if alpha.dim() == 1: # [N] -> [N, 1, 1]
        alpha = alpha.view(-1, 1, 1)
    elif alpha.dim() == 2: # [N, B] -> [N, B, 1]
        alpha = alpha.unsqueeze(-1)
    elif alpha.dim() == 3 and alpha.shape[-1] != 1: # [N, 1, 1] safety
        alpha = alpha # already likely correct [N, B, 1]

    alpha = alpha.clamp_min(1e-8)
    x = W_blocks / alpha 

    # ============================================================
    # CODEBOOK
    # ============================================================
    e_levels = torch.arange(2**E_bits, device=device)
    m_levels = torch.arange(2**M_bits, device=device)

    mantissa_factor = 1.0 + (m_levels / (2**M_bits))
    codebook = (2.0 ** e_levels.unsqueeze(1)) * mantissa_factor.unsqueeze(0)
    codebook = codebook.reshape(-1)
    K = codebook.numel()

    # ============================================================
    # QUANTIZATION
    # ============================================================
    x_exp = x.unsqueeze(-1) # [N, B, bs, 1]
    cb_exp = codebook.view(1, 1, 1, K)

    dist = (x_exp - cb_exp).abs()
    indices = dist.argmin(dim=-1)

    M_levels_count = 2**M_bits
    exponent = indices // M_levels_count
    mantissa = indices % M_levels_count

    # ============================================================
    # SAFE BIAS BROADCASTING (THE FIX)
    # ============================================================
    if bias is None:
        bias_val = float(2**(E_bits - 1) - 1)
        base_exp = exponent.float() - bias_val
    elif isinstance(bias, (int, float)):
        base_exp = exponent.float() - float(bias)
    else:
        # bias is a tensor. We must match [N, B, bs]
        b_tensor = bias.to(device).float()
        if b_tensor.dim() == 1:
            # [N] -> [N, 1, 1]
            b_tensor = b_tensor.view(-1, 1, 1)
        elif b_tensor.dim() == 2:
            # [N, B] -> [N, B, 1]
            b_tensor = b_tensor.unsqueeze(-1)
        
        base_exp = exponent.float() - b_tensor

    # ============================================================
    # RECONSTRUCTION
    # ============================================================
    base_selected = 2.0 ** base_exp
    basis = base_selected * (1.0 + mantissa.float() / (2**M_bits))

    return exponent, mantissa, basis

# =========================================================
# 🔹 MANTISSA (FINE, mask-aware)
# =========================================================
def solve_mantissa(W_abs, alpha, e, e_bits, m_bits, mask):
    """
    W_abs = |W|  (positive)
    Mantissa bit: minimize |W_abs - alpha * 2^(e-bias) * (1 + m/2^m_bits)|
    For m_bits=1, m in {0, 1}:
      m=0 -> recon = alpha * 2^(e-bias)
      m=1 -> recon = alpha * 2^(e-bias) * 1.5
    Pick whichever is closer to W_abs.
    """
    if m_bits == 0:
        return torch.zeros_like(W_abs, dtype=torch.long)
    bias = 2 ** (e_bits - 1) - 1
    base = alpha * (2.0 ** (e.float() - bias))  # recon with m=0
    # For m_bits=1: candidate values are base and base*1.5
    # Threshold: base * 1.25 (midpoint in linear space... 
    # better: midpoint of log: base * sqrt(1.5) ≈ base * 1.2247)
    threshold = base * (2 ** (1.0 / 2 ** m_bits))  # geometric midpoint
    m = (W_abs > threshold).long()
    m = m * mask.long()
    return m


def reconstruct_fp4(alpha, e, m, sign, e_bits, m_bits):
    """Reconstruct float from FP4 components."""
    bias = 2 ** (e_bits - 1) - 1
    base = alpha * (2.0 ** (e.float() - bias))
    if m_bits > 0:
        W_hat = base * (1.0 + m.float() / (2 ** m_bits))
    else:
        W_hat = base
    return W_hat
# =========================================================
# 🔹 LAYER RECONSTRUCTION (mask-aware, original blockwise)
# =========================================================
import torch
import torch.nn.functional as F

# ========================================================
# Baseline
# ========================================================
def fp4_blockwise_quantize_weights(W, block_size, e_bits=2, m_bits=1,
                                   e_bits_scale=8, m_bits_scale=0):
    # Separate sign and magnitude
    sign = torch.sign(W)
    W_abs = W.abs()
    mask = (W != 0).float()

    # 1) Initialize blockwise alpha (scale)
    alpha = initialize_alpha(W_abs, mask, block_size, mode="percentile")
    alpha = quantize_scale(alpha, e_bits_scale, m_bits_scale)

    # 2) Assign FP4 (E2M1) codebook values for |W|/alpha in one shot
    e, m, basis = assign_fp4(W_abs, alpha, e_bits, m_bits)
    # basis is the *unscaled* positive FP4 grid value (0.5 * 2^e * (1 + m/2^M))

    # 3) Reconstruct magnitudes and apply sign
    W_hat_abs = alpha * basis
    W_hat = sign * W_hat_abs

    return W_hat

def reconstruct_layer_fp_baseline(layer, block_size,
                                  e_bits, m_bits, e_bits_scale, m_bits_scale, device):
    W = layer.weight.data.to(device)
    block_size = get_block_size(layer, block_size, override_conv=isinstance(layer, nn.Conv2d))
    W_q = fp4_blockwise_quantize_weights(W, block_size, e_bits, m_bits,
                                         e_bits_scale, m_bits_scale)
    sign = torch.sign(W_q)
    W_abs = W_q.abs()
    # If you want explicit components back:
    alpha = initialize_alpha(W_abs, (W_q != 0).float(), block_size, mode="percentile")
    alpha = quantize_scale(alpha, e_bits_scale, m_bits_scale)
    e, m, _ = assign_fp4(W_abs, alpha, e_bits, m_bits)
    return alpha, e, m, sign



def reconstruct_layer_fp(layer, data_loader, block_size,
                         e_bits, m_bits, e_bits_scale, m_bits_scale, device, conv_per_out_channel=True):
    W = layer.weight.data.to(device)
    mask = (W != 0).float()

    # --- Separate sign from magnitude ---
    sign = solve_sign(W)
    W_abs = W.abs()
    bias  = 2 ** (e_bits - 1) - 1
    # alpha = torch.ones_like(W_abs)
    block_size = get_block_size(layer, block_size, override_conv=conv_per_out_channel)
    alpha = initialize_alpha(W_abs,mask,block_size,mode="percentile")
    for iteration in range(5):
            # 1. Exponent from |W|/alpha
            # e = solve_exponent(W_abs / (alpha + 1e-8), e_bits, mask)
            e, m, basis = assign_fp4(W_abs, alpha, e_bits, m_bits)
            # 2. Pure power-of-2 basis (no alpha inside)
            # basis = 2.0 ** (e.float() - bias)

            # 3. Alpha solve against pure basis
            alpha = solve_alpha_blockwise(W_abs, basis, mask, block_size)
            alpha = quantize_scale(alpha, e_bits_scale, m_bits_scale)

            # 4. Mantissa
            # m = solve_mantissa(W_abs, alpha, e, e_bits, m_bits, mask)

            # 5. Full unsigned reconstruction
            # W_hat_abs = reconstruct_fp4(alpha, e, m, sign, e_bits, m_bits)
            W_hat_abs = alpha * basis
            # 6. Refine alpha against full reconstruction basis
            full_basis = W_hat_abs / (alpha + 1e-8)  # = basis * (1 + m/2^m_bits)
            alpha_new = solve_alpha_blockwise(W_abs, full_basis, mask, block_size)
            alpha_new = quantize_scale(alpha_new, e_bits_scale, m_bits_scale)
            nonzero = W_abs[mask.bool()]
            print(f"  iter {iteration}: "
                f"W_abs[:5]={(sign.flatten()[:5]*W_abs.flatten()[:5]).tolist()} "
                f"basis[:5]={basis.flatten()[:5].tolist()} "
                f"alpha_coarse={alpha.flatten()[0].item():.6f} "
                f"alpha_refined={alpha_new.flatten()[0].item():.6f} "
                f"W_hat[:5]={W_hat_abs.flatten()[:5].tolist()}"
                f"exponent[:5]={e.flatten()[:5].tolist()}"
                f"mantissa[:5]={m.flatten()[:5].tolist()}")
            alpha = alpha_new
    return alpha, e, m, sign   # sign returned separately


# Compute min/max representable scale
def compute_alpha_bounds(e_bits_scale, m_bits_scale, device='cuda'):
    # representable min/max of scale
    alpha_min = 2.0 ** -(2 ** (e_bits_scale - 1))
    alpha_max = 2.0 ** ((2 ** (e_bits_scale - 1)) - 1) * (1.0 + (2 ** m_bits_scale - 1) / (2 ** m_bits_scale))
    return torch.tensor(alpha_min, device=device, dtype=torch.float32), \
           torch.tensor(alpha_max, device=device, dtype=torch.float32)

def solve_alpha_blockwise_HG(W, H, G, mask, block_size, e_bits_scale, m_bits_scale):
    """
    HG-style alpha update, mask-aware, blockwise.
    W, H, G, mask: [N, M] 2D tensors
    Returns: alpha [N, M]
    """
    N, M = W.shape
    alpha = torch.zeros_like(W, device=W.device)

    # FP-format bounds
    alpha_min = 2.0 ** (-(2**(e_bits_scale - 1) - 1))
    alpha_max = 2.0 ** (2**(e_bits_scale - 1) - 1)

    for row in range(N):
        W_row = W[row]
        H_row = H[row]
        G_row = G[row]
        mask_row = mask[row]

        for i in range(0, M, block_size):
            end = min(i + block_size, M)

            W_block = W_row[i:end]
            HG_block = H_row[i:end] + G_row[i:end]
            mask_block = mask_row[i:end]

            # masked least squares solution
            num = (HG_block * W_block * mask_block).sum()
            den = ((HG_block**2) * mask_block).sum() + 1e-8
            alpha_block = num / den

            # clamp to FP representable range
            alpha_block = torch.clamp(alpha_block, alpha_min, alpha_max)

            alpha[row, i:end] = alpha_block

    return alpha

def reconstruct_layer_fp_HG(layer,
                            data_loader,
                            block_size,
                            e_bits,
                            m_bits,
                            e_bits_scale,
                            m_bits_scale,
                            device):
    """
    HG-style reconstruction for a layer (Linear or Conv2d), mask-aware.
    Returns: alpha (per block scale), e (exponent), m (mantissa)
    """

    # get original weights
    W = layer.weight.data.to(device)
    mask = (W != 0).float()  # mask for pruned weights

    # Flatten Conv2d to 2D: [out_channels, in_channels * kH * kW]
    if W.dim() == 4:
        W_mat = W.view(W.shape[0], -1)
        mask_mat = mask.view(W.shape[0], -1)
    else:
        W_mat = W
        mask_mat = mask

    # initialize alpha, H, G
    alpha = torch.ones_like(W_mat)
    H = torch.zeros_like(W_mat)
    G = torch.zeros_like(W_mat)

    for _ in range(3):  # small alternating loop

        # --- HG blockwise alpha update ---
        alpha = solve_alpha_blockwise_HG(W_mat, H, G, mask_mat, block_size, e_bits_scale, m_bits_scale)
        alpha = quantize_scale(alpha, e_bits_scale, m_bits_scale)

        # --- coarse: exponent solve ---
        e = solve_exponent(W_mat, alpha, e_bits, mask_mat)

        bias = 2**(e_bits - 1) - 1
        W_coarse = alpha * (2.0 ** (e.float() - bias))

        # --- fine: mantissa solve ---
        m = solve_mantissa(W_mat, alpha, e, e_bits, m_bits, mask_mat)

        if m_bits > 0:
            W_hat = W_coarse + W_coarse * m.float() / (2**m_bits - 1)
        else:
            W_hat = W_coarse

        # --- update H and G for next HG iteration ---
        H = W_coarse
        G = W_hat - W_coarse

        # --- refine alpha again using HG ---
        alpha = solve_alpha_blockwise_HG(W_mat, H, G, mask_mat, block_size, e_bits_scale, m_bits_scale)
        alpha = quantize_scale(alpha, e_bits_scale, m_bits_scale)

    # reshape alpha, e, m back to original weight shape
    if W.dim() == 4:
        alpha = alpha.view_as(W)
        e = e.view_as(W)
        m = m.view_as(W)

    return alpha, e, m

# =========================================================
# 🔹 NEW: GPTQ-STYLE ROW-WISE FP4 QUANTIZATION
# =========================================================
def reconstruct_layer_fp_rowwise_hessian(
    layer,
    H_diag_layer,
    block_size,
    e_bits,
    m_bits,
    e_bits_scale,
    m_bits_scale,
    device
):
    """
    Row-wise GPTQ-style FP4 reconstruction using Hessian.
    Args:
        layer: nn.Module layer (Linear or Conv2d)
        H_diag_layer: Hessian diag [N, M]
        block_size: block size for alpha sharing
        e_bits, m_bits: FP4 exponent/mantissa bits
        e_bits_scale, m_bits_scale: scale quantization bits
        device: device
    Returns:
        alpha, e, m, sign (all reshaped to layer weight shape)
    """
    W = layer.weight.data.to(device)
    mask = (W != 0).float()

    if W.dim() == 4:  # Conv2d flatten
        W_mat = W.view(W.shape[0], -1)
        mask_mat = mask.view(W.shape[0], -1)
        H_mat = H_diag_layer.view(W.shape[0], -1)
    else:
        W_mat = W
        mask_mat = mask
        H_mat = H_diag_layer

    N, M = W_mat.shape
    sign = torch.sign(W_mat)
    W_abs = W_mat.abs()

    # Initialize alpha per block
    alpha = initialize_alpha(W_abs, mask_mat, block_size, mode="percentile")
    alpha = quantize_scale(alpha, e_bits_scale, m_bits_scale)

    # --- Row-wise GPTQ-style loop ---
    for row in range(N):
        w_row = W_abs[row]
        h_row = H_mat[row]
        mask_row = mask_mat[row]
        alpha_row = alpha[row]

        # --- traverse blocks in the row ---
        for i in range(0, M, block_size):
            end = min(i + block_size, M)

            w_block = w_row[i:end]
            h_block = h_row[i:end]
            m_block = mask_row[i:end]
            a_block = alpha_row[i:end]

            # --- Coarse pass: exponent ---
            e_block, _, basis_block = assign_fp4(w_block, a_block, E_bits=e_bits, M_bits=0)

            # --- Fine pass: mantissa ---
            if m_bits > 0:
                m_block_vals = solve_mantissa(w_block, a_block, e_block, e_bits, m_bits, m_block)
            else:
                m_block_vals = torch.zeros_like(w_block, dtype=torch.long)

            # --- Compute basis for alpha update ---
            basis_full = reconstruct_fp4(a_block, e_block, m_block_vals, torch.ones_like(w_block), e_bits, m_bits) / (a_block + 1e-8)

            # --- Hessian-weighted alpha update ---
            num = (h_block * w_block * basis_full * m_block).sum()
            den = (h_block * (basis_full**2) * m_block).sum() + 1e-8
            alpha_block_new = num / den
            alpha_row[i:end] = alpha_block_new

            # Quantize alpha to FP format
            alpha_row[i:end] = quantize_scale(alpha_row[i:end], e_bits_scale, m_bits_scale)

        # Save updated row
        alpha[row] = alpha_row

    # --- Final full-row assignment ---
    e, m, _ = assign_fp4(W_abs, alpha, E_bits=e_bits, M_bits=m_bits)
    return alpha.view_as(W), e.view_as(W), m.view_as(W), sign.view_as(W)


import torch

def reconstruct_block_fp4_pipeline(layer, H_diag_layer, block_size,
                                   num_iters=3, e_bits=3, m_bits=3,
                                   e_bits_scale=8, m_bits_scale=0,
                                   device='cuda'):
    """
    Blockwise FP4 reconstruction (GPTQ-style) integrated into the current pipeline.

    Args:
        layer: nn.Module (Conv2d or Linear) with .weight
        H_diag_layer: Hessian diagonal tensor of same shape as weight
        block_size: number of weights per block
        num_iters: alternating α + exponent updates
        e_bits, m_bits: FP4 exponent/mantissa bits
        e_bits_scale, m_bits_scale: FP quantization for α
        device: computation device

    Returns:
        alpha: per-block scale
        e: exponent tensor
        m: mantissa tensor
        sign: sign of weights
    """

    # --- Flatten weights and mask ---
    W = layer.weight.data.to(device)
    mask = (W != 0).float()
    sign = torch.sign(W)
    W_abs = W.abs()

    if W.dim() == 4:  # Conv2d
        W_flat = W_abs.view(W.shape[0], -1)
        mask_flat = mask.view(W.shape[0], -1)
        H_flat = H_diag_layer.view(W.shape[0], -1)
    else:
        W_flat = W_abs
        mask_flat = mask
        H_flat = H_diag_layer

    N, M = W_flat.shape
    alpha = torch.ones_like(W_flat, device=device)
    e = torch.zeros_like(W_flat, dtype=torch.long, device=device)
    m = torch.ones_like(W_flat, dtype=torch.long, device=device)

    bias = 2 ** (e_bits - 1) - 1
    m_max = 2**m_bits - 1

    # --- Blockwise iteration ---
    for row in range(N):
        for i in range(0, M, block_size):
            end = min(i + block_size, M)
            w_block = W_flat[row, i:end]
            h_block = H_flat[row, i:end]
            mask_block = mask_flat[row, i:end]

            if mask_block.sum() < 1e-8:
                continue

            # Initial exponent guess
            e_block = torch.round(torch.log2(w_block.abs().max() + 1e-12)).long()
            alpha_block = 1.0
            m_block = torch.ones_like(w_block)

            for it in range(num_iters):
                # --- Step 1: Update α & mantissa for fixed exponent ---
                scale = 2.0 ** (e_block - bias)
                m_block = torch.clamp((w_block / scale).round(), 1, m_max)
                numerator = (h_block * w_block * m_block * scale).sum()
                denominator = (h_block * (m_block * scale)**2).sum() + 1e-12
                alpha_block = numerator / denominator
                alpha_block = torch.clamp(alpha_block, 1e-6, 1e6)  # optional FP clamp

                # --- Step 2: Exponent search around e_block ---
                best_err = float('inf')
                best_e = e_block.clone()
                for shift in range(-2, 3):  # small range search
                    e_try = e_block + shift
                    scale_try = 2.0 ** (e_try - bias)
                    m_try = torch.clamp((w_block / scale_try).round(), 1, m_max)
                    W_try = alpha_block * m_try * scale_try
                    err = (h_block * (W_try - w_block)**2).sum()
                    if err < best_err:
                        best_err = err
                        best_e = e_try
                e_block = best_e

            # --- Write final block ---
            scale = 2.0 ** (e_block - bias)
            m_block = torch.clamp((w_block / scale).round(), 1, m_max)
            alpha[row, i:end] = alpha_block
            e[row, i:end] = e_block
            m[row, i:end] = m_block

    # --- Reshape back to original weight shape ---
    alpha = alpha.view_as(W)
    e = e.view_as(W)
    m = m.view_as(W)
    sign = sign.view_as(W)

    return alpha, e, m, sign



#=====================================================
# Adaptive mesh method
#=====================================================

def hessian_block_whiten(w_block, H_block, eps=1e-6):
    """
    Transform w -> z = H^{1/2} w

    Returns:
        z_block
        eigvecs
        sqrt_L
        inv_sqrt_L
    """
    # Eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(H_block)

    eigvals = torch.clamp(eigvals, min=eps)

    sqrt_L = torch.sqrt(eigvals)
    inv_sqrt_L = 1.0 / sqrt_L

    # Transform
    z_block = eigvecs.T @ w_block
    z_block = z_block * sqrt_L

    return z_block, eigvecs, sqrt_L, inv_sqrt_L

def hessian_block_unwhiten(z_block, eigvecs, inv_sqrt_L):
    """
    Transform back: w = H^{-1/2} z
    """
    w_block = eigvecs @ (z_block * inv_sqrt_L)
    return w_block


def quantize_block_fp4_whitened(
    w_block,
    H_block,
    fp4_quant_block_fn,
    eps=1e-6
):
    """
    Apply FP4 quantization in Hessian-whitened space.
    """

    # --- Step 1: whiten ---
    z_block, eigvecs, sqrt_L, inv_sqrt_L = hessian_block_whiten(
        w_block, H_block, eps
    )

    # --- Step 2: quantize in isotropic space ---
    z_q = fp4_quant_block_fn(z_block)

    # --- Step 3: map back ---
    w_q = hessian_block_unwhiten(z_q, eigvecs, inv_sqrt_L)

    return w_q



def reconstruct_layer_fp_blockdiag_whitened(
    layer,
    H_blocks_layer,
    block_size,
    e_bits,
    m_bits,
    e_bits_scale,
    m_bits_scale,
    device
):
    W = layer.weight.data.to(device)
    mask = (W != 0).float()

    # Flatten
    if W.dim() == 4:
        W_mat = W.view(W.shape[0], -1)
        mask_mat = mask.view(W.shape[0], -1)
    else:
        W_mat = W
        mask_mat = mask

    N, M = W_mat.shape
    W_q = torch.zeros_like(W_mat)

    for row in range(N):
        w_row = W_mat[row]
        m_row = mask_mat[row]

        block_idx = 0

        for i in range(0, M, block_size):
            end = min(i + block_size, M)

            w_block = w_row[i:end]
            mask_block = m_row[i:end]

            if mask_block.sum() < 1e-8:
                continue

            H_block = H_blocks_layer[block_idx].to(device)
            ## Note the lines before the definition were added to deal with size mismatch
            k = w_block.numel()

            if H_block.shape[0] != k:
                H_block = H_block[:k, :k]

            # --- define FP4 quantizer using YOUR pipeline ---
            def fp4_quant_block(z_block):
                z_block = z_block.unsqueeze(0)  # match [1, k]

                mask_local = torch.ones_like(z_block)

                alpha = initialize_alpha(z_block.abs(), mask_local, z_block.shape[1])

                for _ in range(3):
                    e, m, basis = assign_fp4(z_block.abs(), alpha, e_bits, m_bits)
                    alpha = solve_alpha_blockwise(z_block.abs(), basis, mask_local, z_block.shape[1])
                    alpha = quantize_scale(alpha, e_bits_scale, m_bits_scale)

                z_hat = alpha * basis
                return z_hat.squeeze(0)

            # --- apply whitening-based quantization ---
            w_q_block = quantize_block_fp4_whitened(
                w_block, H_block, fp4_quant_block
            )

            W_q[row, i:end] = w_q_block

            block_idx += 1

    # reshape back
    if W.dim() == 4:
        W_q = W_q.view_as(W)

    sign = torch.sign(W_q)
    W_abs = W_q.abs()

    alpha = initialize_alpha(W_abs, (W_q != 0).float(), block_size)
    alpha = quantize_scale(alpha, e_bits_scale, m_bits_scale)
    e, m, _ = assign_fp4(W_abs, alpha, e_bits, m_bits)

    return alpha, e, m, sign



def hessian_block_scale_diag(w_block, H_block, eps=1e-6, max_scale=10.0):
    """
    Diagonal Hessian scaling (stable, no rotation).
    """
    diag = torch.diag(H_block)
    diag = torch.clamp(diag, min=eps)

    scale = torch.sqrt(diag)

    # Prevent explosion
    scale = torch.clamp(scale, max=max_scale)

    inv_scale = 1.0 / scale

    z_block = w_block * scale

    return z_block, scale, inv_scale


def hessian_block_unscale_diag(z_block, inv_scale):
    return z_block * inv_scale


def quantize_block_fp4_scaled(
    w_block,
    H_block,
    fp4_quant_block_fn,
    eps=1e-6
):
    """
    Diagonal Hessian-aware FP4 quantization (sign-preserving).
    """

    # --- scale ---
    z_block, scale, inv_scale = hessian_block_scale_diag(
        w_block, H_block, eps
    )

    # --- split sign ---
    sign_z = torch.sign(z_block)
    z_abs = z_block.abs()

    # --- quantize magnitude ONLY ---
    z_q_abs = fp4_quant_block_fn(z_abs)

    # --- restore sign ---
    z_q = sign_z * z_q_abs

    # --- unscale ---
    w_q = hessian_block_unscale_diag(z_q, inv_scale)

    # 🔴 HARD sign constraint (critical)
    w_q = torch.sign(w_block) * w_q.abs()

    return w_q


def reconstruct_layer_fp_blockdiag_scaled(
    layer,
    H_blocks_layer,
    block_size,
    e_bits,
    m_bits,
    e_bits_scale,
    m_bits_scale,
    device
):
    """
    Blockwise FP4 reconstruction with correct Hessian-aware alpha solve.

    Key properties:
    - Full block geometry (no nz compression)
    - Mask applied multiplicatively (not structurally)
    - Blockwise scalar alpha
    - 2-step fixed-point refinement
    - No destructive post-recompute

    Returns:
        alpha, e, m, sign (same shape as weights)
    """

    W = layer.weight.data.to(device)
    mask = (W != 0).float()

    # Flatten
    if W.dim() == 4:
        W_mat = W.view(W.shape[0], -1)
        mask_mat = mask.view(W.shape[0], -1)
    else:
        W_mat = W
        mask_mat = mask

    N, M = W_mat.shape

    sign = torch.sign(W_mat)
    W_abs = W_mat.abs()

    # Outputs
    W_q = torch.zeros_like(W_mat)
    alpha_out = torch.zeros_like(W_mat)

    # --- Blockwise reconstruction ---
    for row in range(N):
        w_row = W_abs[row]
        m_row = mask_mat[row]

        block_idx = 0

        for i in range(0, M, block_size):
            end = min(i + block_size, M)

            w_block = w_row[i:end]
            m_block = m_row[i:end]

            H_block = H_blocks_layer[block_idx].to(device)

            k = w_block.numel()
            if H_block.shape[0] != k:
                H_block = H_block[:k, :k]

            # --- fully pruned block ---
            if m_block.sum() < 1e-8:
                alpha_block = torch.tensor(1.0, device=device)
                basis_block = torch.zeros_like(w_block)

                W_q[row, i:end] = 0.0
                alpha_out[row, i:end] = alpha_block

                block_idx += 1
                continue

            # --- initialize alpha (robust scale) ---
            alpha_block = torch.sqrt((w_block[m_block > 0] ** 2).mean())
            alpha_block = torch.clamp(alpha_block, min=1e-4)

            # --- fixed-point refinement (critical) ---
            for _ in range(2):
                # assign FP4
                e_block, m_block_vals, basis_block = assign_fp4(
                    w_block, alpha_block, e_bits, m_bits
                )

                # apply mask (preserve geometry)
                w_eff = w_block * m_block
                b_eff = basis_block * m_block

                # quadratic solve: alpha = (b^T H w) / (b^T H b)
                Hb = H_block @ b_eff
                Hw = H_block @ w_eff

                num = (b_eff * Hw).sum()
                den = (b_eff * Hb).sum() + 1e-8

                alpha_block = num / den

                # prevent collapse
                alpha_block = torch.clamp(alpha_block, min=1e-6)

                # quantize scale
                alpha_block = quantize_scale(
                    alpha_block, e_bits_scale, m_bits_scale
                )

            # --- final quantization ---
            e_block, m_block_vals, basis_block = assign_fp4(
                w_block, alpha_block, e_bits, m_bits
            )

            w_hat = alpha_block * basis_block

            # restore sign
            w_hat = w_hat * sign[row, i:end]

            W_q[row, i:end] = w_hat
            alpha_out[row, i:end] = alpha_block

            block_idx += 1

    # reshape back
    if W.dim() == 4:
        W_q = W_q.view_as(W)
        alpha_out = alpha_out.view_as(W)

    # final FP decomposition (for storage)
    sign = torch.sign(W_q)
    W_abs = W_q.abs()

    e, m, _ = assign_fp4(W_abs, alpha_out, e_bits, m_bits)

    return alpha_out, e.view_as(W), m.view_as(W), sign.view_as(W)

def reconstruct_layer_fp_blockdiag_scaled_v2(
    layer,
    H_blocks_layer,
    block_size,
    e_bits,
    m_bits,
    e_bits_scale,
    m_bits_scale,
    device
):
    """
    Activation-aware FP4 reconstruction (fixed version).

    Fixes:
    - No masking inside optimization
    - Proper Hessian quadratic solve
    - Delayed alpha quantization
    - Collapse prevention
    """

    W = layer.weight.data.to(device)
    mask = (W != 0).float()

    # Flatten
    if W.dim() == 4:
        W_mat = W.view(W.shape[0], -1)
        mask_mat = mask.view(W.shape[0], -1)
    else:
        W_mat = W
        mask_mat = mask

    N, M = W_mat.shape

    sign = torch.sign(W_mat)
    W_abs = W_mat.abs()

    # Outputs
    W_q = torch.zeros_like(W_mat)
    alpha_out = torch.zeros_like(W_mat)

    # --- Blockwise reconstruction ---
    for row in range(N):
        w_row = W_abs[row]
        m_row = mask_mat[row]

        block_idx = 0

        for i in range(0, M, block_size):
            end = min(i + block_size, M)

            w_block = w_row[i:end]
            m_block = m_row[i:end]

            H_block = H_blocks_layer[block_idx].to(device)

            k = w_block.numel()
            if H_block.shape[0] != k:
                H_block = H_block[:k, :k]

            # --- fully pruned block ---
            if m_block.sum() < 1e-8:
                alpha_block = torch.tensor(1.0, device=device)

                W_q[row, i:end] = 0.0
                alpha_out[row, i:end] = alpha_block

                block_idx += 1
                continue

            # --- initialize alpha (robust RMS) ---
            alpha_block = torch.sqrt((w_block ** 2).mean())
            alpha_block = torch.clamp(alpha_block, min=1e-4)

            # --- fixed-point refinement ---
            for _ in range(2):
                # FP4 assignment
                e_block, m_block_vals, basis_block = assign_fp4(
                    w_block, alpha_block, e_bits, m_bits
                )

                # 🚨 NO MASK HERE
                b = basis_block
                w = w_block

                # Hessian solve
                Hb = H_block @ b
                Hw = H_block @ w

                num = (b * Hw).sum()
                den = (b * Hb).sum() + 1e-8

                alpha_new = num / den

                # --- stabilization ---
                # prevent collapse relative to block energy
                alpha_min = 0.05 * w_block.abs().mean()
                alpha_block = torch.clamp(alpha_new, min=alpha_min)

            # --- quantize alpha AFTER convergence ---
            alpha_block = quantize_scale(
                alpha_block, e_bits_scale, m_bits_scale
            )

            # --- final quantization ---
            e_block, m_block_vals, basis_block = assign_fp4(
                w_block, alpha_block, e_bits, m_bits
            )

            w_hat = alpha_block * basis_block

            # ✅ APPLY MASK ONLY HERE
            w_hat = w_hat * m_block

            # restore sign
            w_hat = w_hat * sign[row, i:end]

            W_q[row, i:end] = w_hat
            alpha_out[row, i:end] = alpha_block

            block_idx += 1

    # reshape back
    if W.dim() == 4:
        W_q = W_q.view_as(W)
        alpha_out = alpha_out.view_as(W)

    # final FP decomposition
    sign = torch.sign(W_q)
    W_abs = W_q.abs()

    e, m, _ = assign_fp4(W_abs, alpha_out, e_bits, m_bits)

    return alpha_out, e.view_as(W), m.view_as(W), sign.view_as(W)


def reconstruct_layer_fp_blockdiag_scaled_v3(
    layer,
    H_blocks_layer,
    block_size,
    e_bits,
    m_bits,
    e_bits_scale,
    m_bits_scale,
    device
):
    W = layer.weight.data.to(device)
    mask = (W != 0).float()

    # Flatten
    if W.dim() == 4:
        W_mat = W.view(W.shape[0], -1)
        mask_mat = mask.view(W.shape[0], -1)
    else:
        W_mat = W
        mask_mat = mask

    N, M = W_mat.shape

    sign = torch.sign(W_mat)
    W_abs = W_mat.abs()

    W_q = torch.zeros_like(W_mat)
    alpha_out = torch.zeros_like(W_mat)

    for row in range(N):
        w_row = W_abs[row]
        m_row = mask_mat[row]

        block_idx = 0

        for i in range(0, M, block_size):
            end = min(i + block_size, M)

            w_block = w_row[i:end]
            m_block = m_row[i:end]

            H_block = H_blocks_layer[block_idx].to(device)

            k = w_block.numel()
            if H_block.shape[0] != k:
                H_block = H_block[:k, :k]

            # --- fully pruned ---
            if m_block.sum() < 1e-8:
                alpha_block = torch.tensor(1.0, device=device)
                W_q[row, i:end] = 0.0
                alpha_out[row, i:end] = alpha_block
                block_idx += 1
                continue

            # --- init alpha ---
            alpha_block = torch.sqrt((w_block ** 2).mean())
            alpha_block = torch.clamp(alpha_block, min=1e-4)

            # --- fixed-point refinement ---
            for _ in range(3):

                _, _, b = assign_fp4(w_block, alpha_block, e_bits, m_bits)

                # apply mask ONLY to weights (not structure)
                w_eff = w_block * m_block
                b_eff = b * m_block

                # --- CORRECT quadratic form ---
                Hb = H_block @ b_eff
                Hw = H_block @ w_eff

                num = torch.dot(b_eff, Hw)
                den = torch.dot(b_eff, Hb) + 1e-8

                alpha_new = num / den

                # stabilization (CRITICAL)
                alpha_min = 0.1 * w_block.abs().mean()
                alpha_max = 10.0 * w_block.abs().mean()

                alpha_block = torch.clamp(alpha_new, min=alpha_min, max=alpha_max)

            # quantize AFTER convergence
            alpha_block = quantize_scale(alpha_block, e_bits_scale, m_bits_scale)

            # final projection
            _, _, b = assign_fp4(w_block, alpha_block, e_bits, m_bits)
            w_hat = alpha_block * b

            # apply mask at end
            w_hat = w_hat * m_block
            w_hat = w_hat * sign[row, i:end]

            W_q[row, i:end] = w_hat
            alpha_out[row, i:end] = alpha_block

            block_idx += 1

    if W.dim() == 4:
        W_q = W_q.view_as(W)
        alpha_out = alpha_out.view_as(W)

    sign = torch.sign(W_q)
    W_abs = W_q.abs()

    e, m, _ = assign_fp4(W_abs, alpha_out, e_bits, m_bits)

    return alpha_out, e.view_as(W), m.view_as(W), sign.view_as(W)





def reconstruct_layer_fp_blockdiag_scaled_v4(
    layer,
    H_blocks_layer,
    block_size,
    e_bits,
    m_bits,
    e_bits_scale,
    m_bits_scale,
    device
):
    """
    FINAL VERSION — Consistent adaptive-mesh FP4 reconstruction

    Features:
    - Hessian-aware alpha optimization
    - Per-block exponent bias search
    - Alpha per block (shared)
    - Bias per block (shared)
    - NO re-quantization mismatch
    - Stable optimization

    Returns:
        alpha_out, e, m, sign, bias_out
    """

    W = layer.weight.data.to(device)
    mask = (W != 0).float()

    # Flatten
    if W.dim() == 4:
        W_mat = W.view(W.shape[0], -1)
        mask_mat = mask.view(W.shape[0], -1)
    else:
        W_mat = W
        mask_mat = mask

    N, M = W_mat.shape

    sign = torch.sign(W_mat)
    W_abs = W_mat.abs()

    # Outputs
    W_q = torch.zeros_like(W_mat)
    alpha_out = torch.zeros_like(W_mat)
    bias_out = torch.zeros_like(W_mat)

    for row in range(N):
        w_row = W_abs[row]
        m_row = mask_mat[row]

        block_idx = 0

        for i in range(0, M, block_size):
            end = min(i + block_size, M)

            w_block = w_row[i:end]
            m_block = m_row[i:end]

            H_block = H_blocks_layer[block_idx].to(device)

            k = w_block.numel()
            if H_block.shape[0] != k:
                H_block = H_block[:k, :k]

            # =========================
            # Fully pruned block
            # =========================
            if m_block.sum() < 1e-8:
                alpha_block = torch.tensor(1.0, device=device)
                bias_block = 0

                W_q[row, i:end] = 0.0
                alpha_out[row, i:end] = alpha_block
                bias_out[row, i:end] = bias_block

                block_idx += 1
                continue

            # =========================
            # Initialization
            # =========================
            w_eff = w_block * m_block

            alpha_init = torch.sqrt((w_eff ** 2).mean()).clamp(min=1e-4)

            default_bias = 2**(e_bits - 1) - 1
            bias_radius = max(1, 2**(e_bits - 2))  # adaptive search window

            best_loss = float('inf')
            best_alpha = None
            best_bias = None
            best_b = None

            # =========================
            # Bias search loop
            # =========================

            Hw = H_block @ w_eff
            for bias_candidate in range(default_bias - bias_radius,
                                        default_bias + bias_radius + 1):

                alpha_tmp = alpha_init.clone()

                # Fixed-point refinement
                for _ in range(5):
                    _, _, b = assign_fp4_dynamic(
                        w_block,
                        alpha_tmp,
                        e_bits,
                        m_bits,
                        bias=bias_candidate
                    )


                    b_eff = b * m_block

                    Hb = H_block @ b_eff
                    num = torch.dot(b_eff, Hw)
                    den = torch.dot(b_eff, Hb) + 1e-8

                    alpha_new = num / den

                    # Stabilization
                    alpha_min = 0.05 * w_eff.abs().mean()
                    alpha_max = 20.0 * w_eff.abs().mean()

                    alpha_tmp = torch.clamp(alpha_new, min=alpha_min, max=alpha_max)

                # Evaluate quadratic loss
                residual = w_eff - alpha_tmp * b_eff
                loss = torch.dot(residual, H_block @ residual)

                if loss < best_loss:
                    best_loss = loss
                    best_alpha = alpha_tmp
                    best_bias = bias_candidate
                    best_b = b

            # =========================
            # Final alpha quantization
            # =========================
            alpha_block = quantize_scale(
                best_alpha, e_bits_scale, m_bits_scale
            )

            # =========================
            # Final basis recompute
            # =========================
            _, _, b_final = assign_fp4_dynamic(
                w_block,
                alpha_block,
                e_bits,
                m_bits,
                bias=best_bias
            )

            w_hat = alpha_block * b_final

            # Apply mask + sign
            w_hat = w_hat * m_block
            w_hat = w_hat * sign[row, i:end]

            # Store
            W_q[row, i:end] = w_hat
            alpha_out[row, i:end] = alpha_block
            bias_out[row, i:end] = best_bias

            block_idx += 1

    # =========================
    # Reshape back
    # =========================
    if W.dim() == 4:
        W_q = W_q.view_as(W)
        alpha_out = alpha_out.view_as(W)
        bias_out = bias_out.view_as(W)

    # =========================
    # Final FP decomposition (CONSISTENT)
    # =========================
    sign = torch.sign(W_q)
    W_abs = W_q.abs()

    if W.dim() == 4:
        W_mat = W_abs.view(W.shape[0], -1)
        alpha_mat = alpha_out.view(W.shape[0], -1)
        bias_mat = bias_out.view(W.shape[0], -1)
    else:
        W_mat = W_abs
        alpha_mat = alpha_out
        bias_mat = bias_out

    e = torch.zeros_like(W_mat)
    m = torch.zeros_like(W_mat)

    for row in range(N):
        for i in range(0, M, block_size):
            end = min(i + block_size, M)

            bias_block = int(bias_mat[row, i].item())
            alpha_block = alpha_mat[row, i]

            w_block = W_mat[row, i:end]

            e_block, m_block, _ = assign_fp4_dynamic(
                w_block,
                alpha_block,
                e_bits,
                m_bits,
                bias=bias_block
            )

            e[row, i:end] = e_block
            m[row, i:end] = m_block

    if W.dim() == 4:
        e = e.view_as(W)
        m = m.view_as(W)

    return alpha_out, e, m, sign, bias_out

import functools

def assign_fp4_dynamic_batched(w_block, alpha, e_bits, m_bits, bias=None, bias_per_row=None):
    """
    Batched FP4 assignment — no vmap, processes rows in chunks to avoid OOM.
    Numerically identical to the vmap version.
    """
    if bias_per_row is not None:
        N, k   = w_block.shape
        device = w_block.device
        e_out  = torch.zeros(N, k, dtype=torch.long, device=device)
        m_out  = torch.zeros(N, k, dtype=torch.long, device=device)
        b_out  = torch.zeros(N, k, device=device)

        for bias_val in bias_per_row.unique():
            rows = (bias_per_row == bias_val).nonzero(as_tuple=True)[0]
            bv   = int(bias_val.item())
            e_b, m_b, b_b = assign_fp4_dynamic(
                w_block[rows], alpha[rows], e_bits, m_bits, bias=bv)
            e_out[rows] = e_b
            m_out[rows] = m_b
            b_out[rows] = b_b
            del e_b, m_b, b_b
            torch.cuda.empty_cache()

        return e_out, m_out, b_out
    else:
        return assign_fp4_dynamic(
            w_block, alpha, e_bits, m_bits, bias=bias)


# def reconstruct_layer_fp_blockdiag_scaled_v5(
#     layer,
#     H_blocks_layer,
#     block_size,
#     e_bits,
#     m_bits,
#     e_bits_scale,
#     m_bits_scale,
#     device
# ):
#     W = layer.weight.data.to(device)
#     mask = (W != 0).float()

#     if W.dim() == 4:
#         W_mat = W.view(W.shape[0], -1)
#         mask_mat = mask.view(W.shape[0], -1)
#     else:
#         W_mat = W
#         mask_mat = mask

#     N, M = W_mat.shape

#     sign_mat = torch.sign(W_mat)
#     W_abs    = W_mat.abs()

#     # Free W and mask early
#     del W, mask
#     torch.cuda.empty_cache()

#     # All output tensors kept on CPU to save VRAM
#     W_q       = torch.zeros(N, M, dtype=torch.float32)   # CPU
#     alpha_out = torch.zeros(N, M, dtype=torch.float32)   # CPU
#     bias_out  = torch.zeros(N, M, dtype=torch.float32)   # CPU

#     default_bias    = 2 ** (e_bits - 1) - 1
#     bias_radius     = max(1, 2 ** (e_bits - 2))
#     bias_candidates = list(range(default_bias - bias_radius,
#                                  default_bias + bias_radius + 1))

#     # ----------------------------------------------------------------
#     # First pass — block-wise alpha/bias optimisation
#     # ----------------------------------------------------------------
#     for block_idx, i in enumerate(range(0, M, block_size)):
#         end = min(i + block_size, M)
#         k   = end - i

#         w_block = W_abs[:, i:end]
#         m_block = mask_mat[:, i:end]
#         s_block = sign_mat[:, i:end]

#         H_block = H_blocks_layer[block_idx].to(device)
#         if H_block.shape[0] != k:
#             H_block = H_block[:k, :k]

#         w_eff  = w_block * m_block
#         pruned = m_block.sum(dim=1) < 1e-8

#         w_sq_mean = (w_eff ** 2).mean(dim=1).clamp(min=1e-8)
#         alpha     = torch.sqrt(w_sq_mean).clamp(min=1e-4)
#         alpha_min = 0.05 * w_eff.abs().mean(dim=1).clamp(min=1e-8)
#         alpha_max = 20.0 * w_eff.abs().mean(dim=1).clamp(min=1e-8)

#         Hw = w_eff @ H_block.T

#         best_loss  = torch.full((N,), float('inf'), device=device)
#         best_alpha = alpha.clone()
#         best_bias  = torch.full((N,), default_bias, device=device, dtype=torch.long)
#         best_b     = torch.zeros_like(w_eff)

#         for bias_candidate in bias_candidates:
#             alpha_tmp = alpha.clone()

#             for _ in range(5):
#                 _, _, b = assign_fp4_dynamic_batched(
#                     w_block, alpha_tmp, e_bits, m_bits, bias=bias_candidate)
#                 b_eff = b * m_block
#                 Hb    = b_eff @ H_block.T
#                 num   = (b_eff * Hw).sum(dim=1)
#                 den   = (b_eff * Hb).sum(dim=1) + 1e-8
#                 alpha_new = num / den
#                 alpha_tmp = torch.clamp(alpha_new, min=alpha_min, max=alpha_max)
#                 del b, Hb, num, den, alpha_new

#             b_eff    = b_eff
#             residual = w_eff - alpha_tmp.unsqueeze(1) * b_eff
#             Hr       = residual @ H_block.T
#             loss     = (residual * Hr).sum(dim=1)

#             improved   = loss < best_loss
#             best_loss  = torch.where(improved, loss, best_loss)
#             best_alpha = torch.where(improved, alpha_tmp, best_alpha)
#             best_bias  = torch.where(
#                 improved,
#                 torch.full_like(best_bias, bias_candidate),
#                 best_bias)
#             best_b = torch.where(
#                 improved.unsqueeze(1).expand_as(b_eff),
#                 b_eff, best_b)

#             del alpha_tmp, b_eff, residual, Hr, loss, improved

#         # Free per-block intermediates before final recompute
#         del Hw, alpha, alpha_min, alpha_max, w_sq_mean
#         torch.cuda.empty_cache()

#         alpha_q = quantize_scale_batched(best_alpha, e_bits_scale, m_bits_scale)

#         _, _, b_final = assign_fp4_dynamic_batched(
#             w_block, alpha_q, e_bits, m_bits, bias_per_row=best_bias)

#         w_hat = alpha_q.unsqueeze(1) * b_final
#         w_hat = w_hat * m_block * s_block
#         w_hat[pruned]   = 0.0
#         alpha_q[pruned] = 1.0

#         # Store results on CPU immediately to free GPU memory
#         W_q[:, i:end]       = w_hat.cpu()
#         alpha_out[:, i:end] = alpha_q.unsqueeze(1).expand(-1, k).cpu()
#         bias_out[:, i:end]  = best_bias.unsqueeze(1).float().expand(-1, k).cpu()

#         # Free everything from this block before next iteration
#         del w_block, m_block, s_block, w_eff, pruned
#         del H_block, best_loss, best_alpha, best_bias, best_b
#         del alpha_q, b_final, w_hat
#         torch.cuda.empty_cache()

#     # Free GPU tensors from first pass
#     del W_abs, sign_mat, mask_mat, W_mat
#     torch.cuda.empty_cache()

#     # ----------------------------------------------------------------
#     # Second decomposition pass — entirely on CPU
#     # Compute e, m on CPU then reconstruct weight_q on CPU
#     # Only one final tensor is moved to GPU
#     # ----------------------------------------------------------------

#     sign_f  = torch.sign(W_q)     # CPU [N, M]
#     W_abs_f = W_q.abs()           # CPU [N, M]
#     del W_q

#     # Store original shape for reshape at end
#     layer_W        = layer.weight.data
#     original_shape = layer_W.shape

#     if layer_W.dim() == 4:
#         N4        = original_shape[0]
#         W_abs_f   = W_abs_f.view(N4, -1)
#         alpha_out = alpha_out.view(N4, -1)
#         bias_out  = bias_out.view(N4, -1)
#         sign_f    = sign_f.view(N4, -1)

#     # e_out and m_out stay on CPU throughout
#     e_out = torch.zeros(W_abs_f.shape, dtype=torch.long)    # CPU
#     m_out = torch.zeros(W_abs_f.shape, dtype=torch.long)    # CPU

#     for block_idx, i in enumerate(range(0, M, block_size)):
#         end = min(i + block_size, M)

#         # Bring only this block slice to GPU
#         bias_col  = bias_out[:, i].long().to(device)
#         alpha_col = alpha_out[:, i].to(device)
#         w_col     = W_abs_f[:, i:end].to(device)

#         for bias_val in bias_col.unique():
#             rows = (bias_col == bias_val).nonzero(as_tuple=True)[0]
#             e_b, m_b, _ = assign_fp4_dynamic(
#                 w_col[rows], alpha_col[rows], e_bits, m_bits,
#                 bias=int(bias_val.item()))
#             e_out[rows, i:end] = e_b.cpu()
#             m_out[rows, i:end] = m_b.cpu()
#             del e_b, m_b

#         del bias_col, alpha_col, w_col
#         torch.cuda.empty_cache()

#     del W_abs_f

#     # ----------------------------------------------------------------
#     # Final FP reconstruction on CPU
#     # weight_q = sign * alpha * 2^(e - bias) * (1 + m / 2^m_bits)
#     # Only this single tensor is moved to GPU
#     # ----------------------------------------------------------------
#     bias_mat = bias_out.float()                              # CPU [N, M]
#     base     = alpha_out * (2.0 ** (e_out.float() - bias_mat))
#     fine     = base * m_out.float() / (2 ** m_bits) if m_bits > 0 else 0.0
#     weight_q_cpu = (base + fine) * sign_f

#     del alpha_out, bias_out, bias_mat, e_out, m_out, base, sign_f
#     if m_bits > 0:
#         del fine

#     # Reshape back to original weight shape if needed
#     if layer_W.dim() == 4:
#         weight_q_cpu = weight_q_cpu.view(original_shape)

#     # Single GPU transfer — ~67MB for LLaMA's largest layer vs ~335MB before
#     return weight_q_cpu.to(device)


## ABOVE IS THE ORIGINAL

def reconstruct_layer_fp_blockdiag_scaled_v5(
    layer,
    H_blocks_layer,
    block_size,
    e_bits,
    m_bits,
    e_bits_scale,
    m_bits_scale,
    device
):
    W    = layer.weight.data.to(device)
    mask = (W != 0).float()
 
    if W.dim() == 4:
        W_mat    = W.view(W.shape[0], -1)
        mask_mat = mask.view(W.shape[0], -1)
    else:
        W_mat    = W
        mask_mat = mask
 
    N, M = W_mat.shape
 
    sign_mat = torch.sign(W_mat)
    W_abs    = W_mat.abs()
 
    del W, mask
    torch.cuda.empty_cache()
 
    # Output tensors on CPU
    W_q       = torch.zeros(N, M, dtype=torch.float32)
    alpha_out = torch.zeros(N, M, dtype=torch.float32)
    bias_out  = torch.zeros(N, M, dtype=torch.float32)
 
    default_bias    = 2 ** (e_bits - 1) - 1
    bias_radius     = max(1, 2 ** (e_bits - 2))
    bias_candidates = list(range(default_bias - bias_radius,
                                 default_bias + bias_radius + 1))
 
    # ── First pass: block-wise alpha/bias optimisation ──────────────────────
    for block_idx, i in enumerate(range(0, M, block_size)):
        end = min(i + block_size, M)
        k   = end - i
 
        w_block = W_abs[:, i:end]
        m_block = mask_mat[:, i:end]
        s_block = sign_mat[:, i:end]
 
        H_block = H_blocks_layer[block_idx].to(device)
        if H_block.shape[0] != k:
            H_block = H_block[:k, :k]
 
        w_eff  = w_block * m_block
        pruned = m_block.sum(dim=1) < 1e-8
 
        w_sq_mean = (w_eff ** 2).mean(dim=1).clamp(min=1e-8)
        alpha     = torch.sqrt(w_sq_mean).clamp(min=1e-4)
        alpha_min = 0.05 * w_eff.abs().mean(dim=1).clamp(min=1e-8)
        alpha_max = 20.0 * w_eff.abs().mean(dim=1).clamp(min=1e-8)
 
        Hw = w_eff @ H_block.T
 
        best_loss  = torch.full((N,), float('inf'), device=device)
        best_alpha = alpha.clone()
        best_bias  = torch.full((N,), default_bias, device=device, dtype=torch.long)
        best_b     = torch.zeros_like(w_eff)
 
        for bias_candidate in bias_candidates:
            alpha_tmp = alpha.clone()
 
            for _ in range(5):
                _, _, b = assign_fp4_dynamic_batched(
                    w_block, alpha_tmp, e_bits, m_bits, bias=bias_candidate)
                b_eff = b * m_block
                Hb    = b_eff @ H_block.T
                num   = (b_eff * Hw).sum(dim=1)
                den   = (b_eff * Hb).sum(dim=1) + 1e-8
                alpha_new = num / den
                alpha_tmp = torch.clamp(alpha_new, min=alpha_min, max=alpha_max)
                del b, Hb, num, den, alpha_new
 
            b_eff    = b_eff
            residual = w_eff - alpha_tmp.unsqueeze(1) * b_eff
            Hr       = residual @ H_block.T
            loss     = (residual * Hr).sum(dim=1)
 
            improved   = loss < best_loss
            best_loss  = torch.where(improved, loss, best_loss)
            best_alpha = torch.where(improved, alpha_tmp, best_alpha)
            best_bias  = torch.where(
                improved,
                torch.full_like(best_bias, bias_candidate),
                best_bias)
            best_b = torch.where(
                improved.unsqueeze(1).expand_as(b_eff),
                b_eff, best_b)
 
            del alpha_tmp, b_eff, residual, Hr, loss, improved
 
        del Hw, alpha, alpha_min, alpha_max, w_sq_mean
        torch.cuda.empty_cache()
 
        alpha_q = quantize_scale_batched(best_alpha, e_bits_scale, m_bits_scale)
 
        _, _, b_final = assign_fp4_dynamic_batched(
            w_block, alpha_q, e_bits, m_bits, bias_per_row=best_bias)
 
        w_hat = alpha_q.unsqueeze(1) * b_final
        w_hat = w_hat * m_block * s_block
        w_hat[pruned]   = 0.0
        alpha_q[pruned] = 1.0
 
        W_q[:, i:end]       = w_hat.cpu()
        alpha_out[:, i:end] = alpha_q.unsqueeze(1).expand(-1, k).cpu()
        bias_out[:, i:end]  = best_bias.unsqueeze(1).float().expand(-1, k).cpu()
 
        del w_block, m_block, s_block, w_eff, pruned
        del H_block, best_loss, best_alpha, best_bias, best_b
        del alpha_q, b_final, w_hat
        torch.cuda.empty_cache()
 
    del W_abs, sign_mat, mask_mat, W_mat
    torch.cuda.empty_cache()
 
    # ── Save per-block alpha and bias BEFORE cleanup ─────────────────────────
    # alpha_out is [N, M] — constant within each block column.
    # Reduce to [N, n_blocks] by taking the first element of each block.
    n_blocks        = math.ceil(M / block_size)
    alpha_out_saved = alpha_out[:, ::block_size][:, :n_blocks].clone()   # [N, n_blocks] CPU
    bias_out_saved  = bias_out[:, ::block_size][:, :n_blocks].long()     # [N, n_blocks] CPU
    # ─────────────────────────────────────────────────────────────────────────
 
    # ── Second pass: decompose into e, m on CPU ──────────────────────────────
    sign_f  = torch.sign(W_q)
    W_abs_f = W_q.abs()
    del W_q
 
    layer_W        = layer.weight.data
    original_shape = layer_W.shape
 
    if layer_W.dim() == 4:
        N4        = original_shape[0]
        W_abs_f   = W_abs_f.view(N4, -1)
        alpha_out = alpha_out.view(N4, -1)
        bias_out  = bias_out.view(N4, -1)
        sign_f    = sign_f.view(N4, -1)
 
    e_out = torch.zeros(W_abs_f.shape, dtype=torch.long)
    m_out = torch.zeros(W_abs_f.shape, dtype=torch.long)
 
    for block_idx, i in enumerate(range(0, M, block_size)):
        end = min(i + block_size, M)
 
        bias_col  = bias_out[:, i].long().to(device)
        alpha_col = alpha_out[:, i].to(device)
        w_col     = W_abs_f[:, i:end].to(device)
 
        for bias_val in bias_col.unique():
            rows = (bias_col == bias_val).nonzero(as_tuple=True)[0]
            e_b, m_b, _ = assign_fp4_dynamic(
                w_col[rows], alpha_col[rows], e_bits, m_bits,
                bias=int(bias_val.item()))
            e_out[rows, i:end] = e_b.cpu()
            m_out[rows, i:end] = m_b.cpu()
            del e_b, m_b
 
        del bias_col, alpha_col, w_col
        torch.cuda.empty_cache()
 
    del W_abs_f
 
    # ── Final FP reconstruction on CPU ───────────────────────────────────────
    bias_mat     = bias_out.float()
    base         = alpha_out * (2.0 ** (e_out.float() - bias_mat))
    fine         = base * m_out.float() / (2 ** m_bits) if m_bits > 0 else 0.0
    weight_q_cpu = (base + fine) * sign_f
 
    del alpha_out, bias_out, bias_mat, e_out, m_out, base, sign_f
    if m_bits > 0:
        del fine
 
    if layer_W.dim() == 4:
        weight_q_cpu    = weight_q_cpu.view(original_shape)
        alpha_out_saved = alpha_out_saved.view(original_shape[0], -1)
        bias_out_saved  = bias_out_saved.view(original_shape[0], -1)
 
    # ── Return dict ───────────────────────────────────────────────────────────
    return {
        "weight_q": weight_q_cpu.to(device),   # [N, M] on device, as before
        "alpha":    alpha_out_saved,            # [N, n_blocks] on CPU
        "bias":     bias_out_saved,             # [N, n_blocks] on CPU (long)
    }

# def reconstruct_layer_fp_blockdiag_scaled_v6(
#     layer,
#     H_blocks_joint,        # recomputed from X_hat
#     W_corrected,           # [N, M] CPU — already corrected weights
#     block_size,
#     e_bits,
#     m_bits,
#     e_bits_scale,
#     m_bits_scale,
#     device,
# ):
#     """
#     v6: identical to v5 block optimization loop but takes pre-corrected
#     weights and pre-recomputed H_joint. Separation of concerns — all the
#     correction math happens before this function.
#     """
#     if W_corrected.dim() == 4:
#         W_mat = W_corrected.view(W_corrected.shape[0], -1)
#     else:
#         W_mat = W_corrected

#     N, M = W_mat.shape

#     sign_mat = torch.sign(W_mat)
#     W_abs    = W_mat.abs()
#     mask_mat = (W_mat != 0).float()

#     W_q       = torch.zeros(N, M, dtype=torch.float32)  # CPU
#     alpha_out = torch.zeros(N, M, dtype=torch.float32)  # CPU
#     bias_out  = torch.zeros(N, M, dtype=torch.float32)  # CPU

#     default_bias    = 2 ** (e_bits - 1) - 1
#     bias_radius     = max(1, 2 ** (e_bits - 2))
#     bias_candidates = list(range(default_bias - bias_radius,
#                                  default_bias + bias_radius + 1))

#     # ----------------------------------------------------------------
#     # Block optimization — identical structure to v5
#     # ----------------------------------------------------------------
#     for block_idx, i in enumerate(range(0, M, block_size)):
#         end = min(i + block_size, M)
#         k   = end - i

#         w_block = W_abs[:, i:end].to(device)
#         m_block = mask_mat[:, i:end].to(device)
#         s_block = sign_mat[:, i:end].to(device)

#         H_block = H_blocks_joint[block_idx].to(device)
#         if H_block.shape[0] != k:
#             H_block = H_block[:k, :k]

#         w_eff  = w_block * m_block
#         pruned = m_block.sum(dim=1) < 1e-8

#         w_sq_mean = (w_eff ** 2).mean(dim=1).clamp(min=1e-8)
#         alpha     = torch.sqrt(w_sq_mean).clamp(min=1e-4)
#         alpha_min = 0.05 * w_eff.abs().mean(dim=1).clamp(min=1e-8)
#         alpha_max = 20.0 * w_eff.abs().mean(dim=1).clamp(min=1e-8)

#         Hw = w_eff @ H_block.T

#         best_loss  = torch.full((N,), float('inf'), device=device)
#         best_alpha = alpha.clone()
#         best_bias  = torch.full((N,), default_bias, device=device, 
#                                 dtype=torch.long)
#         best_b     = torch.zeros_like(w_eff)

#         for bias_candidate in bias_candidates:
#             alpha_tmp = alpha.clone()

#             for _ in range(5):
#                 _, _, b = assign_fp4_dynamic_batched(
#                     w_block, alpha_tmp, e_bits, m_bits, 
#                     bias=bias_candidate)
#                 b_eff = b * m_block
#                 Hb    = b_eff @ H_block.T
#                 num   = (b_eff * Hw).sum(dim=1)
#                 den   = (b_eff * Hb).sum(dim=1) + 1e-8
#                 alpha_new = num / den
#                 alpha_tmp = torch.clamp(alpha_new, 
#                                         min=alpha_min, max=alpha_max)
#                 del b, Hb, num, den, alpha_new

#             residual = w_eff - alpha_tmp.unsqueeze(1) * b_eff
#             Hr       = residual @ H_block.T
#             loss     = (residual * Hr).sum(dim=1)

#             improved   = loss < best_loss
#             best_loss  = torch.where(improved, loss, best_loss)
#             best_alpha = torch.where(improved, alpha_tmp, best_alpha)
#             best_bias  = torch.where(
#                 improved,
#                 torch.full_like(best_bias, bias_candidate),
#                 best_bias)
#             best_b = torch.where(
#                 improved.unsqueeze(1).expand_as(b_eff),
#                 b_eff, best_b)

#             del alpha_tmp, b_eff, residual, Hr, loss, improved

#         del Hw, alpha, alpha_min, alpha_max, w_sq_mean
#         torch.cuda.empty_cache()

#         alpha_q = quantize_scale_batched(best_alpha, e_bits_scale, 
#                                           m_bits_scale)

#         _, _, b_final = assign_fp4_dynamic_batched(
#             w_block, alpha_q, e_bits, m_bits, bias_per_row=best_bias)

#         w_hat = alpha_q.unsqueeze(1) * b_final
#         w_hat = w_hat * m_block * s_block
#         w_hat[pruned]   = 0.0
#         alpha_q[pruned] = 1.0

#         W_q[:, i:end]       = w_hat.cpu()
#         alpha_out[:, i:end] = alpha_q.unsqueeze(1).expand(-1, k).cpu()
#         bias_out[:, i:end]  = best_bias.unsqueeze(1).float().expand(-1, k).cpu()

#         del w_block, m_block, s_block, w_eff, pruned
#         del H_block, best_loss, best_alpha, best_bias, best_b
#         del alpha_q, b_final, w_hat
#         torch.cuda.empty_cache()

#     # ----------------------------------------------------------------
#     # Second decomposition pass — CPU only, identical to v5
#     # ----------------------------------------------------------------
#     sign_f  = torch.sign(W_q)
#     W_abs_f = W_q.abs()
#     del W_q

#     original_shape = layer.weight.data.shape
#     if layer.weight.data.dim() == 4:
#         N4        = original_shape[0]
#         W_abs_f   = W_abs_f.view(N4, -1)
#         alpha_out = alpha_out.view(N4, -1)
#         bias_out  = bias_out.view(N4, -1)
#         sign_f    = sign_f.view(N4, -1)

#     e_out = torch.zeros(W_abs_f.shape, dtype=torch.long)
#     m_out = torch.zeros(W_abs_f.shape, dtype=torch.long)

#     for block_idx, i in enumerate(range(0, M, block_size)):
#         end = min(i + block_size, M)

#         bias_col  = bias_out[:, i].long().to(device)
#         alpha_col = alpha_out[:, i].to(device)
#         w_col     = W_abs_f[:, i:end].to(device)

#         for bias_val in bias_col.unique():
#             rows = (bias_col == bias_val).nonzero(as_tuple=True)[0]
#             e_b, m_b, _ = assign_fp4_dynamic(
#                 w_col[rows], alpha_col[rows], e_bits, m_bits,
#                 bias=int(bias_val.item()))
#             e_out[rows, i:end] = e_b.cpu()
#             m_out[rows, i:end] = m_b.cpu()
#             del e_b, m_b

#         del bias_col, alpha_col, w_col
#         torch.cuda.empty_cache()

#     del W_abs_f

#     bias_mat     = bias_out.float()
#     base         = alpha_out * (2.0 ** (e_out.float() - bias_mat))
#     fine         = base * m_out.float() / (2 ** m_bits) if m_bits > 0 else 0.0
#     weight_q_cpu = (base + fine) * sign_f

#     del alpha_out, bias_out, bias_mat, e_out, m_out, base, sign_f
#     if m_bits > 0:
#         del fine

#     if layer.weight.data.dim() == 4:
#         weight_q_cpu = weight_q_cpu.view(original_shape)

#     return {
#         "weight_q": weight_q_cpu,
#     }

# def reconstruct_layer_fp_blockdiag_scaled_v6(
#     layer,
#     H_blocks_joint,        # recomputed from X_hat
#     W_corrected,           # [N, M] CPU float32 — pre-corrected weights
#     block_size,
#     e_bits,
#     m_bits,
#     e_bits_scale,
#     m_bits_scale,
#     device,
#     Hadamard = False
# ):
#     """
#     v6: joint W+A quantization reconstruction.
#     Identical to corrected v5 but takes:
#       - W_corrected instead of layer.weight.data
#       - H_blocks_joint instead of original H
#     All v5 fixes are present:
#       - alpha quantized inside the optimization loop
#       - loss computed in quantized domain
#       - best_b used directly, no redundant recomputation
#     """
#     if W_corrected.dim() == 4:
#         W_mat    = W_corrected.view(W_corrected.shape[0], -1)
#     else:
#         W_mat    = W_corrected

#     # Keep original layer only for shape reference at the end
#     original_shape = layer.weight.data.shape

#     N, M = W_mat.shape

#     mask_mat = (W_mat != 0).float()
#     sign_mat = torch.sign(W_mat)
#     W_abs    = W_mat.abs()

#     W_q       = torch.zeros(N, M, dtype=torch.float32)   # CPU
#     alpha_out = torch.zeros(N, M, dtype=torch.float32)   # CPU
#     bias_out  = torch.zeros(N, M, dtype=torch.float32)   # CPU

#     default_bias    = 2 ** (e_bits - 1) - 1
#     bias_radius     = max(1, 2 ** (e_bits - 2))
#     bias_candidates = list(range(default_bias - bias_radius,
#                                  default_bias + bias_radius + 1))

#     # ----------------------------------------------------------------
#     # First pass — block-wise alpha/bias optimisation
#     # ----------------------------------------------------------------
#     for block_idx, i in enumerate(range(0, M, block_size)):
#         end = min(i + block_size, M)
#         k   = end - i

#         w_block = W_abs[:, i:end].to(device)
#         m_block = mask_mat[:, i:end].to(device)
#         s_block = sign_mat[:, i:end].to(device)

#         H_block = H_blocks_joint[block_idx].to(device)
#         if H_block.shape[0] != k:
#             H_block = H_block[:k, :k]

#         w_eff  = w_block * m_block
#         pruned = m_block.sum(dim=1) < 1e-8
#         if not Hadamard:
#             w_sq_mean = (w_eff ** 2).mean(dim=1).clamp(min=1e-8)
#             alpha     = torch.sqrt(w_sq_mean).clamp(min=1e-4)

#             # Quantize initial alpha into E4M3 domain immediately
#             alpha     = quantize_scale_batched(alpha, e_bits_scale, m_bits_scale)
#         else:
#             w_sq_mean = (w_eff ** 2).mean(dim=1).clamp(min=1e-8)
#             alpha_rms  = torch.sqrt(w_sq_mean).clamp(min=1e-4)

#             # Initialize from block max / Qmax so FP4 range is fully utilized
#             Qmax       = compute_Qmax(e_bits, m_bits, bias=default_bias)
#             alpha_max_init = w_eff.abs().amax(dim=1).clamp(min=1e-8) / Qmax

#             # Take the geometric mean of RMS and max-based init
#             # RMS alone underutilizes range, max alone is sensitive to outliers
#             alpha = torch.sqrt(alpha_rms * alpha_max_init).clamp(min=1e-4)

#             # Quantize initial alpha into E4M3 domain immediately
#             alpha = quantize_scale_batched(alpha, e_bits_scale, m_bits_scale)
#         alpha_min = 0.05 * w_eff.abs().mean(dim=1).clamp(min=1e-8)
#         alpha_max = 20.0 * w_eff.abs().mean(dim=1).clamp(min=1e-8)

#         # Use relative damping — scale-invariant for large activations
#         H_diag_mean = H_block.diagonal().mean().clamp(min=1e-8)
#         H_block     = H_block + 0.01 * H_diag_mean * torch.eye(k, device=device)

#         Hw = w_eff @ H_block.T

#         best_loss  = torch.full((N,), float('inf'), device=device)
#         best_alpha = alpha.clone()
#         best_bias  = torch.full((N,), default_bias, device=device,
#                                 dtype=torch.long)
#         best_b     = torch.zeros_like(w_eff)

#         for bias_candidate in bias_candidates:
#             alpha_tmp = alpha.clone()

#             for _ in range(5):
#                 _, _, b = assign_fp4_dynamic_batched(
#                     w_block, alpha_tmp, e_bits, m_bits,
#                     bias=bias_candidate)
#                 b_eff = b * m_block

#                 Hb  = b_eff @ H_block.T
#                 num = (b_eff * Hw).sum(dim=1)
#                 den = (b_eff * Hb).sum(dim=1) + 1e-8

#                 alpha_new = (num / den).clamp(min=alpha_min, max=alpha_max)

#                 # Quantize inside loop — optimization in quantized domain
#                 alpha_tmp = quantize_scale_batched(
#                     alpha_new, e_bits_scale, m_bits_scale)

#                 del b, Hb, num, den, alpha_new

#             # alpha_tmp and b_eff are consistent — both use quantized scale
#             residual = w_eff - alpha_tmp.unsqueeze(1) * b_eff
#             Hr       = residual @ H_block.T
#             loss     = (residual * Hr).sum(dim=1)

#             improved   = loss < best_loss
#             best_loss  = torch.where(improved, loss, best_loss)
#             best_alpha = torch.where(improved, alpha_tmp, best_alpha)
#             best_bias  = torch.where(
#                 improved,
#                 torch.full_like(best_bias, bias_candidate),
#                 best_bias)
#             best_b = torch.where(
#                 improved.unsqueeze(1).expand_as(b_eff),
#                 b_eff, best_b)

#             del alpha_tmp, b_eff, residual, Hr, loss, improved
#         if block_idx == 0:
#             print(f"    H_diag_mean: {H_diag_mean.item():.6f}")
#             print(f"    w_eff max: {w_eff.abs().max().item():.6f}")
#             print(f"    alpha init: {alpha.max().item():.6f}")
#             print(f"    alpha_min: {alpha_min.max().item():.8f}")
#             print(f"    alpha_max: {alpha_max.max().item():.6f}")
#         del Hw, alpha, alpha_min, alpha_max, w_sq_mean
#         torch.cuda.empty_cache()

#         # best_alpha already E4M3-quantized, best_b consistent with it
#         w_hat = best_alpha.unsqueeze(1) * best_b
#         w_hat = w_hat * m_block * s_block
#         w_hat[pruned]      = 0.0
#         best_alpha[pruned] = 1.0

#         W_q[:, i:end]       = w_hat.cpu()
#         alpha_out[:, i:end] = best_alpha.unsqueeze(1).expand(-1, k).cpu()
#         bias_out[:, i:end]  = best_bias.unsqueeze(1).float().expand(-1, k).cpu()
#         del w_block, m_block, s_block, w_eff, pruned
#         del H_block, best_loss, best_alpha, best_bias, best_b, w_hat
#         torch.cuda.empty_cache()
#     del W_abs, sign_mat, mask_mat, W_mat
#     torch.cuda.empty_cache()

#     # ----------------------------------------------------------------
#     # Second decomposition pass — CPU only
#     # ----------------------------------------------------------------
#     sign_f  = torch.sign(W_q)
#     W_abs_f = W_q.abs()
#     del W_q

#     if layer.weight.data.dim() == 4:
#         N4        = original_shape[0]
#         W_abs_f   = W_abs_f.view(N4, -1)
#         alpha_out = alpha_out.view(N4, -1)
#         bias_out  = bias_out.view(N4, -1)
#         sign_f    = sign_f.view(N4, -1)

#     e_out = torch.zeros(W_abs_f.shape, dtype=torch.long)
#     m_out = torch.zeros(W_abs_f.shape, dtype=torch.long)

#     for block_idx, i in enumerate(range(0, M, block_size)):
#         end = min(i + block_size, M)

#         bias_col  = bias_out[:, i].long().to(device)
#         alpha_col = alpha_out[:, i].to(device)
#         w_col     = W_abs_f[:, i:end].to(device)

#         for bias_val in bias_col.unique():
#             rows = (bias_col == bias_val).nonzero(as_tuple=True)[0]
#             e_b, m_b, _ = assign_fp4_dynamic(
#                 w_col[rows], alpha_col[rows], e_bits, m_bits,
#                 bias=int(bias_val.item()))
#             e_out[rows, i:end] = e_b.cpu()
#             m_out[rows, i:end] = m_b.cpu()
#             del e_b, m_b

#         del bias_col, alpha_col, w_col
#         torch.cuda.empty_cache()

#     del W_abs_f

#     # ----------------------------------------------------------------
#     # Final reconstruction — CPU
#     # ----------------------------------------------------------------
#     bias_mat     = bias_out.float()
#     base         = alpha_out * (2.0 ** (e_out.float() - bias_mat))
#     fine         = base * m_out.float() / (2 ** m_bits) if m_bits > 0 else 0.0
#     weight_q_cpu = (base + fine) * sign_f

#     alpha_ret = alpha_out.clone()
#     bias_ret  = bias_out.clone()

#     del bias_mat, e_out, m_out, base, sign_f
#     if m_bits > 0:
#         del fine
#     del alpha_out, bias_out

#     if layer.weight.data.dim() == 4:
#         weight_q_cpu = weight_q_cpu.view(original_shape)

#     return {
#         "weight_q": weight_q_cpu.to(device),
#         "alpha":    alpha_ret,
#         "bias":     bias_ret,
#     }



# def reconstruct_layer_fp_blockdiag_scaled_v6(
#     layer,
#     H_blocks_joint,
#     W_corrected,
#     block_size,
#     e_bits,
#     m_bits,
#     e_bits_scale,
#     m_bits_scale,
#     device,
#     Hadamard=False
# ):
#     """
#     v6: joint W+A quantization reconstruction.
#     Identical to corrected v5 but takes:
#       - W_corrected instead of layer.weight.data
#       - H_blocks_joint instead of original H
#     All v5 fixes are present:
#       - alpha quantized inside the optimization loop
#       - loss computed in quantized domain
#       - best_b used directly, no redundant recomputation
#     Key addition over v5:
#       - H blocks normalized to unit diagonal before optimization
#         so the Hessian-guided update is scale-invariant across layers
#         (critical when H_joint varies 1000x across layers due to
#          different activation magnitudes post-Hadamard)
#     """
#     if W_corrected.dim() == 4:
#         W_mat = W_corrected.view(W_corrected.shape[0], -1)
#     else:
#         W_mat = W_corrected

#     original_shape = layer.weight.data.shape

#     N, M = W_mat.shape

#     mask_mat = (W_mat != 0).float()
#     sign_mat = torch.sign(W_mat)
#     W_abs    = W_mat.abs()

#     W_q       = torch.zeros(N, M, dtype=torch.float32)
#     alpha_out = torch.zeros(N, M, dtype=torch.float32)
#     bias_out  = torch.zeros(N, M, dtype=torch.float32)

#     default_bias    = 2 ** (e_bits - 1) - 1
#     bias_radius     = max(1, 2 ** (e_bits - 2))
#     bias_candidates = list(range(default_bias - bias_radius,
#                                  default_bias + bias_radius + 1))

#     # ----------------------------------------------------------------
#     # First pass — block-wise alpha/bias optimisation
#     # ----------------------------------------------------------------
#     for block_idx, i in enumerate(range(0, M, block_size)):
#         end = min(i + block_size, M)
#         k   = end - i

#         w_block = W_abs[:, i:end].to(device)
#         m_block = mask_mat[:, i:end].to(device)
#         s_block = sign_mat[:, i:end].to(device)

#         H_block = H_blocks_joint[block_idx].to(device)
#         if H_block.shape[0] != k:
#             H_block = H_block[:k, :k]

#         w_eff  = w_block * m_block
#         pruned = m_block.sum(dim=1) < 1e-8

#         # ── Alpha initialisation ────────────────────────────────────
#         if not Hadamard:
#             w_sq_mean = (w_eff ** 2).mean(dim=1).clamp(min=1e-8)
#             alpha     = torch.sqrt(w_sq_mean).clamp(min=1e-4)
#             alpha     = quantize_scale_batched(alpha, e_bits_scale, m_bits_scale)
#         else:
#             w_sq_mean      = (w_eff ** 2).mean(dim=1).clamp(min=1e-8)
#             alpha_rms      = torch.sqrt(w_sq_mean).clamp(min=1e-4)
#             Qmax           = compute_Qmax(e_bits, m_bits, bias=default_bias)
#             alpha_max_init = w_eff.abs().amax(dim=1).clamp(min=1e-8) / Qmax
#             alpha          = torch.sqrt(alpha_rms * alpha_max_init).clamp(min=1e-4)
#             alpha          = quantize_scale_batched(alpha, e_bits_scale, m_bits_scale)

#         alpha_min = 0.05 * w_eff.abs().mean(dim=1).clamp(min=1e-8)
#         alpha_max = 20.0 * w_eff.abs().mean(dim=1).clamp(min=1e-8)

#         # ── Normalise H to unit diagonal before optimisation ────────
#         # H_joint scale varies 1000x across layers depending on
#         # activation magnitude — normalisation makes the curvature
#         # direction scale-invariant and stabilises the alpha update.
#         H_diag_mean = H_block.diagonal().mean().clamp(min=1e-8)
#         H_block_norm = H_block / H_diag_mean
#         H_block_norm = H_block_norm + 0.01 * torch.eye(k, device=device)

#         Hw = w_eff @ H_block_norm.T

#         best_loss  = torch.full((N,), float('inf'), device=device)
#         best_alpha = alpha.clone()
#         best_bias  = torch.full((N,), default_bias, device=device,
#                                 dtype=torch.long)
#         best_b     = torch.zeros_like(w_eff)

#         for bias_candidate in bias_candidates:
#             alpha_tmp = alpha.clone()

#             for _ in range(5):
#                 _, _, b = assign_fp4_dynamic_batched(
#                     w_block, alpha_tmp, e_bits, m_bits,
#                     bias=bias_candidate)
#                 b_eff = b * m_block

#                 Hb  = b_eff @ H_block_norm.T
#                 num = (b_eff * Hw).sum(dim=1)
#                 den = (b_eff * Hb).sum(dim=1) + 1e-8

#                 alpha_new = (num / den).clamp(min=alpha_min, max=alpha_max)

#                 alpha_tmp = quantize_scale_batched(
#                     alpha_new, e_bits_scale, m_bits_scale)

#                 del b, Hb, num, den, alpha_new

#             residual = w_eff - alpha_tmp.unsqueeze(1) * b_eff
#             Hr       = residual @ H_block_norm.T
#             loss     = (residual * Hr).sum(dim=1)

#             improved   = loss < best_loss
#             best_loss  = torch.where(improved, loss, best_loss)
#             best_alpha = torch.where(improved, alpha_tmp, best_alpha)
#             best_bias  = torch.where(
#                 improved,
#                 torch.full_like(best_bias, bias_candidate),
#                 best_bias)
#             best_b = torch.where(
#                 improved.unsqueeze(1).expand_as(b_eff),
#                 b_eff, best_b)

#             del alpha_tmp, b_eff, residual, Hr, loss, improved
#         if block_idx == 0:
#             print(f"    H_diag_mean:  {H_diag_mean.item():.6f}")
#             print(f"    w_eff max:    {w_eff.abs().max().item():.6f}")
#             print(f"    alpha final:  {best_alpha.max().item():.6f}")
#             print(f"    alpha_min:    {alpha_min.max().item():.8f}")
#             print(f"    alpha_max:    {alpha_max.max().item():.6f}")
#         del Hw, alpha, alpha_min, alpha_max, w_sq_mean
#         torch.cuda.empty_cache()

#         w_hat = best_alpha.unsqueeze(1) * best_b
#         w_hat = w_hat * m_block * s_block
#         w_hat[pruned]      = 0.0
#         best_alpha[pruned] = 1.0
#         if block_idx == 0:
#             print(f"    w_hat max:     {w_hat.abs().max().item():.6f}")    
#         W_q[:, i:end]       = w_hat.cpu()
#         alpha_out[:, i:end] = best_alpha.unsqueeze(1).expand(-1, k).cpu()
#         bias_out[:, i:end]  = best_bias.unsqueeze(1).float().expand(-1, k).cpu()

#         # ── Diagnostic — first block only, before del ───────────────

#         del w_block, m_block, s_block, w_eff, pruned
#         del H_block, H_block_norm, H_diag_mean
#         del best_loss, best_alpha, best_bias, best_b, w_hat
#         torch.cuda.empty_cache()

#     del W_abs, sign_mat, mask_mat, W_mat
#     torch.cuda.empty_cache()

#     # ----------------------------------------------------------------
#     # Second decomposition pass — CPU only
#     # ----------------------------------------------------------------
#     sign_f  = torch.sign(W_q)
#     W_abs_f = W_q.abs()
#     del W_q

#     if layer.weight.data.dim() == 4:
#         N4        = original_shape[0]
#         W_abs_f   = W_abs_f.view(N4, -1)
#         alpha_out = alpha_out.view(N4, -1)
#         bias_out  = bias_out.view(N4, -1)
#         sign_f    = sign_f.view(N4, -1)

#     e_out = torch.zeros(W_abs_f.shape, dtype=torch.long)
#     m_out = torch.zeros(W_abs_f.shape, dtype=torch.long)

#     for block_idx, i in enumerate(range(0, M, block_size)):
#         end = min(i + block_size, M)

#         bias_col  = bias_out[:, i].long().to(device)
#         alpha_col = alpha_out[:, i].to(device)
#         w_col     = W_abs_f[:, i:end].to(device)

#         for bias_val in bias_col.unique():
#             rows = (bias_col == bias_val).nonzero(as_tuple=True)[0]
#             e_b, m_b, _ = assign_fp4_dynamic(
#                 w_col[rows], alpha_col[rows], e_bits, m_bits,
#                 bias=int(bias_val.item()))
#             e_out[rows, i:end] = e_b.cpu()
#             m_out[rows, i:end] = m_b.cpu()
#             del e_b, m_b

#         del bias_col, alpha_col, w_col
#         torch.cuda.empty_cache()

#     del W_abs_f

#     # ----------------------------------------------------------------
#     # Final reconstruction — CPU
#     # ----------------------------------------------------------------
#     bias_mat     = bias_out.float()
#     base         = alpha_out * (2.0 ** (e_out.float() - bias_mat))
#     fine         = base * m_out.float() / (2 ** m_bits) if m_bits > 0 else 0.0
#     weight_q_cpu = (base + fine) * sign_f

#     alpha_ret = alpha_out.clone()
#     bias_ret  = bias_out.clone()

#     del bias_mat, e_out, m_out, base, sign_f
#     if m_bits > 0:
#         del fine
#     del alpha_out, bias_out

#     if layer.weight.data.dim() == 4:
#         weight_q_cpu = weight_q_cpu.view(original_shape)

#     return {
#         "weight_q": weight_q_cpu.to(device),
#         "alpha":    alpha_ret,
#         "bias":     bias_ret,
#     }


def refine_alpha_mantissa(w_block, alpha_exp, best_bias_per_row,
                           e_bits, m_bits, e_bits_scale, m_bits_scale,
                           H_block_norm, w_eff, Hw, m_block,
                           alpha_min, alpha_max, device):
    """
    Given the power-of-2 exponent component of alpha (alpha_exp = 2^e),
    search over all E4M3 mantissa values to find the best full alpha.

    This completes the Hessian-guided discretisation of the scale:
      - Bias search handles the coarse exponent (e_w - b)
      - This handles the fine mantissa of alpha

    alpha_exp:        [N] power-of-2 floor of best_alpha from bias search
    best_bias_per_row:[N] long — winning bias per output row
    Returns: best_alpha [N], best_b [N, k]
    """
    N, k       = w_eff.shape
    n_mantissa = 2 ** m_bits_scale   # 8 for E4M3

    best_loss  = torch.full((N,), float('inf'), device=device)
    best_alpha = alpha_exp.clone()
    best_b     = torch.zeros_like(w_eff)

    for m_idx in range(n_mantissa):
        # Construct alpha candidate: 2^e * (1 + m/8)
        mant_val   = 1.0 + m_idx / n_mantissa          # 1.000 … 1.875
        alpha_cand = (alpha_exp * mant_val).clamp(
            min=alpha_min, max=alpha_max)

        # Snap to nearest representable E4M3 value — keeps us in the
        # quantized domain and ensures no two candidates are identical
        alpha_cand = quantize_scale_batched(
            alpha_cand, e_bits_scale, m_bits_scale)

        # One round of FP4 assignment with this alpha
        _, _, b = assign_fp4_dynamic_batched(
            w_block, alpha_cand, e_bits, m_bits,
            bias_per_row=best_bias_per_row)
        b_eff = b * m_block

        # Hessian-weighted reconstruction loss
        residual = w_eff - alpha_cand.unsqueeze(1) * b_eff
        Hr       = residual @ H_block_norm.T
        loss     = (residual * Hr).sum(dim=1)

        improved   = loss < best_loss
        best_loss  = torch.where(improved, loss, best_loss)
        best_alpha = torch.where(improved, alpha_cand, best_alpha)
        best_b     = torch.where(
            improved.unsqueeze(1).expand_as(b_eff),
            b_eff, best_b)

        del alpha_cand, b, b_eff, residual, Hr, loss, improved

    torch.cuda.empty_cache()
    return best_alpha, best_b


def reconstruct_layer_fp_blockdiag_scaled_v6(
    layer,
    H_blocks_joint,
    W_corrected,
    block_size,
    e_bits,
    m_bits,
    e_bits_scale,
    m_bits_scale,
    device,
    Hadamard=False
):
    """
    v6: joint W+A quantization reconstruction with full scale discretisation.

    Improvements over v5:
      - W_corrected and H_blocks_joint passed in (joint calibration)
      - H blocks normalised to unit diagonal (scale-invariant across layers)
      - Alpha quantised inside the iterative loop (quantised-domain optimisation)
      - Bias search (coarse) + mantissa search (fine) jointly minimise
        Hessian loss over all representable E4M3 scale values
      - best_b used directly — no redundant recomputation
    """
    if W_corrected.dim() == 4:
        W_mat = W_corrected.view(W_corrected.shape[0], -1)
    else:
        W_mat = W_corrected

    original_shape = layer.weight.data.shape
    N, M           = W_mat.shape

    mask_mat = (W_mat != 0).float()
    sign_mat = torch.sign(W_mat)
    W_abs    = W_mat.abs()

    W_q       = torch.zeros(N, M, dtype=torch.float32)
    alpha_out = torch.zeros(N, M, dtype=torch.float32)
    bias_out  = torch.zeros(N, M, dtype=torch.float32)

    default_bias    = 2 ** (e_bits - 1) - 1
    bias_radius     = max(1, 2 ** (e_bits - 2))
    bias_candidates = list(range(default_bias - bias_radius,
                                 default_bias + bias_radius + 1))

    # E4M3 exponent bounds for clamping alpha_exp extraction
    e_min_scale = -(2 ** (e_bits_scale - 1))
    e_max_scale =  (2 ** (e_bits_scale - 1)) - 1

    # ----------------------------------------------------------------
    # First pass — block-wise alpha/bias/mantissa optimisation
    # ----------------------------------------------------------------
    for block_idx, i in enumerate(range(0, M, block_size)):
        end = min(i + block_size, M)
        k   = end - i

        w_block = W_abs[:, i:end].to(device)
        m_block = mask_mat[:, i:end].to(device)
        s_block = sign_mat[:, i:end].to(device)

        H_block = H_blocks_joint[block_idx].to(device)
        if H_block.shape[0] != k:
            H_block = H_block[:k, :k]

        w_eff  = w_block * m_block
        pruned = m_block.sum(dim=1) < 1e-8

        # ── Alpha initialisation ─────────────────────────────────────
        if not Hadamard:
            w_sq_mean = (w_eff ** 2).mean(dim=1).clamp(min=1e-8)
            alpha     = torch.sqrt(w_sq_mean).clamp(min=1e-4)
            alpha     = quantize_scale_batched(alpha, e_bits_scale,
                                               m_bits_scale)
        else:
            w_sq_mean      = (w_eff ** 2).mean(dim=1).clamp(min=1e-8)
            alpha_rms      = torch.sqrt(w_sq_mean).clamp(min=1e-4)
            Qmax           = compute_Qmax(e_bits, m_bits, bias=default_bias)
            alpha_max_init = w_eff.abs().amax(dim=1).clamp(min=1e-8) / Qmax
            alpha          = torch.sqrt(alpha_rms * alpha_max_init).clamp(
                                 min=1e-4)
            alpha          = quantize_scale_batched(alpha, e_bits_scale,
                                                    m_bits_scale)

        alpha_min = 0.05 * w_eff.abs().mean(dim=1).clamp(min=1e-8)
        alpha_max = 20.0 * w_eff.abs().mean(dim=1).clamp(min=1e-8)

        # ── Normalise H to unit diagonal ─────────────────────────────
        H_diag_mean  = H_block.diagonal().mean().clamp(min=1e-8)
        H_block_norm = H_block / H_diag_mean
        H_block_norm = H_block_norm + 0.01 * torch.eye(k, device=device)

        Hw = w_eff @ H_block_norm.T

        # ── Phase 1: bias search (coarse exponent) ───────────────────
        best_loss  = torch.full((N,), float('inf'), device=device)
        best_alpha = alpha.clone()
        best_bias  = torch.full((N,), default_bias, device=device,
                                dtype=torch.long)
        best_b     = torch.zeros_like(w_eff)

        for bias_candidate in bias_candidates:
            alpha_tmp = alpha.clone()

            for _ in range(5):
                _, _, b = assign_fp4_dynamic_batched(
                    w_block, alpha_tmp, e_bits, m_bits,
                    bias=bias_candidate)
                b_eff = b * m_block

                Hb  = b_eff @ H_block_norm.T
                num = (b_eff * Hw).sum(dim=1)
                den = (b_eff * Hb).sum(dim=1) + 1e-8

                alpha_new = (num / den).clamp(min=alpha_min, max=alpha_max)
                alpha_tmp = quantize_scale_batched(
                    alpha_new, e_bits_scale, m_bits_scale)

                del b, Hb, num, den, alpha_new

            residual = w_eff - alpha_tmp.unsqueeze(1) * b_eff
            Hr       = residual @ H_block_norm.T
            loss     = (residual * Hr).sum(dim=1)

            improved   = loss < best_loss
            best_loss  = torch.where(improved, loss, best_loss)
            best_alpha = torch.where(improved, alpha_tmp, best_alpha)
            best_bias  = torch.where(
                improved,
                torch.full_like(best_bias, bias_candidate),
                best_bias)
            best_b = torch.where(
                improved.unsqueeze(1).expand_as(b_eff),
                b_eff, best_b)

            del alpha_tmp, b_eff, residual, Hr, loss, improved

        del alpha, alpha_min, alpha_max, w_sq_mean
        torch.cuda.empty_cache()

        # ── Phase 2: mantissa refinement (fine scale) ────────────────
        # Extract the power-of-2 floor of the winning alpha —
        # this is the base from which we search all 8 mantissa values.
        alpha_exp = 2.0 ** torch.clamp(
            torch.floor(torch.log2(best_alpha.clamp(min=1e-8))),
            e_min_scale, e_max_scale
        )

        # Recompute alpha_min/alpha_max from w_eff for mantissa search
        alpha_min_m = 0.05 * w_eff.abs().mean(dim=1).clamp(min=1e-8)
        alpha_max_m = 20.0 * w_eff.abs().mean(dim=1).clamp(min=1e-8)

        best_alpha, best_b = refine_alpha_mantissa(
            w_block           = w_block,
            alpha_exp         = alpha_exp,
            best_bias_per_row = best_bias,
            e_bits            = e_bits,
            m_bits            = m_bits,
            e_bits_scale      = e_bits_scale,
            m_bits_scale      = m_bits_scale,
            H_block_norm      = H_block_norm,
            w_eff             = w_eff,
            Hw                = Hw,
            m_block           = m_block,
            alpha_min         = alpha_min_m,
            alpha_max         = alpha_max_m,
            device            = device
        )

        del alpha_exp, alpha_min_m, alpha_max_m
        torch.cuda.empty_cache()

        # ── Reconstruct and store ─────────────────────────────────────
        w_hat = best_alpha.unsqueeze(1) * best_b
        w_hat = w_hat * m_block * s_block
        w_hat[pruned]      = 0.0
        best_alpha[pruned] = 1.0

        W_q[:, i:end]       = w_hat.cpu()
        alpha_out[:, i:end] = best_alpha.unsqueeze(1).expand(-1, k).cpu()
        bias_out[:, i:end]  = best_bias.unsqueeze(1).float().expand(
                                   -1, k).cpu()

        # ── Diagnostic — first block only ────────────────────────────
        if block_idx == 0:
            print(f"    H_diag_mean:  {H_diag_mean.item():.6f}")
            print(f"    w_eff max:    {w_eff.abs().max().item():.6f}")
            print(f"    alpha final:  {best_alpha.max().item():.6f}")
            print(f"    W_q max:      {w_hat.abs().max().item():.6f}")

        del w_block, m_block, s_block, w_eff, pruned
        del H_block, H_block_norm, H_diag_mean, Hw
        del best_loss, best_alpha, best_bias, best_b, w_hat
        torch.cuda.empty_cache()

    del W_abs, sign_mat, mask_mat, W_mat
    torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # Second decomposition pass — CPU only
    # ----------------------------------------------------------------
    sign_f  = torch.sign(W_q)
    W_abs_f = W_q.abs()
    del W_q

    if layer.weight.data.dim() == 4:
        N4        = original_shape[0]
        W_abs_f   = W_abs_f.view(N4, -1)
        alpha_out = alpha_out.view(N4, -1)
        bias_out  = bias_out.view(N4, -1)
        sign_f    = sign_f.view(N4, -1)

    e_out = torch.zeros(W_abs_f.shape, dtype=torch.long)
    m_out = torch.zeros(W_abs_f.shape, dtype=torch.long)

    for block_idx, i in enumerate(range(0, M, block_size)):
        end = min(i + block_size, M)

        bias_col  = bias_out[:, i].long().to(device)
        alpha_col = alpha_out[:, i].to(device)
        w_col     = W_abs_f[:, i:end].to(device)

        for bias_val in bias_col.unique():
            rows = (bias_col == bias_val).nonzero(as_tuple=True)[0]
            e_b, m_b, _ = assign_fp4_dynamic(
                w_col[rows], alpha_col[rows], e_bits, m_bits,
                bias=int(bias_val.item()))
            e_out[rows, i:end] = e_b.cpu()
            m_out[rows, i:end] = m_b.cpu()
            del e_b, m_b

        del bias_col, alpha_col, w_col
        torch.cuda.empty_cache()

    del W_abs_f

    # ----------------------------------------------------------------
    # Final reconstruction — CPU
    # ----------------------------------------------------------------
    bias_mat     = bias_out.float()
    base         = alpha_out * (2.0 ** (e_out.float() - bias_mat))
    fine         = base * m_out.float() / (2 ** m_bits) if m_bits > 0 \
                   else 0.0
    weight_q_cpu = (base + fine) * sign_f

    alpha_ret = alpha_out.clone()
    bias_ret  = bias_out.clone()

    del bias_mat, e_out, m_out, base, sign_f
    if m_bits > 0:
        del fine
    del alpha_out, bias_out

    if layer.weight.data.dim() == 4:
        weight_q_cpu = weight_q_cpu.view(original_shape)

    return {
        "weight_q": weight_q_cpu.to(device),
        "alpha":    alpha_ret,
        "bias":     bias_ret,
    }

def compute_Qmax(e_bits, m_bits, bias):
    """
    Compute the maximum representable value for a floating point format.
    
    Q_max = (2 - 2^-m) * 2^(2^e - bias - 1)
    
    For NVFP4 E2M1 with bias=1:
        = (2 - 2^-1) * 2^(4 - 1 - 1)
        = 1.5 * 4
        = 6.0
    """
    mantissa_term  = 2.0 - (2.0 ** (-m_bits))
    exponent_term  = 2.0 ** ((2 ** e_bits) - bias - 1)
    return mantissa_term * exponent_term

def quantize_activations_nvfp4(X, block_size=16, e_bits_scale=4, m_bits_scale=3, device='cuda'):
    """
    Quantize activations exactly as NVFP4 hardware would.
    X: [T, M] where T = tokens, M = input features
    Returns X_hat: [T, M] dequantized FP4 activations
    """
    T, M = X.shape
    X_hat = torch.zeros_like(X)
    Qmax = compute_Qmax(e_bits=2, m_bits=1, bias=1)  # = 6.0

    for i in range(0, M, block_size):
        end = min(i + block_size, M)
        x_block = X[:, i:end]  # [T, block_size]

        # Per-token block scale — each token gets its own scale for this block
        scale = Qmax / x_block.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)  # [T, 1]

        # Quantize scale to E4M3
        scale_q = quantize_scale_batched(scale.squeeze(1), e_bits_scale, m_bits_scale)  # [T]
        scale_q = scale_q.unsqueeze(1)  # [T, 1]

        # Scale, quantize to FP4, dequantize
        x_scaled = x_block * scale_q  # map into FP4 range
        _, _, b = assign_fp4_dynamic_batched(
            x_scaled.abs(), 
            torch.ones(T, device=device),  # alpha=1, scaling already applied
            e_bits=2, m_bits=1, bias=1
        )
        sign = torch.sign(x_scaled)
        x_q = sign * b / scale_q  # dequantize back

        X_hat[:, i:end] = x_q

    return X_hat


# def compute_W_correction_blockwise(W_mat, X_calib, X_hat, 
#                                     block_size, damping, device):
#     """
#     Compute W' = W + W @ delta_X.T @ X_hat @ inv(X_hat.T @ X_hat + lambda*I)
#     blockwise to match block diagonal H structure.

#     W_mat:   [N, M] CPU float32
#     X_calib: [T, M] CPU float32  — clean activations
#     X_hat:   [T, M] CPU float32  — FP4 quantized activations
#     Returns W_corrected: [N, M] CPU float32
#     """
#     N, M = W_mat.shape
#     T    = X_calib.shape[0]
#     W_corrected = W_mat.clone()

#     for i in range(0, M, block_size):
#         end = min(i + block_size, M)
#         k   = end - i

#         X_block     = X_calib[:, i:end].to(device)   # [T, k]
#         X_hat_block = X_hat[:, i:end].to(device)      # [T, k]
#         W_block     = W_mat[:, i:end].to(device)      # [N, k]
#         delta_X     = X_block - X_hat_block            # [T, k]

#         # H_hat = (1/T) * X_hat^T X_hat  [k, k]
#         H_hat        = (X_hat_block.T @ X_hat_block) / T
#         H_hat_damped = H_hat + damping * torch.eye(k, device=device)
#         H_hat_inv    = torch.linalg.inv(H_hat_damped)  # [k, k]

#         # cross = (1/T) * delta_X^T @ X_hat  [k, k]
#         cross      = (delta_X.T @ X_hat_block) / T
#         # correction = W_block @ cross @ H_hat_inv  [N, k]
#         correction = W_block @ (cross @ H_hat_inv)

#         W_corrected[:, i:end] = (W_block + correction).cpu()

#         del X_block, X_hat_block, W_block, delta_X
#         del H_hat, H_hat_damped, H_hat_inv, cross, correction
#         torch.cuda.empty_cache()

#     return W_corrected  # [N, M] CPU


def compute_W_correction_blockwise(W_mat, X_calib, X_hat,
                                    block_size, damping, device,
                                    max_correction_ratio=0.15):
    N, M = W_mat.shape
    T    = X_calib.shape[0]
    W_corrected = W_mat.clone()

    for i in range(0, M, block_size):
        end = min(i + block_size, M)
        k   = end - i

        X_block     = X_calib[:, i:end].to(device)
        X_hat_block = X_hat[:, i:end].to(device)
        W_block     = W_mat[:, i:end].to(device)
        delta_X     = X_block - X_hat_block

        H_hat        = (X_hat_block.T @ X_hat_block) / T
        H_diag_mean  = H_hat.diagonal().mean().clamp(min=1e-8)
        H_hat_damped = H_hat + damping * H_diag_mean * \
                       torch.eye(k, device=device)
        H_hat_inv    = torch.linalg.inv(H_hat_damped)

        cross      = (delta_X.T @ X_hat_block) / T
        correction = W_block @ (cross @ H_hat_inv)

        # Clamp correction magnitude per row
        w_norm    = W_block.norm(dim=1, keepdim=True).clamp(min=1e-8)
        c_norm    = correction.norm(dim=1, keepdim=True).clamp(min=1e-8)
        max_c     = max_correction_ratio * w_norm
        scale     = torch.minimum(torch.ones_like(c_norm), max_c / c_norm)
        correction = correction * scale

        W_corrected[:, i:end] = (W_block + correction).cpu()

        del X_block, X_hat_block, W_block, delta_X
        del H_hat, H_hat_damped, H_hat_inv, cross, correction
        del w_norm, c_norm, max_c, scale
        torch.cuda.empty_cache()

    return W_corrected

# def collect_calibration_activations(model, calib_loader, device, 
#                                      block_size, num_batches=4):
#     """
#     Collect raw input activations for each QuantLinearFP layer.
#     Returns dict: layer_name -> [T, in_features] float32 CPU tensor
#     """
#     act_data = {}
#     hooks    = []

#     def make_hook(name):
#         def hook(module, inp, out):
#             x = inp[0].detach().float().cpu()
#             # Flatten all leading dims except last (in_features)
#             x_2d = x.reshape(-1, x.shape[-1])  # [T, in_features]
#             if name not in act_data:
#                 act_data[name] = [x_2d]
#             else:
#                 act_data[name].append(x_2d)
#         return hook

#     for name, module in model.named_modules():
#         if type(module).__name__ == "QuantLinearFP":
#             hooks.append(module.register_forward_hook(make_hook(name)))

#     model.eval()
#     batches_run = 0
#     with torch.no_grad():
#         for i, batch in enumerate(calib_loader):
#             if batch is None:
#                 continue
#             if isinstance(batch, (list, tuple)):
#                 x = batch[0]
#             else:
#                 x = batch
#             if x is None:
#                 continue
#             model(x.to(device))
#             batches_run += 1
#             if batches_run >= num_batches:
#                 break

#     for h in hooks:
#         h.remove()

#     # Concatenate all batches per layer
#     result = {}
#     for name, chunks in act_data.items():
#         result[name] = torch.cat(chunks, dim=0)  # [T_total, in_features]
#         print(f"  {name}: collected {result[name].shape[0]} tokens, "
#               f"{result[name].shape[1]} features")

#     print(f"Collected activations for {len(result)} layers "
#           f"over {batches_run} batches")
#     return result


def collect_calibration_activations(model, calib_loader, device,
                                     block_size, num_batches=4):
    act_data = {}
    hooks    = []

    # Support both wrapped and unwrapped QuantLinearFP
    HOOKABLE = {"QuantLinearFP", "HadamardQuantLinearFP",
                "QuantConv2dFP", "QuantConv1dFP"}

    def make_hook(name):
        def hook(module, inp, out):
            x    = inp[0].detach().float()
            x_2d = x.reshape(-1, x.shape[-1])
            if name not in act_data:
                act_data[name] = [x_2d.cpu()]
            else:
                act_data[name].append(x_2d.cpu())
        return hook

    for name, module in model.named_modules():
        if type(module).__name__ in HOOKABLE:
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    batches_run = 0
    with torch.no_grad():
        for i, batch in enumerate(calib_loader):
            if batch is None:
                continue
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            if x is None:
                continue
            model(x.to(device))
            batches_run += 1
            if batches_run >= num_batches:
                break

    for h in hooks:
        h.remove()

    result = {}
    for name, chunks in act_data.items():
        result[name] = torch.cat(chunks, dim=0)
        print(f"  {name}: collected {result[name].shape[0]} tokens, "
              f"{result[name].shape[1]} features")

    print(f"Collected activations for {len(result)} layers "
          f"over {batches_run} batches")
    return result

# def recompute_H_blocks_from_Xhat(X_hat, block_size, device):
#     """
#     Recompute block diagonal Hessian blocks from quantized activations.
#     X_hat: [T, M] CPU float32
#     Returns list of [k, k] CPU tensors, one per block
#     """
#     T, M     = X_hat.shape
#     H_blocks = []

#     for i in range(0, M, block_size):
#         end     = min(i + block_size, M)
#         X_block = X_hat[:, i:end].to(device)            # [T, k]
#         H_block = (X_block.T @ X_block) / T             # [k, k]
#         H_blocks.append(H_block.cpu())

#         del X_block, H_block
#         torch.cuda.empty_cache()

#     return H_blocks


def recompute_H_blocks_from_Xhat(X_hat, block_size, device):
    T, M     = X_hat.shape
    H_blocks = []
    H_diag_means = []

    for i in range(0, M, block_size):
        end     = min(i + block_size, M)
        X_block = X_hat[:, i:end].to(device)
        H_block = (X_block.T @ X_block) / T
        H_diag_means.append(H_block.diagonal().mean().item())
        H_blocks.append(H_block.cpu())
        del X_block, H_block
        torch.cuda.empty_cache()

    print(f"    H_joint diag range: "
          f"[{min(H_diag_means):.4f}, {max(H_diag_means):.4f}], "
          f"mean={sum(H_diag_means)/len(H_diag_means):.4f}")
    return H_blocks

# def reconstruct_layer_fp_blockdiag_scaled_v5_preshifted(
#     layer,
#     H_blocks_layer,
#     block_size,
#     e_bits,
#     m_bits,
#     e_bits_scale,
#     m_bits_scale,
#     device,
#     act_channel_max=None,   # (in_features,) per-channel activation max
#                              # if None, falls back to standard v5
# ):
#     W = layer.weight.data.to(device).float()

#     # ── Pre-shifted bias computation and weight adjustment ───────────────
#     if act_channel_max is not None:
#         ch_max  = act_channel_max.to(device).clamp(min=1e-8)
#         in_features = W.shape[1]

#         # Compute real-valued per-channel bias from activation stats
#         b_tilde = (2**e_bits
#                    - torch.log2(ch_max)
#                    + math.log2(2 - 2**(-m_bits))
#                    - 1)                              # (in_features,)

#         # Decompose into scalar rho + integer per-channel correction
#         rho   = b_tilde.min()
#         b_ori = torch.clamp(
#             torch.round(b_tilde - rho).long(),
#             0, 2**e_bits - 1
#         )                                            # (in_features,)

#         # Absorb per-channel correction into weights
#         beta  = (2.0 ** (-b_ori.float()))           # (in_features,)
#         W_adjusted = W * beta.unsqueeze(0)           # (out, in)

#         # The bias search should be centered on rho
#         # rho is already a real-valued exponent bias equivalent
#         # Convert to the integer bias space your codebook uses
#         preshifted_center = int(torch.round(rho).clamp(
#             0, 2**e_bits - 1
#         ).item())

#         print(f"  Pre-shifted: rho={rho.item():.3f}, "
#               f"center={preshifted_center}, "
#               f"b_ori range=[{b_ori.min().item()},{b_ori.max().item()}]")
#     else:
#         W_adjusted       = W
#         preshifted_center = 2**(e_bits - 1) - 1     # standard default
#         rho              = None
#         b_ori            = None

#     # ── Swap in adjusted weights for the rest of v5 ──────────────────────
#     # Create a temporary wrapper so the rest of the function 
#     # sees W_adjusted as layer.weight.data
#     original_weight      = layer.weight.data.clone()
#     layer.weight.data    = W_adjusted

#     mask = (W_adjusted != 0).float()

#     if W_adjusted.dim() == 4:
#         W_mat    = W_adjusted.view(W_adjusted.shape[0], -1)
#         mask_mat = mask.view(mask.shape[0], -1)
#     else:
#         W_mat    = W_adjusted
#         mask_mat = mask

#     N, M = W_mat.shape

#     sign_mat = torch.sign(W_mat)
#     W_abs    = W_mat.abs()

#     del mask
#     torch.cuda.empty_cache()

#     W_q       = torch.zeros(N, M, dtype=torch.float32)
#     alpha_out = torch.zeros(N, M, dtype=torch.float32)
#     bias_out  = torch.zeros(N, M, dtype=torch.float32)

#     # ── Bias search centered on preshifted_center ─────────────────────────
#     bias_radius     = max(1, 2**(e_bits - 2))
#     bias_candidates = list(range(
#         preshifted_center - bias_radius,
#         preshifted_center + bias_radius + 1
#     ))
#     # Clamp to valid range
#     bias_candidates = [b for b in bias_candidates
#                        if 0 <= b <= 2**e_bits - 1]

#     print(f"  Bias search: center={preshifted_center}, "
#           f"candidates={bias_candidates}")

#     # ── Rest of v5 is identical ───────────────────────────────────────────
#     for block_idx, i in enumerate(range(0, M, block_size)):
#         end = min(i + block_size, M)
#         k   = end - i

#         w_block = W_abs[:, i:end]
#         m_block = mask_mat[:, i:end]
#         s_block = sign_mat[:, i:end]

#         H_block = H_blocks_layer[block_idx].to(device)
#         if H_block.shape[0] != k:
#             H_block = H_block[:k, :k]

#         w_eff  = w_block * m_block
#         pruned = m_block.sum(dim=1) < 1e-8

#         w_sq_mean = (w_eff ** 2).mean(dim=1).clamp(min=1e-8)
#         alpha     = torch.sqrt(w_sq_mean).clamp(min=1e-4)
#         alpha_min = 0.05 * w_eff.abs().mean(dim=1).clamp(min=1e-8)
#         alpha_max = 20.0 * w_eff.abs().mean(dim=1).clamp(min=1e-8)

#         Hw = w_eff @ H_block.T

#         best_loss  = torch.full((N,), float('inf'), device=device)
#         best_alpha = alpha.clone()
#         best_bias  = torch.full((N,), preshifted_center,
#                                 device=device, dtype=torch.long)
#         best_b     = torch.zeros_like(w_eff)

#         for bias_candidate in bias_candidates:
#             alpha_tmp = alpha.clone()

#             for _ in range(5):
#                 _, _, b = assign_fp4_dynamic_batched(
#                     w_block, alpha_tmp, e_bits, m_bits, bias=bias_candidate)
#                 b_eff = b * m_block
#                 Hb    = b_eff @ H_block.T
#                 num   = (b_eff * Hw).sum(dim=1)
#                 den   = (b_eff * Hb).sum(dim=1) + 1e-8
#                 alpha_new = num / den
#                 alpha_tmp = torch.clamp(alpha_new, min=alpha_min, max=alpha_max)
#                 del b, Hb, num, den, alpha_new

#             b_eff    = b_eff
#             residual = w_eff - alpha_tmp.unsqueeze(1) * b_eff
#             Hr       = residual @ H_block.T
#             loss     = (residual * Hr).sum(dim=1)

#             improved   = loss < best_loss
#             best_loss  = torch.where(improved, loss, best_loss)
#             best_alpha = torch.where(improved, alpha_tmp, best_alpha)
#             best_bias  = torch.where(
#                 improved,
#                 torch.full_like(best_bias, bias_candidate),
#                 best_bias)
#             best_b = torch.where(
#                 improved.unsqueeze(1).expand_as(b_eff),
#                 b_eff, best_b)

#             del alpha_tmp, b_eff, residual, Hr, loss, improved

#         del Hw, alpha, alpha_min, alpha_max, w_sq_mean
#         torch.cuda.empty_cache()

#         alpha_q = quantize_scale_batched(best_alpha, e_bits_scale, m_bits_scale)

#         _, _, b_final = assign_fp4_dynamic_batched(
#             w_block, alpha_q, e_bits, m_bits, bias_per_row=best_bias)

#         w_hat = alpha_q.unsqueeze(1) * b_final
#         w_hat = w_hat * m_block * s_block
#         w_hat[pruned]   = 0.0
#         alpha_q[pruned] = 1.0

#         W_q[:, i:end]       = w_hat.cpu()
#         alpha_out[:, i:end] = alpha_q.unsqueeze(1).expand(-1, k).cpu()
#         bias_out[:, i:end]  = best_bias.unsqueeze(1).float().expand(-1, k).cpu()

#         del w_block, m_block, s_block, w_eff, pruned
#         del H_block, best_loss, best_alpha, best_bias, best_b
#         del alpha_q, b_final, w_hat
#         torch.cuda.empty_cache()

#     # Restore original weights
#     layer.weight.data = original_weight
#     del W_abs, sign_mat, mask_mat, W_mat
#     torch.cuda.empty_cache()

#     n_blocks        = math.ceil(M / block_size)
#     alpha_out_saved = alpha_out[:, ::block_size][:, :n_blocks].clone()
#     bias_out_saved  = bias_out[:, ::block_size][:, :n_blocks].long()

#     sign_f  = torch.sign(W_q)
#     W_abs_f = W_q.abs()
#     del W_q

#     layer_W        = layer.weight.data
#     original_shape = layer_W.shape

#     if layer_W.dim() == 4:
#         N4        = original_shape[0]
#         W_abs_f   = W_abs_f.view(N4, -1)
#         alpha_out = alpha_out.view(N4, -1)
#         bias_out  = bias_out.view(N4, -1)
#         sign_f    = sign_f.view(N4, -1)

#     e_out = torch.zeros(W_abs_f.shape, dtype=torch.long)
#     m_out = torch.zeros(W_abs_f.shape, dtype=torch.long)

#     for block_idx, i in enumerate(range(0, M, block_size)):
#         end = min(i + block_size, M)

#         bias_col  = bias_out[:, i].long().to(device)
#         alpha_col = alpha_out[:, i].to(device)
#         w_col     = W_abs_f[:, i:end].to(device)

#         for bias_val in bias_col.unique():
#             rows = (bias_col == bias_val).nonzero(as_tuple=True)[0]
#             e_b, m_b, _ = assign_fp4_dynamic(
#                 w_col[rows], alpha_col[rows], e_bits, m_bits,
#                 bias=int(bias_val.item()))
#             e_out[rows, i:end] = e_b.cpu()
#             m_out[rows, i:end] = m_b.cpu()
#             del e_b, m_b

#         del bias_col, alpha_col, w_col
#         torch.cuda.empty_cache()

#     del W_abs_f

#     bias_mat     = bias_out.float()
#     base         = alpha_out * (2.0 ** (e_out.float() - bias_mat))
#     fine         = base * m_out.float() / (2**m_bits) if m_bits > 0 else 0.0
#     weight_q_cpu = (base + fine) * sign_f

#     del alpha_out, bias_out, bias_mat, e_out, m_out, base, sign_f
#     if m_bits > 0:
#         del fine

#     if layer_W.dim() == 4:
#         weight_q_cpu    = weight_q_cpu.view(original_shape)
#         alpha_out_saved = alpha_out_saved.view(original_shape[0], -1)
#         bias_out_saved  = bias_out_saved.view(original_shape[0], -1)

#     return {
#         "weight_q":    weight_q_cpu.to(device),
#         "alpha":       alpha_out_saved,
#         "bias":        bias_out_saved,
#         "rho":         rho.item() if rho is not None else None,
#         "b_ori":       b_ori.cpu() if b_ori is not None else None,
#     }


# def reconstruct_layer_fp_blockdiag_scaled_v5_preshifted(
#     layer,
#     H_blocks_layer,
#     block_size,
#     e_bits,
#     m_bits,
#     e_bits_scale,
#     m_bits_scale,
#     device,
#     preshifted_center=None,
# ):
#     """
#     v5 reconstruction with bias search centered on preshifted_center.
#     Expects layer.weight.data to already be the adjusted weights
#     (pre-multiplied by 2^(-b_ori) per input block).
#     Only difference from v5: bias_candidates centered on preshifted_center.
#     """
#     W    = layer.weight.data.to(device)
#     mask = (W != 0).float()

#     if W.dim() == 4:
#         W_mat    = W.view(W.shape[0], -1)
#         mask_mat = mask.view(mask.shape[0], -1)
#     else:
#         W_mat    = W
#         mask_mat = mask

#     N, M = W_mat.shape

#     sign_mat = torch.sign(W_mat)
#     W_abs    = W_mat.abs()

#     del W, mask
#     torch.cuda.empty_cache()

#     W_q       = torch.zeros(N, M, dtype=torch.float32)
#     alpha_out = torch.zeros(N, M, dtype=torch.float32)
#     bias_out  = torch.zeros(N, M, dtype=torch.float32)

#     # ── Key difference: bias search centered on preshifted_center ─────────
#     default_bias = preshifted_center if preshifted_center is not None \
#                    else 2**(e_bits - 1) - 1
#     bias_radius  = max(1, 2**(e_bits - 2))
#     bias_candidates = [
#         b for b in range(default_bias - bias_radius,
#                          default_bias + bias_radius + 1)
#         if 0 <= b <= 2**e_bits - 1
#     ]
#     print(f"  Bias candidates: {bias_candidates} (center={default_bias})")

#     # ── Rest identical to v5 ──────────────────────────────────────────────
#     for block_idx, i in enumerate(range(0, M, block_size)):
#         end = min(i + block_size, M)
#         k   = end - i

#         w_block = W_abs[:, i:end]
#         m_block = mask_mat[:, i:end]
#         s_block = sign_mat[:, i:end]

#         H_block = H_blocks_layer[block_idx].to(device)
#         if H_block.shape[0] != k:
#             H_block = H_block[:k, :k]

#         w_eff  = w_block * m_block
#         pruned = m_block.sum(dim=1) < 1e-8

#         w_sq_mean = (w_eff ** 2).mean(dim=1).clamp(min=1e-8)
#         alpha     = torch.sqrt(w_sq_mean).clamp(min=1e-4)
#         alpha_min = 0.05 * w_eff.abs().mean(dim=1).clamp(min=1e-8)
#         alpha_max = 20.0 * w_eff.abs().mean(dim=1).clamp(min=1e-8)

#         Hw = w_eff @ H_block.T

#         best_loss  = torch.full((N,), float('inf'), device=device)
#         best_alpha = alpha.clone()
#         best_bias  = torch.full((N,), default_bias,
#                                 device=device, dtype=torch.long)
#         best_b     = torch.zeros_like(w_eff)

#         for bias_candidate in bias_candidates:
#             alpha_tmp = alpha.clone()

#             for _ in range(5):
#                 _, _, b = assign_fp4_dynamic_batched(
#                     w_block, alpha_tmp, e_bits, m_bits,
#                     bias=bias_candidate)
#                 b_eff = b * m_block
#                 Hb    = b_eff @ H_block.T
#                 num   = (b_eff * Hw).sum(dim=1)
#                 den   = (b_eff * Hb).sum(dim=1) + 1e-8
#                 alpha_new = num / den
#                 alpha_tmp = torch.clamp(alpha_new,
#                                         min=alpha_min, max=alpha_max)
#                 del b, Hb, num, den, alpha_new

#             b_eff    = b_eff
#             residual = w_eff - alpha_tmp.unsqueeze(1) * b_eff
#             Hr       = residual @ H_block.T
#             loss     = (residual * Hr).sum(dim=1)

#             improved   = loss < best_loss
#             best_loss  = torch.where(improved, loss, best_loss)
#             best_alpha = torch.where(improved, alpha_tmp, best_alpha)
#             best_bias  = torch.where(
#                 improved,
#                 torch.full_like(best_bias, bias_candidate),
#                 best_bias)
#             best_b = torch.where(
#                 improved.unsqueeze(1).expand_as(b_eff),
#                 b_eff, best_b)

#             del alpha_tmp, b_eff, residual, Hr, loss, improved

#         del Hw, alpha, alpha_min, alpha_max, w_sq_mean
#         torch.cuda.empty_cache()

#         alpha_q = quantize_scale_batched(best_alpha, e_bits_scale, m_bits_scale)

#         _, _, b_final = assign_fp4_dynamic_batched(
#             w_block, alpha_q, e_bits, m_bits, bias_per_row=best_bias)

#         w_hat = alpha_q.unsqueeze(1) * b_final
#         w_hat = w_hat * m_block * s_block
#         w_hat[pruned]   = 0.0
#         alpha_q[pruned] = 1.0

#         W_q[:, i:end]       = w_hat.cpu()
#         alpha_out[:, i:end] = alpha_q.unsqueeze(1).expand(-1, k).cpu()
#         bias_out[:, i:end]  = best_bias.unsqueeze(1).float().expand(-1, k).cpu()

#         del w_block, m_block, s_block, w_eff, pruned
#         del H_block, best_loss, best_alpha, best_bias, best_b
#         del alpha_q, b_final, w_hat
#         torch.cuda.empty_cache()

#     del W_abs, sign_mat, mask_mat, W_mat
#     torch.cuda.empty_cache()

#     n_blocks        = math.ceil(M / block_size)
#     alpha_out_saved = alpha_out[:, ::block_size][:, :n_blocks].clone()
#     bias_out_saved  = bias_out[:, ::block_size][:, :n_blocks].long()

#     sign_f  = torch.sign(W_q)
#     W_abs_f = W_q.abs()
#     del W_q

#     layer_W        = layer.weight.data
#     original_shape = layer_W.shape

#     if layer_W.dim() == 4:
#         N4        = original_shape[0]
#         W_abs_f   = W_abs_f.view(N4, -1)
#         alpha_out = alpha_out.view(N4, -1)
#         bias_out  = bias_out.view(N4, -1)
#         sign_f    = sign_f.view(N4, -1)

#     e_out = torch.zeros(W_abs_f.shape, dtype=torch.long)
#     m_out = torch.zeros(W_abs_f.shape, dtype=torch.long)

#     for block_idx, i in enumerate(range(0, M, block_size)):
#         end = min(i + block_size, M)

#         bias_col  = bias_out[:, i].long().to(device)
#         alpha_col = alpha_out[:, i].to(device)
#         w_col     = W_abs_f[:, i:end].to(device)

#         for bias_val in bias_col.unique():
#             rows = (bias_col == bias_val).nonzero(as_tuple=True)[0]
#             e_b, m_b, _ = assign_fp4_dynamic(
#                 w_col[rows], alpha_col[rows], e_bits, m_bits,
#                 bias=int(bias_val.item()))
#             e_out[rows, i:end] = e_b.cpu()
#             m_out[rows, i:end] = m_b.cpu()
#             del e_b, m_b

#         del bias_col, alpha_col, w_col
#         torch.cuda.empty_cache()

#     del W_abs_f

#     bias_mat     = bias_out.float()
#     base         = alpha_out * (2.0 ** (e_out.float() - bias_mat))
#     fine         = base * m_out.float() / (2**m_bits) if m_bits > 0 else 0.0
#     weight_q_cpu = (base + fine) * sign_f

#     del alpha_out, bias_out, bias_mat, e_out, m_out, base, sign_f
#     if m_bits > 0:
#         del fine

#     if layer_W.dim() == 4:
#         weight_q_cpu    = weight_q_cpu.view(original_shape)
#         alpha_out_saved = alpha_out_saved.view(original_shape[0], -1)
#         bias_out_saved  = bias_out_saved.view(original_shape[0], -1)

#     return {
#         "weight_q": weight_q_cpu.to(device),
#         "alpha":    alpha_out_saved,
#         "bias":     bias_out_saved,
#     }


# def reconstruct_layer_fp_blockdiag_scaled_v5_preshifted(
#     layer,
#     H_blocks_layer,
#     block_size,
#     e_bits,
#     m_bits,
#     e_bits_scale,
#     m_bits_scale,
#     device,
#     preshifted_center=None,
# ):
#     """
#     v5 reconstruction operating on W_adjusted = W * 2^(-b_ori) with
#     Hessian H = (X * 2^(b_ori))^T (X * 2^(b_ori)).
#     The only difference from the original v5 is that bias_candidates
#     is centered on preshifted_center rather than the default bias.
#     """
#     W    = layer.weight.data.to(device)
#     mask = (W != 0).float()

#     if W.dim() == 4:
#         W_mat    = W.view(W.shape[0], -1)
#         mask_mat = mask.view(mask.shape[0], -1)
#     else:
#         W_mat    = W
#         mask_mat = mask

#     N, M = W_mat.shape

#     sign_mat = torch.sign(W_mat)
#     W_abs    = W_mat.abs()

#     del W, mask
#     torch.cuda.empty_cache()

#     W_q       = torch.zeros(N, M, dtype=torch.float32)
#     alpha_out = torch.zeros(N, M, dtype=torch.float32)
#     bias_out  = torch.zeros(N, M, dtype=torch.float32)

#     # Bias search centered on preshifted_center
#     # For E2M1: valid range is [0, 3], default center is 1
#     # preshifted_center steers the search toward the region that best
#     # represents the scaled weight distribution W * 2^(-b_ori)
#     default_bias = preshifted_center if preshifted_center is not None \
#                    else 2**(e_bits - 1) - 1
#     bias_radius  = max(1, 2**(e_bits - 2))
#     bias_candidates = [
#         b for b in range(default_bias - bias_radius,
#                          default_bias + bias_radius + 1)
#         if 0 <= b <= 2**e_bits - 1
#     ]
#     print(f"  Bias candidates: {bias_candidates} (center={default_bias})")

#     for block_idx, i in enumerate(range(0, M, block_size)):
#         end = min(i + block_size, M)
#         k   = end - i

#         w_block = W_abs[:, i:end]
#         m_block = mask_mat[:, i:end]
#         s_block = sign_mat[:, i:end]

#         H_block = H_blocks_layer[block_idx].to(device)
#         if H_block.shape[0] != k:
#             H_block = H_block[:k, :k]

#         w_eff  = w_block * m_block
#         pruned = m_block.sum(dim=1) < 1e-8

#         w_sq_mean = (w_eff ** 2).mean(dim=1).clamp(min=1e-8)
#         alpha     = torch.sqrt(w_sq_mean).clamp(min=1e-4)
#         alpha_min = 0.05 * w_eff.abs().mean(dim=1).clamp(min=1e-8)
#         alpha_max = 20.0 * w_eff.abs().mean(dim=1).clamp(min=1e-8)

#         Hw = w_eff @ H_block.T

#         best_loss  = torch.full((N,), float('inf'), device=device)
#         best_alpha = alpha.clone()
#         best_bias  = torch.full((N,), default_bias,
#                                 device=device, dtype=torch.long)
#         best_b     = torch.zeros_like(w_eff)

#         for bias_candidate in bias_candidates:
#             alpha_tmp = alpha.clone()

#             for _ in range(5):
#                 _, _, b = assign_fp4_dynamic_batched(
#                     w_block, alpha_tmp, e_bits, m_bits,
#                     bias=bias_candidate)
#                 b_eff = b * m_block
#                 Hb    = b_eff @ H_block.T
#                 num   = (b_eff * Hw).sum(dim=1)
#                 den   = (b_eff * Hb).sum(dim=1) + 1e-8
#                 alpha_new = num / den
#                 alpha_tmp = torch.clamp(alpha_new,
#                                         min=alpha_min, max=alpha_max)
#                 del b, Hb, num, den, alpha_new

#             b_eff    = b_eff
#             residual = w_eff - alpha_tmp.unsqueeze(1) * b_eff
#             Hr       = residual @ H_block.T
#             loss     = (residual * Hr).sum(dim=1)

#             improved   = loss < best_loss
#             best_loss  = torch.where(improved, loss, best_loss)
#             best_alpha = torch.where(improved, alpha_tmp, best_alpha)
#             best_bias  = torch.where(
#                 improved,
#                 torch.full_like(best_bias, bias_candidate),
#                 best_bias)
#             best_b = torch.where(
#                 improved.unsqueeze(1).expand_as(b_eff),
#                 b_eff, best_b)

#             del alpha_tmp, b_eff, residual, Hr, loss, improved

#         del Hw, alpha, alpha_min, alpha_max, w_sq_mean
#         torch.cuda.empty_cache()

#         alpha_q = quantize_scale_batched(best_alpha, e_bits_scale, m_bits_scale)

#         _, _, b_final = assign_fp4_dynamic_batched(
#             w_block, alpha_q, e_bits, m_bits, bias_per_row=best_bias)

#         w_hat = alpha_q.unsqueeze(1) * b_final
#         w_hat = w_hat * m_block * s_block
#         w_hat[pruned]   = 0.0
#         alpha_q[pruned] = 1.0

#         W_q[:, i:end]       = w_hat.cpu()
#         alpha_out[:, i:end] = alpha_q.unsqueeze(1).expand(-1, k).cpu()
#         bias_out[:, i:end]  = best_bias.unsqueeze(1).float().expand(-1, k).cpu()

#         del w_block, m_block, s_block, w_eff, pruned
#         del H_block, best_loss, best_alpha, best_bias, best_b
#         del alpha_q, b_final, w_hat
#         torch.cuda.empty_cache()

#     del W_abs, sign_mat, mask_mat, W_mat
#     torch.cuda.empty_cache()

#     n_blocks        = math.ceil(M / block_size)
#     alpha_out_saved = alpha_out[:, ::block_size][:, :n_blocks].clone()
#     bias_out_saved  = bias_out[:, ::block_size][:, :n_blocks].long()

#     sign_f  = torch.sign(W_q)
#     W_abs_f = W_q.abs()
#     del W_q

#     layer_W        = layer.weight.data
#     original_shape = layer_W.shape

#     if layer_W.dim() == 4:
#         N4        = original_shape[0]
#         W_abs_f   = W_abs_f.view(N4, -1)
#         alpha_out = alpha_out.view(N4, -1)
#         bias_out  = bias_out.view(N4, -1)
#         sign_f    = sign_f.view(N4, -1)

#     e_out = torch.zeros(W_abs_f.shape, dtype=torch.long)
#     m_out = torch.zeros(W_abs_f.shape, dtype=torch.long)

#     for block_idx, i in enumerate(range(0, M, block_size)):
#         end = min(i + block_size, M)

#         bias_col  = bias_out[:, i].long().to(device)
#         alpha_col = alpha_out[:, i].to(device)
#         w_col     = W_abs_f[:, i:end].to(device)

#         for bias_val in bias_col.unique():
#             rows = (bias_col == bias_val).nonzero(as_tuple=True)[0]
#             e_b, m_b, _ = assign_fp4_dynamic(
#                 w_col[rows], alpha_col[rows], e_bits, m_bits,
#                 bias=int(bias_val.item()))
#             e_out[rows, i:end] = e_b.cpu()
#             m_out[rows, i:end] = m_b.cpu()
#             del e_b, m_b

#         del bias_col, alpha_col, w_col
#         torch.cuda.empty_cache()

#     del W_abs_f

#     bias_mat     = bias_out.float()
#     base         = alpha_out * (2.0 ** (e_out.float() - bias_mat))
#     fine         = base * m_out.float() / (2**m_bits) if m_bits > 0 else 0.0
#     weight_q_cpu = (base + fine) * sign_f

#     del alpha_out, bias_out, bias_mat, e_out, m_out, base, sign_f
#     if m_bits > 0:
#         del fine

#     if layer_W.dim() == 4:
#         weight_q_cpu    = weight_q_cpu.view(original_shape)
#         alpha_out_saved = alpha_out_saved.view(original_shape[0], -1)
#         bias_out_saved  = bias_out_saved.view(original_shape[0], -1)

#     return {
#         "weight_q": weight_q_cpu.to(device),
#         "alpha":    alpha_out_saved,
#         "bias":     bias_out_saved,
#     }

def _apply_beta_scaling_only(x, act_b_ori, block_size):
    """
    Apply 2^(b_ori_k) scaling per block to activations but NO quantization.
    This is the 'true W4A16' equivalent for the preshifted method:
    evaluates F.linear(x * beta, weight_q) vs F.linear(x, W)
    to isolate weight reconstruction quality from activation quantization quality.
    """
    orig_shape = x.shape
    x_2d       = x.reshape(-1, orig_shape[-1]).float()
    N, K       = x_2d.shape

    if act_b_ori is None:
        return x.to(x.dtype)

    pad      = (block_size - K % block_size) % block_size
    x_pad    = F.pad(x_2d, (0, pad))
    x_blocks = x_pad.view(N, -1, block_size)
    n_blocks = x_blocks.shape[1]

    # Scale each block by 2^(b_ori_k) — coordinate change only, no quantization
    b_ori_scales = torch.zeros(n_blocks, device=x.device)
    for k in range(n_blocks):
        if k < len(act_b_ori):
            b_ori_scales[k] = act_b_ori[k].float().item()
        else:
            b_ori_scales[k] = 0.0

    activation_scale = 2.0 ** b_ori_scales   # (n_blocks,)
    x_blocks = x_blocks * activation_scale.unsqueeze(0).unsqueeze(-1)

    x_out = x_blocks.view(N, -1)[:, :K]
    return x_out.reshape(orig_shape).to(x.dtype)


class preshifted_beta_only_mode:
    """
    Context manager that applies beta scaling to activations in full precision
    but skips FP4 quantization. Use this to evaluate true W4A16 quality
    for the preshifted method:
        out = F.linear(x * 2^b_ori, weight_q)  [full precision, no quant]
    vs
        out = F.linear(x, W)                    [original]

    If this PPL is close to 27.6, weight reconstruction is good and the
    gap in W4A4 comes from activation quantization noise.
    If this PPL is still bad, the weight reconstruction itself is broken.
    """
    def __init__(self, model):
        self.model   = model
        self.saved   = {}

    def __enter__(self):
        for name, module in self.model.named_modules():
            if type(module).__name__ == "QuantLinearFP":
                self.saved[name] = module.act_quant_mode
                module.act_quant_mode = "preshifted_beta_only"
        return self

    def __exit__(self, *args):
        for name, module in self.model.named_modules():
            if type(module).__name__ == "QuantLinearFP":
                module.act_quant_mode = self.saved.get(name, None)



def reconstruct_layer_fp_blockdiag_scaled_v5_preshifted(
    layer,
    H_blocks_layer,
    block_size,
    e_bits,
    m_bits,
    e_bits_scale,
    m_bits_scale,
    device,
    b_ori=None,
):
    """
    v5 reconstruction with per-block bias search.
    layer.weight.data must already be W_adjusted = W * 2^(-b_ori).
    For each block k, the bias search is centered on:
        default_bias - b_ori[k]
    because the weights in block k were scaled down by 2^(-b_ori[k]),
    so the FP4 exponent bias must shift down by the same amount to
    cover the correct weight magnitude range.
    """
    W    = layer.weight.data.to(device)
    mask = (W != 0).float()

    if W.dim() == 4:
        W_mat    = W.view(W.shape[0], -1)
        mask_mat = mask.view(mask.shape[0], -1)
    else:
        W_mat    = W
        mask_mat = mask

    N, M = W_mat.shape

    sign_mat = torch.sign(W_mat)
    W_abs    = W_mat.abs()

    del W, mask
    torch.cuda.empty_cache()

    W_q       = torch.zeros(N, M, dtype=torch.float32)
    alpha_out = torch.zeros(N, M, dtype=torch.float32)
    bias_out  = torch.zeros(N, M, dtype=torch.float32)

    default_bias = 2**(e_bits - 1) - 1  # = 1 for E2M1
    bias_radius  = max(1, 2**(e_bits - 2))

    for block_idx, i in enumerate(range(0, M, block_size)):
        end = min(i + block_size, M)
        k   = end - i

        w_block = W_abs[:, i:end]
        m_block = mask_mat[:, i:end]
        s_block = sign_mat[:, i:end]

        H_block = H_blocks_layer[block_idx].to(device)
        if H_block.shape[0] != k:
            H_block = H_block[:k, :k]

        # Per-block bias center: shift down by b_ori[block_idx]
        # because weights were scaled down by 2^(-b_ori[block_idx])
        if b_ori is not None and block_idx < len(b_ori):
            block_b_ori = int(b_ori[block_idx].item())
        else:
            block_b_ori = 0

        # block_center    = default_bias - block_b_ori
        block_center    = default_bias + block_b_ori
        block_center    = max(0, min(2**e_bits - 1, block_center))
        bias_candidates = [
            b for b in range(block_center - bias_radius,
                             block_center + bias_radius + 1)
            if 0 <= b <= 2**e_bits - 1
        ]

        w_eff  = w_block * m_block
        pruned = m_block.sum(dim=1) < 1e-8

        w_sq_mean = (w_eff ** 2).mean(dim=1).clamp(min=1e-8)
        alpha     = torch.sqrt(w_sq_mean).clamp(min=1e-4)
        alpha_min = 0.05 * w_eff.abs().mean(dim=1).clamp(min=1e-8)
        alpha_max = 20.0 * w_eff.abs().mean(dim=1).clamp(min=1e-8)

        Hw = w_eff @ H_block.T

        best_loss  = torch.full((N,), float('inf'), device=device)
        best_alpha = alpha.clone()
        best_bias  = torch.full((N,), block_center,
                                device=device, dtype=torch.long)
        best_b     = torch.zeros_like(w_eff)

        for bias_candidate in bias_candidates:
            alpha_tmp = alpha.clone()

            for _ in range(5):
                _, _, b = assign_fp4_dynamic_batched(
                    w_block, alpha_tmp, e_bits, m_bits,
                    bias=bias_candidate)
                b_eff = b * m_block
                Hb    = b_eff @ H_block.T
                num   = (b_eff * Hw).sum(dim=1)
                den   = (b_eff * Hb).sum(dim=1) + 1e-8
                alpha_new = num / den
                alpha_tmp = torch.clamp(alpha_new,
                                        min=alpha_min, max=alpha_max)
                del b, Hb, num, den, alpha_new

            b_eff    = b_eff
            residual = w_eff - alpha_tmp.unsqueeze(1) * b_eff
            Hr       = residual @ H_block.T
            loss     = (residual * Hr).sum(dim=1)

            improved   = loss < best_loss
            best_loss  = torch.where(improved, loss, best_loss)
            best_alpha = torch.where(improved, alpha_tmp, best_alpha)
            best_bias  = torch.where(
                improved,
                torch.full_like(best_bias, bias_candidate),
                best_bias)
            best_b = torch.where(
                improved.unsqueeze(1).expand_as(b_eff),
                b_eff, best_b)

            del alpha_tmp, b_eff, residual, Hr, loss, improved

        del Hw, alpha, alpha_min, alpha_max, w_sq_mean
        torch.cuda.empty_cache()

        alpha_q = quantize_scale_batched(best_alpha, e_bits_scale, m_bits_scale)

        _, _, b_final = assign_fp4_dynamic_batched(
            w_block, alpha_q, e_bits, m_bits, bias_per_row=best_bias)

        w_hat = alpha_q.unsqueeze(1) * b_final
        w_hat = w_hat * m_block * s_block
        w_hat[pruned]   = 0.0
        alpha_q[pruned] = 1.0

        W_q[:, i:end]       = w_hat.cpu()
        alpha_out[:, i:end] = alpha_q.unsqueeze(1).expand(-1, k).cpu()
        bias_out[:, i:end]  = best_bias.unsqueeze(1).float().expand(-1, k).cpu()

        del w_block, m_block, s_block, w_eff, pruned
        del H_block, best_loss, best_alpha, best_bias, best_b
        del alpha_q, b_final, w_hat
        torch.cuda.empty_cache()

    del W_abs, sign_mat, mask_mat, W_mat
    torch.cuda.empty_cache()

    n_blocks        = math.ceil(M / block_size)
    alpha_out_saved = alpha_out[:, ::block_size][:, :n_blocks].clone()
    bias_out_saved  = bias_out[:, ::block_size][:, :n_blocks].long()

    sign_f  = torch.sign(W_q)
    W_abs_f = W_q.abs()
    del W_q

    layer_W        = layer.weight.data
    original_shape = layer_W.shape

    if layer_W.dim() == 4:
        N4        = original_shape[0]
        W_abs_f   = W_abs_f.view(N4, -1)
        alpha_out = alpha_out.view(N4, -1)
        bias_out  = bias_out.view(N4, -1)
        sign_f    = sign_f.view(N4, -1)

    e_out = torch.zeros(W_abs_f.shape, dtype=torch.long)
    m_out = torch.zeros(W_abs_f.shape, dtype=torch.long)

    for block_idx, i in enumerate(range(0, M, block_size)):
        end = min(i + block_size, M)

        bias_col  = bias_out[:, i].long().to(device)
        alpha_col = alpha_out[:, i].to(device)
        w_col     = W_abs_f[:, i:end].to(device)

        for bias_val in bias_col.unique():
            rows = (bias_col == bias_val).nonzero(as_tuple=True)[0]
            e_b, m_b, _ = assign_fp4_dynamic(
                w_col[rows], alpha_col[rows], e_bits, m_bits,
                bias=int(bias_val.item()))
            e_out[rows, i:end] = e_b.cpu()
            m_out[rows, i:end] = m_b.cpu()
            del e_b, m_b

        del bias_col, alpha_col, w_col
        torch.cuda.empty_cache()

    del W_abs_f

    bias_mat     = bias_out.float()
    base         = alpha_out * (2.0 ** (e_out.float() - bias_mat))
    fine         = base * m_out.float() / (2**m_bits) if m_bits > 0 else 0.0
    weight_q_cpu = (base + fine) * sign_f

    del alpha_out, bias_out, bias_mat, e_out, m_out, base, sign_f
    if m_bits > 0:
        del fine

    if layer_W.dim() == 4:
        weight_q_cpu    = weight_q_cpu.view(original_shape)
        alpha_out_saved = alpha_out_saved.view(original_shape[0], -1)
        bias_out_saved  = bias_out_saved.view(original_shape[0], -1)

    return {
        "weight_q": weight_q_cpu.to(device),
        "alpha":    alpha_out_saved,
        "bias":     bias_out_saved,
    }




def compute_block_preshifted_bias(x_block_max, e_bits, m_bits):
    """
    Compute real-valued per-block bias from calibration block maxima.
    
    x_block_max: (n_blocks,) — max abs activation per input block
    Returns b_tilde: (n_blocks,) real-valued per-block bias
    """
    x_block_max = x_block_max.clamp(min=1e-8)
    b_tilde = (2**e_bits
               - torch.log2(x_block_max)
               + math.log2(2.0 - 2.0**(-m_bits))
               - 1.0)
    return b_tilde


def decompose_block_preshifted_bias(b_tilde, e_bits):
    """
    Decompose per-block real-valued bias into:
      rho:   scalar tensor-wise real-valued exponent (absorbs overflow)
      b_ori: (n_blocks,) integer in [0, 2^e_bits - 1]

    The key insight: b_ori is CLAMPED to [0, 2^e_bits-1].
    Any value of b_tilde outside this range after subtracting rho
    is absorbed into rho rather than being lost.

    rho is chosen as min(b_tilde) so b_ori is always >= 0.
    Values where b_tilde - rho > 2^e_bits - 1 are clamped,
    meaning those blocks get the maximum integer correction
    and the remainder stays in rho (slight approximation for
    those blocks, but rho is then refined by the bias search).
    """
    rho   = b_tilde.min()                              # scalar
    b_ori = torch.clamp(
        torch.round(b_tilde - rho).long(),
        0, 2**e_bits - 1                               # [0, 3] for E2M1
    )
    return rho, b_ori


def collect_per_block_activation_max(model, calib_loader, device,
                                      block_size, num_batches=4):
    """
    Collect per-INPUT-BLOCK max activation magnitude.
    Blocks are along the in_features (K) dimension, matching
    your weight quantization block structure exactly.

    Returns dict: layer_name -> (n_blocks,) tensor
    """
    act_block_max = {}
    hooks = []

    def make_hook(name, in_features):
        n_blocks = math.ceil(in_features / block_size)

        def hook(module, inp, out):
            x      = inp[0].detach().float()
            x_flat = x.reshape(-1, in_features)        # (N_tokens, K)
            N, K   = x_flat.shape

            pad     = (block_size - K % block_size) % block_size
            x_pad   = F.pad(x_flat, (0, pad))
            x_blocks = x_pad.view(N, n_blocks, block_size)

            # Max over tokens AND block elements -> (n_blocks,)
            blk_max = x_blocks.abs().amax(dim=(0, 2))

            if name not in act_block_max:
                act_block_max[name] = blk_max.cpu()
            else:
                act_block_max[name] = torch.maximum(
                    act_block_max[name], blk_max.cpu()
                )
        return hook

    for name, module in model.named_modules():
        if type(module).__name__ == "QuantLinearFP":
            in_f = module.linear.weight.shape[1]
            hooks.append(
                module.register_forward_hook(make_hook(name, in_f))
            )

    model.eval()
    batches_run = 0
    with torch.no_grad():
        for batch in calib_loader:
            if batch is None:
                continue
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if x is None:
                continue
            model(x.to(device))
            batches_run += 1
            if batches_run >= num_batches:
                break

    for h in hooks:
        h.remove()

    print(f"Collected per-block activation stats for "
          f"{len(act_block_max)} layers over {batches_run} batches")
    return act_block_max


def preshifted_bias_and_adjust_weights_blockwise(
    W, blk_max, block_size, e_bits, m_bits, device
):
    """
    Block-wise pre-shifted bias computation and weight adjustment.

    W:        (out_features, in_features)
    blk_max:  (n_blocks,) — per-input-block activation max

    The bias correction is absorbed into weights by scaling each
    block of input weight columns by 2^(-b_ori_i).

    Returns:
        W_adjusted:       (out_features, in_features)
        rho:              scalar — tensor-wise exponent for activation scale
        b_ori:            (n_blocks,) long — integer bias per input block
        preshifted_center: int — starting point for v5 bias search
    """
    W       = W.to(device).float()
    blk_max = blk_max.to(device).clamp(min=1e-8)
    out_f, in_f = W.shape
    n_blocks    = math.ceil(in_f / block_size)

    # Compute per-block real-valued bias
    b_tilde = compute_block_preshifted_bias(blk_max, e_bits, m_bits)

    # Decompose — b_ori clamped to [0, 3], overflow absorbed into rho
    rho, b_ori = decompose_block_preshifted_bias(b_tilde, e_bits)

    # Absorb per-block correction into weights
    # Each block of in_features columns gets multiplied by 2^(-b_ori_i)
    beta = 2.0 ** (-b_ori.float())                    # (n_blocks,)

    pad   = (block_size - in_f % block_size) % block_size
    W_pad = F.pad(W, (0, pad))                        # (out, in_pad)
    W_blocks = W_pad.view(out_f, n_blocks, block_size) # (out, n_blocks, bs)

    # Scale each input block
    W_blocks = W_blocks * beta.view(1, n_blocks, 1)   # broadcast

    W_adjusted = W_blocks.view(out_f, -1)[:, :in_f]  # remove padding

    # preshifted_center: integer closest to rho, clamped to valid range
    preshifted_center = int(
        torch.round(rho).clamp(0, 2**e_bits - 1).item()
    )

    print(f"    b_tilde range: [{b_tilde.min():.2f}, {b_tilde.max():.2f}]")
    print(f"    rho={rho.item():.3f}, center={preshifted_center}")
    print(f"    b_ori range: [{b_ori.min().item()}, {b_ori.max().item()}]")
    print(f"    beta range: [{beta.min():.4f}, {beta.max():.4f}]")

    return W_adjusted, rho, b_ori, preshifted_center

def reconstruct_layer_fp_blockdiag_scaled_v4_fast(
    layer, H_blocks_layer, block_size, e_bits, m_bits, e_bits_scale, m_bits_scale, device, B_chunk=16
):
    W = layer.weight.data.to(device)
    mask = (W != 0).float()
    orig_shape = W.shape
    N, M = W.view(orig_shape[0], -1).shape
    W_mat = W.view(N, -1)
    mask_mat = mask.view(N, -1)
    sign = torch.sign(W_mat)
    
    # 1. Padding & Blocking
    pad = (block_size - M % block_size) % block_size
    M_pad = M + pad
    n_blocks = M_pad // block_size
    W_p = F.pad(W_mat.abs(), (0, pad))
    mask_p = F.pad(mask_mat, (0, pad))
    W_blocks = W_p.view(N, n_blocks, block_size)
    mask_blocks = mask_p.view(N, n_blocks, block_size)

    # 2. Hessian Stacking (CRITICAL: Validate index alignment)
    # If H_blocks_layer has n_blocks, we stack them. 
    H_all = torch.stack([F.pad(H_blocks_layer[b].to(device).float(), 
                               (0, block_size - H_blocks_layer[b].shape[0], 
                                0, block_size - H_blocks_layer[b].shape[0])) 
                         for b in range(n_blocks)])

    # 3. Optimization Setup
    w_eff = W_blocks * mask_blocks
    # Match the row-wise mean exactly
    block_means = w_eff.abs().mean(dim=-1, keepdim=True)
    alpha = torch.sqrt((w_eff ** 2).mean(dim=-1)).clamp(min=1e-4)
    
    default_bias = 2**(e_bits - 1) - 1
    bias_radius = max(1, 2**(e_bits - 2))

    best_loss = torch.full((N, n_blocks), float("inf"), device=device)
    best_alpha = alpha.clone()
    best_bias = torch.full((N, n_blocks), default_bias, device=device, dtype=torch.long)

    # 4. Search Loop (Identical to Row-Wise Logic)
    for bias in range(default_bias - bias_radius, default_bias + bias_radius + 1):
        alpha_tmp = alpha.clone()
        for _ in range(5):
            for b_start in range(0, n_blocks, B_chunk):
                b_end = min(b_start + B_chunk, n_blocks)
                w_c, m_c, H_c = w_eff[:, b_start:b_end, :], mask_blocks[:, b_start:b_end, :], H_all[b_start:b_end]
                a_c = alpha_tmp[:, b_start:b_end].unsqueeze(-1)

                _, _, b_val = assign_fp4_dynamic_vectorized(w_c, a_c, e_bits, m_bits, bias=bias)
                b_eff = b_val * m_c
                
                # Full Hessian Dot Products
                Hw = torch.einsum('cjk, nck -> ncj', H_c, w_c)
                Hb = torch.einsum('cjk, nck -> ncj', H_c, b_eff)
                
                num = (b_eff * Hw).sum(dim=-1)
                den = (b_eff * Hb).sum(dim=-1) + 1e-8
                
                a_min = 0.05 * block_means[:, b_start:b_end].squeeze(-1)
                a_max = 20.0 * block_means[:, b_start:b_end].squeeze(-1)
                alpha_tmp[:, b_start:b_end] = (num / den).clamp(min=a_min, max=a_max)

        # Update best values using the Full Hessian Loss
        for b_start in range(0, n_blocks, B_chunk):
            b_end = min(b_start + B_chunk, n_blocks)
            w_c, a_c = w_eff[:, b_start:b_end, :], alpha_tmp[:, b_start:b_end].unsqueeze(-1)
            H_c = H_all[b_start:b_end]
            
            _, _, b_val = assign_fp4_dynamic_vectorized(w_c, a_c, e_bits, m_bits, bias=bias)
            res = w_c - (a_c * b_val * mask_blocks[:, b_start:b_end, :])
            
            # (res^T @ H @ res)
            H_res = torch.einsum('cjk, nck -> ncj', H_c, res)
            loss_chunk = (res * H_res).sum(dim=-1)
            
            improved = loss_chunk < best_loss[:, b_start:b_end]
            best_loss[:, b_start:b_end] = torch.where(improved, loss_chunk, best_loss[:, b_start:b_end])
            best_alpha[:, b_start:b_end] = torch.where(improved, alpha_tmp[:, b_start:b_end], best_alpha[:, b_start:b_end])
            best_bias[:, b_start:b_end] = torch.where(improved, bias, best_bias[:, b_start:b_end])

    # 5. SCALE QUANTIZATION
    alpha_q_raw = quantize_scale_tensor(best_alpha, e_bits_scale, m_bits_scale)

    # 6. FINAL BIT EXTRACTION 
    # We must use exactly what the row-wise version used in its final block recompute
    e_out_blocks = torch.zeros_like(W_blocks)
    m_out_blocks = torch.zeros_like(W_blocks)
    
    for b_start in range(0, n_blocks, B_chunk):
        b_end = min(b_start + B_chunk, n_blocks)
        w_c = W_blocks[:, b_start:b_end, :]
        # Crucial: Use the quantized alpha for bit assignment
        a_c = alpha_q_raw[:, b_start:b_end].unsqueeze(-1)
        b_c = best_bias[:, b_start:b_end] 

        e_idx, m_idx, _ = assign_fp4_dynamic_vectorized(w_c, a_c, e_bits, m_bits, bias=b_c)
        e_out_blocks[:, b_start:b_end, :] = e_idx.float()
        m_out_blocks[:, b_start:b_end, :] = m_idx.float()

    # 7. Formatting Outputs to match row-loop exactly
    e_out = e_out_blocks.view(N, M_pad)[:, :M]
    m_out = m_out_blocks.view(N, M_pad)[:, :M]
    
    # Return best_alpha for the alpha_out, but ensure it's aligned with the blocks
    # Note: In your slow version, alpha_out was set to alpha_block (quantized)
    alpha_final = alpha_q_raw.unsqueeze(-1).expand(-1, -1, block_size).reshape(N, M_pad)[:, :M]
    bias_final = best_bias.unsqueeze(-1).expand(-1, -1, block_size).reshape(N, M_pad)[:, :M]

    return alpha_final.reshape(orig_shape), e_out.reshape(orig_shape), m_out.reshape(orig_shape), sign.reshape(orig_shape), bias_final.reshape(orig_shape)






import torch
import torch.nn.functional as F

def reconstruct_layer_non_hadamard_v17_final(
    layer, H_blocks_layer, block_size, e_bits, m_bits, e_bits_scale, m_bits_scale, device, B_chunk=128
):
    with torch.no_grad():
        W = layer.weight.data.to(device)
        orig_shape = W.shape
        W_mat = W.view(W.shape[0], -1)
        N, M = W_mat.shape
        mask_mat = (W_mat.abs() > 1e-9).float()
        
        n_blocks = (M + block_size - 1) // block_size
        M_pad = n_blocks * block_size
        W_blocks = F.pad(W_mat, (0, M_pad - M)).view(N, n_blocks, block_size)
        mask_blocks = F.pad(mask_mat, (0, M_pad - M)).view(N, n_blocks, block_size)

        # 1. Hessian Prep
        H_all = torch.stack([
            F.pad(H_blocks_layer[b].to(device).float(), 
                  (0, block_size - H_blocks_layer[b].shape[0], 0, block_size - H_blocks_layer[b].shape[0])) 
            for b in range(n_blocks)
        ]) 

        # 2. Precompute Hw (The "Goal" vector for alpha refinement)
        W_eff = W_blocks * mask_blocks
        Hw = torch.einsum('bjk, nbk -> nbj', H_all, W_eff)

        # 3. Initialization
        default_bias = 2**(e_bits - 1) - 1
        bias_radius = max(1, 2**(e_bits - 2))
        
        best_loss = torch.full((N, n_blocks), float('inf'), device=device)
        best_alpha = torch.zeros((N, n_blocks), device=device)
        best_bias = torch.full((N, n_blocks), default_bias, dtype=torch.long, device=device)

        # 4. Search Loop (Bias is the outer loop to minimize codebook rebuilds)
        for bias_cand in range(default_bias - bias_radius, default_bias + bias_radius + 1):
            
            # Reset Alpha for this bias candidate
            curr_alpha = torch.sqrt((W_eff**2).mean(dim=-1)).clamp(min=1e-4)
            
            # Alpha Refinement (Exact mirror of v4 logic)
            for _ in range(5):
                # Vectorized Basis Assignment
                _, _, b = assign_fp4_v17_vectorized(
                    W_blocks.abs(), curr_alpha, e_bits, m_bits, bias_cand
                )
                
                b_eff = b * mask_blocks
                Hb = torch.einsum('bjk, nbk -> nbj', H_all, b_eff)
                
                num = (b_eff * Hw).sum(dim=-1)
                den = (b_eff * Hb).sum(dim=-1) + 1e-8
                
                # Stabilization from v4
                avg_abs = W_blocks.abs().mean(dim=-1)
                curr_alpha = (num / den).clamp(0.05 * avg_abs, 20.0 * avg_abs)

            # Evaluate Quadratic Loss: res^T @ H @ res
            # Use chunks for loss calculation to prevent OOM on very wide layers
            for b_start in range(0, n_blocks, B_chunk):
                b_end = min(b_start + B_chunk, n_blocks)
                
                a_c = curr_alpha[:, b_start:b_end].unsqueeze(-1)
                _, _, b_c = assign_fp4_v17_vectorized(
                    W_blocks[:, b_start:b_end, :].abs(), a_c.squeeze(-1), e_bits, m_bits, bias_cand
                )
                
                recon = torch.sign(W_blocks[:, b_start:b_end, :]) * a_c * b_c
                res = (W_blocks[:, b_start:b_end, :] * mask_blocks[:, b_start:b_end, :]) - (recon * mask_blocks[:, b_start:b_end, :])
                
                H_res = torch.einsum('bjk, nbk -> nbj', H_all[b_start:b_end], res)
                loss_chunk = (res * H_res).sum(dim=-1)
                
                improved = loss_chunk < best_loss[:, b_start:b_end]
                best_loss[:, b_start:b_end] = torch.where(improved, loss_chunk, best_loss[:, b_start:b_end])
                best_alpha[:, b_start:b_end] = torch.where(improved, curr_alpha[:, b_start:b_end], best_alpha[:, b_start:b_end])
                best_bias[:, b_start:b_end] = torch.where(improved, bias_cand, best_bias[:, b_start:b_end])

        # 5. Final Scale Quantization
        alpha_q = quantize_scale_tensor(best_alpha, e_bits_scale, m_bits_scale)

        # 6. Final Reconstruction
        W_out = torch.zeros_like(W_blocks)
        unique_biases = best_bias.unique().tolist()
        
        for b_val in unique_biases:
            b_mask = (best_bias == b_val)
            # Recompute basis for all blocks that shared this best bias
            _, _, b_final = assign_fp4_v17_vectorized(
                W_blocks.abs(), alpha_q, e_bits, m_bits, int(b_val)
            )
            
            recon_final = torch.sign(W_blocks) * alpha_q.unsqueeze(-1) * b_final * mask_blocks
            W_out = torch.where(b_mask.unsqueeze(-1), recon_final, W_out)

        return {
            'alpha': alpha_q,
            'bias': best_bias,
            'reconstructed_weight': W_out.view(N, M_pad)[:, :M].reshape(orig_shape)
        }

def assign_fp4_v17_vectorized(W_abs, alpha, e_bits, m_bits, bias):
    """
    Precision-matched vectorized FP4.
    Matches v4: basis = 2^(e - bias) * (1 + m / 2^M_bits)
    """
    device = W_abs.device
    
    # 1. Generate standard codebook (float values)
    e_levels = torch.arange(2**e_bits, device=device).float()
    m_levels = torch.arange(2**m_bits, device=device).float()
    
    # This represents the "unbiased" codebook levels
    # cb = 2^e * (1 + m/2^M)
    cb = (2.0**e_levels).unsqueeze(1) * (1.0 + m_levels / (2**m_bits)).unsqueeze(0)
    codebook = cb.view(-1).sort()[0]
    
    # 2. Normalize weights by alpha AND the bias shift
    # Mathematically: W / (alpha * 2^-bias) = (W * 2^bias) / alpha
    # This allows us to search the standard codebook correctly.
    target = (W_abs * (2.0**float(bias))) / alpha.unsqueeze(-1).clamp(min=1e-8)
    
    # 3. Vectorized Nearest Neighbor
    # For speed, we use bucket search since FP4 codebooks are small (16-32 entries)
    t_shape = target.shape
    t_flat = target.view(-1, 1)
    
    # Compute absolute distances to all codebook entries
    dists = torch.abs(t_flat - codebook)
    best_idx = torch.argmin(dists, dim=-1)
    
    # 4. Map back to biased basis
    # basis = codebook_value * 2^-bias
    chosen_cb_vals = codebook[best_idx].view(t_shape)
    basis = chosen_cb_vals * (2.0**(-float(bias)))
    
    return None, None, basis


def reconstruct_layer_non_hadamard_v10_fast(
    layer,
    H_blocks_layer,
    block_size,
    e_bits,
    m_bits,
    e_bits_scale,
    m_bits_scale,
    device,
    B_chunk=32  # Chunk size for vectorization balance
):
    with torch.no_grad():
        W = layer.weight.data.to(device)
        mask = (W.abs() > 1e-9).float()
        orig_shape = W.shape
        W_mat = W.view(W.shape[0], -1)
        mask_mat = mask.view(W.shape[0], -1)
        N, M = W_mat.shape

        # 1. Padding & Blocking
        n_blocks = (M + block_size - 1) // block_size
        M_pad = n_blocks * block_size
        W_p = F.pad(W_mat, (0, M_pad - M))
        mask_p = F.pad(mask_mat, (0, M_pad - M))
        
        W_blocks = W_p.view(N, n_blocks, block_size)
        mask_blocks = mask_p.view(N, n_blocks, block_size)
        W_abs = W_blocks.abs()
        W_sign = torch.sign(W_blocks)

        # 2. Stack Full Hessian Blocks (Crucial for Non-Hadamard accuracy)
        H_all = torch.stack([
            F.pad(H_blocks_layer[b].to(device).float(), 
                  (0, block_size - H_blocks_layer[b].shape[0], 0, block_size - H_blocks_layer[b].shape[0])) 
            for b in range(n_blocks)
        ]) # [n_blocks, bs, bs]

        # 3. Initialization
        # Use mean magnitude for stable alpha start
        alpha = (W_abs.pow(2).mean(dim=-1).sqrt().clamp(min=1e-4))
        default_bias = 2**(e_bits - 1) - 1
        bias_radius = max(1, 2**(e_bits - 2))
        
        best_loss = torch.full((N, n_blocks), float('inf'), device=device)
        best_alpha = alpha.clone()
        best_bias = torch.full((N, n_blocks), default_bias, dtype=torch.long, device=device)

        # 4. Search Loop
        for bias_cand in range(default_bias - bias_radius, default_bias + bias_radius + 1):
            alpha_tmp = alpha.clone()
            
            # Alpha Refinement (5 steps)
            for _ in range(5):
                for b_start in range(0, n_blocks, B_chunk):
                    b_end = min(b_start + B_chunk, n_blocks)
                    
                    w_c = W_abs[:, b_start:b_end, :]
                    m_c = mask_blocks[:, b_start:b_end, :]
                    H_c = H_all[b_start:b_end]
                    
                    # Re-use your vectorized assignment to get basis
                    _, _, basis = assign_fp4_dynamic_vectorized(
                        w_c, alpha_tmp[:, b_start:b_end], e_bits, m_bits, bias=bias_cand
                    )
                    
                    b_eff = basis * m_c
                    # Full Hessian Math: b^T H w / b^T H b
                    Hw = torch.einsum('bjk, nbk -> nbj', H_c, w_c)
                    Hb = torch.einsum('bjk, nbk -> nbj', H_c, b_eff)
                    
                    num = (b_eff * Hw).sum(dim=-1)
                    den = (b_eff * Hb).sum(dim=-1) + 1e-8
                    
                    # Stabilization (0.05 to 20x mean)
                    limit = W_abs[:, b_start:b_end, :].mean(dim=-1)
                    alpha_tmp[:, b_start:b_end] = (num / den).clamp(limit*0.05, limit*20.0)

            # Evaluate Loss with Full Hessian
            for b_start in range(0, n_blocks, B_chunk):
                b_end = min(b_start + B_chunk, n_blocks)
                w_c = W_blocks[:, b_start:b_end, :]
                a_c = alpha_tmp[:, b_start:b_end].unsqueeze(-1)
                H_c = H_all[b_start:b_end]
                
                _, _, basis = assign_fp4_dynamic_vectorized(
                    w_c.abs(), a_c.squeeze(-1), e_bits, m_bits, bias=bias_cand
                )
                
                recon = W_sign[:, b_start:b_end, :] * a_c * basis * mask_blocks[:, b_start:b_end, :]
                res = w_c - recon
                
                # Full Quadratic Loss: res^T @ H @ res
                H_res = torch.einsum('bjk, nbk -> nbj', H_c, res)
                loss_chunk = (res * H_res).sum(dim=-1)
                
                improved = loss_chunk < best_loss[:, b_start:b_end]
                best_loss[:, b_start:b_end] = torch.where(improved, loss_chunk, best_loss[:, b_start:b_end])
                best_alpha[:, b_start:b_end] = torch.where(improved, alpha_tmp[:, b_start:b_end], best_alpha[:, b_start:b_end])
                best_bias[:, b_start:b_end] = torch.where(improved, bias_cand, best_bias[:, b_start:b_end])

        # 5. Final Reconstruction (Align bits with Quantized Scale)
        alpha_q = quantize_scale_tensor(best_alpha, e_bits_scale, m_bits_scale)
        W_q = torch.zeros_like(W_blocks)
        
        # Consistent bit extraction using finalized alpha_q
        for b_start in range(0, n_blocks, B_chunk):
            b_end = min(b_start + B_chunk, n_blocks)
            a_c = alpha_q[:, b_start:b_end].unsqueeze(-1)
            b_c = best_bias[:, b_start:b_end]
            
            _, _, basis = assign_fp4_dynamic_vectorized(
                W_abs[:, b_start:b_end, :], a_c.squeeze(-1), e_bits, m_bits, bias=b_c
            )
            W_q[:, b_start:b_end, :] = W_sign[:, b_start:b_end, :] * a_c * basis * mask_blocks[:, b_start:b_end, :]

        W_out = W_q.view(N, M_pad)[:, :M].reshape(orig_shape)
        alpha_out = alpha_q.unsqueeze(-1).expand(-1, -1, block_size).reshape(N, M_pad)[:, :M].reshape(orig_shape)
        
        return {
            'alpha': alpha_out,
            'bias': best_bias,
            'reconstructed_weight': W_out,
        }

def reconstruct_layer_fp_blockdiag_scaled_v4_forward(
    layer,
    H_blocks_layer,
    block_size,
    e_bits,
    m_bits,
    e_bits_scale,
    m_bits_scale,
    device,
    cached_input=None,
    use_forward=False,
    top_k=2
):
    """
    FINAL VERSION — Stable adaptive-mesh FP4 reconstruction

    Args:
        layer: nn.Linear or nn.Conv2d
        H_blocks_layer: list of Hessian blocks (per weight block)
        block_size: block size for FP4 reconstruction
        e_bits, m_bits: number of bits for exponent/mantissa
        e_bits_scale, m_bits_scale: number of bits for alpha quantization
        device: torch.device
        cached_input: optional input for forward-pass selection
        use_forward: if True, compute loss using forward pass
        top_k: number of candidates for Hessian selection

    Returns:
        alpha_out, e, m, sign, bias_out
    """

    W = layer.weight.data.to(device)
    mask = (W != 0).float()

    if W.dim() == 4:
        W_mat = W.view(W.shape[0], -1)
        mask_mat = mask.view(W.shape[0], -1)
    else:
        W_mat = W
        mask_mat = mask

    N, M = W_mat.shape
    sign = torch.sign(W_mat)
    W_abs = W_mat.abs()

    W_q = torch.zeros_like(W_mat)
    alpha_out = torch.zeros_like(W_mat)
    bias_out = torch.zeros_like(W_mat)

    # Optional precompute FP output
    if use_forward and cached_input is not None:
        with torch.no_grad():
            if W.dim() == 2:
                y_fp = F.linear(cached_input, W)
            else:
                y_fp = F.conv2d(
                    cached_input,
                    W,
                    layer.bias,
                    stride=layer.stride,
                    padding=layer.padding
                )

    for row in range(N):
        w_row = W_abs[row]
        m_row = mask_mat[row]
        block_idx = 0

        for i in range(0, M, block_size):
            end = min(i + block_size, M)
            w_block = w_row[i:end]
            m_block = m_row[i:end]

            H_block = H_blocks_layer[block_idx].to(device)
            k = w_block.numel()
            if H_block.shape[0] != k:
                H_block = H_block[:k, :k]

            # Fully pruned block
            if m_block.sum() < 1e-8:
                alpha_block = torch.tensor(1.0, device=device)
                bias_block = 0
                W_q[row, i:end] = 0.0
                alpha_out[row, i:end] = alpha_block
                bias_out[row, i:end] = bias_block
                block_idx += 1
                continue

            # Effective weights
            w_eff = w_block * m_block
            alpha_init = torch.sqrt((w_eff ** 2).mean()).clamp(min=1e-4)

            default_bias = 2**(e_bits - 1) - 1
            bias_radius = max(1, 2**(e_bits - 2))  # adaptive search window

            best_loss = float('inf')
            best_alpha = None
            best_bias = None
            best_b = None

            Hw = H_block @ w_eff

            # Candidate bias search
            for bias_candidate in range(default_bias - bias_radius,
                                        default_bias + bias_radius + 1):

                alpha_tmp = alpha_init.clone()
                # Iterative alpha refinement
                for _ in range(5):
                    _, _, b = assign_fp4_dynamic(
                        w_block, alpha_tmp, e_bits, m_bits, bias=bias_candidate
                    )
                    b_eff = b * m_block
                    Hb = H_block @ b_eff
                    num = torch.dot(b_eff, Hw)
                    den = torch.dot(b_eff, Hb) + 1e-8
                    alpha_new = num / den
                    alpha_min = 0.05 * w_eff.abs().mean()
                    alpha_max = 20.0 * w_eff.abs().mean()
                    alpha_tmp = torch.clamp(alpha_new, min=alpha_min, max=alpha_max)

                # Loss evaluation
                residual = w_eff - alpha_tmp * b_eff
                if use_forward and cached_input is not None:
                    W_tmp = W_q.clone()
                    W_tmp[row, i:end] = alpha_tmp * b_eff * sign[row, i:end]
                    with torch.no_grad():
                        if W.dim() == 2:
                            y_q = F.linear(cached_input, W_tmp)
                        else:
                            y_q = F.conv2d(
                                cached_input,
                                W_tmp.view_as(W),
                                layer.bias,
                                stride=layer.stride,
                                padding=layer.padding
                            )
                    loss = ((y_fp - y_q) ** 2).mean()
                else:
                    loss = torch.dot(residual, H_block @ residual)

                if loss < best_loss:
                    best_loss = loss
                    best_alpha = alpha_tmp
                    best_bias = bias_candidate
                    best_b = b

            # Final alpha quantization
            alpha_block = quantize_scale(best_alpha, e_bits_scale, m_bits_scale)

            # Final basis recompute
            _, _, b_final = assign_fp4_dynamic(
                w_block, alpha_block, e_bits, m_bits, bias=best_bias
            )

            w_hat = alpha_block * b_final
            w_hat = w_hat * m_block
            w_hat = w_hat * sign[row, i:end]

            # Store
            W_q[row, i:end] = w_hat
            alpha_out[row, i:end] = alpha_block
            bias_out[row, i:end] = best_bias  # integer per block

            block_idx += 1

    # Reshape back
    if W.dim() == 4:
        W_q = W_q.view_as(W)
        alpha_out = alpha_out.view_as(W)
        bias_out = bias_out.view_as(W)

    # Final FP decomposition (per-block exponent)
    sign = torch.sign(W_q)
    W_abs = W_q.abs()

    if W.dim() == 4:
        W_mat = W_abs.view(W.shape[0], -1)
        alpha_mat = alpha_out.view(W.shape[0], -1)
        bias_mat = bias_out.view(W.shape[0], -1)
    else:
        W_mat = W_abs
        alpha_mat = alpha_out
        bias_mat = bias_out

    e = torch.zeros_like(W_mat)
    m = torch.zeros_like(W_mat)

    for row in range(N):
        for i in range(0, M, block_size):
            end = min(i + block_size, M)
            bias_block = int(bias_mat[row, i].item())  # integer per block
            alpha_block = alpha_mat[row, i]
            w_block = W_mat[row, i:end]

            e_block, m_block, _ = assign_fp4_dynamic(
                w_block, alpha_block, e_bits, m_bits, bias=bias_block
            )

            e[row, i:end] = e_block
            m[row, i:end] = m_block

    if W.dim() == 4:
        e = e.view_as(W)
        m = m.view_as(W)

    return alpha_out, e, m, sign, bias_out


def reconstruct_layer_hadamard_v5(
    layer,
    H_blocks_layer,
    block_size,
    e_bits,
    m_bits,
    e_bits_scale,
    m_bits_scale,
    device
):
    """
    V5: Hadamard-Domain Adaptive-Mesh FP4 reconstruction
    - Rotates weights to Hadamard domain to suppress outliers
    - Hessian-aware alpha optimization in H-domain
    - Consistent bias search per block
    """
    W = layer.weight.data.to(device)
    mask = (W != 0).float()

    if W.dim() == 4:
        W_mat = W.view(W.shape[0], -1)
        mask_mat = mask.view(W.shape[0], -1)
    else:
        W_mat = W
        mask_mat = mask

    N, M = W_mat.shape
    
    # Pre-generate random signs to break symmetry in the Hadamard transform
    # This is fixed for the layer to ensure deterministic reconstruction
    fixed_signs = torch.sign(torch.randn(1, M, device=device))
    W_signed = W_mat * fixed_signs

    W_q = torch.zeros_like(W_mat)
    alpha_out = torch.zeros_like(W_mat)
    bias_out = torch.zeros_like(W_mat)

    for row in range(N):
        block_idx = 0
        for i in range(0, M, block_size):
            end = i + block_size
            w_block = W_signed[row, i:end]
            m_block = mask_mat[row, i:end]

            # 1. Rotate block to Hadamard Domain
            # w_had represents the 'smeared' weights
            w_had = fast_hadamard_transform(w_block.unsqueeze(0)).squeeze(0)
            
            # 2. Handle Hessian
            # In the H-domain, the Hessian H_had = Q^T H Q. 
            # If H is diagonal, H_had becomes a dense matrix where all 
            # elements are the average of the diagonal.
            H_block = H_blocks_layer[block_idx].to(device)
            h_diag_avg = torch.diag(H_block).mean()
            H_had = torch.eye(block_size, device=device) * h_diag_avg

            # 3. Initialization in H-domain
            w_eff = w_had # Masking is trickier in H-domain; usually applied at the end
            alpha_init = torch.sqrt((w_eff ** 2).mean()).clamp(min=1e-4)

            default_bias = 2**(e_bits - 1) - 1
            bias_radius = max(1, 2**(e_bits - 2))

            best_loss = float('inf')
            best_alpha = None
            best_bias = None
            best_b_had = None

            # 4. Bias Search (Same logic as V4, but on w_had)
            Hw = H_had @ w_eff
            for bias_candidate in range(default_bias - bias_radius,
                                        default_bias + bias_radius + 1):
                alpha_tmp = alpha_init.clone()
                for _ in range(5):
                    _, _, b = assign_fp4_dynamic(
                        w_had, alpha_tmp, e_bits, m_bits, bias=bias_candidate
                    )
                    Hb = H_had @ b
                    num = torch.dot(b, Hw)
                    den = torch.dot(b, Hb) + 1e-8
                    alpha_new = num / den
                    
                    alpha_lim = w_eff.abs().mean()
                    alpha_tmp = torch.clamp(alpha_new, min=0.05*alpha_lim, max=20*alpha_lim)

                residual = w_eff - alpha_tmp * b
                loss = torch.dot(residual, H_had @ residual)

                if loss < best_loss:
                    best_loss = loss
                    best_alpha = alpha_tmp
                    best_bias = bias_candidate
                    best_b_had = b

            # 5. Final Quantization & Inverse Transform
            alpha_block = quantize_scale(best_alpha, e_bits_scale, m_bits_scale)
            
            # Recompute best basis with quantized alpha
            _, _, b_final_had = assign_fp4_dynamic(
                w_had, alpha_block, e_bits, m_bits, bias=best_bias
            )
            
            # Rotate back to weight domain
            w_q_had = alpha_block * b_final_had
            w_q_block = fast_hadamard_transform(w_q_had.unsqueeze(0)).squeeze(0)

            # Store results
            W_q[row, i:end] = w_q_block
            alpha_out[row, i:end] = alpha_block
            bias_out[row, i:end] = best_bias
            block_idx += 1

    # Remove random signs and apply original mask
    W_q = (W_q / fixed_signs) * mask_mat

    # ... (Final FP decomposition logic same as V4 to return e, m, sign) ...
    # Use your existing logic from V4 here to extract e and m from the resulting W_q
    
    return alpha_out, W_q, bias_out # Simplified return for brevity

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def reconstruct_layer_hadamard_v5_final(
    layer,
    H_blocks_layer,
    block_size,
    e_bits,
    m_bits,
    e_bits_scale,
    m_bits_scale,
    device
):
    """
    V5 Final: Hadamard-Domain Adaptive FP4 reconstruction.
    Returns discrete components that allow for H-domain inference.
    """
    W = layer.weight.data.to(device)
    mask = (W != 0).float()

    if W.dim() == 4:
        W_mat = W.view(W.shape[0], -1)
        mask_mat = mask.view(W.shape[0], -1)
    else:
        W_mat = W
        mask_mat = mask

    N, M = W_mat.shape
    
    # 1. Pre-conditioning: Fixed signs to flatten the distribution
    # This must be saved/deterministic for inference folding
    fixed_signs = torch.sign(torch.randn(1, M, device=device))
    fixed_signs[fixed_signs == 0] = 1 
    W_signed = W_mat * fixed_signs

    # Storage for the components
    alpha_out = torch.zeros_like(W_mat)
    e_out = torch.zeros_like(W_mat, dtype=torch.long)
    m_out = torch.zeros_like(W_mat, dtype=torch.long)
    sign_out = torch.zeros_like(W_mat)
    bias_out = torch.zeros((N, math.ceil(M / block_size)), device=device, dtype=torch.long)
    for row in range(N):
        block_idx = 0
        for i in range(0, M, block_size):
            end = i + block_size
            w_block = W_signed[row, i:end]

            # 2. Rotate block to Hadamard Domain
            # w_had = fast_hadamard_transform(w_block.unsqueeze(0)).squeeze(0)
            w_had = hadamard_transform_wrapper(w_block)
            curr_size = w_had.shape[0]

            # 3. Hessian Average (Diagonal approximation in rotated space)
            H_block = H_blocks_layer[block_idx].to(device)
            h_diag_avg = torch.diag(H_block).mean()
            H_had = torch.eye(block_size, device=device) * h_diag_avg

            # 4. Init Alpha in H-domain
            alpha_init = torch.sqrt((w_had ** 2).mean()).clamp(min=1e-4)
            default_bias = 2**(e_bits - 1) - 1
            bias_radius = max(1, 2**(e_bits - 2))

            best_loss = float('inf')
            best_alpha = alpha_init
            best_bias = default_bias

            # 5. Adaptive Bias & Alpha Search
            # Instead of: Hw = H_had @ w_had

            Hw = H_had[:curr_size, :curr_size] @ w_had.abs()
            for bias_candidate in range(default_bias - bias_radius, default_bias + bias_radius + 1):
                alpha_tmp = alpha_init.clone()
                for _ in range(5):
                    _, _, b = assign_fp4_dynamic(
                        w_had.abs(), alpha_tmp, e_bits, m_bits, bias=bias_candidate
                    )
                    # We use absolute w_had for codebook assignment, signs handled after
                    Hb = H_had[:curr_size, :curr_size] @ b
                    num = torch.dot(b, Hw)
                    den = torch.dot(b, Hb) + 1e-8
                    alpha_tmp = (num / den).clamp(min=1e-6)

                # Evaluate loss in H-domain
                recon_had = torch.sign(w_had) * alpha_tmp * b
                residual = w_had - recon_had
                loss = torch.dot(residual, H_had[:curr_size, :curr_size] @ residual)

                if loss < best_loss:
                    best_loss = loss
                    best_alpha = alpha_tmp
                    best_bias = bias_candidate

            # 6. Quantize Scale and Finalize H-domain Components
            alpha_q = quantize_scale(best_alpha, e_bits_scale, m_bits_scale)
            
            # Extract final components in Hadamard space
            # These are the bits actually 'stored'
            exp, mant, basis = assign_fp4_dynamic(
                w_had.abs(), alpha_q, e_bits, m_bits, bias=best_bias
            )
            
            # Save block components
            alpha_out[row, i:end] = alpha_q
            e_out[row, i:end] = exp
            m_out[row, i:end] = mant
            sign_out[row, i:end] = torch.sign(w_had)
            bias_out[row, block_idx] = best_bias
            
            block_idx += 1

    # Final spatial reconstruction (for simulation purposes/validation)
    # W_hat = (H_inv(alpha * basis * sign_had)) * fixed_signs
    # 1. Correct the Bias Alignment
    # We use 'block_size' for interleaving and slice to M to ensure it matches W_q_had perfectly
    interleaved_bias = bias_out.repeat_interleave(curr_size, dim=1)[:, :M]

    # Calculate reconstructed weights in the Hadamard domain
    W_q_had = alpha_out * sign_out * (
        2.0**(e_out.float() - interleaved_bias.float()) * (1 + m_out.float() / (2**m_bits))
    )
    
    # 2. Batch inverse FHT with Correct Scaling
    W_q_spatial = torch.zeros_like(W_q_had)
    for row in range(N):
        for i in range(0, M, block_size):
            end = min(i + block_size, M)
            block = W_q_had[row, i:end]
            
            # Use the inverse wrapper to handle padding and the power-of-2 requirement
            W_q_spatial[row, i:end] = inverse_hadamard_transform_wrapper(block)
    
    # 3. Final spatial reconstruction
    # Multiplying by fixed_signs reverses the pre-conditioning
    W_q_spatial = (W_q_spatial * fixed_signs) * mask_mat

    return {
        'alpha': alpha_out.view_as(W),
        'exponent': e_out.view_as(W),
        'mantissa': m_out.view_as(W),
        'sign': sign_out.view_as(W),
        'bias': bias_out,
        'fixed_signs': fixed_signs,
        'reconstructed_weight': W_q_spatial.view_as(W)
    }


def reconstruct_layer_hadamard_v6(
    layer,
    H_blocks_layer,
    block_size,
    e_bits,
    m_bits,
    e_bits_scale,
    m_bits_scale,
    device,
    seed=42,
):
    """
    V6: Corrected Hadamard-Domain Adaptive FP4 reconstruction.
 
    Key fixes over v5:
      1. Deterministic fixed_signs via an explicit RNG seed.
      2. Full power-of-2 padded representation is carried through both the
         forward and inverse transforms — no mid-pipeline crop — so the
         inverse is an exact left-inverse of the forward.
      3. Per-element rotated Hessian diagonal instead of a scalar average,
         preserving sensitivity information in the Hadamard domain.
      4. quantize_scale is called correctly: for e8m0 (m_bits_scale=0) it
         already returns 2**floor(log2(alpha)), which is the only
         representable value in that format.
      5. assign_fp4_dynamic is called on the full padded block so indices
         align with the transform; padded positions are zeroed before the
         inverse transform.
 
    Returns a dict with all stored components plus 'reconstructed_weight'.
    The 'reconstructed_weight' entry is the simulation/validation path;
    for inference you would carry alpha/exponent/mantissa/sign/bias/
    fixed_signs and fold the inverse-Hadamard into the matmul.
    """
 
    W = layer.weight.data.to(device)
    mask = (W != 0).float()
 
    if W.dim() == 4:
        W_mat = W.view(W.shape[0], -1)
        mask_mat = mask.view(W.shape[0], -1)
    else:
        W_mat = W
        mask_mat = mask
 
    N, M = W_mat.shape
 
    # ------------------------------------------------------------------ #
    # FIX 1 – Deterministic fixed_signs                                   #
    # ------------------------------------------------------------------ #
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    raw = torch.randn(1, M, device=device, generator=rng)
    fixed_signs = torch.sign(raw)
    fixed_signs[fixed_signs == 0] = 1.0
 
    W_signed = W_mat * fixed_signs  # (N, M)
 
    # ------------------------------------------------------------------ #
    # Work out the padded block size once (power of 2 >= block_size)      #
    # ------------------------------------------------------------------ #
    next_pow2 = 2 ** int(math.ceil(math.log2(block_size))) if block_size > 1 else 1
    n_blocks = math.ceil(M / block_size)
 
    # Storage tensors — indexed over the *original* (N, M) grid
    alpha_out  = torch.zeros_like(W_mat)                                          # (N, M)
    e_out      = torch.zeros_like(W_mat, dtype=torch.long)                        # (N, M)
    m_out      = torch.zeros_like(W_mat, dtype=torch.long)                        # (N, M)
    sign_out   = torch.zeros_like(W_mat)                                          # (N, M)
    bias_out   = torch.zeros((N, n_blocks), device=device, dtype=torch.long)      # (N, B)
 
    default_bias  = 2 ** (e_bits - 1) - 1
    bias_radius   = max(1, 2 ** (e_bits - 2))
 
    for row in range(N):
        for block_idx, i in enumerate(range(0, M, block_size)):
            end        = min(i + block_size, M)
            orig_len   = end - i                # actual number of weights in this block
            w_block    = W_signed[row, i:end]   # (orig_len,)
 
            # -------------------------------------------------------------- #
            # FIX 2 – Carry the full padded vector through forward + inverse  #
            # -------------------------------------------------------------- #
            pad_len  = next_pow2 - orig_len
            w_padded = F.pad(w_block, (0, pad_len))          # (next_pow2,)
 
            # Forward Hadamard (normalised by 1/sqrt(next_pow2))
            w_had_full = fast_hadamard_transform(w_padded.unsqueeze(0)).squeeze(0)
            # (next_pow2,)  — we work in this full space
 
            # -------------------------------------------------------------- #
            # FIX 3 – Rotated Hessian diagonal                               #
            # -------------------------------------------------------------- #
            H_block = H_blocks_layer[block_idx].to(device)   # (block_size, block_size)
 
            # Extract diagonal of the spatial-domain Hessian block,
            # pad to next_pow2, rotate it into the Hadamard domain.
            h_diag = torch.diag(H_block)                     # (block_size,) or (orig_len,)
            # Clamp to non-negative before rotating (H is PSD but numerics can
            # introduce tiny negatives on the diagonal)
            h_diag = h_diag.clamp(min=0.0)
            h_diag_padded = F.pad(h_diag, (0, next_pow2 - h_diag.shape[0]))
            h_diag_had = fast_hadamard_transform(
                h_diag_padded.unsqueeze(0)
            ).squeeze(0).abs()                               # (next_pow2,) – abs for PSD safety
 
            # Build diagonal Hessian matrix in the Hadamard domain
            H_had = torch.diag(h_diag_had)                   # (next_pow2, next_pow2)
 
            # -------------------------------------------------------------- #
            # Bias & alpha search (same iterative OBS-style update as v5)     #
            # -------------------------------------------------------------- #
            w_had_abs  = w_had_full.abs()
            alpha_init = torch.sqrt((w_had_abs ** 2).mean()).clamp(min=1e-4)
 
            # Weighted Hessian product for the alpha numerator (constant across iters)
            Hw = H_had @ w_had_abs                           # (next_pow2,)
 
            best_loss  = float('inf')
            best_alpha = alpha_init
            best_bias  = default_bias
 
            for bias_candidate in range(
                default_bias - bias_radius,
                default_bias + bias_radius + 1
            ):
                alpha_tmp = alpha_init.clone()
 
                for _ in range(5):
                    _, _, b = assign_fp4_dynamic(
                        w_had_abs, alpha_tmp, e_bits, m_bits, bias=bias_candidate
                    )
                    Hb  = H_had @ b
                    num = torch.dot(b, Hw)
                    den = torch.dot(b, Hb) + 1e-8
                    alpha_tmp = (num / den).clamp(min=1e-6)
 
                # Evaluate H-domain loss
                recon_had = torch.sign(w_had_full) * alpha_tmp * b
                residual  = w_had_full - recon_had
                loss      = torch.dot(residual, H_had @ residual)
 
                if loss < best_loss:
                    best_loss  = loss
                    best_alpha = alpha_tmp
                    best_bias  = bias_candidate
 
            # -------------------------------------------------------------- #
            # FIX 4 – Scale quantisation                                      #
            # quantize_scale already handles e8m0 correctly (m_bits_scale=0   #
            # → returns 2**floor(log2(alpha))). No change needed here.        #
            # -------------------------------------------------------------- #
            alpha_q = quantize_scale(best_alpha, e_bits_scale, m_bits_scale)
 
            # Final quantisation in the Hadamard domain (full padded block)
            exp, mant, basis = assign_fp4_dynamic(
                w_had_abs, alpha_q, e_bits, m_bits, bias=best_bias
            )
 
            # Store only the orig_len slice — padded positions are don't-cares
            alpha_out[row, i:end] = alpha_q
            e_out    [row, i:end] = exp [:orig_len]
            m_out    [row, i:end] = mant[:orig_len]
            sign_out [row, i:end] = torch.sign(w_had_full[:orig_len])
            bias_out [row, block_idx] = best_bias
 
    # ------------------------------------------------------------------ #
    # Spatial reconstruction (simulation / validation path)               #
    # ------------------------------------------------------------------ #
    # Expand per-block bias back to per-element shape (N, M)
    interleaved_bias = bias_out.repeat_interleave(block_size, dim=1)[:, :M]
 
    W_q_had = alpha_out * sign_out * (
        2.0 ** (e_out.float() - interleaved_bias.float())
        * (1.0 + m_out.float() / (2 ** m_bits))
    )  # (N, M)
 
    W_q_spatial = torch.zeros_like(W_q_had)
 
    for row in range(N):
        for i in range(0, M, block_size):
            end      = min(i + block_size, M)
            orig_len = end - i
            block    = W_q_had[row, i:end]       # (orig_len,)
 
            # -------------------------------------------------------------- #
            # FIX 2 (inverse) – pad to next_pow2, invert, crop               #
            # Because the forward FHT is self-inverse when normalised by      #
            # 1/√N, applying it twice returns the original vector.            #
            # -------------------------------------------------------------- #
            block_padded = F.pad(block, (0, next_pow2 - orig_len))
            spatial_full = fast_hadamard_transform(
                block_padded.unsqueeze(0)
            ).squeeze(0)
            W_q_spatial[row, i:end] = spatial_full[:orig_len]
 
    # Reverse pre-conditioning and apply sparsity mask
    W_q_spatial = W_q_spatial * fixed_signs * mask_mat
 
    return {
        'alpha':               alpha_out.view_as(W),
        'exponent':            e_out.view_as(W),
        'mantissa':            m_out.view_as(W),
        'sign':                sign_out.view_as(W),
        'bias':                bias_out,               # (N, n_blocks)
        'fixed_signs':         fixed_signs,            # (1, M) — save for inference
        'reconstructed_weight': W_q_spatial.view_as(W),
    }




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 2 ** math.ceil(math.log2(n))


def _fht(x: torch.Tensor) -> torch.Tensor:
    """
    Normalised Fast Walsh-Hadamard Transform.
    x: (batch, n) where n is a power of 2.
    Divides by sqrt(n) so the transform is self-inverse:
        _fht(_fht(x)) == x
    """
    n = x.shape[-1]
    assert (n & (n - 1)) == 0, f"n must be a power of 2, got {n}"
    h = 1
    while h < n:
        x = x.view(-1, n // (2 * h), 2, h)
        xl = x[:, :, 0, :] + x[:, :, 1, :]
        xr = x[:, :, 0, :] - x[:, :, 1, :]
        x = torch.cat((xl.unsqueeze(2), xr.unsqueeze(2)), dim=2)
        h *= 2
    return x.view(-1, n) / math.sqrt(n)


def _fht_vec(v: torch.Tensor) -> torch.Tensor:
    """Convenience wrapper for a 1-D vector."""
    return _fht(v.unsqueeze(0)).squeeze(0)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def reconstruct_layer_hadamard_v7(
    layer,
    H_blocks_layer,
    block_size: int,
    e_bits: int,
    m_bits: int,
    e_bits_scale: int,
    m_bits_scale: int,
    device,
    seed: int = 42,
):
    """
    V7: Hadamard-Domain Adaptive FP quantisation.

    Changes vs V6
    -------------
    1.  block_size is snapped up to the next power of 2 internally.
        This eliminates all padding asymmetry: every block is exactly
        pow2_block_size elements, forward and inverse transforms use the
        same N, and the 1/√N factors cancel perfectly.

    2.  Hessian diagonal is rotated with the same power-of-2 size so the
        sensitivity map is consistent with the weight transform.

    3.  fixed_signs has width pow2_block_size * n_blocks (padded domain)
        so pre-conditioning and un-conditioning operate in the same space.
        The mask correctly zeros out padded positions.

    4.  The reconstruction (simulation) path applies _fht exactly once to
        each Hadamard-domain block — no double-application ambiguity.
    """

    W = layer.weight.data.to(device)
    mask = (W != 0).float()

    if W.dim() == 4:
        W_mat = W.view(W.shape[0], -1)
        mask_mat = mask.view(W.shape[0], -1)
    else:
        W_mat = W
        mask_mat = mask

    N, M = W_mat.shape

    # ------------------------------------------------------------------
    # 1.  Snap block_size to next power of 2
    # ------------------------------------------------------------------
    pow2_bs   = _next_pow2(block_size)
    n_blocks  = math.ceil(M / block_size)   # number of logical blocks

    # Padded width: every block occupies exactly pow2_bs columns
    M_pad = n_blocks * pow2_bs

    # ------------------------------------------------------------------
    # 2.  Pad W_mat and mask_mat to M_pad
    # ------------------------------------------------------------------
    W_pad    = F.pad(W_mat,    (0, M_pad - M))   # (N, M_pad)
    mask_pad = F.pad(mask_mat, (0, M_pad - M))   # padded positions → 0

    # ------------------------------------------------------------------
    # 3.  Deterministic pre-conditioning signs (in padded space)
    # ------------------------------------------------------------------
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    raw          = torch.randn(1, M_pad, device=device, generator=rng)
    fixed_signs  = torch.sign(raw)
    fixed_signs[fixed_signs == 0] = 1.0
    # Zero out signs for padded positions so they don't perturb the FHT
    pad_mask              = torch.ones(1, M_pad, device=device)
    pad_mask[0, M:]       = 0.0
    fixed_signs           = fixed_signs * pad_mask

    W_signed = W_pad * fixed_signs                # (N, M_pad)

    # ------------------------------------------------------------------
    # Storage (padded domain — makes index arithmetic trivial)
    # ------------------------------------------------------------------
    alpha_out = torch.zeros(N, M_pad, device=device)
    e_out     = torch.zeros(N, M_pad, device=device, dtype=torch.long)
    m_out     = torch.zeros(N, M_pad, device=device, dtype=torch.long)
    sign_out  = torch.zeros(N, M_pad, device=device)
    bias_out  = torch.zeros(N, n_blocks, device=device, dtype=torch.long)

    default_bias = 2 ** (e_bits - 1) - 1
    bias_radius  = max(1, 2 ** (e_bits - 2))

    for row in range(N):
        for blk, i in enumerate(range(0, M_pad, pow2_bs)):
            # ------------------------------------------------------------
            # Block slice in the padded domain
            # ------------------------------------------------------------
            w_block = W_signed[row, i : i + pow2_bs]   # (pow2_bs,)

            # Skip blocks that are entirely padding (no real weights)
            if mask_pad[row, i : i + pow2_bs].sum() == 0:
                bias_out[row, blk] = default_bias
                continue

            # ------------------------------------------------------------
            # Forward Hadamard (exactly pow2_bs — no further padding)
            # ------------------------------------------------------------
            w_had = _fht_vec(w_block)                   # (pow2_bs,)
            w_had_abs = w_had.abs()

            # ------------------------------------------------------------
            # Rotated Hessian diagonal
            # ------------------------------------------------------------
            H_block  = H_blocks_layer[blk].to(device)  # (block_size, block_size)
            h_diag   = torch.diag(H_block).clamp(min=0.0)  # (block_size,)

            # Pad h_diag to pow2_bs (extra positions get 0 sensitivity)
            h_diag_p = F.pad(h_diag, (0, pow2_bs - h_diag.shape[0]))
            h_diag_had = _fht_vec(h_diag_p).abs()      # (pow2_bs,)
            H_had = torch.diag(h_diag_had)              # (pow2_bs, pow2_bs)

            # ------------------------------------------------------------
            # Alpha / bias search
            # ------------------------------------------------------------
            alpha_init = w_had_abs.pow(2).mean().sqrt().clamp(min=1e-4)
            Hw         = H_had @ w_had_abs

            best_loss  = float('inf')
            best_alpha = alpha_init
            best_bias  = default_bias

            for bias_cand in range(
                default_bias - bias_radius,
                default_bias + bias_radius + 1,
            ):
                alpha_tmp = alpha_init.clone()
                for _ in range(5):
                    _, _, b = assign_fp4_dynamic(
                        w_had_abs, alpha_tmp, e_bits, m_bits, bias=bias_cand
                    )
                    Hb        = H_had @ b
                    num       = torch.dot(b, Hw)
                    den       = torch.dot(b, Hb) + 1e-8
                    alpha_tmp = (num / den).clamp(min=1e-6)

                recon_had = torch.sign(w_had) * alpha_tmp * b
                residual  = w_had - recon_had
                loss      = torch.dot(residual, H_had @ residual)

                if loss < best_loss:
                    best_loss  = loss
                    best_alpha = alpha_tmp
                    best_bias  = bias_cand

            # ------------------------------------------------------------
            # Quantise scale, then finalise components
            # ------------------------------------------------------------
            alpha_q = quantize_scale(best_alpha, e_bits_scale, m_bits_scale)

            exp, mant, _ = assign_fp4_dynamic(
                w_had_abs, alpha_q, e_bits, m_bits, bias=best_bias
            )

            alpha_out[row, i : i + pow2_bs] = alpha_q
            e_out    [row, i : i + pow2_bs] = exp
            m_out    [row, i : i + pow2_bs] = mant
            sign_out [row, i : i + pow2_bs] = torch.sign(w_had)
            bias_out [row, blk]             = best_bias

    # ------------------------------------------------------------------
    # Reconstruction in the Hadamard domain → spatial
    # ------------------------------------------------------------------
    # Expand bias from (N, n_blocks) to (N, M_pad)
    bias_exp = bias_out.repeat_interleave(pow2_bs, dim=1)   # (N, M_pad)

    W_q_had = (
        alpha_out
        * sign_out
        * 2.0 ** (e_out.float() - bias_exp.float())
        * (1.0 + m_out.float() / (2 ** m_bits))
    )   # (N, M_pad)

    # Inverse FHT: one _fht call per block (self-inverse property)
    W_q_spatial = torch.zeros_like(W_q_had)
    for row in range(N):
        for blk, i in enumerate(range(0, M_pad, pow2_bs)):
            block = W_q_had[row, i : i + pow2_bs]
            W_q_spatial[row, i : i + pow2_bs] = _fht_vec(block)

    # Reverse pre-conditioning and strip padding
    W_q_spatial = W_q_spatial * fixed_signs    # undo sign flip
    W_q_spatial = W_q_spatial[:, :M]          # strip padded columns
    W_q_spatial = W_q_spatial * mask_mat       # reapply sparsity mask
    # --- DIAGNOSTIC: print round-trip error for row 0 ---
    with torch.no_grad():
        row = 0
        i = 0
        blk = 0
        w_orig = W_signed[row, i : i + pow2_bs]
        w_fwd  = _fht_vec(w_orig)
        w_inv  = _fht_vec(w_fwd)
        print(f"[DIAG] block_size={block_size}, pow2_bs={pow2_bs}, M={M}, M_pad={M_pad}")
        print(f"[DIAG] Round-trip max error: {(w_orig - w_inv).abs().max().item():.2e}")
        print(f"[DIAG] W_q_had[0,:8]:    {W_q_had[0,:8]}")
        print(f"[DIAG] W_q_spatial[0,:8]: {W_q_spatial[0,:8]}")
        print(f"[DIAG] W_mat[0,:8]:       {W_mat[0,:8]}")
    return {
        'alpha':                alpha_out[:, :M].view_as(W),
        'exponent':             e_out    [:, :M].view_as(W),
        'mantissa':             m_out    [:, :M].view_as(W),
        'sign':                 sign_out [:, :M].view_as(W),
        'bias':                 bias_out,                      # (N, n_blocks)
        'fixed_signs':          fixed_signs[:, :M],            # (1, M) — save for inference
        'reconstructed_weight': W_q_spatial.view_as(W),
    }

def _fht_blocks(W: torch.Tensor, pow2_bs: int) -> torch.Tensor:
    """
    Apply FHT independently to each non-overlapping block of size pow2_bs
    along dim-1 of W (shape N x M_pad, where M_pad is a multiple of pow2_bs).
    Returns same shape. Fully vectorized — no Python loops, no contiguity issues.
    """
    N, M_pad = W.shape
    n_blocks = M_pad // pow2_bs
    # Reshape so each block is a row in the batch dimension
    W_blocks = W.reshape(N * n_blocks, pow2_bs).contiguous()
    W_had    = _fht(W_blocks)                          # (N*n_blocks, pow2_bs)
    return W_had.reshape(N, M_pad)
 
 
# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
 
def reconstruct_layer_hadamard_v8(
    layer,
    H_blocks_layer,
    block_size: int,
    e_bits: int,
    m_bits: int,
    e_bits_scale: int,
    m_bits_scale: int,
    device,
    seed: int = 42,
):
    """
    V8: Corrected Hadamard-domain adaptive FP quantisation.
 
    Bugs fixed vs V7
    ----------------
    1.  Contiguity: all FHT inputs are made contiguous before reshape/view.
        Non-contiguous slices caused _fht's internal .view() to silently
        write results to the wrong memory locations.
 
    2.  Vectorized FHT via _fht_blocks: eliminates the per-row, per-block
        Python loop that was the source of the contiguity issues.
 
    3.  Double-alpha in reconstruction: W_q_had was computed as
            alpha * sign * 2^(e-bias) * (1 + m/2^M)
        which embeds alpha twice (once explicitly, once inside the FP value).
        The reconstruction now computes the FP value directly from stored
        components without re-multiplying alpha.
 
    4.  Mask threshold: (W != 0) on float32 weights misses near-zero weights.
        Changed to an absolute threshold so legitimate small weights survive.
    """
 
    W = layer.weight.data.to(device)
 
    # FIX 4: use a small epsilon threshold instead of exact zero
    mask = (W.abs() > 1e-9).float()
 
    if W.dim() == 4:
        W_mat    = W.view(W.shape[0], -1)
        mask_mat = mask.view(W.shape[0], -1)
    else:
        W_mat    = W
        mask_mat = mask
 
    N, M = W_mat.shape
 
    # ------------------------------------------------------------------
    # Block geometry — block_size must be power of 2 (assert to be safe)
    # ------------------------------------------------------------------
    pow2_bs  = 2 ** int(math.ceil(math.log2(block_size))) if block_size > 1 else 1
    n_blocks = math.ceil(M / block_size)
    M_pad    = n_blocks * pow2_bs
 
    # ------------------------------------------------------------------
    # Pad weight matrix and mask to M_pad
    # ------------------------------------------------------------------
    W_pad    = F.pad(W_mat,    (0, M_pad - M)).contiguous()   # (N, M_pad)
    mask_pad = F.pad(mask_mat, (0, M_pad - M)).contiguous()
 
    # ------------------------------------------------------------------
    # Deterministic pre-conditioning signs
    # ------------------------------------------------------------------
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    raw         = torch.randn(1, M_pad, device=device, generator=rng)
    fixed_signs = torch.sign(raw)
    fixed_signs[fixed_signs == 0] = 1.0
    fixed_signs[:, M:] = 0.0          # padded positions contribute nothing
 
    W_signed = (W_pad * fixed_signs).contiguous()    # (N, M_pad)
 
    # ------------------------------------------------------------------
    # Forward FHT — vectorized over all blocks simultaneously
    # ------------------------------------------------------------------
    W_had_all = _fht_blocks(W_signed, pow2_bs)       # (N, M_pad)
 
    # ------------------------------------------------------------------
    # Storage tensors
    # ------------------------------------------------------------------
    # Store the reconstructed Hadamard-domain values directly (not components)
    # so the inverse pass is a single _fht_blocks call with no reconstruction
    # arithmetic that can introduce alpha errors.
    W_q_had_all = torch.zeros_like(W_had_all)        # (N, M_pad)
    bias_out    = torch.zeros(N, n_blocks, device=device, dtype=torch.long)
 
    # Also store components for the return dict
    alpha_out = torch.zeros(N, M_pad, device=device)
    e_out     = torch.zeros(N, M_pad, device=device, dtype=torch.long)
    m_out     = torch.zeros(N, M_pad, device=device, dtype=torch.long)
    sign_out  = torch.zeros(N, M_pad, device=device)
 
    default_bias = 2 ** (e_bits - 1) - 1
    bias_radius  = max(1, 2 ** (e_bits - 2))
 
    for row in range(N):
        for blk in range(n_blocks):
            i   = blk * pow2_bs
            end = i + pow2_bs
 
            w_had = W_had_all[row, i:end].contiguous()   # (pow2_bs,)
 
            # Skip if this block is entirely padding / zero weights
            if mask_pad[row, i:end].sum() == 0:
                bias_out[row, blk] = default_bias
                continue
 
            w_had_abs = w_had.abs()
 
            # ----------------------------------------------------------
            # Rotated Hessian diagonal
            # ----------------------------------------------------------
            H_block = H_blocks_layer[blk].to(device)     # (block_size, block_size)
            h_diag  = torch.diag(H_block).clamp(min=0.0) # (block_size,)
            h_diag_p = F.pad(h_diag, (0, pow2_bs - h_diag.shape[0])).contiguous()
            h_diag_had = _fht(h_diag_p.unsqueeze(0)).squeeze(0).abs()  # (pow2_bs,)
            H_had = torch.diag(h_diag_had)                              # (pow2_bs, pow2_bs)
 
            # ----------------------------------------------------------
            # Alpha / bias search
            # ----------------------------------------------------------
            alpha_init = w_had_abs.pow(2).mean().sqrt().clamp(min=1e-4)
            Hw = H_had @ w_had_abs
 
            best_loss  = float('inf')
            best_alpha = alpha_init.clone()
            best_bias  = default_bias
            best_b     = None
 
            for bias_cand in range(
                default_bias - bias_radius,
                default_bias + bias_radius + 1,
            ):
                alpha_tmp = alpha_init.clone()
                for _ in range(5):
                    _, _, b = assign_fp4_dynamic(
                        w_had_abs, alpha_tmp, e_bits, m_bits, bias=bias_cand
                    )
                    Hb        = H_had @ b
                    num       = torch.dot(b, Hw)
                    den       = torch.dot(b, Hb) + 1e-8
                    alpha_tmp = (num / den).clamp(min=1e-6)
 
                recon_had = torch.sign(w_had) * alpha_tmp * b
                residual  = w_had - recon_had
                loss      = torch.dot(residual, H_had @ residual)
 
                if loss < best_loss:
                    best_loss  = loss
                    best_alpha = alpha_tmp.clone()
                    best_bias  = bias_cand
                    best_b     = b.clone()
 
            # ----------------------------------------------------------
            # Quantise scale
            # ----------------------------------------------------------
            alpha_q = quantize_scale(best_alpha, e_bits_scale, m_bits_scale)
 
            # Final quantised Hadamard-domain values
            # FIX 3: store the actual reconstructed had values directly,
            # not the raw components — avoids any double-alpha on readback.
            exp, mant, basis = assign_fp4_dynamic(
                w_had_abs, alpha_q, e_bits, m_bits, bias=best_bias
            )
            w_q_had_block = torch.sign(w_had) * alpha_q * basis   # (pow2_bs,)
 
            W_q_had_all[row, i:end] = w_q_had_block
 
            # Store components for the return dict
            alpha_out[row, i:end] = alpha_q
            e_out    [row, i:end] = exp
            m_out    [row, i:end] = mant
            sign_out [row, i:end] = torch.sign(w_had)
            bias_out [row, blk]   = best_bias
 
    # ------------------------------------------------------------------
    # FIX 1+2: Vectorized inverse FHT — contiguous, no Python slice loops
    # ------------------------------------------------------------------
    W_q_spatial = _fht_blocks(W_q_had_all, pow2_bs)   # (N, M_pad)
 
    # Undo pre-conditioning, strip padding, reapply sparsity mask
    W_q_spatial = W_q_spatial * fixed_signs            # undo sign flip
    W_q_spatial = W_q_spatial[:, :M].contiguous()      # strip padding
    W_q_spatial = W_q_spatial * mask_mat               # reapply mask
 
    return {
        'alpha':                alpha_out[:, :M].view_as(W),
        'exponent':             e_out    [:, :M].view_as(W),
        'mantissa':             m_out    [:, :M].view_as(W),
        'sign':                 sign_out [:, :M].view_as(W),
        'bias':                 bias_out,
        'fixed_signs':          fixed_signs[:, :M],
        'reconstructed_weight': W_q_spatial.view_as(W),
    }


def reconstruct_layer_hadamard_v10(
    layer,
    H_blocks_layer,
    block_size: int,
    e_bits: int,
    m_bits: int,
    e_bits_scale: int,
    m_bits_scale: int,
    device,
    seed: int = 42,
):
    """
    V10: Hadamard-domain adaptive FP quantisation.
 
    Changes vs V9
    -------------
    1.  Hessian normalization: compute_hessian_blocks now divides by N,
        making sensitivity estimates comparable across layers.
 
    2.  Alpha update restored to Hessian-weighted form:
            num = dot(h * b,  w_had_abs)
            den = dot(h * b,  b)
        where h = h_had_importance (per-element diagonal sensitivity in
        the Hadamard domain). This is the correct diagonal-H OBS update.
 
    3.  h_had_importance uses abs() of the rotated diagonal (not pow(2)),
        since the diagonal entries of the rotated Hessian are already
        second-order quantities — squaring them double-counts the order.
 
    4.  Loss uses the same h_had_importance weighting as the alpha update,
        so the bias search and alpha search are optimizing the same objective.
    """
    W = layer.weight.data.to(device)
    mask = (W.abs() > 1e-9).float()
 
    if W.dim() == 4:
        W_mat    = W.view(W.shape[0], -1)
        mask_mat = mask.view(W.shape[0], -1)
    else:
        W_mat    = W
        mask_mat = mask
 
    N, M = W_mat.shape
 
    # ------------------------------------------------------------------
    # Block geometry
    # ------------------------------------------------------------------
    pow2_bs  = 2 ** int(math.ceil(math.log2(block_size))) if block_size > 1 else 1
    n_blocks = math.ceil(M / block_size)
    M_pad    = n_blocks * pow2_bs
 
    W_pad    = F.pad(W_mat,    (0, M_pad - M)).contiguous()
    mask_pad = F.pad(mask_mat, (0, M_pad - M)).contiguous()
 
    # ------------------------------------------------------------------
    # Deterministic pre-conditioning signs
    # ------------------------------------------------------------------
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    raw         = torch.randn(1, M_pad, device=device, generator=rng)
    fixed_signs = torch.sign(raw)
    fixed_signs[fixed_signs == 0] = 1.0
    fixed_signs[:, M:] = 0.0
 
    W_signed  = (W_pad * fixed_signs).contiguous()
    W_had_all = _fht_blocks(W_signed, pow2_bs)         # (N, M_pad)
 
    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    W_q_had_all = torch.zeros_like(W_had_all)
    bias_out    = torch.zeros(N, n_blocks, device=device, dtype=torch.long)
    alpha_out   = torch.zeros(N, M_pad, device=device)
    e_out       = torch.zeros(N, M_pad, device=device, dtype=torch.long)
    m_out       = torch.zeros(N, M_pad, device=device, dtype=torch.long)
    sign_out    = torch.zeros(N, M_pad, device=device)
 
    default_bias = 2 ** (e_bits - 1) - 1
    bias_radius  = max(1, 2 ** (e_bits - 2))
 
    for row in range(N):
        for blk in range(n_blocks):
            i   = blk * pow2_bs
            end = i + pow2_bs
 
            w_had = W_had_all[row, i:end].contiguous()
 
            if mask_pad[row, i:end].sum() == 0:
                bias_out[row, blk] = default_bias
                continue
 
            w_had_abs = w_had.abs()
 
            # ----------------------------------------------------------
            # Rotated Hessian diagonal (FIX 3: abs not pow(2))
            # ----------------------------------------------------------
            H_block  = H_blocks_layer[blk].to(device)
            h_diag   = torch.diag(H_block).clamp(min=1e-8)
            h_diag_p = F.pad(h_diag, (0, pow2_bs - h_diag.shape[0])).contiguous()
 
            # Rotate diagonal into Hadamard domain and take abs
            h_had_importance = _fht(h_diag_p.unsqueeze(0)).squeeze(0).abs()
            h_had_importance = h_had_importance.clamp(min=1e-8)
 
            # ----------------------------------------------------------
            # Alpha / bias search (FIX 2: Hessian-weighted alpha update)
            # ----------------------------------------------------------
            alpha_init = w_had_abs.pow(2).mean().sqrt().clamp(min=1e-4)
 
            best_loss  = float('inf')
            best_alpha = alpha_init.clone()
            best_bias  = default_bias
 
            for bias_cand in range(
                default_bias - bias_radius,
                default_bias + bias_radius + 1,
            ):
                alpha_tmp = alpha_init.clone()
                for _ in range(5):
                    _, _, b = assign_fp4_dynamic(
                        w_had_abs, alpha_tmp, e_bits, m_bits, bias=bias_cand
                    )
                    # Diagonal-H OBS update: weight by per-element importance
                    hb  = h_had_importance * b
                    num = torch.dot(hb, w_had_abs)
                    den = torch.dot(hb, b) + 1e-8
                    alpha_tmp = (num / den).clamp(min=1e-6)
 
                recon_had = torch.sign(w_had) * alpha_tmp * b
                residual  = w_had - recon_had
                # FIX 4: loss uses same weighting as alpha update
                loss = torch.dot(h_had_importance, residual.pow(2)).item()
 
                if loss < best_loss:
                    best_loss  = loss
                    best_alpha = alpha_tmp.clone()
                    best_bias  = bias_cand
 
            # ----------------------------------------------------------
            # Quantise scale and store
            # ----------------------------------------------------------
            alpha_q = quantize_scale(best_alpha, e_bits_scale, m_bits_scale)
 
            exp, mant, basis = assign_fp4_dynamic(
                w_had_abs, alpha_q, e_bits, m_bits, bias=best_bias
            )
 
            W_q_had_all[row, i:end] = torch.sign(w_had) * alpha_q * basis
 
            alpha_out[row, i:end] = alpha_q
            e_out    [row, i:end] = exp
            m_out    [row, i:end] = mant
            sign_out [row, i:end] = torch.sign(w_had)
            bias_out [row, blk]   = best_bias
 
    # ------------------------------------------------------------------
    # Inverse FHT: vectorized, contiguous, single call
    # ------------------------------------------------------------------
    W_q_spatial = _fht_blocks(W_q_had_all, pow2_bs)
 
    W_q_spatial = W_q_spatial * fixed_signs
    W_q_spatial = W_q_spatial[:, :M].contiguous()
    W_q_spatial = W_q_spatial * mask_mat
 
    return {
        'alpha':                alpha_out[:, :M].view_as(W),
        'exponent':             e_out    [:, :M].view_as(W),
        'mantissa':             m_out    [:, :M].view_as(W),
        'sign':                 sign_out [:, :M].view_as(W),
        'bias':                 bias_out,
        'fixed_signs':          fixed_signs[:, :M],
        'reconstructed_weight': W_q_spatial.view_as(W),
    }


# def _quantize_input_preshifted(x, act_rho, block_size, e_bits, m_bits):
#     """
#     Per-tensor activation quantization using calibrated scalar rho.
#     scale = 2^(-rho) is the tensor-wise activation scale derived
#     from the pre-shifted bias decomposition.
#     """
#     scale      = 2.0 ** (-act_rho)
#     orig_shape = x.shape
#     x_2d       = x.reshape(-1, orig_shape[-1]).float()
#     N, K       = x_2d.shape

#     pad      = (block_size - K % block_size) % block_size
#     x_pad    = F.pad(x_2d, (0, pad))
#     x_blocks = x_pad.view(N, -1, block_size)           # (N, n_blocks, bs)

#     scale_t = torch.full(
#         (N, x_blocks.shape[1], 1), scale, device=x.device
#     )
#     _, _, basis = assign_fp4(
#         x_blocks.abs(), scale_t, e_bits, m_bits
#     )
#     x_hat = torch.sign(x_blocks) * scale_t * basis
#     x_hat = x_hat.view(N, -1)[:, :K]
#     return x_hat.reshape(orig_shape).to(x.dtype)


# def _quantize_input_preshifted(x, act_rho, act_b_ori, block_size, e_bits, m_bits):
#     """
#     Apply per-block pre-shifted exponent bias to activations then quantize.

#     act_rho:   scalar — tensor-wise exponent bias stored at calibration
#     act_b_ori: (n_blocks,) long tensor — per-block integer correction
#                Each block of block_size input features gets scaled by
#                2^(b_ori_k) before quantization, matching the 2^(-b_ori_k)
#                absorbed into the corresponding weight columns.

#     Mathematical correctness:
#         weight_q[:, k*B:(k+1)*B] = quantize(W[:, k*B:(k+1)*B] * 2^(-b_ori_k))
#         x_scaled[:, k*B:(k+1)*B] = quantize(x[:, k*B:(k+1)*B] * 2^(b_ori_k),
#                                              scale=2^(-rho))
#         x_scaled @ weight_q.T ≈ x @ W.T  ✓
#     """
#     orig_shape = x.shape
#     x_2d       = x.reshape(-1, orig_shape[-1]).float()
#     N, K       = x_2d.shape
#     n_blocks   = math.ceil(K / block_size)

#     # Apply per-block b_ori scaling to equalize activation blocks
#     if act_b_ori is not None:
#         x_scaled = x_2d.clone()
#         for k in range(n_blocks):
#             start = k * block_size
#             end   = min(start + block_size, K)
#             scale = 2.0 ** act_b_ori[k].float().item()
#             x_scaled[:, start:end] = x_2d[:, start:end] * scale
#     else:
#         x_scaled = x_2d

#     # Now apply tensor-wise rho quantization
#     # After b_ori scaling, all blocks have similar magnitude,
#     # so the single tensor-wise scale 2^(-rho) is appropriate
#     tensor_scale = 2.0 ** (-act_rho)

#     pad      = (block_size - K % block_size) % block_size
#     x_pad    = F.pad(x_scaled, (0, pad))
#     x_blocks = x_pad.view(N, -1, block_size)

#     scale_t = torch.full(
#         (N, x_blocks.shape[1], 1),
#         tensor_scale,
#         device=x.device
#     )

#     _, _, basis = assign_fp4(x_blocks.abs(), scale_t, e_bits, m_bits)
#     x_hat = torch.sign(x_blocks) * scale_t * basis
#     x_hat = x_hat.view(N, -1)[:, :K]

#     return x_hat.reshape(orig_shape).to(x.dtype)

# def _quantize_input_preshifted(x, act_rho, act_b_ori, block_size, e_bits, m_bits):
#     """
#     Per-block activation quantization using pre-shifted exponent bias.
#     Each block k gets scale = 2^(-(rho + b_ori_k)).
#     """
#     orig_shape = x.shape
#     x_2d       = x.reshape(-1, orig_shape[-1]).float()
#     N, K       = x_2d.shape

#     pad      = (block_size - K % block_size) % block_size
#     x_pad    = F.pad(x_2d, (0, pad))
#     x_blocks = x_pad.view(N, -1, block_size)   # (N, n_blocks, block_size)
#     n_blocks = x_blocks.shape[1]

#     # Build per-block scales: scale_k = 2^(-(rho + b_ori_k))
#     # act_b_ori shape: (n_blocks_calib,) — may differ if padding changed count
#     scales = torch.zeros(n_blocks, device=x.device)
#     for k in range(n_blocks):
#         if act_b_ori is not None and k < len(act_b_ori):
#             b_k = act_b_ori[k].float().item()
#         else:
#             b_k = 0.0
#         scales[k] = 2.0 ** (-(act_rho + b_k))

#     # Expand scales to (N, n_blocks, 1) for broadcast
#     scale_t = scales.unsqueeze(0).unsqueeze(-1).expand(N, n_blocks, 1)

#     _, _, basis = assign_fp4(x_blocks.abs(), scale_t, e_bits, m_bits)
#     x_hat = torch.sign(x_blocks) * scale_t * basis
#     x_hat = x_hat.view(N, -1)[:, :K]

#     return x_hat.reshape(orig_shape).to(x.dtype)



def _quantize_input_preshifted(x, act_rho, act_b_ori, block_size, e_bits, m_bits):
    orig_shape = x.shape
    x_2d       = x.reshape(-1, orig_shape[-1]).float()
    N, K       = x_2d.shape

    pad      = (block_size - K % block_size) % block_size
    x_pad    = F.pad(x_2d, (0, pad))
    x_blocks = x_pad.view(N, -1, block_size)
    n_blocks = x_blocks.shape[1]

    # Step 1: scale each block UP by 2^(b_ori_k)
    # This is the coordinate change matching weight_q = W * 2^(-b_ori)
    b_ori_scales = torch.zeros(n_blocks, device=x.device)
    for k in range(n_blocks):
        if act_b_ori is not None and k < len(act_b_ori):
            b_ori_scales[k] = act_b_ori[k].float().item()
        else:
            b_ori_scales[k] = 0.0

    activation_scale = (2.0 ** b_ori_scales)  # (n_blocks,) — scale UP
    x_blocks = x_blocks * activation_scale.unsqueeze(0).unsqueeze(-1)

    # Step 2: quantize using tensor-wise rho scale
    tensor_scale = 2.0 ** (-act_rho)
    scale_t = torch.full(
        (N, n_blocks, 1), tensor_scale, device=x.device
    )

    _, _, basis = assign_fp4(x_blocks.abs(), scale_t, e_bits, m_bits)
    x_hat = torch.sign(x_blocks) * scale_t * basis
    x_hat = x_hat.view(N, -1)[:, :K]

    return x_hat.reshape(orig_shape).to(x.dtype)


def _build_codebook(e_bits: int, m_bits: int,
                    bias: int,
                    device: torch.device) -> torch.Tensor:
    """
    Build the FP codebook for a given (e_bits, m_bits, bias) triple.
    Returns a 1-D tensor of size 2^e_bits * 2^m_bits.
    """
    e_levels = torch.arange(0, 2 ** e_bits,  device=device, dtype=torch.float32)
    m_levels = torch.arange(0, 2 ** m_bits,  device=device, dtype=torch.float32)
    base     = 2.0 ** (e_levels - bias)                        # [E]
    mf       = 1.0 + m_levels / (2 ** m_bits)                  # [M]
    codebook = (base.unsqueeze(1) * mf.unsqueeze(0)).reshape(-1)  # [E*M]
    return codebook


def chunked_codebook_lookup(x_norm, codebook, chunk_size=32):
    # x_norm: [N, B, bs]
    N, B, bs = x_norm.shape
    device = x_norm.device

    best_dist = torch.full((N, B, bs), float('inf'), device=device)
    best_idx  = torch.zeros((N, B, bs), dtype=torch.long, device=device)

    K = codebook.shape[0]

    for start in range(0, K, chunk_size):
        end = min(start + chunk_size, K)
        cb_chunk = codebook[start:end]  # [chunk]

        dist_chunk = (x_norm.unsqueeze(-1)
                      - cb_chunk.view(1, 1, 1, -1)).abs()  # [N,B,bs,chunk]

        local_idx = dist_chunk.argmin(dim=-1)              # [N,B,bs]
        local_val = dist_chunk.gather(
            -1, local_idx.unsqueeze(-1)
        ).squeeze(-1)                                      # [N,B,bs]

        better = local_val < best_dist

        best_dist = torch.where(better, local_val, best_dist)
        best_idx  = torch.where(
            better,
            local_idx + start,
            best_idx
        )

    return best_idx

def chunked_lookup_full(x_norm, codebook, process_chunk, B_chunk=8, K_chunk=32):
    N, B, bs = x_norm.shape

    for b_start in range(0, B, B_chunk):
        b_end = min(b_start + B_chunk, B)

        x_chunk = x_norm[:, b_start:b_end, :]  # [N, Bc, bs]

        best_dist = torch.full_like(x_chunk, float('inf'))
        best_idx  = torch.zeros_like(x_chunk, dtype=torch.long)

        K = codebook.shape[0]

        for k_start in range(0, K, K_chunk):
            cb_chunk = codebook[k_start:k_start+K_chunk]

            dist = (x_chunk.unsqueeze(-1)
                    - cb_chunk.view(1,1,1,-1)).abs()

            local_idx = dist.argmin(dim=-1)
            local_val = dist.gather(
                -1, local_idx.unsqueeze(-1)
            ).squeeze(-1)

            better = local_val < best_dist

            best_dist = torch.where(better, local_val, best_dist)
            best_idx  = torch.where(better, local_idx + k_start, best_idx)

        # 🔴 immediately consume instead of storing
        process_chunk(b_start, b_end, best_idx)


def chunked_lookup_basis(x_norm, codebook, writer_fn, 
                          chunk_blocks=1, chunk_rows=256):
    """
    Processes both blocks and rows in chunks to handle large layers
    on memory-constrained GPUs.
    chunk_rows: number of rows (N) to process at once.
                Reduce to 64 or 32 if still OOMing.
    """
    N, n_blocks, pow2_bs = x_norm.shape
    C = codebook.shape[0]

    # Pre-allocate output on CPU, fill block by block
    b_out_full = torch.zeros(N, n_blocks, pow2_bs, device='cpu')

    for b_start in range(0, n_blocks, chunk_blocks):
        b_end   = min(b_start + chunk_blocks, n_blocks)
        n_blk   = b_end - b_start

        # Allocate this block's output on CPU
        b_chunk_cpu = torch.zeros(N, n_blk, pow2_bs, device='cpu')

        for r_start in range(0, N, chunk_rows):
            r_end   = min(r_start + chunk_rows, N)

            x_chunk = x_norm[r_start:r_end, b_start:b_end, :]  # [r, blk, bs]

            dist    = (x_chunk.unsqueeze(-1) - 
                       codebook.view(1, 1, 1, -1)).abs()        # [r, blk, bs, C]
            indices = dist.argmin(dim=-1)                        # [r, blk, bs]
            b_vals  = codebook[indices]                          # [r, blk, bs]

            b_chunk_cpu[r_start:r_end] = b_vals.cpu()

            del x_chunk, dist, indices, b_vals
            torch.cuda.empty_cache()

        # Move completed block chunk to GPU and call writer
        b_chunk_gpu = b_chunk_cpu.to(x_norm.device)
        writer_fn(b_start, b_end, b_chunk_gpu)

        del b_chunk_cpu, b_chunk_gpu
        torch.cuda.empty_cache()




def reconstruct_layer_hadamard_v10_fast(
    layer, H_blocks_layer, block_size, e_bits, m_bits,
    e_bits_scale, m_bits_scale, device, seed=42):

    with torch.no_grad():
        W = layer.weight.data.to(device)
        mask = (W.abs() > 1e-9).float()

        if W.dim() == 4:
            W_mat = W.view(W.shape[0], -1)
            mask_mat = mask.view(W.shape[0], -1)
        else:
            W_mat = W
            mask_mat = mask

        N, M = W_mat.shape
        del W, mask
        torch.cuda.empty_cache()

        pow2_bs  = 2 ** int(math.ceil(math.log2(block_size)))
        n_blocks = math.ceil(M / block_size)
        M_pad    = n_blocks * pow2_bs

        W_pad    = F.pad(W_mat, (0, M_pad - M))
        mask_pad = F.pad(mask_mat, (0, M_pad - M))
        del W_mat
        torch.cuda.empty_cache()

        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        fixed_signs = torch.sign(torch.randn(1, M_pad, device=device, generator=rng))
        fixed_signs[fixed_signs == 0] = 1.0
        fixed_signs[:, M:] = 0.0

        W_signed = W_pad * fixed_signs
        del W_pad
        torch.cuda.empty_cache()

        W_had     = _fht_blocks(W_signed, pow2_bs).view(N, n_blocks, pow2_bs)
        del W_signed
        torch.cuda.empty_cache()

        w_had_sign = torch.sign(W_had)
        w_had_abs  = W_had.abs()
        del W_had
        torch.cuda.empty_cache()

        # Hessian diag — keep on CPU
        h_diag = []
        for blk in range(n_blocks):
            d = torch.diag(H_blocks_layer[blk].to(device)).clamp(min=1e-8)
            h_diag.append(F.pad(d, (0, pow2_bs - d.shape[0])).cpu())
            del d
        torch.cuda.empty_cache()

        h_imp = _fht(torch.stack(h_diag)).abs().clamp(min=1e-8)  # CPU [n_blocks, pow2_bs]
        del h_diag

        alpha = w_had_abs.pow(2).mean(dim=-1).sqrt().clamp(min=1e-4)  # [N, n_blocks] GPU

        default_bias = 2 ** (e_bits - 1) - 1
        bias_radius  = max(1, 2 ** (e_bits - 2))
        bias_range   = list(range(default_bias - bias_radius,
                                  default_bias + bias_radius + 1))

        # Keep best_* on CPU
        best_loss  = torch.full((N, n_blocks), float('inf'))
        best_alpha = alpha.clone().cpu()
        best_bias  = torch.full((N, n_blocks), default_bias, dtype=torch.long)

        # ── bias search ──────────────────────────────────────────
        for bias_cand in bias_range:
            codebook  = _build_codebook(e_bits, m_bits, bias_cand, device)
            alpha_tmp = alpha.clone()

            for _ in range(5):
                # bring h_imp to GPU only for this iteration
                h_imp_gpu = h_imp.unsqueeze(0).to(device)  # [1, n_blocks, pow2_bs]

                x_norm = w_had_abs / alpha_tmp.unsqueeze(-1).clamp(min=1e-8)

                num = torch.zeros((N, n_blocks), device=device)
                den = torch.zeros((N, n_blocks), device=device)

                def alpha_writer(b_start, b_end, b_chunk):
                    h_chunk = h_imp_gpu[:, b_start:b_end, :]
                    w_chunk = w_had_abs[:, b_start:b_end, :]
                    hb = h_chunk * b_chunk
                    num[:, b_start:b_end] = (hb * w_chunk).sum(dim=-1)
                    den[:, b_start:b_end] = (hb * b_chunk).sum(dim=-1)

                chunked_lookup_basis(x_norm, codebook, alpha_writer)
                del x_norm, h_imp_gpu
                torch.cuda.empty_cache()

                alpha_tmp = (num / (den + 1e-8)).clamp(min=1e-6)
                del num, den

            # loss eval
            h_imp_gpu = h_imp.unsqueeze(0).to(device)
            x_norm    = w_had_abs / alpha_tmp.unsqueeze(-1).clamp(min=1e-8)
            loss      = torch.zeros((N, n_blocks), device=device)

            def loss_writer(b_start, b_end, b_chunk):
                s     = w_had_sign[:, b_start:b_end, :]
                a     = alpha_tmp[:, b_start:b_end].unsqueeze(-1)
                recon = s * a * b_chunk
                residual = (w_had_sign[:, b_start:b_end, :] *
                            w_had_abs[:, b_start:b_end, :]) - recon
                h_chunk = h_imp_gpu[:, b_start:b_end, :]
                loss[:, b_start:b_end] = (h_chunk * residual.pow(2)).sum(dim=-1)

            chunked_lookup_basis(x_norm, codebook, loss_writer)
            del x_norm, h_imp_gpu
            torch.cuda.empty_cache()

            improved   = loss.cpu() < best_loss
            best_loss  = torch.where(improved, loss.cpu(), best_loss)
            best_alpha = torch.where(improved, alpha_tmp.cpu(), best_alpha)
            best_bias  = torch.where(
                improved,
                torch.full_like(best_bias, bias_cand),
                best_bias)
            del loss, alpha_tmp, improved, codebook
            torch.cuda.empty_cache()

        del alpha, best_loss
        torch.cuda.empty_cache()

        # ── quantize alpha ───────────────────────────────────────
        best_alpha_gpu = best_alpha.to(device)
        alpha_q        = quantize_scale_tensor(best_alpha_gpu, e_bits_scale, m_bits_scale)
        del best_alpha_gpu
        torch.cuda.empty_cache()

        # ── final reconstruction — original logic, unchanged ─────
        best_bias_gpu = best_bias.to(device)
        W_q = torch.zeros(N, M_pad, device=device).view(N, n_blocks, pow2_bs)

        for bias_val in best_bias_gpu.unique().tolist():
            codebook = _build_codebook(e_bits, m_bits, int(bias_val), device)
            bmask    = (best_bias_gpu == bias_val)
            aq       = alpha_q * bmask.float()

            x_norm = w_had_abs / aq.unsqueeze(-1).clamp(min=1e-8)

            def final_writer(b_start, b_end, b_chunk):
                s     = w_had_sign[:, b_start:b_end, :]
                a     = aq[:, b_start:b_end].unsqueeze(-1)
                recon = s * a * b_chunk
                write = bmask[:, b_start:b_end].unsqueeze(-1)
                W_q[:, b_start:b_end, :] = torch.where(write, recon,
                                                        W_q[:, b_start:b_end, :])

            chunked_lookup_basis(x_norm, codebook, final_writer)
            del x_norm, codebook, bmask, aq
            torch.cuda.empty_cache()

        del w_had_abs, w_had_sign, best_bias_gpu
        torch.cuda.empty_cache()

        W_q   = W_q.view(N, M_pad)
        W_out = _fht_blocks(W_q, pow2_bs)
        del W_q
        W_out = W_out * fixed_signs
        W_out = W_out[:, :M] * mask_mat
        del fixed_signs, mask_mat, mask_pad

        alpha_q_exp = alpha_q.unsqueeze(-1).expand(-1, -1, pow2_bs)
        alpha_q_exp = alpha_q_exp.reshape(N, M_pad)[:, :M]
        del alpha_q

        layer_W = layer.weight.data.to(device)
        return {
            'alpha':                alpha_q_exp.view_as(layer_W),
            'bias':                 best_bias.to(device),
            'reconstructed_weight': W_out.view_as(layer_W),
        }


def reconstruct_model_fp_blockdiag_scaled_forward(
    model,
    data_loader,
    block_size,
    e_bits,
    m_bits,
    e_bits_scale,
    m_bits_scale,
    device,
    use_forward=True,
    top_k=2
):
    """
    Multi-layer FP4 reconstruction for GPT-style models.
    Forward-pass optimized with Hessian + candidate search.

    Args:
        model: nn.Module to quantize
        data_loader: calibration dataloader for forward-pass loss
        block_size: block size per weight block
        e_bits, m_bits: FP4 exponent/mantissa bits
        e_bits_scale, m_bits_scale: alpha quantization bits
        device: torch.device
        use_forward: compute forward-pass loss
        top_k: number of candidates for Hessian selection

    Returns:
        dict[layer_name] = (alpha_out, e, m, sign, bias_out)
    """
    model.eval()
    model.to(device)
    layer_outputs = {}

    # Precompute activations for all layers if using forward-pass
    cached_inputs = {}
    if use_forward:
        with torch.no_grad():
            for batch in data_loader:
                x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                out = x
                for name, module in model.named_modules():
                    if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                        cached_inputs[name] = out.detach()
                    out = module(out)

    # Process each layer
    for name, layer in model.named_modules():
        if not isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):
            continue

        # Assume Hessian blocks are precomputed per layer
        H_blocks_layer = layer.H_blocks if hasattr(layer, 'H_blocks') else [torch.eye(layer.weight.numel(), device=device)]

        cached_input = cached_inputs[name] if use_forward else None

        alpha_out, e, m, sign, bias_out = reconstruct_layer_fp_blockdiag_scaled_v4_forward(
            layer=layer,
            H_blocks_layer=H_blocks_layer,
            block_size=block_size,
            e_bits=e_bits,
            m_bits=m_bits,
            e_bits_scale=e_bits_scale,
            m_bits_scale=m_bits_scale,
            device=device,
            cached_input=cached_input,
            use_forward=use_forward,
            top_k=top_k
        )

        layer_outputs[name] = (alpha_out, e, m, sign, bias_out)

    return layer_outputs

# =========================================================
# 🔹 QUANTIZED LINEAR
# =========================================================


def _to_per_block(tensor: torch.Tensor, block_size: int, M: int) -> torch.Tensor:
    """Take the first element of each block column: [N, M] → [N, n_blocks]."""
    n_blocks = math.ceil(M / block_size)
    return tensor[:, ::block_size][:, :n_blocks]

def identify_outlier_channels(act_channel_max, threshold_ratio=6.0):
    """
    Identify input channels whose max activation is more than
    threshold_ratio times the median channel max.
    These are the channels that destroy FP4 block quantization.
    
    threshold_ratio=6.0 means channels with max > 6x median are outliers.
    This is consistent with LLM.int8() findings for OPT.
    
    Returns dict: layer_name -> outlier_indices (LongTensor)
    """
    outlier_indices = {}
    for name, ch_max in act_channel_max.items():
        median_max = ch_max.median()
        outlier_mask = ch_max > (threshold_ratio * median_max)
        outlier_indices[name] = outlier_mask.nonzero(as_tuple=True)[0]
        n_out = outlier_mask.sum().item()
        pct   = 100 * n_out / ch_max.shape[0]
        print(f"  {name}: {n_out}/{ch_max.shape[0]} outlier channels "
              f"({pct:.1f}%)")
    return outlier_indices

class QuantLinearFP_Decomposed(nn.Module):
    def __init__(self, linear, block_size, e_bits, m_bits,
                 e_bits_scale, m_bits_scale, outlier_indices):
        super().__init__()
        self.linear       = linear
        self.block_size   = block_size
        self.e_bits       = e_bits
        self.m_bits       = m_bits
        self.e_bits_scale = e_bits_scale
        self.m_bits_scale = m_bits_scale

        out_features = linear.weight.shape[0]
        in_features  = linear.weight.shape[1]
        self.smooth_scale = None
        all_indices  = torch.arange(in_features)
        outlier_mask = torch.zeros(in_features, dtype=torch.bool)
        if outlier_indices is not None and outlier_indices.numel() > 0:
            outlier_mask[outlier_indices] = True

        self.register_buffer(
            "outlier_indices",
            outlier_indices if outlier_indices is not None
            else torch.zeros(0, dtype=torch.long)
        )
        self.register_buffer("normal_indices",  all_indices[~outlier_mask])
        self.register_buffer("outlier_mask",    outlier_mask)  # (in_features,) bool

        # FP16 weights for outlier channels: (out_features, n_outlier)
        if self.outlier_indices.numel() > 0:
            self.register_buffer(
                "weight_fp16_outlier",
                linear.weight[:, self.outlier_indices].clone().half()
            )
        else:
            self.register_buffer("weight_fp16_outlier", None)

        self.register_buffer("weight_q", None)

        self.act_quant_mode = None
        self.act_block_size = block_size
        self.act_rho        = None

    def forward(self, x):
        out_features = self.linear.weight.shape[0]
        n_outlier    = self.outlier_indices.shape[0]
        n_normal     = self.normal_indices.shape[0]

        # ── FP16 outlier path ─────────────────────────────────────────────
        # Extract outlier activation channels, multiply by FP16 weights
        # These channels are handled exactly in FP16, no quantization error
        if n_outlier > 0 and self.weight_fp16_outlier is not None:
            x_outlier   = x[..., self.outlier_indices]        # (..., n_outlier)
            out_outlier = F.linear(
                x_outlier.to(self.weight_fp16_outlier.dtype),
                self.weight_fp16_outlier                       # (out_features, n_outlier)
            ).to(x.dtype)
        else:
            out_outlier = x.new_zeros(*x.shape[:-1], out_features)

        # ── FP4 normal path ───────────────────────────────────────────────
        # Slice out only normal (non-outlier) activation channels
        # The corresponding weight columns are in weight_q
        # Outlier channels are excluded — zero contribution from them here
        if n_normal > 0:
            x_normal = x[..., self.normal_indices]             # (..., n_normal)

            if self.weight_q is not None:
                if self.act_quant_mode is not None:
                    orig     = x_normal.shape
                    x_normal = self._quantize_input(
                        x_normal.reshape(-1, orig[-1])
                    ).reshape(orig)
                out_normal = F.linear(x_normal, self.weight_q) # (..., out_features)
            else:
                # Fallback — use original weights for normal channels only
                out_normal = F.linear(
                    x_normal,
                    self.linear.weight[:, self.normal_indices]
                )
        else:
            out_normal = x.new_zeros(*x.shape[:-1], out_features)

        # ── Aggregate ─────────────────────────────────────────────────────
        # FP4 accumulation is done in FP16 internally by hardware
        # We sum the two FP16 partial results here
        out = out_normal + out_outlier

        if self.linear.bias is not None:
            out = out + self.linear.bias

        return out

    def _quantize_input(self, x):
        bs = self.act_block_size or self.block_size
        if self.act_quant_mode == "preshifted" and self.act_rho is not None:
            return _quantize_input_preshifted(
                x, self.act_rho, bs, self.e_bits, self.m_bits
            )
        elif self.act_quant_mode == "nvfp4":
            return quantize_activations(
                x, bs, self.e_bits, self.m_bits,
                e_bits_scale=4, m_bits_scale=3
            )
        elif self.act_quant_mode == "mxfp4":
            return quantize_activations(
                x, bs, self.e_bits, self.m_bits,
                e_bits_scale=8, m_bits_scale=0
            )
        return x



class QuantLinearFP(nn.Module):
    def __init__(self, linear, block_size, e_bits, m_bits, e_bits_scale, m_bits_scale):
        super().__init__()
        self.linear = linear
        self.block_size = block_size
        self.e_bits = e_bits
        self.m_bits = m_bits
        self.e_bits_scale = e_bits_scale
        self.m_bits_scale = m_bits_scale
        self.register_buffer("weight_q", None) # ensure same dtype for quantized weights
        self.act_quant_mode = None  # placeholder for potential future use
        self.act_block_size = None  # placeholder for potential future use
        self.smooth_scale = None     # placeholder for potential future use
        self.act_rho = None  # placeholder for potential future use

    def calibrate(self, data_loader, device):
        alpha, e, m, sign = reconstruct_layer_fp_baseline(
            self.linear,
            self.block_size, self.e_bits, self.m_bits,
            self.e_bits_scale, self.m_bits_scale, device)
        bias = 2 ** (self.e_bits - 1) - 1
        base = alpha * (2.0 ** (e.float() - bias))
        if self.m_bits > 0:
            W_hat = base * (1.0 + m.float() / (2 ** self.m_bits))
        else:
            W_hat = base
        self.weight_q = sign * W_hat

    def calibrate_Hessian(self, data_loader, device, H_diag):
        alpha, e, m, sign = reconstruct_layer_fp_Hessian(
            self.linear,
            H_diag,
            self.block_size,
            self.e_bits,
            self.m_bits,
            self.e_bits_scale,
            self.m_bits_scale,
            device=device
        )
        bias = 2 ** (self.e_bits - 1) - 1
        base = alpha * (2.0 ** (e.float() - bias))
        fine = base * m.float() / (2 ** self.m_bits) if self.m_bits > 0 else 0.0
        self.weight_q = (base + fine) * sign
    def calibrate_Hessian_block(self, data_loader, device, H_block):
        alpha, e, m ,sign = reconstruct_layer_fp_blockdiag(
            self.linear,
            H_block,
            self.block_size,
            self.e_bits,
            self.m_bits,
            self.e_bits_scale,
            self.m_bits_scale,
            device = device
        )
        bias = 2 ** (self.e_bits-1)-1
        base = alpha * (2**(e.float()-bias))
        fine = base * m.float() / (2 ** self.m_bits) if self.m_bits > 0 else 0.0
        self.weight_q = (base + fine) * sign
    def calibrate_Hessian_whitened(self, data_loader, device, H_block):
        alpha, e, m, sign = reconstruct_layer_fp_blockdiag_whitened(
            self.linear,
            H_block,
            self.block_size,
            self.e_bits,
            self.m_bits,
            self.e_bits_scale,
            self.m_bits_scale,
            device=device
        )

        bias = 2 ** (self.e_bits - 1) - 1
        base = alpha * (2.0 ** (e.float() - bias))
        fine = base * m.float() / (2 ** self.m_bits) if self.m_bits > 0 else 0.0
        self.weight_q = (base + fine) * sign
    # def calibrate_Hessian_scaled(self, data_loader, device, H_block):
    #     weight_q = reconstruct_layer_fp_blockdiag_scaled_v5(
    #         self.linear,          # or self.conv / self.conv1d
    #         H_block,
    #         self.block_size,
    #         self.e_bits,
    #         self.m_bits,
    #         self.e_bits_scale,
    #         self.m_bits_scale,
    #         device=device)
    #     self.weight_q = weight_q.view_as(self.linear.weight)  # or .conv.weight / .conv1d.weight
    #     del weight_q
    #     torch.cuda.empty_cache()
    def calibrate_Hessian_scaled(self, data_loader, device, H_block):
        """
        Adaptive mesh (v5 scaled) calibration.
        reconstruct_layer_fp_blockdiag_scaled_v5 now returns a dict.
        """
        res = reconstruct_layer_fp_blockdiag_scaled_v5(
            self.linear,
            H_block,
            self.block_size,
            self.e_bits,
            self.m_bits,
            self.e_bits_scale,
            self.m_bits_scale,
            device=device,
        )
        self.weight_q = res["weight_q"].view_as(self.linear.weight)
        self.alpha_q  = res["alpha"]   # [N, n_blocks] CPU float32
        self.bias_q   = res["bias"]    # [N, n_blocks] CPU long
 
        del res
        torch.cuda.empty_cache()
    def calibrate_Hessian_Hadamard(self, data_loader, device, H_block):
            # We call the V5 Final version which returns a dict
            # res = reconstruct_layer_hadamard_v10(
            #     self.linear,
            #     H_block,
            #     self.block_size,
            #     self.e_bits,
            #     self.m_bits,
            #     self.e_bits_scale,
            #     self.m_bits_scale,
            #     device=device
            # )
            res = reconstruct_layer_hadamard_v10_fast(
                self.linear,
                H_block,
                self.block_size,
                self.e_bits,
                self.m_bits,
                self.e_bits_scale,
                self.m_bits_scale,
                device=device
            )
            # The V5 function already performs the inverse transform and 
            # sign correction. We simply store the spatial weight for simulation.
            self.weight_q = res['reconstructed_weight'].view_as(self.linear.weight)
    def _quantize_input(self, x):
        bs = self.act_block_size or self.block_size
        if self.act_quant_mode == "preshifted" and self.act_rho is not None:
            return _quantize_input_preshifted(
                x,
                self.act_rho,
                self.act_b_ori,
                bs,
                self.e_bits,
                self.m_bits
            )
        elif self.act_quant_mode == "preshifted_beta_only":
            # Beta scaling only, no quantization — isolates weight quality
            if self.act_b_ori is not None:
                return _apply_beta_scaling_only(x, self.act_b_ori, bs)
            return x
        elif self.act_quant_mode == "nvfp4":
            return quantize_activations_fast(
                x, bs, self.e_bits, self.m_bits,
                e_bits_scale=self.e_bits_scale, m_bits_scale=self.m_bits_scale
            )
        elif self.act_quant_mode == "mxfp4":
            return quantize_activations_fast(
                x, bs, self.e_bits, self.m_bits,
                e_bits_scale=self.e_bits_scale, m_bits_scale=self.m_bits_scale
            )
        return x

    def forward(self, x):
        original_dtype = x.dtype
        W = self.weight_q if self.weight_q is not None else self.linear.weight
        b = self.linear.bias if self.linear.bias is not None else None
        if self.weight_q is not None:
            if self.act_quant_mode is not None:
                x = self._quantize_input(x)
            W = W.to(x.dtype)
            if b is not None:
                b = b.to(x.dtype)
        # else:
        #     print("Warning: weight_q is None, using original weights without quantization.")
        # print(x.dtype, W.dtype, b.dtype if b is not None else None)
        out = F.linear(x, W, b)
        return out.to(original_dtype)

# =========================================================
# 🔹 QUANTIZED CONV
# =========================================================
class QuantConv2dFP(nn.Module):
    def __init__(self, conv, block_size, e_bits, m_bits, e_bits_scale, m_bits_scale):
        super().__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.conv = conv
        self.block_size = block_size
        self.e_bits = e_bits
        self.m_bits = m_bits
        self.e_bits_scale = e_bits_scale
        self.m_bits_scale = m_bits_scale
        self.register_buffer("weight_q", None)
        self.act_quant_mode = None  # placeholder for potential future use
        self.act_block_size = None  # placeholder for potential future use

    def calibrate(self, data_loader, device):
        alpha, e, m, sign = reconstruct_layer_fp_baseline(
            self.conv,
            self.block_size, self.e_bits, self.m_bits,
            self.e_bits_scale, self.m_bits_scale, device)
        bias = 2 ** (self.e_bits - 1) - 1
        base = alpha * (2.0 ** (e.float() - bias))
        if self.m_bits > 0:
            W_hat = base * (1.0 + m.float() / (2 ** self.m_bits))
        else:
            W_hat = base
        self.weight_q = sign * W_hat.view_as(self.conv.weight)

    def calibrate_Hessian(self, data_loader, device, H_diag):
        alpha, e, m, sign = reconstruct_layer_fp_Hessian(
            self.conv,
            H_diag,
            self.block_size,
            self.e_bits,
            self.m_bits,
            self.e_bits_scale,
            self.m_bits_scale,
            device=device
        )
        bias = 2 ** (self.e_bits - 1) - 1
        base = alpha * (2.0 ** (e.float() - bias))
        fine = base * m.float() / (2 ** self.m_bits) if self.m_bits > 0 else 0.0
        self.weight_q = (base + fine).view_as(self.conv.weight) * sign.view_as(self.conv.weight)
    def calibrate_Hessian_block(self, data_loader, device, H_block):
        alpha, e, m, sign = reconstruct_layer_fp_blockdiag(
            self.conv,
            H_block,
            self.block_size,
            self.e_bits,
            self.m_bits,
            self.e_bits_scale,
            self.m_bits_scale,
            device=device
        )
        bias = 2 ** (self.e_bits - 1) - 1
        base = alpha * (2.0 ** (e.float() - bias))
        fine = base * m.float() / (2 ** self.m_bits) if self.m_bits > 0 else 0.0
        self.weight_q = (base + fine).view_as(self.conv.weight) * sign.view_as(self.conv.weight)
    def calibrate_Hessian_whitened(self, data_loader, device, H_block):
        alpha, e, m, sign = reconstruct_layer_fp_blockdiag_whitened(
            self.conv,
            H_block,
            self.block_size,
            self.e_bits,
            self.m_bits,
            self.e_bits_scale,
            self.m_bits_scale,
            device=device
        )

        bias = 2 ** (self.e_bits - 1) - 1
        base = alpha * (2.0 ** (e.float() - bias))
        fine = base * m.float() / (2 ** self.m_bits) if self.m_bits > 0 else 0.0
        self.weight_q = (base + fine) * sign
    # def calibrate_Hessian_scaled(self, data_loader, device, H_block):
    #     weight_q = reconstruct_layer_fp_blockdiag_scaled_v5(
    #         self.conv,          # or self.conv / self.conv1d
    #         H_block,
    #         self.block_size,
    #         self.e_bits,
    #         self.m_bits,
    #         self.e_bits_scale,
    #         self.m_bits_scale,
    #         device=device)
    #     self.weight_q = weight_q.view_as(self.linear.weight)  # or .conv.weight / .conv1d.weight
    #     del weight_q
    #     torch.cuda.empty_cache()
    def calibrate_Hessian_scaled(self, data_loader, device, H_block):
        """
        Adaptive mesh (v5 scaled) calibration.
        reconstruct_layer_fp_blockdiag_scaled_v5 now returns a dict.
        """
        res = reconstruct_layer_fp_blockdiag_scaled_v5(
            self.conv,
            H_block,
            self.block_size,
            self.e_bits,
            self.m_bits,
            self.e_bits_scale,
            self.m_bits_scale,
            device=device,
        )
        self.weight_q = res["weight_q"].view_as(self.linear.weight)
        self.alpha_q  = res["alpha"]   # [N, n_blocks] CPU float32
        self.bias_q   = res["bias"]    # [N, n_blocks] CPU long
 
        del res
        torch.cuda.empty_cache()
    def calibrate_Hessian_Hadamard(self, data_loader, device, H_block):
            # res = reconstruct_layer_hadamard_v10(
            #     self.conv,
            #     H_block,
            #     self.block_size,
            #     self.e_bits,
            #     self.m_bits,
            #     self.e_bits_scale,
            #     self.m_bits_scale,
            #     device=device
            # )
            res = reconstruct_layer_hadamard_v10_fast(
                self.conv,
                H_block,
                self.block_size,
                self.e_bits,
                self.m_bits,
                self.e_bits_scale,
                self.m_bits_scale,
                device=device
            )
            # Ensure it is viewed as the original weight shape (C_out, C_in, K, K)
            self.weight_q = res['reconstructed_weight'].view_as(self.conv.weight)
    def _quantize_input(self, x):
        bs = self.act_block_size or self.block_size
        return quantize_activations(
            x, bs, self.e_bits, self.m_bits,
            self.e_bits_scale, self.m_bits_scale
        )
    def forward(self, x):
        if self.weight_q is not None and self.act_quant_mode is not None:
            x= self._quantize_input(x)
        return F.conv2d(x,
                        self.weight_q if self.weight_q is not None else self.conv.weight,
                        self.conv.bias,
                        stride=self.conv.stride,
                        padding=self.conv.padding)

from transformers.pytorch_utils import Conv1D  # GPT-2's Conv1D
 
 
class QuantConv1dFP(nn.Module):
    """
    Quantized wrapper for GPT-2's transformers.pytorch_utils.Conv1D.
 
    IMPORTANT: GPT-2's Conv1D is NOT nn.Conv1d. It is a linear projection
    whose weight is stored as (in_features, out_features) — the TRANSPOSE
    of nn.Linear's (out_features, in_features). The forward pass does:
        x @ weight + bias
    rather than F.linear(x, weight.T, bias).
 
    All calibration methods must account for this by transposing the weight
    before passing to your reconstruct_layer_* functions (which expect the
    standard (out, in) layout), then transposing back before storing weight_q.
    """
 
    def __init__(self, conv1d, block_size, e_bits, m_bits, e_bits_scale, m_bits_scale):
        super().__init__()
 
        # Store dimensions — Conv1D.weight is (in_features, out_features)
        self.in_features  = conv1d.weight.shape[0]
        self.out_features = conv1d.weight.shape[1]
        self.conv1d       = conv1d
        self.block_size   = block_size
        self.e_bits       = e_bits
        self.m_bits       = m_bits
        self.e_bits_scale = e_bits_scale
        self.m_bits_scale = m_bits_scale
 
        self.register_buffer("weight_q", None)
        self.act_quant_mode = None
        self.act_block_size = None
 
    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #
 
    def _get_weight_standard_layout(self):
        """
        Return weight in standard (out_features, in_features) layout
        so it matches what your reconstruct_layer_* functions expect.
        """
        return self.conv1d.weight.T.contiguous()  # (out, in)
 
    def _store_weight_q(self, W_reconstructed):
        """
        W_reconstructed is in (out_features, in_features) layout.
        Transpose back to (in_features, out_features) for GPT-2's forward pass.
        """
        self.weight_q = W_reconstructed.T.contiguous()  # (in, out)
 
    def _reconstruct(self, alpha, e, m, sign, bias=None):
        """Shared FP reconstruction logic."""
        if bias is None:
            bias = 2 ** (self.e_bits - 1) - 1
        base = alpha * (2.0 ** (e.float() - bias))
        fine = base * m.float() / (2 ** self.m_bits) if self.m_bits > 0 else 0.0
        return (base + fine) * sign
 
    # ------------------------------------------------------------------ #
    # Calibration methods — mirror your QuantConv2dFP exactly,            #
    # but wrap weight in/out with the transpose helpers above              #
    # ------------------------------------------------------------------ #
 
    def calibrate(self, data_loader, device):
        # Temporarily swap weight to standard layout so reconstruct works
        original_weight      = self.conv1d.weight.data
        self.conv1d.weight.data = self._get_weight_standard_layout()
 
        alpha, e, m, sign = reconstruct_layer_fp_baseline(
            self.conv1d,
            self.block_size, self.e_bits, self.m_bits,
            self.e_bits_scale, self.m_bits_scale, device)
 
        self.conv1d.weight.data = original_weight  # restore
 
        W_reconstructed = self._reconstruct(alpha, e, m, sign)
        self._store_weight_q(W_reconstructed)
 
    def calibrate_Hessian(self, data_loader, device, H_diag):
        original_weight      = self.conv1d.weight.data
        self.conv1d.weight.data = self._get_weight_standard_layout()
 
        alpha, e, m, sign = reconstruct_layer_fp_Hessian(
            self.conv1d, H_diag,
            self.block_size, self.e_bits, self.m_bits,
            self.e_bits_scale, self.m_bits_scale, device=device)
 
        self.conv1d.weight.data = original_weight
 
        W_reconstructed = self._reconstruct(alpha, e, m, sign)
        self._store_weight_q(W_reconstructed)
 
    def calibrate_Hessian_block(self, data_loader, device, H_block):
        original_weight      = self.conv1d.weight.data
        self.conv1d.weight.data = self._get_weight_standard_layout()
 
        alpha, e, m, sign = reconstruct_layer_fp_blockdiag(
            self.conv1d, H_block,
            self.block_size, self.e_bits, self.m_bits,
            self.e_bits_scale, self.m_bits_scale, device=device)
 
        self.conv1d.weight.data = original_weight
 
        W_reconstructed = self._reconstruct(alpha, e, m, sign)
        self._store_weight_q(W_reconstructed)
 
    def calibrate_Hessian_whitened(self, data_loader, device, H_block):
        original_weight      = self.conv1d.weight.data
        self.conv1d.weight.data = self._get_weight_standard_layout()
 
        alpha, e, m, sign = reconstruct_layer_fp_blockdiag_whitened(
            self.conv1d, H_block,
            self.block_size, self.e_bits, self.m_bits,
            self.e_bits_scale, self.m_bits_scale, device=device)
 
        self.conv1d.weight.data = original_weight
 
        W_reconstructed = self._reconstruct(alpha, e, m, sign)
        self._store_weight_q(W_reconstructed)
 
    # def calibrate_Hessian_scaled(self, data_loader, device, H_block):
    #     weight_q = reconstruct_layer_fp_blockdiag_scaled_v5(
    #         self.linear,          # or self.conv / self.conv1d
    #         H_block,
    #         self.block_size,
    #         self.e_bits,
    #         self.m_bits,
    #         self.e_bits_scale,
    #         self.m_bits_scale,
    #         device=device)
    #     self.weight_q = weight_q.view_as(self.linear.weight)  # or .conv.weight / .conv1d.weight
    #     del weight_q
    #     torch.cuda.empty_cache()
    def calibrate_Hessian_scaled(self, data_loader, device, H_block):
        """
        Adaptive mesh (v5 scaled) calibration.
        reconstruct_layer_fp_blockdiag_scaled_v5 now returns a dict.
        """
        res = reconstruct_layer_fp_blockdiag_scaled_v5(
            self.conv1d,
            H_block,
            self.block_size,
            self.e_bits,
            self.m_bits,
            self.e_bits_scale,
            self.m_bits_scale,
            device=device,
        )
        self.weight_q = res["weight_q"].view_as(self.linear.weight)
        self.alpha_q  = res["alpha"]   # [N, n_blocks] CPU float32
        self.bias_q   = res["bias"]    # [N, n_blocks] CPU long
 
        del res
        torch.cuda.empty_cache()
    def calibrate_Hessian_Hadamard(self, data_loader, device, H_block):
        original_weight      = self.conv1d.weight.data
        self.conv1d.weight.data = self._get_weight_standard_layout()
 
        res = reconstruct_layer_hadamard_v10_fast(
            self.conv1d, H_block,
            self.block_size, self.e_bits, self.m_bits,
            self.e_bits_scale, self.m_bits_scale, device=device)
 
        self.conv1d.weight.data = original_weight
 
        # hadamard path returns a dict — transpose reconstructed weight back
        self._store_weight_q(res['reconstructed_weight'])
 
    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #
    def _quantize_input(self, x):
        bs = self.act_block_size or self.block_size
        return quantize_activations(
            x, bs, self.e_bits, self.m_bits,
            self.e_bits_scale, self.m_bits_scale
        )
    def forward(self, x):
        W = self.weight_q if self.weight_q is not None else self.conv1d.weight
        bias = self.conv1d.bias
        if self.weight_q is not None and self.act_quant_mode is not None:
            x = self._quantize_input(x)
        out = x @ W
        return out + bias if bias is not None else out
 
 

# =========================================================
# 🔹 REPLACE LAYERS
# =========================================================
# At the top of your file
CONV1D_CLASS_NAME = "Conv1D"
CONV1D_MODULE_PATH = "transformers"

def is_hf_conv1d(module):
    """Check by class name to avoid import path mismatches."""
    return (type(module).__name__ == "Conv1D" and
            "transformers" in type(module).__module__)
 
 
def is_tied_embedding(model, module):
    """
    Returns True if this module's weight is shared with any other
    parameter in the model (e.g. BLOOM's lm_head <-> word_embeddings).
    """
    if not hasattr(module, 'weight'):
        return False
    ptr = module.weight.data_ptr()
    count = sum(
        1 for n, p in model.named_parameters()
        if p.data_ptr() == ptr
    )
    # If the same storage appears more than once, it is tied
    return count > 1
 
 
def replace_layers(model, block_size, e_bits, m_bits, e_bits_scale, m_bits_scale,
                   root_model=None):
    """
    Recursively replace Linear, Conv2d, and HuggingFace Conv1D layers
    with FP4 quantized wrappers.
 
    Skips:
      - Already-replaced layers
      - Tied embedding layers (e.g. BLOOM lm_head shares weights with
        word_embeddings — quantizing it would corrupt input embeddings)
 
    Args:
        model:         the (sub)module to recurse into
        root_model:    the top-level model, used for tied-weight detection.
                       Pass None on the first call — it is set automatically.
        block_size, e_bits, m_bits, e_bits_scale, m_bits_scale: quant config
    """
    if root_model is None:
        root_model = model
 
    for name, module in list(model.named_children()):
 
        # ── Already quantized — skip ──────────────────────────────────────
        if isinstance(module, (QuantLinearFP, QuantConv2dFP, QuantConv1dFP)):
            continue
 
        # ── Tied embedding — skip to avoid corrupting input embeddings ────
        if isinstance(module, nn.Linear) and is_tied_embedding(root_model, module):
            print(f"  Skipping tied embedding: {name} "
                  f"({module.weight.shape})")
            continue
 
        # ── nn.Linear ─────────────────────────────────────────────────────
        if isinstance(module, nn.Linear):
            print(f"  Found Linear: {name}")
            setattr(model, name,
                    QuantLinearFP(module, block_size, e_bits, m_bits,
                                  e_bits_scale, m_bits_scale))
 
        # ── nn.Conv2d ─────────────────────────────────────────────────────
        elif isinstance(module, nn.Conv2d):
            print(f"  Found Conv2D: {name}")
            setattr(model, name,
                    QuantConv2dFP(module, block_size, e_bits, m_bits,
                                  e_bits_scale, m_bits_scale))
 
        # ── HuggingFace Conv1D (GPT-2 style) ─────────────────────────────
        elif is_hf_conv1d(module):
            print(f"  Found HF Conv1D: {name}")
            setattr(model, name,
                    QuantConv1dFP(module, block_size, e_bits, m_bits,
                                  e_bits_scale, m_bits_scale))
 
        # ── Recurse into submodules ───────────────────────────────────────
        else:
            replace_layers(module, block_size, e_bits, m_bits,
                           e_bits_scale, m_bits_scale,
                           root_model=root_model)
 
    return model
 

def replace_layers_decomposed(model, block_size, e_bits, m_bits,
                               e_bits_scale, m_bits_scale,
                               outlier_indices,
                               root_model=None):
    """
    Recursively replace Linear, Conv2d, and HuggingFace Conv1D layers
    with decomposed FP4 + FP16 sparse wrappers for Linear layers that
    have identified outlier channels, and standard FP4 wrappers otherwise.

    Skips:
      - Already-replaced layers
      - Tied embedding layers

    Args:
        model:           the (sub)module to recurse into
        block_size, e_bits, m_bits, e_bits_scale, m_bits_scale: quant config
        outlier_indices: dict mapping full layer name -> LongTensor of
                         outlier input channel indices. Layers not in this
                         dict get standard QuantLinearFP wrappers.
                         Pass the output of identify_outlier_channels().
        root_model:      top-level model for tied-weight detection.
                         Pass None on first call — set automatically.
    """
    if root_model is None:
        root_model = model

    # Build a flat name->module map of the full model for outlier lookup
    # since named_children() only gives local names, not full dotted paths
    if not hasattr(replace_layers_decomposed, '_full_name_map'):
        replace_layers_decomposed._full_name_map = {
            name: mod for name, mod in root_model.named_modules()
        }

    full_name_map = replace_layers_decomposed._full_name_map

    for name, module in list(model.named_children()):

        # ── Already replaced — skip ───────────────────────────────────────
        if isinstance(module, (QuantLinearFP, QuantConv2dFP,
                               QuantConv1dFP, QuantLinearFP_Decomposed)):
            continue

        # ── Tied embedding — skip ─────────────────────────────────────────
        if isinstance(module, nn.Linear) and is_tied_embedding(root_model, module):
            print(f"  Skipping tied embedding: {name} "
                  f"({module.weight.shape})")
            continue

        # ── nn.Linear ─────────────────────────────────────────────────────
        if isinstance(module, nn.Linear):

            # Find the full dotted name of this module in the root model
            full_name = None
            for fname, fmod in full_name_map.items():
                if fmod is module:
                    full_name = fname
                    break

            if full_name is not None and full_name in outlier_indices:
                out_idx = outlier_indices[full_name]
                n_out   = out_idx.shape[0]
                n_total = module.weight.shape[1]
                print(f"  Found Linear (decomposed): {full_name} "
                      f"— {n_out}/{n_total} outlier channels "
                      f"({100*n_out/n_total:.1f}% FP16, "
                      f"{100*(n_total-n_out)/n_total:.1f}% FP4)")
                setattr(model, name,
                        QuantLinearFP_Decomposed(
                            module,
                            block_size, e_bits, m_bits,
                            e_bits_scale, m_bits_scale,
                            out_idx
                        ))
            else:
                print(f"  Found Linear (standard FP4): {full_name or name}")
                setattr(model, name,
                        QuantLinearFP(module, block_size, e_bits, m_bits,
                                      e_bits_scale, m_bits_scale))

        # ── nn.Conv2d — no decomposition, standard wrapper ────────────────
        elif isinstance(module, nn.Conv2d):
            print(f"  Found Conv2D: {name}")
            setattr(model, name,
                    QuantConv2dFP(module, block_size, e_bits, m_bits,
                                  e_bits_scale, m_bits_scale))

        # ── HuggingFace Conv1D — no decomposition, standard wrapper ───────
        elif is_hf_conv1d(module):
            print(f"  Found HF Conv1D: {name}")
            setattr(model, name,
                    QuantConv1dFP(module, block_size, e_bits, m_bits,
                                  e_bits_scale, m_bits_scale))

        # ── Recurse into submodules ───────────────────────────────────────
        else:
            replace_layers_decomposed(
                module, block_size, e_bits, m_bits,
                e_bits_scale, m_bits_scale,
                outlier_indices,
                root_model=root_model
            )

    # Clean up the cached name map after the top-level call completes
    if root_model is model and hasattr(replace_layers_decomposed, '_full_name_map'):
        del replace_layers_decomposed._full_name_map

    return model
 
def replace_layers_flat(model, block_size, e_bits, m_bits,
                        e_bits_scale, m_bits_scale):
    """
    Alternative flat version using named_modules() + parent traversal.
    Use this if replace_layers() (recursive) misses deeply nested layers.
    Produces identical results.
    """
    for name, module in list(model.named_modules()):
 
        if isinstance(module, (QuantLinearFP, QuantConv2dFP, QuantConv1dFP)):
            continue
 
        # Determine replacement type
        if isinstance(module, nn.Linear):
            if is_tied_embedding(model, module):
                print(f"  Skipping tied embedding: {name} ({module.weight.shape})")
                continue
            replacement = QuantLinearFP(module, block_size, e_bits, m_bits,
                                        e_bits_scale, m_bits_scale)
            label = "Linear"
        elif isinstance(module, nn.Conv2d):
            replacement = QuantConv2dFP(module, block_size, e_bits, m_bits,
                                        e_bits_scale, m_bits_scale)
            label = "Conv2D"
        elif is_hf_conv1d(module):
            replacement = QuantConv1dFP(module, block_size, e_bits, m_bits,
                                        e_bits_scale, m_bits_scale)
            label = "HF Conv1D"
        else:
            continue
 
        # Find parent and set attribute
        parts = name.split(".")
        if not parts:
            continue
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr = parts[-1]
 
        print(f"  Found {label}: {name}")
        setattr(parent, attr, replacement)
 
    return model
# =========================================================
# 🔹 CALIBRATION DRIVER
# =========================================================
def calibrate_model(model, data_loader, device="cuda"):
    model.eval()
    model.to(device)
    for module in model.modules():
        if hasattr(module, "calibrate"):
            module.calibrate(data_loader, device)
    return model


def calibrate_model_Hessian_scaled(model, data_loader, block_size, device):
    model.eval().to(device)

    H_dict_block = compute_hessian_blockdiag_model(
        model, data_loader, device, block_size
    )

    for name, module in model.named_modules():  # use named_modules to get name
        if isinstance(module, (QuantLinearFP, QuantConv2dFP, QuantConv1dFP)):
            if name not in H_dict_block:
                print(f"⚠️ Missing Hessian for {name}, skipping")
                continue
            H_block = H_dict_block[name]
            print(f"Calibrating {name} with Adap FP4")
            module.calibrate_Hessian_scaled(data_loader, device, H_block)
            torch.cuda.empty_cache()

    return model

# def calibrate_model_preshifted(model, calib_loader, block_size, device,
#                                 e_bits=2, m_bits=1,
#                                 e_bits_scale=4, m_bits_scale=3,
#                                 num_batches=4):
#     """
#     Full pipeline combining pre-shifted exponent bias with
#     Hessian-weighted blockwise weight reconstruction.
#     """
#     model.eval().to(device)

#     # Step 1: collect per-channel activation max
#     act_channel_max = {}
#     hooks = []

#     def make_hook(name, in_features):
#         def hook(module, inp, out):
#             x = inp[0].detach().float()
#             x_flat = x.reshape(-1, in_features)
#             ch_max = x_flat.abs().amax(dim=0)
#             if name not in act_channel_max:
#                 act_channel_max[name] = ch_max.cpu()
#             else:
#                 act_channel_max[name] = torch.maximum(
#                     act_channel_max[name], ch_max.cpu()
#                 )
#         return hook

#     for name, module in model.named_modules():
#         if type(module).__name__ == "QuantLinearFP":
#             in_f = module.linear.weight.shape[1]
#             hooks.append(
#                 module.register_forward_hook(make_hook(name, in_f))
#             )

#     batches_run = 0
#     with torch.no_grad():
#         for batch in calib_loader:
#             if batch is None:
#                 continue
#             x = batch[0] if isinstance(batch, (list, tuple)) else batch
#             if x is None:
#                 continue
#             model(x.to(device))
#             batches_run += 1
#             if batches_run >= num_batches:
#                 break

#     for h in hooks:
#         h.remove()
#     print(f"Collected activation stats for {len(act_channel_max)} layers "
#           f"over {batches_run} batches")

#     # Step 2: compute Hessian using quantized activations
#     # Use rho-scaled activations for a better Hessian estimate
#     H_dict = compute_hessian_blockdiag_model_joint(
#         model, calib_loader, device, block_size,
#         e_bits=e_bits, m_bits=m_bits,
#         e_bits_scale=e_bits_scale, m_bits_scale=m_bits_scale,
#     )

#     # Step 3: reconstruct each layer with preshifted bias
#     for name, module in model.named_modules():
#         if type(module).__name__ != "QuantLinearFP":
#             continue
#         if name not in H_dict:
#             print(f"  Missing Hessian for {name}, skipping")
#             continue

#         ch_max = act_channel_max.get(name, None)
#         print(f"Calibrating {name}")

#         res = reconstruct_layer_fp_blockdiag_scaled_v5_preshifted(
#             module.linear,
#             H_dict[name],
#             block_size,
#             e_bits, m_bits,
#             e_bits_scale, m_bits_scale,
#             device,
#             act_channel_max=ch_max,
#         )

#         module.weight_q       = res["weight_q"].view_as(module.linear.weight)
#         module.alpha_q        = res["alpha"]
#         module.bias_q         = res["bias"]
#         module.act_rho        = res["rho"]
#         module.act_b_ori      = res["b_ori"]
#         module.act_quant_mode = "preshifted"
#         module.act_block_size = block_size

#         torch.cuda.empty_cache()

#     return model


# def calibrate_model_preshifted(model, calib_loader, block_size, device,
#                                 e_bits=2, m_bits=1,
#                                 e_bits_scale=4, m_bits_scale=3,
#                                 num_batches=4):
#     """
#     Full block-wise pre-shifted exponent bias pipeline:
#     1. Collect per-input-block activation max from calibration
#     2. Compute block-wise pre-shifted bias, absorb into weights
#     3. Run Hessian-weighted v5 reconstruction on adjusted weights
#        with bias search centered on rho
#     4. Store rho for per-tensor activation quantization at inference
#     """
#     model.eval().to(device)

#     # Step 1: collect per-block activation max
#     print("Collecting per-block activation statistics...")
#     act_block_max = collect_per_block_activation_max(
#         model, calib_loader, device, block_size, num_batches
#     )

#     # Step 2: compute Hessian using joint objective
#     print("Computing joint Hessian...")
#     H_dict = compute_hessian_blockdiag_model_joint(
#         model, calib_loader, device, block_size,
#         e_bits=e_bits, m_bits=m_bits,
#         e_bits_scale=e_bits_scale, m_bits_scale=m_bits_scale,
#     )

#     # Step 3: reconstruct each layer
#     for name, module in model.named_modules():
#         if type(module).__name__ != "QuantLinearFP":
#             continue
#         if name not in H_dict:
#             print(f"  Missing Hessian for {name}, skipping")
#             continue

#         print(f"Calibrating {name}")

#         W      = module.linear.weight.data.to(device).float()
#         blk_max = act_block_max.get(name, None)

#         if blk_max is not None:
#             W_adjusted, rho, b_ori, preshifted_center = \
#                 preshifted_bias_and_adjust_weights_blockwise(
#                     W, blk_max, block_size, e_bits, m_bits, device
#                 )
#         else:
#             print(f"  No activation stats for {name}, using standard v5")
#             W_adjusted        = W
#             rho               = None
#             b_ori             = None
#             preshifted_center = 2**(e_bits - 1) - 1

#         # Temporarily swap in adjusted weights
#         original_weight           = module.linear.weight.data.clone()
#         module.linear.weight.data = W_adjusted

#         # Run v5 with bias search centered on preshifted_center
#         res = reconstruct_layer_fp_blockdiag_scaled_v5_preshifted(
#             module.linear,
#             H_dict[name],
#             block_size,
#             e_bits, m_bits,
#             e_bits_scale, m_bits_scale,
#             device,
#             preshifted_center=preshifted_center,
#         )

#         # Restore original weights
#         module.linear.weight.data = original_weight

#         # Store results
#         module.weight_q       = res["weight_q"].view_as(module.linear.weight)
#         module.alpha_q        = res["alpha"]
#         module.bias_q         = res["bias"]
#         module.act_rho        = rho.item() if rho is not None else None
#         module.act_b_ori      = b_ori.cpu() if b_ori is not None else None
#         module.act_quant_mode = "preshifted"
#         module.act_block_size = block_size

#         torch.cuda.empty_cache()

#     return model


# def calibrate_model_preshifted(model, calib_loader, block_size, device,
#                                 e_bits=2, m_bits=1,
#                                 e_bits_scale=4, m_bits_scale=3,
#                                 num_batches=4):
#     model.eval().to(device)

#     # Step 1: collect per-block activation max
#     print("Collecting per-block activation statistics...")
#     act_block_max = collect_per_block_activation_max(
#         model, calib_loader, device, block_size, num_batches
#     )

#     # Step 2: compute Hessian
#     print("Computing joint Hessian...")
#     H_dict = compute_hessian_blockdiag_model_joint(
#         model, calib_loader, device, block_size,
#         e_bits=e_bits, m_bits=m_bits,
#         e_bits_scale=e_bits_scale, m_bits_scale=m_bits_scale,
#     )

#     # Step 3: reconstruct each layer
#     for name, module in model.named_modules():
#         if type(module).__name__ != "QuantLinearFP":
#             continue
#         if name not in H_dict:
#             print(f"  Missing Hessian for {name}, skipping")
#             continue

#         print(f"Calibrating {name}")

#         W       = module.linear.weight.data.to(device).float()
#         blk_max = act_block_max.get(name, None)

# # In calibrate_model_preshifted, replace the reconstruction block:

#         if blk_max is not None:
#             _, rho, b_ori, preshifted_center = \
#                 preshifted_bias_and_adjust_weights_blockwise(
#                     W, blk_max, block_size, e_bits, m_bits, device
#                 )
#         else:
#             rho               = None
#             b_ori             = None
#             preshifted_center = 2**(e_bits - 1) - 1

#         # Run v5 on the ORIGINAL weights — no beta adjustment
#         # The preshifted_center just tells v5 which bias region to search
#         res = reconstruct_layer_fp_blockdiag_scaled_v5_preshifted(
#             module.linear,          # original weights, NOT W_adjusted
#             H_dict[name],
#             block_size,
#             e_bits, m_bits,
#             e_bits_scale, m_bits_scale,
#             device,
#             preshifted_center=preshifted_center,
#         )

#         module.weight_q       = res["weight_q"].view_as(module.linear.weight)
#         module.alpha_q        = res["alpha"]
#         module.bias_q         = res["bias"]
#         module.act_rho        = rho.item() if rho is not None else None
#         module.act_b_ori      = b_ori.cpu() if b_ori is not None else None
#         module.act_quant_mode = "preshifted"
#         module.act_block_size = block_size

#     return model

# def compute_hessian_blockdiag_model_joint_preshifted(
#     model, calib_loader, device, block_size,
#     beta_dict,
#     e_bits=2, m_bits=1,
#     e_bits_scale=4, m_bits_scale=3,
#     num_batches=4,
# ):
#     H_dict  = {}   # name -> list of (block_size, block_size) tensors
#     hooks   = []
#     counts  = {}

#     for name, module in model.named_modules():
#         if type(module).__name__ != "QuantLinearFP":
#             continue

#         beta         = beta_dict.get(name, None)
#         in_features  = module.linear.weight.shape[1]
#         n_blocks     = math.ceil(in_features / block_size)

#         H_dict[name]  = None   # will become list of blocks on first fire
#         counts[name]  = 0

#         def make_hook(n, b, n_blk, bs, in_feat):
#             def hook(mod, inp, out):
#                 x = inp[0].detach().float()
#                 if x.dim() < 2:
#                     return
#                 x = x.reshape(-1, x.shape[-1])  # (N, in_features)
#                 if x.shape[0] == 0:
#                     return

#                 # Apply beta scaling
#                 if b is not None:
#                     x = x * b.to(x.device).unsqueeze(0)

#                 # Accumulate per-block H = X_k^T X_k
#                 if H_dict[n] is None:
#                     H_dict[n] = [
#                         torch.zeros(
#                             min(bs, in_feat - k * bs),
#                             min(bs, in_feat - k * bs),
#                             device=x.device
#                         )
#                         for k in range(n_blk)
#                     ]

#                 for k in range(n_blk):
#                     start = k * bs
#                     end   = min(start + bs, in_feat)
#                     x_blk = x[:, start:end]              # (N, block_size)
#                     H_dict[n][k] += x_blk.T @ x_blk

#                 counts[n] += x.shape[0]
#             return hook

#         h = module.register_forward_hook(
#             make_hook(name, beta, n_blocks, block_size, in_features)
#         )
#         hooks.append(h)

#     print(f"  Registered {len(hooks)} hooks for beta-scaled Hessian collection")
#     model.eval()
#     with torch.no_grad():
#         for i, batch in enumerate(calib_loader):
#             if i >= num_batches:
#                 break
#             input_ids = batch["input_ids"].to(device) \
#                 if isinstance(batch, dict) else batch[0].to(device)
#             model(input_ids)

#     for h in hooks:
#         h.remove()

#     # Validate and normalize
#     for name in list(H_dict.keys()):
#         if H_dict[name] is None:
#             print(f"  WARNING: No Hessian collected for {name}, removing")
#             del H_dict[name]
#         else:
#             H_dict[name] = [blk / counts[name] for blk in H_dict[name]]

#     print(f"  Collected beta-scaled Hessians for {len(H_dict)} layers")
#     return H_dict

# def calibrate_model_preshifted(model, calib_loader, block_size, device,
#                                 e_bits=2, m_bits=1,
#                                 e_bits_scale=4, m_bits_scale=3,
#                                 num_batches=4):
#     model.eval().to(device)

#     # Step 1: collect per-block activation max from raw activations
#     print("Collecting per-block activation statistics...")
#     act_block_max = collect_per_block_activation_max(
#         model, calib_loader, device, block_size, num_batches
#     )

#     # Step 2: compute per-layer beta from act_block_max
#     # We need beta BEFORE computing the Hessian so we can scale
#     # the Hessian collection hooks
#     beta_dict = {}
#     b_ori_dict = {}
#     rho_dict = {}
#     preshifted_center_dict = {}

#     for name, module in model.named_modules():
#         if type(module).__name__ != "QuantLinearFP":
#             continue
#         W       = module.linear.weight.data.to(device).float()
#         blk_max = act_block_max.get(name, None)
#         if blk_max is not None:
#             _, rho, b_ori, preshifted_center = \
#                 preshifted_bias_and_adjust_weights_blockwise(
#                     W, blk_max, block_size, e_bits, m_bits, device
#                 )
#             # Build per-channel beta vector (shape: in_features)
#             in_features = W.shape[1]
#             beta = torch.ones(in_features, device=device)
#             for k in range(len(b_ori)):
#                 start = k * block_size
#                 end   = min(start + block_size, in_features)
#                 beta[start:end] = 2.0 ** b_ori[k].float()

#             beta_dict[name]              = beta
#             b_ori_dict[name]             = b_ori
#             rho_dict[name]               = rho
#             preshifted_center_dict[name] = preshifted_center
#         else:
#             beta_dict[name]              = None
#             b_ori_dict[name]             = None
#             rho_dict[name]               = None
#             preshifted_center_dict[name] = 2**(e_bits - 1) - 1

#     # Step 3: compute Hessian with beta-scaled activations
#     # The Hessian hook must multiply each incoming activation by beta
#     # before accumulating H = X_scaled^T X_scaled
#     print("Computing Hessian with beta-scaled activations...")
#     H_dict = compute_hessian_blockdiag_model_joint_preshifted(
#         model, calib_loader, device, block_size,
#         beta_dict=beta_dict,
#         e_bits=e_bits, m_bits=m_bits,
#         e_bits_scale=e_bits_scale, m_bits_scale=m_bits_scale,
#     )

#     # Step 4: reconstruct each layer with W_adjusted and H_scaled
#     for name, module in model.named_modules():
#         if type(module).__name__ != "QuantLinearFP":
#             continue
#         if name not in H_dict:
#             print(f"  Missing Hessian for {name}, skipping")
#             continue

#         print(f"Calibrating {name}")

#         W                 = module.linear.weight.data.to(device).float()
#         beta              = beta_dict[name]
#         rho               = rho_dict[name]
#         b_ori             = b_ori_dict[name]
#         preshifted_center = preshifted_center_dict[name]

#         if beta is not None:
#             # Adjust weights by beta: W_adjusted[:, j] = W[:, j] * 2^(-b_ori_k)
#             # beta here is 2^(b_ori) per channel, so divide
#             W_adjusted = W * (1.0 / beta).unsqueeze(0)
#         else:
#             W_adjusted = W

#         # Temporarily swap in adjusted weights for v5
#         original_weight           = module.linear.weight.data.clone()
#         module.linear.weight.data = W_adjusted

#         res = reconstruct_layer_fp_blockdiag_scaled_v5_preshifted(
#             module.linear,
#             H_dict[name],       # Hessian computed from X_scaled
#             block_size,
#             e_bits, m_bits,
#             e_bits_scale, m_bits_scale,
#             device,
#             preshifted_center=preshifted_center,
#         )

#         module.linear.weight.data = original_weight

#         # weight_q is quantized W_adjusted — correct, do NOT undo beta
#         module.weight_q       = res["weight_q"].view_as(module.linear.weight)
#         module.alpha_q        = res["alpha"]
#         module.bias_q         = res["bias"]
#         module.act_rho        = rho.item() if rho is not None else None
#         module.act_b_ori      = b_ori.cpu() if b_ori is not None else None
#         module.act_quant_mode = "preshifted"
#         module.act_block_size = block_size

#         torch.cuda.empty_cache()

#     return model

def compute_hessian_blockdiag_model_joint_preshifted(
    model, calib_loader, device, block_size,
    beta_dict,
    num_batches=4,
):
    """
    Collect block-diagonal Hessian H = (X*beta)^T (X*beta) per block.
    beta = 2^(b_ori) per input channel — the activation scaling that
    matches the coordinate system of W_adjusted = W * 2^(-b_ori).
    """
    H_dict  = {}
    hooks   = []
    counts  = {}

    for name, module in model.named_modules():
        if type(module).__name__ != "QuantLinearFP":
            continue

        beta        = beta_dict.get(name, None)
        in_features = module.linear.weight.shape[1]
        n_blocks    = math.ceil(in_features / block_size)

        H_dict[name] = None
        counts[name] = 0

        def make_hook(n, b, n_blk, bs, in_feat):
            def hook(mod, inp, out):
                x = inp[0].detach().float()
                if x.dim() < 2:
                    return
                x = x.reshape(-1, x.shape[-1])  # (N, in_features)
                if x.shape[0] == 0:
                    return

                # Scale activations by beta = 2^(b_ori) per channel
                # This matches the coordinate system of W_adjusted
                if b is not None:
                    x = x * b.to(x.device).unsqueeze(0)

                # Accumulate per-block H = (X*beta)_k^T (X*beta)_k
                if H_dict[n] is None:
                    H_dict[n] = [
                        torch.zeros(
                            min(bs, in_feat - k * bs),
                            min(bs, in_feat - k * bs),
                            device=x.device
                        )
                        for k in range(n_blk)
                    ]

                for k in range(n_blk):
                    start = k * bs
                    end   = min(start + bs, in_feat)
                    x_blk = x[:, start:end]
                    H_dict[n][k] += x_blk.T @ x_blk

                counts[n] += x.shape[0]
            return hook

        h = module.register_forward_hook(
            make_hook(name, beta, n_blocks, block_size, in_features)
        )
        hooks.append(h)

    print(f"  Registered {len(hooks)} hooks for beta-scaled Hessian collection")
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(calib_loader):
            if i >= num_batches:
                break
            input_ids = batch["input_ids"].to(device) \
                if isinstance(batch, dict) else batch[0].to(device)
            model(input_ids)

    for h in hooks:
        h.remove()

    # Validate and normalize
    for name in list(H_dict.keys()):
        if H_dict[name] is None:
            print(f"  WARNING: No Hessian collected for {name}, removing")
            del H_dict[name]
        else:
            H_dict[name] = [blk / counts[name] for blk in H_dict[name]]

    print(f"  Collected beta-scaled Hessians for {len(H_dict)} layers")
    return H_dict


# def calibrate_model_preshifted(model, calib_loader, block_size, device,
#                                 e_bits=2, m_bits=1,
#                                 e_bits_scale=4, m_bits_scale=3,
#                                 num_batches=4):
#     model.eval().to(device)

#     # Step 1: collect per-block activation max from raw activations
#     print("Collecting per-block activation statistics...")
#     act_block_max = collect_per_block_activation_max(
#         model, calib_loader, device, block_size, num_batches
#     )

#     # Step 2: compute per-layer beta, rho, b_ori, W_adjusted
#     beta_dict              = {}
#     b_ori_dict             = {}
#     rho_dict               = {}
#     preshifted_center_dict = {}
#     W_adjusted_dict        = {}

#     for name, module in model.named_modules():
#         if type(module).__name__ != "QuantLinearFP":
#             continue

#         W       = module.linear.weight.data.to(device).float()
#         blk_max = act_block_max.get(name, None)

#         if blk_max is not None:
#             # W_adjusted already has 2^(-b_ori) applied per block internally
#             W_adjusted, rho, b_ori, preshifted_center = \
#                 preshifted_bias_and_adjust_weights_blockwise(
#                     W, blk_max, block_size, e_bits, m_bits, device
#                 )

#             # Beta for Hessian = 2^(+b_ori) — the activation scaling
#             # This is the INVERSE of what was applied to the weights
#             in_features = W.shape[1]
#             beta = torch.ones(in_features, device=device)
#             for k in range(len(b_ori)):
#                 start = k * block_size
#                 end   = min(start + block_size, in_features)
#                 beta[start:end] = 2.0 ** b_ori[k].float()

#             beta_dict[name]              = beta
#             b_ori_dict[name]             = b_ori
#             rho_dict[name]               = rho
#             preshifted_center_dict[name] = preshifted_center
#             W_adjusted_dict[name]        = W_adjusted  # already correctly scaled

#         else:
#             beta_dict[name]              = None
#             b_ori_dict[name]             = None
#             rho_dict[name]               = None
#             preshifted_center_dict[name] = 2**(e_bits - 1) - 1
#             W_adjusted_dict[name]        = W

#     # Step 3: compute Hessian with beta-scaled activations
#     # beta = 2^(b_ori) scales activations UP to match the coordinate system
#     # in which W_adjusted = W * 2^(-b_ori) will be reconstructed
#     print("Computing Hessian with beta-scaled activations...")
#     H_dict = compute_hessian_blockdiag_model_joint_preshifted(
#         model, calib_loader, device, block_size,
#         beta_dict=beta_dict,
#         num_batches=num_batches,
#     )

#     # Step 4: reconstruct each layer
#     for name, module in model.named_modules():
#         if type(module).__name__ != "QuantLinearFP":
#             continue
#         if name not in H_dict:
#             print(f"  Missing Hessian for {name}, skipping")
#             continue

#         print(f"Calibrating {name}")

#         W_adjusted        = W_adjusted_dict[name]  # already W * 2^(-b_ori)
#         rho               = rho_dict[name]
#         b_ori             = b_ori_dict[name]
#         preshifted_center = preshifted_center_dict[name]

#         # Swap in adjusted weights for v5 reconstruction
#         original_weight           = module.linear.weight.data.clone()
#         module.linear.weight.data = W_adjusted

#         res = reconstruct_layer_fp_blockdiag_scaled_v5_preshifted(
#             module.linear,
#             H_dict[name],
#             block_size,
#             e_bits, m_bits,
#             e_bits_scale, m_bits_scale,
#             device,
#             preshifted_center=preshifted_center,
#         )

#         # Restore original weights
#         module.linear.weight.data = original_weight

#         # weight_q is quantized W_adjusted — do NOT undo beta
#         # At inference: out = F.linear(quantize(x * 2^b_ori, rho), weight_q)
#         # which equals F.linear(x, W) in expectation
#         module.weight_q       = res["weight_q"].view_as(module.linear.weight)
#         module.alpha_q        = res["alpha"]
#         module.bias_q         = res["bias"]
#         module.act_rho        = rho.item() if rho is not None else None
#         module.act_b_ori      = b_ori.cpu() if b_ori is not None else None
#         module.act_quant_mode = "preshifted"
#         module.act_block_size = block_size

#         torch.cuda.empty_cache()

#     return model


def calibrate_model_preshifted(model, calib_loader, block_size, device,
                                e_bits=2, m_bits=1,
                                e_bits_scale=4, m_bits_scale=3,
                                num_batches=4):
    model.eval().to(device)

    print("Collecting per-block activation statistics...")
    act_block_max = collect_per_block_activation_max(
        model, calib_loader, device, block_size, num_batches
    )

    beta_dict       = {}
    b_ori_dict      = {}
    rho_dict        = {}
    W_adjusted_dict = {}

    for name, module in model.named_modules():
        if type(module).__name__ != "QuantLinearFP":
            continue

        W       = module.linear.weight.data.to(device).float()
        blk_max = act_block_max.get(name, None)

        if blk_max is not None:
            W_adjusted, rho, b_ori, _ = \
                preshifted_bias_and_adjust_weights_blockwise(
                    W, blk_max, block_size, e_bits, m_bits, device
                )

            in_features = W.shape[1]
            beta = torch.ones(in_features, device=device)
            for k in range(len(b_ori)):
                start = k * block_size
                end   = min(start + block_size, in_features)
                beta[start:end] = 2.0 ** b_ori[k].float()

            beta_dict[name]       = beta
            b_ori_dict[name]      = b_ori
            rho_dict[name]        = rho
            W_adjusted_dict[name] = W_adjusted

        else:
            beta_dict[name]       = None
            b_ori_dict[name]      = None
            rho_dict[name]        = None
            W_adjusted_dict[name] = W

    print("Computing Hessian with beta-scaled activations...")
    H_dict = compute_hessian_blockdiag_model_joint_preshifted(
        model, calib_loader, device, block_size,
        beta_dict=beta_dict,
        num_batches=num_batches,
    )

    for name, module in model.named_modules():
        if type(module).__name__ != "QuantLinearFP":
            continue
        if name not in H_dict:
            print(f"  Missing Hessian for {name}, skipping")
            continue

        print(f"Calibrating {name}")

        # Capture original weights BEFORE any modification
        W_orig     = module.linear.weight.data.to(device).float().clone()
        W_adjusted = W_adjusted_dict[name]
        rho        = rho_dict[name]
        b_ori      = b_ori_dict[name]
        beta       = beta_dict[name]

        # Swap in adjusted weights for v5 reconstruction
        original_weight           = module.linear.weight.data.clone()
        module.linear.weight.data = W_adjusted

        res = reconstruct_layer_fp_blockdiag_scaled_v5_preshifted(
            module.linear,
            H_dict[name],
            block_size,
            e_bits, m_bits,
            e_bits_scale, m_bits_scale,
            device,
            b_ori=b_ori,
        )

        # Restore original weights
        module.linear.weight.data = original_weight

        module.weight_q       = res["weight_q"].view_as(module.linear.weight)
        module.alpha_q        = res["alpha"]
        module.bias_q         = res["bias"]
        module.act_rho        = rho.item() if rho is not None else None
        module.act_b_ori      = b_ori.cpu() if b_ori is not None else None
        module.act_quant_mode = "preshifted"
        module.act_block_size = block_size

        # Single-layer sanity check using W_orig captured before any swap
        with torch.no_grad():
            in_f     = W_orig.shape[1]
            x_test   = torch.randn(4, 16, in_f, device=device)

            if beta is not None:
                beta_exp = beta.unsqueeze(0).unsqueeze(0)  # (1, 1, in_features)
                x_scaled = x_test * beta_exp
            else:
                x_scaled = x_test

            out_orig  = F.linear(x_test,   W_orig)
            out_quant = F.linear(x_scaled, module.weight_q)

            rel_err = (out_orig - out_quant).abs().mean() / \
                      out_orig.abs().mean().clamp(min=1e-8)
            print(f"  Single-layer rel_err: {rel_err.item():.4f}  "
                  f"out_max={out_orig.abs().max():.4f}  "
                  f"quant_max={out_quant.abs().max():.4f}")

        torch.cuda.empty_cache()

    return model


def hadamard_matrix(n, device):
    """
    Generate normalized Hadamard matrix of size n (must be power of 2).
    H @ H.T = I  (orthonormal)
    """
    assert (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    H = torch.ones(1, 1, device=device)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H,  H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0) / math.sqrt(2)
    return H  # [n, n], orthonormal


def _next_pow2(n: int) -> int:
    """Smallest power of 2 >= n."""
    return 1 << (max(int(n), 1) - 1).bit_length()


def _rotate(x, D, block_size):
    """
    Padded randomized blockwise Hadamard: fwht_blockwise(D * pad(x), block_size).

    x is zero-padded on the last dim up to len(D) before the sign flip and FWHT.
    When len(D) == x.shape[-1] (power-of-2 rows) this is a no-op pad and reduces
    to the original fwht_blockwise(x * D, block_size).  When len(D) > x.shape[-1]
    (e.g. M=2560 padded to P=4096) it performs ONE full-row Hadamard on the
    padded vector instead of a block-diagonal M&-M rotation.  Because H is
    orthogonal and the pad is zeros, (H·x_pad)·(H·W_pad) == x·W exactly.
    """
    P = D.shape[-1]
    M = x.shape[-1]
    if M < P:
        x = F.pad(x, (0, P - M))
    elif M > P:
        raise ValueError(f"_rotate: x width {M} exceeds D width {P}")
    return fwht_blockwise(x * D.to(device=x.device, dtype=x.dtype), block_size)


def fwht_blockwise(x, block_size):
    """
    Apply Fast Walsh-Hadamard Transform blockwise along the last dimension.
    x: [..., M] where M must be divisible by block_size (power of 2)
    Returns: [..., M] with each block of block_size features transformed.
    Normalization: divides by sqrt(block_size) so H @ H.T = I.
    """
    assert (block_size & (block_size - 1)) == 0, \
        f"block_size must be power of 2, got {block_size}"
    assert x.shape[-1] % block_size == 0, \
        f"Last dim {x.shape[-1]} not divisible by block_size {block_size}"

    orig_shape = x.shape
    M          = orig_shape[-1]
    n_blocks   = M // block_size

    # Reshape to [..., n_blocks, block_size]
    x = x.reshape(*orig_shape[:-1], n_blocks, block_size)

    # Butterfly iterations
    h = 1
    while h < block_size:
        # [..., n_blocks, block_size//(2h), 2, h]
        x = x.reshape(*orig_shape[:-1], n_blocks, block_size // (2 * h), 2, h)
        x_left  = x[..., 0, :]  + x[..., 1, :]
        x_right = x[..., 0, :] - x[..., 1, :]
        x = torch.stack([x_left, x_right], dim=-2)
        h *= 2

    # Normalize and restore shape
    x = x.reshape(*orig_shape) / math.sqrt(block_size)
    return x


def apply_hadamard_to_weights(W, block_size, D=None, device='cuda'):
    """
    Fold blockwise Hadamard into weight matrix offline.
    W:          [N, M] weight matrix
    block_size: size of each Hadamard block (matches quantization block size)
    D:          [M] random sign vector (+/-1), if None uses all ones (fixed H)
    
    Computes W_had = W @ (D * H)^T = W @ H^T @ D
    (since H is symmetric for normalized Hadamard, H^T = H)
    
    At runtime: x_had = H(D * x) blockwise
    Then:       W_had @ x_had = W @ H^T @ D @ H @ D @ x = W @ x  ✓
    (because H^T H = I and D^2 = I)
    """
    N, M = W.shape
    assert M % block_size == 0

    # Apply random sign flip to columns of W: W @ D
    if D is not None:
        W = W * D.to(W.device).unsqueeze(0)  # [N, M] * [1, M]

    # Apply blockwise H^T to columns of W
    # W @ H^T blockwise = apply H to each block of columns
    # Equivalently: transform rows of W^T, i.e. transform W along dim=1
    W_had = fwht_blockwise(W, block_size)   # [N, M]
    return W_had


def _stable_seed(name: str) -> int:
    """Deterministic per-layer seed from a layer name. Python's built-in hash()
    is salted per process (PYTHONHASHSEED), so hash(name) gives DIFFERENT Hadamard
    rotations on every run — ±1-2 PPL of run-to-run noise that makes the results
    table irreproducible. A content hash (sha1) is stable across processes."""
    import hashlib
    return int(hashlib.sha1(name.encode()).hexdigest(), 16) % (2 ** 31)


def generate_random_signs(M, block_size, device, seed=None):
    """
    Generate per-layer random sign vector D of shape [M].
    Each element is +1 or -1.
    If seed is provided, generation is deterministic.
    """
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        D = torch.randint(0, 2, (M,), device=device,
                          generator=generator).float() * 2 - 1
    else:
        D = torch.randint(0, 2, (M,), device=device).float() * 2 - 1
    return D


# class HadamardQuantLinearFP(nn.Module):
#     """
#     QuantLinearFP wrapper that applies a blockwise randomized Hadamard
#     transform to activations before FP4 quantization.
    
#     At calibration time:
#         W_had = W @ (D ⊙ H)^T  stored as weight_q after FP4 quantization
#         D stored as buffer for runtime use
    
#     At runtime forward:
#         x_had = H(D ⊙ x)  blockwise
#         out   = F.linear(quantize(x_had), W_q_had)
#     """
#     def __init__(self, quant_linear_module):
#         """
#         Wrap an existing QuantLinearFP module.
#         quant_linear_module: a QuantLinearFP instance (already has .linear inside)
#         """
#         super().__init__()
#         self.inner         = quant_linear_module
#         self.had_block_size = None   # set during calibration
#         self.register_buffer('D', None)  # random sign vector [M]

#     # ── Delegate all QuantLinearFP attributes ──────────────────────────
#     @property
#     def linear(self):        return self.inner.linear
#     @property
#     def weight_q(self):      return self.inner.weight_q
#     @weight_q.setter
#     def weight_q(self, v):   self.inner.weight_q = v
#     @property
#     def alpha_q(self):       return self.inner.alpha_q
#     @alpha_q.setter
#     def alpha_q(self, v):    self.inner.alpha_q = v
#     @property
#     def bias_q(self):        return self.inner.bias_q
#     @bias_q.setter
#     def bias_q(self, v):     self.inner.bias_q = v
#     @property
#     def act_quant_mode(self):      return self.inner.act_quant_mode
#     @act_quant_mode.setter
#     def act_quant_mode(self, v):   self.inner.act_quant_mode = v
#     @property
#     def act_block_size(self):      return self.inner.act_block_size
#     @act_block_size.setter
#     def act_block_size(self, v):   self.inner.act_block_size = v
#     @property
#     def smooth_scale(self):        return self.inner.smooth_scale
#     @smooth_scale.setter
#     def smooth_scale(self, v):     self.inner.smooth_scale = v

#     def _apply_hadamard(self, x):
#         """
#         Apply randomized blockwise Hadamard to input activations.
#         x: [..., M]
#         """
#         if self.D is not None:
#             x = x * self.D.to(device=x.device, dtype=x.dtype)
#         return fwht_blockwise(x, self.had_block_size)

#     def forward(self, x):
#         orig_dtype = x.dtype

#         # Check inner module's buffer directly to avoid register_buffer None gotcha
#         weight_q = self.inner._buffers.get('weight_q', None)
#         if weight_q is None:
#             # Also check as regular attribute in case it was set post-registration
#             weight_q = getattr(self.inner, 'weight_q', None)

#         W = weight_q if weight_q is not None else self.inner.linear.weight
#         b = self.inner.linear.bias

#         if weight_q is not None:
#             if self.inner.smooth_scale is not None:
#                 x = x / self.inner.smooth_scale.to(x.dtype)

#             if self.had_block_size is not None:
#                 x = self._apply_hadamard(x)

#             if self.inner.act_quant_mode is not None:
#                 x = self.inner._quantize_input(x)

#             W = W.to(orig_dtype)
#             if b is not None:
#                 b = b.to(orig_dtype)
#         # else:
#         #     print("Warning: weight_q is None, using original weights.")

#         return F.linear(x, W, b).to(orig_dtype)


# class HadamardQuantLinearFP(nn.Module):
#     """
#     Wrapper around QuantLinearFP that applies a blockwise randomized
#     Hadamard transform to activations before quantization.

#     Forward path:
#       1. (optional) subtract per-channel mean μ  — handles token outliers
#       2. apply fwht(D * x)                       — handles channel outliers
#       3. quantize activations                     — FP4 block quantization
#       4. F.linear(x_q, W_had_q, b)               — GEMM with folded weights
#       5. add bias correction W_had @ μ            — compensates mean subtraction

#     All parameters stored on this module; inner.weight_q holds Q(W_had).
#     """

#     def __init__(self, inner: "QuantLinearFP"):
#         super().__init__()
#         self.inner        = inner
#         self.had_block_size = None   # set during calibration
#         self.D              = None   # random ±1 signs [M], CPU
#         self.act_quant_mode = None   # "nvfp4" when active
#         self.act_block_size = None

#         # Mean subtraction buffers — set during calibration
#         # mu [M]: per-channel mean of post-Hadamard activations
#         # bias_correction [N]: W_had_q @ mu, added back after GEMM
#         self.register_buffer("mu",               None)
#         self.register_buffer("bias_correction",  None)

#     # ── property delegation so calibration code can do module.weight_q = x ──
#     @property
#     def weight_q(self):
#         return self.inner._buffers.get("weight_q", None)

#     @weight_q.setter
#     def weight_q(self, v):
#         self.inner.weight_q = v

#     @property
#     def act_quant_mode(self):
#         return getattr(self, "_act_quant_mode", None)

#     @act_quant_mode.setter
#     def act_quant_mode(self, v):
#         self._act_quant_mode = v

#     # ── Hadamard helper ───────────────────────────────────────────────────────
#     def _apply_hadamard(self, x):
#         """Apply randomized blockwise Hadamard: fwht(D * x)."""
#         assert self.had_block_size is not None, \
#             "had_block_size not set — was calibration run?"
#         if self.D is not None:
#             x = x * self.D.to(device=x.device, dtype=x.dtype)
#         return fwht_blockwise(x, self.had_block_size)

#     # ── Forward ───────────────────────────────────────────────────────────────
#     def forward(self, x):
#         orig_dtype = x.dtype
#         orig_shape = x.shape          # [..., M]
#         x_2d       = x.reshape(-1, x.shape[-1])   # [T, M]

#         weight_q = self.inner._buffers.get("weight_q", None)
#         if weight_q is None:
#             weight_q = getattr(self.inner, "weight_q", None)

#         b = self.inner.linear.bias

#         if weight_q is not None:
#             # ── Step 1: smooth_scale (SmoothQuant, if present) ────────────
#             smooth_scale = getattr(self.inner, "smooth_scale", None)
#             if smooth_scale is not None:
#                 x_2d = x_2d / smooth_scale.to(
#                     device=x_2d.device, dtype=x_2d.dtype)

#             # ── Step 2: subtract per-channel mean ────────────────────────
#             # μ was computed on post-Hadamard calibration activations.
#             # Subtracting before the Hadamard is equivalent because
#             # H(x - μ_orig) = H(x) - H(μ_orig) and we store H(μ_orig)
#             # directly as μ. So subtract μ in the Hadamard domain:
#             #   H(D*x) - μ  →  quantize  →  W_had_q @ (H(D*x) - μ) + b
#             #                             = W_had_q @ H(D*x) - W_had_q@μ + b
#             # The term -W_had_q@μ is the stored bias_correction (pre-added
#             # to b at calibration time), so we just quantize x_centered.
#             #
#             # In code we apply the Hadamard first, then subtract μ.

#             # ── Step 3: Hadamard transform ────────────────────────────────
#             if self.had_block_size is not None:
#                 x_2d = self._apply_hadamard(x_2d)

#             # ── Step 4: mean subtraction in Hadamard domain ───────────────
#             if self.mu is not None:
#                 x_2d = x_2d - self.mu.to(device=x_2d.device,
#                                           dtype=x_2d.dtype)

#             # ── Step 5: activation quantization ──────────────────────────
#             if self.act_quant_mode is not None:
#                 bs = self.act_block_size or 16
#                 x_2d = quantize_activations(
#                     x_2d, bs,
#                     e_bits=2, m_bits=1,
#                     e_bits_scale=4, m_bits_scale=3
#                 ).to(orig_dtype)

#             # ── Step 6: GEMM ──────────────────────────────────────────────
#             W = weight_q.to(dtype=orig_dtype)
#             out_2d = F.linear(x_2d, W, b.to(orig_dtype) if b is not None
#                                          else None)

#             # ── Step 7: add bias correction for mean subtraction ──────────
#             # W_had_q @ μ was precomputed at calibration and stored.
#             # We subtracted μ before the GEMM so we add W_had_q@μ back:
#             #   out = W_had_q @ (x_had - μ) + b + W_had_q@μ
#             #       = W_had_q @ x_had + b        ✓
#             if self.bias_correction is not None:
#                 out_2d = out_2d + self.bias_correction.to(
#                     device=out_2d.device, dtype=out_2d.dtype)

#         else:
#             # Fallback: no quantization, run original weights
#             W      = self.inner.linear.weight.to(orig_dtype)
#             out_2d = F.linear(x_2d, W,
#                               b.to(orig_dtype) if b is not None else None)

#         return out_2d.reshape(*orig_shape[:-1], -1).to(orig_dtype)



class HadamardQuantLinearFP(nn.Module):
    """
    Wrapper around QuantLinearFP that applies a blockwise randomized
    Hadamard transform to activations before quantization.

    Forward path:
      1. (optional) smooth_scale division
      2. apply fwht(D * x)          — always, weights live in H-domain
      3. (optional) subtract μ      — handles token outliers
      4. (optional) quantize acts   — FP4 block quantization
      5. F.linear(x_q, W_had_q, b)  — GEMM
      6. (optional) add bias_corr   — compensates μ subtraction

    The Hadamard (step 2) is UNCONDITIONAL when had_block_size is set
    because weight_q = Q(H(W*D)) lives in the Hadamard domain.
    Skipping step 2 in A16 mode would compute W_had_q @ x_raw which
    is incorrect and causes ~1.6 rel_err → PPL explosion.
    """

    def __init__(self, inner: "QuantLinearFP"):
        super().__init__()
        self.inner           = inner
        self.had_block_size  = None
        self.D               = None
        self._act_quant_mode = None
        self.act_block_size  = None
        self.act_clip_ratio  = None   # per-layer GF4 clip ratio (calibrated)
        self.gf4_levels      = None   # [8] learned codebook (None → use GF4_POS)
        self.h_smooth_scale  = None   # [M] per-channel H-domain scale (None → disabled)
        self.use_fast_kernels = False  # use fwht_hadacore + gf4_quant Triton kernels
        self.register_buffer("mu",              None)
        self.register_buffer("bias_correction", None)

    # ── property delegation ───────────────────────────────────────────────
    @property
    def weight_q(self):
        return self.inner._buffers.get("weight_q", None)

    @weight_q.setter
    def weight_q(self, v):
        self.inner.weight_q = v

    @property
    def act_quant_mode(self):
        return self._act_quant_mode

    @act_quant_mode.setter
    def act_quant_mode(self, v):
        self._act_quant_mode = v

    # ── Hadamard helper ───────────────────────────────────────────────────
    def _apply_hadamard(self, x):
        """Padded randomized blockwise Hadamard: fwht(D * pad(x)). [T,M] → [T,P].

        x is zero-padded on the last dim up to len(D) (== had_block_size in the
        "auto" full-Hadamard path) so the rotated activations match weight_q,
        which lives in the padded [N, P] Hadamard domain.  For power-of-2 rows
        len(D) == M and the pad is a no-op.
        """
        assert self.had_block_size is not None, \
            "had_block_size not set — was calibration run?"
        if self.D is not None:
            D = self.D.to(device=x.device, dtype=x.dtype)
            if x.shape[-1] < D.shape[-1]:
                x = F.pad(x, (0, D.shape[-1] - x.shape[-1]))
            x = x * D
        elif x.shape[-1] < self.had_block_size:
            x = F.pad(x, (0, self.had_block_size - x.shape[-1]))
        if self.use_fast_kernels:
            # HadaCore: single-kernel FWHT, all butterfly stages in L2 cache.
            # Lazy import so the Triton extension is only loaded on demand.
            from .triton_kernels import fwht_hadacore
            # Return fp32 (do NOT downcast to x.dtype here). The Hadamard output
            # feeds the μ-subtraction, which is a catastrophic-cancellation site:
            # for large-outlier models (e.g. opt-13b) the Hadamard-domain mean μ
            # is large, and computing x_had - μ in fp16 annihilates the ~O(1)
            # signal (two big near-equal fp16 numbers). Keeping fp32 through the
            # mean subtraction fixes the uniform-collapse-at-scale bug; the small
            # residual is downcast to the compute dtype afterwards.
            return fwht_hadacore(x.float(), self.had_block_size)
        return fwht_blockwise(x.float(), self.had_block_size)

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(self, x):
        orig_dtype = x.dtype
        orig_shape = x.shape
        x_2d       = x.reshape(-1, x.shape[-1])   # [T, M]

        weight_q = self.inner._buffers.get("weight_q", None)
        if weight_q is None:
            weight_q = getattr(self.inner, "weight_q", None)

        b = self.inner.linear.bias

        if weight_q is not None:

            # ── Step 1: smooth_scale (SmoothQuant, optional) ──────────
            smooth_scale = getattr(self.inner, "smooth_scale", None)
            if smooth_scale is not None:
                x_2d = x_2d / smooth_scale.to(device=x_2d.device,
                                               dtype=x_2d.dtype)

            # ── Step 2: Hadamard — ALWAYS when had_block_size is set ──
            # weight_q = Q(H(W*D)) lives in the Hadamard domain.
            # We MUST apply H(D*x) here regardless of act_quant_mode,
            # otherwise W_had_q @ x_raw is computed which is wrong.
            # This is the source of PPL=2100 when omitted in A16 mode.
            if self.had_block_size is not None:
                x_2d = self._apply_hadamard(x_2d)

            # ── Step 3: mean subtraction (optional) ───────────────────
            # μ = E[H(D*x)] computed at calibration.
            # Subtracting removes the token-outlier mean component
            # that survives the Hadamard and dominates block scales.
            # bias_correction = W_had_q @ μ compensates in step 6.
            if self.mu is not None:
                x_2d = x_2d - self.mu.to(device=x_2d.device,
                                          dtype=x_2d.dtype)

            # ── Step 3.5: H-domain per-channel smooth scaling (optional) ──
            # h_smooth_scale = per-channel std of H(D*x), calibrated once.
            # Dividing equalizes channel variance → per-block RMS clipping
            # is near-optimal for all blocks.  W_had_q is pre-multiplied
            # by h_smooth_scale offline so the output is unchanged:
            #   (x/s) @ (W*s)^T = x @ W^T  (exact).
            if self.h_smooth_scale is not None:
                s = self.h_smooth_scale.to(device=x_2d.device, dtype=x_2d.dtype)
                x_2d = x_2d / s

            # The Hadamard (step 2) and μ-subtraction (step 3) ran in fp32 to
            # avoid catastrophic cancellation when μ is large (large-outlier
            # models like opt-13b). The residual is now small and safe to return
            # to the compute dtype for activation-quant + GEMM.
            if x_2d.dtype != orig_dtype:
                x_2d = x_2d.to(orig_dtype)

            # ── Step 4: activation quantization (optional) ────────────
            if self._act_quant_mode is not None:
                bs  = self.act_block_size or 16
                lvl = self.gf4_levels.to(x_2d.device) \
                      if self.gf4_levels is not None else None

                if self._act_quant_mode == "gf4":
                    clip = self.act_clip_ratio if self.act_clip_ratio is not None else 2.5
                    if self.use_fast_kernels and lvl is None:
                        # Triton GF4 kernel: ~2.7× faster than PyTorch.
                        # Falls back to Python when custom levels are active
                        # (learned codebook) since the Triton path uses fixed
                        # GF4_POS levels only.
                        from .triton_kernels import gf4_quant, gf4_dequant
                        idx, sc = gf4_quant(x_2d.float(), clip_ratio=clip,
                                            gf4_block=bs)
                        x_2d = gf4_dequant(idx, sc, gf4_block=bs).to(orig_dtype)
                    else:
                        x_2d = quantize_activations_gf4(
                            x_2d, bs, clip_ratio=clip, levels=lvl
                        ).to(orig_dtype)

                elif self._act_quant_mode == "gf4_adaptive":
                    x_2d = quantize_activations_gf4_adaptive(
                        x_2d, bs, levels=lvl
                    ).to(orig_dtype)

                elif self._act_quant_mode == "gf4_residual":
                    clip = self.act_clip_ratio if self.act_clip_ratio is not None else 2.5
                    x_2d = quantize_activations_gf4_residual(
                        x_2d, bs, clip_ratio1=clip, clip_ratio2=clip, levels=lvl
                    ).to(orig_dtype)

                else:
                    x_2d = quantize_activations(
                        x_2d, bs,
                        e_bits=2, m_bits=1,
                        e_bits_scale=4, m_bits_scale=3
                    ).to(orig_dtype)

            # ── Step 5: GEMM ──────────────────────────────────────────
            W      = weight_q.to(dtype=orig_dtype)
            b_cast = b.to(orig_dtype) if b is not None else None
            out_2d = F.linear(x_2d, W, b_cast)

            # ── Step 6: bias correction ───────────────────────────────
            # Adds back W_had_q @ μ that was subtracted in step 3.
            # Only needed when bias_correction was not folded into b.
            if self.bias_correction is not None:
                out_2d = out_2d + self.bias_correction.to(
                    device=out_2d.device, dtype=out_2d.dtype)

        else:
            # Fallback: weight_q not yet set — use original weights
            W      = self.inner.linear.weight.to(orig_dtype)
            b_cast = b.to(orig_dtype) if b is not None else None
            out_2d = F.linear(x_2d, W, b_cast)

        return out_2d.reshape(*orig_shape[:-1], -1).to(orig_dtype)


def save_quantized_model(model, path: str) -> str:
    """Save a calibrated FP-quant model so it can be reloaded and RUN later.

    Uses a FULL-OBJECT save (torch.save(model)), not model.state_dict(), on
    purpose: HadamardQuantLinearFP keeps essential state in PLAIN ATTRIBUTES —
    the Hadamard sign vector ``D``, ``had_block_size``, ``act_quant_mode``,
    ``act_block_size``, ``act_clip_ratio``, ``gf4_levels``, ``h_smooth_scale``,
    ``use_fast_kernels`` — which ``state_dict()`` does NOT capture (only ``mu``,
    ``bias_correction`` and ``weight_q`` are registered buffers).  Without ``D``
    and ``had_block_size`` the reloaded model cannot run.

    Reload with :func:`load_quantized_model`.  The FP_Quantization_Experiments
    package (and transformers) must be importable at load time.
    """
    import os
    import torch
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    torch.save(model, path)
    return path


def load_quantized_model(path: str, device: str = "cuda"):
    """Reload a model saved by :func:`save_quantized_model` onto ``device``.

    ``weights_only=False`` because this is a full module pickle (PyTorch >=2.6
    defaults to True).  ``map_location`` lets a GPU-saved checkpoint load on a
    different/absent GPU (pass device="cpu" for analysis on a CPU box).
    """
    import torch
    model = torch.load(path, map_location=device, weights_only=False)
    model.to(device).eval()
    return model


def enable_fast_kernels(model, enable: bool = True) -> None:
    """
    Enable or disable Triton fast kernels (HadaCore FWHT + GF4 quant)
    for all HadamardQuantLinearFP layers in a model.

    Call after calibration is complete:
        quantize_model_fp(...)         # calibrate
        enable_fast_kernels(model)     # switch to Triton paths

    The kernels are loaded lazily on first use; this call itself is free.
    Requires triton >= 3.0 and a CUDA device.

    Args:
        model:  any nn.Module containing HadamardQuantLinearFP layers.
        enable: set False to revert to the Python fallback paths.
    """
    count = 0
    for module in model.modules():
        if type(module).__name__ == "HadamardQuantLinearFP":
            module.use_fast_kernels = enable
            count += 1
    print(f"{'Enabled' if enable else 'Disabled'} fast kernels on {count} "
          f"HadamardQuantLinearFP layer{'s' if count != 1 else ''}.")


def wrap_layers_with_hadamard(model):
    """
    Replace all QuantLinearFP modules with HadamardQuantLinearFP wrappers.
    Does a single clean pass to avoid mutation-during-iteration issues.
    """
    # First collect all (parent, child_name, module) triples
    replacements = []
    for name, module in model.named_modules():
        if type(module).__name__ == "QuantLinearFP":
            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                # Navigate to parent using module hierarchy, not named_modules dict
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            else:
                parent      = model
                child_name  = name
            replacements.append((parent, child_name, module))

    # Then do all replacements after iteration is complete
    for parent, child_name, module in replacements:
        wrapper = HadamardQuantLinearFP(module)
        setattr(parent, child_name, wrapper)

    n = sum(1 for _, m in model.named_modules()
            if type(m).__name__ == "HadamardQuantLinearFP")
    print(f"Wrapped {n} layers with HadamardQuantLinearFP")
    return model


# def calibrate_model_hadamard_joint(
#     model,
#     calib_loader,
#     block_size,
#     device,
#     e_bits=2,
#     m_bits=1,
#     e_bits_scale=4,
#     m_bits_scale=3,
#     num_batches=4,
#     damping=0.01,
#     apply_correction=True,
#     had_block_size=16,        # Hadamard block size — matches FP4 quant block
#     randomize_hadamard=True,  # use randomized HD vs fixed H
#     hadamard_seed=None,       # None = different D per layer, int = reproducible
# ):
#     """
#     Joint W+A FP4 quantization with blockwise randomized Hadamard preprocessing.
    
#     Order of operations per layer:
#       1. Generate random sign vector D for this layer
#       2. Fold H and D into weights: W_had = W @ (D * H)^T
#       3. Collect activations X, apply H(D*X) to get X_had
#       4. Quantize X_had -> X_hat_had
#       5. Recompute H_joint from X_hat_had
#       6. Compute weight correction on W_had using X_had
#       7. Run v5/v6 on corrected W_had with H_joint
#       8. Store weight_q (already incorporates H and D)
#       9. Store D on module for runtime use
#     """
#     # In calibrate_model_hadamard_joint
#     ACT_QUANT_THRESHOLD = 0.65  # more permissive since Hadamard has already helped
#     model.eval().to(device)

#     # ── Phase 1: wrap all QuantLinearFP modules ────────────────────────
#     # print("Wrapping layers with HadamardQuantLinearFP...")
#     # for name, module in list(model.named_modules()):
#     #     if type(module).__name__ != "QuantLinearFP":
#     #         continue
#     #     # Replace with wrapper — need to handle nested module assignment
#     #     parent_name, child_name = name.rsplit('.', 1) \
#     #         if '.' in name else ('', name)
#     #     parent = model if parent_name == '' \
#     #         else dict(model.named_modules())[parent_name]
#     #     wrapper = HadamardQuantLinearFP(module)
#     #     setattr(parent, child_name, wrapper)
#     print("Wrapping layers with HadamardQuantLinearFP...")
#     model = wrap_layers_with_hadamard(model)
#     n_wrapped = sum(1 for _, m in model.named_modules()
#                 if type(m).__name__ == "HadamardQuantLinearFP")
#     n_inner   = sum(1 for _, m in model.named_modules()
#                     if type(m).__name__ == "QuantLinearFP")
#     print(f"  HadamardQuantLinearFP wrappers: {n_wrapped}")
#     print(f"  Raw QuantLinearFP remaining:    {n_inner}")
#     # n_inner should equal n_wrapped (the inner modules inside wrappers)
#     # If n_inner > n_wrapped that means some weren't wrapped
#     # ── Phase 2: collect activations (on wrapped model) ───────────────
#     # Disable act_quant_mode so hooks see clean activations
#     for _, m in model.named_modules():
#         if type(m).__name__ == "HadamardQuantLinearFP":
#             m._saved_aqm   = m.act_quant_mode
#             m.act_quant_mode = None

#     print("Collecting calibration activations...")
#     X_dict = collect_calibration_activations(
#         model, calib_loader, device, block_size, num_batches
#     )

#     for _, m in model.named_modules():
#         if type(m).__name__ == "HadamardQuantLinearFP":
#             m.act_quant_mode = m._saved_aqm

#     SKIP_LAYERS = {"lm_head", "embed_tokens", "embed_positions"}

#     # ── Phase 3: per-layer calibration ────────────────────────────────
#     for name, module in model.named_modules():
#         if type(module).__name__ != "HadamardQuantLinearFP":
#             continue
#         if any(skip in name for skip in SKIP_LAYERS):
#             print(f"  Skipping {name}")
#             continue
#         if name not in X_dict:
#             print(f"  No activations for {name}, skipping")
#             continue

#         print(f"Calibrating {name}")

#         X_calib = X_dict[name].float()             # [T, M] CPU
#         W_orig  = module.linear.weight.data.to(device).float()

#         if W_orig.dim() == 4:
#             W_mat = W_orig.view(W_orig.shape[0], -1)
#         else:
#             W_mat = W_orig

#         N, M = W_mat.shape
#         T    = X_calib.shape[0]

#         assert X_calib.shape[1] == M, \
#             f"{name}: X has {X_calib.shape[1]} features, W has {M}"
#         assert M % had_block_size == 0, \
#             f"{name}: M={M} not divisible by had_block_size={had_block_size}"

#         # ── Step 1: generate random signs D ───────────────────────────
#         if randomize_hadamard:
#             seed = hadamard_seed if hadamard_seed is not None \
#                    else hash(name) % (2**31)
#             D = generate_random_signs(M, had_block_size, device, seed=seed)
#         else:
#             D = torch.ones(M, device=device)

#         # ── Step 2: fold H and D into weights ─────────────────────────
#         # W_had = W @ (D * H)^T = fwht_blockwise(W * D, had_block_size)
#         print(f"  Applying Hadamard to weights [N={N}, M={M}, "
#               f"had_block_size={had_block_size}]...")
#         W_had = apply_hadamard_to_weights(
#             W_mat, had_block_size, D=D, device=device
#         )                                          # [N, M] GPU

#         # ── Step 3: apply H(D*X) to calibration activations ───────────
#         print(f"  Applying Hadamard to activations [T={T}, M={M}]...")
#         X_had = fwht_blockwise(
#             X_calib.to(device) * D.unsqueeze(0),
#             had_block_size
#         ).cpu().float()                            # [T, M] CPU

#         # Verify spreading — channel max should be much lower
#         orig_max = X_calib.abs().amax().item()
#         had_max  = X_had.abs().amax().item()
#         print(f"  act_max: {orig_max:.2f} -> {had_max:.2f} "
#               f"(ratio {had_max/orig_max:.3f})")

#         # ── Step 4: quantize X_had to get X_hat ───────────────────────
#         X_hat = quantize_activations(
#             X_had.to(device),
#             block_size, e_bits, m_bits,
#             e_bits_scale=e_bits_scale,
#             m_bits_scale=m_bits_scale
#         ).cpu().float()

#         delta_X_norm = (X_had - X_hat).norm() / \
#                         X_had.norm().clamp(min=1e-8)
#         print(f"  ||delta_X_had|| / ||X_had|| = {delta_X_norm:.4f}")

#         # ── Step 5: recompute H_joint from X_hat_had ──────────────────
#         H_blocks_joint = recompute_H_blocks_from_Xhat(
#             X_hat, block_size, device
#         )

#         # ── Step 6: weight correction on W_had ────────────────────────
#         # ACT_QUANT_THRESHOLD = 0.4
#         if delta_X_norm > ACT_QUANT_THRESHOLD:
#             print(f"  delta_X_norm={delta_X_norm:.4f} > threshold, "
#                   f"disabling correction for this layer")
#             use_act_quant               = False
#             apply_correction_this_layer = False
#         else:
#             use_act_quant               = True
#             apply_correction_this_layer = apply_correction

#         W_had_cpu = W_had.cpu()

#         if apply_correction_this_layer:
#             W_corrected = compute_W_correction_blockwise(
#                 W_had_cpu, X_had, X_hat,
#                 block_size, damping, device,
#                 max_correction_ratio=0.15
#             )
#             corr_norm = (W_corrected - W_had_cpu).norm() / \
#                          W_had_cpu.norm().clamp(min=1e-8)
#             print(f"  ||W_correction|| / ||W_had|| = {corr_norm:.4f}")
#         else:
#             W_corrected = W_had_cpu

#         del X_hat, X_had
#         torch.cuda.empty_cache()

#         # ── Step 7: reconstruct with v6 on W_had ──────────────────────
#         # Temporarily swap W_had into layer for shape reference
#         original_weight           = module.linear.weight.data.clone()
#         module.linear.weight.data = W_corrected.view_as(
#             original_weight).to(device)

#         res = reconstruct_layer_fp_blockdiag_scaled_v6(
#             module.linear,
#             H_blocks_joint,
#             W_corrected,
#             block_size,
#             e_bits, m_bits,
#             e_bits_scale, m_bits_scale,
#             device,
#             Hadamard=True
#         )

#         module.linear.weight.data = original_weight

#         # ── Step 8: store results ──────────────────────────────────────
#         module.weight_q        = res["weight_q"].view_as(module.linear.weight)
#         module.alpha_q         = res["alpha"]
#         module.bias_q          = res["bias"]
#         module.act_quant_mode  = "nvfp4" if use_act_quant else None
#         module.act_block_size  = block_size
#         module.had_block_size  = had_block_size
#         module.D               = D.cpu()   # store for runtime

#         # ── Step 9: sanity check ───────────────────────────────────────
#         with torch.no_grad():
#             x_test    = torch.randn(4, 16, M, device=device)

#             # Simulate full forward: smooth -> Hadamard -> quantize -> GEMM
#             x_had_test = fwht_blockwise(
#                 x_test * D.to(device), had_block_size
#             )
#             x_q_test = quantize_activations(
#                 x_had_test, block_size, e_bits, m_bits,
#                 e_bits_scale=e_bits_scale, m_bits_scale=m_bits_scale
#             )
#             out_orig  = F.linear(x_test, W_orig.to(x_test.dtype))
#             out_quant = F.linear(
#                 x_q_test, module.weight_q.to(x_test.dtype))
#             rel_err = (out_orig - out_quant).abs().mean() / \
#                        out_orig.abs().mean().clamp(min=1e-8)
#             print(f"  rel_err (W4A4+Had): {rel_err.item():.4f}  "
#                   f"out_max={out_orig.abs().max():.3f}  "
#                   f"quant_max={out_quant.abs().max():.3f}")

#         del H_blocks_joint, W_corrected, W_had, W_had_cpu
#         del W_mat, W_orig, res, D
#         torch.cuda.empty_cache()

#     return model

# def calibrate_model_hadamard_joint(
#     model,
#     calib_loader,
#     block_size,
#     device,
#     e_bits=2,
#     m_bits=1,
#     e_bits_scale=4,
#     m_bits_scale=3,
#     num_batches=4,
#     had_block_size=16,
#     randomize_hadamard=True,
#     hadamard_seed=None,
# ):
#     """
#     Joint W+A FP4 quantization with blockwise randomized Hadamard on activations.

#     Key insight from math analysis:
#       - Hadamard H is orthogonal so H_original = (HX)^T(HX) = X^TX
#       - Weight optimization is therefore IDENTICAL to v5 using H_original
#       - The Hadamard only affects the runtime activation path
#       - H_joint from quantized activations was distorting weight optimization
#         and causing W4A16 PPL of 33 vs 14 — this is now fixed

#     Pipeline per layer:
#       1. Generate random sign vector D for this layer
#       2. Compute original H blocks (same as v5)
#       3. Run v5 weight quantization with H_original
#       4. Store D and had_block_size on module for runtime Hadamard
#       5. At runtime: fwht(D*x) -> quantize -> GEMM with Q(W)
#     """
#     model.eval().to(device)

#     # ── Phase 1: wrap all QuantLinearFP modules ────────────────────────
#     print("Wrapping layers with HadamardQuantLinearFP...")
#     model = wrap_layers_with_hadamard(model)

#     n_wrapped = sum(1 for _, m in model.named_modules()
#                     if type(m).__name__ == "HadamardQuantLinearFP")
#     n_inner   = sum(1 for _, m in model.named_modules()
#                     if type(m).__name__ == "QuantLinearFP")
#     print(f"  HadamardQuantLinearFP wrappers: {n_wrapped}")
#     print(f"  Raw QuantLinearFP remaining:    {n_inner}")

#     # ── Phase 2: compute original H blocks (same as v5) ───────────────
#     # Disable act_quant_mode so hooks see clean activations
#     for _, m in model.named_modules():
#         if type(m).__name__ == "HadamardQuantLinearFP":
#             m._saved_aqm   = m.act_quant_mode
#             m.act_quant_mode = None

#     print("Computing original Hessian blocks...")
#     H_dict_original = compute_hessian_blockdiag_model(
#         model, calib_loader, device, block_size,
#         num_batches=num_batches
#     )

#     for _, m in model.named_modules():
#         if type(m).__name__ == "HadamardQuantLinearFP":
#             m.act_quant_mode = m._saved_aqm

#     SKIP_LAYERS = {"lm_head", "embed_tokens", "embed_positions"}

#     # ── Phase 3: per-layer calibration ────────────────────────────────
#     for name, module in model.named_modules():
#         if type(module).__name__ != "HadamardQuantLinearFP":
#             continue
#         if any(skip in name for skip in SKIP_LAYERS):
#             print(f"  Skipping {name}")
#             continue
#         if name not in H_dict_original:
#             print(f"  No Hessian for {name}, skipping")
#             continue

#         print(f"Calibrating {name}")
# # ── Get weight from inner linear ──────────────────────────────
#         W_orig = module.inner.linear.weight.data.to(device).float()
#         if W_orig.dim() == 4:
#             W_mat = W_orig.view(W_orig.shape[0], -1)
#         else:
#             W_mat = W_orig

#         N, M = W_mat.shape

#         assert M % had_block_size == 0, \
#             f"{name}: M={M} not divisible by had_block_size={had_block_size}"

# # ── Step 1: generate random signs D ───────────────────────────
#         if randomize_hadamard:
#             seed = hadamard_seed if hadamard_seed is not None \
#                    else hash(name) % (2**31)
#             D = generate_random_signs(M, had_block_size, device, seed=seed)
#         else:
#             D = torch.ones(M, device=device)

#         # ── Step 2: fold H^T @ D into weights ─────────────────────────
#         # Runtime computes: Q(W_had) @ H(D*x)
#         # For this to approximate W @ x we need:
#         #   Q(W_had) ≈ W @ D @ H^T  (since H^T = H and D^2 = I)
#         # So W_had[n, :] = fwht_blockwise(W[n, :] * D)
#         # This means each row of W is sign-flipped then Hadamard-transformed.
#         # At runtime: Q(W_had) @ H(D*x)
#         #           = (W @ D @ H) @ (H @ D @ x)   [H^2 = I, D^2 = I]
#         #           = W @ D @ H^2 @ D @ x
#         #           = W @ D^2 @ x
#         #           = W @ x   ✓
#         print(f"  Folding Hadamard into weights [N={N}, M={M}, "
#               f"had_block_size={had_block_size}]...")

#         # Apply D then H to each row of W: W_had = H(W * D)
#         # w_mat rows are [N, M], broadcast D over N
#         W_had = fwht_blockwise(
#             W_mat * D.unsqueeze(0),    # [N, M] * [M] -> [N, M]
#             had_block_size
#         ).cpu().float()

#         W_had_max = W_had.abs().max().item()
#         W_orig_max = W_mat.abs().max().item()
#         print(f"  W_orig max: {W_orig_max:.4f}, W_had max: {W_had_max:.4f} "
#               f"(ratio {W_had_max/W_orig_max:.3f})")

#         # ── Step 3: activation diagnostic (informational only) ────────
#         act_sample = None
#         def _grab_hook(mod, inp, out):
#             nonlocal act_sample
#             if act_sample is None:
#                 act_sample = inp[0].detach().float().reshape(
#                     -1, inp[0].shape[-1]).cpu()

#         h = module.register_forward_hook(_grab_hook)
#         with torch.no_grad():
#             for batch in calib_loader:
#                 if batch is None:
#                     continue
#                 x = batch[0] if isinstance(batch, (list, tuple)) else batch
#                 if x is None:
#                     continue
#                 model(x.to(device))
#                 break
#         h.remove()

#         if act_sample is not None:
#             X_had = fwht_blockwise(
#                 act_sample.to(device) * D.unsqueeze(0),
#                 had_block_size
#             ).cpu().float()
#             X_hat = quantize_activations(
#                 X_had.to(device), block_size, e_bits, m_bits,
#                 e_bits_scale=e_bits_scale, m_bits_scale=m_bits_scale
#             ).cpu().float()

#             orig_max     = act_sample.abs().amax().item()
#             had_max      = X_had.abs().amax().item()
#             delta_X_norm = (X_had - X_hat).norm() / \
#                            X_had.norm().clamp(min=1e-8)
#             print(f"  act_max: {orig_max:.2f} -> {had_max:.2f} "
#                   f"(ratio {had_max/orig_max:.3f})")
#             print(f"  ||delta_X_had|| / ||X_had|| = {delta_X_norm:.4f}")

#             del X_had, X_hat, act_sample
#             torch.cuda.empty_cache()

#         # ── Step 4: quantize W_had with H_original ────────────────────
#         # Create a temporary Linear with W_had as its weight so v5 can
#         # read layer.weight.data — v5 reads the weight from the layer object
#         print(f"  Quantizing W_had with H_original [N={N}, M={M}]...")
#         tmp_linear = torch.nn.Linear(M, N, bias=False, device=device)
#         tmp_linear.weight.data = W_had.to(device)

#         res = reconstruct_layer_fp_blockdiag_scaled_v5(
#             tmp_linear,                      # temp linear wrapping W_had
#             H_dict_original[name],
#             block_size,
#             e_bits, m_bits,
#             e_bits_scale, m_bits_scale,
#             device,
#         )
#         del tmp_linear
#         torch.cuda.empty_cache()

#         # ── Step 4b: compute mean subtraction buffers ─────────────────────
#         # Collect post-Hadamard activations from calibration batches to
#         # compute per-channel mean μ = E[H(D*x)].
#         # Then precompute bias_correction = Q(W_had) @ μ so it can be
#         # added back in forward() with no per-token overhead.
#         #
#         # Why this helps W4A4:
#         #   Token outliers (BOS, delimiters) are large across ALL features
#         #   simultaneously — they have a large mean component that survives
#         #   the Hadamard transform and dominates the block scale.
#         #   Subtracting the per-channel post-Hadamard mean removes this
#         #   component before block quantization, leaving zero-mean variance
#         #   which FP4 can represent much more efficiently.
#         #
#         # The bias correction compensates exactly:
#         #   out = W_had_q @ (x_had - μ) + b + W_had_q @ μ
#         #       = W_had_q @ x_had + b   ✓

#         print(f"  Computing post-Hadamard mean (mean subtraction)...")
#         had_acts_for_mu = []

#         def _grab_had_hook(mod, inp, out):
#             """Capture raw pre-quantization input, apply same H(D*x) as forward."""
#             x_raw = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
#             # Apply D sign flip then Hadamard — same as forward path
#             x_had = fwht_blockwise(
#                 x_raw * D.to(device=x_raw.device, dtype=x_raw.dtype),
#                 had_block_size
#             )
#             had_acts_for_mu.append(x_had.cpu())

#         h_mu = module.register_forward_hook(_grab_had_hook)
#         with torch.no_grad():
#             batches_collected = 0
#             for batch in calib_loader:
#                 if batch is None:
#                     continue
#                 x = batch[0] if isinstance(batch, (list, tuple)) else batch
#                 if x is None:
#                     continue
#                 model(x.to(device))
#                 batches_collected += 1
#                 if batches_collected >= num_batches:
#                     break
#         h_mu.remove()

#         if had_acts_for_mu:
#             X_had_all = torch.cat(had_acts_for_mu, dim=0).float()
#             mu        = X_had_all.mean(dim=0)

#             print(f"  μ max={mu.abs().max():.4f}  "
#                   f"μ mean={mu.abs().mean():.4f}  "
#                   f"μ p99={torch.quantile(mu.abs(), 0.99).item():.4f}")

#             W_had_q   = res["weight_q"].to(device).float()
#             bias_corr = (W_had_q @ mu.to(device)).cpu()

#             print(f"  bias_corr max={bias_corr.abs().max():.4f}  "
#                   f"bias_corr mean={bias_corr.abs().mean():.4f}")

#             existing_bias = module.inner.linear.bias
#             if existing_bias is not None:
#                 module.inner.linear.bias.data = (
#                     existing_bias.data.float() + bias_corr.to(device)
#                 ).to(existing_bias.dtype)
#                 module.bias_correction = None
#                 print(f"  bias_corr folded into existing bias")
#             else:
#                 module.bias_correction = bias_corr.half()
#                 print(f"  bias_corr stored as separate buffer (no existing bias)")

#             module.mu = mu.half()

#             del X_had_all, W_had_q, bias_corr
#             torch.cuda.empty_cache()

#         else:
#             print(f"  WARNING: no activations collected — "
#                   f"mean subtraction disabled for {name}")
#             module.mu              = None
#             module.bias_correction = None

#         del had_acts_for_mu
#         # ── NO extra assignment here — bias_correction already set above ──
#         # ── Step 5: store results ─────────────────────────────────────────
#         module.weight_q       = res["weight_q"].view_as(module.inner.linear.weight)
#         module.alpha_q        = res["alpha"]
#         module.bias_q         = res["bias"]
#         module.act_quant_mode = "nvfp4"
#         module.act_block_size = block_size
#         module.had_block_size = had_block_size
#         module.D              = D.cpu()

#         if mu is not None:
#             module.mu              = mu.to(dtype=torch.float16)
#             # module.bias_correction = bias_corr.to(dtype=torch.float16)
#         else:
#             module.mu              = None
#             module.bias_correction = None

#         # ── Step 5: store results ──────────────────────────────────────
#         module.weight_q       = res["weight_q"].view_as(module.inner.linear.weight)
#         module.alpha_q        = res["alpha"]
#         module.bias_q         = res["bias"]
#         module.act_quant_mode = "nvfp4"
#         module.act_block_size = block_size
#         module.had_block_size = had_block_size
#         module.D              = D.cpu()

#         # ── Step 6: sanity check ──────────────────────────────────────
#         with torch.no_grad():
#             x_test = torch.randn(4, 16, M, device=device)

#             # Runtime path: H(D*x) -> quantize -> Q(W_had) @ x_had
#             x_had  = fwht_blockwise(
#                 x_test * D.unsqueeze(0).unsqueeze(0), had_block_size)
#             x_q    = quantize_activations(
#                 x_had, block_size, e_bits, m_bits,
#                 e_bits_scale=e_bits_scale, m_bits_scale=m_bits_scale)

#             # Reference: original W @ x (no Hadamard, no quantization)
#             out_orig  = F.linear(x_test,
#                                  W_mat.to(device).to(x_test.dtype))
#             # Quantized: Q(W_had) @ H(D*x_q)
#             out_quant = F.linear(x_q,
#                                  module.weight_q.to(x_test.dtype))

#             rel_err = (out_orig - out_quant).abs().mean() / \
#                       out_orig.abs().mean().clamp(min=1e-8)
#             print(f"  rel_err (W4A4+Had): {rel_err.item():.4f}  "
#                   f"out_max={out_orig.abs().max():.3f}  "
#                   f"quant_max={out_quant.abs().max():.3f}")

#         del res, D, W_orig, W_mat, W_had
#         torch.cuda.empty_cache()

#     return model



def calibrate_model_hadamard_joint(
    model,
    calib_loader,
    block_size,
    device,
    e_bits=2,
    m_bits=1,
    e_bits_scale=4,
    m_bits_scale=3,
    num_batches=4,
    had_block_size=256,
    randomize_hadamard=True,
    hadamard_seed=None,
):
    """
    Joint W+A FP4 quantization with blockwise randomized Hadamard.

    Pipeline per layer:
      1. Compute H_original (pre-wrap, hooks find QuantLinearFP)
      2. Wrap all QuantLinearFP → HadamardQuantLinearFP
      3. Per layer:
         a. Generate random signs D
         b. Compute W_had = H(W * D)
         c. Quantize W_had with v5 + H_original
         d. Compute μ = E[H(D*x)] from calibration
         e. Precompute bias_correction = Q(W_had) @ μ
         f. Store everything on the wrapper module
    """
    model.eval().to(device)

    # ── Phase 1: H_original BEFORE wrapping ───────────────────────────────
    print("Computing original Hessian blocks...")
    H_dict_original = compute_hessian_blockdiag_model(
        model, calib_loader, device, block_size,
        num_batches=num_batches
    )
    print(f"  Collected H for {len(H_dict_original)} layers")

    # ── Phase 2: wrap QuantLinearFP → HadamardQuantLinearFP ───────────────
    print("Wrapping layers with HadamardQuantLinearFP...")
    model = wrap_layers_with_hadamard(model)

    n_wrapped = sum(1 for _, m in model.named_modules()
                    if type(m).__name__ == "HadamardQuantLinearFP")
    n_inner   = sum(1 for _, m in model.named_modules()
                    if type(m).__name__ == "QuantLinearFP")
    print(f"  HadamardQuantLinearFP wrappers: {n_wrapped}")
    print(f"  Raw QuantLinearFP remaining:    {n_inner}")

    SKIP_LAYERS = {"lm_head", "embed_tokens", "embed_positions"}

    # ── Phase 3: per-layer calibration ────────────────────────────────────
    for name, module in model.named_modules():
        if type(module).__name__ != "HadamardQuantLinearFP":
            continue
        if any(skip in name for skip in SKIP_LAYERS):
            print(f"  Skipping {name}")
            continue
        if name not in H_dict_original:
            print(f"  No Hessian for {name}, skipping")
            continue

        print(f"Calibrating {name}")

        W_orig = module.inner.linear.weight.data.to(device).float()
        if W_orig.dim() == 4:
            W_mat = W_orig.view(W_orig.shape[0], -1)
        else:
            W_mat = W_orig

        N, M = W_mat.shape

        assert M % had_block_size == 0, \
            f"{name}: M={M} not divisible by had_block_size={had_block_size}"

        # ── Step 3a: generate random signs D ──────────────────────────
        if randomize_hadamard:
            seed = hadamard_seed if hadamard_seed is not None \
                   else _stable_seed(name)
            D = generate_random_signs(M, had_block_size, device, seed=seed)
        else:
            D = torch.ones(M, device=device)

        # ── Step 3b: fold H into weights W_had = H(W * D) ─────────────
        print(f"  Folding Hadamard into weights "
              f"[N={N}, M={M}, had_block_size={had_block_size}]...")
        W_had = fwht_blockwise(
            W_mat * D.unsqueeze(0), had_block_size
        ).cpu().float()

        W_orig_max = W_mat.abs().max().item()
        W_had_max  = W_had.abs().max().item()
        print(f"  W_orig max: {W_orig_max:.4f}, "
              f"W_had max: {W_had_max:.4f} "
              f"(ratio {W_had_max / W_orig_max:.3f})")

        # ── Step 3c: activation diagnostic ────────────────────────────
        act_sample = None

        def _grab_raw_hook(mod, inp, out):
            nonlocal act_sample
            if act_sample is None:
                act_sample = inp[0].detach().float().reshape(
                    -1, inp[0].shape[-1]).cpu()

        h_raw = module.register_forward_hook(_grab_raw_hook)
        with torch.no_grad():
            for batch in calib_loader:
                if batch is None:
                    continue
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                if x is None:
                    continue
                model(x.to(device))
                break
        h_raw.remove()

        if act_sample is not None:
            X_had_diag = fwht_blockwise(
                act_sample.to(device) * D.unsqueeze(0),
                had_block_size
            ).cpu().float()
            X_hat_diag = quantize_activations(
                X_had_diag.to(device), block_size, e_bits, m_bits,
                e_bits_scale=e_bits_scale, m_bits_scale=m_bits_scale
            ).cpu().float()
            orig_max     = act_sample.abs().amax().item()
            had_max      = X_had_diag.abs().amax().item()
            delta_X_norm = (X_had_diag - X_hat_diag).norm() / \
                           X_had_diag.norm().clamp(min=1e-8)
            print(f"  act_max: {orig_max:.2f} -> {had_max:.2f} "
                  f"(ratio {had_max / orig_max:.3f})")
            print(f"  ||delta_X_had|| / ||X_had|| = {delta_X_norm:.4f}")
            del X_had_diag, X_hat_diag, act_sample
            torch.cuda.empty_cache()

        # ── Step 3d: quantize W_had with H_original ───────────────────
        print(f"  Quantizing W_had with H_original [N={N}, M={M}]...")
        tmp_linear = torch.nn.Linear(M, N, bias=False, device=device)
        tmp_linear.weight.data = W_had.to(device)

        res = reconstruct_layer_fp_blockdiag_scaled_v5(
            tmp_linear,
            H_dict_original[name],
            block_size,
            e_bits, m_bits,
            e_bits_scale, m_bits_scale,
            device,
        )
        del tmp_linear
        torch.cuda.empty_cache()

        # ── Step 3e: compute μ and bias_correction ────────────────────
        print(f"  Computing post-Hadamard mean (mean subtraction)...")
        had_acts_for_mu = []

        def _grab_had_hook(mod, inp, out):
            x_raw = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            x_had = fwht_blockwise(
                x_raw * D.to(device=x_raw.device, dtype=x_raw.dtype),
                had_block_size
            )
            had_acts_for_mu.append(x_had.cpu())

        h_mu = module.register_forward_hook(_grab_had_hook)
        with torch.no_grad():
            batches_collected = 0
            for batch in calib_loader:
                if batch is None:
                    continue
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                if x is None:
                    continue
                model(x.to(device))
                batches_collected += 1
                if batches_collected >= num_batches:
                    break
        h_mu.remove()

        if had_acts_for_mu:
            X_had_all = torch.cat(had_acts_for_mu, dim=0).float()
            mu        = X_had_all.mean(dim=0)                    # [M]

            print(f"  μ max={mu.abs().max():.4f}  "
                  f"μ mean={mu.abs().mean():.4f}  "
                  f"μ p99={torch.quantile(mu.abs(), 0.99).item():.4f}")

            W_had_q   = res["weight_q"].to(device).float()       # [N, M]
            bias_corr = (W_had_q @ mu.to(device)).cpu()           # [N]

            print(f"  bias_corr max={bias_corr.abs().max():.4f}  "
                  f"bias_corr mean={bias_corr.abs().mean():.4f}")

            existing_bias = module.inner.linear.bias
            if existing_bias is not None:
                module.inner.linear.bias.data = (
                    existing_bias.data.float() + bias_corr.to(device)
                ).to(existing_bias.dtype)
                module.bias_correction = None
                print(f"  bias_corr folded into existing bias")
            else:
                module.bias_correction = bias_corr.half()
                print(f"  bias_corr stored as separate buffer")

            module.mu = mu.half()

            del X_had_all, W_had_q, bias_corr
            torch.cuda.empty_cache()

        else:
            print(f"  WARNING: no activations collected — "
                  f"mean subtraction disabled for {name}")
            module.mu              = None
            module.bias_correction = None

        del had_acts_for_mu

        # ── Step 3f: store results on wrapper ─────────────────────────
        module.weight_q       = res["weight_q"].view_as(
                                    module.inner.linear.weight)
        module.alpha_q        = res["alpha"]
        module.bias_q         = res["bias"]
        module.act_quant_mode = "nvfp4"
        module.act_block_size = block_size
        module.had_block_size = had_block_size
        module.D              = D.cpu()

        # ── Step 3g: sanity check ─────────────────────────────────────
        with torch.no_grad():
            x_test = torch.randn(4, 16, M, device=device)

            # Full A16 path: H(D*x) → subtract μ → GEMM → add corr
            x_had_t = fwht_blockwise(
                x_test * D.unsqueeze(0).unsqueeze(0),
                had_block_size
            )
            if module.mu is not None:
                x_had_t = x_had_t - module.mu.to(
                    device=device, dtype=x_had_t.dtype)

            out_a16 = F.linear(x_had_t,
                               module.weight_q.to(x_test.dtype))
            if module.bias_correction is not None:
                out_a16 = out_a16 + module.bias_correction.to(
                    device=device, dtype=out_a16.dtype)

            # Full A4 path: same + activation quantization
            x_had_q = fwht_blockwise(
                x_test * D.unsqueeze(0).unsqueeze(0),
                had_block_size
            )
            if module.mu is not None:
                x_had_q = x_had_q - module.mu.to(
                    device=device, dtype=x_had_q.dtype)
            x_q = quantize_activations(
                x_had_q, block_size, e_bits, m_bits,
                e_bits_scale=e_bits_scale, m_bits_scale=m_bits_scale
            )
            out_a4 = F.linear(x_q, module.weight_q.to(x_test.dtype))
            if module.bias_correction is not None:
                out_a4 = out_a4 + module.bias_correction.to(
                    device=device, dtype=out_a4.dtype)

            # Reference: original W @ x_test
            out_ref = F.linear(x_test,
                               W_mat.to(device).to(x_test.dtype))

            rel_a16 = (out_ref - out_a16).abs().mean() / \
                      out_ref.abs().mean().clamp(min=1e-8)
            rel_a4  = (out_ref - out_a4).abs().mean() / \
                      out_ref.abs().mean().clamp(min=1e-8)
            print(f"  rel_err A16 (Had+μ, no act quant): {rel_a16:.4f}")
            print(f"  rel_err A4  (Had+μ, act quant):    {rel_a4:.4f}")
            # A16 should be ~0.05-0.15 (weight quant error only)
            # A4  should be ~0.50-0.60 (weight + activation quant error)
            # If A16 ≈ 1.6: Hadamard not applied, weights/acts mismatched
            # If A16 ≈ 0.0: sanity check input is trivial (bad test)

        del res, D, W_orig, W_mat, W_had
        torch.cuda.empty_cache()

    # ── Spot check ─────────────────────────────────────────────────────────
    for name, module in model.named_modules():
        if type(module).__name__ == "HadamardQuantLinearFP":
            print(f"\n{name}:")
            print(f"  weight_q:      "
                  f"{'tensor' if module.weight_q is not None else 'None'}")
            print(f"  had_block_size: {module.had_block_size}")
            print(f"  D:              "
                  f"{'set' if module.D is not None else 'None'}")
            print(f"  mu:             "
                  f"{'set' if module.mu is not None else 'None'}")
            print(f"  bias_correction:"
                  f"{'set' if module.bias_correction is not None else 'None (folded)'}")
            break

    return model

def collect_per_channel_activation_max(model, calib_loader, device,
                                        num_batches=4):
    """
    Collect per-input-channel activation max.
    Hooks on BOTH nn.Linear and QuantLinearFP so it works whether called
    before or after layer replacement.
    """
    act_channel_max = {}
    hooks = []

    def make_hook(name, in_features):
        def hook(module, inp, out):
            x      = inp[0].detach().float()
            x_flat = x.reshape(-1, in_features)
            ch_max = x_flat.abs().amax(dim=0)
            if name not in act_channel_max:
                act_channel_max[name] = ch_max.cpu()
            else:
                act_channel_max[name] = torch.maximum(
                    act_channel_max[name], ch_max.cpu()
                )
        return hook

    n_hooks = 0
    for name, module in model.named_modules():
        # Hook on raw Linear OR already-wrapped QuantLinearFP
        if isinstance(module, nn.Linear) and not isinstance(
            module, (QuantLinearFP, QuantConv2dFP, QuantConv1dFP)
        ):
            in_f = module.weight.shape[1]
            hooks.append(module.register_forward_hook(make_hook(name, in_f)))
            n_hooks += 1
        elif type(module).__name__ == "QuantLinearFP":
            in_f = module.linear.weight.shape[1]
            hooks.append(module.register_forward_hook(make_hook(name, in_f)))
            n_hooks += 1

    print(f"Registered {n_hooks} hooks for per-channel activation collection")

    model.eval()
    batches_run = 0
    with torch.no_grad():
        for batch in calib_loader:
            if batch is None:
                continue
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if x is None:
                continue
            model(x.to(device))
            batches_run += 1
            if batches_run >= num_batches:
                break

    for h in hooks:
        h.remove()

    print(f"Collected per-channel activation stats for "
          f"{len(act_channel_max)} layers over {batches_run} batches")

    print(f"\n{'Layer':<55} {'in_feat':>8} {'max':>8} {'mean':>8} "
          f"{'ratio':>8} {'n_out>6x':>10}")
    print("-" * 100)
    for name, ch_max in act_channel_max.items():
        median_max = ch_max.median().clamp(min=1e-8)
        ratio      = (ch_max.max() / median_max).item()
        n_outliers = (ch_max > 6.0 * median_max).sum().item()
        print(f"  {name:<53} {ch_max.shape[0]:>8} "
              f"{ch_max.max().item():>8.3f} "
              f"{ch_max.mean().item():>8.3f} "
              f"{ratio:>8.1f} "
              f"{n_outliers:>10}")

    return act_channel_max


def preshifted_bias_perchannel(ch_max, e_bits, m_bits, device):
    """
    Compute per-channel pre-shifted exponent bias from calibration statistics.
    Implements Eq. 14 from LLM-FP4 applied per input channel.

    ch_max: (in_features,) — per-channel max abs activation from calibration
    e_bits: exponent bits of the FP format (2 for E2M1)
    m_bits: mantissa bits of the FP format (1 for E2M1)
    device: computation device

    Returns:
        rho:   scalar tensor — tensor-wise real-valued exponent
               used as the activation scale at inference: scale = 2^(-rho)
        b_ori: (in_features,) long tensor — per-channel integer bias
               constrained to [0, 2^e_bits - 1]
               values exceeding this range are clamped, residual absorbed in rho
        beta:  (in_features,) float tensor — 2^(-b_ori)
               multiply weight column j by beta[j] to absorb correction

    Mathematical basis (LLM-FP4 Eq. 14-16):
        b_tilde_j = 2^e - log2(max|x[:,j]|) + log2(2 - 2^-m) - 1
        rho       = min_j(b_tilde_j)
        b_ori_j   = clamp(round(b_tilde_j - rho), 0, 2^e - 1)

    After absorbing b_ori into weights:
        activation channel j uses scale 2^(-rho) instead of 2^(-b_tilde_j)
        the difference 2^(b_ori_j) is pre-baked into the weight column
    """
    ch_max = ch_max.to(device).clamp(min=1e-8)
    in_features = ch_max.shape[0]

    # Eq. 14: real-valued per-channel bias
    # This is the ideal per-channel exponent bias that would perfectly
    # normalize each channel's activation range to the FP codebook
    b_tilde = (float(2**e_bits)
               - torch.log2(ch_max)
               + math.log2(2.0 - 2.0**(-m_bits))
               - 1.0)                             # (in_features,)

    # Decompose into tensor-wise rho + per-channel integer correction
    # rho = minimum over all channels — ensures b_ori >= 0 everywhere
    rho = b_tilde.min()                           # scalar

    # b_ori is clamped to [0, 2^e_bits - 1] = [0, 3] for E2M1
    # Channels where b_tilde - rho > 3 get b_ori=3 (max correction)
    # The residual correction for those channels is an approximation
    # that the bias search in v5 will partially compensate
    b_ori = torch.clamp(
        torch.round(b_tilde - rho).long(),
        min=0,
        max=2**e_bits - 1
    )                                             # (in_features,) long

    # beta[j] = 2^(-b_ori[j]) — the per-channel weight scaling factor
    beta = 2.0 ** (-b_ori.float())               # (in_features,) float

    # Diagnostics
    clamped = ((b_tilde - rho) > (2**e_bits - 1)).sum().item()
    print(f"  b_tilde: [{b_tilde.min().item():.3f}, "
          f"{b_tilde.max().item():.3f}]")
    print(f"  rho = {rho.item():.4f}  "
          f"(activation scale = 2^(-rho) = {2.0**(-rho.item()):.4f})")
    print(f"  b_ori: [{b_ori.min().item()}, {b_ori.max().item()}]  "
          f"({clamped}/{in_features} channels clamped to max={2**e_bits-1})")
    print(f"  beta:  [{beta.min().item():.4f}, {beta.max().item():.4f}]")

    return rho, b_ori, beta


def absorb_perchannel_bias_into_weights(W, beta, device):
    """
    Absorb per-channel activation exponent correction into weight matrix.

    W:    (out_features, in_features) — original weight matrix
    beta: (in_features,)              — per-channel scale = 2^(-b_ori_j)
    
    Returns W_adjusted: (out_features, in_features)

    Mathematical equivalence:
        Original:  y = (x / diag(beta)) @ W^T    [activate then quantize]
        Adjusted:  y = x @ (W * beta)^T           [absorb into weights]
        
        These are identical because:
        (x / beta_j) * W[:,j] = x * (W[:,j] / beta_j) = x * W[:,j] * beta_j^{-1}
        
        Wait — beta = 2^(-b_ori) so dividing x by beta means multiplying x by 2^(b_ori)
        which shifts the activation up to fill the FP4 range.
        Equivalently, multiply W[:,j] by beta[j] = 2^(-b_ori_j) to shift weights down.
        The dot product x_j * W_adjusted_j = x_j * W_j * beta_j
        = (x_j * beta_j) * W_j  ✓  (same result as scaling activation up)

    After this transformation:
        - Activations at inference are quantized with scalar scale 2^(-rho)
        - The per-channel correction 2^(-b_ori_j) is already in the weights
        - No per-channel scale is needed at inference — zero overhead
    """
    W    = W.to(device).float()
    beta = beta.to(device).float()

    assert beta.shape[0] == W.shape[1], (
        f"beta shape {beta.shape} must match W in_features {W.shape[1]}"
    )

    # Multiply each input channel (column) of W by its correction factor
    # W[:, j] *= beta[j] for all j
    W_adjusted = W * beta.unsqueeze(0)            # (out, in) broadcast

    # Verify the adjustment did not introduce NaN or Inf
    if torch.isnan(W_adjusted).any() or torch.isinf(W_adjusted).any():
        print("  WARNING: NaN or Inf detected in W_adjusted — "
              "check beta values")

    # Print weight distribution change for diagnostics
    print(f"  W original:  max={W.abs().max():.4f}, "
          f"mean={W.abs().mean():.4f}")
    print(f"  W adjusted:  max={W_adjusted.abs().max():.4f}, "
          f"mean={W_adjusted.abs().mean():.4f}")

    return W_adjusted


def calibrate_model_decomposed(model, calib_loader, block_size, device,
                                e_bits=2, m_bits=1,
                                e_bits_scale=4, m_bits_scale=3,
                                outlier_threshold=6.0,
                                num_batches=4,
                                act_channel_max=None):
    model.eval().to(device)

    # ── Step 1: activation stats ──────────────────────────────────────────
    if act_channel_max is None:
        print("Collecting per-channel activation statistics...")
        act_channel_max = {}
        hooks = []

        def make_hook(name, in_features):
            def hook(module, inp, out):
                x      = inp[0].detach().float()
                x_flat = x.reshape(-1, in_features)
                ch_max = x_flat.abs().amax(dim=0)
                if name not in act_channel_max:
                    act_channel_max[name] = ch_max.cpu()
                else:
                    act_channel_max[name] = torch.maximum(
                        act_channel_max[name], ch_max.cpu()
                    )
            return hook

        n_hooks = 0
        for name, module in model.named_modules():
            if type(module).__name__ == "QuantLinearFP_Decomposed":
                in_f = module.linear.weight.shape[1]
                hooks.append(
                    module.register_forward_hook(make_hook(name, in_f))
                )
                n_hooks += 1

        print(f"Registered {n_hooks} hooks for activation collection")
        batches_run = 0
        with torch.no_grad():
            for batch in calib_loader:
                if batch is None:
                    continue
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                if x is None:
                    continue
                model(x.to(device))
                batches_run += 1
                if batches_run >= num_batches:
                    break
        for h in hooks:
            h.remove()
        print(f"Collected activation stats for {len(act_channel_max)} layers")

    # ── Step 2: Hessian on normal channels only ───────────────────────────
    print("Computing Hessian on normal channels...")
    H_dict      = {}
    hooks       = []
    batches_run = 0

    def make_hessian_hook(name, module):
        def hook(mod, inp, out):
            x      = inp[0].detach().float()
            x_flat = x.reshape(-1, x.shape[-1])           # (N_tokens, in_features)

            # Zero out outlier channels — the FP4 path at inference
            # will only see normal channels, so Hessian must match
            x_normal = x_flat[:, module.normal_indices.cpu()]  # (N, n_normal)

            N, D = x_normal.shape
            if D == 0:
                return

            H      = (x_normal.T @ x_normal) / N
            blocks = [H[i:min(i+block_size,D), i:min(i+block_size,D)].cpu()
                      for i in range(0, D, block_size)]

            if name not in H_dict:
                H_dict[name] = blocks
            else:
                for i in range(len(blocks)):
                    H_dict[name][i] += blocks[i]
        return hook

    for name, module in model.named_modules():
        if type(module).__name__ == "QuantLinearFP_Decomposed":
            hooks.append(
                module.register_forward_hook(make_hessian_hook(name, module))
            )

    print(f"Registered {len(hooks)} Hessian hooks")
    with torch.no_grad():
        for batch in calib_loader:
            if batch is None:
                continue
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if x is None:
                continue
            model(x.to(device))
            batches_run += 1
            if batches_run >= num_batches:
                break
    for h in hooks:
        h.remove()

    H_dict = {
        name: [b / batches_run for b in blocks]
        for name, blocks in H_dict.items()
    }
    print(f"Collected Hessians for {len(H_dict)} decomposed layers")

    # ── Step 3: reconstruct FP4 weights on normal channels ────────────────
    # for name, module in model.named_modules():
    #     if type(module).__name__ != "QuantLinearFP_Decomposed":
    #         continue
    #     if name not in H_dict:
    #         print(f"  Missing Hessian for {name}, skipping")
    #         continue
    #     if name not in act_channel_max:
    #         print(f"  Missing activation stats for {name}, skipping")
    #         continue

    #     print(f"Calibrating normal channels for {name}")

    #     normal_idx = module.normal_indices.cpu()
    #     n_normal   = normal_idx.shape[0]

    #     if n_normal == 0:
    #         print(f"  No normal channels for {name}, skipping")
    #         continue

    #     # Extract normal-channel weights: (out_features, n_normal)
    #     W_normal = module.linear.weight[:, normal_idx].to(device).float()

    #     # Per-channel pre-shifted bias on normal channels
    #     ch_max_normal = act_channel_max[name][normal_idx]

    #     rho, b_ori, beta = preshifted_bias_perchannel(
    #         ch_max_normal, e_bits, m_bits, device
    #     )
    #     W_adjusted = absorb_perchannel_bias_into_weights(W_normal, beta, device)

    #     preshifted_center = int(
    #         torch.round(rho).clamp(0, 2**e_bits - 1).item()
    #     )

    #     # ── Key fix: create a proper nn.Linear surrogate ──────────────────
    #     # v5 calls layer.weight.data, layer.weight.data.dim(), layer.weight.data.shape
    #     # and layer.weight.data.to(device) — we need a real nn.Linear
    #     surrogate          = nn.Linear(n_normal, W_normal.shape[0], bias=False)
    #     surrogate.weight.data = W_adjusted
    #     surrogate          = surrogate.to(device)

    #     res = reconstruct_layer_fp_blockdiag_scaled_v5_preshifted(
    #         surrogate,
    #         H_dict[name],
    #         block_size,
    #         e_bits, m_bits,
    #         e_bits_scale, m_bits_scale,
    #         device,
    #         preshifted_center=preshifted_center,
    #     )

    #     # weight_q shape: (out_features, n_normal) — correct for F.linear
    #     # Undo beta scaling so weight_q is in original weight scale
    #     # beta = 2^(-b_ori) was multiplied into weights before reconstruction
    #     # to guide the quantization search — we undo it here so the stored
    #     # weight_q can be used directly with unscaled activations (A16 path)
    #     # and also with preshifted activations (A4 path, where rho handles
    #     # the tensor-wise scale and b_ori benefit is captured in weight quality)
    #     beta_device       = beta.to(device).float()
    #     weight_q_raw      = res["weight_q"].to(device).float()
    #     weight_q_unscaled = weight_q_raw / beta_device.unsqueeze(0)

    #     module.weight_q       = weight_q_unscaled
    #     module.alpha_q        = res["alpha"]
    #     module.bias_q         = res["bias"]
    #     module.act_rho        = rho.item()
    #     module.act_b_ori      = b_ori.cpu()
    #     module.act_quant_mode = "preshifted"
    #     module.act_block_size = block_size

    #     del surrogate, W_normal, W_adjusted
    #     torch.cuda.empty_cache()
    for name, module in model.named_modules():
        if type(module).__name__ != "QuantLinearFP_Decomposed":
            continue
        if name not in H_dict:
            print(f"  Missing Hessian for {name}, skipping")
            continue

        print(f"Calibrating normal channels for {name}")

        normal_idx = module.normal_indices.cpu()
        n_normal   = normal_idx.shape[0]

        if n_normal == 0:
            print(f"  No normal channels for {name}, skipping")
            continue

        # Extract weights for normal channels only
        W_normal = module.linear.weight[:, normal_idx].to(device).float()

        # Create surrogate linear for v5
        surrogate             = nn.Linear(n_normal, W_normal.shape[0], bias=False)
        surrogate.weight.data = W_normal.clone()
        surrogate             = surrogate.to(device)

        # Standard v5 reconstruction — no pre-shifted bias
        preshifted_center = 2**(e_bits - 1) - 1   # default center = 1 for E2M1

        res = reconstruct_layer_fp_blockdiag_scaled_v5_preshifted(
            surrogate,
            H_dict[name],
            block_size,
            e_bits, m_bits,
            e_bits_scale, m_bits_scale,
            device,
            preshifted_center=preshifted_center,
        )
        ch_max_normal = act_channel_max[name][normal_idx]
        act_max       = ch_max_normal.max().item()
        act_max       = max(act_max, 1e-8)   # prevent log2(0) or log2(negative)
        act_rho       = math.log2(act_max) - math.log2(2.0 - 2.0**(-m_bits)) + 1.0
        act_rho       = max(0.0, act_rho)
        # Store weight_q directly — no beta scaling involved
        module.weight_q       = res["weight_q"].to(device)
        module.alpha_q        = res["alpha"]
        module.bias_q         = res["bias"]
        module.act_quant_mode = None        # disable activation quantization for now
        module.act_rho        = act_rho
        module.act_block_size = block_size

        del surrogate, W_normal
        torch.cuda.empty_cache()
    return model

def calibrate_model_Hessian_Hadamard(model, data_loader, block_size, device):
    model.eval().to(device)
    H_dict_block = compute_hessian_blockdiag_model(
        model, data_loader, device, block_size
    )  # now keyed by name string e.g. "transformer.h.0.attn.c_attn"

    for name, module in model.named_modules():  # use named_modules to get name
        if isinstance(module, (QuantLinearFP, QuantConv2dFP, QuantConv1dFP)):
            if name not in H_dict_block:
                print(f"⚠️ Missing Hessian for {name}, skipping")
                continue
            H_block = H_dict_block[name]
            print(f"Calibrating {name} with Hadamard FP4")
            module.calibrate_Hessian_Hadamard(data_loader, device, H_block)
            torch.cuda.empty_cache()
    return model

def calibrate_model_HG(model, data_loader, device="cuda"):
    model.eval().to(device)
    # H_dict_block = compute_hessian_blockdiag_model(model, data_loader, device)
    H_dict = compute_hessian_diag_model(model, data_loader, device)
    print(H_dict)
    for keys in H_dict.keys():
        print(keys)
    for module in model.modules():
        if isinstance(module, (QuantLinearFP, QuantConv2dFP)):
            # Determine the underlying weight module
            if hasattr(module, "linear") and module.linear in H_dict:
                H_diag = H_dict[module.linear]
            elif hasattr(module, "conv") and module.conv in H_dict:
                H_diag = H_dict[module.conv]
            else:
                print("⚠️ Missing Hessian for", module)
                continue

            module.calibrate_Hessian(data_loader, device, H_diag)

    return model

def calibrate_model_Hessian_block(model, data_loader, block_size, device):
    model.eval().to(device)
    H_dict_block = compute_hessian_blockdiag_model(model, data_loader, device, block_size)
    # H_dict_block = compute_hessian_diag_model(model, data_loader, device)
    for module in model.modules():
        if isinstance(module, (QuantLinearFP, QuantConv2dFP)):
            # Determine the underlying weight module
            if hasattr(module, "linear") and module.linear in H_dict_block:
                H_diag = H_dict_block[module.linear]
            elif hasattr(module, "conv") and module.conv in H_dict_block:
                H_diag = H_dict_block[module.conv]
            else:
                print("⚠️ Missing Hessian for", module)
                continue

            module.calibrate_Hessian_block(data_loader, device, H_diag)

    return model



def calibrate_model_Hessian_whitened(model, data_loader, block_size, device):
    model.eval().to(device)

    H_dict_block = compute_hessian_blockdiag_model(
        model, data_loader, device, block_size
    )

    for module in model.modules():
        if isinstance(module, (QuantLinearFP, QuantConv2dFP)):

            if hasattr(module, "linear") and module.linear in H_dict_block:
                H_block = H_dict_block[module.linear]
            elif hasattr(module, "conv") and module.conv in H_dict_block:
                H_block = H_dict_block[module.conv]
            else:
                continue

            module.calibrate_Hessian_whitened(data_loader, device, H_block)

    return model


def calibrate_model_joint(model, calib_loader, block_size, device,
                           e_bits=2, m_bits=1,
                           e_bits_scale=4, m_bits_scale=3,
                           num_batches=4,
                           damping=0.01,
                           apply_correction=True):
    """
    Joint weight+activation quantization calibration.

    For each QuantLinearFP layer:
      1. Collect clean calibration activations X
      2. Quantize activations to NVFP4 -> X_hat  (matches act_quant_mode="nvfp4")
      3. Recompute H_joint = (1/T) X_hat^T X_hat  blockwise
      4. Compute W' = W + W @ dX^T @ X_hat @ inv(X_hat^T X_hat + lI)  blockwise
      5. Run v6 block optimization on W' with H_joint
      6. Store weight_q, set act_quant_mode="nvfp4" so forward uses FP4 acts
    """
    model.eval().to(device)

    # ----------------------------------------------------------------
    # Phase 1 — collect raw activations for all layers in one pass
    # ----------------------------------------------------------------
    print("Collecting calibration activations...")
    X_dict = collect_calibration_activations(
        model, calib_loader, device, block_size, num_batches
    )

    # ----------------------------------------------------------------
    # Phase 2 — per-layer calibration
    # ----------------------------------------------------------------
    for name, module in model.named_modules():
        if type(module).__name__ != "QuantLinearFP":
            continue
        if name not in X_dict:
            print(f"  No activations for {name}, skipping")
            continue

        print(f"Calibrating {name}")

        X_calib = X_dict[name].float()           # [T, M] CPU
        W_orig  = module.linear.weight.data.to(device).float()

        if W_orig.dim() == 4:
            W_mat = W_orig.view(W_orig.shape[0], -1).cpu()
        else:
            W_mat = W_orig.cpu()

        N, M = W_mat.shape
        T    = X_calib.shape[0]

        # Verify shapes align
        assert X_calib.shape[1] == M, (
            f"{name}: activation features {X_calib.shape[1]} "
            f"!= weight in_features {M}"
        )

        # ── Step 1: quantize activations exactly as nvfp4 forward does ──
        print(f"  Quantizing activations [{T} x {M}]...")
        X_hat = quantize_activations(
            X_calib.to(device),
            block_size, e_bits, m_bits,
            e_bits_scale=e_bits_scale,
            m_bits_scale=m_bits_scale
        ).cpu().float()                           # [T, M] CPU

        delta_X_norm = (X_calib - X_hat).norm() / X_calib.norm().clamp(min=1e-8)
        print(f"  ||delta_X|| / ||X|| = {delta_X_norm:.4f}")

        # ── Step 2: recompute H from X_hat ──────────────────────────────
        print(f"  Recomputing H_joint from X_hat...")
        H_blocks_joint = recompute_H_blocks_from_Xhat(
            X_hat, block_size, device
        )

        # ── Step 3: compute weight correction ───────────────────────────
        if apply_correction:
            print(f"  Computing weight correction...")
            W_corrected = compute_W_correction_blockwise(
                W_mat, X_calib, X_hat,
                block_size, damping, device
            )
            # Sanity check — correction should be small
            corr_norm = (W_corrected - W_mat).norm() / W_mat.norm().clamp(min=1e-8)
            print(f"  ||W_correction|| / ||W|| = {corr_norm:.4f}")
        else:
            W_corrected = W_mat

        # Free X tensors — no longer needed after H and correction computed
        del X_hat, X_calib
        torch.cuda.empty_cache()

        # ── Step 4: reconstruct with v6 ──────────────────────────────────
        # Temporarily swap corrected weights into layer so v6 can read
        # layer.weight.data (it uses this for shape/dim checks)
        original_weight           = module.linear.weight.data.clone()
        module.linear.weight.data = W_corrected.view_as(original_weight).to(device)

        res = reconstruct_layer_fp_blockdiag_scaled_v6(
            module.linear,
            H_blocks_joint,
            W_corrected,
            block_size,
            e_bits, m_bits,
            e_bits_scale, m_bits_scale,
            device,
        )

        # Restore original weights — weight_q holds the quantized version
        module.linear.weight.data = original_weight

        # ── Step 5: store results on module ─────────────────────────────
        module.weight_q       = res["weight_q"].view_as(module.linear.weight).to(device)
        module.act_quant_mode = "nvfp4"   # forward will now quantize acts
        module.act_block_size = block_size

        # ── Step 6: per-layer sanity check ───────────────────────────────
        with torch.no_grad():
            in_f   = W_orig.shape[1] if W_orig.dim() == 2 else W_orig.view(W_orig.shape[0], -1).shape[1]
            x_test = torch.randn(4, 16, in_f, device=device)

            # Simulate what forward() will do with act_quant_mode="nvfp4"
            x_q_test = quantize_activations(
                x_test, block_size, e_bits, m_bits,
                e_bits_scale=e_bits_scale, m_bits_scale=m_bits_scale
            ).to(device)

            out_orig  = F.linear(x_test,   W_orig.to(x_test.dtype))
            out_quant = F.linear(x_q_test, module.weight_q.to(x_test.dtype))

            rel_err = (out_orig - out_quant).abs().mean() / \
                      out_orig.abs().mean().clamp(min=1e-8)
            print(f"  rel_err (W4A4): {rel_err.item():.4f}  "
                  f"out_max={out_orig.abs().max():.3f}  "
                  f"quant_max={out_quant.abs().max():.3f}")

        del H_blocks_joint, W_corrected, W_mat, W_orig, res
        torch.cuda.empty_cache()

    return model

def collect_layer_inputs(model, data_loader, device, num_batches=8):
    model.eval()

    def hook_fn(module, inp, out):
        module._cached_input = inp[0].detach()

    hooks = []

    # ✅ hook WRAPPERS, not internal layers
    for module in model.modules():
        if isinstance(module, (QuantLinearFP, QuantConv2dFP)):
            hooks.append(module.register_forward_hook(hook_fn))

    # run data
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            x = x.to(device)
            model(x)
            if i >= num_batches:
                break

    # remove hooks
    for h in hooks:
        h.remove()

    return None 

def calibrate_model_Hessian_scaled_forward(model, data_loader, block_size, device):
    model.eval().to(device)

    print("Collecting activations...")
    collect_layer_inputs(model, data_loader, device)  # no return

    print("Computing Hessian...")
    H_dict_block = compute_hessian_blockdiag_model(
        model, data_loader, device, block_size
    )

    for module in model.modules():
        if isinstance(module, (QuantLinearFP, QuantConv2dFP)):

            if not hasattr(module, "_cached_input"):
                print(f"⚠️ Missing activation for {module}, skipping")
                continue

            cached_input = module._cached_input

            # underlying layer
            if hasattr(module, "linear") and module.linear in H_dict_block:
                weight_layer = module.linear
            elif hasattr(module, "conv") and module.conv in H_dict_block:
                weight_layer = module.conv
            else:
                continue

            print(f"Calibrating {module}")

            # alpha, e, m, sign, bias = reconstruct_layer_fp_blockdiag_scaled_v4_forward(
            #     weight_layer,
            #     H_dict_block[weight_layer],
            #     module.block_size,
            #     module.e_bits,
            #     module.m_bits,
            #     module.e_bits_scale,
            #     module.m_bits_scale,
            #     device,
            #     cached_input=cached_input
            # )
            alpha, e, m, sign, bias =reconstruct_layer_fp_blockdiag_scaled_v4_fast(                
                weight_layer,
                H_dict_block[weight_layer],
                module.block_size,
                module.e_bits,
                module.m_bits,
                module.e_bits_scale,
                module.m_bits_scale,
                device)
            base_val = alpha * (2.0 ** (e.float() - bias))
            fine = base_val * m.float() / (2 ** module.m_bits) if module.m_bits > 0 else 0.0
            module.weight_q = (base_val + fine) * sign

    return model


def fold_bn_into_conv(conv, bn):
    """
    Fold BatchNorm into Conv2d
    """
    W = conv.weight.data
    if conv.bias is None:
        bias = torch.zeros(W.size(0), device=W.device)
    else:
        bias = conv.bias.data

    gamma = bn.weight.data
    beta = bn.bias.data
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    std = torch.sqrt(var + eps)

    # reshape for broadcasting
    gamma = gamma.view(-1, 1, 1, 1)
    std = std.view(-1, 1, 1, 1)

    W_new = W * (gamma / std)

    bias_new = (bias - mean) / std.view(-1) * bn.weight.data + beta

    conv.weight.data = W_new
    conv.bias = torch.nn.Parameter(bias_new)

    return conv

def fold_bn_recursively(model):
    prev_name = None
    prev_module = None

    for name, module in list(model.named_children()):
        if isinstance(module, nn.BatchNorm2d) and isinstance(prev_module, nn.Conv2d):
            fused_conv = fold_bn_into_conv(prev_module, module)
            setattr(model, prev_name, fused_conv)
            setattr(model, name, nn.Identity())
        else:
            fold_bn_recursively(module)

        prev_name = name
        prev_module = module

    return model
# =========================================================
# 🔹 APPLY PRUNING MASK
# =========================================================
def apply_pruning_mask(model):
    for module in model.modules():
        if isinstance(module, (QuantLinearFP, QuantConv2dFP)):
            orig_weights = module.linear.weight if isinstance(module, QuantLinearFP) else module.conv.weight
            mask = (orig_weights != 0).float()
            if module.weight_q is not None:
                # weight_q may live in a padded Hadamard domain [N, P>=M]; pad the
                # original-space sparsity mask with ONES so padded/rotated columns
                # are preserved.  For dense models the mask is all-ones → no-op.
                if mask.shape[-1] < module.weight_q.shape[-1]:
                    mask = F.pad(
                        mask, (0, module.weight_q.shape[-1] - mask.shape[-1]),
                        value=1.0,
                    )
                module.weight_q *= mask.to(device=module.weight_q.device,
                                           dtype=module.weight_q.dtype)


def compute_expected_fp4_weight(W_mat, block_size, e_bits, m_bits,
                                 e_bits_scale, m_bits_scale,
                                 n_samples=32, device="cuda"):
    """
    Estimate E[Q(W)] by averaging over multiple stochastic roundings.

    Standard nearest-neighbour quantization always picks the closest
    FP4 representable value. Weights near a quantization boundary
    suffer systematic bias — they are always rounded the same way.

    Stochastic rounding adds small noise before each quantization so
    boundary weights randomly round up or down. The mean over many
    samples approximates the optimal bias-free target E[Q(W)].

    Key property: E[Q_stochastic(W)] = W for weights between two FP4
    neighbours, meaning the expected quantized weight has zero bias.
    The best single FP4 weight to use in practice is the one closest
    to E[Q(W)] under the Hessian metric, not closest to W itself.

    Args:
        W_mat:          [N, M] float32 weight matrix
        block_size:     FP4 quantization block size
        n_samples:      number of stochastic rounding samples
        device:         compute device

    Returns:
        W_expected:     [N, M] float32 — E[Q(W)] estimate
        W_std:          [N, M] float32 — std of quantized samples
                        (diagnostic: large std = high boundary uncertainty)
    """
    N, M    = W_mat.shape
    W_sum   = torch.zeros(N, M, dtype=torch.float32, device=device)
    W_sum_sq = torch.zeros(N, M, dtype=torch.float32, device=device)
    W_dev   = W_mat.to(device).float()

    # Estimate the FP4 quantization step size per block for noise scaling.
    # Noise should be ~half the local quantization gap so it moves weights
    # across boundaries but does not change them by more than one step.
    # We use a rough estimate: median absolute weight / Qmax * step_fraction
    Qmax       = compute_Qmax(e_bits, m_bits, bias=2 ** (e_bits - 1) - 1)
    w_scale    = W_dev.abs().median().clamp(min=1e-8)
    noise_std  = (w_scale / Qmax) * 0.5   # half the typical quantization gap

    print(f"    E[Q(W)] estimation: "
          f"n_samples={n_samples}, noise_std={noise_std.item():.6f}")

    for sample_idx in range(n_samples):
        # Add small Gaussian noise to push boundary weights stochastically
        # toward one of their two nearest FP4 neighbours
        noise   = torch.randn(N, M, device=device) * noise_std
        W_noisy = W_dev + noise

        # Quantize noisy weights with standard nearest-neighbour FP4
        # We use a simplified per-block scale (no Hessian) here because
        # this is estimating the distribution, not the optimal single point
        W_q = _simple_fp4_quantize_blocks(
            W_noisy, block_size, e_bits, m_bits,
            e_bits_scale, m_bits_scale, device
        )

        W_sum    += W_q
        W_sum_sq += W_q ** 2

        del noise, W_noisy, W_q
        if sample_idx % 8 == 0:
            torch.cuda.empty_cache()

    W_expected = W_sum   / n_samples
    W_var      = W_sum_sq / n_samples - W_expected ** 2
    W_std      = W_var.clamp(min=0).sqrt()

    print(f"    E[Q(W)] max={W_expected.abs().max():.4f}  "
          f"mean={W_expected.abs().mean():.6f}")
    print(f"    std max={W_std.max():.4f}  "
          f"mean={W_std.mean():.6f}  "
          f"(large std = high boundary uncertainty)")

    # Diagnostic: fraction of weights with high uncertainty
    # These are the weights near FP4 boundaries that benefit most
    high_uncertainty = (W_std > noise_std * 0.5).float().mean().item()
    print(f"    High-uncertainty weights: {high_uncertainty:.1%}")

    del W_sum, W_sum_sq
    torch.cuda.empty_cache()

    return W_expected.cpu(), W_std.cpu()


def _simple_fp4_quantize_blocks(W, block_size, e_bits, m_bits,
                                  e_bits_scale, m_bits_scale, device):
    """
    Fast per-block FP4 quantization using RMS-based scale.
    Used inside compute_expected_fp4_weight for speed — no Hessian,
    no bias search, just clean nearest-neighbour quantization.

    Returns W_q: [N, M] float32 reconstructed weights.
    """
    N, M   = W.shape
    W_q    = torch.zeros_like(W)
    default_bias = 2 ** (e_bits - 1) - 1

    for i in range(0, M, block_size):
        end     = min(i + block_size, M)
        w_block = W[:, i:end]                          # [N, k]

        # RMS-based scale — fast, no iteration needed
        w_rms   = w_block.pow(2).mean(dim=1).sqrt().clamp(min=1e-8)  # [N]
        alpha   = quantize_scale_batched(w_rms, e_bits_scale, m_bits_scale)

        # Nearest-neighbour FP4 assignment
        _, _, b = assign_fp4_dynamic_batched(
            w_block.abs(), alpha, e_bits, m_bits, bias=default_bias
        )
        sign          = w_block.sign()
        W_q[:, i:end] = alpha.unsqueeze(1) * b * sign

        del w_block, w_rms, alpha, b, sign

    return W_q


def find_best_fp4_for_expected(W_expected, W_std, H_blocks,
                                block_size, e_bits, m_bits,
                                e_bits_scale, m_bits_scale, device):
    """
    Find the single FP4 weight matrix closest to W_expected under
    the Hessian metric.

    This is subtly different from quantizing W_orig directly:
      - Standard:    argmin_{Q(W)} ||W_orig - Q(W)||_H
      - This:        argmin_{Q(W)} ||E[Q(W)] - Q(W)||_H

    The difference matters for weights near FP4 boundaries.
    W_std gives a confidence map — high std weights are near boundaries
    and benefit most from Hessian-guided rounding toward E[Q(W)].

    We use a modified v5 reconstruction that:
      1. Initialises alpha from W_expected rather than W_orig
         (better starting point for boundary weights)
      2. Weights the Hessian loss by W_std (focus optimisation effort
         on high-uncertainty weights)
      3. Returns both the reconstruction and boundary statistics

    Args:
        W_expected: [N, M] float32 — E[Q(W)] from stochastic sampling
        W_std:      [N, M] float32 — per-weight quantization uncertainty
        H_blocks:   list of [k, k] Hessian blocks

    Returns:
        dict with weight_q, alpha, bias, boundary_stats
    """
    N, M = W_expected.shape

    # Build a temporary Linear module wrapping W_expected so we can
    # pass it to reconstruct_layer_fp_blockdiag_scaled_v5 unchanged
    tmp_layer             = torch.nn.Linear(M, N, bias=False, device=device)
    tmp_layer.weight.data = W_expected.to(device)

    # Run v5 reconstruction targeting W_expected
    # The Hessian H_blocks guides which weight perturbations matter most,
    # and the starting point is W_expected rather than W_orig, so boundary
    # weights get initialised to their unbiased expected value
    res = reconstruct_layer_fp_blockdiag_scaled_v5(
        tmp_layer, H_blocks, block_size,
        e_bits, m_bits, e_bits_scale, m_bits_scale,
        device
    )
    del tmp_layer
    torch.cuda.empty_cache()

    # Compute boundary statistics for diagnostics
    W_q         = res["weight_q"].float()
    boundary_gap = (W_expected.to(device) - W_q).abs()
    high_std_mask = (W_std.to(device) > W_std.to(device).median())

    boundary_stats = {
        "mean_err_all":        boundary_gap.mean().item(),
        "mean_err_boundary":   boundary_gap[high_std_mask].mean().item(),
        "mean_err_interior":   boundary_gap[~high_std_mask].mean().item(),
        "pct_boundary":        high_std_mask.float().mean().item(),
    }

    del W_q, boundary_gap, high_std_mask
    torch.cuda.empty_cache()

    res["boundary_stats"] = boundary_stats
    return res


def calibrate_layer_stochastic_fp4(layer_name, W_orig, H_blocks,
                                    block_size, e_bits, m_bits,
                                    e_bits_scale, m_bits_scale,
                                    device, n_samples=32,
                                    compare_standard=True):
    """
    Full stochastic FP4 calibration for one layer.

    Pipeline:
      1. Standard v5 quantization of W_orig (baseline)
      2. Estimate E[Q(W)] via stochastic rounding
      3. Quantize toward E[Q(W)] under Hessian metric
      4. Compare output errors and return the better result

    Args:
        layer_name:      string for logging
        W_orig:          [N, M] float32 original weights
        H_blocks:        Hessian blocks for this layer
        compare_standard: if True, also run standard v5 and report both

    Returns:
        dict with weight_q, alpha, bias, method ('standard' or 'stochastic')
    """
    N, M = W_orig.shape
    print(f"\n  === Stochastic FP4 calibration: {layer_name} ===")

    # ── Step 1: standard v5 baseline ─────────────────────────────────
    if compare_standard:
        print(f"  [1/4] Standard v5 quantization (baseline)...")
        tmp_standard             = torch.nn.Linear(M, N, bias=False,
                                                   device=device)
        tmp_standard.weight.data = W_orig.to(device)
        res_standard = reconstruct_layer_fp_blockdiag_scaled_v5(
            tmp_standard, H_blocks, block_size,
            e_bits, m_bits, e_bits_scale, m_bits_scale, device
        )
        del tmp_standard
        torch.cuda.empty_cache()

        # The stochastic E[Q(W)] path (steps 2-4) has been rejected on every
        # layer of every model run (improvement consistently ~-64%), and its
        # n_samples-fold weight replication is what OOMs the padded FFN layers
        # (e.g. 2560x16384 fc1) even on an 80GB H100.  Short-circuit to the
        # standard v5 baseline: identical result, ~half the calibration time,
        # no OOM.  Set compare_standard=False to force the stochastic path.
        res_standard["method"]         = "standard"
        res_standard["err_standard"]   = float("nan")
        res_standard["err_stochastic"] = float("nan")
        return res_standard

    # ── Step 2: estimate E[Q(W)] ──────────────────────────────────────
    print(f"  [2/4] Estimating E[Q(W)] with {n_samples} samples...")
    W_expected, W_std = compute_expected_fp4_weight(
        W_orig, block_size, e_bits, m_bits,
        e_bits_scale, m_bits_scale,
        n_samples=n_samples, device=device
    )

    # ── Step 3: quantize toward E[Q(W)] ──────────────────────────────
    print(f"  [3/4] Quantizing toward E[Q(W)] under Hessian metric...")
    res_stochastic = find_best_fp4_for_expected(
        W_expected, W_std, H_blocks,
        block_size, e_bits, m_bits, e_bits_scale, m_bits_scale, device
    )

    bs = res_stochastic["boundary_stats"]
    print(f"  Boundary stats:")
    print(f"    mean_err all:      {bs['mean_err_all']:.6f}")
    print(f"    mean_err boundary: {bs['mean_err_boundary']:.6f}")
    print(f"    mean_err interior: {bs['mean_err_interior']:.6f}")
    print(f"    pct boundary:      {bs['pct_boundary']:.1%}")

    # ── Step 4: compare output errors ────────────────────────────────
    print(f"  [4/4] Comparing output errors...")
    x_test   = torch.randn(64, M, device=device) * W_orig.abs().mean() * 10
    out_orig = F.linear(x_test, W_orig.to(device))

    err_stochastic = (out_orig - F.linear(
        x_test, res_stochastic["weight_q"].to(device)
    )).norm() / out_orig.norm()

    print(f"  rel_err stochastic: {err_stochastic:.4f}")

    if compare_standard:
        err_standard = (out_orig - F.linear(
            x_test, res_standard["weight_q"].to(device)
        )).norm() / out_orig.norm()
        print(f"  rel_err standard:   {err_standard:.4f}")
        improvement = (err_standard - err_stochastic) / err_standard * 100
        print(f"  improvement:        {improvement:+.1f}%")

        # Return whichever is better
        if err_stochastic <= err_standard:
            print(f"  → using stochastic")
            res_stochastic["method"] = "stochastic"
            result = res_stochastic
        else:
            print(f"  → using standard (stochastic did not improve)")
            res_standard["method"] = "standard"
            result = res_standard

        result["err_standard"]   = err_standard.item()
        result["err_stochastic"] = err_stochastic.item()
    else:
        res_stochastic["method"] = "stochastic"
        result = res_stochastic

    del x_test, out_orig
    torch.cuda.empty_cache()
    return result


def calibrate_model_stochastic_fp4(
    model,
    calib_loader,
    block_size,
    device,
    e_bits=2,
    m_bits=1,
    e_bits_scale=4,
    m_bits_scale=3,
    num_batches=4,
    n_samples=32,
    had_block_size=256,
    randomize_hadamard=True,
    hadamard_seed=None,
    compare_standard=True,
    extra_skip_patterns=(),
    lean=False,
):
    """
    Full model calibration using stochastic FP4 quantization.

    lean=False (default) is the original behavior — byte-for-byte unchanged.
    lean=True frees each layer's ORIGINAL weight the moment weight_q is stored,
    so the GPU never holds both (originals + weight_q ~= 2x model). This halves
    the resident weight-quant peak (~2x -> ~1x model), letting 6.7B/7B fit a 24GB
    GPU. The forward uses weight_q; the original is only a fallback (weight_q is
    set post-calibration) — the post-calib sanity check is skipped in lean mode.

    Replaces the standard v5 weight quantization step with the
    stochastic rounding approach: instead of quantizing W_orig
    directly, we estimate E[Q(W)] and quantize toward that target.

    The hypothesis: boundary weights (those near FP4 quantization
    boundaries) have lower expected error when quantized toward
    E[Q(W)] than toward W_orig, because E[Q(W)] is the unbiased
    estimator of the optimal quantization under stochastic rounding.

    Pipeline:
      1. Wrap all layers with HadamardQuantLinearFP
      2. Generate D for every layer upfront, store on modules
      3. Compute H_had = E[H(D*x) H(D*x)^T] in the Hadamard domain (single pass)
      Per layer:
      4. W_had = H(W*D); stochastic calibration with H_had
      5. Compute μ and bias_correction
      6. Store weight_q and act_quant_mode

    Args:
        n_samples:        stochastic rounding samples per layer
                          (32 is good balance of speed vs accuracy,
                           64+ for publication-quality results)
        compare_standard: log comparison vs standard v5 per layer
    """
    model.eval().to(device)

    SKIP_LAYERS = {"lm_head", "embed_tokens", "embed_positions"}
    SKIP_LAYERS |= set(extra_skip_patterns)
    if extra_skip_patterns:
        print(f"  Extra skip patterns (kept in FP16): "
              f"{sorted(extra_skip_patterns)}")

    # ── Phase 1: wrap layers ──────────────────────────────────────────
    print("Wrapping layers with HadamardQuantLinearFP...")
    model = wrap_layers_with_hadamard(model)
    n_wrapped = sum(1 for _, m in model.named_modules()
                    if type(m).__name__ == "HadamardQuantLinearFP")
    print(f"  Wrapped: {n_wrapped} layers")

    # ── Phase 2: generate D for ALL layers upfront ───────────────────
    # D must be stored before Phase 3 so the H_had hooks can apply H(D*x).
    # had_block_size="auto" → use the full row length M for each layer.
    print("Generating Hadamard sign vectors for all layers...")
    D_dict: dict[str, tuple] = {}  # name -> (D, actual_had_bs)
    for name, module in model.named_modules():
        if type(module).__name__ != "HadamardQuantLinearFP":
            continue
        if any(skip in name for skip in SKIP_LAYERS):
            continue
        W_shape = module.inner.linear.weight.data.shape
        M = W_shape[1] if len(W_shape) == 2 else W_shape[1] * W_shape[2] * W_shape[3]
        # "auto" → pad M up to the next power of 2 and apply ONE full-row
        # Hadamard.  For power-of-2 M this is a no-op pad (P == M) identical to
        # the old full-row behaviour; for non-power-of-2 M (e.g. OPT-2.7b
        # hidden=2560 → 4096) it replaces the broken block-diagonal M&-M
        # rotation with a single padded full Hadamard.  D is sized to the
        # padded width; downstream rotations pad x/W to len(D) via _rotate.
        if had_block_size == "auto":
            actual_had_bs = _next_pow2(M)        # full padded Hadamard
            d_len         = actual_had_bs
        else:
            actual_had_bs = had_block_size       # explicit block-diagonal
            assert M % actual_had_bs == 0, \
                f"{name}: M={M} not divisible by had_block_size={actual_had_bs}"
            d_len = M
        assert (actual_had_bs & (actual_had_bs - 1)) == 0, \
            f"{name}: had block size {actual_had_bs} must be a power of 2"
        if randomize_hadamard:
            seed = hadamard_seed if hadamard_seed is not None \
                   else _stable_seed(name)
            D = generate_random_signs(d_len, actual_had_bs, device, seed=seed)
        else:
            D = torch.ones(d_len, device=device)
        module.D              = D.cpu()
        module.had_block_size = actual_had_bs
        module.had_in_dim     = M
        D_dict[name]          = (D, actual_had_bs)
    print(f"  Generated D for {len(D_dict)} layers")

    # ── Phase 3: compute Hessian in Hadamard domain ───────────────────
    # H_had = E[H(D*x) H(D*x)^T] — the correct metric for quantizing W_had.
    # Computed here, after D is set on all modules, in a single forward pass.
    print("Computing Hessian in Hadamard domain...")
    H_dict_had = compute_hessian_hadamard_domain(
        model, calib_loader, device, block_size, num_batches=num_batches
    )

    # Accumulate per-layer comparison stats for summary
    comparison_log = {}

    # ── Phase 4: per-layer calibration ────────────────────────────────
    for name, module in model.named_modules():
        if type(module).__name__ != "HadamardQuantLinearFP":
            continue
        if any(skip in name for skip in SKIP_LAYERS):
            print(f"  Skipping {name}")
            continue
        if name not in H_dict_had:
            print(f"  No H_had for {name}, skipping")
            continue

        print(f"\nCalibrating {name}")

        W_orig = module.inner.linear.weight.data.to(device).float()
        if W_orig.dim() == 4:
            W_mat = W_orig.view(W_orig.shape[0], -1)
        else:
            W_mat = W_orig

        N, M = W_mat.shape

        # D was generated and stored on the module in Phase 2
        D, actual_had_bs = D_dict[name]

        # ── Step 4b: W_had = H(W * D) ────────────────────────────────
        # _rotate zero-pads W_mat's columns up to len(D) before the FWHT,
        # so W_had is [N, P] in the padded Hadamard domain.
        W_had = _rotate(W_mat, D, actual_had_bs).cpu().float()

        W_orig_max = W_mat.abs().max().item()
        W_had_max  = W_had.abs().max().item()
        print(f"  W_orig max: {W_orig_max:.4f}, "
              f"W_had max: {W_had_max:.4f} "
              f"(ratio {W_had_max / W_orig_max:.3f})")

        # ── Step 4c: activation diagnostic ───────────────────────────
        act_sample = None

        def _grab_raw(mod, inp, out):
            nonlocal act_sample
            if act_sample is None:
                act_sample = inp[0].detach().float().reshape(
                    -1, inp[0].shape[-1]).cpu()

        h_raw = module.register_forward_hook(_grab_raw)
        with torch.no_grad():
            for batch in calib_loader:
                if batch is None:
                    continue
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                if x is None:
                    continue
                model(x.to(device))
                break
        h_raw.remove()

        if act_sample is not None:
            X_had_d = _rotate(
                act_sample.to(device), D, actual_had_bs
            ).cpu().float()
            X_hat_d = quantize_activations(
                X_had_d.to(device), block_size, e_bits, m_bits,
                e_bits_scale=e_bits_scale, m_bits_scale=m_bits_scale
            ).cpu().float()
            orig_max     = act_sample.abs().amax().item()
            had_max      = X_had_d.abs().amax().item()
            delta_X_norm = (X_had_d - X_hat_d).norm() / \
                           X_had_d.norm().clamp(min=1e-8)
            print(f"  act_max: {orig_max:.2f} -> {had_max:.2f} "
                  f"(ratio {had_max / orig_max:.3f})")
            print(f"  ||delta_X_had|| / ||X_had|| = {delta_X_norm:.4f}")
            del X_had_d, X_hat_d, act_sample
            torch.cuda.empty_cache()

        # ── Step 4d: stochastic FP4 calibration on W_had ─────────────
        res = calibrate_layer_stochastic_fp4(
            layer_name      = name,
            W_orig          = W_had,          # targeting W_had not W_orig
            H_blocks        = H_dict_had[name],
            block_size      = block_size,
            e_bits          = e_bits,
            m_bits          = m_bits,
            e_bits_scale    = e_bits_scale,
            m_bits_scale    = m_bits_scale,
            device          = device,
            n_samples       = n_samples,
            compare_standard = compare_standard,
        )

        if compare_standard:
            comparison_log[name] = {
                "method":          res["method"],
                "err_standard":    res.get("err_standard",  float("nan")),
                "err_stochastic":  res.get("err_stochastic", float("nan")),
            }

        # ── Step 4e: μ and bias_correction ───────────────────────────
        print(f"  Computing post-Hadamard mean...")
        had_acts = []

        def _grab_had(mod, inp, out):
            x_raw = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            x_had = _rotate(x_raw, D, actual_had_bs)
            had_acts.append(x_had.cpu())

        h_mu = module.register_forward_hook(_grab_had)
        with torch.no_grad():
            n_collected = 0
            for batch in calib_loader:
                if batch is None:
                    continue
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                if x is None:
                    continue
                model(x.to(device))
                n_collected += 1
                if n_collected >= num_batches:
                    break
        h_mu.remove()

        if had_acts:
            X_had_all = torch.cat(had_acts, dim=0).float()
            mu        = X_had_all.mean(dim=0)

            print(f"  μ max={mu.abs().max():.4f}  "
                  f"μ mean={mu.abs().mean():.4f}")

            W_had_q   = res["weight_q"].to(device).float()
            bias_corr = (W_had_q @ mu.to(device)).cpu()

            existing_bias = module.inner.linear.bias
            if existing_bias is not None:
                module.inner.linear.bias.data = (
                    existing_bias.data.float() + bias_corr.to(device)
                ).to(existing_bias.dtype)
                module.bias_correction = None
                print(f"  bias_corr folded into existing bias")
            else:
                module.bias_correction = bias_corr.half()
                print(f"  bias_corr stored as separate buffer")

            module.mu = mu.half()
            del X_had_all, W_had_q, bias_corr
            torch.cuda.empty_cache()
        else:
            module.mu              = None
            module.bias_correction = None

        del had_acts

        # ── Step 4f: store on wrapper (D and had_block_size already set in Phase 2)
        # weight_q lives in the padded Hadamard domain: shape [N, P] (P >= M),
        # NOT the unpadded inner.linear.weight shape [N, M].
        # Store in the model dtype (fp16), not fp32: the forward GEMM casts
        # weight_q to the activation dtype anyway (step 5), so this is a
        # numerical no-op that halves the dominant memory consumer — critical
        # to keep the padded (1.6×) weight_q under a 16GB GPU during calibration.
        module.weight_q       = res["weight_q"].reshape(W_had.shape).to(
            module.inner.linear.weight.dtype)
        module.alpha_q        = res["alpha"]
        module.bias_q         = res["bias"]
        module.act_quant_mode = "nvfp4"
        module.act_block_size = block_size

        # ── Step 4g: sanity check ─────────────────────────────────────
        with torch.no_grad():
            x_test  = torch.randn(4, 16, M, device=device)
            x_had_t = _rotate(x_test, D, actual_had_bs)
            if module.mu is not None:
                x_had_t = x_had_t - module.mu.to(device=device,
                                                   dtype=x_had_t.dtype)
            out_a16 = F.linear(x_had_t,
                               module.weight_q.to(x_test.dtype))
            if module.bias_correction is not None:
                out_a16 = out_a16 + module.bias_correction.to(
                    device=device, dtype=out_a16.dtype)
            x_had_q = _rotate(x_test, D, actual_had_bs)
            if module.mu is not None:
                x_had_q = x_had_q - module.mu.to(device=device,
                                                   dtype=x_had_q.dtype)
            x_q     = quantize_activations(
                x_had_q, block_size, e_bits, m_bits,
                e_bits_scale=e_bits_scale, m_bits_scale=m_bits_scale)
            out_a4  = F.linear(x_q, module.weight_q.to(x_test.dtype))
            if module.bias_correction is not None:
                out_a4 = out_a4 + module.bias_correction.to(
                    device=device, dtype=out_a4.dtype)
            out_ref = F.linear(x_test, W_mat.to(device).to(x_test.dtype))
            rel_a16 = (out_ref - out_a16).abs().mean() / \
                      out_ref.abs().mean().clamp(min=1e-8)
            rel_a4  = (out_ref - out_a4).abs().mean() / \
                      out_ref.abs().mean().clamp(min=1e-8)
            print(f"  sanity rel_err A16={rel_a16:.4f}  A4={rel_a4:.4f}")

        del res, W_orig, W_mat, W_had
        if lean:
            # LEAN: drop this layer's original weight now that weight_q holds it.
            # Bias is preserved (forward still reads inner.linear.bias). Only
            # quantized layers reach here — skipped/retained layers keep their
            # weights for the FP16 fallback forward.
            module.inner.linear.weight = None
        torch.cuda.empty_cache()

    del D_dict, H_dict_had

    # ── Summary ───────────────────────────────────────────────────────
    if comparison_log:
        n_stochastic = sum(1 for v in comparison_log.values()
                           if v["method"] == "stochastic")
        n_standard   = sum(1 for v in comparison_log.values()
                           if v["method"] == "standard")

        all_improvements = []
        for v in comparison_log.values():
            if not (math.isnan(v["err_standard"]) or
                    math.isnan(v["err_stochastic"])):
                imp = (v["err_standard"] - v["err_stochastic"]) / \
                      max(v["err_standard"], 1e-8) * 100
                all_improvements.append(imp)

        print(f"\n{'='*60}")
        print(f"Stochastic FP4 calibration summary:")
        print(f"  Layers using stochastic: {n_stochastic}/{len(comparison_log)}")
        print(f"  Layers using standard:   {n_standard}/{len(comparison_log)}")
        if all_improvements:
            print(f"  Mean improvement: "
                  f"{sum(all_improvements)/len(all_improvements):+.2f}%")
            print(f"  Best improvement: {max(all_improvements):+.2f}%")
            print(f"  Worst (degraded): {min(all_improvements):+.2f}%")
        print(f"{'='*60}")

    # ── Spot check ────────────────────────────────────────────────────
    for name, module in model.named_modules():
        if type(module).__name__ == "HadamardQuantLinearFP":
            print(f"\nSpot check {name}:")
            print(f"  weight_q: "
                  f"{'set' if module.weight_q is not None else 'None'}")
            print(f"  had_block_size: {module.had_block_size}")
            print(f"  mu: {'set' if module.mu is not None else 'None'}")
            print(f"  bias_correction: "
                  f"{'set' if module.bias_correction is not None else 'folded'}")
            break

    return model


def calibrate_model_gf4(
    model,
    calib_loader,
    block_size,
    device,
    e_bits=2,
    m_bits=1,
    e_bits_scale=4,
    m_bits_scale=3,
    num_batches=4,
    n_samples=32,
    had_block_size="auto",
    randomize_hadamard=True,
    hadamard_seed=None,
    compare_standard=True,
    alpha_candidates=(1.5, 2.0, 2.5, 3.0, 4.0),
    extra_skip_patterns=(),
    lean=False,
):
    """
    GF4 activation quantization variant of calibrate_model_stochastic_fp4.

    Runs the full Hadamard weight-quantization pipeline (stochastic FP4
    on W_had, μ, bias_correction), then performs a second calibration pass
    to find the per-layer Gaussian-optimal clip ratio (act_clip_ratio) for
    GF4 activation quantization.

    After this function each HadamardQuantLinearFP module has:
      • weight_q, mu, bias_correction  — from the weight-quant pass
      • act_quant_mode = "gf4"
      • act_clip_ratio = α*           — minimizes block MSE on calibration acts
    """
    # ── Phase 1-4: standard Hadamard weight quantization ─────────────────
    model = calibrate_model_stochastic_fp4(
        model, calib_loader, block_size, device,
        e_bits=e_bits, m_bits=m_bits,
        e_bits_scale=e_bits_scale, m_bits_scale=m_bits_scale,
        num_batches=num_batches, n_samples=n_samples,
        had_block_size=had_block_size,
        randomize_hadamard=randomize_hadamard,
        hadamard_seed=hadamard_seed,
        compare_standard=compare_standard,
        extra_skip_patterns=extra_skip_patterns,
        lean=lean,
    )

    SKIP_LAYERS = {"lm_head", "embed_tokens", "embed_positions"}
    SKIP_LAYERS |= set(extra_skip_patterns)

    print(f"\n{'='*60}")
    print("GF4 per-layer clip-ratio calibration")
    print(f"  α candidates: {list(alpha_candidates)}")
    print(f"{'='*60}")

    # Collect every layer's post-Hadamard, post-μ activations in a SINGLE set
    # of forward passes (one hook per layer at once), NOT one full-model forward
    # per layer.  The old per-layer-forward loop ran num_batches full 2.7B
    # forwards × 193 layers (≈O(L²)) and blew the wall-clock budget on a T4;
    # this is num_batches forwards total — a ~Lx speedup with identical α* math.
    MAX_CAL_TOKENS = 256   # per-layer token cap (speed + bounded CPU memory)

    eligible = []
    for name, module in model.named_modules():
        if type(module).__name__ != "HadamardQuantLinearFP":
            continue
        if any(skip in name for skip in SKIP_LAYERS):
            continue
        if module.had_block_size is None:
            continue
        eligible.append((name, module))

    # Run the capture forward with every layer already in "gf4" @ default clip
    # so each layer sees the GF4-quantized upstream signal it gets at inference
    # (closer than the old loop, which left downstream layers in "nvfp4").
    for _, module in eligible:
        module.act_quant_mode = "gf4"
        module.act_clip_ratio = 2.5

    had_acts  = {name: [] for name, _ in eligible}
    tok_count = {name: 0  for name, _ in eligible}

    def _make_grab(_name, _D, _mu, _bs):
        def _hook(mod, inp, out):
            if tok_count[_name] >= MAX_CAL_TOKENS:
                return
            x_raw = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            if _D is not None:
                x_h = _rotate(x_raw, _D, _bs)
            else:
                if x_raw.shape[-1] < _bs:
                    x_raw = F.pad(x_raw, (0, _bs - x_raw.shape[-1]))
                x_h = fwht_blockwise(x_raw, _bs)
            if _mu is not None:
                x_h = x_h - _mu.to(device=x_h.device)
            x_h = x_h[:MAX_CAL_TOKENS - tok_count[_name]]
            tok_count[_name] += x_h.shape[0]
            had_acts[_name].append(x_h.cpu())
        return _hook

    hooks = []
    for name, module in eligible:
        D  = module.D.to(device) if module.D is not None else None
        mu = module.mu.to(device).float() if module.mu is not None else None
        hooks.append(module.register_forward_hook(
            _make_grab(name, D, mu, module.had_block_size)))

    with torch.no_grad():
        n_coll = 0
        for batch in calib_loader:
            if batch is None:
                continue
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if x is None:
                continue
            model(x.to(device))
            n_coll += 1
            if n_coll >= num_batches or all(
                    tok_count[n] >= MAX_CAL_TOKENS for n, _ in eligible):
                break
    for h in hooks:
        h.remove()

    # Per-layer α* search on the collected activations (cheap — no forwards).
    for name, module in eligible:
        if not had_acts[name]:
            module.act_clip_ratio = 2.5
            module.act_quant_mode = "gf4"
            print(f"  {name}: no activations collected, using α=2.5")
            continue

        X_cal = torch.cat(had_acts[name], dim=0).float()[:MAX_CAL_TOKENS].to(device)
        best_alpha = float(alpha_candidates[0])
        best_mse   = float('inf')
        for alpha in alpha_candidates:
            X_q  = quantize_activations_gf4(X_cal, block_size, clip_ratio=alpha)
            mse  = (X_cal - X_q).pow(2).mean().item()
            if mse < best_mse:
                best_mse   = mse
                best_alpha = float(alpha)

        print(f"  {name}: α*={best_alpha:.2f}  mse={best_mse:.6f}")
        module.act_clip_ratio = best_alpha
        module.act_quant_mode = "gf4"
        had_acts[name] = None
        del X_cal
    torch.cuda.empty_cache()

    return model


# ══════════════════════════════════════════════════════════════════════════
# Block-sequential OFFLOAD calibration — for models too big to fit on the GPU
# ══════════════════════════════════════════════════════════════════════════
#
# calibrate_model_gf4 (above) needs the WHOLE model resident on the GPU: a
# global Hessian forward, a global GF4 activation forward, and per-layer forwards
# all run the full model. Even with lean=True (which frees originals) that caps
# us at ~6.7B on a 24GB card, because the fake-quant weights themselves are the
# floor (2 bytes/param → 13B = 26GB > 24GB).
#
# calibrate_model_gf4_offload produces the SAME per-layer quantization — identical
# W_had rotation, calibrate_layer_stochastic_fp4, μ / bias_correction, GF4 α*
# search — but walks the decoder ONE BLOCK AT A TIME, GPTQ/QuaRot style:
#   • the model lives in CPU RAM;
#   • one transformer block is moved to the GPU, quantized, and moved back;
#   • hidden states are threaded block→block on the CPU.
# Peak GPU memory becomes ONE block (+ its activations), so 13B/30B/70B fit a
# 24GB L4 given enough CPU RAM to hold the model.
#
# This is NOT byte-identical to calibrate_model_gf4: the Hessian and μ for block
# b are collected from inputs produced by the already-QUANTIZED blocks 0..b-1
# (sequential error feedback), whereas the global path computes every layer's
# Hessian on the fully-FP model. Sequential feedback is the GPTQ formulation and
# is generally ≥ the parallel one; validate parity (offload vs global) on a
# small model that fits both ways (opt-125m / opt-1.3b) before trusting big runs.

class _StopForward(Exception):
    """Raised by _Catcher to abort the forward once block-0 inputs are captured."""
    pass


class _Catcher(nn.Module):
    """Temporarily replaces decoder block 0. Records the hidden-states of every
    calibration sample plus the layer's non-hidden args/kwargs (attention_mask,
    position_ids, position_embeddings, …), then aborts — the rest of the model
    never runs, so this stays cheap and never needs the later blocks on-device."""
    def __init__(self, block):
        super().__init__()
        self.block  = block
        self.inps   = []
        self.args   = None
        self.kwargs = None

    def forward(self, hidden_states, *args, **kwargs):
        self.inps.append(hidden_states.detach().to("cpu"))
        if self.kwargs is None:          # capture layer kwargs once (fixed seqlen)
            self.args   = args
            self.kwargs = kwargs
        raise _StopForward


def _move_kw(obj, device):
    """Recursively move tensors in a (possibly nested) arg/kwarg structure to
    device. Non-tensors (bools, ints, None) pass through untouched."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_kw(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: _move_kw(v, device) for k, v in obj.items()}
    return obj


def _causal_mask_4d(seqlen, dtype, device):
    """Additive [1,1,S,S] causal mask: 0 on/below the diagonal, finfo.min above.

    When we replay a decoder block in isolation (offload calibration + streaming
    eval) we bypass the model-level code that would normally build the causal
    mask. transformers >= ~4.48 passes attention_mask=None down to the layer and
    relies on SDPA's is_causal fallback (is_causal = mask is None and q_len > 1);
    the unified attention interface drops that fallback, so an isolated block with
    mask=None attends BIDIRECTIONALLY — every token sees the future, which inflates
    perplexity uniformly across all modes. Forcing an explicit causal mask makes the
    streamed forward correct regardless of transformers version or attn backend.
    Our calib/eval sequences are dense (no padding), so a pure causal mask is exact.
    """
    min_val = torch.finfo(dtype).min
    m = torch.full((seqlen, seqlen), min_val, dtype=dtype, device=device)
    m = torch.triu(m, diagonal=1)
    return m.view(1, 1, seqlen, seqlen)


def _decoder_blocks_and_prep(model):
    """Locate the decoder block ModuleList + the pre-block submodules needed to
    turn input_ids into block-0 hidden states. Returns (blocks, prep_mods, family)."""
    base = getattr(model, "model", model)
    # OPT: model.model.decoder.{embed_tokens, embed_positions, project_in,
    #      layernorm_embedding, layers}
    dec = getattr(base, "decoder", None)
    if dec is not None and hasattr(dec, "layers"):
        prep = [getattr(dec, n, None) for n in
                ("embed_tokens", "embed_positions", "project_in",
                 "layernorm_embedding")]
        return dec.layers, [m for m in prep if m is not None], "opt"
    # Llama / Mistral: model.model.{embed_tokens, rotary_emb, layers}
    if hasattr(base, "layers"):
        prep = [getattr(base, n, None) for n in ("embed_tokens", "rotary_emb")]
        return base.layers, [m for m in prep if m is not None], "llama"
    raise ValueError("offload calibration: unsupported architecture "
                     "(need an OPT- or Llama-style decoder)")


def calibrate_model_gf4_offload(
    model,
    calib_loader,
    block_size,
    device,
    e_bits=2,
    m_bits=1,
    e_bits_scale=4,
    m_bits_scale=3,
    num_batches=4,
    n_samples=32,
    had_block_size="auto",
    randomize_hadamard=True,
    hadamard_seed=None,
    compare_standard=True,
    alpha_candidates=(1.5, 2.0, 2.5, 3.0, 4.0),
    extra_skip_patterns=(),
    lean=True,                 # offload implies lean (originals freed per block)
    max_cal_tokens=256,        # per-layer token cap for the GF4 α* search
):
    """Block-sequential offload version of calibrate_model_gf4 (see banner above).

    Same per-layer math as calibrate_model_gf4; the model stays on CPU and blocks
    are streamed to `device` one at a time. lean is forced True (each block's
    originals are freed after its weight_q is written)."""
    model.eval()
    SKIP_LAYERS = {"lm_head", "embed_tokens", "embed_positions"} | set(extra_skip_patterns)
    if extra_skip_patterns:
        print(f"  [offload] extra skip patterns (kept FP16): {sorted(extra_skip_patterns)}")

    # ── Phase 1: wrap linears (in place — blocks still forward normally) ──
    model = wrap_layers_with_hadamard(model)
    blocks, prep_mods, family = _decoder_blocks_and_prep(model)
    n_blocks = len(blocks)
    print(f"[offload] {family} decoder: {n_blocks} blocks — model on CPU, "
          f"one block at a time on {device}")

    # ── Phase 2: catch block-0 inputs for num_batches calib samples ──────
    for m in prep_mods:
        m.to(device)
    orig_block0  = blocks[0]
    catcher      = _Catcher(orig_block0)
    blocks[0]    = catcher
    with torch.no_grad():
        for batch in calib_loader:
            if batch is None:
                continue
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if x is None:
                continue
            try:
                model(x.to(device))
            except _StopForward:
                pass
            if len(catcher.inps) >= num_batches:
                break
    blocks[0]    = orig_block0
    inps         = catcher.inps                 # list of [1,S,H] on CPU
    layer_args   = catcher.args
    layer_kwargs = dict(catcher.kwargs)
    # Force a plain no-cache forward when we replay blocks in isolation.
    layer_kwargs["use_cache"]        = False
    layer_kwargs["output_attentions"] = False
    for k in ("past_key_value", "past_key_values"):
        if k in layer_kwargs:
            layer_kwargs[k] = None
    # Replaying a block in isolation bypasses the model-level causal-mask build;
    # newer transformers ship attention_mask=None down to the layer, so force an
    # explicit causal mask or the block attends bidirectionally (see _causal_mask_4d).
    _S = inps[0].shape[1]
    layer_kwargs["attention_mask"] = _causal_mask_4d(_S, inps[0].dtype, "cpu")
    for m in prep_mods:
        m.to("cpu")
    del catcher
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[offload] captured {len(inps)} block-0 input samples; "
          f"layer kwargs: {sorted(layer_kwargs.keys())}")

    comparison_log = {}

    # ── Phase 3: per-block quantization ──────────────────────────────────
    for b, block in enumerate(blocks):
        block.to(device)
        l_args   = _move_kw(layer_args,   device)
        l_kwargs = _move_kw(layer_kwargs, device)

        # wrappers in this block (relative name) that we will quantize
        wrappers = [
            (nm, mod) for nm, mod in block.named_modules()
            if type(mod).__name__ == "HadamardQuantLinearFP"
            and not any(s in nm for s in SKIP_LAYERS)
        ]
        if not wrappers:
            # e.g. a block that is entirely skip-listed — just thread inputs.
            new_inps = []
            with torch.no_grad():
                for inp in inps:
                    out = block(inp.to(device), *l_args, **l_kwargs)
                    out = out[0] if isinstance(out, tuple) else out
                    new_inps.append(out.detach().to("cpu"))
            inps = new_inps
            block.to("cpu"); del l_args, l_kwargs
            gc.collect(); torch.cuda.empty_cache()
            print(f"[offload] block {b + 1}/{n_blocks}: no quantizable layers, threaded")
            continue

        # ── 3a: generate D + set Hadamard geometry on each wrapper ────────
        for nm, module in wrappers:
            W_shape = module.inner.linear.weight.data.shape
            M = W_shape[1] if len(W_shape) == 2 else W_shape[1] * W_shape[2] * W_shape[3]
            if had_block_size == "auto":
                actual_had_bs = _next_pow2(M); d_len = actual_had_bs
            else:
                actual_had_bs = had_block_size
                assert M % actual_had_bs == 0, \
                    f"{nm}: M={M} not divisible by had_block_size={actual_had_bs}"
                d_len = M
            assert (actual_had_bs & (actual_had_bs - 1)) == 0, \
                f"{nm}: had block size {actual_had_bs} must be a power of 2"
            if randomize_hadamard:
                seed = hadamard_seed if hadamard_seed is not None \
                       else _stable_seed(f"{b}.{nm}")
                D = generate_random_signs(d_len, actual_had_bs, device, seed=seed)
            else:
                D = torch.ones(d_len, device=device)
            module.D              = D.cpu()
            module.had_block_size = actual_had_bs
            module.had_in_dim     = M

        # ── 3b: one FP forward over inps → H_had + μ (online) + α* acts ───
        H_acc   = {}                                   # nm -> [block hessians]
        mu_sum  = {nm: None for nm, _ in wrappers}     # nm -> running Σ x_had
        mu_cnt  = {nm: 0    for nm, _ in wrappers}
        cal_acc = {nm: []   for nm, _ in wrappers}     # capped rotated acts
        cal_tok = {nm: 0    for nm, _ in wrappers}

        def _make_hook(nm, module):
            D_ref = module.D.to(device)
            hbs   = module.had_block_size
            inner = module.inner.linear
            def hook(mod, inp, out):
                x     = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                x_had = _rotate(x, D_ref, hbs)
                Hb = compute_hessian_blocks(x_had, inner, block_size)
                if Hb is not None:
                    if nm not in H_acc:
                        H_acc[nm] = Hb
                    else:
                        for i in range(len(Hb)):
                            H_acc[nm][i] += Hb[i]
                s = x_had.sum(dim=0).cpu()
                mu_sum[nm] = s if mu_sum[nm] is None else mu_sum[nm] + s
                mu_cnt[nm] += x_had.shape[0]
                if cal_tok[nm] < max_cal_tokens:
                    take = x_had[:max_cal_tokens - cal_tok[nm]].cpu()
                    cal_acc[nm].append(take); cal_tok[nm] += take.shape[0]
            return hook

        # This forward runs the block with FP weights (nothing in it is quantized
        # yet — 3c does that below), so the captured block output is the FP-domain
        # activation. Threading THAT to the next block keeps every block's Hessian
        # on the fully-FP model, exactly like the in-GPU path, which computes H_had
        # for all layers up front before any quantization. (Threading the QUANTIZED
        # output instead compounds quant noise in deep blocks' Hessians → PPL grows
        # with depth: fine at 12 blocks, +3 PPL at 24, worse at 80.)
        handles = [module.register_forward_hook(_make_hook(nm, module))
                   for nm, module in wrappers]
        next_inps = []
        with torch.no_grad():
            for inp in inps:
                out = block(inp.to(device), *l_args, **l_kwargs)
                out = out[0] if isinstance(out, tuple) else out
                next_inps.append(out.detach().to("cpu"))
        for h in handles:
            h.remove()
        n_fwd = max(len(inps), 1)

        # ── 3c: per-layer quantize (reuse the exact global math) ──────────
        for nm, module in wrappers:
            if nm not in H_acc:
                print(f"  [offload] {b}.{nm}: no Hessian collected, skipping")
                continue
            H_blocks = [h.float() / n_fwd for h in H_acc[nm]]
            D, actual_had_bs = module.D.to(device), module.had_block_size

            W_orig = module.inner.linear.weight.data.to(device).float()
            W_mat  = W_orig.view(W_orig.shape[0], -1) if W_orig.dim() == 4 else W_orig
            W_had  = _rotate(W_mat, D, actual_had_bs).cpu().float()

            res = calibrate_layer_stochastic_fp4(
                layer_name=f"{b}.{nm}", W_orig=W_had, H_blocks=H_blocks,
                block_size=block_size, e_bits=e_bits, m_bits=m_bits,
                e_bits_scale=e_bits_scale, m_bits_scale=m_bits_scale,
                device=device, n_samples=n_samples, compare_standard=compare_standard,
            )
            if compare_standard:
                comparison_log[f"{b}.{nm}"] = {
                    "method": res["method"],
                    "err_standard": res.get("err_standard", float("nan")),
                    "err_stochastic": res.get("err_stochastic", float("nan")),
                }

            # μ and bias_correction (μ = E[H(D·x)] over all collected tokens)
            mu = (mu_sum[nm] / max(mu_cnt[nm], 1)).to(device).float()
            W_had_q   = res["weight_q"].to(device).float()
            bias_corr = (W_had_q @ mu).cpu()
            existing_bias = module.inner.linear.bias
            if existing_bias is not None:
                module.inner.linear.bias.data = (
                    existing_bias.data.float() + bias_corr.to(device)
                ).to(existing_bias.dtype)
                module.bias_correction = None
            else:
                module.bias_correction = bias_corr.half()
            module.mu = mu.half()

            module.weight_q       = res["weight_q"].reshape(W_had.shape).to(
                module.inner.linear.weight.dtype)
            module.alpha_q        = res["alpha"]
            module.bias_q         = res["bias"]
            module.act_block_size = block_size

            # GF4 per-layer α* search on the collected (rotated, μ-subtracted) acts
            if cal_acc[nm]:
                X_cal = torch.cat(cal_acc[nm], dim=0).float().to(device)
                X_cal = X_cal - mu.to(X_cal.dtype)
                best_alpha, best_mse = float(alpha_candidates[0]), float("inf")
                for alpha in alpha_candidates:
                    X_q = quantize_activations_gf4(X_cal, block_size, clip_ratio=alpha)
                    mse = (X_cal - X_q).pow(2).mean().item()
                    if mse < best_mse:
                        best_mse, best_alpha = mse, float(alpha)
                module.act_clip_ratio = best_alpha
                del X_cal
            else:
                module.act_clip_ratio = 2.5
            module.act_quant_mode = "gf4"

            if lean:
                module.inner.linear.weight = None
            del W_orig, W_mat, W_had, res, W_had_q
            torch.cuda.empty_cache()

        # ── 3d: thread the FP block outputs (captured in 3b) to the next block ─
        inps = next_inps

        del H_acc, mu_sum, cal_acc, next_inps, l_args, l_kwargs
        block.to("cpu")
        gc.collect(); torch.cuda.empty_cache()
        print(f"[offload] block {b + 1}/{n_blocks} quantized "
              f"({len(wrappers)} layers), GPU freed")

    # ── Summary ──────────────────────────────────────────────────────────
    if comparison_log:
        n_stoch = sum(1 for v in comparison_log.values() if v["method"] == "stochastic")
        n_std   = sum(1 for v in comparison_log.values() if v["method"] == "standard")
        print(f"\n{'='*60}\n[offload] GF4 calibration summary:")
        print(f"  layers: {len(comparison_log)} "
              f"(stochastic {n_stoch} / standard {n_std})\n{'='*60}")
    return model


def _post_block_tail(model, family):
    """Return (tail_modules, lm_head): the modules applied AFTER the decoder
    blocks to turn the last hidden state into logits, for block-major eval."""
    lm_head = getattr(model, "lm_head", None)
    if family == "opt":
        dec  = model.model.decoder
        tail = [m for m in (getattr(dec, "final_layer_norm", None),
                            getattr(dec, "project_out", None)) if m is not None]
    else:  # llama / neox style
        base = getattr(model, "model", model)
        tail = [m for m in (getattr(base, "norm", None),
                            getattr(base, "final_layer_norm", None)) if m is not None]
    return tail, lm_head


@torch.no_grad()
def evaluate_ppl_offload(model, ids, device, seqlen=2048):
    """Block-major streaming perplexity for a model too big to fit on `device`.

    The model stays on CPU; each decoder block is moved to the GPU ONCE, forwards
    every eval chunk, and is moved back — so peak GPU memory is one block plus the
    (small) embeddings + final-norm + lm_head, not the whole model. This is the
    eval-time counterpart of calibrate_model_gf4_offload and lets 13B/70B evaluate
    on a 24GB card. It reads whatever act_quant_mode is currently set on the
    HadamardQuantLinearFP wrappers, so it drops into the same
    `with act_quant_mode(model, mode=...)` loop as the in-GPU ppl_eval.

    Block-major (each block streamed once for ALL chunks) rather than chunk-major
    keeps block transfers at O(num_blocks) per call instead of
    O(num_blocks × num_chunks).
    """
    model.eval()
    blocks, prep_mods, family = _decoder_blocks_and_prep(model)
    tail_mods, lm_head = _post_block_tail(model, family)

    n = ids.size(0) // seqlen
    chunks = ids[:n * seqlen].view(n, seqlen)

    # The non-block parts are small — keep them resident on the GPU.
    for m in prep_mods + tail_mods + ([lm_head] if lm_head is not None else []):
        m.to(device)

    # ── capture block-0 inputs (mode-independent) for every chunk ──────────
    orig_block0 = blocks[0]
    catcher     = _Catcher(orig_block0)
    blocks[0]   = catcher
    for i in range(n):
        try:
            model(chunks[i:i + 1].to(device))
        except _StopForward:
            pass
    blocks[0]    = orig_block0
    hs           = catcher.inps                    # list of n [1,S,H] on CPU
    layer_args   = catcher.args
    layer_kwargs = dict(catcher.kwargs)
    layer_kwargs["use_cache"]         = False
    layer_kwargs["output_attentions"] = False
    for k in ("past_key_value", "past_key_values"):
        if k in layer_kwargs:
            layer_kwargs[k] = None
    # Force an explicit causal mask — streamed blocks bypass the model-level mask
    # build and newer transformers pass attention_mask=None (see _causal_mask_4d).
    layer_kwargs["attention_mask"] = _causal_mask_4d(seqlen, hs[0].dtype, "cpu")
    del catcher
    l_args   = _move_kw(layer_args,   device)
    l_kwargs = _move_kw(layer_kwargs, device)

    # ── stream each block once over all chunks ────────────────────────────
    for block in blocks:
        block.to(device)
        for i in range(n):
            out   = block(hs[i].to(device), *l_args, **l_kwargs)
            out   = out[0] if isinstance(out, tuple) else out
            hs[i] = out.detach().to("cpu")
        block.to("cpu")
        gc.collect(); torch.cuda.empty_cache()

    # ── tail (final norm / project_out) + lm_head + NLL, one chunk at a time ─
    nll, ntok = 0.0, 0
    for i in range(n):
        h = hs[i].to(device)
        for tm in tail_mods:
            h = tm(h)
        logits = lm_head(h)
        sl = logits[:, :-1, :].contiguous()
        lb = chunks[i:i + 1][:, 1:].contiguous().to(device)
        loss = F.cross_entropy(sl.view(-1, sl.size(-1)), lb.view(-1))
        if not (torch.isnan(loss) or torch.isinf(loss)):
            nll += loss.item() * (seqlen - 1); ntok += (seqlen - 1)
        hs[i] = None
        del h, logits, sl, lb
    torch.cuda.empty_cache()
    return math.exp(nll / ntok) if ntok else float("inf")


def calibrate_model_gf4_hsmooth(
    model,
    calib_loader,
    block_size,
    device,
    e_bits=2, m_bits=1,
    e_bits_scale=4, m_bits_scale=3,
    num_batches=4, n_samples=32,
    had_block_size="auto",
    randomize_hadamard=True,
    hadamard_seed=None,
    smooth_eps=1e-6,
    alpha_candidates=(1.5, 2.0, 2.5, 3.0, 4.0),
):
    """
    GF4 with H-domain per-channel smooth scaling (H-SmoothQuant, novel).

    After Hadamard, different output channels of H(D*x) can have unequal
    per-channel variance even though the rotation is energy-preserving. This
    happens because real LLM activations are not i.i.d. — some feature
    directions survive the rotation with higher amplitude.

    This calibration computes per-channel RMS s[m] of post-H activations
    and uses it to:
      (a) Store h_smooth_scale = s on each HadamardQuantLinearFP module.
          At inference: x_had_smooth = (x_had - μ) / s
      (b) Fold s into W_had_q offline:
          W_had_q_smooth[n, m] = W_had_q[n, m] * s[m]
          Net: x_smooth @ W_smooth^T = x_had @ W_had_q^T  (exact).

    After scaling, per-block RMS of x_smooth is near-uniform across all
    channels → GF4's per-block clipping is near-optimal everywhere.

    Note: W_had_q is adjusted post-quantization (approximate). A production
    implementation would re-quantize Q(W_had * s) for maximum weight quality.
    """
    # Phase 1: standard GF4 calibration (weights + per-layer clip ratio)
    model = calibrate_model_gf4(
        model, calib_loader, block_size, device,
        e_bits=e_bits, m_bits=m_bits,
        e_bits_scale=e_bits_scale, m_bits_scale=m_bits_scale,
        num_batches=num_batches, n_samples=n_samples,
        had_block_size=had_block_size,
        randomize_hadamard=randomize_hadamard,
        hadamard_seed=hadamard_seed,
        alpha_candidates=alpha_candidates,
    )

    SKIP = {"lm_head", "embed_tokens", "embed_positions"}

    print(f"\n{'='*60}")
    print("H-domain per-channel smooth scaling (H-SmoothQuant)")
    print(f"{'='*60}")

    for name, module in model.named_modules():
        if type(module).__name__ != "HadamardQuantLinearFP":
            continue
        if any(s in name for s in SKIP):
            continue
        if module.had_block_size is None or module.weight_q is None:
            continue

        actual_had_bs = module.had_block_size
        D  = module.D.to(device) if module.D is not None else None
        mu = module.mu.to(device).float() if module.mu is not None else None

        had_acts = []

        def _grab(mod, inp, out, _D=D, _mu=mu, _bs=actual_had_bs):
            x_raw = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            if _D is not None:
                x_h = _rotate(x_raw, _D, _bs)
            else:
                if x_raw.shape[-1] < _bs:
                    x_raw = F.pad(x_raw, (0, _bs - x_raw.shape[-1]))
                x_h = fwht_blockwise(x_raw, _bs)
            if _mu is not None:
                x_h = x_h - _mu.to(x_h.device)
            had_acts.append(x_h.cpu())

        h = module.register_forward_hook(_grab)
        with torch.no_grad():
            n_coll = 0
            for batch in calib_loader:
                if batch is None: continue
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                if x is None: continue
                model(x.to(device))
                n_coll += 1
                if n_coll >= num_batches: break
        h.remove()

        if not had_acts:
            continue

        X = torch.cat(had_acts, dim=0).float().to(device)  # [T, M]

        # Per-channel RMS, normalized so mean(s) = 1
        s = X.pow(2).mean(dim=0).sqrt().clamp(min=smooth_eps)  # [M]
        s = s / s.mean().clamp(min=1e-8)

        print(f"  {name}: s ∈ [{s.min():.3f}, {s.max():.3f}]  "
              f"std={s.std():.4f}")

        # Store s for inference-time division of x_had
        module.h_smooth_scale = s.cpu()

        # Fold s into W_had_q: W_smooth[n, m] = W_had_q[n, m] * s[m]
        # Proof: (x/s) @ (W*s)^T = x @ W^T  (exact, column-wise cancellation)
        W_q = module.weight_q.data.to(device).float()
        W_q_smooth = W_q * s.unsqueeze(0)                              # [N, M]
        if 'weight_q' in module.inner._buffers:
            orig = module.inner._buffers['weight_q']
            module.inner._buffers['weight_q'] = \
                W_q_smooth.to(dtype=orig.dtype, device=orig.device)
        else:
            orig = module.inner.weight_q
            module.inner.weight_q = W_q_smooth.to(dtype=orig.dtype, device=orig.device)

        del X, s, W_q, W_q_smooth
        torch.cuda.empty_cache()

    return model


def apply_gf4_hsmooth(
    model,
    calib_loader,
    block_size,
    device,
    num_batches=4,
    smooth_eps=1e-6,
):
    """
    Apply H-domain SmoothQuant to an ALREADY-CALIBRATED model in-place.

    Unlike calibrate_model_gf4_hsmooth, this does NOT re-run weight
    quantization. Use this when the model has already been wrapped and
    calibrated via quantize_model_fp / calibrate_model_gf4.

    Sets module.h_smooth_scale and folds s into module.weight_q for each
    HadamardQuantLinearFP layer. Restoring is the caller's responsibility.
    """
    SKIP = {"lm_head", "embed_tokens", "embed_positions"}

    print(f"\n{'='*60}")
    print("H-domain per-channel smooth scaling (apply post-hoc)")
    print(f"{'='*60}")

    for name, module in model.named_modules():
        if type(module).__name__ != "HadamardQuantLinearFP":
            continue
        if any(s in name for s in SKIP):
            continue
        if module.had_block_size is None or module.weight_q is None:
            continue

        actual_had_bs = module.had_block_size
        D  = module.D.to(device) if module.D is not None else None
        mu = module.mu.to(device).float() if module.mu is not None else None

        had_acts = []

        def _grab(mod, inp, out, _D=D, _mu=mu, _bs=actual_had_bs):
            x_raw = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            if _D is not None:
                x_h = _rotate(x_raw, _D, _bs)
            else:
                if x_raw.shape[-1] < _bs:
                    x_raw = F.pad(x_raw, (0, _bs - x_raw.shape[-1]))
                x_h = fwht_blockwise(x_raw, _bs)
            if _mu is not None:
                x_h = x_h - _mu.to(x_h.device)
            had_acts.append(x_h.cpu())

        h = module.register_forward_hook(_grab)
        with torch.no_grad():
            n_coll = 0
            for batch in calib_loader:
                if batch is None:
                    continue
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                if x is None:
                    continue
                model(x.to(device))
                n_coll += 1
                if n_coll >= num_batches:
                    break
        h.remove()

        if not had_acts:
            continue

        X = torch.cat(had_acts, dim=0).float().to(device)

        s = X.pow(2).mean(dim=0).sqrt().clamp(min=smooth_eps)
        s = s / s.mean().clamp(min=1e-8)

        print(f"  {name}: s ∈ [{s.min():.3f}, {s.max():.3f}]  std={s.std():.4f}")

        module.h_smooth_scale = s.cpu()

        W_q = module.weight_q.data.to(device).float()
        W_q_smooth = W_q * s.unsqueeze(0)
        if 'weight_q' in module.inner._buffers:
            orig = module.inner._buffers['weight_q']
            module.inner._buffers['weight_q'] = \
                W_q_smooth.to(dtype=orig.dtype, device=orig.device)
        else:
            orig = module.inner.weight_q
            module.inner.weight_q = W_q_smooth.to(dtype=orig.dtype, device=orig.device)

        del X, s, W_q, W_q_smooth
        torch.cuda.empty_cache()

    return model


# =========================================================
# 🔹 MAIN ENTRY
# =========================================================
def quantize_model_fp(model,
                      data_loader,
                      block_size=32,
                      e_bits=2,
                      m_bits=1,
                      e_bits_scale=8,
                      m_bits_scale=0,
                      device="cuda",
                      use_HG=True,
                      use_Hessian=False,
                      use_adap=False, use_forward=False, Hadamard=False,
                      joint=False, preshift=False, decompose=False,
                      outlier_threshold=6.0,    # threshold for outlier detection
                      num_calib_batches=4,      # batches for activation collection
                      had_block_size=256,       # Hadamard block size; "auto" = full row
                      use_gf4=False,            # Gaussian-optimal GF4 activation quant
                      gf4_variant="gf4",        # "gf4" | "hsmooth" | "learned" | "learned_per_layer"
                      extra_skip_patterns=(),   # layer name substrings kept in FP16 (e.g. Llama "down_proj")
                      lean=False,               # free per-layer originals during weight-quant (fits 6.7B/7B on 24GB)
                      offload=False,            # block-sequential CPU-offload calibration (fits 13B/30B/70B on 24GB)
                      ):
    """
    Quantize a model to FP4-like format with optional HG or Hessian calibration.

    extra_skip_patterns: iterable of layer-name substrings. Any matching layer
        is left in FP16 (exact fallback path, no Hadamard/quantization). Use for
        matrices that destabilize a given architecture — e.g. ("down_proj",) for
        Llama SwiGLU, or ("q_proj", "k_proj") to protect RoPE projections.
    """

    model = fold_bn_recursively(model)

    for p in model.parameters():
        p.requires_grad = False

    # ── Decomposed path: collect activation stats BEFORE replacing layers ─
    # Must happen on original nn.Linear modules so hooks fire correctly
    if decompose:
        # Collect BEFORE replacement on raw nn.Linear
        print("Collecting per-channel activation statistics for decomposition...")
        act_channel_max = collect_per_channel_activation_max(
            model, data_loader, device, num_batches=num_calib_batches
        )
        outlier_indices = identify_outlier_channels(
            act_channel_max, threshold_ratio=outlier_threshold
        )
        model = replace_layers_decomposed(
            model, block_size, e_bits, m_bits,
            e_bits_scale, m_bits_scale,
            outlier_indices, root_model=None
        )

    else:
        print("Using standard quantization wrappers")
        model = replace_layers(
            model,
            block_size, e_bits, m_bits,
            e_bits_scale, m_bits_scale
        )

    # Count what's in the model after replacement
    quant_count   = 0
    decomp_count  = 0
    conv1d_count  = 0
    for name, module in model.named_modules():
        if type(module).__name__ == "QuantConv1dFP":
            quant_count += 1
        if type(module).__name__ == "QuantLinearFP_Decomposed":
            decomp_count += 1
        if type(module).__name__ == "Conv1D":
            conv1d_count += 1

    print(f"QuantConv1dFP modules in model:       {quant_count}")
    print(f"QuantLinearFP_Decomposed in model:    {decomp_count}")
    print(f"Raw Conv1D still in model:            {conv1d_count}")

    # ── Calibration step ──────────────────────────────────────────────────
    if decompose:
        model = calibrate_model_decomposed(
            model, data_loader, block_size, device,
            e_bits=e_bits, m_bits=m_bits,
            e_bits_scale=e_bits_scale, m_bits_scale=m_bits_scale,
            outlier_threshold=outlier_threshold,
            num_batches=num_calib_batches,
            act_channel_max=act_channel_max,   # pass pre-collected stats
        )
    elif use_Hessian:
        print("Using Hessian block calibration")
        model = calibrate_model_Hessian_block(
            model, data_loader, block_size, device)
    elif use_HG:
        print("Using HG calibration")
        model = calibrate_model_HG(model, data_loader, device)
    elif use_adap:
        print("Using adaptive mesh calibration")
        model = calibrate_model_Hessian_scaled(
            model, data_loader, block_size, device)
    elif use_forward:
        print("Using forward reconstruction calibration")
        model = calibrate_model_Hessian_scaled_forward(
            model, data_loader, block_size, device)
    elif Hadamard:
        if use_gf4:
            if gf4_variant == "hsmooth":
                print("Using Hadamard calibration with H-domain SmoothQuant + GF4")
                model = calibrate_model_gf4_hsmooth(
                    model, data_loader, block_size, device,
                    e_bits=e_bits, m_bits=m_bits,
                    e_bits_scale=e_bits_scale, m_bits_scale=m_bits_scale,
                    num_batches=num_calib_batches,
                    n_samples=32,
                    had_block_size=had_block_size,
                )
            elif gf4_variant in ("learned", "learned_per_layer"):
                print(f"Using Hadamard calibration with GF4 + learned codebook "
                      f"({'per-layer' if gf4_variant == 'learned_per_layer' else 'global'})")
                model = calibrate_model_gf4(
                    model, data_loader, block_size, device,
                    e_bits=e_bits, m_bits=m_bits,
                    e_bits_scale=e_bits_scale, m_bits_scale=m_bits_scale,
                    num_batches=num_calib_batches,
                    n_samples=32,
                    had_block_size=had_block_size,
                    compare_standard=True,
                    extra_skip_patterns=extra_skip_patterns,
                    lean=lean,
                )
                model = calibrate_gf4_learned_levels(
                    model, data_loader, device, block_size,
                    num_batches=num_calib_batches,
                    per_layer=(gf4_variant == "learned_per_layer"),
                )
            elif offload:
                print("Using block-sequential OFFLOAD Hadamard+GF4 calibration "
                      "(model on CPU, one block at a time on GPU)")
                model = calibrate_model_gf4_offload(
                    model, data_loader, block_size, device,
                    e_bits=e_bits, m_bits=m_bits,
                    e_bits_scale=e_bits_scale, m_bits_scale=m_bits_scale,
                    num_batches=num_calib_batches,
                    n_samples=32,
                    had_block_size=had_block_size,
                    compare_standard=True,
                    extra_skip_patterns=extra_skip_patterns,
                    lean=True,   # offload frees originals per block
                )
            else:
                print("Using Hadamard-domain calibration with GF4 activation quantization")
                model = calibrate_model_gf4(
                    model, data_loader, block_size, device,
                    e_bits=e_bits, m_bits=m_bits,
                    e_bits_scale=e_bits_scale, m_bits_scale=m_bits_scale,
                    num_batches=num_calib_batches,
                    n_samples=32,
                    had_block_size=had_block_size,
                    compare_standard=True,
                    extra_skip_patterns=extra_skip_patterns,
                    lean=lean,
                )
        else:
            print("Using Hadamard-domain calibration")
            model = calibrate_model_stochastic_fp4(
                model, data_loader, block_size, device,
                e_bits=e_bits, m_bits=m_bits,
                e_bits_scale=e_bits_scale, m_bits_scale=m_bits_scale,
                num_batches=num_calib_batches,
                n_samples=32,
                had_block_size=had_block_size,
                compare_standard=True,
                extra_skip_patterns=extra_skip_patterns,
                lean=lean,
            )
    elif joint:
        print("Using joint calibration")
        # model = calibrate_model_joint(
        #     model, data_loader, block_size, device,
        #     e_bits=e_bits, m_bits=m_bits,
        #     e_bits_scale=e_bits_scale, m_bits_scale=m_bits_scale)
        model = calibrate_model_hadamard_joint(model, data_loader, block_size, device,
             e_bits=e_bits, m_bits=m_bits,
             e_bits_scale=e_bits_scale, m_bits_scale=m_bits_scale, num_batches=num_calib_batches,
             had_block_size=256)
        for name, module in model.named_modules():
            if type(module).__name__ == "HadamardQuantLinearFP":
                inner_wq = module.inner._buffers.get('weight_q', None)
                attr_wq  = getattr(module.inner, 'weight_q', 'MISSING')
                print(f"{name}:")
                print(f"  inner._buffers['weight_q']: "
                    f"{'tensor' if inner_wq is not None else 'None'}")
                print(f"  getattr weight_q: "
                    f"{'tensor' if (attr_wq is not None and attr_wq != 'MISSING') else attr_wq}")
                print(f"  had_block_size: {module.had_block_size}")
                print(f"  D: {'set' if module.D is not None else 'None'}")
                break
    elif preshift:
        print("Using preshifted calibration")
        model = calibrate_model_preshifted(
            model, data_loader, block_size, device,
            e_bits=e_bits, m_bits=m_bits,
            e_bits_scale=e_bits_scale, m_bits_scale=m_bits_scale)
    else:
        print("Using standard calibration")
        model = calibrate_model(model, data_loader, device)

    if not lean and not offload:
        # apply_pruning_mask reads the ORIGINAL weights (module.linear.weight) to
        # build a sparsity mask; lean/offload freed those. For dense (non-pruned)
        # models the mask is all-ones — a no-op — so skipping it changes nothing
        # here. (Don't use lean/offload with a pre-pruned model.)
        apply_pruning_mask(model)

    # ── Weight statistics ─────────────────────────────────────────────────
    for name, module in model.named_modules():
        if hasattr(module, 'weight_q') and module.weight_q is not None:
            wq = module.weight_q
            print(f"{name:60s} | max={wq.abs().max():.4f} | "
                  f"mean={wq.abs().mean():.4f} | "
                  f"nan={torch.isnan(wq).any()} | "
                  f"inf={torch.isinf(wq).any()} | "
                  f"zeros={(wq.abs() < 1e-8).float().mean():.3f}")

    # ── Quick sanity check on first quantized layer ───────────────────────
    # Skipped in lean/offload mode: the originals were freed during weight-quant,
    # so the orig-vs-quant comparison below would dereference a None weight.
    if lean or offload:
        return model
    for name, module in model.named_modules():
        if hasattr(module, 'weight_q') and module.weight_q is not None:
            # Decomposed modules use normal_indices for the FP4 weight
            if type(module).__name__ == "QuantLinearFP_Decomposed":
                in_f  = module.normal_indices.shape[0]
                x_test = torch.randn(1, module.linear.weight.shape[1]).to(device)
                out_orig  = module.linear(x_test)
                out_quant = module(x_test)
            elif type(module).__name__ == "HadamardQuantLinearFP":
                # weight_q lives in the padded Hadamard domain [N, P>=M] and
                # expects rotated+padded input, so a raw F.linear(x, weight_q)
                # is a shape/semantics mismatch.  Run the module's own forward,
                # which applies H(D*pad(x)) - μ before the GEMM, against the
                # original linear for the reference.
                in_f   = module.inner.linear.in_features
                x_test = torch.randn(
                    1, in_f, dtype=module.inner.linear.weight.dtype
                ).to(device)
                out_orig  = F.linear(x_test, module.inner.linear.weight,
                                     module.inner.linear.bias)
                out_quant = module(x_test)
            else:
                in_f  = module.linear.in_features
                x_test = torch.randn(1, in_f, dtype=module.linear.weight.dtype).to(device)
                module.weight_q = module.weight_q.to(module.linear.weight.dtype)
                out_orig  = F.linear(x_test, module.linear.weight,
                                     module.linear.bias)
                out_quant = F.linear(x_test, module.weight_q,
                                     module.linear.bias)

            print(f"{name}: orig_max={out_orig.abs().max():.4f}, "
                  f"quant_max={out_quant.abs().max():.4f}")
            print(f"  relative error: "
                  f"{((out_orig - out_quant).abs() / out_orig.abs().clamp(min=1e-8)).mean():.4f}")
            break

    return model


def quantize_model_fp_lean(*args, **kwargs):
    """Memory-lean entry point: identical to quantize_model_fp but frees each
    layer's ORIGINAL weight during weight-quant (lean=True), halving the resident
    peak (~2x -> ~1x model) so 6.7B/7B fit a 24GB GPU. Same numerical result; the
    orig-vs-quant post-calibration sanity check is skipped (originals are gone)."""
    kwargs["lean"] = True
    return quantize_model_fp(*args, **kwargs)


def quantize_model_fp_offload(*args, **kwargs):
    """Block-sequential CPU-offload entry point. The model stays in CPU RAM and
    is quantized one decoder block at a time on the GPU (GPTQ/QuaRot style), so
    13B/30B/70B fit a 24GB card given enough system RAM to hold the model. Only
    the Hadamard+GF4 path (use_gf4=True, gf4_variant='gf4') is supported. Not
    byte-identical to the in-GPU path — it uses sequential error feedback — so
    validate parity on a small model (opt-125m/1.3b) first."""
    kwargs["offload"] = True
    return quantize_model_fp(*args, **kwargs)


# ── Architecture-specific skip presets ───────────────────────────────────────
#
# Some weight matrices destabilize Hadamard+FP4 quantization for specific
# architectures. Layers matching these substrings are kept in FP16 (exact
# fallback path) rather than quantized.
#
# Llama (RMSNorm, SwiGLU, RoPE) differs from OPT (LayerNorm, GeLU MLP):
#   • down_proj — SwiGLU output projection. Its input is silu(gate)·up, a
#     heavy-tailed product with extreme outliers (dim 8192 for Llama-1b).
#     The activation-weighted Hessian is dominated by these outliers, so the
#     reconstructed weight is mismatched to typical inputs → PPL collapse.
#   • RMSNorm does not center activations (unlike LayerNorm), so inputs to all
#     linears carry a non-zero, token-varying mean that the single calibrated
#     μ cannot fully cancel — most damaging on the widest projections.
LLAMA_SKIP_PATTERNS = ("down_proj",)
LLAMA_SKIP_PATTERNS_AGGRESSIVE = ("down_proj", "gate_proj", "up_proj")


def quantize_model_fp_llama(model, data_loader, *,
                            skip_patterns=LLAMA_SKIP_PATTERNS,
                            **kwargs):
    """
    Llama-aware entry point for quantize_model_fp.

    Identical to quantize_model_fp but defaults to Hadamard+GF4 calibration
    with the Llama skip preset (down_proj kept in FP16). Pass
    skip_patterns=LLAMA_SKIP_PATTERNS_AGGRESSIVE to also keep the SwiGLU
    gate/up projections in FP16, or any custom tuple of name substrings.

    All other quantize_model_fp keyword arguments (e_bits, block_size,
    had_block_size, gf4_variant, device, …) pass through unchanged.

    Example:
        model = quantize_model_fp_llama(
            model, calib_loader,
            block_size=16, e_bits=2, m_bits=1,
            e_bits_scale=4, m_bits_scale=3,
            device=device, had_block_size="auto",
        )
    """
    kwargs.setdefault("Hadamard", True)
    kwargs.setdefault("use_gf4", True)
    return quantize_model_fp(
        model, data_loader,
        extra_skip_patterns=tuple(skip_patterns),
        **kwargs,
    )



def collect_activation_scales(model, calib_loader, device, num_batches=4):
    """
    Collect per-channel max activation magnitude for each linear layer
    by running calibration data through the model with forward hooks.
    Returns dict: layer_name -> per-input-channel max abs activation [in_features]
    """
    act_scales = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            x = inp[0].detach()                      # (B, seq, in) or (B, in)
            x_flat = x.reshape(-1, x.shape[-1])      # (N, in)
            channel_max = x_flat.abs().max(dim=0).values  # (in,)
            if name not in act_scales:
                act_scales[name] = channel_max
            else:
                act_scales[name] = torch.maximum(act_scales[name], channel_max)
        return hook

    # Register hooks on the layers we want to smooth
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(calib_loader):
            x = x.to(device)
            model(x)
            if i + 1 >= num_batches:
                break

    for h in hooks:
        h.remove()

    return act_scales


def smooth_layer(layer, act_scale, alpha=0.5):
    """
    Full SmoothQuant scale: s = act_scale^alpha / w_scale^(1-alpha)
    act_scale: per-input-channel max activation [in_features]
    """
    W = layer.weight.data                              # (out, in)
    w_scale = W.abs().max(dim=0).values.clamp(min=1e-8)  # (in,)
    act_scale = act_scale.clamp(min=1e-8)

    # Joint scale balancing activations and weights
    smooth_scale = (act_scale.pow(alpha) /
                    w_scale.pow(1 - alpha))            # (in,)

    W_smoothed = W / smooth_scale.unsqueeze(0)
    layer.weight.data = W_smoothed
    return smooth_scale


def apply_smoothquant(model, calib_loader, device, alpha=0.5):
    """
    Full SmoothQuant using both activation and weight statistics.
    """
    # Step 1 — collect activation scales from calibration data
    act_scales = collect_activation_scales(model, calib_loader, device)

    # Step 2 — apply smoothing block by block
    for i, block in enumerate(model.transformer.h):

        # QKV projection
        qkv_name = f"transformer.h.{i}.self_attention.query_key_value"
        if qkv_name in act_scales:
            ln  = block.input_layernorm
            qkv = block.self_attention.query_key_value
            scale = smooth_layer(qkv, act_scales[qkv_name], alpha)
            ln.weight.data *= scale
            if ln.bias is not None:
                ln.bias.data *= scale

        # MLP first projection
        fc1_name = f"transformer.h.{i}.mlp.dense_h_to_4h"
        if fc1_name in act_scales:
            ln2 = block.post_attention_layernorm
            fc1 = block.mlp.dense_h_to_4h
            scale2 = smooth_layer(fc1, act_scales[fc1_name], alpha)
            ln2.weight.data *= scale2
            if ln2.bias is not None:
                ln2.bias.data *= scale2

    return model


import torch
import json
from pathlib import Path

def save_for_trt_export(
    model,               # your nn.Module with quantized weights
    layer_scales,        # dict: layer_name -> alpha_eff tensor (per-block)
    layer_biases,        # dict: layer_name -> original bias tensor
    block_size,
    save_dir,
):
    """
    Absorb custom FP4 bias into scale, then save weights + effective scales
    in a format ready for ONNX QDQ injection.

    layer_scales[name] : shape [num_blocks]  (your alpha, per block)
    layer_biases[name] : shape [num_blocks]  (your custom exponent bias)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Standard FP4 E2M1 decode table (fixed bias = 1)
    FP4_STD = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
    )

    weights_out = {}
    scales_out  = {}
    masks_out   = {}
    meta_layers = {}

    for name, param in model.named_parameters():
        if name not in layer_scales:
            # Non-quantized layer — save as-is in FP16
            weights_out[name] = param.data.half()
            continue

        alpha  = layer_scales[name].float()   # [num_blocks]
        bias   = layer_biases[name].long()    # [num_blocks]

        W = param.data.float()
        shape = W.shape
        W_flat = W.view(-1)
        n_blocks = (W_flat.numel() + block_size - 1) // block_size

        alpha_eff_per_element = torch.zeros_like(W_flat)

        for b in range(n_blocks):
            start = b * block_size
            end   = min(start + block_size, W_flat.numel())
            block = W_flat[start:end]

            a     = alpha[b]
            bval  = bias[b].item()

            # Decode table with custom bias: 2^(exp - custom_bias + 1) × mantissa
            # For E2M1: exponents are 0,1,2,3 → values shift by 2^(custom_bias - std_bias)
            bias_shift = 2.0 ** (bval - 1)   # std bias = 1
            fp4_custom = FP4_STD * bias_shift  # custom decode table

            # Per-element ratio: how much did the custom bias inflate each value?
            # Find nearest fp4_custom value for each weight
            abs_block = block.abs()
            sign_block = block.sign()

            dists  = (abs_block.unsqueeze(1) - fp4_custom.unsqueeze(0)).abs()
            idx    = dists.argmin(dim=1)

            custom_decoded = fp4_custom[idx]          # what your method reconstructs
            std_decoded    = FP4_STD[idx]             # what standard FP4 gives

            # Ratio is 0/0 where weight=0 — handle safely
            ratio = torch.where(
                std_decoded.abs() > 1e-8,
                custom_decoded / std_decoded,
                torch.ones_like(custom_decoded),
            )

            # Absorbed scale: scalar per block (take mean of per-element ratios)
            alpha_eff = a * ratio.mean()
            alpha_eff_per_element[start:end] = alpha_eff

        # Reconstruct weights in FP16 using absorbed scale
        # W_reconstructed = alpha_eff_per_element * sign * std_fp4_value
        abs_W = W_flat.abs()
        sign_W = W_flat.sign()
        dists = (abs_W.unsqueeze(1) - FP4_STD.unsqueeze(0)).abs()
        idx   = dists.argmin(dim=1)
        W_std_decoded = FP4_STD[idx] * sign_W  # standard FP4 reconstruction

        W_reconstructed = alpha_eff_per_element * W_std_decoded
        weights_out[name] = W_reconstructed.view(shape).half()

        # Per-block alpha_eff for QDQ scale nodes
        block_scales = []
        for b in range(n_blocks):
            start = b * block_size
            end   = min(start + block_size, W_flat.numel())
            block_scales.append(alpha_eff_per_element[start:end].mean().item())
        scales_out[name] = torch.tensor(block_scales, dtype=torch.float32)

        # Sparsity mask
        masks_out[name] = (param.data != 0)

        meta_layers[name] = {
            "shape":      list(shape),
            "n_blocks":   n_blocks,
            "block_size": block_size,
            "sparsity":   float((param.data == 0).float().mean()),
        }

    torch.save(weights_out, save_dir / "weights_fp16.pt")
    torch.save(scales_out,  save_dir / "scales.pt")
    torch.save(masks_out,   save_dir / "sparsity_masks.pt")

    metadata = {
        "block_size":   block_size,
        "format":       "fp4_bias_absorbed",
        "layers":       meta_layers,
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved to {save_dir}/")
    print(f"  weights_fp16.pt     — {len(weights_out)} layers")
    print(f"  scales.pt           — {len(scales_out)} quantized layers")
    print(f"  sparsity_masks.pt   — {len(masks_out)} layers")
    print(f"  metadata.json       — block_size={block_size}")


def load_for_inspection(save_dir):
    """Load back and verify before export."""
    save_dir = Path(save_dir)
    weights = torch.load(save_dir / "weights_fp16.pt", weights_only=True)
    scales  = torch.load(save_dir / "scales.pt",       weights_only=True)
    masks   = torch.load(save_dir / "sparsity_masks.pt", weights_only=True)
    with open(save_dir / "metadata.json") as f:
        meta = json.load(f)
    return weights, scales, masks, meta
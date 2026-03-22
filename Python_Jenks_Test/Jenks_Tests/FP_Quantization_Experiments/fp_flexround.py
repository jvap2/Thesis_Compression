import torch
import torch.nn as nn
from Quantization_Experiments.brecq import get_reconstruct_blocks, cache_block_input, apply_bias_correction, replace_block
from .quant_layers_fp import convert_to_fp_quant, convert_to_fp_quant_flex, FlexFPQuantizer, apply_layerwise_scales, convert_to_fp_quant_flex_scale, FPScaledLinear, FPScaledConv2d, quantize_to_pow2
from .utils import solve_output_scale_channelwise
import copy
import torch.nn.functional as F 
import os


def capture_s_global_init(block_q: nn.Module) -> dict:
    """Capture s_global values before any optimizer steps."""
    init = {}
    for name, mod in block_q.named_modules():
        if isinstance(mod, (FPScaledLinear, FPScaledConv2d)):
            init[name] = mod.s_global.item()
    return init

def reconstruct_block_fp(block_fp, block_q, inputs, iters=2000, lr=1e-3):

    block_fp.eval()
    for p in block_fp.parameters():
        p.requires_grad = False

    inputs = inputs.detach().clone()

    with torch.no_grad():
        target = block_fp(inputs).detach()

    params = []
    for m in block_q.modules():
        if isinstance(m, FlexFPQuantizer):
            params.append(m.theta)

    optimizer = torch.optim.Adam(params, lr=lr)

    for i in range(iters):

        optimizer.zero_grad()

        pred = block_q(inputs)

        loss = F.mse_loss(pred, target)
        if i%100==0:
            print("Loss: ", loss.item())
        loss.backward()

        optimizer.step()



def brecq_quantize_exp_fp(model, calibration_loader, name, bitwidth=4, geometry=False, batch_size=1024, config=None):
    """
    FP4 quantization with per-channel output scale.
    Uses FlexFPQuantizer with:
    - Hard exponent quantization
    - Optional mantissa STE
    - Per-channel learnable scale
    """

    if config is None:
        config = {
            "first": (4, 3),
            "last": (4, 3),
            "default": (3, 0),
            "conv": (3, 1),
        }

    model.eval()
    device = next(model.parameters()).device
    print("Device:", device)

    # --- get reconstruct blocks ---
    blocks = get_reconstruct_blocks(model, name)
    last_name = blocks[-1][0]
    first_name = blocks[0][0]
    print("Blocks have been reconstructed")

    for block_name, block in blocks:

        # --- cache inputs for calibration ---
        inputs = cache_block_input(
            model,
            block,
            calibration_loader,
            device=device,
            num_batches=2
        )

        # --- deepcopy block and move to device ---
        block_q = copy.deepcopy(block)
        block_q.to(device)

        # --- convert to FP4 wrapper ---
        block_q = convert_to_fp_quant_flex(
            block_q,
            block_name,
            is_first=(block_name == first_name),
            is_last=(block_name == last_name)
        )

        # --- apply quantizer to weights (no optimization) ---
        for name, module in block_q.named_modules():
            if hasattr(module, "flex"):
                # Compute FP4 quantized weights
                w_q = module.flex(module.weight)
                # Replace weight in module
                module.weight.data = w_q
        # apply_layerwise_scales(block, block_q, inputs)
        # --- optional: bias correction ---
        apply_bias_correction(block, block_q, inputs)

        # --- replace block in model ---
        replace_block(model, block, block_q)

    return model


def brecq_quantize_exp_fp_scale(model, calibration_loader, name, bitwidth=4, geometry=False, batch_size=1024, config=None):
    """
    FP4 quantization with per-channel output scale.
    Uses FlexFPQuantizer with:
    - Hard exponent quantization
    - Optional mantissa STE
    - Per-channel learnable scale
    """

    if config is None:
        config = {
            "first": (4, 3),
            "last": (4, 3),
            "default": (3, 0),
            "conv": (3, 1),
        }

    model.eval()
    device = next(model.parameters()).device
    print("Device:", device)

    # --- get reconstruct blocks ---
    blocks = get_reconstruct_blocks(model, name)
    last_name = blocks[-1][0]
    first_name = blocks[0][0]
    print("Blocks have been reconstructed")

    for block_name, block in blocks:

        # --- cache inputs for calibration ---
        inputs = cache_block_input(
            model,
            block,
            calibration_loader,
            device=device,
            num_batches=2
        )

        # --- deepcopy block and move to device ---
        block_q = copy.deepcopy(block)
        block_q.to(device)

        # --- convert to FP4 wrapper ---
        block_q = convert_to_fp_quant_flex_scale(
            block_q,
            block_name,
            is_first=(block_name == first_name),
            is_last=(block_name == last_name)
        )
        for name, mod in block_q.named_modules():
            print(name, type(mod).__name__)
        ## Reconstruction
        for name, mod in block_q.named_modules():
            if isinstance(mod, (FPScaledLinear, FPScaledConv2d)):
                # print(f"  {name}: s1={mod.s1.item():.4f}  "
                    print(f"W_max={mod.weight.abs().max().item():.4f}  "
                    f"u_max={mod.get_u().abs().max().item():.4f}  "
                    f"codebook_max={mod.codebook.abs().max().item():.4f}")
        reconstruct_block_scale(block, block_q, inputs, last_layer = (block_name==last_name))
        # apply_layerwise_scales(block, block_q, inputs)
        # --- optional: bias correction ---
        apply_bias_correction(block, block_q, inputs)

        # --- replace block in model ---
        replace_block(model, block, block_q)

    return model

def clamp_positive_params(module: nn.Module):
    for name, p in module.named_parameters():
        if "s_" in name or "S_" in name:
            p.data.clamp_(1e-6, 1e6)

def reconstruct_block_fp_fp4(block_fp, block_q, inputs, quant_modules, iters=2000, lr=1e-4):

    block_fp.eval()
    for p in block_fp.parameters():
        p.requires_grad = False

    inputs = inputs.detach()

    with torch.no_grad():
        target = block_fp(inputs).detach()

    # Only optimize rounding (NOT scale)
    params = []
    for m in quant_modules:
        if hasattr(m, "delta"):
            m.delta = None  # will be populated in forward

    # (optional) if you later add rounding params, collect here

    optimizer = torch.optim.Adam(
        [p for p in block_q.parameters() if p.requires_grad],
        lr=lr
    )

    for i in range(iters):

        optimizer.zero_grad()

        pred = block_q(inputs)

        # --- reconstruction loss ---
        loss = F.mse_loss(pred, target)

        # --- optional rounding regularizer ---
        round_loss = 0
        for m in quant_modules:
            if hasattr(m, "delta") and m.delta is not None:
                round_loss += torch.mean(torch.abs(m.delta))

        loss = loss + 1e-3 * round_loss

        if i % 100 == 0:
            print(f"Iter {i} | Loss: {loss.item():.6f}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(block_q.parameters(), 1.0)
        optimizer.step()

        # if i % 200 == 0:
        #     for m in quant_modules:
        #         solve_output_scale_channelwise(block_fp, block_q, inputs, m)



# def reconstruct_block_scale(
#     block_fp:    nn.Module,
#     block_q:     nn.Module,
#     inputs:      torch.Tensor,
#     iters:       int   = 1000,
#     lr:          float = 5e-4,
#     batch_size:  int   = 256,
#     lambda_r:    float = 1e-5,
#     beta_warmup: int   = 400,
#     beta_max:    float = 6.0,
#     log_every:   int   = 200,
#     last_layer:  bool  = False,   # activates all three last-layer fixes
#     temperature: float = 1.0,     # KL softening temperature
# ) -> nn.Module:
#     """
#     last_layer=True applies:
#       - s1 frozen (lr=0) — preserves the RTN scale that was good at iter 0
#       - KL loss instead of MSE — correct objective for logits
#       - lambda_r reduced 10x — less regularizer pressure on small layer
#       - fewer iterations — small layer converges fast, more iters = overfit
 
#     Tuning note for last_layer:
#       Start with lambda_r=1e-5. At iter 0, lambda*reg should be < 10% of
#       recon loss. If not, halve lambda_r. The KL loss at iter 0 for a
#       well-calibrated last layer should be in the range [0.01, 0.5].
#       If it starts at 0.001 or below, the layer is essentially lossless
#       at RTN and you should consider skipping reconstruction entirely
#       and just using RTN for this layer.
#     """
#     if last_layer:
#         lambda_r  = lambda_r * 0.1    # 10x smaller regularizer
#         iters     = min(iters, 500)   # small layer, fewer iters
 
#     # Separate s1 from all other parameters
#     s1_params    = [p for n, p in block_q.named_parameters() if n.endswith(".s1") or n == "s1"]
#     other_params = [p for n, p in block_q.named_parameters() if not (n.endswith(".s1") or n == "s1")]
 
#     param_groups = [
#         {"params": other_params, "lr": lr},
#         {"params": s1_params,    "lr": 0.0 if last_layer else lr * 0.1},
#     ]
#     optimizer = torch.optim.Adam(param_groups)
#     N         = inputs.shape[0]
 
#     block_fp.eval()
#     block_q.train()
 
#     for i in range(iters):
#         beta = anneal_beta(i, warmup=beta_warmup, max_beta=beta_max)
#         idx  = torch.randint(0, N, (batch_size,), device=inputs.device)
#         x    = inputs[idx]
 
#         optimizer.zero_grad()
 
#         with torch.no_grad():
#             y_fp = block_fp(x)
 
#         y_q   = block_q(x)
#         loss = reconstruction_loss(y_q, y_fp, use_kl=last_layer, temperature=temperature)
#         # reg   = compute_reg_loss(block_q, beta=beta)
#         # loss  += lambda_r * reg
 
#         loss.backward()
#         optimizer.step()
#         clamp_positive_params(block_q)
 
#         if i % log_every == 0:
#             print(
#                 f"  [iter {i:4d}]  "
#                 f"recon={loss.item():.6f}  "
#                 # f"reg={reg.item():.4f}  "
#                 # f"lambda*reg={lambda_r * reg.item():.6f}  "
#                 f"beta={beta:.2f}"
#             )
 
#     block_q.eval()
#     return block_q

def fp_directional_regularizer(u, codebook, kappa_norm=None):
    u_flat = u.reshape(-1).contiguous()
    idx_hi = torch.searchsorted(codebook, u_flat).clamp(1, len(codebook) - 1)
    idx_lo = idx_hi - 1
    c_lo   = codebook[idx_lo]
    c_hi   = codebook[idx_hi]
    kappa  = (c_hi - c_lo).clamp(min=1e-8)
    if kappa_norm is None:
        kappa_norm = codebook_mean_interval(codebook)
    w      = kappa / kappa_norm
    phi    = ((u_flat - c_lo) / kappa).clamp(0.0, 1.0)
    target = (phi > 0.5).float()
    reg    = (w / kappa) * (phi - target).pow(2)
    return reg.reshape_as(u)

def fp_parameter_regularizer(
    block_q:   nn.Module,
    s_global_init: dict,       # captured before training
    lambda_global: float = 1.0,
    lambda_block:  float = 0.5,
    lambda_S2:     float = 0.1,
) -> torch.Tensor:
    """
    Three-level parameter regularizer matching the scale hierarchy.

    R_global: anchor s_global to its calibrated initialization.
              Strongest regularization — this is the layer-wide anchor.

    R_block:  pull s_block exponents toward their initialized values
              and toward uniformity within each output channel.
              We regularize in log2 space since that's where the
              meaningful variation is for power-of-two scales.

    R_S2:     pull per-element factors toward 1 (RTN baseline).
              Weakest — we want S2 to be free to deviate, just not wildly.
    """
    device = next(block_q.parameters()).device
    total  = torch.tensor(0.0, device=device)

    for name, mod in block_q.named_modules():
        if not isinstance(mod, (FPScaledLinear, FPScaledConv2d)):
            continue

        # ---- s_global anchor ----
        if name in s_global_init:
            ref   = torch.tensor(s_global_init[name], device=device)
            total = total + lambda_global * (mod.s_global - ref).pow(2)

        # ---- s_block in log2 space ----
        log2_block      = torch.log2(mod.s_block.abs().clamp(min=1e-8))
        log2_block_init = torch.log2(
            quantize_to_pow2(mod.s_block.abs().clamp(min=1e-8)).detach()
        )
        # Anchor to initialized exponents + penalize variance across blocks
        total = total + lambda_block * (
            (log2_block - log2_block_init.detach()).pow(2).mean() +
            log2_block.var(dim=-1).mean()
        )

        # ---- S2 toward 1 ----
        total = total + lambda_S2 * (mod.S2 - 1.0).pow(2).mean()

    return total

def reconstruct_block_scale(
    block_fp:       nn.Module,
    block_q:        nn.Module,
    inputs:         torch.Tensor,
    iters:          int   = 2000,
    lr:             float = 1e-3,
    batch_size:     int   = 64,
    lambda_r:       float = 1e-4,
    lambda_global:  float = 1.0,
    lambda_block:   float = 0.5,
    lambda_S2:      float = 0.1,
    log_every:      int   = 200,
    last_layer:     bool  = False,
) -> nn.Module:

    if last_layer:
        iters    = min(iters, 500)
        lambda_r = lambda_r * 0.1

    s_global_init = capture_s_global_init(block_q)

    # Separate parameter groups — s_global gets lower lr for stability
    s_global_params = [p for n, p in block_q.named_parameters() if "s_global" in n]
    other_params    = [p for n, p in block_q.named_parameters() if "s_global" not in n]

    optimizer = torch.optim.Adam([
        {"params": other_params,    "lr": lr},
        {"params": s_global_params, "lr": lr * 0.1 if not last_layer else 0.0},
    ])

    N = inputs.shape[0]
    block_fp.eval()
    block_q.train()
# At the very start of reconstruct_block, before the loop
    for name, mod in block_q.named_modules():
        if isinstance(mod, (FPScaledLinear, FPScaledConv2d)):
            print(f"\n{name}:")
            print(f"  s_global: {mod.s_global.item():.6f}")
            print(f"  s_block min/max: {mod.s_block.min().item():.6f} / {mod.s_block.max().item():.6f}")
            print(f"  S2 min/max: {mod.S2.min().item():.6f} / {mod.S2.max().item():.6f}")
            print(f"  S min/max: {mod.get_S().min().item():.6f} / {mod.get_S().max().item():.6f}")
            print(f"  u min/max: {mod.get_u().min().item():.6f} / {mod.get_u().max().item():.6f}")
            print(f"  W min/max: {mod.weight.min().item():.6f} / {mod.weight.max().item():.6f}")
            # Check for NaN before training even starts
            for pname, p in mod.named_parameters():
                if torch.isnan(p).any():
                    print(f"  NaN in {pname} at initialization!")
    for i in range(iters):
        idx   = torch.randint(0, N, (batch_size,), device=inputs.device)
        x     = inputs[idx]

        optimizer.zero_grad()

        with torch.no_grad():
            y_fp = block_fp(x)

        y_q    = block_q(x)
        recon  = reconstruction_loss(y_q, y_fp, use_kl=last_layer)
        # reg_cb = compute_reg_loss(block_q)
        # reg_p  = fp_parameter_regularizer(
        #              block_q, s_global_init,
        #              lambda_global=lambda_global,
        #              lambda_block=lambda_block,
        #              lambda_S2=lambda_S2,
        #          )
        # loss   = recon + lambda_r * reg_cb + reg_p
        loss = recon

        loss.backward()

        # kappa-scaled gradient for S2
        for mod in block_q.modules():
            if isinstance(mod, (FPScaledLinear, FPScaledConv2d)) and mod.S2.grad is not None:
                kappa_norm = codebook_mean_interval(mod.codebook)
                u          = mod.get_u().detach()
                if isinstance(mod, FPScaledConv2d):
                    u = u  # already [O, flat_dim]
                idx_hi = torch.searchsorted(
                    mod.codebook, u.reshape(-1).contiguous()
                ).clamp(1, len(mod.codebook) - 1)
                kappa  = (mod.codebook[idx_hi] - mod.codebook[idx_hi - 1]).clamp(min=1e-8)
                scale  = (kappa / kappa_norm).reshape_as(mod.S2)
                mod.S2.grad.mul_(scale)

        optimizer.step()
        clamp_positive_params(block_q)

        if i % log_every == 0:
            print(
                f"  [iter {i:4d}]  recon={recon.item():.6f}  "
                # f"reg_cb={reg_cb.item():.4f}  reg_p={reg_p.item():.6f}"
            )

    block_q.eval()
    return block_q


def reconstruction_loss(
    y_q:       torch.Tensor,
    y_fp:      torch.Tensor,
    use_kl:    bool  = False,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    MSE for hidden layers.
    KL divergence between softmax outputs for the classification head.
 
    KL is scale-invariant — it measures whether probability mass lands in
    the same places, not whether raw logit values match numerically.
    This is the correct objective for a layer whose outputs feed directly
    into a softmax, because what matters is the ranking and ratio of logits,
    not their absolute values.
    """
    if not use_kl:
        return F.mse_loss(y_q, y_fp)
 
    p_fp = F.softmax(y_fp / temperature, dim=-1)
    p_q  = F.log_softmax(y_q / temperature, dim=-1)
    return F.kl_div(p_q, p_fp, reduction="batchmean")

def diagnostic_file(block):
    filename="ResNet32_CIFAR100/4_bit/diag.txt"
    quant_modules = [
            m for m in block.modules()
            if isinstance(m, FlexFPQuantizer)
        ]
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a") as f:
        for module in quant_modules:
            with torch.no_grad():
                w = module.weight
                w_q = module(w)

                print("W stats:", w.mean().item(), w.std().item(), file=f)
                print("W_q stats:", w_q.mean().item(), w_q.std().item(), file=f)
                print("Unique values:", w_q.unique().numel(), file=f)
                print("theta:", module.theta.min().item(), module.theta.max().item(), file=f)




 
 
def anneal_beta(step: int, warmup: int = 200, max_beta: float = 6.0) -> float:
    """
    Linear beta warmup from 1 to max_beta over `warmup` steps,
    then held at max_beta.  Mirrors AdaRound's temperature schedule.
    """
    if step < warmup:
        return 1.0 + (max_beta - 1.0) * (step / warmup)
    return max_beta

def fp_directional_regularizer(u, codebook, kappa_norm=None):
    u_flat = u.reshape(-1).contiguous()
    idx_hi = torch.searchsorted(codebook, u_flat).clamp(1, len(codebook) - 1)
    idx_lo = idx_hi - 1
    c_lo   = codebook[idx_lo]
    c_hi   = codebook[idx_hi]
    kappa  = (c_hi - c_lo).clamp(min=1e-8)
    if kappa_norm is None:
        kappa_norm = codebook_mean_interval(codebook)
    w      = kappa / kappa_norm
    phi    = ((u_flat - c_lo) / kappa).clamp(0.0, 1.0)
    target = (phi > 0.5).float()
    reg    = (w / kappa) * (phi - target).pow(2)
    return reg.reshape_as(u)

def compute_reg_loss(block_q: nn.Module) -> torch.Tensor:
    total = None
    for mod in block_q.modules():
        if not isinstance(mod, (FPScaledLinear, FPScaledConv2d)):
            continue
        kappa_norm = codebook_mean_interval(mod.codebook)
        reg  = fp_directional_regularizer(mod.get_u(), mod.codebook, kappa_norm)
        term = reg.sum()
        total = term if total is None else total + term
    return total if total is not None else torch.tensor(0.0)


def fp_boundary_regularizer(
    u:           torch.Tensor,
    codebook:    torch.Tensor,
    beta:        float = 1.0,
    kappa_norm:  torch.Tensor = None,
) -> torch.Tensor:
    """
    Returns a NON-NEGATIVE tensor of the same shape as u.
    Minimizing its sum pushes each element toward its nearest codebook boundary.
 
    phi = 0 or 1  =>  reg = 0              (on a boundary — no penalty)
    phi = 0.5     =>  reg = kappa/kappa_norm  (dead zone center — max penalty)
 
    Relative interval weighting (kappa / kappa_norm):
      - Dead zones (wide intervals): weight > 1, stronger pull toward boundary
      - Dense zones (narrow intervals): weight < 1, weaker pull
      - Mean interval: weight = 1
 
    This is bounded: kappa/kappa_norm has a finite maximum determined by the
    codebook structure (for FP4 E=2 M=1, max ratio is ~8).
    """
    u_flat = u.reshape(-1).contiguous()
 
    idx_hi = torch.searchsorted(codebook, u_flat).clamp(1, len(codebook) - 1)
    idx_lo = idx_hi - 1
    c_lo   = codebook[idx_lo]
    c_hi   = codebook[idx_hi]
    kappa  = (c_hi - c_lo).clamp(min=1e-8)
 
    if kappa_norm is None:
        kappa_norm = codebook_mean_interval(codebook)
 
    # Relative width: dimensionless, bounded, > 1 in dead zones
    w = kappa / kappa_norm
 
    # Normalized position in local interval — differentiable w.r.t. u_flat
    phi = ((u_flat - c_lo) / kappa).clamp(0.0, 1.0)
 
    # Boundary pressure weighted by relative interval width
    reg = w * (1.0 - (2.0 * phi - 1.0).abs().pow(beta))
 
    return reg.reshape_as(u)

def codebook_mean_interval(codebook: torch.Tensor) -> torch.Tensor:
    """Mean width of all adjacent intervals. Used to normalize kappa."""
    gaps = codebook[1:] - codebook[:-1]
    return gaps.mean().clamp(min=1e-8)



def should_reconstruct_last_layer(
    block_fp:  nn.Module,
    block_q:   nn.Module,
    inputs:    torch.Tensor,
    threshold: float = 0.01,
) -> bool:
    """
    Compute KL loss at RTN (before any reconstruction).
    If it's already below threshold, reconstruction will likely hurt more
    than it helps — just keep the RTN quantization for the last layer.
 
    For CIFAR-10 with a well-trained model, a healthy RTN KL is typically
    in [0.05, 0.3]. Below 0.01 means the last layer quantizes cleanly
    without any optimization needed.
    """
    block_fp.eval()
    block_q.eval()
    with torch.no_grad():
        sample = inputs[:64]
        y_fp   = block_fp(sample)
        y_q    = block_q(sample)
        kl     = reconstruction_loss(y_q, y_fp, use_kl=True)
    print(f"  [last layer RTN KL] = {kl.item():.6f}  (threshold={threshold})")
    return kl.item() > threshold
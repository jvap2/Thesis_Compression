import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *


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
        H_blocks.append(H[i:end, i:end])
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

def compute_hessian_blockdiag_model(model, data_loader, device, block_size, num_batches=4):
    H_data = {}
    hook_map = {}

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

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, QuantConv2dFP):
            handles.append(module.register_forward_hook(make_hook(name, module.conv)))
            hook_map[name] = module.conv
        elif isinstance(module, QuantLinearFP):
            handles.append(module.register_forward_hook(make_hook(name, module.linear)))
            hook_map[name] = module.linear
        elif isinstance(module, QuantConv1dFP):
            print(f"  Registering hook for Conv1D: {name}")
            handles.append(module.register_forward_hook(make_hook(name, module.conv1d)))
            hook_map[name] = module.conv1d

    print(f"Total hooks registered: {len(handles)}")
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            x = x.to(device)
            model(x)
            if i + 1 >= num_batches:
                break

    for h in handles:
        h.remove()

    H_final = {}
    for name, blocks in H_data.items():
        H_final[name] = [b / num_batches for b in blocks]

    return H_final



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


# def compute_hessian_diag_model(model, data_loader, device, num_batches=10):
#     """
#     Collect diagonal Hessian (E[x^2]) for all Conv2d and Linear layers (handles Quant wrappers).
#     Conv2d: returns H_diag of shape [out_channels, in_channels * kh * kw]
#     Linear: returns H_diag of shape [out_features, in_features]
#     Returns: dict {inner_module: H_diag}
#     """
#     H_data = {}
#     hook_name_to_submodule = {}

#     def make_hook(name, inner_mod):
#         def hook(mod, inp, out):
#             x = inp[0].detach()

#             if isinstance(inner_mod, nn.Conv2d):
#                 # Unfold input to patches
#                 unfold = nn.Unfold(
#                     kernel_size=inner_mod.kernel_size,
#                     dilation=inner_mod.dilation,
#                     padding=inner_mod.padding,
#                     stride=inner_mod.stride
#                 )
#                 x_unf = unfold(x)  # [B, in_ch*kh*kw, L]
#                 x_unf = x_unf.permute(0, 2, 1)  # [B, L, in_ch*kh*kw]
#                 # Diagonal per output channel: approximate by same diag for each filter
#                 diag = (x_unf ** 2).sum(dim=(0, 1))  # [in_ch*kh*kw]
#                 diag = diag.unsqueeze(0).repeat(inner_mod.out_channels, 1)  # [out_ch, in_ch*kh*kw]

#             elif isinstance(inner_mod, nn.Linear):
#                 x_flat = x.reshape(-1, x.shape[-1])  # [batch*..., in_features]
#                 diag = (x_flat ** 2).sum(dim=0, keepdim=True)  # [1, in_features]
#                 diag = diag.repeat(inner_mod.out_features, 1)  # [out_features, in_features]

#             else:
#                 return

#             if name not in H_data:
#                 H_data[name] = [diag, x.shape[0]]
#                 print(f"  [Hook fired] {name}, diag shape: {diag.shape}")
#             else:
#                 H_data[name][0] += diag
#                 H_data[name][1] += x.shape[0]

#         return hook

#     handles = []
#     for name, module in model.named_modules():
#         if isinstance(module, QuantConv2dFP):
#             # Hook the WRAPPER but compute diag for inner conv
#             handles.append(module.register_forward_hook(make_hook(name, module.conv)))
#             hook_name_to_submodule[name] = module.conv
#         elif isinstance(module, QuantLinearFP):
#             handles.append(module.register_forward_hook(make_hook(name, module.linear)))
#             hook_name_to_submodule[name] = module.linear

#     print(f"=== Total hooks registered: {len(handles)} ===")

#     model.eval()
#     with torch.no_grad():
#         for i, (x, _) in enumerate(data_loader):
#             x = x.to(device)
#             _ = model(x)
#             if i + 1 >= num_batches:
#                 break

#     for h in handles:
#         h.remove()

#     print(f"H_data keys collected: {len(H_data)}")

#     # Finalize H_diag_dict mapping to actual inner modules
#     H_diag_dict = {}
#     for name, (diag_sum, count) in H_data.items():
#         actual_mod = hook_name_to_submodule[name]
#         H_diag_dict[actual_mod] = diag_sum / count

#     print(f"Collected Hessians for {len(H_diag_dict)} modules")
#     return H_diag_dict

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




# def assign_fp4_dynamic(W_abs, alpha, E_bits=2, M_bits=1, bias=None):
#     """
#     Bit-accurate FP assignment with configurable exponent bias.
#     """

#     x = W_abs / alpha.clamp_min(1e-8)
#     device = W_abs.device

#     if bias is None:
#         bias = 2**(E_bits - 1) - 1

#     # Exponent + mantissa grids
#     e_levels = torch.arange(0, 2**E_bits, device=device)
#     m_levels = torch.arange(0, 2**M_bits, device=device)

#     # Correct base using SAME bias everywhere
#     base = 2.0 ** (e_levels - bias)
#     mantissa_factor = 1.0 + (m_levels / (2**M_bits))

#     codebook = (base.unsqueeze(1) * mantissa_factor.unsqueeze(0)).view(-1)

#     # Assign nearest
#     x_expanded = x.unsqueeze(-1)
#     codebook_expanded = codebook.view(*([1]*x.dim()), -1)

#     dist = (x_expanded - codebook_expanded).abs()
#     indices = dist.argmin(dim=-1)

#     M = 2**M_bits
#     exponent = indices // M
#     mantissa = indices % M

#     base_selected = 2.0 ** (exponent - bias)
#     basis = base_selected * (1.0 + mantissa / (2**M_bits))

#     return exponent, mantissa, basis


# def assign_fp4_dynamic(W_abs, alpha, E_bits=2, M_bits=1, bias=None):
#     """
#     Bit-accurate FP assignment with configurable exponent bias.
#     Memory-efficient version using cdist instead of broadcast expansion.
#     """
#     x = W_abs / alpha.clamp_min(1e-8)
#     device = W_abs.device
    
#     if bias is None:
#         bias = 2**(E_bits - 1) - 1

#     # Build codebook — small, e.g. 8 values for E2M1
#     e_levels = torch.arange(0, 2**E_bits, device=device)
#     m_levels = torch.arange(0, 2**M_bits, device=device)
#     base     = 2.0 ** (e_levels - bias)
#     mantissa_factor = 1.0 + (m_levels / (2**M_bits))
#     codebook = (base.unsqueeze(1) * mantissa_factor.unsqueeze(0)).view(-1)  # [C]

#     original_shape = x.shape
#     x_flat = x.reshape(-1, 1)          # [N*block, 1]
#     cb     = codebook.unsqueeze(0)     # [1, C]

#     # cdist never materializes more than [N*block, C] — much smaller
#     # than [N, block, C] from the broadcast approach
#     dist    = torch.cdist(x_flat, cb.T.unsqueeze(0).squeeze(0).unsqueeze(0)
#                           .reshape(-1,1), p=1)
    
#     # Simpler: just use subtraction on the flattened form
#     dist    = (x_flat - cb).abs()      # [N*block, C]
#     indices = dist.argmin(dim=-1)      # [N*block]
    
#     del x_flat, cb, dist              # free immediately
#     torch.cuda.empty_cache()

#     M         = 2**M_bits
#     exponent  = (indices // M).reshape(original_shape)
#     mantissa  = (indices %  M).reshape(original_shape)
    
#     del indices

#     base_selected = 2.0 ** (exponent.float() - bias)
#     basis         = base_selected * (1.0 + mantissa.float() / (2**M_bits))

#     return exponent, mantissa, basis


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

# def assign_fp4_dynamic_vectorized(W_abs, alpha, E_bits=2, M_bits=1, bias=None):
#     """
#     Fully vectorized, shape-safe FP4 assignment.

#     Supports:
#     - W_abs: [N, B, bs] or [N, M]
#     - alpha: [N, B] or [N, M] (broadcastable last dim)
#     - bias:
#         * int (global bias)
#         * tensor [N, B] (per-block bias)

#     Returns:
#         exponent, mantissa, basis (same shape as W_abs)
#     """

#     device = W_abs.device

#     # ------------------------------------------------------------
#     # Normalize input
#     # ------------------------------------------------------------
#     # ensure block view
#     if W_abs.dim() == 2 and alpha.dim() == 2:
#         N, M = W_abs.shape
#         B = alpha.shape[1]
#         bs = M // B

#         W_blocks = W_abs.view(N, B, bs)
#         alpha = alpha.clamp_min(1e-8).unsqueeze(-1)   # [N,B,1]

#         x = W_blocks / alpha                         # [N,B,bs]
#     else:
#         x = W_abs / alpha.clamp_min(1e-8)

#     # ------------------------------------------------------------
#     # Build codebook (same as original, scalar over E/M)
#     # ------------------------------------------------------------
#     e_levels = torch.arange(2**E_bits, device=device)          # [E]
#     m_levels = torch.arange(2**M_bits, device=device)          # [M]

#     mantissa_factor = 1.0 + (m_levels / (2**M_bits))           # [M]
#     codebook = (2.0 ** (e_levels.unsqueeze(1) - 0.0)) * mantissa_factor.unsqueeze(0)
#     codebook = codebook.reshape(-1)                            # [K]

#     K = codebook.numel()

#     # ------------------------------------------------------------
#     # Bias handling (THIS is the main fix)
#     # ------------------------------------------------------------
#     if bias is None:
#         bias_t = torch.tensor(2**(E_bits - 1) - 1, device=device)
#     elif isinstance(bias, (int, float)):
#         bias_t = torch.tensor(bias, device=device)
#     else:
#         bias_t = bias.to(device)

#     # ------------------------------------------------------------
#     # Correct exponent grid (FIXED BROADCASTING)
#     # ------------------------------------------------------------
#     e_levels = torch.arange(2**E_bits, device=device).view(-1, 1)  # [E,1]

#     # broadcast-safe bias subtraction:
#     # if scalar -> works
#     # if tensor [N,B] -> we expand later per-x
#     base_exp = e_levels  # [E,1]

#     # ------------------------------------------------------------
#     # Expand x for distance computation
#     # ------------------------------------------------------------
#     x_exp = x.unsqueeze(-1)                        # [..., 1]
#     cb_exp = codebook.view(*([1] * x.dim()), K)    # [..., K]

#     dist = (x_exp - cb_exp).abs()
#     indices = dist.argmin(dim=-1)

#     # ------------------------------------------------------------
#     # Decode indices -> exponent/mantissa
#     # ------------------------------------------------------------
#     M = 2**M_bits
#     exponent = indices // M
#     mantissa = indices % M

#     # ------------------------------------------------------------
#     # IMPORTANT FIX: correct basis reconstruction
#     # (bias must match exponent shape exactly)
#     # ------------------------------------------------------------

#     # Handle scalar vs tensor bias safely
#     if isinstance(bias_t, torch.Tensor) and bias_t.numel() > 1:
#         # per-element bias [N,B] or [N,M]
#         bias_exp = bias_t.unsqueeze(-1).expand_as(exponent)
#     else:
#         bias_exp = bias_t

#     base_selected = 2.0 ** (exponent.float() - bias_exp.float())
#     basis = base_selected * (1.0 + mantissa.float() / (2**M_bits))

#     return exponent, mantissa, basis


# import torch

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
# =========================================================
# 🔹 HG-STYLE OPTIMIZATION (mask-aware)
# =========================================================
# def solve_alpha_blockwise_HG(W, H, G, mask, block_size):
#     print(W.shape)
#     N, M = W.shape
#     alpha = torch.zeros(M, device=W.device)
#     for i in range(0, M, block_size):
#         idx = slice(i, min(i + block_size, M))
#         H_block = H[:, idx] * mask[:, idx]
#         G_block = G[idx, :] * mask[idx, :]
#         W_block = W[:, idx] * mask[:, idx]
#         H_G = H_block @ G_block
#         num = (H_G * W_block).sum()
#         den = (H_G * H_G).sum() + 1e-8
#         alpha[idx] = num / den
#     return alpha

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

#     num_blocks = (M + block_size - 1) // block_size

#     # Output tensors
#     W_q       = torch.zeros_like(W_mat)
#     alpha_out = torch.zeros_like(W_mat)
#     bias_out  = torch.zeros_like(W_mat)

#     default_bias    = 2 ** (e_bits - 1) - 1
#     bias_radius     = max(1, 2 ** (e_bits - 2))
#     bias_candidates = list(range(default_bias - bias_radius,
#                                  default_bias + bias_radius + 1))

#     for block_idx, i in enumerate(range(0, M, block_size)):
#         end = min(i + block_size, M)
#         k   = end - i

#         # [N, k]
#         w_block = W_abs[:, i:end]
#         m_block = mask_mat[:, i:end]
#         s_block = sign_mat[:, i:end]

#         H_block = H_blocks_layer[block_idx].to(device)  # [k, k]
#         if H_block.shape[0] != k:
#             H_block = H_block[:k, :k]

#         w_eff = w_block * m_block  # [N, k]

#         # Rows where block is fully pruned
#         pruned = m_block.sum(dim=1) < 1e-8  # [N]

#         # --- Alpha initialization [N] ---
#         w_sq_mean = (w_eff ** 2).mean(dim=1).clamp(min=1e-8)
#         alpha     = torch.sqrt(w_sq_mean).clamp(min=1e-4)  # [N]

#         alpha_min = 0.05 * w_eff.abs().mean(dim=1).clamp(min=1e-8)  # [N]
#         alpha_max = 20.0 * w_eff.abs().mean(dim=1).clamp(min=1e-8)  # [N]

#         # Precompute H @ w_eff for all rows: [N, k]
#         Hw = w_eff @ H_block.T  # [N, k]

#         best_loss  = torch.full((N,), float('inf'), device=device)
#         best_alpha = alpha.clone()
#         best_bias  = torch.full((N,), default_bias, device=device, dtype=torch.long)
#         best_b     = torch.zeros_like(w_eff)

#         for bias_candidate in bias_candidates:

#             alpha_tmp = alpha.clone()  # [N]

#             for _ in range(5):
#                 _, _, b = assign_fp4_dynamic_batched(
#                     w_block,
#                     alpha_tmp,
#                     e_bits,
#                     m_bits,
#                     bias=bias_candidate
#                 )

#                 b_eff = b * m_block  # [N, k]

#                 Hb  = b_eff @ H_block.T                    # [N, k]
#                 num = (b_eff * Hw).sum(dim=1)              # [N]
#                 den = (b_eff * Hb).sum(dim=1) + 1e-8      # [N]

#                 alpha_new = num / den                       # [N]
#                 alpha_tmp = torch.clamp(alpha_new,
#                                         min=alpha_min,
#                                         max=alpha_max)

#             # Loss per row [N]
#             residual = w_eff - alpha_tmp.unsqueeze(1) * b_eff  # [N, k]
#             Hr       = residual @ H_block.T                     # [N, k]
#             loss     = (residual * Hr).sum(dim=1)               # [N]

#             improved   = loss < best_loss
#             best_loss  = torch.where(improved, loss, best_loss)
#             best_alpha = torch.where(improved, alpha_tmp, best_alpha)
#             best_bias  = torch.where(
#                 improved,
#                 torch.full_like(best_bias, bias_candidate),
#                 best_bias
#             )
#             best_b = torch.where(
#                 improved.unsqueeze(1).expand_as(b_eff),
#                 b_eff,
#                 best_b
#             )

#         # --- Quantize scale [N] ---
#         alpha_q = quantize_scale_batched(best_alpha, e_bits_scale, m_bits_scale)

#         # --- Final basis recompute with quantized alpha ---
#         _, _, b_final = assign_fp4_dynamic_batched(
#             w_block,
#             alpha_q,
#             e_bits,
#             m_bits,
#             bias_per_row=best_bias
#         )

#         w_hat = alpha_q.unsqueeze(1) * b_final  # [N, k]
#         w_hat = w_hat * m_block * s_block

#         # Zero out pruned rows
#         w_hat[pruned]  = 0.0
#         alpha_q[pruned] = 1.0

#         W_q[:, i:end]       = w_hat
#         alpha_out[:, i:end] = alpha_q.unsqueeze(1).expand(-1, k)
#         bias_out[:, i:end]  = best_bias.unsqueeze(1).float().expand(-1, k)

#     # --- Reshape back ---
#     if W.dim() == 4:
#         W_q       = W_q.view_as(W)
#         alpha_out = alpha_out.view_as(W)
#         bias_out  = bias_out.view_as(W)

#     # --- Final FP decomposition ---
#     sign_f  = torch.sign(W_q)
#     W_abs_f = W_q.abs()

#     if W.dim() == 4:
#         W_mat2    = W_abs_f.view(W.shape[0], -1)
#         alpha_mat = alpha_out.view(W.shape[0], -1)
#         bias_mat  = bias_out.view(W.shape[0], -1)
#     else:
#         W_mat2    = W_abs_f
#         alpha_mat = alpha_out
#         bias_mat  = bias_out

#     # Initialize as long for FP index outputs
#     e_out = torch.zeros_like(W_mat2, dtype=torch.long)
#     m_out = torch.zeros_like(W_mat2, dtype=torch.long)

#     for block_idx, i in enumerate(range(0, M, block_size)):
#         end = min(i + block_size, M)

#         bias_col  = bias_mat[:, i].long()   # [N]
#         alpha_col = alpha_mat[:, i]         # [N]
#         w_col     = W_mat2[:, i:end]        # [N, k]

#         for bias_val in bias_col.unique():
#             rows = (bias_col == bias_val).nonzero(as_tuple=True)[0]

#             e_b, m_b, _ = assign_fp4_dynamic_batched(
#                 w_col[rows],
#                 alpha_col[rows],
#                 e_bits,
#                 m_bits,
#                 bias=int(bias_val.item())
#             )

#             e_out[rows, i:end] = e_b
#             m_out[rows, i:end] = m_b

#     if W.dim() == 4:
#         e_out = e_out.view_as(W)
#         m_out = m_out.view_as(W)

#     return alpha_out, e_out, m_out, sign_f, bias_out


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

#     # Free W and mask early — we only need W_mat, mask_mat, sign_mat, W_abs
#     del W, mask
#     torch.cuda.empty_cache()

#     num_blocks = (M + block_size - 1) // block_size

#     # Keep output tensors on CPU to save VRAM, move to GPU only at the end
#     W_q       = torch.zeros_like(W_mat, device='cpu')
#     alpha_out = torch.zeros_like(W_mat, device='cpu')
#     bias_out  = torch.zeros_like(W_mat, device='cpu')

#     default_bias    = 2 ** (e_bits - 1) - 1
#     bias_radius     = max(1, 2 ** (e_bits - 2))
#     bias_candidates = list(range(default_bias - bias_radius,
#                                  default_bias + bias_radius + 1))

#     for block_idx, i in enumerate(range(0, M, block_size)):
#         end = min(i + block_size, M)
#         k   = end - i

#         w_block = W_abs[:, i:end]
#         m_block = mask_mat[:, i:end]
#         s_block = sign_mat[:, i:end]

#         H_block = H_blocks_layer[block_idx].to(device)
#         if H_block.shape[0] != k:
#             H_block = H_block[:k, :k]

#         w_eff = w_block * m_block
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

#                 # Free inner loop intermediates
#                 del b, Hb, num, den, alpha_new
            
#             b_eff = b_eff  # already computed in last inner iteration
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

#             # Free outer loop intermediates immediately
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

#     # Move final outputs to GPU for the decomposition pass
#     W_q       = W_q.to(device)
#     alpha_out = alpha_out.to(device)
#     bias_out  = bias_out.to(device)

#     # --- Reshape back ---
#     layer_W = layer.weight.data.to(device)
#     if layer_W.dim() == 4:
#         W_q       = W_q.view_as(layer_W)
#         alpha_out = alpha_out.view_as(layer_W)
#         bias_out  = bias_out.view_as(layer_W)

#     sign_f  = torch.sign(W_q)
#     W_abs_f = W_q.abs()

#     if layer_W.dim() == 4:
#         W_mat2    = W_abs_f.view(layer_W.shape[0], -1)
#         alpha_mat = alpha_out.view(layer_W.shape[0], -1)
#         bias_mat  = bias_out.view(layer_W.shape[0], -1)
#     else:
#         W_mat2    = W_abs_f
#         alpha_mat = alpha_out
#         bias_mat  = bias_out


#     e_out = torch.zeros_like(W_mat2, dtype=torch.long)
#     m_out = torch.zeros_like(W_mat2, dtype=torch.long)

#     for block_idx, i in enumerate(range(0, M, block_size)):
#         end = min(i + block_size, M)

#         bias_col  = bias_mat[:, i].long()
#         alpha_col = alpha_mat[:, i]
#         w_col     = W_mat2[:, i:end]

#         for bias_val in bias_col.unique():
#             rows = (bias_col == bias_val).nonzero(as_tuple=True)[0]
#             e_b, m_b, _ = assign_fp4_dynamic_batched(
#                 w_col[rows], alpha_col[rows], e_bits, m_bits,
#                 bias=int(bias_val.item()))
#             e_out[rows, i:end] = e_b
#             m_out[rows, i:end] = m_b
#             del e_b, m_b

#         del bias_col, alpha_col, w_col
#         torch.cuda.empty_cache()

#     if layer_W.dim() == 4:
#         e_out = e_out.view_as(layer_W)
#         m_out = m_out.view_as(layer_W)
#     del layer_W
#     return alpha_out, e_out, m_out, sign_f, bias_out


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
    W = layer.weight.data.to(device)
    mask = (W != 0).float()

    if W.dim() == 4:
        W_mat = W.view(W.shape[0], -1)
        mask_mat = mask.view(W.shape[0], -1)
    else:
        W_mat = W
        mask_mat = mask

    N, M = W_mat.shape

    sign_mat = torch.sign(W_mat)
    W_abs    = W_mat.abs()

    # Free W and mask early
    del W, mask
    torch.cuda.empty_cache()

    # All output tensors kept on CPU to save VRAM
    W_q       = torch.zeros(N, M, dtype=torch.float32)   # CPU
    alpha_out = torch.zeros(N, M, dtype=torch.float32)   # CPU
    bias_out  = torch.zeros(N, M, dtype=torch.float32)   # CPU

    default_bias    = 2 ** (e_bits - 1) - 1
    bias_radius     = max(1, 2 ** (e_bits - 2))
    bias_candidates = list(range(default_bias - bias_radius,
                                 default_bias + bias_radius + 1))

    # ----------------------------------------------------------------
    # First pass — block-wise alpha/bias optimisation
    # ----------------------------------------------------------------
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

        # Free per-block intermediates before final recompute
        del Hw, alpha, alpha_min, alpha_max, w_sq_mean
        torch.cuda.empty_cache()

        alpha_q = quantize_scale_batched(best_alpha, e_bits_scale, m_bits_scale)

        _, _, b_final = assign_fp4_dynamic_batched(
            w_block, alpha_q, e_bits, m_bits, bias_per_row=best_bias)

        w_hat = alpha_q.unsqueeze(1) * b_final
        w_hat = w_hat * m_block * s_block
        w_hat[pruned]   = 0.0
        alpha_q[pruned] = 1.0

        # Store results on CPU immediately to free GPU memory
        W_q[:, i:end]       = w_hat.cpu()
        alpha_out[:, i:end] = alpha_q.unsqueeze(1).expand(-1, k).cpu()
        bias_out[:, i:end]  = best_bias.unsqueeze(1).float().expand(-1, k).cpu()

        # Free everything from this block before next iteration
        del w_block, m_block, s_block, w_eff, pruned
        del H_block, best_loss, best_alpha, best_bias, best_b
        del alpha_q, b_final, w_hat
        torch.cuda.empty_cache()

    # Free GPU tensors from first pass
    del W_abs, sign_mat, mask_mat, W_mat
    torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # Second decomposition pass — entirely on CPU
    # Compute e, m on CPU then reconstruct weight_q on CPU
    # Only one final tensor is moved to GPU
    # ----------------------------------------------------------------

    sign_f  = torch.sign(W_q)     # CPU [N, M]
    W_abs_f = W_q.abs()           # CPU [N, M]
    del W_q

    # Store original shape for reshape at end
    layer_W        = layer.weight.data
    original_shape = layer_W.shape

    if layer_W.dim() == 4:
        N4        = original_shape[0]
        W_abs_f   = W_abs_f.view(N4, -1)
        alpha_out = alpha_out.view(N4, -1)
        bias_out  = bias_out.view(N4, -1)
        sign_f    = sign_f.view(N4, -1)

    # e_out and m_out stay on CPU throughout
    e_out = torch.zeros(W_abs_f.shape, dtype=torch.long)    # CPU
    m_out = torch.zeros(W_abs_f.shape, dtype=torch.long)    # CPU

    for block_idx, i in enumerate(range(0, M, block_size)):
        end = min(i + block_size, M)

        # Bring only this block slice to GPU
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
    # Final FP reconstruction on CPU
    # weight_q = sign * alpha * 2^(e - bias) * (1 + m / 2^m_bits)
    # Only this single tensor is moved to GPU
    # ----------------------------------------------------------------
    bias_mat = bias_out.float()                              # CPU [N, M]
    base     = alpha_out * (2.0 ** (e_out.float() - bias_mat))
    fine     = base * m_out.float() / (2 ** m_bits) if m_bits > 0 else 0.0
    weight_q_cpu = (base + fine) * sign_f

    del alpha_out, bias_out, bias_mat, e_out, m_out, base, sign_f
    if m_bits > 0:
        del fine

    # Reshape back to original weight shape if needed
    if layer_W.dim() == 4:
        weight_q_cpu = weight_q_cpu.view(original_shape)

    # Single GPU transfer — ~67MB for LLaMA's largest layer vs ~335MB before
    return weight_q_cpu.to(device)

# def reconstruct_layer_fp_blockdiag_scaled_v4_fast(
#     layer, H_blocks_layer, block_size,
#     e_bits, m_bits, e_bits_scale, m_bits_scale, device
# ):
#     W = layer.weight.data.to(device)
#     mask = (W.abs() > 1e-9).float()

#     if W.dim() == 4:
#         W_mat    = W.view(W.shape[0], -1)
#         mask_mat = mask.view(W.shape[0], -1)
#     else:
#         W_mat    = W
#         mask_mat = mask

#     N, M = W_mat.shape
#     sign  = torch.sign(W_mat)
#     W_abs = W_mat.abs()

#     # --- Stack all Hessian blocks into a single tensor [n_blocks, bs, bs] ---
#     n_blocks   = (M + block_size - 1) // block_size
#     H_stacked  = torch.stack(
#         [H_blocks_layer[b].to(device) for b in range(n_blocks)]
#     )  # [n_blocks, block_size, block_size]

#     # --- Pad W_abs and mask to be divisible by block_size ---
#     pad = (block_size - M % block_size) % block_size
#     W_p    = F.pad(W_abs,    (0, pad))   # [N, M+pad]
#     mask_p = F.pad(mask_mat, (0, pad))

#     # --- Reshape to [N, n_blocks, block_size] ---
#     W_blocks    = W_p.view(N, n_blocks, block_size)     # [N, B, bs]
#     mask_blocks = mask_p.view(N, n_blocks, block_size)

#     default_bias = 2 ** (e_bits - 1) - 1
#     bias_radius  = max(1, 2 ** (e_bits - 2))
#     bias_candidates = list(range(default_bias - bias_radius,
#                                  default_bias + bias_radius + 1))

#     # Pre-build codebook once — shape [K] where K = 2^e_bits * 2^m_bits
#     e_levels = torch.arange(0, 2**e_bits, device=device)
#     m_levels = torch.arange(0, 2**m_bits, device=device)

#     alpha_out = torch.zeros(N, n_blocks, device=device)
#     best_bias_out = torch.full((N, n_blocks), default_bias,
#                                 device=device, dtype=torch.long)

#     # --- alpha init: RMS per block ---
#     rms = W_blocks.pow(2).mean(dim=-1).sqrt().clamp(min=1e-4)  # [N, B]
#     alpha_cur = rms.clone()

#     # --- Vectorized bias search ---
#     best_loss = torch.full((N, n_blocks), float('inf'), device=device)

#     for bias_cand in bias_candidates:
#         alpha_tmp = alpha_cur.clone()  # [N, B]

#         for _ in range(5):
#             # Build codebook for this bias
#             base = 2.0 ** (e_levels.float() - bias_cand)       # [E]
#             mf   = 1.0 + m_levels.float() / (2 ** m_bits)      # [M]
#             cb   = (base.unsqueeze(1) * mf.unsqueeze(0)).view(-1)  # [K]

#             # Normalized weights: [N, B, bs, 1]
#             x_norm = (W_blocks / alpha_tmp.unsqueeze(-1).clamp(min=1e-8)
#                       ).unsqueeze(-1)
#             # Distance to codebook: [N, B, bs, K]
#             dist = (x_norm - cb.view(1, 1, 1, -1)).abs()
#             idx  = dist.argmin(dim=-1)          # [N, B, bs]
#             b    = cb[idx]                      # [N, B, bs] — basis values

#             # Hessian-weighted alpha update (vectorized over N)
#             # H_stacked: [n_blocks, bs, bs]
#             # b, W_blocks: [N, n_blocks, bs]
#             # b_eff = b * mask_blocks
#             b_eff = b * mask_blocks             # [N, B, bs]
#             w_eff = W_blocks * mask_blocks

#             # Hb = einsum('bij,nbj->nbi', H_stacked, b_eff)
#             Hb = torch.einsum('bij,nbj->nbi', H_stacked, b_eff)   # [N, B, bs]
#             Hw = torch.einsum('bij,nbj->nbi', H_stacked, w_eff)

#             num = (b_eff * Hw).sum(dim=-1)      # [N, B]
#             den = (b_eff * Hb).sum(dim=-1) + 1e-8

#             alpha_new = (num / den).clamp(min=1e-6)

#             # Stabilize
#             alpha_min = 0.05 * W_blocks.abs().mean(dim=-1)
#             alpha_max = 20.0 * W_blocks.abs().mean(dim=-1)
#             alpha_tmp = alpha_new.clamp(min=alpha_min, max=alpha_max)

#         # Evaluate loss for this bias candidate
#         recon   = alpha_tmp.unsqueeze(-1) * b          # [N, B, bs]
#         res     = w_eff - recon * mask_blocks
#         # Diagonal H loss: sum_i h_ii * res_i^2
#         H_diag  = torch.diagonal(H_stacked, dim1=-2, dim2=-1)  # [B, bs]
#         loss    = (H_diag.unsqueeze(0) * res.pow(2)).sum(dim=-1)  # [N, B]

#         improved = loss < best_loss
#         best_loss = torch.where(improved, loss, best_loss)
#         alpha_out = torch.where(improved, alpha_tmp, alpha_out)
#         best_bias_out = torch.where(
#             improved,
#             torch.full_like(best_bias_out, bias_cand),
#             best_bias_out
#         )

#     # --- Quantize alpha ---
#     alpha_q_flat = torch.stack([
#         quantize_scale(alpha_out[n, b].item(), e_bits_scale, m_bits_scale)
#         for n in range(N) for b in range(n_blocks)
#     ]).view(N, n_blocks)

#     # --- Final FP assignment ---
#     alpha_expanded = alpha_q_flat.unsqueeze(-1).expand_as(W_blocks).to(device)
#     base = 2.0 ** (e_levels.float() - default_bias)
#     mf   = 1.0 + m_levels.float() / (2 ** m_bits)
#     cb   = (base.unsqueeze(1) * mf.unsqueeze(0)).view(-1)
#     print(W_blocks.device, alpha_expanded.device, cb.device)
#     x_norm = (W_blocks / alpha_expanded.clamp(min=1e-8)).unsqueeze(-1)
#     dist   = (x_norm - cb.view(1, 1, 1, -1)).abs()
#     idx    = dist.argmin(dim=-1)
#     b_final = cb[idx]

#     W_q = (sign.view(N, n_blocks, block_size)
#            * alpha_expanded * b_final * mask_blocks)
#     W_q = W_q.view(N, -1)[:, :M]

#     # Recover e, m for return
#     K = len(cb)
#     M_bits_count = 2 ** m_bits
#     e_out = (idx // M_bits_count).view(N, -1)[:, :M].view_as(W)
#     m_out = (idx  % M_bits_count).view(N, -1)[:, :M].view_as(W)

#     alpha_ret = alpha_expanded.view(N, -1)[:, :M].view_as(W)
#     sign_ret  = sign.view_as(W)

#     return alpha_ret, e_out, m_out, sign_ret, best_bias_out

# def reconstruct_layer_fp_blockdiag_scaled_v4_fast(
#     layer,
#     H_blocks_layer,
#     block_size: int,
#     e_bits: int,
#     m_bits: int,
#     e_bits_scale: int,
#     m_bits_scale: int,
#     device,
# ):
#     """
#     Drop-in replacement for reconstruct_layer_fp_blockdiag_scaled_v4.
 
#     Returns
#     -------
#     alpha_out : Tensor  same shape as layer.weight
#     e_out     : Tensor  same shape as layer.weight  (long)
#     m_out     : Tensor  same shape as layer.weight  (long)
#     sign_out  : Tensor  same shape as layer.weight
#     bias_out  : Tensor  same shape as layer.weight  (long)
#                 *** per-element, matching what calibrate_Hessian_scaled
#                     expects when it computes
#                         base = alpha * (2.0 ** (e.float() - bias))  ***
 
#     Speedups over v4
#     ----------------
#     * The double Python loop (for row / for block) is replaced by
#       batched tensor ops over [N, n_blocks, block_size] simultaneously.
#     * The codebook is built once per bias candidate (not once per block).
#     * quantize_scale is replaced by quantize_scale_tensor which works on
#       the full [N, n_blocks] alpha tensor at once.
#     * The final e/m decomposition iterates only over unique bias values
#       (usually 1-3) rather than over every block.
#     """
 
#     W    = layer.weight.data.to(device)
#     mask = (W.abs() > 1e-9).float()
 
#     if W.dim() == 4:
#         W_mat    = W.view(W.shape[0], -1)
#         mask_mat = mask.view(W.shape[0], -1)
#     else:
#         W_mat    = W
#         mask_mat = mask
 
#     N, M = W_mat.shape
 
#     sign_mat = torch.sign(W_mat)
#     W_abs    = W_mat.abs()
 
#     # ------------------------------------------------------------------
#     # Pad to an exact multiple of block_size
#     # ------------------------------------------------------------------
#     pad      = (block_size - M % block_size) % block_size
#     W_p      = F.pad(W_abs,    (0, pad)).contiguous()   # [N, M_pad]
#     mask_p   = F.pad(mask_mat, (0, pad)).contiguous()
#     sign_p   = F.pad(sign_mat, (0, pad)).contiguous()
 
#     M_pad    = M + pad
#     n_blocks = M_pad // block_size
 
#     # Reshape to blocks: [N, n_blocks, block_size]
#     W_blocks    = W_p   .view(N, n_blocks, block_size)
#     mask_blocks = mask_p.view(N, n_blocks, block_size)
 
#     # Dead blocks (all zero / all padding)  [N, n_blocks]
#     block_dead = (mask_blocks.sum(dim=-1) == 0)
 
#     # ------------------------------------------------------------------
#     # Stack Hessian blocks: [n_blocks, block_size, block_size]
#     # Guard against size mismatches (last block may be smaller)
#     # ------------------------------------------------------------------
#     H_list = []
#     for b in range(n_blocks):
#         H_b = H_blocks_layer[b].to(device)          # [bs, bs] or smaller
#         if H_b.shape[0] != block_size:
#             # Pad small trailing block
#             diff = block_size - H_b.shape[0]
#             H_b  = F.pad(H_b, (0, diff, 0, diff))
#         H_list.append(H_b)
#     H_stacked = torch.stack(H_list)                  # [B, bs, bs]
 
#     # Diagonal of each Hessian block: [B, bs]
#     H_diag = torch.diagonal(H_stacked, dim1=-2, dim2=-1)  # [B, bs]
 
#     # ------------------------------------------------------------------
#     # Alpha initialisation: per-block RMS  [N, B]
#     # ------------------------------------------------------------------
#     w_mean_sq  = (W_blocks.pow(2) * mask_blocks).sum(dim=-1)          # [N,B]
#     w_count    = mask_blocks.sum(dim=-1).clamp(min=1)                  # [N,B]
#     alpha      = (w_mean_sq / w_count).sqrt().clamp(min=1e-4)         # [N,B]
 
#     # ------------------------------------------------------------------
#     # Stabilisation bounds (computed once)  [N, B]
#     # ------------------------------------------------------------------
#     w_mean     = (W_blocks * mask_blocks).sum(dim=-1) / w_count        # [N,B]
#     alpha_min  = (0.05 * w_mean).clamp(min=1e-6)
#     alpha_max  = (20.0 * w_mean).clamp(min=1e-4)
 
#     # ------------------------------------------------------------------
#     # Bias search
#     # ------------------------------------------------------------------
#     default_bias = 2 ** (e_bits - 1) - 1
#     bias_radius  = max(1, 2 ** (e_bits - 2))
#     bias_range   = list(range(default_bias - bias_radius,
#                                default_bias + bias_radius + 1))
 
#     best_loss  = torch.full((N, n_blocks), float('inf'), device=device)
#     best_alpha = alpha.clone()                                         # [N,B]
#     best_bias  = torch.full((N, n_blocks), default_bias,
#                              device=device, dtype=torch.long)
 
#     # h_imp: [1, B, bs]  — broadcast over N
#     h_imp = H_diag.unsqueeze(0)
 
#     # w_eff: [N, B, bs]
#     w_eff = W_blocks * mask_blocks
 
#     for bias_cand in bias_range:
#         codebook = _build_codebook(e_bits, m_bits, bias_cand, device)  # [K]
 
#         alpha_tmp = alpha.clone()    # [N, B]
 
#         # ---- 5-iteration alpha refinement ----
#         for _ in range(5):
#             x_norm = (W_blocks
#                       / alpha_tmp.unsqueeze(-1).clamp(min=1e-8))       # [N,B,bs]
#             dist   = (x_norm.unsqueeze(-1)
#                       - codebook.view(1, 1, 1, -1)).abs()              # [N,B,bs,K]
#             idx    = dist.argmin(dim=-1)                               # [N,B,bs]
#             b      = codebook[idx]                                     # [N,B,bs]
 
#             # Hessian-weighted diagonal OBS update
#             # num_i = h_i * b_i * w_i,  den_i = h_i * b_i^2
#             b_eff  = b * mask_blocks                                   # [N,B,bs]
#             num    = (h_imp * b_eff * w_eff).sum(dim=-1)               # [N,B]
#             den    = (h_imp * b_eff * b_eff).sum(dim=-1) + 1e-8       # [N,B]
#             alpha_tmp = (num / den).clamp(min=1e-6)                    # [N,B]
#             alpha_tmp = alpha_tmp.clamp(min=alpha_min, max=alpha_max)
 
#         # ---- Evaluate diagonal-H loss ----
#         x_norm  = (W_blocks
#                    / alpha_tmp.unsqueeze(-1).clamp(min=1e-8))
#         b       = codebook[
#                     (x_norm.unsqueeze(-1)
#                      - codebook.view(1, 1, 1, -1)).abs().argmin(dim=-1)
#                   ]                                                     # [N,B,bs]
#         b_eff   = b * mask_blocks
#         recon   = alpha_tmp.unsqueeze(-1) * b_eff                      # [N,B,bs]
#         residual = w_eff - recon                                        # [N,B,bs]
#         loss     = (h_imp * residual.pow(2)).sum(dim=-1)               # [N,B]
 
#         # Update bests
#         improved   = loss < best_loss
#         best_loss  = torch.where(improved, loss,       best_loss)
#         best_alpha = torch.where(improved, alpha_tmp,  best_alpha)
#         best_bias  = torch.where(
#             improved,
#             torch.full_like(best_bias, bias_cand),
#             best_bias
#         )                                                              # [N,B]
 
#     # ------------------------------------------------------------------
#     # Quantise alpha  [N, B] → [N, B]
#     # ------------------------------------------------------------------
#     alpha_q = quantize_scale_tensor(best_alpha, e_bits_scale, m_bits_scale)
 
#     # ------------------------------------------------------------------
#     # Final FP decomposition — iterate over unique bias values only
#     # (typically 1-3, not N*B)
#     # ------------------------------------------------------------------
#     W_q_flat    = torch.zeros(N, M_pad, device=device)
#     alpha_flat  = torch.zeros(N, M_pad, device=device)
#     e_flat      = torch.zeros(N, M_pad, device=device, dtype=torch.long)
#     m_flat      = torch.zeros(N, M_pad, device=device, dtype=torch.long)
#     # bias per element — will be filled below [N, M_pad]
#     bias_flat   = torch.full((N, M_pad), default_bias,
#                               device=device, dtype=torch.long)
 
#     M_count = 2 ** m_bits
 
#     for bias_val in best_bias.unique().tolist():
#         bias_val = int(bias_val)
#         codebook = _build_codebook(e_bits, m_bits, bias_val, device)   # [K]
 
#         # Which (row, block) pairs use this bias?  [N, B]
#         bmask = (best_bias == bias_val)
 
#         # alpha for these pairs, 0 elsewhere  [N, B]
#         aq = alpha_q * bmask.float()                                   # [N,B]
 
#         # Lookup for all blocks (inactive ones will be masked out)
#         x_norm = (W_blocks
#                   / aq.unsqueeze(-1).clamp(min=1e-8))                  # [N,B,bs]
#         dist   = (x_norm.unsqueeze(-1)
#                   - codebook.view(1, 1, 1, -1)).abs()                  # [N,B,bs,K]
#         idx    = dist.argmin(dim=-1)                                   # [N,B,bs]
#         basis  = codebook[idx]                                         # [N,B,bs]
 
#         exp_t  = (idx // M_count)                                      # [N,B,bs]
#         mant_t = (idx  % M_count)                                      # [N,B,bs]
 
#         # Reconstructed weights (with sign)
#         sign_b = sign_p.view(N, n_blocks, block_size)
#         w_q    = sign_b * aq.unsqueeze(-1) * basis * mask_blocks       # [N,B,bs]
 
#         # Write mask: [N, B, bs]
#         write  = bmask.unsqueeze(-1).expand_as(w_q)
 
#         # Flatten to [N, M_pad] for scatter
#         w_q_f   = w_q  .reshape(N, M_pad)
#         aq_f    = aq.unsqueeze(-1).expand_as(w_q).reshape(N, M_pad)
#         exp_f   = exp_t.reshape(N, M_pad)
#         mant_f  = mant_t.reshape(N, M_pad)
#         write_f = write .reshape(N, M_pad)
#         bias_f  = torch.full((N, M_pad), bias_val,
#                               device=device, dtype=torch.long)
 
#         W_q_flat   = torch.where(write_f, w_q_f,   W_q_flat)
#         alpha_flat = torch.where(write_f, aq_f,    alpha_flat)
#         e_flat     = torch.where(write_f, exp_f,   e_flat)
#         m_flat     = torch.where(write_f, mant_f,  m_flat)
#         bias_flat  = torch.where(write_f, bias_f,  bias_flat)
 
#     # ------------------------------------------------------------------
#     # Strip padding and reshape to original weight shape
#     # ------------------------------------------------------------------
#     W_q_out    = W_q_flat  [:, :M].view_as(W)
#     alpha_out  = alpha_flat[:, :M].view_as(W)
#     e_out      = e_flat    [:, :M].view_as(W)
#     m_out      = m_flat    [:, :M].view_as(W)
#     sign_out   = sign_mat          .view_as(W)
#     # *** bias_out has the SAME shape as W so that calibrate_Hessian_scaled
#     #     can do:  base = alpha * (2.0 ** (e.float() - bias))  directly ***
#     bias_out   = bias_flat [:, :M].view_as(W)
 
#     return alpha_out, e_out, m_out, sign_out, bias_out

# def reconstruct_layer_fp_blockdiag_scaled_v4_fast(
#     layer,
#     H_blocks_layer,
#     block_size: int,
#     e_bits: int,
#     m_bits: int,
#     e_bits_scale: int,
#     m_bits_scale: int,
#     device,
# ):
#     import torch
#     import torch.nn.functional as F
#     import math

#     # ============================================================
#     # STREAMING LOOKUP (CRITICAL FIX)
#     # ============================================================
#     def chunked_lookup_fp(x_norm, codebook, process_chunk, B_chunk=1, K_chunk=16):
#         N, B, bs = x_norm.shape

#         for b_start in range(0, B, B_chunk):
#             b_end = min(b_start + B_chunk, B)

#             x_chunk = x_norm[:, b_start:b_end, :]

#             best_dist = torch.full_like(x_chunk, float("inf"))
#             best_basis = torch.zeros_like(x_chunk)
#             best_idx = torch.zeros_like(x_chunk, dtype=torch.long)

#             for k_start in range(0, codebook.shape[0], K_chunk):
#                 cb = codebook[k_start:k_start + K_chunk]

#                 dist = (x_chunk.unsqueeze(-1) - cb.view(1, 1, 1, -1)).abs()

#                 idx = dist.argmin(dim=-1)
#                 val = dist.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

#                 better = val < best_dist

#                 best_dist = torch.where(better, val, best_dist)
#                 best_basis = torch.where(better, cb[idx], best_basis)
#                 best_idx = torch.where(better, idx + k_start, best_idx)

#             process_chunk(b_start, b_end, best_basis, best_idx)

#             del x_chunk, best_dist, best_basis, best_idx
#             torch.cuda.empty_cache()

#     # ============================================================
#     # SETUP
#     # ============================================================
#     with torch.no_grad():

#         W = layer.weight.data.to(device)
#         mask = (W.abs() > 1e-9).float()

#         if W.dim() == 4:
#             W_mat = W.view(W.shape[0], -1)
#             mask_mat = mask.view(W.shape[0], -1)
#         else:
#             W_mat = W
#             mask_mat = mask

#         N, M = W_mat.shape

#         sign_mat = torch.sign(W_mat)
#         W_abs = W_mat.abs()

#         # ------------------------------------------------------------
#         # BLOCK STRUCTURE
#         # ------------------------------------------------------------
#         pad = (block_size - M % block_size) % block_size
#         W_p = F.pad(W_abs, (0, pad))
#         mask_p = F.pad(mask_mat, (0, pad))
#         sign_p = F.pad(sign_mat, (0, pad))

#         M_pad = M + pad
#         n_blocks = M_pad // block_size

#         W_blocks = W_p.view(N, n_blocks, block_size)
#         mask_blocks = mask_p.view(N, n_blocks, block_size)

#         block_dead = (mask_blocks.sum(dim=-1) == 0)

#         # ------------------------------------------------------------
#         # Hessian diagonal
#         # ------------------------------------------------------------
#         H_list = []
#         for b in range(n_blocks):
#             H_b = H_blocks_layer[b].to(device)
#             if H_b.shape[0] != block_size:
#                 diff = block_size - H_b.shape[0]
#                 H_b = F.pad(H_b, (0, diff, 0, diff))
#             H_list.append(H_b)

#         H_diag = torch.diagonal(torch.stack(H_list), dim1=-2, dim2=-1)

#         # ------------------------------------------------------------
#         # alpha init
#         # ------------------------------------------------------------
#         w_mean_sq = (W_blocks.pow(2) * mask_blocks).sum(dim=-1)
#         w_count = mask_blocks.sum(dim=-1).clamp(min=1)
#         alpha = (w_mean_sq / w_count).sqrt().clamp(min=1e-4)

#         w_mean = (W_blocks * mask_blocks).sum(dim=-1) / w_count
#         alpha_min = (0.05 * w_mean).clamp(min=1e-6)
#         alpha_max = (20.0 * w_mean).clamp(min=1e-4)

#         default_bias = 2 ** (e_bits - 1) - 1
#         bias_radius = max(1, 2 ** (e_bits - 2))
#         bias_range = list(range(
#             default_bias - bias_radius,
#             default_bias + bias_radius + 1
#         ))

#         best_loss = torch.full((N, n_blocks), float("inf"), device=device)
#         best_alpha = alpha.clone()
#         best_bias = torch.full((N, n_blocks), default_bias, device=device)

#         h_imp = H_diag.unsqueeze(0)
#         w_eff = W_blocks * mask_blocks

#         # ============================================================
#         # BIAS SEARCH LOOP
#         # ============================================================
#         for bias_cand in bias_range:

#             codebook = _build_codebook(e_bits, m_bits, bias_cand, device)
#             alpha_tmp = alpha.clone()

#             # -----------------------------
#             # alpha refinement (STREAMED)
#             # -----------------------------
#             for _ in range(5):

#                 num = torch.zeros((N, n_blocks), device=device)
#                 den = torch.zeros((N, n_blocks), device=device)

#                 x_norm = W_blocks / alpha_tmp.unsqueeze(-1).clamp(min=1e-8)

#                 def alpha_writer(b_start, b_end, b_chunk, _):
#                     b_eff = b_chunk * mask_blocks[:, b_start:b_end, :]
#                     h_chunk = h_imp[:, b_start:b_end, :]
#                     w_chunk = w_eff[:, b_start:b_end, :]

#                     num[:, b_start:b_end] = (h_chunk * b_eff * w_chunk).sum(dim=-1)
#                     den[:, b_start:b_end] = (h_chunk * b_eff * b_eff).sum(dim=-1)

#                 chunked_lookup_fp(x_norm, codebook, alpha_writer)

#                 alpha_tmp = (num / (den + 1e-8)).clamp(min=1e-6)
#                 alpha_tmp = alpha_tmp.clamp(min=alpha_min, max=alpha_max)

#             # -----------------------------
#             # LOSS EVALUATION (STREAMED)
#             # -----------------------------
#             loss = torch.zeros((N, n_blocks), device=device)

#             x_norm = W_blocks / alpha_tmp.unsqueeze(-1).clamp(min=1e-8)

#             def loss_writer(b_start, b_end, b_chunk, _):
#                 a = alpha_tmp[:, b_start:b_end].unsqueeze(-1)
#                 b_eff = b_chunk * mask_blocks[:, b_start:b_end, :]
#                 recon = a * b_eff
#                 residual = w_eff[:, b_start:b_end, :] - recon

#                 h_chunk = h_imp[:, b_start:b_end, :]
#                 loss[:, b_start:b_end] = (h_chunk * residual.pow(2)).sum(dim=-1)

#             chunked_lookup_fp(x_norm, codebook, loss_writer)

#             improved = loss < best_loss
#             best_loss = torch.where(improved, loss, best_loss)
#             best_alpha = torch.where(improved, alpha_tmp, best_alpha)
#             best_bias = torch.where(
#                 improved,
#                 torch.full_like(best_bias, bias_cand),
#                 best_bias
#             )

#         # ============================================================
#         # QUANTIZE SCALE
#         # ============================================================
#         alpha_q = quantize_scale_tensor(best_alpha, e_bits_scale, m_bits_scale)

#         # ============================================================
#         # FINAL RECONSTRUCTION (STREAMED)
#         # ============================================================
#         W_q = torch.zeros(N, M_pad, device=device).view(N, n_blocks, block_size)

#         sign_b = sign_p.view(N, n_blocks, block_size)

#         for bias_val in best_bias.unique().tolist():

#             bias_val = int(bias_val)
#             codebook = _build_codebook(e_bits, m_bits, bias_val, device)

#             bmask = (best_bias == bias_val)
#             aq = alpha_q * bmask.float()

#             x_norm = W_blocks / aq.unsqueeze(-1).clamp(min=1e-8)

#             def final_writer(b_start, b_end, b_chunk, _):
#                 a = aq[:, b_start:b_end].unsqueeze(-1)
#                 s = sign_b[:, b_start:b_end, :]
#                 m = mask_blocks[:, b_start:b_end, :]

#                 w_q = s * a * b_chunk * m

#                 write = bmask[:, b_start:b_end].unsqueeze(-1)

#                 W_q[:, b_start:b_end, :] = torch.where(
#                     write, w_q, W_q[:, b_start:b_end, :]
#                 )

#             chunked_lookup_fp(x_norm, codebook, final_writer)

#         # ============================================================
#         # CLEANUP + OUTPUT
#         # ============================================================
#         W_q = W_q.view(N, M_pad)

#         W_out = _fht_blocks(W_q, 1 if block_size == 1 else block_size)
#         W_out = W_out * sign_mat
#         W_out = W_out[:, :M]
#         W_out = W_out * mask_mat
#         pow2_bs = 2 ** int(math.ceil(math.log2(block_size)))
#         alpha_expanded = best_alpha.unsqueeze(-1).expand(-1, -1, block_size)
#         alpha_expanded = alpha_expanded.reshape(N, M_pad)[:, :M]

#         return {
#             "alpha": alpha_expanded.view_as(W),
#             "exponent": None,   # (not fully tracked per-stream; keep API consistent)
#             "mantissa": None,
#             "sign": sign_mat.view_as(W),
#             "bias": best_bias,
#             "reconstructed_weight": W_out.view_as(W),
#         }

# def reconstruct_layer_fp_blockdiag_scaled_v4_fast(
#     layer,
#     H_blocks_layer,
#     block_size,
#     e_bits,
#     m_bits,
#     e_bits_scale,
#     m_bits_scale,
#     device,
#     B_chunk=4,
#     K_chunk=32,
# ):


#     # ------------------------------------------------------------
#     # Setup
#     # ------------------------------------------------------------
#     W = layer.weight.data.to(device)
#     mask = (W.abs() > 1e-9).float()

#     if W.dim() == 4:
#         W_mat = W.view(W.shape[0], -1)
#         mask_mat = mask.view(W.shape[0], -1)
#     else:
#         W_mat = W
#         mask_mat = mask

#     N, M = W_mat.shape
#     sign_mat = torch.sign(W_mat)
#     W_abs = W_mat.abs()

#     # ------------------------------------------------------------
#     # Block structure
#     # ------------------------------------------------------------
#     pad = (block_size - M % block_size) % block_size
#     M_pad = M + pad
#     n_blocks = M_pad // block_size

#     W_p = F.pad(W_abs, (0, pad))
#     mask_p = F.pad(mask_mat, (0, pad))
#     sign_p = F.pad(sign_mat, (0, pad))

#     W_blocks = W_p.view(N, n_blocks, block_size)
#     mask_blocks = mask_p.view(N, n_blocks, block_size)
#     sign_blocks = sign_p.view(N, n_blocks, block_size)

#     block_dead = (mask_blocks.sum(dim=-1) == 0)

#     # ------------------------------------------------------------
#     # Hessian
#     # ------------------------------------------------------------
#     H_list = []
#     for b in range(n_blocks):
#         H_b = H_blocks_layer[b].to(device)
#         if H_b.shape[0] != block_size:
#             diff = block_size - H_b.shape[0]
#             H_b = F.pad(H_b, (0, diff, 0, diff))
#         H_list.append(H_b)

#     H_diag = torch.diagonal(torch.stack(H_list), dim1=-2, dim2=-1)
#     h_imp = H_diag.unsqueeze(0)  # [1, B, bs]

#     w_eff = W_blocks * mask_blocks

#     # ------------------------------------------------------------
#     # alpha init
#     # ------------------------------------------------------------
#     w_mean_sq = (W_blocks.pow(2) * mask_blocks).sum(dim=-1)
#     w_count = mask_blocks.sum(dim=-1).clamp(min=1)

#     alpha = (w_mean_sq / w_count).sqrt().clamp(min=1e-4)

#     w_mean = (W_blocks * mask_blocks).sum(dim=-1) / w_count
#     alpha_min = (0.05 * w_mean).clamp(min=1e-6)
#     alpha_max = (20.0 * w_mean).clamp(min=1e-4)

#     # ------------------------------------------------------------
#     # bias search
#     # ------------------------------------------------------------
#     default_bias = 2 ** (e_bits - 1) - 1
#     bias_radius = max(1, 2 ** (e_bits - 2))

#     bias_range = range(
#         default_bias - bias_radius,
#         default_bias + bias_radius + 1
#     )

#     best_loss = torch.full((N, n_blocks), float("inf"), device=device)
#     best_alpha = alpha.clone()
#     best_bias = torch.full((N, n_blocks), default_bias, device=device)

#     # ============================================================
#     # CORE LOOP
#     # ============================================================
#     for bias_cand in bias_range:

#         codebook = _build_codebook(e_bits, m_bits, bias_cand, device)
#         K = codebook.numel()

#         alpha_tmp = alpha.clone()

#         # ========================================================
#         # 1. ALPHA ITERATION (SAFE STREAMED CODEBOOK LOOKUP)
#         # ========================================================
#         for _ in range(5):

#             x_norm = W_blocks / alpha_tmp.unsqueeze(-1).clamp(min=1e-8)

#             num = torch.zeros((N, n_blocks), device=device)
#             den = torch.zeros((N, n_blocks), device=device)

#             # ---- streaming over BLOCKS ONLY ----
#             for b_start in range(0, n_blocks, B_chunk):
#                 b_end = min(b_start + B_chunk, n_blocks)

#                 x_chunk = x_norm[:, b_start:b_end, :]
#                 mask_chunk = mask_blocks[:, b_start:b_end, :]
#                 h_chunk = h_imp[:, b_start:b_end, :]
#                 w_chunk = w_eff[:, b_start:b_end, :]

#                 # full local search but chunked over K
#                 best_b = torch.zeros_like(x_chunk)
#                 best_dist = torch.full_like(x_chunk, float("inf"))

#                 for k_start in range(0, K, K_chunk):
#                     k_end = min(k_start + K_chunk, K)
#                     cb = codebook[k_start:k_end]

#                     dist = (x_chunk.unsqueeze(-1) - cb.view(1, 1, 1, -1)).abs()

#                     idx = dist.argmin(dim=-1)
#                     val = dist.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

#                     better = val < best_dist
#                     best_dist = torch.where(better, val, best_dist)
#                     best_b = torch.where(better, cb[idx], best_b)

#                 b_eff = best_b * mask_chunk

#                 num[:, b_start:b_end] = (h_chunk * b_eff * w_chunk).sum(dim=-1)
#                 den[:, b_start:b_end] = (h_chunk * b_eff * b_eff).sum(dim=-1)

#             alpha_tmp = (num / (den + 1e-8)).clamp(min=1e-6)
#             alpha_tmp = alpha_tmp.clamp(min=alpha_min, max=alpha_max)

#         # ========================================================
#         # 2. LOSS EVALUATION (FULLY CONSISTENT)
#         # ========================================================
#         x_norm = W_blocks / alpha_tmp.unsqueeze(-1).clamp(min=1e-8)

#         loss = torch.zeros((N, n_blocks), device=device)

#         for b_start in range(0, n_blocks, B_chunk):
#             b_end = min(b_start + B_chunk, n_blocks)

#             x_chunk = x_norm[:, b_start:b_end, :]
#             mask_chunk = mask_blocks[:, b_start:b_end, :]
#             h_chunk = h_imp[:, b_start:b_end, :]
#             w_chunk = w_eff[:, b_start:b_end, :]

#             best_b = torch.zeros_like(x_chunk)
#             best_dist = torch.full_like(x_chunk, float("inf"))

#             for k_start in range(0, K, K_chunk):
#                 k_end = min(k_start + K_chunk, K)
#                 cb = codebook[k_start:k_end]

#                 dist = (x_chunk.unsqueeze(-1) - cb.view(1, 1, 1, -1)).abs()

#                 idx = dist.argmin(dim=-1)
#                 val = dist.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

#                 better = val < best_dist
#                 best_dist = torch.where(better, val, best_dist)
#                 best_b = torch.where(better, cb[idx], best_b)

#             b_eff = best_b * mask_chunk
#             recon = alpha_tmp[:, b_start:b_end].unsqueeze(-1) * b_eff
#             residual = w_chunk - recon

#             loss[:, b_start:b_end] = (h_chunk * residual.pow(2)).sum(dim=-1)

#         improved = loss < best_loss
#         best_loss = torch.where(improved, loss, best_loss)
#         best_alpha = torch.where(improved, alpha_tmp, best_alpha)
#         best_bias = torch.where(
#             improved,
#             torch.full_like(best_bias, bias_cand),
#             best_bias
#         )

#     # ============================================================
#     # QUANTIZE SCALE
#     # ============================================================
#     alpha_q = quantize_scale_tensor(best_alpha, e_bits_scale, m_bits_scale)

#     # ============================================================
#     # FINAL RECONSTRUCTION (SAFE STREAMING)
#     # ============================================================
#     W_q = torch.zeros((N, n_blocks, block_size), device=device)

#     for bias_val in best_bias.unique().tolist():

#         bias_val = int(bias_val)
#         codebook = _build_codebook(e_bits, m_bits, bias_val, device)

#         bmask = (best_bias == bias_val)
#         aq = alpha_q * bmask.float()

#         for b_start in range(0, n_blocks, B_chunk):
#             b_end = min(b_start + B_chunk, n_blocks)

#             x_chunk = W_blocks[:, b_start:b_end, :] / aq[:, b_start:b_end].unsqueeze(-1).clamp(min=1e-8)

#             best_b = torch.zeros_like(x_chunk)
#             best_dist = torch.full_like(x_chunk, float("inf"))

#             for k_start in range(0, codebook.numel(), K_chunk):
#                 cb = codebook[k_start:k_start + K_chunk]

#                 dist = (x_chunk.unsqueeze(-1) - cb.view(1, 1, 1, -1)).abs()

#                 idx = dist.argmin(dim=-1)
#                 val = dist.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

#                 better = val < best_dist
#                 best_dist = torch.where(better, val, best_dist)
#                 best_b = torch.where(better, cb[idx], best_b)

#             write = bmask[:, b_start:b_end].unsqueeze(-1)

#             w_q = sign_blocks[:, b_start:b_end, :] * aq[:, b_start:b_end].unsqueeze(-1) * best_b * mask_blocks[:, b_start:b_end, :]

#             W_q[:, b_start:b_end, :] = torch.where(write, w_q, W_q[:, b_start:b_end, :])

#     # ============================================================
#     # OUTPUT
#     # ============================================================
#     W_q = W_q.view(N, M_pad)


#     W_q = W_q * sign_mat
#     W_q = W_q[:, :M] * mask_mat

#     return {
#         "alpha": best_alpha.view(N, n_blocks, 1).expand(-1, -1, block_size).reshape(N, M_pad)[:, :M].view_as(W_mat.view_as(W)),
#         "bias": best_bias,
#         "reconstructed_weight": W_q.view_as(W),
#     }

# def reconstruct_layer_fp_blockdiag_scaled_v4_fast(
#     layer,
#     H_blocks_layer,
#     block_size,
#     e_bits,
#     m_bits,
#     e_bits_scale,
#     m_bits_scale,
#     device,
#     B_chunk=4,
#     K_chunk=32,
# ):
#     import torch
#     import torch.nn.functional as F

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
#     W_abs = W_mat.abs()

#     # ============================================================
#     # OUTPUT BUFFERS (same as original)
#     # ============================================================
#     W_q = torch.zeros_like(W_mat)
#     alpha_out = torch.zeros_like(W_mat)
#     bias_out = torch.zeros_like(W_mat)

#     # ============================================================
#     # PROCESS PER BLOCK (BUT BATCHED OVER ROWS)
#     # ============================================================
#     block_idx = 0

#     for i in range(0, M, block_size):
#         end = min(i + block_size, M)
#         bs = end - i

#         W_block = W_abs[:, i:end]          # [N, bs]
#         M_block = mask_mat[:, i:end]

#         H_block = H_blocks_layer[block_idx].to(device)
#         if H_block.shape[0] != bs:
#             H_block = H_block[:bs, :bs]

#         # ========================================================
#         # HANDLE DEAD BLOCK
#         # ========================================================
#         alive = (M_block.sum(dim=1) > 1e-8)

#         if not alive.any():
#             alpha_out[:, i:end] = 1.0
#             bias_out[:, i:end] = 0
#             block_idx += 1
#             continue

#         w_eff = W_block * M_block

#         # ========================================================
#         # INIT
#         # ========================================================
#         alpha = torch.sqrt((w_eff ** 2).mean(dim=1)).clamp(min=1e-4)

#         default_bias = 2**(e_bits - 1) - 1
#         bias_radius = max(1, 2**(e_bits - 2))

#         best_loss = torch.full((N,), float("inf"), device=device)
#         best_alpha = alpha.clone()
#         best_bias = torch.full((N,), default_bias, device=device)

#         Hw = (H_block @ w_eff.T).T  # [N, bs]

#         # ========================================================
#         # BIAS SEARCH
#         # ========================================================
#         for bias_candidate in range(
#             default_bias - bias_radius,
#             default_bias + bias_radius + 1
#         ):

#             codebook = _build_codebook(e_bits, m_bits, bias_candidate, device)
#             alpha_tmp = alpha.clone()

#             # ---------------------------
#             # alpha refinement
#             # ---------------------------
#             for _ in range(5):

#                 x_norm = W_block / alpha_tmp.unsqueeze(-1).clamp(min=1e-8)

#                 best_b = torch.zeros_like(x_norm)
#                 best_dist = torch.full_like(x_norm, float("inf"))

#                 for k_start in range(0, codebook.numel(), K_chunk):
#                     cb = codebook[k_start:k_start + K_chunk]

#                     dist = (x_norm.unsqueeze(-1) - cb.view(1,1,-1)).abs()

#                     idx = dist.argmin(dim=-1)
#                     val = dist.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

#                     better = val < best_dist
#                     best_dist = torch.where(better, val, best_dist)
#                     best_b = torch.where(better, cb[idx], best_b)

#                 b_eff = best_b * M_block

#                 Hb = (H_block @ b_eff.T).T

#                 num = (b_eff * Hw).sum(dim=1)
#                 den = (b_eff * Hb).sum(dim=1) + 1e-8

#                 alpha_tmp = (num / den).clamp(min=1e-6)

#             # ---------------------------
#             # LOSS
#             # ---------------------------
#             residual = w_eff - alpha_tmp.unsqueeze(-1) * b_eff
#             loss = (residual * (H_block @ residual.T).T).sum(dim=1)

#             improved = loss < best_loss

#             best_loss = torch.where(improved, loss, best_loss)
#             best_alpha = torch.where(improved, alpha_tmp, best_alpha)
#             best_bias = torch.where(
#                 improved,
#                 torch.full_like(best_bias, bias_candidate),
#                 best_bias
#             )

#         # ========================================================
#         # FINAL QUANTIZATION (MATCH ORIGINAL)
#         # ========================================================
#         alpha_q = quantize_scale_tensor(
#             best_alpha.unsqueeze(-1),
#             e_bits_scale,
#             m_bits_scale
#         ).squeeze(-1)

#         for n in range(N):
#             if not alive[n]:
#                 continue

#             _, _, b_final = assign_fp4_dynamic(
#                 W_block[n],
#                 alpha_q[n],
#                 e_bits,
#                 m_bits,
#                 bias=int(best_bias[n].item())
#             )

#             w_hat = alpha_q[n] * b_final
#             w_hat = w_hat * M_block[n]
#             w_hat = w_hat * sign_mat[n, i:end]

#             W_q[n, i:end] = w_hat
#             alpha_out[n, i:end] = alpha_q[n]
#             bias_out[n, i:end] = best_bias[n]

#         block_idx += 1

#     # ============================================================
#     # FINAL FP DECOMPOSITION (CRITICAL FOR PERPLEXITY)
#     # ============================================================
#     sign = torch.sign(W_q)
#     W_abs = W_q.abs()

#     e = torch.zeros_like(W_mat)
#     m = torch.zeros_like(W_mat)

#     for row in range(N):
#         for i in range(0, M, block_size):
#             end = min(i + block_size, M)

#             bias_block = int(bias_out[row, i].item())
#             alpha_block = alpha_out[row, i]

#             e_block, m_block, _ = assign_fp4_dynamic(
#                 W_abs[row, i:end],
#                 alpha_block,
#                 e_bits,
#                 m_bits,
#                 bias=bias_block
#             )

#             e[row, i:end] = e_block
#             m[row, i:end] = m_block

#     if W.dim() == 4:
#         return (
#             alpha_out.view_as(W),
#             e.view_as(W),
#             m.view_as(W),
#             sign.view_as(W),
#             bias_out.view_as(W),
#         )

#     return alpha_out, e, m, sign, bias_out

# 
# def reconstruct_layer_fp_blockdiag_scaled_v4_fast(
#     layer,
#     H_blocks_layer,
#     block_size,
#     e_bits,
#     m_bits,
#     e_bits_scale,
#     m_bits_scale,
#     device,
#     B_chunk=16, # Increased chunk size for better GPU utilization
# ):
#     W = layer.weight.data.to(device)
#     mask = (W != 0).float()
#     N, M = (W.view(W.shape[0], -1)).shape
#     W_mat = W.view(N, -1)
#     mask_mat = mask.view(N, -1)
    
#     sign = torch.sign(W_mat)
#     W_abs = W_mat.abs()

#     # Padding
#     pad = (block_size - M % block_size) % block_size
#     M_pad = M + pad
#     n_blocks = M_pad // block_size
#     W_p = torch.nn.functional.pad(W_abs, (0, pad))
#     mask_p = torch.nn.functional.pad(mask_mat, (0, pad))
#     W_blocks = W_p.view(N, n_blocks, block_size)
#     mask_blocks = mask_p.view(N, n_blocks, block_size)

#     # 1. Load Full Hessian Blocks (Crucial for Perplexity)
#     # Shape: [n_blocks, block_size, block_size]
#     H_all = []
#     for b in range(n_blocks):
#         H_b = H_blocks_layer[b].to(device).float()
#         if H_b.shape[0] != block_size:
#             diff = block_size - H_b.shape[0]
#             H_b = torch.nn.functional.pad(H_b, (0, diff, 0, diff))
#         H_all.append(H_b)
#     H_all = torch.stack(H_all) 

#     # 2. Alpha Init (Matching your original mean-based scaling)
#     w_eff = W_blocks * mask_blocks
#     # Mean across the block size dimension
#     block_means = w_eff.abs().mean(dim=-1, keepdim=True) 
#     alpha = torch.sqrt((w_eff ** 2).mean(dim=-1)).clamp(min=1e-4)

#     default_bias = 2**(e_bits - 1) - 1
#     bias_radius = max(1, 2**(e_bits - 2))

#     best_loss = torch.full((N, n_blocks), float("inf"), device=device)
#     best_alpha = alpha.clone()
#     best_bias = torch.full((N, n_blocks), default_bias, device=device, dtype=torch.long)

#     # 3. Optimization Loop
#     for bias in range(default_bias - bias_radius, default_bias + bias_radius + 1):
#         alpha_tmp = alpha.clone()
        
#         for _ in range(5):
#             for b_start in range(0, n_blocks, B_chunk):
#                 b_end = min(b_start + B_chunk, n_blocks)
                
#                 # Slices
#                 w_c = w_eff[:, b_start:b_end, :]      # [N, chunk, bs]
#                 m_c = mask_blocks[:, b_start:b_end, :]
#                 H_c = H_all[b_start:b_end]             # [chunk, bs, bs]
#                 a_c = alpha_tmp[:, b_start:b_end].unsqueeze(-1)

#                 _, _, b_val = assign_fp4_dynamic_vectorized(w_c, a_c, e_bits, m_bits, bias=bias)
#                 b_eff = b_val * m_c                    # [N, chunk, bs]

#                 # Matrix Math: Restore Full Hessian Dot Products
#                 # Hw = H @ w
#                 Hw = torch.einsum('cjk, nck -> ncj', H_c, w_c)
#                 # Hb = H @ b
#                 Hb = torch.einsum('cjk, nck -> ncj', H_c, b_eff)
                
#                 num = (b_eff * Hw).sum(dim=-1)
#                 den = (b_eff * Hb).sum(dim=-1) + 1e-8
                
#                 # Stabilization logic matching your original code
#                 a_new = num / den
#                 a_min = 0.05 * block_means[:, b_start:b_end].squeeze(-1)
#                 a_max = 20.0 * block_means[:, b_start:b_end].squeeze(-1)
                
#                 alpha_tmp[:, b_start:b_end] = a_new.clamp(min=a_min, max=a_max)

#         # 4. Loss Evaluation (Full Hessian Residual)
#         for b_start in range(0, n_blocks, B_chunk):
#             b_end = min(b_start + B_chunk, n_blocks)
#             w_c = w_eff[:, b_start:b_end, :]
#             H_c = H_all[b_start:b_end]
#             a_c = alpha_tmp[:, b_start:b_end].unsqueeze(-1)
            
#             _, _, b_val = assign_fp4_dynamic_vectorized(w_c, a_c, e_bits, m_bits, bias=bias)
#             recon = a_c * b_val * mask_blocks[:, b_start:b_end, :]
#             res = w_c - recon
            
#             # Quadratic form: res^T @ H @ res
#             H_res = torch.einsum('cjk, nck -> ncj', H_c, res)
#             loss_chunk = (res * H_res).sum(dim=-1)
            
#             improved = loss_chunk < best_loss[:, b_start:b_end]
#             best_loss[:, b_start:b_end] = torch.where(improved, loss_chunk, best_loss[:, b_start:b_end])
#             best_alpha[:, b_start:b_end] = torch.where(improved, alpha_tmp[:, b_start:b_end], best_alpha[:, b_start:b_end])
#             best_bias[:, b_start:b_end] = torch.where(improved, bias, best_bias[:, b_start:b_end])

#     # Re-run the final reconstruction loop and decomposition logic as previously fixed...
#     # (Ensure you use best_alpha and best_bias for the final e, m extraction)

#     # ------------------------------------------------------------
#     # Quantize alpha (CRITICAL: AFTER optimization)
#     # ------------------------------------------------------------
#     alpha_q = quantize_scale_tensor(best_alpha, e_bits_scale, m_bits_scale)

#     # ------------------------------------------------------------
#     # Final reconstruction (CONSISTENT)
#     # ------------------------------------------------------------
#     W_q = torch.zeros_like(W_blocks)

#     for bias in best_bias.unique().tolist():
#         bias = int(bias)

#         mask_bias = (best_bias == bias)

#         for b_start in range(0, n_blocks, B_chunk):
#             b_end = min(b_start + B_chunk, n_blocks)

#             w_chunk = W_blocks[:, b_start:b_end, :]
#             m_chunk = mask_blocks[:, b_start:b_end, :]
#             s_chunk = sign[:, b_start*block_size:(b_end*block_size)].view(N, -1, block_size)

#             alpha_chunk = alpha_q[:, b_start:b_end]

#             _, _, b_val = assign_fp4_dynamic_vectorized(
#                 w_chunk,
#                 alpha_chunk.unsqueeze(-1),
#                 e_bits,
#                 m_bits,
#                 bias=bias
#             )

#             w_hat = alpha_chunk.unsqueeze(-1) * b_val * m_chunk

#             write = mask_bias[:, b_start:b_end].unsqueeze(-1)

#             W_q[:, b_start:b_end, :] = torch.where(
#                 write,
#                 w_hat,
#                 W_q[:, b_start:b_end, :]
#             )

#     # ------------------------------------------------------------
#     # Reshape back
#     # ------------------------------------------------------------
#     W_q = W_q.view(N, M_pad)[:, :M]
#     W_q = W_q * sign

#     alpha_out = best_alpha.unsqueeze(-1).expand(-1, -1, block_size)
#     alpha_out = alpha_out.reshape(N, M_pad)[:, :M]

#     bias_out = best_bias.unsqueeze(-1).expand(-1, -1, block_size)
#     bias_out = bias_out.reshape(N, M_pad)[:, :M]

#     if W.dim() == 4:
#         W_q = W_q.view_as(W)
#         alpha_out = alpha_out.view_as(W)
#         bias_out = bias_out.view_as(W)

#     # ------------------------------------------------------------
#     # Final FP decomposition (EXACT MATCH)
#     # ------------------------------------------------------------
#     e = torch.zeros_like(W_q)
#     m = torch.zeros_like(W_q)

#     W_abs = W_q.abs()

#     if W.dim() == 4:
#         W_flat = W_abs.view(N, -1)
#         alpha_flat = alpha_out.view(N, -1)
#         bias_flat = bias_out.view(N, -1)
#     else:
#         W_flat = W_abs
#         alpha_flat = alpha_out
#         bias_flat = bias_out

#     # Get flat references for e and m to avoid view_as issues inside the loop
#     e_flat = e.view(N, -1) if W.dim() == 4 else e
#     m_flat = m.view(N, -1) if W.dim() == 4 else m

#     for i in range(0, M, block_size):
#         end = min(i + block_size, M)
#         curr_bs = end - i

#         current_bias = bias_flat[:, i].long() # Shape [N]

#         # b_val shape: [N, 1, curr_bs] (since we pass a slice, B=1)
#         _, _, b_val = assign_fp4_dynamic_vectorized(
#             W_flat[:, i:end],
#             alpha_flat[:, i:end],
#             e_bits,
#             m_bits,
#             bias=current_bias
#         )

#         M_count = 2**m_bits
        
#         # FIX: Expand b_offset to [N, 1, 1] to match [N, 1, curr_bs]
#         b_offset = current_bias.view(-1, 1, 1).float()
        
#         # Log recovery
#         # (torch.log2(b_val) + b_offset) maintains [N, 1, curr_bs]
#         idx = (torch.log2(b_val + 1e-10) + b_offset).round()

#         # Reshape to [N, curr_bs] before assignment to e_flat/m_flat
#         idx_reshaped = idx.view(N, curr_bs)

#         e_flat[:, i:end] = idx_reshaped // M_count
#         m_flat[:, i:end] = idx_reshaped % M_count

#     if W.dim() == 4:
#         e = e.view_as(W)
#         m = m.view_as(W)

#     return alpha_out, e, m, sign, bias_out


# def reconstruct_layer_fp_blockdiag_scaled_v4_fast(
#     layer,
#     H_blocks_layer,
#     block_size,
#     e_bits,
#     m_bits,
#     e_bits_scale,
#     m_bits_scale,
#     device,
#     B_chunk=16,
# ):
#     # 1. Setup and Flattening
#     W = layer.weight.data.to(device)
#     mask = (W != 0).float()
#     N, M = W.view(W.shape[0], -1).shape
#     W_mat = W.view(N, -1)
#     mask_mat = mask.view(N, -1)
    
#     sign = torch.sign(W_mat)
#     W_abs = W_mat.abs()

#     # 2. Block Padding
#     pad = (block_size - M % block_size) % block_size
#     M_pad = M + pad
#     n_blocks = M_pad // block_size
#     W_p = F.pad(W_abs, (0, pad))
#     mask_p = F.pad(mask_mat, (0, pad))
#     W_blocks = W_p.view(N, n_blocks, block_size)
#     mask_blocks = mask_p.view(N, n_blocks, block_size)

#     # 3. Full Hessian Block Stacking
#     H_all = []
#     for b in range(n_blocks):
#         H_b = H_blocks_layer[b].to(device).float()
#         if H_b.shape[0] != block_size:
#             diff = block_size - H_b.shape[0]
#             H_b = F.pad(H_b, (0, diff, 0, diff))
#         H_all.append(H_b)
#     H_all = torch.stack(H_all) # [n_blocks, bs, bs]

#     # 4. Initialization (Matching Original Row-wise Logic)
#     w_eff = W_blocks * mask_blocks
#     block_means = w_eff.abs().mean(dim=-1, keepdim=True)
#     alpha = torch.sqrt((w_eff ** 2).mean(dim=-1)).clamp(min=1e-4)

#     default_bias = 2**(e_bits - 1) - 1
#     bias_radius = max(1, 2**(e_bits - 2))

#     best_loss = torch.full((N, n_blocks), float("inf"), device=device)
#     best_alpha = alpha.clone()
#     best_bias = torch.full((N, n_blocks), default_bias, device=device, dtype=torch.long)

#     # 5. Bias & Alpha Search (The Optimization)
#     for bias in range(default_bias - bias_radius, default_bias + bias_radius + 1):
#         alpha_tmp = alpha.clone()
        
#         for _ in range(5):
#             for b_start in range(0, n_blocks, B_chunk):
#                 b_end = min(b_start + B_chunk, n_blocks)
                
#                 w_c = w_eff[:, b_start:b_end, :]
#                 m_c = mask_blocks[:, b_start:b_end, :]
#                 H_c = H_all[b_start:b_end]
#                 a_c = alpha_tmp[:, b_start:b_end].unsqueeze(-1)

#                 _, _, b_val = assign_fp4_dynamic_vectorized(w_c, a_c, e_bits, m_bits, bias=bias)
#                 b_eff = b_val * m_c

#                 # Full Hessian Math: b^T H w / b^T H b
#                 Hw = torch.einsum('cjk, nck -> ncj', H_c, w_c)
#                 Hb = torch.einsum('cjk, nck -> ncj', H_c, b_eff)
                
#                 num = (b_eff * Hw).sum(dim=-1)
#                 den = (b_eff * Hb).sum(dim=-1) + 1e-8
                
#                 # Stabilization
#                 a_min = 0.05 * block_means[:, b_start:b_end].squeeze(-1)
#                 a_max = 20.0 * block_means[:, b_start:b_end].squeeze(-1)
#                 alpha_tmp[:, b_start:b_end] = (num / den).clamp(min=a_min, max=a_max)

#         # Loss Update
#         for b_start in range(0, n_blocks, B_chunk):
#             b_end = min(b_start + B_chunk, n_blocks)
#             w_c = w_eff[:, b_start:b_end, :]
#             H_c = H_all[b_start:b_end]
#             a_c = alpha_tmp[:, b_start:b_end].unsqueeze(-1)
            
#             _, _, b_val = assign_fp4_dynamic_vectorized(w_c, a_c, e_bits, m_bits, bias=bias)
#             recon = a_c * b_val * mask_blocks[:, b_start:b_end, :]
#             res = w_c - recon
            
#             H_res = torch.einsum('cjk, nck -> ncj', H_c, res)
#             loss_chunk = (res * H_res).sum(dim=-1)
            
#             improved = loss_chunk < best_loss[:, b_start:b_end]
#             best_loss[:, b_start:b_end] = torch.where(improved, loss_chunk, best_loss[:, b_start:b_end])
#             best_alpha[:, b_start:b_end] = torch.where(improved, alpha_tmp[:, b_start:b_end], best_alpha[:, b_start:b_end])
#             best_bias[:, b_start:b_end] = torch.where(improved, bias, best_bias[:, b_start:b_end])

#     # 6. Quantize Optimized Scales
#     # Ensure quantize_scale_tensor matches your scale quantization method
#     alpha_q = quantize_scale_tensor(best_alpha, e_bits_scale, m_bits_scale)

#     # 7. Final Extraction (The "Golden" Pass)
#     e_out = torch.zeros((N, M_pad), device=device)
#     m_out = torch.zeros((N, M_pad), device=device)
    
#     for b_start in range(0, n_blocks, B_chunk):
#         b_end = min(b_start + B_chunk, n_blocks)
        
#         w_c = W_blocks[:, b_start:b_end, :]
#         a_c = alpha_q[:, b_start:b_end].unsqueeze(-1)
#         b_c = best_bias[:, b_start:b_end] # Tensor shape [N, chunk]

#         # Use the fixed assign function that handles tensor biases
#         exp_idx, mant_idx, _ = assign_fp4_dynamic_vectorized(
#             w_c, a_c, e_bits, m_bits, bias=b_c
#         )
        
#         e_out.view(N, n_blocks, block_size)[:, b_start:b_end, :] = exp_idx.float()
#         m_out.view(N, n_blocks, block_size)[:, b_start:b_end, :] = mant_idx.float()

#     # 8. Final Reshaping
#     e_out = e_out[:, :M]
#     m_out = m_out[:, :M]
#     alpha_final = best_alpha.unsqueeze(-1).expand(-1, -1, block_size).reshape(N, M_pad)[:, :M]
#     bias_final = best_bias.unsqueeze(-1).expand(-1, -1, block_size).reshape(N, M_pad)[:, :M]

#     if W.dim() == 4:
#         out_shape = W.shape
#         return alpha_final.view(out_shape), e_out.view(out_shape), m_out.view(out_shape), sign.view(out_shape), bias_final.view(out_shape)
    
#     return alpha_final, e_out, m_out, sign, bias_final

# def reconstruct_layer_fp_blockdiag_scaled_v4_fast(
#     layer, H_blocks_layer, block_size, e_bits, m_bits, e_bits_scale, m_bits_scale, device, B_chunk=16
# ):
#     W = layer.weight.data.to(device)
#     mask = (W != 0).float()
#     N, M = W.view(W.shape[0], -1).shape
#     W_mat = W.view(N, -1)
#     mask_mat = mask.view(N, -1)
#     sign = torch.sign(W_mat)
    
#     # 1. Padding & Blocking
#     pad = (block_size - M % block_size) % block_size
#     M_pad = M + pad
#     n_blocks = M_pad // block_size
#     W_p = F.pad(W_mat.abs(), (0, pad))
#     mask_p = F.pad(mask_mat, (0, pad))
#     W_blocks = W_p.view(N, n_blocks, block_size)
#     mask_blocks = mask_p.view(N, n_blocks, block_size)

#     # 2. Stack Full Hessian Blocks
#     H_all = torch.stack([F.pad(H_blocks_layer[b].to(device).float(), 
#                                (0, max(0, block_size - H_blocks_layer[b].shape[0]), 
#                                 0, max(0, block_size - H_blocks_layer[b].shape[0]))) 
#                          for b in range(n_blocks)])

#     # 3. Initialization
#     w_eff = W_blocks * mask_blocks
#     block_means = w_eff.abs().mean(dim=-1, keepdim=True)
#     alpha = torch.sqrt((w_eff ** 2).mean(dim=-1)).clamp(min=1e-4)
#     default_bias = 2**(e_bits - 1) - 1
#     bias_radius = max(1, 2**(e_bits - 2))

#     best_loss = torch.full((N, n_blocks), float("inf"), device=device)
#     best_alpha = alpha.clone()
#     best_bias = torch.full((N, n_blocks), default_bias, device=device, dtype=torch.long)

#     # 4. Optimization Loop (Hessian-Aware)
#     for bias in range(default_bias - bias_radius, default_bias + bias_radius + 1):
#         alpha_tmp = alpha.clone()
#         for _ in range(5):
#             for b_start in range(0, n_blocks, B_chunk):
#                 b_end = min(b_start + B_chunk, n_blocks)
#                 w_c, m_c, H_c = w_eff[:, b_start:b_end, :], mask_blocks[:, b_start:b_end, :], H_all[b_start:b_end]
#                 a_c = alpha_tmp[:, b_start:b_end].unsqueeze(-1)

#                 _, _, b_val = assign_fp4_dynamic_vectorized(w_c, a_c, e_bits, m_bits, bias=bias)
#                 b_eff = b_val * m_c
#                 Hw, Hb = torch.einsum('cjk, nck -> ncj', H_c, w_c), torch.einsum('cjk, nck -> ncj', H_c, b_eff)
                
#                 num = (b_eff * Hw).sum(dim=-1)
#                 den = (b_eff * Hb).sum(dim=-1) + 1e-8
#                 a_min, a_max = 0.05 * block_means[:, b_start:b_end].squeeze(-1), 20.0 * block_means[:, b_start:b_end].squeeze(-1)
#                 alpha_tmp[:, b_start:b_end] = (num / den).clamp(min=a_min, max=a_max)

#         # Loss Update (Quadratic)
#         for b_start in range(0, n_blocks, B_chunk):
#             b_end = min(b_start + B_chunk, n_blocks)
#             a_c = alpha_tmp[:, b_start:b_end].unsqueeze(-1)
#             _, _, b_val = assign_fp4_dynamic_vectorized(w_eff[:, b_start:b_end, :], a_c, e_bits, m_bits, bias=bias)
#             res = w_eff[:, b_start:b_end, :] - (a_c * b_val * mask_blocks[:, b_start:b_end, :])
#             loss_chunk = (res * torch.einsum('cjk, nck -> ncj', H_all[b_start:b_end], res)).sum(dim=-1)
#             improved = loss_chunk < best_loss[:, b_start:b_end]
#             best_loss[:, b_start:b_end] = torch.where(improved, loss_chunk, best_loss[:, b_start:b_end])
#             best_alpha[:, b_start:b_end] = torch.where(improved, alpha_tmp[:, b_start:b_end], best_alpha[:, b_start:b_end])
#             best_bias[:, b_start:b_end] = torch.where(improved, bias, best_bias[:, b_start:b_end])

#     # 5. SCALE QUANTIZATION (CRITICAL: Match the slow version exactly)
#     alpha_q = quantize_scale_tensor(best_alpha, e_bits_scale, m_bits_scale)

#     # 6. FINAL BASIS EXTRACTION (The "Inference-Consistent" Pass)
#     e_out = torch.zeros((N, n_blocks, block_size), device=device)
#     m_out = torch.zeros((N, n_blocks, block_size), device=device)
    
#     for b_start in range(0, n_blocks, B_chunk):
#         b_end = min(b_start + B_chunk, n_blocks)
#         w_c, a_c, b_c = W_blocks[:, b_start:b_end, :], alpha_q[:, b_start:b_end].unsqueeze(-1), best_bias[:, b_start:b_end]
        
#         # Get indices using the QUANTIZED scale
#         e_idx, m_idx, _ = assign_fp4_dynamic_vectorized(w_c, a_c, e_bits, m_bits, bias=b_c)
#         e_out[:, b_start:b_end, :] = e_idx.float()
#         m_out[:, b_start:b_end, :] = m_idx.float()

#     # 7. Formatting Outputs
#     e_out = e_out.view(N, M_pad)[:, :M].view_as(W)
#     m_out = m_out.view(N, M_pad)[:, :M].view_as(W)
#     alpha_out = alpha_q.unsqueeze(-1).expand(-1, -1, block_size).reshape(N, M_pad)[:, :M].view_as(W)
#     bias_out = best_bias.unsqueeze(-1).expand(-1, -1, block_size).reshape(N, M_pad)[:, :M].view_as(W)

#     return alpha_out, e_out, m_out, sign.view_as(W), bias_out

# def reconstruct_layer_fp_blockdiag_scaled_v4_fast(
#     layer, H_blocks_layer, block_size, e_bits, m_bits, e_bits_scale, m_bits_scale, device, B_chunk=16
# ):
#     W = layer.weight.data.to(device)
#     mask = (W != 0).float()
#     orig_shape = W.shape
#     N, M = W.view(orig_shape[0], -1).shape
#     W_mat = W.view(N, -1)
#     mask_mat = mask.view(N, -1)
#     sign = torch.sign(W_mat)
    
#     # 1. Padding & Blocking
#     pad = (block_size - M % block_size) % block_size
#     M_pad = M + pad
#     n_blocks = M_pad // block_size
#     W_p = F.pad(W_mat.abs(), (0, pad))
#     mask_p = F.pad(mask_mat, (0, pad))
#     W_blocks = W_p.view(N, n_blocks, block_size)
#     mask_blocks = mask_p.view(N, n_blocks, block_size)

#     # 2. Hessian Stacking (Ensure alignment with blocks)
#     H_all = torch.stack([F.pad(H_blocks_layer[b].to(device).float(), 
#                                (0, block_size - H_blocks_layer[b].shape[0], 
#                                 0, block_size - H_blocks_layer[b].shape[0])) 
#                          for b in range(n_blocks)])

#     # 3. Optimization Setup
#     w_eff = W_blocks * mask_blocks
#     block_means = w_eff.abs().mean(dim=-1, keepdim=True)
#     alpha = torch.sqrt((w_eff ** 2).mean(dim=-1)).clamp(min=1e-4)
#     default_bias = 2**(e_bits - 1) - 1
#     bias_radius = max(1, 2**(e_bits - 2))

#     best_loss = torch.full((N, n_blocks), float("inf"), device=device)
#     best_alpha = alpha.clone()
#     best_bias = torch.full((N, n_blocks), default_bias, device=device, dtype=torch.long)

#     # 4. Search Loop
#     for bias in range(default_bias - bias_radius, default_bias + bias_radius + 1):
#         alpha_tmp = alpha.clone()
#         for _ in range(5):
#             for b_start in range(0, n_blocks, B_chunk):
#                 b_end = min(b_start + B_chunk, n_blocks)
#                 w_c, m_c, H_c = w_eff[:, b_start:b_end, :], mask_blocks[:, b_start:b_end, :], H_all[b_start:b_end]
#                 a_c = alpha_tmp[:, b_start:b_end].unsqueeze(-1)

#                 _, _, b_val = assign_fp4_dynamic_vectorized(w_c, a_c, e_bits, m_bits, bias=bias)
#                 b_eff = b_val * m_c
#                 Hw = torch.einsum('cjk, nck -> ncj', H_c, w_c)
#                 Hb = torch.einsum('cjk, nck -> ncj', H_c, b_eff)
                
#                 num = (b_eff * Hw).sum(dim=-1)
#                 den = (b_eff * Hb).sum(dim=-1) + 1e-8
#                 a_min, a_max = 0.05 * block_means[:, b_start:b_end].squeeze(-1), 20.0 * block_means[:, b_start:b_end].squeeze(-1)
#                 alpha_tmp[:, b_start:b_end] = (num / den).clamp(min=a_min, max=a_max)

#         # Loss Update
#         for b_start in range(0, n_blocks, B_chunk):
#             b_end = min(b_start + B_chunk, n_blocks)
#             w_c, a_c = w_eff[:, b_start:b_end, :], alpha_tmp[:, b_start:b_end].unsqueeze(-1)
#             _, _, b_val = assign_fp4_dynamic_vectorized(w_c, a_c, e_bits, m_bits, bias=bias)
#             res = w_c - (a_c * b_val * mask_blocks[:, b_start:b_end, :])
#             loss_chunk = (res * torch.einsum('cjk, nck -> ncj', H_all[b_start:b_end], res)).sum(dim=-1)
            
#             improved = loss_chunk < best_loss[:, b_start:b_end]
#             best_loss[:, b_start:b_end] = torch.where(improved, loss_chunk, best_loss[:, b_start:b_end])
#             best_alpha[:, b_start:b_end] = torch.where(improved, alpha_tmp[:, b_start:b_end], best_alpha[:, b_start:b_end])
#             best_bias[:, b_start:b_end] = torch.where(improved, bias, best_bias[:, b_start:b_end])

#     # 5. SCALE QUANTIZATION (CRITICAL: MUST happen before Step 6)
#     alpha_q = quantize_scale_tensor(best_alpha, e_bits_scale, m_bits_scale)

#     # 6. FINAL BIT EXTRACTION (Match indices to the QUANTIZED scale)
#     e_out_blocks = torch.zeros_like(W_blocks)
#     m_out_blocks = torch.zeros_like(W_blocks)
    
#     for b_start in range(0, n_blocks, B_chunk):
#         b_end = min(b_start + B_chunk, n_blocks)
#         w_c = W_blocks[:, b_start:b_end, :]
#         a_c = alpha_q[:, b_start:b_end].unsqueeze(-1)
#         b_c = best_bias[:, b_start:b_end] # Shape [N, chunk]

#         e_idx, m_idx, _ = assign_fp4_dynamic_vectorized(w_c, a_c, e_bits, m_bits, bias=b_c)
#         e_out_blocks[:, b_start:b_end, :] = e_idx.float()
#         m_out_blocks[:, b_start:b_end, :] = m_idx.float()

#     # 7. Reshape and Trim (Ensure original M dimensions)
#     e_out = e_out_blocks.view(N, M_pad)[:, :M]
#     m_out = m_out_blocks.view(N, M_pad)[:, :M]
    
#     # Broadcast alpha_q and best_bias back to the full matrix
#     alpha_final = alpha_q.unsqueeze(-1).expand(-1, -1, block_size).reshape(N, M_pad)[:, :M]
#     bias_final = best_bias.unsqueeze(-1).expand(-1, -1, block_size).reshape(N, M_pad)[:, :M]

#     # Reshape all to original 4D or 2D shape
#     return alpha_final.reshape(orig_shape), e_out.reshape(orig_shape), m_out.reshape(orig_shape), sign.reshape(orig_shape), bias_final.reshape(orig_shape)


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



# def reconstruct_layer_non_hadamard_v11_precision(
#     layer, H_blocks_layer, block_size, e_bits, m_bits, e_bits_scale, m_bits_scale, device, B_chunk=32
# ):
#     with torch.no_grad():
#         W = layer.weight.data.to(device)
#         orig_shape = W.shape
#         W_mat = W.view(W.shape[0], -1)
#         N, M = W_mat.shape
#         mask_mat = (W_mat.abs() > 1e-9).float()
        
#         # 1. Padding
#         n_blocks = (M + block_size - 1) // block_size
#         M_pad = n_blocks * block_size
#         W_p = F.pad(W_mat, (0, M_pad - M))
#         mask_p = F.pad(mask_mat, (0, M_pad - M))
        
#         W_blocks = W_p.view(N, n_blocks, block_size)
#         mask_blocks = mask_p.view(N, n_blocks, block_size)

#         # 2. Hessian Stacking
#         H_all = torch.stack([
#             F.pad(H_blocks_layer[b].to(device).float(), 
#                   (0, block_size - H_blocks_layer[b].shape[0], 0, block_size - H_blocks_layer[b].shape[0])) 
#             for b in range(n_blocks)
#         ]) 

#         # 3. Initialization
#         # Use a more aggressive initialization: Frobenius norm per block
#         alpha = torch.sqrt((W_blocks**2).mean(dim=-1)).clamp(min=1e-4)
        
#         default_bias = 2**(e_bits - 1) - 1
#         bias_radius = max(1, 2**(e_bits - 2))
        
#         best_loss = torch.full((N, n_blocks), float('inf'), device=device)
#         best_alpha = alpha.clone()
#         best_bias = torch.full((N, n_blocks), default_bias, dtype=torch.long, device=device)

#         # 4. Search Loop
#         for bias_cand in range(default_bias - bias_radius, default_bias + bias_radius + 1):
#             curr_alpha = alpha.clone()
            
#             # Alpha Refinement
#             for _ in range(8): # Increased iterations for better convergence
#                 for b_start in range(0, n_blocks, B_chunk):
#                     b_end = min(b_start + B_chunk, n_blocks)
                    
#                     w_c = W_blocks[:, b_start:b_end, :]
#                     m_c = mask_blocks[:, b_start:b_end, :]
#                     H_c = H_all[b_start:b_end]
                    
#                     # Compute basis based on absolute weights
#                     _, _, basis = assign_fp4_dynamic_vectorized(
#                         w_c.abs(), curr_alpha[:, b_start:b_end], e_bits, m_bits, bias=bias_cand
#                     )
                    
#                     # Apply sign back for the reconstruction math
#                     b_eff = torch.sign(w_c) * basis * m_c
                    
#                     # Full Hessian Math
#                     Hw = torch.einsum('bjk, nbk -> nbj', H_c, w_c)
#                     Hb = torch.einsum('bjk, nbk -> nbj', H_c, b_eff)
                    
#                     num = (b_eff * Hw).sum(dim=-1)
#                     den = (b_eff * Hb).sum(dim=-1) + 1e-8
                    
#                     # Clamp relative to current block scale
#                     curr_alpha[:, b_start:b_end] = (num / den).abs().clamp(min=1e-6)

#             # Evaluate Loss
#             for b_start in range(0, n_blocks, B_chunk):
#                 b_end = min(b_start + B_chunk, n_blocks)
#                 w_c = W_blocks[:, b_start:b_end, :]
#                 a_c = curr_alpha[:, b_start:b_end].unsqueeze(-1)
                
#                 _, _, basis = assign_fp4_dynamic_vectorized(
#                     w_c.abs(), a_c.squeeze(-1), e_bits, m_bits, bias=bias_cand
#                 )
                
#                 recon = torch.sign(w_c) * a_c * basis * mask_blocks[:, b_start:b_end, :]
#                 res = w_c - recon
                
#                 H_res = torch.einsum('bjk, nbk -> nbj', H_all[b_start:b_end], res)
#                 loss_chunk = (res * H_res).sum(dim=-1)
                
#                 improved = loss_chunk < best_loss[:, b_start:b_end]
#                 best_loss[:, b_start:b_end] = torch.where(improved, loss_chunk, best_loss[:, b_start:b_end])
#                 best_alpha[:, b_start:b_end] = torch.where(improved, curr_alpha[:, b_start:b_end], best_alpha[:, b_start:b_end])
#                 best_bias[:, b_start:b_end] = torch.where(improved, bias_cand, best_bias[:, b_start:b_end])

#         # 5. Final Reconstruction
#         alpha_q = quantize_scale_tensor(best_alpha, e_bits_scale, m_bits_scale)
#         W_q = torch.zeros_like(W_blocks)
        
#         for b_start in range(0, n_blocks, B_chunk):
#             b_end = min(b_start + B_chunk, n_blocks)
#             a_c = alpha_q[:, b_start:b_end].unsqueeze(-1)
#             b_c = best_bias[:, b_start:b_end]
            
#             _, _, basis = assign_fp4_dynamic_vectorized(
#                 W_blocks[:, b_start:b_end, :].abs(), a_c.squeeze(-1), e_bits, m_bits, bias=b_c
#             )
#             W_q[:, b_start:b_end, :] = torch.sign(W_blocks[:, b_start:b_end, :]) * a_c * basis * mask_blocks[:, b_start:b_end, :]

#         return {
#             'alpha': alpha_q, # Return the quantized scale directly
#             'bias': best_bias,
#             'reconstructed_weight': W_q.view(N, M_pad)[:, :M].reshape(orig_shape),
#         }


# def reconstruct_layer_non_hadamard_v12_sync(
#     layer, H_blocks_layer, block_size, e_bits, m_bits, e_bits_scale, m_bits_scale, device, B_chunk=32
# ):
#     with torch.no_grad():
#         W = layer.weight.data.to(device)
#         orig_shape = W.shape
#         W_mat = W.view(W.shape[0], -1)
#         N, M = W_mat.shape
#         mask_mat = (W_mat.abs() > 1e-9).float()
        
#         n_blocks = (M + block_size - 1) // block_size
#         M_pad = n_blocks * block_size
#         W_p = F.pad(W_mat, (0, M_pad - M))
#         mask_p = F.pad(mask_mat, (0, M_pad - M))
#         W_blocks = W_p.view(N, n_blocks, block_size)
#         mask_blocks = mask_p.view(N, n_blocks, block_size)

#         H_all = torch.stack([
#             F.pad(H_blocks_layer[b].to(device).float(), 
#                   (0, block_size - H_blocks_layer[b].shape[0], 0, block_size - H_blocks_layer[b].shape[0])) 
#             for b in range(n_blocks)
#         ]) 

#         # Initial Alpha: Use the standard Frobenius norm initialization
#         alpha = torch.sqrt((W_blocks**2).mean(dim=-1)).clamp(min=1e-4)
        
#         default_bias = 2**(e_bits - 1) - 1
#         bias_radius = max(1, 2**(e_bits - 2))
        
#         best_loss = torch.full((N, n_blocks), float('inf'), device=device)
#         best_alpha_q = alpha.clone() # We will store the quantized version
#         best_bias = torch.full((N, n_blocks), default_bias, dtype=torch.long, device=device)

#         for bias_cand in range(default_bias - bias_radius, default_bias + bias_radius + 1):
#             curr_alpha = alpha.clone()
            
#             # 1. Alpha Refinement (Float domain)
#             for _ in range(4): # Back to 4 for stability
#                 for b_start in range(0, n_blocks, B_chunk):
#                     b_end = min(b_start + B_chunk, n_blocks)
#                     w_c, m_c, H_c = W_blocks[:, b_start:b_end, :], mask_blocks[:, b_start:b_end, :], H_all[b_start:b_end]
                    
#                     _, _, basis = assign_fp4_dynamic_vectorized(
#                         w_c.abs(), curr_alpha[:, b_start:b_end], e_bits, m_bits, bias=bias_cand
#                     )
                    
#                     b_eff = torch.sign(w_c) * basis * m_c
#                     Hw = torch.einsum('bjk, nbk -> nbj', H_c, w_c)
#                     Hb = torch.einsum('bjk, nbk -> nbj', H_c, b_eff)
                    
#                     num = (b_eff * Hw).sum(dim=-1)
#                     den = (b_eff * Hb).sum(dim=-1) + 1e-8
#                     curr_alpha[:, b_start:b_end] = (num / den).abs().clamp(min=1e-6)

#             # 2. Sync Quantization: Quantize alpha BEFORE evaluating best loss
#             curr_alpha_q = quantize_scale_tensor(curr_alpha, e_bits_scale, m_bits_scale)

#             # 3. Evaluate Loss with the ACTUAL quantized scale
#             for b_start in range(0, n_blocks, B_chunk):
#                 b_end = min(b_start + B_chunk, n_blocks)
#                 w_c = W_blocks[:, b_start:b_end, :]
#                 # Use quantized alpha for the final check
#                 a_q_c = curr_alpha_q[:, b_start:b_end].unsqueeze(-1)
                
#                 _, _, basis = assign_fp4_dynamic_vectorized(
#                     w_c.abs(), a_q_c.squeeze(-1), e_bits, m_bits, bias=bias_cand
#                 )
                
#                 recon = torch.sign(w_c) * a_q_c * basis * mask_blocks[:, b_start:b_end, :]
#                 res = w_c - recon
                
#                 H_res = torch.einsum('bjk, nbk -> nbj', H_all[b_start:b_end], res)
#                 loss_chunk = (res * H_res).sum(dim=-1)
                
#                 improved = loss_chunk < best_loss[:, b_start:b_end]
#                 best_loss[:, b_start:b_end] = torch.where(improved, loss_chunk, best_loss[:, b_start:b_end])
#                 best_alpha_q[:, b_start:b_end] = torch.where(improved, curr_alpha_q[:, b_start:b_end], best_alpha_q[:, b_start:b_end])
#                 best_bias[:, b_start:b_end] = torch.where(improved, bias_cand, best_bias[:, b_start:b_end])

#         # 4. Final Reconstruction (No further quantization needed)
#         W_q = torch.zeros_like(W_blocks)
#         for b_start in range(0, n_blocks, B_chunk):
#             b_end = min(b_start + B_chunk, n_blocks)
#             a_c = best_alpha_q[:, b_start:b_end].unsqueeze(-1)
#             b_c = best_bias[:, b_start:b_end]
            
#             _, _, basis = assign_fp4_dynamic_vectorized(
#                 W_blocks[:, b_start:b_end, :].abs(), a_c.squeeze(-1), e_bits, m_bits, bias=b_c
#             )
#             W_q[:, b_start:b_end, :] = torch.sign(W_blocks[:, b_start:b_end, :]) * a_c * basis * mask_blocks[:, b_start:b_end, :]

#         return {
#             'alpha': best_alpha_q,
#             'bias': best_bias,
#             'reconstructed_weight': W_q.view(N, M_pad)[:, :M].reshape(orig_shape),
#         }
# def reconstruct_layer_non_hadamard_v14_signed(
#     layer, H_blocks_layer, block_size, e_bits, m_bits, e_bits_scale, m_bits_scale, device, B_chunk=32
# ):
#     with torch.no_grad():
#         W = layer.weight.data.to(device)
#         orig_shape = W.shape
#         W_mat = W.view(W.shape[0], -1)
#         N, M = W_mat.shape
#         mask_mat = (W_mat.abs() > 1e-9).float()
        
#         n_blocks = (M + block_size - 1) // block_size
#         M_pad = n_blocks * block_size
#         W_blocks = F.pad(W_mat, (0, M_pad - M)).view(N, n_blocks, block_size)
#         mask_blocks = F.pad(mask_mat, (0, M_pad - M)).view(N, n_blocks, block_size)

#         # 1. Hessian Setup
#         H_all = torch.stack([
#             F.pad(H_blocks_layer[b].to(device).float(), 
#                   (0, block_size - H_blocks_layer[b].shape[0], 0, block_size - H_blocks_layer[b].shape[0])) 
#             for b in range(n_blocks)
#         ]) 

#         # 2. Signed Initialization
#         # We optimize Alpha based on the SIGNED reconstruction
#         alpha = torch.sqrt((W_blocks**2).mean(dim=-1)).clamp(min=1e-4)
        
#         default_bias = 2**(e_bits - 1) - 1
#         bias_radius = max(1, 2**(e_bits - 2))
        
#         best_loss = torch.full((N, n_blocks), float('inf'), device=device)
#         best_alpha = alpha.clone()
#         best_bias = torch.full((N, n_blocks), default_bias, dtype=torch.long, device=device)

#         for bias_cand in range(default_bias - bias_radius, default_bias + bias_radius + 1):
#             curr_alpha = alpha.clone()
            
#             # Alpha Refinement (Signed Logic)
#             for _ in range(5):
#                 for b_start in range(0, n_blocks, B_chunk):
#                     b_end = min(b_start + B_chunk, n_blocks)
#                     w_c = W_blocks[:, b_start:b_end, :]
#                     H_c = H_all[b_start:b_end]
                    
#                     # 1. Get basis using ABS but preserve original SIGN
#                     _, _, basis_abs = assign_fp4_dynamic_vectorized(
#                         w_c.abs(), curr_alpha[:, b_start:b_end], e_bits, m_bits, bias=bias_cand
#                     )
#                     b_signed = torch.sign(w_c) * basis_abs * mask_blocks[:, b_start:b_end, :]
                    
#                     # 2. Standard Hessian-Weighted Least Squares (Signed)
#                     # alpha = (b^T H w) / (b^T H b)
#                     Hw = torch.einsum('bjk, nbk -> nbj', H_c, w_c)
#                     Hb = torch.einsum('bjk, nbk -> nbj', H_c, b_signed)
                    
#                     num = (b_signed * Hw).sum(dim=-1)
#                     den = (b_signed * Hb).sum(dim=-1) + 1e-8
                    
#                     # Clamp to ensure alpha stays positive and reasonable
#                     curr_alpha[:, b_start:b_end] = (num / den).clamp(min=1e-6)

#             # 3. Final Loss Evaluation for this Bias candidate
#             for b_start in range(0, n_blocks, B_chunk):
#                 b_end = min(b_start + B_chunk, n_blocks)
#                 w_c = W_blocks[:, b_start:b_end, :]
#                 a_c = curr_alpha[:, b_start:b_end].unsqueeze(-1)
                
#                 _, _, basis_abs = assign_fp4_dynamic_vectorized(
#                     w_c.abs(), a_c.squeeze(-1), e_bits, m_bits, bias=bias_cand
#                 )
#                 recon = torch.sign(w_c) * a_c * basis_abs * mask_blocks[:, b_start:b_end, :]
#                 res = w_c - recon
                
#                 H_res = torch.einsum('bjk, nbk -> nbj', H_all[b_start:b_end], res)
#                 loss_chunk = (res * H_res).sum(dim=-1)
                
#                 improved = loss_chunk < best_loss[:, b_start:b_end]
#                 best_loss[:, b_start:b_end] = torch.where(improved, loss_chunk, best_loss[:, b_start:b_end])
#                 best_alpha[:, b_start:b_end] = torch.where(improved, curr_alpha[:, b_start:b_end], best_alpha[:, b_start:b_end])
#                 best_bias[:, b_start:b_end] = torch.where(improved, bias_cand, best_bias[:, b_start:b_end])

#         # 4. Critical Step: Quantize Alpha then Re-extract Bits
#         alpha_q = quantize_scale_tensor(best_alpha, e_bits_scale, m_bits_scale)
#         W_final = torch.zeros_like(W_blocks)
        
#         for b_start in range(0, n_blocks, B_chunk):
#             b_end = min(b_start + B_chunk, n_blocks)
#             a_q_c = alpha_q[:, b_start:b_end].unsqueeze(-1)
#             b_c = best_bias[:, b_start:b_end]
#             w_c = W_blocks[:, b_start:b_end, :]
            
#             _, _, basis_abs = assign_fp4_dynamic_vectorized(
#                 w_c.abs(), a_q_c.squeeze(-1), e_bits, m_bits, bias=b_c
#             )
#             W_final[:, b_start:b_end, :] = torch.sign(w_c) * a_q_c * basis_abs * mask_blocks[:, b_start:b_end, :]

#         return {
#             'alpha': alpha_q,
#             'bias': best_bias,
#             'reconstructed_weight': W_final.view(N, M_pad)[:, :M].reshape(orig_shape),
#         }


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

# def assign_fp4_dynamic_vectorized(W_abs, alpha, E_bits, M_bits, bias):
#     """
#     Vectorized FP4 assignment matching the logic of assign_fp4_dynamic.
#     """
#     device = W_abs.device
#     if alpha.dim() == 2:
#         alpha = alpha.unsqueeze(-1)
    
#     # Normalize
#     x = W_abs / alpha.clamp_min(1e-8)
    
#     # Build Codebook
#     e_levels = torch.arange(2**E_bits, device=device).float()
#     m_levels = torch.arange(2**M_bits, device=device).float()
#     mantissa_factor = 1.0 + (m_levels / (2**M_bits))
#     cb = (2.0 ** e_levels.unsqueeze(1)) * mantissa_factor.unsqueeze(0)
#     codebook = cb.view(-1).sort()[0] 
    
#     # Distance Search
#     x_flat = x.reshape(-1, 1)
#     dist = (x_flat - codebook.view(1, -1)).abs()
#     indices = dist.argmin(dim=-1)
    
#     # Map back to Basis
#     exponent = indices // (2**M_bits)
#     mantissa = indices % (2**M_bits)
    
#     # Bias handling
#     if isinstance(bias, torch.Tensor):
#         # Expand bias to match flattened indices
#         b_flat = bias.unsqueeze(-1).expand_as(W_abs).reshape(-1)
#         eff_exp = exponent.float() - b_flat.float()
#     else:
#         eff_exp = exponent.float() - float(bias)
        
#     basis = (2.0 ** eff_exp) * (1.0 + mantissa.float() / (2**M_bits))
    
#     return exponent.view_as(W_abs), mantissa.view_as(W_abs), basis.view_as(W_abs)

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

# def chunked_lookup_full(x_norm, codebook, B_chunk=8, K_chunk=32):
#     N, B, bs = x_norm.shape
#     device = x_norm.device

#     idx_out = torch.empty((N, B, bs), dtype=torch.long, device=device)

#     for b_start in range(0, B, B_chunk):
#         b_end = min(b_start + B_chunk, B)

#         x_chunk = x_norm[:, b_start:b_end, :]  # [N, Bc, bs]

#         # local best
#         best_dist = torch.full_like(x_chunk, float('inf'))
#         best_idx  = torch.zeros_like(x_chunk, dtype=torch.long)

#         K = codebook.shape[0]

#         for k_start in range(0, K, K_chunk):
#             k_end = min(k_start + K_chunk, K)
#             cb_chunk = codebook[k_start:k_end]

#             dist = (x_chunk.unsqueeze(-1)
#                     - cb_chunk.view(1,1,1,-1)).abs()

#             local_idx = dist.argmin(dim=-1)
#             local_val = dist.gather(
#                 -1, local_idx.unsqueeze(-1)
#             ).squeeze(-1)

#             better = local_val < best_dist

#             best_dist = torch.where(better, local_val, best_dist)
#             best_idx  = torch.where(better, local_idx + k_start, best_idx)

#         idx_out[:, b_start:b_end, :] = best_idx

#     return idx_out


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


# def chunked_lookup_basis(x_norm, codebook, process_chunk, B_chunk=1, K_chunk=16):
#     """
#     Streaming nearest-codebook lookup.

#     Instead of returning idx or basis, it calls:
#         process_chunk(b_start, b_end, best_basis_chunk)

#     Shapes:
#         x_norm: [N, B, bs]
#         best_basis_chunk: [N, B_chunk, bs]
#     """
#     N, B, bs = x_norm.shape

#     for b_start in range(0, B, B_chunk):
#         b_end = min(b_start + B_chunk, B)

#         x_chunk = x_norm[:, b_start:b_end, :]  # [N, Bc, bs]

#         best_dist = torch.full_like(x_chunk, float('inf'))
#         best_basis = torch.zeros_like(x_chunk)

#         K = codebook.shape[0]

#         for k_start in range(0, K, K_chunk):
#             cb = codebook[k_start:k_start + K_chunk]  # [Kc]

#             dist = (x_chunk.unsqueeze(-1)
#                     - cb.view(1, 1, 1, -1)).abs()

#             idx = dist.argmin(dim=-1)
#             val = dist.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

#             better = val < best_dist

#             best_dist = torch.where(better, val, best_dist)
#             best_basis = torch.where(better, cb[idx], best_basis)

#         process_chunk(b_start, b_end, best_basis)

#         # free memory early
#         del best_dist, best_basis, x_chunk
#         torch.cuda.empty_cache()


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


# def reconstruct_layer_hadamard_v10_fast(
#     layer,
#     H_blocks_layer,
#     block_size: int,
#     e_bits: int,
#     m_bits: int,
#     e_bits_scale: int,
#     m_bits_scale: int,
#     device,
#     seed: int = 42,
#     chunk_row = 64,
# ):
#     """
#     Vectorised drop-in replacement for reconstruct_layer_hadamard_v10.
 
#     Key differences / speedups
#     --------------------------
#     1.  The double Python loop  ``for row / for blk``  is replaced by
#         fully batched tensor operations across all (N, n_blocks) pairs
#         simultaneously.
 
#     2.  The Hessian diagonal rotation is batched:
#             h_diag_stacked → _fht_blocks → h_had_importance  [n_blocks, pow2_bs]
#         One _fht call instead of n_blocks individual calls.
 
#     3.  The inner alpha-refinement loop still iterates 5× per bias
#         candidate, but every iteration operates on the full
#         [N, n_blocks, pow2_bs] tensor — no per-block Python overhead.
 
#     4.  quantize_scale is replaced by quantize_scale_tensor which works
#         directly on the [N, n_blocks] alpha tensor.
 
#     5.  The codebook is built once per bias candidate (not once per block).
 
#     6.  The final e/m decomposition is a single vectorised lookup over
#         the whole [N, M_pad] tensor.
 
#     The return dict is identical to reconstruct_layer_hadamard_v10.
#     """
 
#     W    = layer.weight.data.to(device)
#     mask = (W.abs() > 1e-9).float()
 
#     if W.dim() == 4:
#         W_mat    = W.view(W.shape[0], -1)
#         mask_mat = mask.view(W.shape[0], -1)
#     else:
#         W_mat    = W
#         mask_mat = mask
 
#     N, M = W_mat.shape
 
#     # ------------------------------------------------------------------
#     # Block geometry
#     # ------------------------------------------------------------------
#     pow2_bs  = (2 ** int(math.ceil(math.log2(block_size)))
#                 if block_size > 1 else 1)
#     n_blocks = math.ceil(M / block_size)
#     M_pad    = n_blocks * pow2_bs
 
#     W_pad    = F.pad(W_mat,    (0, M_pad - M)).contiguous()   # [N, M_pad]
#     mask_pad = F.pad(mask_mat, (0, M_pad - M)).contiguous()
 
#     # ------------------------------------------------------------------
#     # Deterministic pre-conditioning signs  [1, M_pad]
#     # ------------------------------------------------------------------
#     rng = torch.Generator(device=device)
#     rng.manual_seed(seed)
#     raw         = torch.randn(1, M_pad, device=device, generator=rng)
#     fixed_signs = torch.sign(raw)
#     fixed_signs[fixed_signs == 0] = 1.0
#     fixed_signs[:, M:] = 0.0                      # zero out padding columns
 
#     W_signed  = (W_pad * fixed_signs).contiguous()
#     W_had_all = _fht_blocks(W_signed, pow2_bs)    # [N, M_pad]
 
#     # ------------------------------------------------------------------
#     # Reshape into blocks:  [N, n_blocks, pow2_bs]
#     # ------------------------------------------------------------------
#     W_had   = W_had_all.view(N, n_blocks, pow2_bs)      # [N, B, bs]
#     mask_b  = mask_pad.view(N, n_blocks, pow2_bs)       # [N, B, bs]
 
#     # Identify fully-masked blocks (all padding / all zero)
#     # shape [N, B]  — True means we skip this block
#     block_dead = (mask_b.sum(dim=-1) == 0)
 
#     w_had_sign = torch.sign(W_had)                      # [N, B, bs]
#     w_had_abs  = W_had.abs()                            # [N, B, bs]
 
#     # ------------------------------------------------------------------
#     # Batch-rotate Hessian diagonals into the Hadamard domain
#     # H_blocks_layer is a list of n_blocks tensors, each [block_size, block_size]
#     # ------------------------------------------------------------------
#     # Build [n_blocks, block_size] diagonal matrix, pad to pow2_bs, rotate
#     h_diag_list = []
#     for blk in range(n_blocks):
#         H_blk  = H_blocks_layer[blk].to(device)        # [block_size, block_size]
#         h_diag = torch.diag(H_blk).clamp(min=1e-8)     # [block_size]
#         h_diag_p = F.pad(h_diag, (0, pow2_bs - h_diag.shape[0]))
#         h_diag_list.append(h_diag_p)
 
#     h_diag_stacked = torch.stack(h_diag_list)           # [B, pow2_bs]
 
#     # Single batched FHT over all blocks at once
#     h_had_importance = _fht(h_diag_stacked).abs().clamp(min=1e-8)
#     # shape [B, pow2_bs]
 
#     # Broadcast to [1, B, pow2_bs] for operations over N
#     h_imp = h_had_importance.unsqueeze(0)               # [1, B, bs]
 
#     # ------------------------------------------------------------------
#     # Alpha initialisation  [N, B]
#     # ------------------------------------------------------------------
#     alpha = (w_had_abs.pow(2).mean(dim=-1).sqrt()
#              .clamp(min=1e-4))                          # [N, B]
 
#     # ------------------------------------------------------------------
#     # Bias search — vectorised over (N, B) simultaneously
#     # ------------------------------------------------------------------
#     default_bias = 2 ** (e_bits - 1) - 1
#     bias_radius  = max(1, 2 ** (e_bits - 2))
#     bias_range   = list(range(default_bias - bias_radius,
#                                default_bias + bias_radius + 1))
 
#     best_loss  = torch.full((N, n_blocks), float('inf'), device=device)
#     best_alpha = alpha.clone()                          # [N, B]
#     best_bias  = torch.full((N, n_blocks), default_bias,
#                              device=device, dtype=torch.long)
 
#     for bias_cand in bias_range:
#         codebook = _build_codebook(e_bits, m_bits, bias_cand, device)
#         # [K]  where K = 2^e_bits * 2^m_bits
 
#         alpha_tmp = alpha.clone()   # [N, B]  — reset each candidate
 
#         # ----- 5-iteration alpha refinement -----
#         for _ in range(5):
#             # Normalise:  [N, B, bs] / [N, B, 1]
#             x_norm = (w_had_abs
#                       / alpha_tmp.unsqueeze(-1).clamp(min=1e-8))  # [N,B,bs]
 
#             # Nearest-codebook lookup: [N, B, bs, 1] vs [K]
#             # dist = (x_norm.unsqueeze(-1)
#             #         - codebook.view(1, 1, 1, -1)).abs()           # [N,B,bs,K]
#             # idx = chunked_lookup_full(x_norm, codebook, B_chunk=1, K_chunk=8)
#             N, B, bs = x_norm.shape
#             idx_out = torch.empty((N, B, bs), dtype=torch.long, device="cpu")

#             def writer(b_start, b_end, idx_chunk):
#                 idx_out[:, b_start:b_end, :] = idx_chunk.cpu()

#             chunked_lookup_full(x_norm, codebook, writer)
#             # idx  = dist.argmin(dim=-1)                             # [N,B,bs]
#             b    = codebook[idx_out]                                   # [N,B,bs]
 
#             # Diagonal-H OBS alpha update (all in one shot)
#             # num = sum_i  h_i * b_i * w_i
#             # den = sum_i  h_i * b_i * b_i
#             hb  = h_imp * b                                        # [N,B,bs]
#             num = (hb * w_had_abs).sum(dim=-1)                     # [N,B]
#             den = (hb * b).sum(dim=-1) + 1e-8                      # [N,B]
#             alpha_tmp = (num / den).clamp(min=1e-6)                # [N,B]
 
#         # ----- Evaluate H-weighted loss for this bias candidate -----
#         # Reconstruct in Hadamard domain (sign still absent for abs path)
#         x_norm = (w_had_abs
#                   / alpha_tmp.unsqueeze(-1).clamp(min=1e-8))
#         dist   = (x_norm.unsqueeze(-1)
#                   - codebook.view(1, 1, 1, -1)).abs()
#         b      = codebook[dist.argmin(dim=-1)]                     # [N,B,bs]
 
#         recon    = w_had_sign * alpha_tmp.unsqueeze(-1) * b        # [N,B,bs]
#         residual = W_had - recon                                    # [N,B,bs]
#         loss     = (h_imp * residual.pow(2)).sum(dim=-1)           # [N,B]
 
#         # Keep results where this candidate is better
#         improved   = loss < best_loss                              # [N,B] bool
#         best_loss  = torch.where(improved, loss,       best_loss)
#         best_alpha = torch.where(improved, alpha_tmp,  best_alpha)
#         best_bias  = torch.where(improved,
#                                   torch.full_like(best_bias, bias_cand),
#                                   best_bias)
 
#     # ------------------------------------------------------------------
#     # Quantise alpha  (vectorised, no Python loop)   [N, B]
#     # ------------------------------------------------------------------
#     alpha_q = quantize_scale_tensor(best_alpha, e_bits_scale, m_bits_scale)
#     # [N, B]
 
#     # ------------------------------------------------------------------
#     # Final FP assignment using per-block best_bias
#     # We need per-block codebooks; since bias varies per block we loop
#     # only over *unique* bias values — usually just 1-3 values total.
#     # ------------------------------------------------------------------
#     W_q_had_all_out = torch.zeros(N, M_pad, device=device)
#     alpha_out_full  = torch.zeros(N, M_pad, device=device)
#     e_out_full      = torch.zeros(N, M_pad, device=device, dtype=torch.long)
#     m_out_full      = torch.zeros(N, M_pad, device=device, dtype=torch.long)
#     sign_out_full   = torch.zeros(N, M_pad, device=device)
 
#     unique_biases = best_bias.unique().tolist()
 
#     for bias_val in unique_biases:
#         bias_val = int(bias_val)
#         codebook = _build_codebook(e_bits, m_bits, bias_val, device)   # [K]
#         K        = codebook.shape[0]
#         M_count  = 2 ** m_bits
 
#         # Mask: which (row, block) pairs use this bias value
#         bmask = (best_bias == bias_val)  # [N, B]
 
#         if not bmask.any():
#             continue
 
#         # alpha for these pairs  [N, B]  (0 elsewhere — harmless)
#         aq = alpha_q * bmask.float()   # [N, B]
 
#         # Nearest lookup for ALL blocks (inactive ones will be overwritten)
#         x_norm = (w_had_abs
#                   / (aq.unsqueeze(-1).clamp(min=1e-8)))            # [N,B,bs]
#         dist   = (x_norm.unsqueeze(-1)
#                   - codebook.view(1, 1, 1, -1)).abs()              # [N,B,bs,K]
#         idx    = dist.argmin(dim=-1)                               # [N,B,bs]
#         basis  = codebook[idx]                                     # [N,B,bs]
#         exp_t  = idx // M_count                                    # [N,B,bs]
#         mant_t = idx  % M_count                                    # [N,B,bs]
 
#         # Reconstructed Hadamard-domain values
#         w_q = w_had_sign * aq.unsqueeze(-1) * basis                # [N,B,bs]
 
#         # Write only for blocks that use this bias
#         write = bmask.unsqueeze(-1).expand_as(w_q)                 # [N,B,bs]
 
#         # Reshape to [N, M_pad] for writing
#         W_q_had_all_out = W_q_had_all_out.view(N, n_blocks, pow2_bs)
#         alpha_out_full  = alpha_out_full .view(N, n_blocks, pow2_bs)
#         e_out_full      = e_out_full     .view(N, n_blocks, pow2_bs)
#         m_out_full      = m_out_full     .view(N, n_blocks, pow2_bs)
#         sign_out_full   = sign_out_full  .view(N, n_blocks, pow2_bs)
 
#         W_q_had_all_out = torch.where(write, w_q,
#                                        W_q_had_all_out)
#         alpha_out_full  = torch.where(write,
#                                        aq.unsqueeze(-1).expand_as(w_q),
#                                        alpha_out_full)
#         e_out_full      = torch.where(write, exp_t,  e_out_full)
#         m_out_full      = torch.where(write, mant_t, m_out_full)
#         sign_out_full   = torch.where(write, w_had_sign, sign_out_full)
 
#         W_q_had_all_out = W_q_had_all_out.view(N, M_pad)
#         alpha_out_full  = alpha_out_full .view(N, M_pad)
#         e_out_full      = e_out_full     .view(N, M_pad)
#         m_out_full      = m_out_full     .view(N, M_pad)
#         sign_out_full   = sign_out_full  .view(N, M_pad)
 
#     # ------------------------------------------------------------------
#     # Zero out fully-dead blocks (all-padding / all-zero input)
#     # ------------------------------------------------------------------
#     dead_mask = block_dead.unsqueeze(-1).expand(N, n_blocks, pow2_bs)
#     dead_mask = dead_mask.reshape(N, M_pad)
#     W_q_had_all_out = W_q_had_all_out.masked_fill(dead_mask, 0.0)
 
#     # ------------------------------------------------------------------
#     # Inverse FHT  (vectorised, single call)
#     # ------------------------------------------------------------------
#     W_q_spatial = _fht_blocks(W_q_had_all_out.contiguous(), pow2_bs)
 
#     W_q_spatial = W_q_spatial * fixed_signs       # undo pre-conditioning
#     W_q_spatial = W_q_spatial[:, :M].contiguous() # strip padding
#     W_q_spatial = W_q_spatial * mask_mat          # reapply sparsity mask
 
#     # ------------------------------------------------------------------
#     # Return dict — identical interface to the original v10
#     # ------------------------------------------------------------------
#     return {
#         'alpha':                alpha_out_full[:, :M].view_as(W),
#         'exponent':             e_out_full    [:, :M].view_as(W),
#         'mantissa':             m_out_full    [:, :M].view_as(W),
#         'sign':                 sign_out_full [:, :M].view_as(W),
#         'bias':                 best_bias,              # [N, n_blocks]
#         'fixed_signs':          fixed_signs[:, :M],     # [1, M]
#         'reconstructed_weight': W_q_spatial.view_as(W),
#     }


# def reconstruct_layer_hadamard_v10_fast(
#     layer,
#     H_blocks_layer,
#     block_size: int,
#     e_bits: int,
#     m_bits: int,
#     e_bits_scale: int,
#     m_bits_scale: int,
#     device,
#     seed: int = 42,
# ):



#     # -----------------------------
#     # Setup
#     # -----------------------------
#     with torch.no_grad():

#         W = layer.weight.data.to(device)
#         mask = (W.abs() > 1e-9).float()

#         if W.dim() == 4:
#             W_mat = W.view(W.shape[0], -1)
#             mask_mat = mask.view(W.shape[0], -1)
#         else:
#             W_mat = W
#             mask_mat = mask

#         N, M = W_mat.shape

#         pow2_bs = 2 ** int(math.ceil(math.log2(block_size)))
#         n_blocks = math.ceil(M / block_size)
#         M_pad = n_blocks * pow2_bs

#         W_pad = F.pad(W_mat, (0, M_pad - M))
#         mask_pad = F.pad(mask_mat, (0, M_pad - M))

#         # -----------------------------
#         # Hadamard transform
#         # -----------------------------
#         rng = torch.Generator(device=device)
#         rng.manual_seed(seed)
#         fixed_signs = torch.sign(torch.randn(1, M_pad, device=device, generator=rng))
#         fixed_signs[fixed_signs == 0] = 1.0
#         fixed_signs[:, M:] = 0.0

#         W_signed = W_pad * fixed_signs
#         W_had = _fht_blocks(W_signed, pow2_bs).view(N, n_blocks, pow2_bs)

#         # mask_b = mask_pad.view(N, n_blocks, pow2_bs)
#         w_had_sign = torch.sign(W_had)
#         w_had_abs = W_had.abs()

#         # -----------------------------
#         # Hessian diag → Hadamard
#         # -----------------------------
#         h_diag = []
#         for blk in range(n_blocks):
#             d = torch.diag(H_blocks_layer[blk].to(device)).clamp(min=1e-8)
#             h_diag.append(F.pad(d, (0, pow2_bs - d.shape[0])))

#         h_imp = _fht(torch.stack(h_diag)).abs().clamp(min=1e-8)
#         h_imp = h_imp.unsqueeze(0)  # [1, B, bs]

#         # -----------------------------
#         # Init
#         # -----------------------------
#         alpha = (w_had_abs.pow(2).mean(dim=-1).sqrt().clamp(min=1e-4))

#         default_bias = 2 ** (e_bits - 1) - 1
#         bias_radius = max(1, 2 ** (e_bits - 2))
#         bias_range = list(range(default_bias - bias_radius,
#                                 default_bias + bias_radius + 1))

#         best_loss = torch.full((N, n_blocks), float('inf'), device=device)
#         best_alpha = alpha.clone()
#         best_bias = torch.full((N, n_blocks), default_bias, dtype=torch.long, device=device)

#         # =============================
#         # Bias search
#         # =============================
#         for bias_cand in bias_range:

#             codebook = _build_codebook(e_bits, m_bits, bias_cand, device)
#             alpha_tmp = alpha.clone()

#             # ----- alpha refinement -----
#             for _ in range(5):

#                 num = torch.zeros((N, n_blocks), device=device)
#                 den = torch.zeros((N, n_blocks), device=device)

#                 x_norm = w_had_abs / alpha_tmp.unsqueeze(-1).clamp(min=1e-8)

#                 def alpha_writer(b_start, b_end, b_chunk):
#                     h_chunk = h_imp[:, b_start:b_end, :]
#                     w_chunk = w_had_abs[:, b_start:b_end, :]

#                     hb = h_chunk * b_chunk

#                     num[:, b_start:b_end] = (hb * w_chunk).sum(dim=-1)
#                     den[:, b_start:b_end] = (hb * b_chunk).sum(dim=-1)

#                 chunked_lookup_basis(x_norm, codebook, alpha_writer)

#                 alpha_tmp = (num / (den + 1e-8)).clamp(min=1e-6)

#             # ----- loss eval -----
#             loss = torch.zeros((N, n_blocks), device=device)

#             x_norm = w_had_abs / alpha_tmp.unsqueeze(-1).clamp(min=1e-8)

#             def loss_writer(b_start, b_end, b_chunk):
#                 s = w_had_sign[:, b_start:b_end, :]
#                 a = alpha_tmp[:, b_start:b_end].unsqueeze(-1)

#                 recon = s * a * b_chunk
#                 residual = W_had[:, b_start:b_end, :] - recon

#                 h_chunk = h_imp[:, b_start:b_end, :]
#                 loss[:, b_start:b_end] = (h_chunk * residual.pow(2)).sum(dim=-1)

#             chunked_lookup_basis(x_norm, codebook, loss_writer)

#             improved = loss < best_loss
#             best_loss = torch.where(improved, loss, best_loss)
#             best_alpha = torch.where(improved, alpha_tmp, best_alpha)
#             best_bias = torch.where(
#                 improved,
#                 torch.full_like(best_bias, bias_cand),
#                 best_bias
#             )
#         del alpha_tmp, num, den, loss
#         # -----------------------------
#         # Quantize alpha
#         # -----------------------------
#         alpha_q = quantize_scale_tensor(best_alpha, e_bits_scale, m_bits_scale)

#         # -----------------------------
#         # Final reconstruction
#         # -----------------------------
#         W_q = torch.zeros(N, M_pad, device=device).view(N, n_blocks, pow2_bs)

#         for bias_val in best_bias.unique().tolist():
#             codebook = _build_codebook(e_bits, m_bits, int(bias_val), device)

#             bmask = (best_bias == bias_val)
#             aq = alpha_q * bmask.float()

#             x_norm = w_had_abs / aq.unsqueeze(-1).clamp(min=1e-8)

#             def final_writer(b_start, b_end, b_chunk):
#                 s = w_had_sign[:, b_start:b_end, :]
#                 a = aq[:, b_start:b_end].unsqueeze(-1)

#                 recon = s * a * b_chunk
#                 write = bmask[:, b_start:b_end].unsqueeze(-1)

#                 W_q[:, b_start:b_end, :] = torch.where(
#                     write,
#                     recon,
#                     W_q[:, b_start:b_end, :]
#                 )

#             chunked_lookup_basis(x_norm, codebook, final_writer)

#         W_q = W_q.view(N, M_pad)

#         # -----------------------------
#         # Inverse Hadamard
#         # -----------------------------
#         W_out = _fht_blocks(W_q, pow2_bs)
#         W_out = W_out * fixed_signs
#         W_out = W_out[:, :M]
#         W_out = W_out * mask_mat
#         alpha_expanded = best_alpha.unsqueeze(-1).expand(-1, -1, pow2_bs)
#         alpha_expanded = alpha_expanded.reshape(N, M_pad)[:, :M]
#         del W_q
#         return {
#             'alpha': alpha_expanded.view_as(W),
#             'bias': best_bias,
#             'reconstructed_weight': W_out.view_as(W),
#         }


# def reconstruct_layer_hadamard_v10_fast(
#     layer,
#     H_blocks_layer,
#     block_size: int,
#     e_bits: int,
#     m_bits: int,
#     e_bits_scale: int,
#     m_bits_scale: int,
#     device,
#     seed: int = 42,
# ):
#     with torch.no_grad():

#         W = layer.weight.data.to(device)
#         mask = (W.abs() > 1e-9).float()

#         if W.dim() == 4:
#             W_mat = W.view(W.shape[0], -1)
#             mask_mat = mask.view(W.shape[0], -1)
#         else:
#             W_mat = W
#             mask_mat = mask

#         N, M = W_mat.shape
#         del W, mask
#         torch.cuda.empty_cache()

#         pow2_bs  = 2 ** int(math.ceil(math.log2(block_size)))
#         n_blocks = math.ceil(M / block_size)
#         M_pad    = n_blocks * pow2_bs

#         W_pad    = F.pad(W_mat, (0, M_pad - M))
#         mask_pad = F.pad(mask_mat, (0, M_pad - M))
#         del W_mat
#         torch.cuda.empty_cache()

#         # Hadamard transform
#         rng = torch.Generator(device=device)
#         rng.manual_seed(seed)
#         fixed_signs = torch.sign(torch.randn(1, M_pad, device=device, generator=rng))
#         fixed_signs[fixed_signs == 0] = 1.0
#         fixed_signs[:, M:] = 0.0

#         W_signed = W_pad * fixed_signs
#         del W_pad
#         torch.cuda.empty_cache()

#         W_had      = _fht_blocks(W_signed, pow2_bs).view(N, n_blocks, pow2_bs)
#         del W_signed
#         torch.cuda.empty_cache()

#         w_had_sign = torch.sign(W_had)
#         w_had_abs  = W_had.abs()
#         del W_had
#         torch.cuda.empty_cache()

#         # Hessian diag → Hadamard — keep on CPU until needed
#         h_diag = []
#         for blk in range(n_blocks):
#             d = torch.diag(H_blocks_layer[blk].to(device)).clamp(min=1e-8)
#             h_diag.append(F.pad(d, (0, pow2_bs - d.shape[0])).cpu())
#             del d

#         h_imp_cpu = _fht(torch.stack(h_diag)).abs().clamp(min=1e-8)  # [n_blocks, pow2_bs] on CPU
#         del h_diag
#         torch.cuda.empty_cache()

#         # Init — keep best_* on CPU to save VRAM
#         alpha = w_had_abs.pow(2).mean(dim=-1).sqrt().clamp(min=1e-4)  # [N, n_blocks]

#         default_bias = 2 ** (e_bits - 1) - 1
#         bias_radius  = max(1, 2 ** (e_bits - 2))
#         bias_range   = list(range(default_bias - bias_radius,
#                                   default_bias + bias_radius + 1))

#         best_loss  = torch.full((N, n_blocks), float('inf'))          # CPU
#         best_alpha = alpha.clone().cpu()                               # CPU
#         best_bias  = torch.full((N, n_blocks), default_bias,
#                                 dtype=torch.long)                      # CPU

#         # Bias search — process one block at a time to limit peak memory
#         for bias_cand in bias_range:
#             codebook  = _build_codebook(e_bits, m_bits, bias_cand, device)
#             alpha_tmp = alpha.clone()  # [N, n_blocks] on GPU

#             # Alpha refinement
#             for _ in range(5):
#                 num = torch.zeros((N, n_blocks), device=device)
#                 den = torch.zeros((N, n_blocks), device=device)

#                 for blk in range(n_blocks):
#                     h_chunk = h_imp_cpu[blk].to(device).unsqueeze(0)  # [1, pow2_bs]
#                     w_chunk = w_had_abs[:, blk, :]                     # [N, pow2_bs]
#                     a_chunk = alpha_tmp[:, blk].unsqueeze(1)           # [N, 1]

#                     x_norm = w_chunk / a_chunk.clamp(min=1e-8)
#                     # lookup basis via codebook
#                     dists  = (x_norm.unsqueeze(-1) - codebook.unsqueeze(0).unsqueeze(0)).abs()
#                     b_chunk = codebook[dists.argmin(dim=-1)]
#                     del dists, x_norm

#                     hb = h_chunk * b_chunk
#                     num[:, blk] = (hb * w_chunk).sum(dim=-1)
#                     den[:, blk] = (hb * b_chunk).sum(dim=-1)
#                     del hb, b_chunk, h_chunk, w_chunk, a_chunk
#                     torch.cuda.empty_cache()

#                 alpha_tmp = (num / (den + 1e-8)).clamp(min=1e-6)
#                 del num, den

#             # Loss eval per block
#             loss = torch.zeros((N, n_blocks), device=device)
#             for blk in range(n_blocks):
#                 h_chunk = h_imp_cpu[blk].to(device).unsqueeze(0)
#                 w_chunk = w_had_abs[:, blk, :]
#                 s_chunk = w_had_sign[:, blk, :]
#                 a_chunk = alpha_tmp[:, blk].unsqueeze(1)

#                 x_norm  = w_chunk / a_chunk.clamp(min=1e-8)
#                 dists   = (x_norm.unsqueeze(-1) - codebook.unsqueeze(0).unsqueeze(0)).abs()
#                 b_chunk = codebook[dists.argmin(dim=-1)]
#                 del dists, x_norm

#                 recon    = s_chunk * a_chunk * b_chunk
#                 residual = (W_had_orig[:, blk, :] if False else
#                             w_had_sign[:, blk, :] * w_had_abs[:, blk, :]) - recon
#                 loss[:, blk] = (h_chunk * residual.pow(2)).sum(dim=-1)
#                 del h_chunk, w_chunk, s_chunk, a_chunk, b_chunk, recon, residual
#                 torch.cuda.empty_cache()

#             improved   = loss.cpu() < best_loss
#             best_loss  = torch.where(improved, loss.cpu(), best_loss)
#             best_alpha = torch.where(improved, alpha_tmp.cpu(), best_alpha)
#             best_bias  = torch.where(
#                 improved,
#                 torch.full_like(best_bias, bias_cand),
#                 best_bias)
#             del loss, alpha_tmp, improved, codebook
#             torch.cuda.empty_cache()

#         del alpha, best_loss
#         torch.cuda.empty_cache()

#         # Quantize alpha
#         best_alpha_gpu = best_alpha.to(device)
#         alpha_q        = quantize_scale_tensor(best_alpha_gpu, e_bits_scale, m_bits_scale)
#         del best_alpha_gpu
#         torch.cuda.empty_cache()

#         # Final reconstruction — one block at a time
#         best_bias_gpu = best_bias.to(device)
#         W_q           = torch.zeros(N, M_pad, device=device).view(N, n_blocks, pow2_bs)

#         for bias_val in best_bias_gpu.unique().tolist():
#             codebook = _build_codebook(e_bits, m_bits, int(bias_val), device)
#             bmask    = (best_bias_gpu == bias_val)

#             for blk in range(n_blocks):
#                 blk_mask = bmask[:, blk]
#                 if not blk_mask.any():
#                     continue

#                 aq      = alpha_q[:, blk] * blk_mask.float()
#                 w_chunk = w_had_abs[:, blk, :]
#                 s_chunk = w_had_sign[:, blk, :]

#                 x_norm  = w_chunk / aq.unsqueeze(1).clamp(min=1e-8)
#                 dists   = (x_norm.unsqueeze(-1) - codebook.unsqueeze(0).unsqueeze(0)).abs()
#                 b_chunk = codebook[dists.argmin(dim=-1)]
#                 del dists, x_norm

#                 recon = s_chunk * aq.unsqueeze(1) * b_chunk
#                 W_q[:, blk, :] = torch.where(
#                     blk_mask.unsqueeze(1).expand_as(recon),
#                     recon,
#                     W_q[:, blk, :])
#                 del recon, b_chunk, w_chunk, s_chunk, aq, blk_mask
#                 torch.cuda.empty_cache()

#             del codebook, bmask
#             torch.cuda.empty_cache()

#         del w_had_abs, w_had_sign, best_bias_gpu
#         torch.cuda.empty_cache()

#         W_q  = W_q.view(N, M_pad)

#         # Inverse Hadamard
#         W_out = _fht_blocks(W_q, pow2_bs)
#         del W_q
#         W_out = W_out * fixed_signs
#         W_out = W_out[:, :M] * mask_mat
#         del fixed_signs, mask_mat

#         alpha_q_exp = alpha_q.unsqueeze(-1).expand(-1, -1, pow2_bs)
#         alpha_q_exp = alpha_q_exp.reshape(N, M_pad)[:, :M]
#         del alpha_q

#         layer_W = layer.weight.data.to(device)
#         return {
#             'alpha':                alpha_q_exp.view_as(layer_W),
#             'bias':                 best_bias.to(device),
#             'reconstructed_weight': W_out.view_as(layer_W),
#         }


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
# def reconstruct_layer_fp_blockdiag_scaled(
#     layer,
#     H_blocks_layer,
#     block_size,
#     e_bits,
#     m_bits,
#     e_bits_scale,
#     m_bits_scale,
#     device
# ):
#     """
#     Blockwise FP4 reconstruction with optional Hessian scaling.
    
#     Args:
#         layer: nn.Module layer (Linear or Conv)
#         H_blocks_layer: list of Hessian blocks per weight row
#         block_size: block size for reconstruction
#         e_bits, m_bits: FP4 bits for exponent and mantissa
#         e_bits_scale, m_bits_scale: FP4 bits for alpha scale
#         device: torch device
#     Returns:
#         alpha: per-block scales
#         e: FP4 exponent tensor
#         m: FP4 mantissa tensor
#         sign: sign of reconstructed weights
#     """

#     W = layer.weight.data.to(device)
#     mask = (W != 0).float()

#     # Flatten convolution weights
#     if W.dim() == 4:
#         W_mat = W.view(W.shape[0], -1)
#         mask_mat = mask.view(W.shape[0], -1)
#     else:
#         W_mat = W
#         mask_mat = mask

#     N, M = W_mat.shape
#     W_q = torch.zeros_like(W_mat)

#     # Loop over rows
#     for row in range(N):
#         w_row = W_mat[row]
#         m_row = mask_mat[row]

#         for i in range(0, M, block_size):
#             end = min(i + block_size, M)
#             w_block = w_row[i:end]
#             mask_block = m_row[i:end]

#             if mask_block.sum() < 1e-8:
#                 continue

#             block_idx = i // block_size
#             H_block = H_blocks_layer[block_idx].to(device)
#             k_block = w_block.numel()

#             # Ensure H_block matches block size
#             if H_block.shape[0] != k_block:
#                 H_block = H_block[:k_block, :k_block]

#             # --- FP4 block reconstruction ---
#             def fp4_quant_block(z_block):
#                 """
#                 z_block: 1D tensor for this block
#                 Returns reconstructed FP4 block
#                 """
#                 mask_local = (z_block != 0).float()
#                 k_local = z_block.numel()

#                 # initialize alpha on full block
#                 alpha = initialize_alpha(z_block, mask_local, k_local)

#                 for _ in range(3):
#                     e, m, basis = assign_fp4(z_block.unsqueeze(0), alpha, e_bits, m_bits)
#                     basis = basis.squeeze(0)
#                     alpha = solve_alpha_blockwise_Hessian_correct(
#                         z_block, basis, H_block[:k_local, :k_local], mask_local, k_local
#                     )
#                     alpha = quantize_scale(alpha, e_bits_scale, m_bits_scale)

#                 return alpha * basis

#             # Quantize block
#             w_q_block = fp4_quant_block(w_block)
#             W_q[row, i:end] = w_q_block

#     # Reshape back to original shape
#     if W.dim() == 4:
#         W_q = W_q.view_as(W)

#     sign = torch.sign(W_q)
#     W_abs = W_q.abs()

#     # Recover FP4 components (optional, for logging/consistency)
#     alpha = initialize_alpha(W_abs, (W_q != 0).float(), block_size)
#     alpha = quantize_scale(alpha, e_bits_scale, m_bits_scale)
#     e, m, _ = assign_fp4(W_abs, alpha, e_bits, m_bits)

#     return alpha, e, m, sign
# =========================================================
# 🔹 QUANTIZED LINEAR
# =========================================================
class QuantLinearFP(nn.Module):
    def __init__(self, linear, block_size, e_bits, m_bits, e_bits_scale, m_bits_scale):
        super().__init__()
        self.linear = linear
        self.block_size = block_size
        self.e_bits = e_bits
        self.m_bits = m_bits
        self.e_bits_scale = e_bits_scale
        self.m_bits_scale = m_bits_scale
        self.register_buffer("weight_q", None)

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
    def calibrate_Hessian_scaled(self, data_loader, device, H_block):
        weight_q = reconstruct_layer_fp_blockdiag_scaled_v5(
            self.linear,          # or self.conv / self.conv1d
            H_block,
            self.block_size,
            self.e_bits,
            self.m_bits,
            self.e_bits_scale,
            self.m_bits_scale,
            device=device)
        self.weight_q = weight_q.view_as(self.linear.weight)  # or .conv.weight / .conv1d.weight
        del weight_q
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


    def forward(self, x):
        return F.linear(x, self.weight_q if self.weight_q is not None else self.linear.weight, self.linear.bias)

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
    def calibrate_Hessian_scaled(self, data_loader, device, H_block):
        weight_q = reconstruct_layer_fp_blockdiag_scaled_v5(
            self.conv,          # or self.conv / self.conv1d
            H_block,
            self.block_size,
            self.e_bits,
            self.m_bits,
            self.e_bits_scale,
            self.m_bits_scale,
            device=device)
        self.weight_q = weight_q.view_as(self.linear.weight)  # or .conv.weight / .conv1d.weight
        del weight_q
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
    def forward(self, x):
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
 
    def calibrate_Hessian_scaled(self, data_loader, device, H_block):
        weight_q = reconstruct_layer_fp_blockdiag_scaled_v5(
            self.linear,          # or self.conv / self.conv1d
            H_block,
            self.block_size,
            self.e_bits,
            self.m_bits,
            self.e_bits_scale,
            self.m_bits_scale,
            device=device)
        self.weight_q = weight_q.view_as(self.linear.weight)  # or .conv.weight / .conv1d.weight
        del weight_q
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
 
    def forward(self, x):
        """
        GPT-2's Conv1D forward is: x @ W + b  (weight is (in, out))
        We replicate that exactly using the quantized weight when available.
        """
        W = self.weight_q if self.weight_q is not None else self.conv1d.weight
        bias = self.conv1d.bias
        return x @ W + bias if bias is not None else x @ W
 
 

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


# def calibrate_model_Hessian_Hadamard(model, data_loader, block_size, device):
#     model.eval().to(device)
#     H_dict_block = compute_hessian_blockdiag_model(
#         model, data_loader, device, block_size
#     )
#     for module in model.modules():
#         if isinstance(module, (QuantLinearFP, QuantConv2dFP, QuantConv1dFP)):
#             if hasattr(module, "linear") and module.linear in H_dict_block:
#                 H_block = H_dict_block[module.linear]
#             elif hasattr(module, "conv") and module.conv in H_dict_block:
#                 H_block = H_dict_block[module.conv]
#             elif hasattr(module, "conv1d") and module.conv1d in H_dict_block:
#                 H_block = H_dict_block[module.conv1d]
#             else:
#                 print(f"⚠️ Missing Hessian for {module}, skipping Hadamard calibration")
#                 continue
#             print(f"Calibrating {module} with Hessian-scaled FP4")
#             module.calibrate_Hessian_Hadamard(data_loader, device, H_block)
#     return model

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
                module.weight_q *= mask

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
                      use_adap=False, use_forward=False, Hadamard=False):
    """
    Quantize a model to FP4-like format with optional HG or Hessian calibration.
    """

    model = fold_bn_recursively(model)

    # Freeze weights after folding
    for p in model.parameters():
        p.requires_grad = False

    # Replace layers with quantized wrappers
    model = replace_layers(model,
                           block_size,
                           e_bits,
                           m_bits,
                           e_bits_scale,
                           m_bits_scale)
    # Count what's in the model before quantization
    quant_count = 0
    conv1d_count = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantConv1dFP):
            quant_count += 1
        if type(module).__name__ == "Conv1D":
            conv1d_count += 1

    print(f"QuantConv1dFP modules in model: {quant_count}")
    print(f"Raw Conv1D still in model: {conv1d_count}")

    # Calibration step
    if use_Hessian:
        print("Using Hessian block calibration")
        model = calibrate_model_Hessian_block(model, data_loader, block_size, device)
    elif use_HG:
        print("Using HG calibration")
        model = calibrate_model_HG(model, data_loader, device)
    elif use_adap:
        print("Using adaptive mesh calibration")
        model = calibrate_model_Hessian_scaled(model, data_loader, block_size, device)
    elif use_forward:
        print("Using forward reconstruction calibration")
        model = calibrate_model_Hessian_scaled_forward(model, data_loader, block_size, device)
    elif Hadamard:
        print("Using Hadamard-domain calibration")
        model = calibrate_model_Hessian_Hadamard(model, data_loader, block_size, device)
    else:
        print("Using standard calibration")
        model = calibrate_model(model, data_loader, device)

    # Apply pruning masks (if any)
    apply_pruning_mask(model)
# Check for nan/inf in quantized weights
# After quantization, check weight_q statistics per layer
    for name, module in model.named_modules():
        if hasattr(module, 'weight_q') and module.weight_q is not None:
            wq = module.weight_q
            print(f"{name:60s} | max={wq.abs().max():.4f} | mean={wq.abs().mean():.4f} | "
                f"nan={torch.isnan(wq).any()} | inf={torch.isinf(wq).any()} | "
                f"zeros={( wq.abs() < 1e-8).float().mean():.3f}")
    # After quantization, manually test one layer
    for name, module in model.named_modules():
        if hasattr(module, 'weight_q') and module.weight_q is not None:
            # Compare output of quantized vs original weight on same input
            x_test = torch.randn(1, module.linear.in_features).to(device)
            out_orig = F.linear(x_test, module.linear.weight, module.linear.bias)
            out_quant = F.linear(x_test, module.weight_q, module.linear.bias)
            print(f"{name}: orig_max={out_orig.abs().max():.4f}, quant_max={out_quant.abs().max():.4f}")
            print(f"  relative error: {((out_orig - out_quant).abs() / out_orig.abs().clamp(min=1e-8)).mean():.4f}")
            break
    return model




def smooth_layer(layer, alpha=0.5):
    """
    Migrate outliers from weights to the preceding activation scale.
    alpha controls the migration strength — 0.5 is the standard default.
    layer.weight is (out, in) — we scale input channels.
    """
    W = layer.weight.data                          # (out, in)
    
    # Per input-channel max absolute value
    w_scale = W.abs().max(dim=0).values            # (in,)
    w_scale = w_scale.clamp(min=1e-8)
    
    # Smooth scale — alpha controls balance between W and X
    smooth_scale = w_scale.pow(alpha)              # (in,)
    
    # Absorb into weight — divide each input channel by smooth_scale
    W_smoothed = W / smooth_scale.unsqueeze(0)     # (out, in)
    layer.weight.data = W_smoothed
    
    return smooth_scale  # caller absorbs this into preceding LayerNorm


def apply_smoothquant(model):
    """
    Apply SmoothQuant channel migration to BLOOM's linear layers.
    For each transformer block, smooth the QKV/dense/MLP weights
    and absorb the scale into the preceding LayerNorm.
    """
    for i, block in enumerate(model.transformer.h):
        # --- Attention input LayerNorm → QKV projection ---
        ln   = block.input_layernorm
        qkv  = block.self_attention.query_key_value
        scale = smooth_layer(qkv, alpha=0.5)
        # Absorb into LayerNorm weight and bias
        ln.weight.data *= scale
        if ln.bias is not None:
            ln.bias.data *= scale

        # --- Post-attention LayerNorm → MLP ---
        ln2  = block.post_attention_layernorm
        fc1  = block.mlp.dense_h_to_4h
        scale2 = smooth_layer(fc1, alpha=0.5)
        ln2.weight.data *= scale2
        if ln2.bias is not None:
            ln2.bias.data *= scale2

        # dense (attention output projection) takes post-attention hidden states
        # These don't have a preceding LayerNorm to absorb into,
        # so we skip or use a smaller alpha
        # smooth_layer(block.self_attention.dense, alpha=0.3)

    return model
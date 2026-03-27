import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """
    Returns block-diagonal Hessian approximation.

    Output:
        H_blocks: list of tensors per row
                  each is [num_blocks, block_size, block_size]
    """
    if isinstance(layer, nn.Conv2d):
        unfold = nn.Unfold(
            kernel_size=layer.kernel_size,
            dilation=layer.dilation,
            padding=layer.padding,
            stride=layer.stride
        )
        x = unfold(x)  # [B, C*k*k, L]
        x = x.permute(0, 2, 1).reshape(-1, x.shape[1])  # [N, D]

    elif isinstance(layer, nn.Linear):
        x = x.reshape(-1, x.shape[-1])  # [N, D]

    else:
        return None

    N, D = x.shape
    H = x.T @ x  # [D, D]

    # Split into blocks
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

def compute_hessian_blockdiag_model(model, data_loader, device, block_size, num_batches=10):
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
                # accumulate
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

    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            x = x.to(device)
            model(x)
            if i + 1 >= num_batches:
                break

    for h in handles:
        h.remove()

    # normalize
    H_final = {}
    for name, blocks in H_data.items():
        H_final[hook_map[name]] = [b / num_batches for b in blocks]

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

def compute_hessian_diag_model(model, data_loader, device, num_batches=10):
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
    def forward(self, x):
        return F.conv2d(x,
                        self.weight_q if self.weight_q is not None else self.conv.weight,
                        self.conv.bias,
                        stride=self.conv.stride,
                        padding=self.conv.padding)

# =========================================================
# 🔹 REPLACE LAYERS
# =========================================================
def replace_layers(model, block_size, e_bits, m_bits, e_bits_scale, m_bits_scale):
    """
    Recursively replace Linear and Conv2d layers with FP4 wrappers.
    Preserves residual connections and downsample blocks in ResNet.
    """
    for name, module in model.named_children():
        # Skip already replaced
        if isinstance(module, (QuantLinearFP, QuantConv2dFP)):
            continue

        # Replace linear
        if isinstance(module, nn.Linear):
            print("Found Linear")
            setattr(model, name, QuantLinearFP(module, block_size, e_bits, m_bits,
                                               e_bits_scale, m_bits_scale))

        # Replace conv2d
        elif isinstance(module, nn.Conv2d):
            print("Found Conv2D")
            setattr(model, name, QuantConv2dFP(module, block_size, e_bits, m_bits,
                                               e_bits_scale, m_bits_scale))

        # Recursively replace inside sequences, modules, downsample, etc.
        elif isinstance(module, nn.Sequential) or isinstance(module, nn.Module):
            # Preserve downsample by recursing
            replace_layers(module, block_size, e_bits, m_bits, e_bits_scale, m_bits_scale)
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
                      use_Hessian=False):
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

    # Calibration step
    if use_Hessian:
        print("Using Hessian block calibration")
        model = calibrate_model_Hessian_block(model, data_loader, block_size, device)
    elif use_HG:
        print("Using HG calibration")
        model = calibrate_model_HG(model, data_loader, device)
    else:
        print("Using standard calibration")
        model = calibrate_model(model, data_loader, device)

    # Apply pruning masks (if any)
    apply_pruning_mask(model)

    return model
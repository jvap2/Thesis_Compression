import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from Quantization_Experiments.brecq import get_reconstruct_blocks, cache_block_input, apply_bias_correction, replace_block

# =========================================================
# 🔹 FP FORMAT (WEIGHTS)
# =========================================================
class FPFormat:
    def __init__(self, e_bits=2, m_bits=1, bias=0):
        self.e_bits = e_bits
        self.m_bits = m_bits
        self.bias = bias
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exponents = self._build_exponents()
        self.mantissas = self._build_mantissas()

    def _build_exponents(self):
        vals = torch.arange(2 ** self.e_bits, device=self.device)
        return vals - self.bias

    def _build_mantissas(self):
        if self.m_bits == 0:
            return torch.tensor([0.0])
        levels = 2 ** self.m_bits
        return torch.arange(levels, device=self.device) / levels


# =========================================================
# 🔹 SCALE QUANTIZATION
# =========================================================
def quantize_scale(alpha, e_bits=8, m_bits=0):
    alpha = alpha.clamp(min=1e-12)

    e_min = -(2 ** (e_bits - 1))
    e_max = (2 ** (e_bits - 1)) - 1

    e = torch.floor(torch.log2(alpha))
    e = torch.clamp(e, e_min, e_max)

    base = 2.0 ** e

    if m_bits > 0:
        frac = alpha / base - 1.0
        levels = 2 ** m_bits
        frac_q = torch.round(frac * levels) / levels
        return base * (1.0 + frac_q)
    else:
        return 2.0 ** torch.round(e)


# =========================================================
# 🔹 LAYER → MATRIX (Linear + Conv2d)
# =========================================================
def extract_matrix(layer, X):
    if isinstance(layer, nn.Linear):
        return layer.weight, X

    elif isinstance(layer, nn.Conv2d):
        W = layer.weight

        B = X.shape[0]
        unfold = F.unfold(
            X,
            kernel_size=layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride
        )  # [B, C_in*kH*kW, L]

        # Permute and flatten to [B*L, C_in*kH*kW]
        X_mat = unfold.permute(0, 2, 1).reshape(B * unfold.shape[-1], -1)  # [B*L, C_in*kH*kW]

        # W_mat stays the same: [C_out, C_in*kH*kW]
        W_mat = W.view(W.shape[0], -1)

        return W_mat, X_mat

    else:
        raise NotImplementedError


# =========================================================
# 🔹 BUILD WEIGHT FROM FP PARAMS
# =========================================================
def build_weight(s, e, m):
    return s * torch.pow(2.0, e) * (1.0 + m)


# =========================================================
# 🔹 PER-WEIGHT DISCRETE OPTIMIZATION
# =========================================================
def optimize_weight(
    i, W_q_flat, s, e, m,
    fp_format,
    Z_block, R, alpha_q,
    X_mat, W_mat
):
    row = i // W_mat.shape[1]
    col = i % W_mat.shape[1]

    best_loss = float("inf")
    best_e = e[i].item()
    best_m = m[i].item()

    for e_cand in fp_format.exponents:
        for m_cand in fp_format.mantissas:

            w_new = s[i] * (2.0 ** e_cand) * (1.0 + m_cand)
            delta = w_new - W_q_flat[i]

            delta_Z = delta * X_mat[col]

            Z_block_new = Z_block.clone()
            Z_block_new[row] += delta_Z

            loss = ((R - alpha_q * Z_block_new) ** 2).sum()

            if loss < best_loss:
                best_loss = loss
                best_e = e_cand
                best_m = m_cand

    return best_e, best_m



def generate_layer_blocks_fixed(layer_weight, layer_input, block_size=64, layer_type='linear'):
    """
    Generate X_block and Y_block for a layer using a fixed number of weights per block.
    
    Args:
        layer_weight: torch.Tensor
            - Linear: [out_features, in_features]
            - Conv2d: [C_out, C_in, kH, kW]
        layer_input: torch.Tensor
            - Linear: [batch, in_features]
            - Conv2d: [batch, C_in, H, W]
        block_size: int, number of weights per block
        layer_type: 'linear' or 'conv'
    
    Returns:
        blocks: list of dicts
            Each dict contains:
                - 'X_block': input activations for the block
                - 'Y_block': output activations for the block
                - 'weight_indices': (start_idx, end_idx) in flattened weight tensor
    """
    blocks = []

    if layer_type == 'linear':
        B, in_features = layer_input.shape
        out_features, _ = layer_weight.shape
        W_flat = layer_weight.flatten()  # [out_features * in_features]
        y_full = layer_input @ layer_weight.T  # [B, out_features]

        for start in range(0, W_flat.numel(), block_size):
            end = min(start + block_size, W_flat.numel())

            # Compute which output/input indices this block corresponds to
            flat_indices = torch.arange(start, end)
            out_indices = (flat_indices // in_features).unique()
            in_indices = (flat_indices % in_features).unique()

            X_block = layer_input[:, in_indices]  # [B, len(in_indices)]
            Y_block = y_full[:, out_indices]      # [B, len(out_indices)]

            blocks.append({
                'X_block': X_block,
                'Y_block': Y_block,
                'weight_indices': (start, end)
            })

    elif layer_type == 'conv':
        C_out, C_in, kH, kW = layer_weight.shape
        B, _, H, W = layer_input.shape
        W_flat = layer_weight.flatten()  # [C_out * C_in * kH * kW]
        y_full = F.conv2d(layer_input, layer_weight, bias=None, stride=1, padding=0)  # [B, C_out, H_out, W_out]

        # Unfold input for convolution
        X_unf = F.unfold(layer_input, kernel_size=(kH, kW))  # [B, C_in*kH*kW, L]
        L = X_unf.shape[-1]  # number of sliding positions

        for start in range(0, W_flat.numel(), block_size):
            end = min(start + block_size, W_flat.numel())

            # Compute which C_out, C_in, kH, kW indices this block corresponds to
            flat_indices = torch.arange(start, end)
            out_c = (flat_indices // (C_in * kH * kW)).unique()
            in_c = ((flat_indices % (C_in * kH * kW)) // (kH * kW)).unique()

            # Slice input channels in unfolded input
            idx_start = in_c[0] * kH * kW
            idx_end = (in_c[-1] + 1) * kH * kW
            X_block = X_unf[:, idx_start:idx_end, :].permute(0, 2, 1).reshape(B*L, -1)

            # Slice corresponding outputs
            Y_block = y_full[:, out_c, :, :].permute(0, 2, 3, 1).reshape(B*L, -1)

            blocks.append({
                'X_block': X_block,
                'Y_block': Y_block,
                'weight_indices': (start, end)
            })

    else:
        raise ValueError(f"Unsupported layer_type {layer_type}")

    return blocks


def optimize_block_activation(w_block, X_block, Y_block,
                              alpha_init=None, iters=300, lr=1e-2,
                              exponent_bits=2, mantissa_bits=1, device='cuda'):
    """
    Optimize a block of weights to minimize activation-space perturbation:
        loss = || X_block @ w_hat - Y_block ||^2
    """
    N = w_block.numel()
    Y_block = Y_block.detach()
    w_block = w_block.to(device)
    X_block = X_block.to(device)
    Y_block = Y_block.to(device)

    # initialize alpha
    if alpha_init is None:
        alpha_b = w_block.abs().max().detach().clone().requires_grad_(True)
    else:
        alpha_b = torch.tensor(alpha_init, device=device).detach().clone().requires_grad_(True)
    # print("alpha requires_grad:", alpha_b.requires_grad)
    # print("alpha is leaf:", alpha_b.is_leaf)
    # exponent setup
    e_range = 2 ** exponent_bits
    bias = 2**(exponent_bits-1) - 1
    log_abs = torch.log2(w_block.abs() + 1e-8)
    e_opt = torch.clamp(torch.floor(log_abs).long() + bias, 0, e_range-1)
    m_opt = torch.zeros_like(w_block, dtype=torch.long, device=device)

    optimizer = torch.optim.Adam([alpha_b], lr=lr)

    for it in range(iters):
        optimizer.zero_grad()

        scale = (2.0 ** (e_opt.float() - bias))   # constant

        coarse = alpha_b * scale                  # MUST depend on alpha_b

        if mantissa_bits > 0:
            fine = coarse * m_opt.float() / (2**mantissa_bits - 1)
        else:
            fine = 0.0

        w_hat = coarse + fine
        # print("alpha:", alpha_b.requires_grad)
        # print("coarse:", coarse.requires_grad)
        # print("w_hat:", w_hat.requires_grad)
        # print(X_block.requires_grad)
        # print(Y_block.requires_grad)
        # Activation-space loss using Y_block
        pred = (X_block@w_hat).view(-1)
        target = Y_block.view(-1)
        loss = F.mse_loss(pred, target)
        # print(alpha_b.requires_grad)   # MUST be True
        # print(w_hat.requires_grad)     # MUST be True
        # print(loss.requires_grad)      # MUST be True
        loss.backward()
        optimizer.step()

        # greedy mantissa update from residual
        with torch.no_grad():
            resid = X_block.T @ (X_block @ (w_block - coarse.detach()))
            m_opt = torch.clamp(torch.round(resid / coarse * (2**mantissa_bits - 1)), 0, 2**mantissa_bits-1).long()

            # optional exponent update every 50 steps
            if it % 50 == 0:
                resid_abs = torch.abs(X_block.T @ (X_block @ w_block))
                e_opt = torch.clamp(torch.round(torch.log2(resid_abs / alpha_b)).long() + bias, 0, e_range-1)

    return alpha_b.detach(), e_opt.detach(), m_opt.detach()

# =========================================================
# 🔹 CORE RECONSTRUCTION
# =========================================================
def reconstruct_layer_fp(
    layer,
    X,
    Y,
    block_size=32,
    iters=2,
    weight_format=FPFormat(2, 1, 0),
    e_bits_scale=8,
    m_bits_scale=0,
):
    device = X.device

    # ---- Matrix form ----
    W_mat, X_mat = extract_matrix(layer, X)

    if Y.dim() > 2:
        Y_mat = Y.view(Y.shape[0], -1)
    else:
        Y_mat = Y

    W_flat = W_mat.view(-1)
    numel = W_flat.numel()

    # ---- Init FP params ----
    s = torch.sign(W_flat)
    s[s == 0] = 1

    e = torch.round(torch.log2(torch.abs(W_flat) + 1e-12))
    e = torch.clamp(e,
        weight_format.exponents.min(),
        weight_format.exponents.max()
    )

    m = torch.zeros_like(W_flat)

    W_q_flat = build_weight(s, e, m)
    Z_full = W_q_flat.view_as(W_mat) @ X_mat

    # =====================================================
    # 🔹 BLOCK LOOP
    # =====================================================
    for start in range(0, numel, block_size):
        end = min(start + block_size, numel)
        idx = torch.arange(start, end, device=device)

        for _ in range(iters):

            # ---- Block contribution ----
            W_tmp = torch.zeros_like(W_flat)
            W_tmp[idx] = W_q_flat[idx]
            Z_block = W_tmp.view_as(W_mat) @ X_mat
            if isinstance(layer, nn.Conv2d):
                B, C_out, H, W = Y.shape
                Y_mat = Y.permute(1, 0, 2, 3).reshape(C_out, -1)
            else:
                Y_mat = Y
            Z_rest = Z_full - Z_block
            R = Y_mat - Z_rest

            # ---- Scale solve ----
            denom = (Z_block * Z_block).sum().clamp(min=1e-8)
            alpha = (R * Z_block).sum() / denom
            alpha_q = quantize_scale(alpha, e_bits_scale, m_bits_scale)

            # ---- Weight optimization ----
            for i in idx:

                e_new, m_new = optimize_weight(
                    i, W_q_flat, s, e, m,
                    weight_format,
                    Z_block, R, alpha_q,
                    X_mat, W_mat
                )

                e[i] = e_new
                m[i] = m_new

                w_new = build_weight(s[i], e[i], m[i])
                delta = w_new - W_q_flat[i]

                row = i // W_mat.shape[1]
                col = i % W_mat.shape[1]

                delta_Z = delta * X_mat[col]
                Z_block[row] += delta_Z

                W_q_flat[i] = w_new

            # ---- Update global output ----
            Z_full = Z_rest + Z_block

        # ----  SCALE FOLDING ----
        W_q_flat[idx] *= alpha_q

    return W_q_flat.view_as(W_mat)





def reconstruct_layer_fp_block(layer, layer_input, block_size=64,
                               iters=300, lr=1e-2,
                               exponent_bits=2, mantissa_bits=1,
                               geometry=False, device='cuda'):
    """
    Block-wise FP4 reconstruction of a layer in activation space.
    
    Args:
        layer: nn.Linear or nn.Conv2d
        layer_input: torch.Tensor, input to the layer
        block_size: int, number of weights per block
        iters: int, iterations for block optimization
        lr: float, learning rate for alpha
        exponent_bits: int, number of exponent bits
        mantissa_bits: int, number of mantissa bits
        geometry: bool, if True, mantissa is binary
        device: 'cuda' or 'cpu'
        
    Returns:
        alpha_blocks: list of alpha_b for each block
        e_blocks: list of exponent tensors per block
        m_blocks: list of mantissa tensors per block
        weight_indices: list of (start, end) in flattened weight tensor
    """
    layer = layer.to(device)
    layer_input = layer_input.to(device)

    # Extract flattened weight matrix and unfolded input matrix
    W_mat, X_mat = extract_matrix(layer, layer_input)
    W_mat = W_mat.to(device)
    X_mat = X_mat.to(device)

    # Compute full target outputs in activation space
    # Y_full = X_mat @ W_mat.T
    out_features, in_features = W_mat.shape
    # Flatten weight matrix for block-wise slicing
    W_flat = W_mat.flatten()
    total_weights = W_flat.numel()

    alpha_blocks, e_blocks, m_blocks, weight_indices = [], [], [], []

    # Loop over fixed-size blocks
    for o in range(out_features):

        # block over input dimension
        for start in range(0, in_features, block_size):
            end = min(start + block_size, in_features)

            in_indices = torch.arange(start, end, device=device)

            w_block = W_mat[o, in_indices]          # [k]
            X_block = X_mat[:, in_indices]          # [N, k]

            # ✅ correct target
            Y_block = X_block @ w_block             # [N]

            alpha_b, e_opt, m_opt = optimize_block_activation(
                w_block, X_block, Y_block,
                iters=iters, lr=lr,
                exponent_bits=exponent_bits,
                mantissa_bits=mantissa_bits,
                device=device
            )

            if geometry:
                m_opt = torch.round(m_opt.float())

            alpha_blocks.append(alpha_b)
            e_blocks.append(e_opt)
            m_blocks.append(m_opt)

            # ✅ NEW indexing format
            weight_indices.append((o, in_indices))

    return alpha_blocks, e_blocks, m_blocks, weight_indices


# =========================================================
# 🔹 QUANTIZED LINEAR LAYER
# =========================================================
class QuantLinearFP(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        block_size=32,
        weight_format=FPFormat(2, 1),
        e_bits_scale=8,
        m_bits_scale=0,
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.block_size = block_size
        self.weight_format = weight_format
        self.e_bits_scale = e_bits_scale
        self.m_bits_scale = m_bits_scale

        self.register_buffer("weight_q", None)

    def calibrate(self, X):
        # with torch.no_grad():
        print("Reconstructing", self.__class__.__name__)

        # Run block-wise FP4 reconstruction
        alpha_blocks, e_blocks, m_blocks, weight_indices = reconstruct_layer_fp_block(
            self.conv,
            X,
            block_size=self.block_size,
            exponent_bits=self.e_bits_scale,
            mantissa_bits=self.m_bits_scale
        )

        # Reconstruct the full FP4 weight tensor from blocks
        W_q = torch.zeros_like(self.conv.weight, device=self.conv.weight.device)
        bias = 2**(self.e_bits_scale - 1) - 1

        for alpha_b, e_b, m_b, (o, in_indices) in zip(alpha_blocks, e_blocks, m_blocks, weight_indices):

            scale = (2.0 ** (e_b.float() - bias))

            coarse = alpha_b * scale

            if self.m_bits_scale > 0:
                fine = coarse * m_b.float() / (2**self.m_bits_scale - 1)
            else:
                fine = 0.0
            W_q[o, in_indices] = coarse + fine

        # Reshape back to the original conv weight shape
        self.weight_q = W_q

    def forward(self, x):
        if self.weight_q is None:
            return self.linear(x)
        return F.linear(x, self.weight_q, self.linear.bias)


# =========================================================
# 🔹 QUANTIZED CONV LAYER
# =========================================================
class QuantConv2dFP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        block_size=32,
        weight_format=FPFormat(2, 1),
        e_bits_scale=8,
        m_bits_scale=0,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        self.block_size = block_size
        self.weight_format = weight_format
        self.e_bits_scale = e_bits_scale
        self.m_bits_scale = m_bits_scale

        self.register_buffer("weight_q", None)

    def calibrate(self, X):
        # with torch.no_grad():
        print("Reconstructing", self.__class__.__name__)

        # Run block-wise FP4 reconstruction
        alpha_blocks, e_blocks, m_blocks, weight_indices = reconstruct_layer_fp_block(
            self.conv,
            X,
            block_size=self.block_size,
            exponent_bits=self.e_bits_scale,
            mantissa_bits=self.m_bits_scale
        )

        # Reconstruct the full FP4 weight tensor from blocks
        W_mat, _ = extract_matrix(self.conv, X)
        W_q = torch.zeros_like(W_mat)
        bias = 2**(self.e_bits_scale - 1) - 1

        for alpha_b, e_b, m_b, (o, in_indices) in zip(alpha_blocks, e_blocks, m_blocks, weight_indices):

            scale = (2.0 ** (e_b.float() - bias))

            coarse = alpha_b * scale

            if self.m_bits_scale > 0:
                fine = coarse * m_b.float() / (2**self.m_bits_scale - 1)
            else:
                fine = 0.0
            W_q[o, in_indices] = coarse + fine

        W_q = W_q.view_as(self.conv.weight)
        self.weight_q = W_q

    def forward(self, x):
        if self.weight_q is None:
            return self.conv(x)

        return F.conv2d(
            x,
            self.weight_q,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
        )
    


def get_reconstruct_layers(model,name):

    blocks = []

        # last_dense_name, last_dense_module = dense_layers[-1]
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            blocks.append((name, module))
            print(f"--> Found Linear layer: {name}")
            continue
        elif isinstance(module, nn.Conv2d):
            ## Check if the name contains the same name as the last block to avoid double counting conv layers that are already part of a block
            blocks.append((name, module))
            print(f"--> Found Conv2d layer: {name}")
            continue

    return blocks


def replace_layers(
    model,
    block_size,
    weight_format,
    e_bits_scale,
    m_bits_scale,
):
    for name, module in model.named_children():

        if isinstance(module, nn.Linear):
            new_layer = QuantLinearFP(
                module.in_features,
                module.out_features,
                bias=(module.bias is not None),
                block_size=block_size,
                weight_format=weight_format,
                e_bits_scale=e_bits_scale,
                m_bits_scale=m_bits_scale,
            )
            new_layer.linear.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new_layer.linear.bias.data.copy_(module.bias.data)

            setattr(model, name, new_layer)

        elif isinstance(module, nn.Conv2d):
            new_layer = QuantConv2dFP(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                bias=(module.bias is not None),
                block_size=block_size,
                weight_format=weight_format,
                e_bits_scale=e_bits_scale,
                m_bits_scale=m_bits_scale,
            )
            new_layer.conv.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new_layer.conv.bias.data.copy_(module.bias.data)

            setattr(model, name, new_layer)

        else:
            replace_layers(
                module,
                block_size,
                weight_format,
                e_bits_scale,
                m_bits_scale,
            )

    return model


def calibrate_model(model, data_loader, device="cuda"):

    model.eval()
    model.to(device)

    inputs, _ = next(iter(data_loader))
    inputs = inputs.to(device)

    hooks = []

    def make_hook(module):
        def hook(module, input, output):
            if hasattr(module, "calibrate") and module.weight_q is None:
                module.calibrate(input[0])
        return hook

    # register hooks
    for m in model.modules():
        if hasattr(m, "calibrate"):
            hooks.append(m.register_forward_hook(make_hook(m)))

    # run forward pass
    model(inputs)

    # remove hooks
    for h in hooks:
        h.remove()

    return model

def quantize_model_fp(
    model,
    data_loader,
    block_size=32,
    weight_exp = 2, 
    weight_mant = 1,
    e_bits_scale=8,
    m_bits_scale=0,
    device="cuda",
):
    # -----------------------------------------------------
    # 1. Replace layers
    # -----------------------------------------------------
    weight_format = FPFormat(weight_exp, weight_mant)
    print("Building Model layers")
    model = replace_layers(
        model,
        block_size,
        weight_format,
        e_bits_scale,
        m_bits_scale,
    )

    # -----------------------------------------------------
    # 2. Calibrate sequentially
    # -----------------------------------------------------
    model = calibrate_model(model, data_loader, device)

    return model
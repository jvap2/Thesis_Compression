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

        self.exponents = self._build_exponents()
        self.mantissas = self._build_mantissas()

    def _build_exponents(self):
        vals = torch.arange(2 ** self.e_bits)
        return vals - self.bias

    def _build_mantissas(self):
        if self.m_bits == 0:
            return torch.tensor([0.0])
        levels = 2 ** self.m_bits
        return torch.arange(levels) / levels


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

        unfold = F.unfold(
            X,
            kernel_size=layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride,
        )

        # [C*k*k, B*L]
        X_mat = unfold.permute(1, 0, 2).reshape(unfold.shape[1], -1)
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

            delta_Z = delta * X_mat[col].unsqueeze(0)

            Z_block_new = Z_block.clone()
            Z_block_new[row] += delta_Z

            loss = ((R - alpha_q * Z_block_new) ** 2).sum()

            if loss < best_loss:
                best_loss = loss
                best_e = e_cand
                best_m = m_cand

    return best_e, best_m


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

                delta_Z = delta * X_mat[col].unsqueeze(0)
                Z_block[row] += delta_Z

                W_q_flat[i] = w_new

            # ---- Update global output ----
            Z_full = Z_rest + Z_block

        # ----  SCALE FOLDING ----
        W_q_flat[idx] *= alpha_q

    return W_q_flat.view_as(W_mat)


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
        with torch.no_grad():
            Y = self.linear(X)

            W_q = reconstruct_layer_fp(
                self.linear,
                X,
                Y,
                block_size=self.block_size,
                weight_format=self.weight_format,
                e_bits_scale=self.e_bits_scale,
                m_bits_scale=self.m_bits_scale,
            )

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
        with torch.no_grad():
            Y = self.conv(X)

            W_q = reconstruct_layer_fp(
                self.conv,
                X,
                Y,
                block_size=self.block_size,
                weight_format=self.weight_format,
                e_bits_scale=self.e_bits_scale,
                m_bits_scale=self.m_bits_scale,
            )

            self.weight_q = W_q.view_as(self.conv.weight)

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

    # take one batch (or a few)
    inputs, _ = next(iter(data_loader))
    inputs = inputs.to(device)

    x = inputs

    for module in model.modules():

        if hasattr(module, "calibrate"):
            # 🔹 calibrate using current activations
            module.calibrate(x)

            # 🔹 forward through quantized layer
            x = module(x)

        else:
            x = module(x)

    return model

def quantize_model_fp(
    model,
    data_loader,
    block_size=32,
    weight_format=FPFormat(2, 1),
    e_bits_scale=8,
    m_bits_scale=0,
    device="cuda",
):
    # -----------------------------------------------------
    # 1. Replace layers
    # -----------------------------------------------------
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_layer_config

class FlexRoundFP(nn.Module):

    def __init__(self, weight, S_min=0.5, S_max=2.0):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.theta = nn.Parameter(torch.zeros_like(weight, device=self.device))
        self.theta.to(device=self.device)
        self.S_min = S_min
        self.S_max = S_max
        nonz_w = torch.abs(weight[weight!=0])
        min_val = torch.min(nonz_w)
        max_val = torch.max(nonz_w)
        log_min = torch.log2(min_val)
        log_max = torch.log2(max_val)
        self.S_max = ((log_max - log_min) / 8).item()
        self.S_min = self.S_max / 2
        
    def forward(self, w):
        w = w.detach()
        sign = torch.sign(w)
        abs_w = torch.abs(w) + 1e-8

        log_w = torch.log2(abs_w)

        S = torch.exp(self.theta).to(device=self.device)
        S = torch.clamp(S, self.S_min, self.S_max)
        log_scaled = log_w / S

        # exp_q = (torch.round(log_scaled)-log_scaled).detach()+log_scaled
        delta = log_scaled - torch.round(log_scaled)
        exp_q = log_scaled + (delta.tanh()).detach() - delta.detach()
        log_q = exp_q * S

        w_q = sign * torch.pow(2.0, log_q)

        return w_q
    


class FlexRoundFPChannel(nn.Module):

    def __init__(self, weight, channel_dim=0, S_min=0.5, S_max=2.0):
        super().__init__()

        self.channel_dim = channel_dim
        self.S_min = S_min
        self.S_max = S_max
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        channels = weight.shape[channel_dim]

        # log parameterization
        self.theta = nn.Parameter(torch.zeros(channels, device=self.device))
        self.theta.to(device=self.device)
        nonz_w = torch.abs(weight[weight!=0])
        min_val = torch.min(nonz_w)
        max_val = torch.max(nonz_w)
        log_min = torch.log2(min_val)
        log_max = torch.log2(max_val)
        self.S_max = ((log_max - log_min) / 8).item()
        self.S_min = self.S_max / 2
        S_init = (self.S_min + self.S_max) / 2
        self.theta.data = torch.log(torch.ones_like(self.theta) * S_init)
    def forward(self, w):

        theta = self.theta
        w = w.detach()
        sign = torch.sign(w)
        abs_w = torch.abs(w) + 1e-8

        log_w = torch.log2(abs_w)

        S = torch.exp(theta)
        S = torch.clamp(S, self.S_min, self.S_max)

        # ✅ FIX: reshape for broadcasting
        shape = [1] * w.dim()
        shape[self.channel_dim] = -1
        S = S.view(shape)

        log_scaled = log_w / S

        # exp_q = (torch.round(log_scaled) - log_scaled).detach() + log_scaled
        delta = log_scaled - torch.round(log_scaled)
        exp_q = log_scaled + (delta.tanh()).detach() - delta.detach()
        exp_q = torch.clamp(exp_q, -7, 7)

        log_q = exp_q * S

        return sign * torch.pow(2.0, log_q)


import torch
import torch.nn as nn

# class FlexFPQuantizer(nn.Module):
#     def __init__(self, weight, exp_bits=3, man_bits=0, channel_wise=True, channel_dim=0):
#         super().__init__()
#         self.weight = weight
#         self.exp_bits = exp_bits
#         self.man_bits = man_bits
#         self.channel_wise = channel_wise
#         self.channel_dim = channel_dim
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # --- Exponent range ---
#         self.exp_min = -(2 ** (exp_bits - 1))
#         self.exp_max = (2 ** (exp_bits - 1)) - 1

#         # --- Mantissa levels ---
#         self.man_levels = 2 ** man_bits
#         self.delta = 1 / self.man_levels if man_bits > 0 else 0

#         # --- Compute bounds a, b for theta sigmoid ---
#         self.a = self.exp_min - self.delta
#         self.b = self.exp_max + self.delta

#         # --- Compute init scale per channel or global ---
#         with torch.no_grad():
#             if channel_wise:
#                 reduce_dims = list(range(weight.dim()))
#                 reduce_dims.pop(channel_dim)
#                 max_val = weight.abs().amax(dim=reduce_dims)
#             else:
#                 max_val = weight.abs().max()

#             max_val = torch.clamp(max_val, min=1e-8)
#             init_log_s = torch.log2(max_val)

#         # Map initial scale to sigmoid space
#         p = (init_log_s - self.a) / (self.b - self.a)
#         p = torch.clamp(p, 1e-6, 1 - 1e-6)  # avoid infinities
#         theta_init = torch.log(p / (1 - p))  # logit

#         self.theta = nn.Parameter(theta_init.to(self.device), requires_grad=True)

#     def get_log_scale(self):
#         # Sigmoid-bounded log-scale
#         log_s = self.a + (self.b - self.a) * torch.sigmoid(self.theta)
#         if self.channel_wise:
#             shape = [1] * self.weight.dim()
#             shape[self.channel_dim] = -1
#             log_s = log_s.view(shape)
#         return log_s

#     def forward(self, w):
#         w = w.detach()
#         sign = torch.sign(w)
#         abs_w = torch.abs(w) + 1e-8

#         log_w = torch.log2(abs_w)
#         log_s = self.get_log_scale()

#         # --- exponent quantization ---
#         log_scaled = log_w - log_s
#         delta = log_scaled - torch.round(log_scaled)
#         exp_q = log_scaled + (delta.tanh()).detach() - delta.detach()
#         exp_q = torch.clamp(exp_q, self.exp_min, self.exp_max)

#         # --- mantissa quantization ---
#         if self.man_bits > 0:
#             exp_floor = torch.floor(log_scaled)
#             frac = log_scaled - exp_floor
#             frac_q = torch.round(frac * self.man_levels) / self.man_levels
#             frac_q = (frac_q - frac).detach() + frac
#             log_q = exp_q + frac_q + log_s
#         else:
#             log_q = exp_q + log_s

#         return sign * torch.pow(2.0, log_q)

import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexFPQuantizer(nn.Module):
    """
    FP4-style quantizer:
    - Hard exponent quantization
    - Optional mantissa STE
    - Learnable per-channel output scale
    """
    def __init__(self, weight, exp_bits=3, man_bits=0, channel_wise=True, channel_dim=0):
        super().__init__()
        self.weight = weight
        self.exp_bits = exp_bits
        self.man_bits = man_bits
        self.channel_wise = channel_wise
        self.channel_dim = channel_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Exponent range
        self.exp_min = -(2 ** (exp_bits - 1))
        self.exp_max = (2 ** (exp_bits - 1)) - 1

        # Mantissa levels
        self.man_levels = 2 ** man_bits

        # --- Learnable output scale per channel ---
        if channel_wise:
            shape = [1] * weight.dim()
            shape[channel_dim] = weight.size(channel_dim)
            self.scale = nn.Parameter(torch.ones(shape, device=self.device))
        else:
            self.scale = nn.Parameter(torch.ones(1))

    def forward(self, w):
        w_detached = w.detach()

        # --- log2 weight ---
        log_w = torch.log2(w_detached.abs() + 1e-8)
        sign = torch.sign(w_detached)

        # --- exponent hard quantization ---
        exp = torch.clamp(torch.floor(log_w), self.exp_min, self.exp_max)

        # --- mantissa (optional STE) ---
        if self.man_bits > 0:
            frac = log_w - exp
            frac_q = torch.round(frac * self.man_levels) / self.man_levels
            # STE
            frac_q = (frac_q - frac).detach() + frac
        else:
            frac_q = 0.0

        log_q = exp + frac_q
        w_q = sign * (2 ** log_q)

        # --- apply learnable scale ---
        w_q = w_q * self.scale

        return w_q


# -----------------------------
# QuantConv2dFP
# -----------------------------
class QuantConv2dFP(nn.Conv2d):
    def __init__(self, conv, exp_bits=3, man_bits=0):
        super().__init__(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            bias=(conv.bias is not None)
        )

        self.weight = nn.Parameter(conv.weight.detach().clone(), requires_grad=False)
        if conv.bias is not None:
            self.bias = nn.Parameter(conv.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

        # Pruning mask
        self.mask = (conv.weight != 0).float()

        # FP quantizer
        self.flex = FlexFPQuantizer(
            self.weight,
            exp_bits=exp_bits,
            man_bits=man_bits,
            channel_wise=True,
            channel_dim=0
        )

    def forward(self, x):
        w_q = self.flex(self.weight)
        w_q = w_q * self.mask
        return F.conv2d(x, w_q, self.bias, self.stride, self.padding)


# -----------------------------
# QuantLinearFP
# -----------------------------
class QuantLinearFP(nn.Linear):
    def __init__(self, linear, exp_bits=3, man_bits=0, channel_wise=True):
        super().__init__(
            linear.in_features,
            linear.out_features,
            bias=(linear.bias is not None)
        )

        self.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

        self.mask = (linear.weight != 0).float()

        self.flex = FlexFPQuantizer(
            self.weight,
            exp_bits=exp_bits,
            man_bits=man_bits,
            channel_wise=channel_wise,
            channel_dim=0
        )

    def forward(self, x):
        w_q = self.flex(self.weight)
        w_q = w_q * self.mask
        return F.linear(x, w_q, self.bias)
    

class FPScaledLinear(nn.Module):
    def __init__(self, weight, E=2, M=1, block_size=32):
        super().__init__()

        self.weight = weight.clone().detach()
        self.weight.requires_grad = False

        self.out_features, self.in_features = weight.shape
        self.block_size = block_size

        assert self.in_features % block_size == 0
        self.num_blocks = self.in_features // block_size

        # Generate codebook
        self.codebook = generate_fp_codebook(E, M, device=weight.device)

        # Scales
        self.s_global = nn.Parameter(torch.ones(1))
        self.s_block = nn.Parameter(torch.ones(self.num_blocks))

    def forward(self, x):
        W = self.weight

        W_blocks = W.view(self.out_features, self.num_blocks, self.block_size)

        s = self.s_global * self.s_block.view(1, -1, 1)

        W_scaled = W_blocks / (s + 1e-8)

        W_q = fp_quantize(W_scaled, self.codebook)

        W_hat = W_q * s

        W_hat = W_hat.view(self.out_features, self.in_features)

        return F.linear(x, W_hat)
    



class FPScaledConv2d(nn.Module):
    def __init__(self, conv_layer, E=2, M=1, block_size=32):
        super().__init__()

        # Copy weights
        self.weight = conv_layer.weight.detach().clone()
        self.weight.requires_grad = False

        self.bias = None
        if conv_layer.bias is not None:
            self.bias = conv_layer.bias.detach().clone()

        # Conv params
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups

        # Shape
        self.out_channels, self.in_channels, self.kH, self.kW = self.weight.shape

        # Flatten per output channel
        self.flat_dim = self.in_channels * self.kH * self.kW

        assert self.flat_dim % block_size == 0, "flat_dim must be divisible by block_size"

        self.block_size = block_size
        self.num_blocks = self.flat_dim // block_size

        # Codebook
        self.codebook = generate_fp_codebook(E, M, device=self.weight.device)

        # ===== SCALE PARAMETERS =====

        # Per-channel scale: [O, 1]
        self.s_channel = nn.Parameter(torch.ones(self.out_channels, 1))

        # Per-channel, per-block scale: [O, B]
        self.s_block = nn.Parameter(torch.ones(self.out_channels, self.num_blocks))

    def forward(self, x):
        W = self.weight

        # Flatten: [O, flat_dim]
        W_flat = W.view(self.out_channels, self.flat_dim)

        # Block reshape: [O, B, block_size]
        W_blocks = W_flat.view(self.out_channels, self.num_blocks, self.block_size)

        # Construct full scale: [O, B, 1]
        s = self.s_channel.view(self.out_channels, 1, 1) * \
            self.s_block.view(self.out_channels, self.num_blocks, 1)

        # Scale down
        W_scaled = W_blocks / (s + 1e-8)

        # Quantize
        W_q = fp_quantize(W_scaled, self.codebook)

        # Dequantize
        W_hat_blocks = W_q * s

        # Reshape back
        W_hat_flat = W_hat_blocks.view(self.out_channels, self.flat_dim)
        W_hat = W_hat_flat.view(
            self.out_channels,
            self.in_channels,
            self.kH,
            self.kW
        )

        return F.conv2d(
            x,
            W_hat,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )


def convert_to_fp_quant(module, module_config=None):

    # Handle root modules first
    if isinstance(module, nn.Conv2d):
        return QuantConv2dFP(module)

    if isinstance(module, nn.Linear):
        return QuantLinearFP(module)

    # Otherwise recurse through children
    for name, child in module.named_children():

        new_child = convert_to_fp_quant(child)

        if new_child is not child:
            setattr(module, name, new_child)

    return module


def convert_to_fp_quant_flex(module, layer_name="", is_first=False, is_last=False):

    if isinstance(module, nn.Conv2d):

        cfg = get_layer_config(layer_name, is_first, is_last)

        return QuantConv2dFP(module, **cfg)

    if isinstance(module, nn.Linear):

        cfg = get_layer_config(layer_name, is_first, is_last)

        return QuantLinearFP(module, **cfg)

    for name, child in module.named_children():

        new_child = convert_to_fp_quant_flex(
            child,
            layer_name=name,
            is_first=is_first,
            is_last=is_last
        )

        if new_child is not child:
            setattr(module, name, new_child)

    return module


def solve_channelwise_scale(y_fp, y_q, layer_type):
    eps = 1e-8

    if layer_type == "conv":
        dims = (0, 2, 3)
        keepdim = True
    elif layer_type == "linear":
        dims = (0,)
        keepdim = True
    else:
        raise ValueError

    num = (y_fp * y_q).sum(dim=dims, keepdim=keepdim)
    den = (y_q * y_q).sum(dim=dims, keepdim=keepdim) + eps

    return num / den


@torch.no_grad()
def apply_layerwise_scales(block_fp, block_q, inputs):
    """
    Compute and assign channel-wise output scale for EACH quantized layer.
    """

    device = inputs.device

    # Forward hooks storage
    fp_outputs = {}
    q_outputs = {}

    def get_hook(storage, name):
        def hook(module, inp, out):
            storage[name] = out.detach()
        return hook

    hooks = []

    # Register hooks for matching layers
    for (name_fp, m_fp), (name_q, m_q) in zip(block_fp.named_modules(), block_q.named_modules()):

        if isinstance(m_q, (QuantConv2dFP, QuantLinearFP)):

            hooks.append(m_fp.register_forward_hook(get_hook(fp_outputs, name_fp)))
            hooks.append(m_q.register_forward_hook(get_hook(q_outputs, name_q)))

    # Run forward
    _ = block_fp(inputs)
    _ = block_q(inputs)

    # Compute scales per layer
    for (name_fp, m_fp), (name_q, m_q) in zip(block_fp.named_modules(), block_q.named_modules()):

        if isinstance(m_q, QuantConv2dFP):

            y_fp = fp_outputs[name_fp]
            y_q = q_outputs[name_q]

            scale = solve_channelwise_scale(y_fp, y_q, "conv")

            m_q.output_scale = scale

        elif isinstance(m_q, QuantLinearFP):

            y_fp = fp_outputs[name_fp]
            y_q = q_outputs[name_q]

            scale = solve_channelwise_scale(y_fp, y_q, "linear")

            m_q.output_scale = scale

    # Clean hooks
    for h in hooks:
        h.remove()



class FPQuantizerSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, codebook):
        x_flat = x.view(-1, 1)

        dist = torch.abs(x_flat - codebook.view(1, -1))
        idx = torch.argmin(dist, dim=1)

        q = codebook[idx].view_as(x)
        return q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def fp_quantize(x, codebook):
    return FPQuantizerSTE.apply(x, codebook)




def generate_fp_codebook(E, M, device="cuda", include_subnormals=False):
    """
    Generate FP(e, m) codebook (normalized only by default)

    E: exponent bits
    M: mantissa bits
    """

    bias = 2 ** (E - 1) - 1

    codebook = []

    # Normalized exponents
    e_min = 1 - bias
    e_max = (2 ** E - 2) - bias

    for e in range(e_min, e_max + 1):
        for m in range(2 ** M):
            val = (1.0 + m / (2 ** M)) * (2 ** e)
            codebook.append(val)
            codebook.append(-val)

    # Zero
    codebook.append(0.0)

    codebook = torch.tensor(codebook, device=device)

    # Remove duplicates + sort
    codebook = torch.unique(codebook)
    codebook, _ = torch.sort(codebook)

    return codebook
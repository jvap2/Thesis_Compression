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


class FlexFPQuantizer(nn.Module):

    def __init__(self, weight, exp_bits=3, man_bits=0, channel_wise=True, channel_dim=0):
        super().__init__()

        self.exp_bits = exp_bits
        self.man_bits = man_bits
        self.channel_wise = channel_wise
        self.channel_dim = channel_dim

        if channel_wise:
            channels = weight.shape[channel_dim]
            self.theta = nn.Parameter(torch.zeros(channels))
        else:
            self.theta = nn.Parameter(torch.tensor(0.0))

        # exponent range
        self.exp_min = -(2 ** (exp_bits - 1))
        self.exp_max = (2 ** (exp_bits - 1)) - 1

        # mantissa levels
        self.man_levels = 2 ** man_bits
    def forward(self, w):

        w = w.detach()

        sign = torch.sign(w)
        abs_w = torch.abs(w) + 1e-8

        log_w = torch.log2(abs_w)

        # scaling
        S = torch.exp(self.theta)

        if self.channel_wise:
            shape = [1] * w.dim()
            shape[self.channel_dim] = -1
            S = S.view(shape)

        log_scaled = log_w / S

        # --- EXPONENT QUANTIZATION ---
        exp_q = (torch.round(log_scaled) - log_scaled).detach() + log_scaled
        exp_q = torch.clamp(exp_q, self.exp_min, self.exp_max)

        # --- MANTISSA QUANTIZATION ---
        if self.man_bits > 0:

            frac = log_scaled - torch.floor(log_scaled)

            frac_q = torch.round(frac * self.man_levels) / self.man_levels
            frac_q = (frac_q - frac).detach() + frac

            log_q = (torch.floor(exp_q) + frac_q) * S

        else:
            log_q = exp_q * S

        return sign * torch.pow(2.0, log_q)



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
            self.bias.data = conv.bias.data.clone()

        self.mask = (conv.weight != 0).float()

        self.flex = FlexFPQuantizer(
            self.weight,
            exp_bits=exp_bits,
            man_bits=man_bits,
            channel_wise=True
        )

    def forward(self, x):

        w_q = self.flex(self.weight)
        w_q = w_q * self.mask
        return F.conv2d(
            x,
            w_q,
            self.bias,
            self.stride,
            self.padding
        )
    



class QuantLinearFP(nn.Linear):

    def __init__(self, linear, exp_bits=3, man_bits=0, channel_wise=True):

        super().__init__(
            linear.in_features,
            linear.out_features,
            bias=(linear.bias is not None)
        )

        # --- Freeze weights (CRITICAL) ---
        self.weight = nn.Parameter(
            linear.weight.detach().clone(),
            requires_grad=False
        )

        if linear.bias is not None:
            self.bias = nn.Parameter(
                linear.bias.detach().clone(),
                requires_grad=False
            )
        else:
            self.bias = None

        # --- Pruning mask ---
        self.mask = (linear.weight != 0).float()

        # --- FP quantizer ---
        # For Linear: channel-wise = per output neuron → dim=0
        self.quantizer = FlexFPQuantizer(
            self.weight,
            exp_bits=exp_bits,
            man_bits=man_bits,
            channel_wise=channel_wise,
            channel_dim=0
        )

    def forward(self, x):

        # Quantize weights
        w_q = self.quantizer(self.weight)

        # Apply pruning mask (NO in-place ops)
        w_q = w_q * self.mask

        return F.linear(x, w_q, self.bias)

def convert_to_fp_quant(module):

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

        new_child = convert_to_fp_quant(
            child,
            layer_name=name,
            is_first=is_first,
            is_last=is_last
        )

        if new_child is not child:
            setattr(module, name, new_child)

    return module
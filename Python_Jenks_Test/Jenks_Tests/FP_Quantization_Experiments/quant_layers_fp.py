import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexRoundFP(nn.Module):

    def __init__(self, weight, S_min=0.5, S_max=2.0):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.theta = nn.Parameter(torch.zeros_like(weight), requires_grad=True)
        self.theta.to(device=self.device)
        self.S_min = S_min
        self.S_max = S_max

    def forward(self, w):

        sign = torch.sign(w)
        abs_w = torch.abs(w) + 1e-8

        log_w = torch.log2(abs_w)

        S = torch.exp(self.theta).to(device=self.device)
        S = torch.clamp(S, self.S_min, self.S_max)
        log_scaled = log_w / S

        exp_q = torch.round(log_scaled)

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
        self.theta = nn.Parameter(torch.zeros(channels), requires_grad=True)
        self.theta.to(device=self.device)

    def forward(self, w):

        sign = torch.sign(w)
        abs_w = torch.abs(w) + 1e-8

        log_w = torch.log2(abs_w)

        S = torch.exp(self.theta).to(device=self.device)
        S = torch.clamp(S, self.S_min, self.S_max)

        # reshape for broadcasting
        shape = [1] * w.dim()
        shape[self.channel_dim] = -1
        S = S.view(shape)
        log_scaled = log_w / S

        exp_q = torch.round(log_scaled)

        log_q = exp_q * S

        w_q = sign * torch.pow(2.0, log_q)

        return w_q


class QuantConv2dFP(nn.Conv2d):

    def __init__(self, conv):

        super().__init__(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            bias=(conv.bias is not None)
        )

        self.weight.data = conv.weight.data.clone()

        self.flex = FlexRoundFPChannel(self.weight)

    def forward(self, x):

        w_q = self.flex(self.weight)

        return F.conv2d(
            x,
            w_q,
            self.bias,
            self.stride,
            self.padding
        )
    



class QuantLinearFP(nn.Linear):

    def __init__(self, linear):

        super().__init__(
            linear.in_features,
            linear.out_features,
            bias=(linear.bias is not None)
        )

        # copy weights
        self.weight.data = linear.weight.data.clone()

        if linear.bias is not None:
            self.bias.data = linear.bias.data.clone()

        # flexround module
        self.quantizer = FlexRoundFPChannel(self.weight)

    def forward(self, x):

        w_q = self.quantizer(self.weight)

        return F.linear(x, w_q, self.bias)

def convert_to_fp_quant(module):

    for name, child in module.named_children():

        if isinstance(child, nn.Conv2d):
            setattr(module, name, QuantConv2dFP(child))

        elif isinstance(child, nn.Linear):
            setattr(module, name, QuantLinearFP(child))

        else:
            convert_to_fp_quant(child)
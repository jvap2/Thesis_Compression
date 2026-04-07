import torch
import torch.nn as nn
import copy




    

def compute_fp4_range_pruned(weight, percentile=0.01):

    w = weight.abs().view(-1)

    w = w[w > 0]   # remove pruned weights

    lower = torch.quantile(w, percentile)

    e_min = torch.floor(torch.log2(lower))

    return int(e_min)

def get_layer_config(name, is_first, is_last):

    if is_first or is_last:
        return dict(exp_bits=4, man_bits=3)  # higher precision

    if "conv" in name:
        return dict(exp_bits=2, man_bits=1)

    return dict(exp_bits=2, man_bits=1)




def solve_output_scale_channelwise(block_fp, block_q, inputs, m, n_grid=20):

    with torch.no_grad():

        target = block_fp(inputs).detach()

        w = m.weight
        C = w.shape[m.channel_dim]

        log_w = torch.log2(w.abs() + 1e-8)

        reduce_dims = list(range(w.dim()))
        reduce_dims.pop(m.channel_dim)

        log_min = log_w.amin(dim=reduce_dims)
        log_max = log_w.amax(dim=reduce_dims)

        search_min = log_min - m.exp_max
        search_max = log_max - m.exp_min

        best_log_s = m.theta.data.clone()

        for c in range(C):

            best_err = float("inf")

            for alpha in torch.linspace(0, 1, n_grid, device=w.device):

                log_s_c = search_min[c] * (1 - alpha) + search_max[c] * alpha

                old = m.theta.data[c].clone()
                m.theta.data[c] = log_s_c

                pred = block_q(inputs)
                err = (pred - target).pow(2).mean().item()

                if err < best_err:
                    best_err = err
                    best_log_s[c] = log_s_c

                m.theta.data[c] = old  # restore

        m.theta.data = best_log_s

def hadamard_transform(x):
    """
    In-place Fast Walsh-Hadamard Transform
    x: (k,) tensor, k must be power of 2
    """
    n = x.shape[0]
    h = 1
    y = x.clone()

    while h < n:
        for i in range(0, n, h * 2):
            a = y[i:i+h]
            b = y[i+h:i+2*h]
            y[i:i+h] = a + b
            y[i+h:i+2*h] = a - b
        h *= 2

    return y / torch.sqrt(torch.tensor(n, device=x.device))

def hadamard_safe(x):
    k = x.numel()
    next_pow2 = 1 << (k - 1).bit_length()

    if next_pow2 == k:
        return hadamard_transform(x), None

    # pad
    pad = next_pow2 - k
    x_pad = torch.cat([x, torch.zeros(pad, device=x.device)])

    y = hadamard_transform(x_pad)
    return y[:k], pad

def hadamard_inverse(x, pad):
    k = x.numel()

    if pad is None:
        return hadamard_transform(x)

    next_pow2 = k + pad
    x_pad = torch.cat([x, torch.zeros(pad, device=x.device)])
    y = hadamard_transform(x_pad)
    return y[:k]
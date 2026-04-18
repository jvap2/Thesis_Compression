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

import math
import torch.nn.functional as F


# def fast_hadamard_transform(x):
#     """
#     Standard Walsh-Hadamard Transform. 
#     x: (B, N) where N is a power of 2.
#     """
#     n = x.shape[-1]
#     if n == 1:
#         return x
#     x = x.view(-1, 2, n // 2)
#     x = torch.stack([x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]], dim=1)
#     return fast_hadamard_transform(x).view(-1, n) / torch.sqrt(torch.tensor(2.0))

def apply_hadamard_to_layer(W, mask):
    # Padding to power of 2 for FHT
    orig_shape = W.shape
    M = W.shape[-1]
    next_pow2 = 2**math.ceil(math.log2(M))
    
    W_padded = F.pad(W, (0, next_pow2 - M))
    # Transform
    W_had = fast_hadamard_transform(W_padded)
    return W_had, next_pow2

def apply_blockwise_hadamard(W, block_size=32):
    """
    W: (N, M) - flattened layer weights
    block_size: power of 2
    """
    N, M = W.shape
    # Pad M to be a multiple of block_size if necessary
    pad_len = (block_size - (M % block_size)) % block_size
    if pad_len > 0:
        W = F.pad(W, (0, pad_len))
    
    new_M = W.shape[-1]
    # Reshape to treat blocks as the last dimension
    # (N * (new_M // block_size), block_size)
    W_blocks = W.view(-1, block_size)
    
    # Transform each block
    W_had_blocks = fast_hadamard_transform(W_blocks)
    
    # Return to (N, new_M)
    return W_had_blocks.view(N, new_M), pad_len

def invert_blockwise_hadamard(W_had, block_size, pad_len):
    N, M = W_had.shape
    W_blocks = W_had.view(-1, block_size)
    # FHT is its own inverse (with normalization)
    W_inv_blocks = fast_hadamard_transform(W_blocks)
    W_inv = W_inv_blocks.view(N, M)
    
    if pad_len > 0:
        W_inv = W_inv[:, :-pad_len]
    return W_inv




def fast_hadamard_transform(x):
    """
    Vectorized Fast Walsh-Hadamard Transform.
    x: tensor of shape (batch, n) where n is a power of 2.
    """
    n = x.shape[-1]
    if n == 1:
        return x
    
    # Check if power of 2
    if (n & (n - 1)) != 0:
        raise ValueError(f"Input size {n} must be a power of 2 for FHT.")

    # Reshape and compute butterfly operations iteratively
    h = 1
    while h < n:
        x = x.view(-1, n // (2 * h), 2, h)
        # Apply the Hadamard butterfly: [1, 1; 1, -1]
        x_left = x[:, :, 0, :] + x[:, :, 1, :]
        x_right = x[:, :, 0, :] - x[:, :, 1, :]
        x = torch.cat((x_left.unsqueeze(2), x_right.unsqueeze(2)), dim=2)
        h *= 2
    
    # Return flattened and normalized (optional: / sqrt(n))
    # For smoothing weights, usually you normalize by sqrt(n)
    return x.view(-1, n) / math.sqrt(n)

def hadamard_transform_wrapper(w_block):
    """
    Handles padding for non-power-of-2 inputs (like your size 27)
    and applies the FHT.
    """
    device = w_block.device
    original_shape = w_block.shape
    n = original_shape[-1]
    
    # 1. Calculate the next power of 2
    next_pow2 = 2**int(math.ceil(math.log2(n)))
    
    # 2. Pad if necessary
    if n != next_pow2:
        padding_size = next_pow2 - n
        # Pad the last dimension with zeros
        w_padded = F.pad(w_block, (0, padding_size))
    else:
        w_padded = w_block
        
    # 3. Reshape for FHT (ensure batch dim)
    input_tensor = w_padded.view(-1, next_pow2)
    
    # 4. Apply Transform
    w_had_padded = fast_hadamard_transform(input_tensor)
    
    # 5. Crop back to original size and reshape back to original dimensions
    # Note: Information is spread, so cropping back is common in block-wise smoothing
    w_had = w_had_padded[:, :n]
    return w_had.view(original_shape)

def inverse_hadamard_transform_wrapper(w_had_block):
    """
    Inverse logic to correctly recover spatial weights from Hadamard components.
    """
    n = w_had_block.shape[-1]
    next_pow2 = 2**int(math.ceil(math.log2(n)))
    
    # 1. Re-pad to the power-of-2 used in the forward pass
    if n != next_pow2:
        padding_size = next_pow2 - n
        w_padded = torch.nn.functional.pad(w_had_block, (0, padding_size))
    else:
        w_padded = w_had_block
    
    # 2. Apply FHT
    # Note: fast_hadamard_transform divides by sqrt(N). 
    # To invert a normalized FHT, we just run it again! 
    # Because (H/√N) * (H/√N) = (H*H)/N = I
    w_spatial_padded = fast_hadamard_transform(w_padded.view(-1, next_pow2))
    
    # 3. Crop back to original size
    return w_spatial_padded[:, :n].view(w_had_block.shape)
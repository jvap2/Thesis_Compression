import torch
import torch.nn as nn
import torch.nn.functional as F
import math




def collect_block_activation_scales(model, calib_loader, device,
                                     block_size, num_batches=4):
    """
    Collect per-input-block max activation magnitude for each linear layer.
    Returns dict: layer_name -> tensor of shape [n_blocks] where
    n_blocks = ceil(in_features / block_size).
    Unlike collect_activation_scales which is per-channel, this gives
    one scale per block of block_size input channels.
    """
    act_scales = {}
    hooks = []

    def make_hook(name, in_features):
        n_blocks = math.ceil(in_features / block_size)

        def hook(module, inp, out):
            x = inp[0].detach().float()
            x_flat = x.reshape(-1, in_features)   # (N_tokens, in_features)
            N, K = x_flat.shape

            # Pad to multiple of block_size
            pad = (block_size - K % block_size) % block_size
            x_pad = F.pad(x_flat, (0, pad))
            x_blocks = x_pad.view(N, n_blocks, block_size)

            # Per-block max abs across all tokens
            block_max = x_blocks.abs().amax(dim=(0, 2))  # (n_blocks,)

            if name not in act_scales:
                act_scales[name] = block_max
            else:
                act_scales[name] = torch.maximum(act_scales[name], block_max)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(
                module.register_forward_hook(
                    make_hook(name, module.in_features)
                )
            )

    model.eval()
    batches_run = 0
    with torch.no_grad():
        for batch in calib_loader:
            if batch is None:
                continue
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if x is None:
                continue
            model(x.to(device))
            batches_run += 1
            if batches_run >= num_batches:
                break

    for h in hooks:
        h.remove()

    return act_scales


def compute_block_smooth_scale(W, act_block_max, block_size, alpha=0.5):
    """
    Compute per-block smoothing scale.

    W:             (out_features, in_features)
    act_block_max: (n_blocks,) — max abs activation per input block
    block_size:    int
    alpha:         float in [0, 1] — 0 = all on weights, 1 = all on activations

    Returns scale: (n_blocks,)
    """
    out_features, in_features = W.shape
    n_blocks = math.ceil(in_features / block_size)

    pad = (block_size - in_features % block_size) % block_size
    W_pad = F.pad(W, (0, pad))                           # (out, in_pad)
    W_blocks = W_pad.view(out_features, n_blocks, block_size)

    # Per-block max abs weight across output channels and block elements
    w_block_max = W_blocks.abs().amax(dim=(0, 2))        # (n_blocks,)

    act_block_max = act_block_max.to(W.device).clamp(min=1e-8)
    w_block_max   = w_block_max.clamp(min=1e-8)

    scale = (act_block_max.pow(alpha) / w_block_max.pow(1 - alpha))
    return scale                                          # (n_blocks,)


def apply_block_smooth_layer(layer, act_block_max, block_size, alpha=0.5):
    """
    Apply block-wise smoothing to a single linear layer in-place.
    Divides activations by scale (absorbed into preceding norm/bias),
    multiplies weights by scale to preserve the product W x.

    Returns scale: (n_blocks,) for absorbing into the preceding LayerNorm.
    """
    W = layer.weight.data                                # (out, in)
    scale = compute_block_smooth_scale(W, act_block_max, block_size, alpha)

    out_features, in_features = W.shape
    n_blocks = math.ceil(in_features / block_size)

    pad = (block_size - in_features % block_size) % block_size
    W_pad = F.pad(W, (0, pad))
    W_blocks = W_pad.view(out_features, n_blocks, block_size)

    # Multiply each weight block by its scale
    W_blocks = W_blocks * scale.view(1, n_blocks, 1)

    # Write back, removing padding
    layer.weight.data = W_blocks.view(out_features, -1)[:, :in_features]

    return scale                                         # (n_blocks,)


def absorb_block_scale_into_layernorm(ln, scale, block_size):
    """
    Absorb the inverse block smooth scale into a LayerNorm.
    LayerNorm has per-element weight/bias of shape (hidden,).
    We expand the per-block scale to per-element and divide.

    scale: (n_blocks,)
    """
    hidden = ln.weight.shape[0]
    n_blocks = scale.shape[0]

    # Expand scale from (n_blocks,) to (hidden,)
    # Each block covers block_size elements
    scale_expanded = scale.repeat_interleave(block_size)[:hidden].to(ln.weight.device)

    ln.weight.data /= scale_expanded
    if ln.bias is not None:
        ln.bias.data /= scale_expanded


def apply_block_smoothquant_opt(model, calib_loader, device,
                                 block_size, alpha=0.5, num_batches=4):
    """
    Block-wise SmoothQuant for OPT.
    Smoothing granularity matches the FP4 quantization block size exactly,
    so the activation quantizer sees a distribution that is already
    block-normalized to a consistent scale.

    alpha: 0.5 is balanced, 0.7-0.85 works better at very low bitwidths.
    """
    print(f"Collecting block activation scales "
          f"(block_size={block_size}, alpha={alpha})...")
    act_scales = collect_block_activation_scales(
        model, calib_loader, device, block_size, num_batches
    )

    for i, block in enumerate(model.model.decoder.layers):

        # ── Attention: LayerNorm -> q, k, v projections ──────────────────
        q_name = f"model.decoder.layers.{i}.self_attn.q_proj"
        if q_name in act_scales:
            ln  = block.self_attn_layer_norm
            q   = block.self_attn.q_proj
            k   = block.self_attn.k_proj
            v   = block.self_attn.v_proj

            scale = apply_block_smooth_layer(q, act_scales[q_name],
                                             block_size, alpha)

            # k and v see the same input — apply same scale to their weights
            out_f = k.weight.shape[0]
            in_f  = k.weight.shape[1]
            n_blocks = math.ceil(in_f / block_size)
            pad = (block_size - in_f % block_size) % block_size

            for proj in [k, v]:
                W_pad = F.pad(proj.weight.data, (0, pad))
                W_b   = W_pad.view(out_f, n_blocks, block_size)
                W_b   = W_b * scale.view(1, n_blocks, 1)
                proj.weight.data = W_b.view(out_f, -1)[:, :in_f]

            # Absorb inverse scale into LayerNorm
            absorb_block_scale_into_layernorm(ln, scale, block_size)

            print(f"  Layer {i} attn: scale min={scale.min():.3f} "
                  f"max={scale.max():.3f} mean={scale.mean():.3f}")

        # ── MLP: final LayerNorm -> fc1 ───────────────────────────────────
        fc1_name = f"model.decoder.layers.{i}.fc1"
        if fc1_name in act_scales:
            ln2 = block.final_layer_norm
            fc1 = block.fc1

            scale2 = apply_block_smooth_layer(fc1, act_scales[fc1_name],
                                              block_size, alpha)
            absorb_block_scale_into_layernorm(ln2, scale2, block_size)

            print(f"  Layer {i} mlp:  scale min={scale2.min():.3f} "
                  f"max={scale2.max():.3f} mean={scale2.mean():.3f}")

    return model
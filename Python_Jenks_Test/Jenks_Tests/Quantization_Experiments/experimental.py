import torch
import torch.nn as nn
import math


###############################################
# OPERATOR SKETCH LOSS
###############################################

def operator_sketch_loss(block_fp, block_q, input_shape, probes=32):

    device = next(block_fp.parameters()).device
    loss = 0
    count = 0

    for _ in range(probes):

        x = torch.randn(input_shape, device=device)

        with torch.no_grad():
            y_fp = block_fp(x)

        y_q = block_q(x)

        loss += torch.mean((y_q - y_fp) ** 2)
        count += 1

    return loss / count


###############################################
# FOURIER PROBES
###############################################

def generate_fourier_probe(shape, freq_x, freq_y):

    B, C, H, W = shape

    xs = torch.arange(W).float()
    ys = torch.arange(H).float()

    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")

    probe = torch.sin(
        2 * math.pi * (
            freq_x * grid_x / W +
            freq_y * grid_y / H
        )
    )

    probe = probe.unsqueeze(0).unsqueeze(0)
    probe = probe.repeat(B, C, 1, 1)

    return probe


def fourier_probe_loss(block_fp, block_q, input_shape, freqs=None):

    if freqs is None:
        freqs = [
            (0,0),
            (1,0),
            (0,1),
            (2,0),
            (0,2),
            (1,1)
        ]

    device = next(block_fp.parameters()).device
    loss = 0
    count = 0
    for (name_fp, layer_fp), (name_q, layer_q) in zip(
        block_fp.named_modules(), block_q.named_modules()
    ):

        if isinstance(layer_fp, (nn.Conv2d)) and isinstance(layer_q, (nn.Conv2d)):
            for fx, fy in freqs:

                probe = generate_fourier_probe(input_shape, fx, fy).to(device)

                with torch.no_grad():
                    y_fp = layer_fp(probe)

                y_q = layer_q(probe)

                loss += torch.mean((y_q - y_fp) ** 2)
                count += 1
    if count == 0:
        return torch.tensor(0)
    else:
        return loss / count


###############################################
# GRAM OPERATOR LOSS
###############################################

def gram_operator_loss_blocks(block_fp, block_q):

    loss = 0
    count = 0

    for (name_fp, layer_fp), (name_q, layer_q) in zip(
        block_fp.named_modules(), block_q.named_modules()
    ):

        if isinstance(layer_fp, (nn.Conv2d, nn.Linear)) and isinstance(layer_q, (nn.Conv2d, nn.Linear)):

            W = layer_fp.weight
            Wq = layer_q.weight_quantizer()

            W = W.view(W.shape[0], -1)
            Wq = Wq.view(Wq.shape[0], -1)

            G = W @ W.T
            Gq = Wq @ Wq.T

            loss += torch.mean((G - Gq) ** 2)
            count += 1

    return loss / max(count,1)


def geometry_preservation_loss(
    block_fp,
    block_q,
    input_shape,
    sketch_weight=1.0,
    fourier_weight=0.5,
    gram_weight=0.5
):

    loss = 0

    loss += sketch_weight * operator_sketch_loss(
        block_fp, block_q, input_shape
    )

    loss += fourier_weight * fourier_probe_loss(
        block_fp, block_q, input_shape
    )

    loss += gram_weight * gram_operator_loss_blocks(
        block_fp, block_q
    )

    return loss
import torch
import torch.nn as nn
from Quantization_Experiments.brecq import get_reconstruct_blocks, cache_block_input, apply_bias_correction, replace_block
from .quant_layers_fp import convert_to_fp_quant, convert_to_fp_quant_flex, FlexFPQuantizer, apply_layerwise_scales
from .utils import solve_output_scale_channelwise
import copy
import torch.nn.functional as F 
import os




def reconstruct_block_fp(block_fp, block_q, inputs, iters=2000, lr=1e-3):

    block_fp.eval()
    for p in block_fp.parameters():
        p.requires_grad = False

    inputs = inputs.detach().clone()

    with torch.no_grad():
        target = block_fp(inputs).detach()

    params = []
    for m in block_q.modules():
        if isinstance(m, FlexFPQuantizer):
            params.append(m.theta)

    optimizer = torch.optim.Adam(params, lr=lr)

    for i in range(iters):

        optimizer.zero_grad()

        pred = block_q(inputs)

        loss = F.mse_loss(pred, target)
        if i%100==0:
            print("Loss: ", loss.item())
        loss.backward()

        optimizer.step()



def brecq_quantize_exp_fp(model, calibration_loader, name, bitwidth=4, geometry=False, batch_size=1024, config=None):
    """
    FP4 quantization with per-channel output scale.
    Uses FlexFPQuantizer with:
    - Hard exponent quantization
    - Optional mantissa STE
    - Per-channel learnable scale
    """

    if config is None:
        config = {
            "first": (4, 3),
            "last": (4, 3),
            "default": (3, 0),
            "conv": (3, 1),
        }

    model.eval()
    device = next(model.parameters()).device
    print("Device:", device)

    # --- get reconstruct blocks ---
    blocks = get_reconstruct_blocks(model, name)
    last_name = blocks[-1][0]
    first_name = blocks[0][0]
    print("Blocks have been reconstructed")

    for block_name, block in blocks:

        # --- cache inputs for calibration ---
        inputs = cache_block_input(
            model,
            block,
            calibration_loader,
            device=device,
            num_batches=2
        )

        # --- deepcopy block and move to device ---
        block_q = copy.deepcopy(block)
        block_q.to(device)

        # --- convert to FP4 wrapper ---
        block_q = convert_to_fp_quant_flex(
            block_q,
            block_name,
            is_first=(block_name == first_name),
            is_last=(block_name == last_name)
        )

        # --- apply quantizer to weights (no optimization) ---
        for name, module in block_q.named_modules():
            if hasattr(module, "flex"):
                # Compute FP4 quantized weights
                w_q = module.flex(module.weight)
                # Replace weight in module
                module.weight.data = w_q
        # apply_layerwise_scales(block, block_q, inputs)
        # --- optional: bias correction ---
        apply_bias_correction(block, block_q, inputs)

        # --- replace block in model ---
        replace_block(model, block, block_q)

    return model



def reconstruct_block_fp_fp4(block_fp, block_q, inputs, quant_modules, iters=2000, lr=1e-4):

    block_fp.eval()
    for p in block_fp.parameters():
        p.requires_grad = False

    inputs = inputs.detach()

    with torch.no_grad():
        target = block_fp(inputs).detach()

    # Only optimize rounding (NOT scale)
    params = []
    for m in quant_modules:
        if hasattr(m, "delta"):
            m.delta = None  # will be populated in forward

    # (optional) if you later add rounding params, collect here

    optimizer = torch.optim.Adam(
        [p for p in block_q.parameters() if p.requires_grad],
        lr=lr
    )

    for i in range(iters):

        optimizer.zero_grad()

        pred = block_q(inputs)

        # --- reconstruction loss ---
        loss = F.mse_loss(pred, target)

        # --- optional rounding regularizer ---
        round_loss = 0
        for m in quant_modules:
            if hasattr(m, "delta") and m.delta is not None:
                round_loss += torch.mean(torch.abs(m.delta))

        loss = loss + 1e-3 * round_loss

        if i % 100 == 0:
            print(f"Iter {i} | Loss: {loss.item():.6f}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(block_q.parameters(), 1.0)
        optimizer.step()

        # if i % 200 == 0:
        #     for m in quant_modules:
        #         solve_output_scale_channelwise(block_fp, block_q, inputs, m)



def diagnostic_file(block):
    filename="ResNet32_CIFAR100/4_bit/diag.txt"
    quant_modules = [
            m for m in block.modules()
            if isinstance(m, FlexFPQuantizer)
        ]
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a") as f:
        for module in quant_modules:
            with torch.no_grad():
                w = module.weight
                w_q = module(w)

                print("W stats:", w.mean().item(), w.std().item(), file=f)
                print("W_q stats:", w_q.mean().item(), w_q.std().item(), file=f)
                print("Unique values:", w_q.unique().numel(), file=f)
                print("theta:", module.theta.min().item(), module.theta.max().item(), file=f)
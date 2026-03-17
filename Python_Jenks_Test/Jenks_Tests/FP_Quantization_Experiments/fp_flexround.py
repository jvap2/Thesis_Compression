import torch
import torch.nn as nn
from Quantization_Experiments.brecq import get_reconstruct_blocks, cache_block_input, apply_bias_correction, replace_block
from .quant_layers_fp import convert_to_fp_quant
import copy
import torch.nn.functional as F 




def reconstruct_block_fp(block_fp, block_q, inputs, iters=2000, lr=1e-3):

    block_fp.eval()
    for p in block_fp.parameters():
        p.requires_grad = False

    inputs = inputs.detach().clone()

    with torch.no_grad():
        target = block_fp(inputs).detach()

    optimizer = torch.optim.Adam(
        [p for n, p in block_q.named_parameters() if "theta" in n],
        lr=lr
    )

    for i in range(iters):

        optimizer.zero_grad()

        pred = block_q(inputs)

        loss = F.mse_loss(pred, target)
        if i%100==0:
            print("Loss: ", loss.item())
        loss.backward()

        optimizer.step()



def brecq_quantize_exp_fp(model, calibration_loader, name, bitwidth, geometry=False, batch_size=1024):

    iters = 2000

    model.eval()
    device = next(model.parameters()).device
    print(device)

    blocks = get_reconstruct_blocks(model, name)
    last_name = blocks[-1][0]
    print("Blocks have been reconstructed")

    for block_name, block in blocks:

        inputs = cache_block_input(
            model,
            block,
            calibration_loader,
            device=device,
            num_batches=2
        )
        block_q = copy.deepcopy(block)
        block_q.to(device)
        block_q = convert_to_fp_quant(block_q)
        for name, module in block_q.named_modules():
            print(name, type(module))
        reconstruct_block_fp(
            block,
            block_q,
            inputs,
            iters=iters,
        )
        apply_bias_correction(block, block_q, inputs)

        replace_block(model, block, block_q)

    return model
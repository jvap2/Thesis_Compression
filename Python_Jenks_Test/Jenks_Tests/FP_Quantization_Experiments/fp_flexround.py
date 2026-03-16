import torch
import torch.nn as nn
from Quantization_Experiments.brecq import get_reconstruct_blocks, cache_block_input, apply_bias_correction, replace_block
from .quant_layers_fp import convert_to_fp_quant
import copy




def reconstruct_block_fp(block_fp,
                         block_q,
                         inputs,
                         iters=2000,
                         lr=1e-3):

    optimizer = torch.optim.Adam(block_q.parameters(), lr=lr)

    for i in range(iters):

        optimizer.zero_grad()

        with torch.no_grad():
            out_fp = block_fp(inputs)

        out_q = block_q(inputs)

        loss = ((out_q - out_fp) ** 2).mean()
        if i%100==0:
            print("Loss: ", loss.item())
        loss.backward()

        optimizer.step()

    return block_q



def brecq_quantize_exp_fp(model, calibration_loader, name, bitwidth, geometry=False, batch_size=1024):

    iters = 5000

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
        convert_to_fp_quant(block_q)


        reconstruct_block_fp(
            block,
            block_q,
            inputs,
            iters=iters,
        )

        apply_bias_correction(block, block_q, inputs)

        replace_block(model, block, block_q)

    return model
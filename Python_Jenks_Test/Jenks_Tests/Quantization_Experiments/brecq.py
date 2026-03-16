import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import cache_block_inputs,geometry_preserve_original_weights,nc_alignment_loss,etf_weight_loss_v2,simplex_etf_gram
import copy
from densenet import DenseBlock, TransitionBlock, BottleneckBlock, BasicBlock
from resnet import BasicBlock as ResNetBasicBlock
from .experimental import gram_operator_loss_blocks,fourier_probe_loss, operator_sketch_loss




class UniformAffineQuantizer(nn.Module):
    def __init__(self, bitwidth=8, per_channel=True, symmetric=True):
        super().__init__()
        self.bitwidth = bitwidth
        self.per_channel = per_channel
        self.symmetric = symmetric
        
        self.qmin = -2 ** (bitwidth - 1)
        self.qmax = 2 ** (bitwidth - 1) - 1
        
        self.delta = None
        self.zero_point = None

    def init_from_weight(self, W):
        if self.per_channel:
            if W.ndim == 4:
                max_val = W.abs().amax(dim=(1,2,3), keepdim=True)
            elif W.ndim ==2:
                max_val = W.abs().amax(dim=1, keepdim=True)
            else:
                raise ValueError("Bad")
        else:
            max_val = W.abs().max()
            
        self.delta = max_val / self.qmax
        self.zero_mask = (max_val==0)
        print(self.qmax)
        self.zero_point = 0
        self.delta=torch.clamp(self.delta,min=1e-8)

    def forward(self, W):
        delta = self.delta
        print(f"delta={delta}")
        W_q = torch.round(W / delta)

        W_q = torch.clamp(W_q, self.qmin, self.qmax)

        W_q = W_q * delta

        # Force zero channels back to zero
        if self.per_channel:
            W_q[self.zero_mask.expand_as(W_q)] = 0

        return W_q
        

class AdaRoundQuantizer(nn.Module):
    def __init__(self, weight, quantizer):
        super().__init__()
        self.quantizer = quantizer
        self.quantizer.init_from_weight(weight)
        self.device = weight.device
        print(f"self.quantizer.delta={self.quantizer.delta}")
        W = weight/self.quantizer.delta
        W_floor = torch.floor(W).to(device=self.device)
        frac = W - W_floor

        # avoid exact 0 or 1
        eps = 1e-6
        frac = torch.clamp(frac, eps, 1 - eps)

        alpha = torch.log(frac / (1 - frac))

        self.alpha = nn.Parameter(alpha)
        self.mask = torch.ones_like(weight.data, device= self.device)
        self.mask[weight==0]=0
        self.register_buffer("W_floor", W_floor)

    def forward(self):
        delta = self.quantizer.delta
        s = torch.sigmoid(self.alpha).to(device=self.device)
        s = torch.clamp(s*1.2-.1,0,1)
        # print(s.device)
        # print(self.W_floor.device)
        # print(self.mask.device)
        W_q = self.mask*(self.W_floor + s)
        W_q = torch.clamp(W_q,
                          self.quantizer.qmin,
                          self.quantizer.qmax)
        return W_q * delta
    

def block_reconstruction_loss(block_fp, block_q, inputs):
    with torch.no_grad():
        target = block_fp(inputs)

    output = block_q(inputs)

    return F.mse_loss(output, target)



def spectral_distortion_block(block_fp, block_q):
    """
    Compute spectral distortion between a floating point block
    and its quantized counterpart.

    block_fp : original block
    block_q  : quantized block (with AdaRound weights)
    """

    total_error = 0.0
    n_layers = 0

    for (name_fp, mod_fp), (name_q, mod_q) in zip(
        block_fp.named_modules(), block_q.named_modules()
    ):

        if isinstance(mod_fp, nn.Conv2d) and isinstance(mod_q, nn.Conv2d) and hasattr(mod_q, "weight_quantizer"):

            W_fp = mod_fp.weight
            W_q = mod_q.weight_quantizer()

            # fft_fp = torch.fft.fftshift(
            #     torch.fft.fft2(W_fp, dim=(-2,-1)),
            #     dim=(-2,-1)
            # )

            # fft_q = torch.fft.fftshift(
            #     torch.fft.fft2(W_q, dim=(-2,-1)),
            #     dim=(-2,-1)
            # )
            fft_fp = torch.fft.fftn(W_fp, dim=(-2,-1))
            fft_q  = torch.fft.fftn(W_q, dim=(-2,-1))

            power_fp = torch.abs(fft_fp)**2
            power_q = torch.abs(fft_q)**2
            # error = torch.abs(fft_q - fft_fp) ** 2
            error = torch.mean((power_fp-power_q)**2)
            # print("Spec Error from function:", error)

            # total_error += error.mean()
            total_error +=error
            n_layers += 1

    if n_layers == 0:
        # Find a module with weight_quantizer to get device
        device = None
        for module in block_q.modules():
            if hasattr(module, "weight_quantizer"):
                device = module.weight_quantizer.quantizer.delta.device
                break
        if device is None:
            device = torch.device("cpu")
        return torch.tensor(0.0, device=device)

    return total_error / n_layers

def block_reconstruction_loss_opt(target, block_q, inputs):
    output = block_q(inputs)

    return F.mse_loss(output, target)

def block_regularization_loss(block_q, lamb, beta):

    binary_reg = 0.0

    for module in block_q.modules():

        if hasattr(module, "weight_quantizer"):

            alpha = module.weight_quantizer.alpha

            Theta = torch.clamp(
                torch.sigmoid(alpha) * 1.2 - 0.1,
                0.0,
                1.0
            )

            binary_reg = binary_reg + lamb * (
                1 - torch.abs(2 * Theta - 1) ** beta
            ).sum()

    return binary_reg

import math

def etf_weight_loss_v3(module):
    # get differentiable quantized weight
    W = module.weight_quantizer()

    if W.dim() > 2:
        W = W.view(W.shape[0], -1)

    C = W.shape[0]

    W_centered = W - W.mean(dim=1, keepdim=True)
    G = W_centered @ W_centered.T

    G_target = simplex_etf_gram(C, W.device)

    alpha = (G * G_target).sum() / (G_target * G_target).sum()

    return torch.norm(G - alpha * G_target, p='fro')**2

def nc_alignment_loss_v2(W, M):
    W_q = W.weight_quantizer()
    Wn = W_q / (W_q.norm(dim=1, keepdim=True) + 1e-8)
    Mn = M / (M.norm(dim=1, keepdim=True) + 1e-8)
    return torch.norm(Wn - Mn, p='fro')**2

def geometry_preserve_original_weights_v2(fp_module, quant_module):
    total_loss = 0.0
    n_layers = 0
    
    # Handle both single layer modules and composite blocks
    for (name_fp, mod_fp), (name_q, mod_q) in zip(
        fp_module.named_modules(), quant_module.named_modules()
    ):
        if isinstance(mod_fp, (nn.Conv2d, nn.Linear)) and isinstance(mod_q, (nn.Conv2d, nn.Linear)) and hasattr(mod_q, "weight_quantizer"):
            W_fp = mod_fp.weight
            W_quant = mod_q.weight_quantizer()
            Wc_fp = W_fp - W_fp.mean(dim=1, keepdim=True)
            Wc_quant = W_quant - W_quant.mean(dim=1, keepdim=True)
            G_fp = Wc_fp @ Wc_fp.T
            G_quant = Wc_quant @ Wc_quant.T
            total_loss += torch.norm(G_fp - G_quant, p='fro')**2
            n_layers += 1
    
    if n_layers == 0:
        return torch.tensor(0.0, device=next(quant_module.parameters()).device)
    
    return total_loss / n_layers

def anneal_lambda(lamb_max, progress):
    lambda_t = lamb_max * 0.5 * (1 - math.cos(math.pi * progress))
    return lambda_t

def anneal_beta(progress,beta_start,beta_end):
    # Beta ramp
    beta_t = beta_start + (beta_end - beta_start) * \
            0.5 * (1 - math.cos(math.pi * progress))
    return beta_t
def reconstruct_block(block_fp, block_q, inputs, last_layer=False, geometry=False,
                      iters=3000, lr=1e-3):

    optimizer = torch.optim.Adam(block_q.parameters(), lr=lr)
    lamb = 5e-4
    beta = 2
    beta_final = 8
    beta_step = (beta_final-beta)/iters
    warmup = int(0.2 * iters)
    with torch.no_grad():
        target = block_fp(inputs)
    quant_modules = [
        m for m in block_q.modules()
        if hasattr(m, "weight_quantizer")
    ]
    for i in range(iters):
        optimizer.zero_grad()

        loss = block_reconstruction_loss_opt(
            target, block_q, inputs
        )
        if i < warmup:
            lambda_t = 0
            beta_t = beta
        else:
            progress = (i - warmup) / (iters - warmup)
            lambda_t = anneal_lambda(lamb,progress=progress)
            beta_t = anneal_beta(progress,beta,beta_final)
        loss_reg = block_regularization_loss(block_q,lambda_t,beta_t)
        if i%100==0:
            print("Reconstruction Loss", loss.item())
            print("Regularization Loss:", loss_reg.item())
        if geometry:
            loss_spec = spectral_distortion_block(block_fp,block_q)
            if i%100==0:
                print("Spectral Loss:",loss_spec.item())
            loss += beta_t*loss_spec
        if last_layer:
            ## Perform Geometric term
            # loss_geo = etf_weight_loss_v3(block_q)
            loss_geo = geometry_preserve_original_weights_v2(block_fp,block_q)
            # loss_geo = nc_alignment_loss_v2(block_q,inputs)
            if i%100 ==0:
                print("Geometric loss:",loss_geo.item())
            loss += beta_t*loss_geo
        loss += loss_reg
        loss.backward()
        optimizer.step()

    return block_q



def reconstruct_block_exp(
        block_fp,
        block_q,
        inputs,
        last_layer=False,
        geometry=False,
        iters=3000,
        lr=1e-3,
        batch_size=1024):

    device = next(block_fp.parameters()).device

    optimizer = torch.optim.Adam(block_q.parameters(), lr=lr)

    lamb = 5e-4
    beta = 2
    beta_final = 8

    warmup = int(0.2 * iters)

    ########################################
    # Precompute targets (BRECQ only)
    ########################################

    if not geometry:
        with torch.no_grad():
            target = block_fp(inputs)

    ########################################
    # Precompute probes (geometry only)
    ########################################

    input_shape = inputs.shape
    ########################################
    # Optimization loop
    ########################################

    for i in range(iters):

        optimizer.zero_grad()

        ####################################
        # Annealing schedule
        ####################################

        if i < warmup:

            lambda_t = 0
            beta_t = beta

        else:

            progress = (i - warmup) / (iters - warmup)

            lambda_t = anneal_lambda(lamb, progress=progress)
            beta_t = anneal_beta(progress, beta, beta_final)

        ####################################
        # Primary loss
        ####################################

        if not geometry:

            ################################
            # Standard BRECQ reconstruction
            ################################

            loss = block_reconstruction_loss_opt(
                target,
                block_q,
                inputs
            )

            if i % 100 == 0:
                print("Reconstruction Loss:", loss.item())

        else:

            ################################
            # Operator sketch loss
            ################################

            sketch_loss = 0

            sketch_loss = operator_sketch_loss(block_fp,block_q,input_shape, probes=batch_size)

            ################################
            # Fourier probe loss
            ################################
            fourier_loss = 0

            fourier_loss = fourier_probe_loss(block_fp,block_q,input_shape)

            ################################
            # Gram operator loss
            ################################

            gram_loss = gram_operator_loss_blocks(
                block_fp,
                block_q
            )

            ################################
            # Combined geometry loss
            ################################

            loss = (
                1.0 * sketch_loss +
                0.5 * fourier_loss +
                0.5 * gram_loss
            )

            if i % 100 == 0:
                print("Sketch Loss:", sketch_loss.item())
                print("Fourier Loss:", fourier_loss.item())
                print("Gram Loss:", gram_loss.item())

        ####################################
        # Last layer constraint (optional)
        ####################################

        # if last_layer:

        #     loss_geo_last = geometry_preserve_original_weights_v2(
        #         block_fp,
        #         block_q
        #     )

        #     if i % 100 == 0:
        #         print("Last Layer Geometry:", loss_geo_last.item())

        #     loss += beta_t * loss_geo_last

        ####################################
        # Binary regularization (ALWAYS)
        ####################################

        loss_reg = block_regularization_loss(
            block_q,
            lambda_t,
            beta_t
        )

        if i % 100 == 0:
            print("Regularization Loss:", loss_reg.item())

        loss += loss_reg

        ####################################
        # Optimization
        ####################################

        loss.backward()
        optimizer.step()

    return block_q

def bias_correction(conv_fp, conv_q, inputs):
    with torch.no_grad():
        out_fp = conv_fp(inputs)
        out_q = conv_q(inputs)

        correction = (out_fp - out_q).mean(dim=(0,2,3))
        conv_q.bias.data += correction

def replace_forward(module):

    if isinstance(module, nn.Conv2d):

        def new_forward(x):
            W_q = module.weight_quantizer()
            return nn.functional.conv2d(
                x,
                W_q,
                module.bias,
                module.stride,
                module.padding,
                module.dilation,
                module.groups
            )

    elif isinstance(module, nn.Linear):

        def new_forward(x):
            W_q = module.weight_quantizer()
            return nn.functional.linear(
                x,
                W_q,
                module.bias
            )

    module.forward = new_forward

def copy_block_with_quantizers(block,
                               bitwidth=8,
                               per_channel=True,
                               symmetric=True):

    block_q = copy.deepcopy(block)

    for module in block_q.modules():

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            # Create base quantizer
            base_quantizer = UniformAffineQuantizer(
                bitwidth=bitwidth,
                per_channel=per_channel,
                symmetric=symmetric
            )

            # Initialize scale from FP weight
            base_quantizer.init_from_weight(module.weight.data)

            # Create AdaRound wrapper
            ada_quant = AdaRoundQuantizer(
                module.weight.data,
                base_quantizer
            )

            # Attach quantizer to module
            module.weight_quantizer = ada_quant

            # Remove original weight parameter
            module.weight.requires_grad = False

            # Replace forward
            replace_forward(module)

    return block_q



# def apply_bias_correction(block_fp, block_q, inputs):
#     print("applying bias correction")
#     block_fp.eval()
#     block_q.eval()

#     with torch.no_grad():

#         # Forward once to ensure quantizers are active
#         out_fp = block_fp(inputs)
#         out_q = block_q(inputs)

#         # Now correct each layer individually
#         for (name,m_fp), (name_q,m_q) in zip(block_fp.named_modules(), block_q.named_modules()):
#             print(name_q, type(m_fp), type(m_q))
#             if isinstance(m_fp, (nn.Conv2d, nn.Linear)) and isinstance(m_q, (nn.Conv2d, nn.Linear)):
#                 print(f"Correcting bias for {name}")
#                 if not hasattr(m_q, "bias"):
#                     continue
#                 if m_q.bias is None:
#                     continue
#                 # Compute layer outputs only
#                 def layer_forward(module, x):
#                     return module(x)

#                 # Run through this single layer only
#                 x = inputs

#                 # For proper BRECQ you should cache layer inputs,
#                 # but this simplified version works block-wise.
#                 out_fp_layer = m_fp(x)
#                 out_q_layer = m_q(x)

#                 if isinstance(m_fp, nn.Conv2d):
#                     correction = (out_fp_layer - out_q_layer).mean(dim=(0,2,3))
#                 else:
#                     correction = (out_fp_layer - out_q_layer).mean(dim=0)

#                 m_q.bias.data += correction



import torch
import torch.nn as nn


def apply_bias_correction(block_fp, block_q, inputs):
    """
    Applies operator-level bias correction to Conv2d and Linear layers
    inside a quantized block.

    Args:
        block_fp : floating-point block
        block_q  : quantized block
        inputs   : calibration inputs (tensor)
    """

    print("applying bias correction")

    block_fp.eval()
    block_q.eval()

    with torch.no_grad():

        # Ensure quantizers are initialized/activated
        _ = block_fp(inputs)
        _ = block_q(inputs)

        # Build lookup dictionary for quantized modules
        q_modules = dict(block_q.named_modules())

        for name, m_fp in block_fp.named_children():

            # Only correct operator layers
            if not isinstance(m_fp, (nn.Conv2d, nn.Linear)):
                print(f"{name} is not Linear or Conv")
                print(f"m_fp is type {m_fp}")
                continue

            if name not in q_modules:
                print(f"{name} is not in q_modules")
                continue

            m_q = q_modules[name]

            if not isinstance(m_q, (nn.Conv2d, nn.Linear)):
                continue

            if m_q.bias is None:
                continue

            print(f"Correcting bias for {name}")

            # --------------------------------------------------
            # Compute bias correction term
            # Δb = E[W_fp x] - E[W_q x]
            # --------------------------------------------------

            # Forward through this layer only
            def forward_layer(layer, x):
                if isinstance(layer, nn.Conv2d):
                    return nn.functional.conv2d(
                        x,
                        layer.weight,
                        None,
                        stride=layer.stride,
                        padding=layer.padding,
                        dilation=layer.dilation,
                        groups=layer.groups,
                    )
                elif isinstance(layer, nn.Linear):
                    return nn.functional.linear(x, layer.weight, None)

            # Get layer input
            # We extract it by registering a forward hook

            layer_input = []

            def hook_fn(module, inp, out):
                layer_input.append(inp[0])

            handle = m_fp.register_forward_hook(hook_fn)
            _ = block_fp(inputs)
            handle.remove()

            if len(layer_input) == 0:
                continue

            x = layer_input[0]

            # Compute floating and quant outputs (without bias)
            out_fp = forward_layer(m_fp, x)
            out_q = forward_layer(m_q, x)

            # Compute mean difference over batch + spatial dims
            # Keep channel dimension
            dims = list(range(out_fp.dim()))
            dims.pop(1)  # remove channel dim

            bias_correction = (out_fp - out_q).mean(dim=dims)

            # Apply correction
            m_q.bias.data += bias_correction

            print(f"Bias updated for {name}")


def replace_block(model, old_block, new_block):

    for name, module in model.named_children():

        if module is old_block:
            setattr(model, name, new_block)
            return True

        else:
            replaced = replace_block(module, old_block, new_block)
            if replaced:
                return True

    return False


def get_reconstruct_blocks(model,name):

    blocks = []

    if 'Dense' in name or 'Res' in name:
        dense_layers = []
        for name, module in model.named_modules():
            if isinstance(module, BasicBlock) or isinstance(module, BottleneckBlock) or isinstance(module, TransitionBlock):
                dense_layers.append((name, module))

        # last_dense_name, last_dense_module = dense_layers[-1]
    for name, module in model.named_modules():

        # Skip root
        if module is model:
            continue
        # DenseLayer blocks
        if isinstance(module, BasicBlock) or isinstance(module, BottleneckBlock) or isinstance(module, TransitionBlock):
            blocks.append((name, module))
            print(f"--> Found DenseLayer block: {name}")
            continue

        # Residual blocks
        elif isinstance(module, ResNetBasicBlock):
            blocks.append((name, module))
            print(f"--> Found Residual block: {name}")
            continue

        elif isinstance(module, nn.Linear):
            blocks.append((name, module))
            print(f"--> Found Linear layer: {name}")
            continue
        elif isinstance(module, nn.Conv2d):
            ## Check if the name contains the same name as the last block to avoid double counting conv layers that are already part of a block
            if len(blocks) > 0:
                last_block_name, _ = blocks[-1]
                if last_block_name in name:
                    print(f"--> Skipping Conv2d layer {name} as it is part of block {last_block_name}")
                    continue
            else:
                blocks.append((name, module))
                print(f"--> Found Conv2d layer: {name}")
            continue

    return blocks

    

def cache_block_input(model, block, loader, device, num_batches=2):

    cached = []

    def hook(module, input, output):
        print("Hook fired for:", module)
        cached.append(input[0].detach())

    print("Registering hook on:", block)

    handle = block.register_forward_hook(hook)

    model.eval()

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(device)
            model(images)
            if i + 1 >= num_batches:
                break

    handle.remove()

    print("Cached tensors:", len(cached))

    if len(cached) == 0:
        raise RuntimeError("Hook never triggered for block")

    return torch.cat(cached, dim=0)

def brecq_quantize(model, calibration_loader,name,bitwidth, geometry=False):

    iters = 2000
    if bitwidth==2:
        if geometry ==True:
            iters=600
        else:
            iters=800
    elif bitwidth ==4:
        if geometry ==True:
            iters=1200
        else:
            iters=1600
    elif bitwidth ==6:
        if geometry ==True:
            iters=1800
        else:
            iters=2400
    elif bitwidth ==8:
        if geometry ==True:
            iters=2400
        else:
            iters=3200
    model.eval()
    device = next(model.parameters()).device
    print(device)
    blocks = get_reconstruct_blocks(model,name)
    last_name = blocks[-1][0]
    print("Blocks have been reconstructed")
    for name, block in blocks:
        inputs = cache_block_input(model, block, calibration_loader, device = device, num_batches=2)

        block_q = copy_block_with_quantizers(block, bitwidth=bitwidth)
        if geometry==False:
            reconstruct_block(block, block_q, inputs, last_layer=False,geometry=geometry, iters=iters)
        else:
            reconstruct_block(block, block_q, inputs,name==last_name, geometry=geometry, iters=iters)

        apply_bias_correction(block, block_q, inputs)

        replace_block(model, block, block_q)

    return model

def brecq_quantize_exp(model, calibration_loader, name, bitwidth, geometry=False, batch_size=1024):

    iters = 2000

    if bitwidth == 2:
        iters = 600 if geometry else 800
    elif bitwidth == 4:
        iters = 1200 if geometry else 1600
    elif bitwidth == 6:
        iters = 1800 if geometry else 2400
    elif bitwidth == 8:
        iters = 2400 if geometry else 3200

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

        block_q = copy_block_with_quantizers(block, bitwidth=bitwidth)

        if geometry:

            input_shape = inputs[0].shape

            reconstruct_block_exp(
                block,
                block_q,
                inputs,
                last_layer=(block_name == last_name),
                geometry=True,
                iters=iters,
                batch_size=batch_size
            )

        else:

            reconstruct_block_exp(
                block,
                block_q,
                inputs,
                last_layer=(block_name == last_name),
                geometry=False,
                iters=iters,
                batch_size=batch_size
            )

        apply_bias_correction(block, block_q, inputs)

        replace_block(model, block, block_q)

    return model
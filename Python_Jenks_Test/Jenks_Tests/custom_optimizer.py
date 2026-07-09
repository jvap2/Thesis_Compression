import numpy as np
from jenkspy import JenksNaturalBreaks
import torch
from torch.optim import Optimizer
from cuda_helpers import module_weights, module_bias, Conv_Mask, Linear_Mask, Bias_Mask
from statistics import mean, stdev
import time
import gc
from torchvision.transforms.v2 import CutMix, MixUp, RandomChoice

# MixUp/CutMix augmentation toggle for train_one_step_prune_HPO.
# MIXUP=False -> original behavior (no change). When True, each training batch is
# randomly CutMix'd or MixUp'd (soft labels into the loss); MIXUP_OFF_EPOCH lets us
# disable it for the final epochs so the model fine-tunes on clean labels.
MIXUP = False
MIXUP_OFF_EPOCH = 10**9


def apply_gradient_centralization(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad.data
            if grad.ndim > 1:  # apply only to weight tensors, not biases
                grad.sub_(grad.mean(dim=tuple(range(1, grad.ndim)), keepdim=True))

class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch

def train_one_step_gsm(net, data, label, optimizer, criterion, nonzero_ratio):
    pred = net(data)
    loss = criterion(pred, label)
    loss.backward()

    to_concat_g = []
    to_concat_v = []
    for name, param in net.named_parameters():
        if param.dim() in [2, 4]:
            to_concat_g.append(param.grad.data.view(-1))
            to_concat_v.append(param.data.view(-1))
    all_g = torch.cat(to_concat_g)
    all_v = torch.cat(to_concat_v)
    metric = torch.abs(all_g * all_v)
    num_params = all_v.size(0)
    nz = int(nonzero_ratio * num_params)
    top_values, _ = torch.topk(metric, nz)
    thresh = top_values[-1]

    for name, param in net.named_parameters():
        if param.dim() in [2, 4]:
            mask = (torch.abs(param.data * param.grad.data) >= thresh).type(torch.cuda.FloatTensor)
            param.grad.data = mask * param.grad.data

    optimizer.step()
    optimizer.zero_grad()
    acc, acc5 = torch_accuracy(pred, label, (1,5))
    return acc, acc5, loss

def prune_by_magnitude(model, nonzero_ratio):
    ##Prune only the weight matrices or convolutional layers, and we will do this one globally
    vals = []
    for name, param in model.named_parameters():
        if param.dim() in [2, 4]:
            vals.append(torch.abs(param.data.view(-1)))
    all_vals = torch.cat(vals)
    num_params = all_vals.size(0)
    nz = int((1-nonzero_ratio) * num_params)
    top_values, _ = torch.topk(all_vals, nz)
    thresh = top_values[-1]
    for name, param in model.named_parameters():
        if param.dim() in [2,4]:
            mask = (torch.abs(param.data) >= thresh).type(torch.cuda.FloatTensor)
            param.data = mask * param.data
    return model

class JenksSGD(Optimizer):
    def __init__(self, params, lr=5e-3, scale=5e-4, momentum=0.99):
        defaults = dict(lr=lr, scale=scale, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            scale = group['scale']
            momentum = group['momentum']

            for param in group['params']:
                if param.grad is None:
                    continue

                # Initialize velocity if not already done
                if 'velocity' not in self.state[param]:
                    self.state[param]['velocity'] = torch.zeros_like(param.data)

                velocity = self.state[param]['velocity']

                # Check if the parameter is a weight matrix or bias vector
                if len(param.shape) > 1:  # Assuming weight matrices have more than 1 dimension
                    # Custom weight update: Scale gradients before applying update
                    # print(torch.mul(param, param.grad).cpu().numpy())
                    s_W = torch.mul(param, param.grad)  # Move to CPU, convert to NumPy, and flatten
                    s_W = torch.abs(s_W)
                    unique_values = torch.unique(s_W)
                    n_classes = min(2, len(unique_values))  # Ensure n_classes is valid
                    if n_classes > 1:
                        # jnb = JenksNaturalBreaks(n_classes)
                        # jnb.fit(s_W)
                        # labels = jnb.labels_
                        # indices = np.where(labels == 1)[0]
                        # indices_ = np.where(labels == 0)[0]

                        # # Update velocity
                        velocity_flat = velocity.view(-1)
                        param_data_flat = param.data.view(-1)
                        param_grad_flat = param.grad.data.view(-1)
                        WB_cuda_flatten = s_W.flatten()
                        WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                        WB_cuda_sorted = WB_cuda_sorted.reshape(s_W.shape)

                        var = module_weights.jenks_optimization_cuda(WB_cuda_sorted)
                        var_min = var.argmin().item()

                        indices_ = WB_cuda_indices[:var_min]
                        indices = WB_cuda_indices[var_min:]

                        velocity_flat[indices] = momentum * velocity_flat[indices] + scale * param_data_flat[indices] + param_grad_flat[indices]
                        velocity_flat[indices_] = momentum * velocity_flat[indices_] + scale * param_data_flat[indices_]

                        # Update parameters
                        param_data_flat[indices] -= lr * velocity_flat[indices]
                        param_data_flat[indices_] -= lr * velocity_flat[indices_]
                        param.data = param_data_flat.view(param.data.shape)
                        self.state[param]['velocity'] = velocity_flat.view(velocity.shape)
                    else:
                        velocity = momentum * velocity + scale * param.grad
                        param.data -= lr * velocity
                else:  # Assuming bias vectors have 1 dimension
                    s_B = torch.mul(param, param.grad)  # Move to CPU, convert to NumPy, and flatten
                    s_B = torch.abs(s_B)
                    unique_values = torch.unique(s_B)
                    n_classes = min(2, len(unique_values))  # Ensure n_classes is valid
                    if n_classes > 1:
                        # jnb = JenksNaturalBreaks(n_classes)
                        # jnb.fit(s_B)
                        # labels = jnb.labels_
                        # indices = np.where(labels == 1)[0]
                        # indices_ = np.where(labels == 0)[0]
                        B_cuda_sorted, B_cuda_indices = s_B.sort()
                        var = module_bias.jenks_optimization_biases_cuda(B_cuda_sorted)
                        var_min = var.argmin().item()
                        # Print the output
                        indices_ = B_cuda_indices[:var_min]
                        indices = B_cuda_indices[var_min:]
                        # Update velocity
                        velocity_flat = velocity.view(-1)
                        param_data_flat = param.data.view(-1)
                        param_grad_flat = param.grad.data.view(-1)

                        velocity_flat[indices] = momentum * velocity_flat[indices] + scale * param_data_flat[indices] + param_grad_flat[indices]
                        velocity_flat[indices_] = momentum * velocity_flat[indices_] + scale * param_data_flat[indices_]

                        # Update parameters
                        param_data_flat[indices] -= lr * velocity_flat[indices]
                        param_data_flat[indices_] -= lr * velocity_flat[indices_]
                        param.data = param_data_flat.view(param.data.shape)
                        self.state[param]['velocity'] = velocity_flat.view(velocity.shape)
                        
                    else:
                        velocity = momentum * velocity + scale * param.grad
                        param.data -= lr * velocity
        return loss
    
class JenksSGD_Test(Optimizer):
    def __init__(self, params, warmup_epochs, lr=5e-3, scale=5e-4, momentum=0.9, nestrov=False, bias = True):
        defaults = dict(lr=lr, scale=scale, momentum=momentum)
        super().__init__(params, defaults)
        self.warmup_epochs = warmup_epochs
        self.nestrov = nestrov
        self.bias = bias
        if self.bias:
            self.name = "JenksSGD_Test"
        else:
            self.name = "JenksSGD_Test_Weights"
        ##Define agg scores which should be the same size as the parameters
        for group in self.param_groups:
            for param in group['params']:
                self.state[param]['agg_score'] = torch.zeros_like(param.data)
        for group in self.param_groups:
            for param in group['params']:
                self.state[param]['lookahead'] = torch.zeros_like(param.data, requires_grad=True)
        for group in self.param_groups:
            for param in group['params']:
                self.state[param]['prev_weights'] = torch.zeros_like(param.data, requires_grad=True)

    # @torch.no_grad()
    def step(self, epoch, closure=None):
        torch.cuda.empty_cache()
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            scale = group['scale']
            momentum = group['momentum']

            for param in group['params']:
                if param.grad is None:
                    continue

                # Initialize velocity if not already done
                if 'velocity' not in self.state[param]:
                    self.state[param]['velocity'] = torch.zeros_like(param.data)
                if 'agg_score' not in self.state[param]:
                    print("Still no agg_score")
                    self.state[param]['agg_score'] = torch.zeros_like(param.data)
                # self.state[param]['velocity'] = param.data - self.state[param]['prev_weights']
                # self.state[param]['prev_weights'] = param.data
                self.state[param]['agg_score']
                # agg_score = self.state[param]['agg_score']

                if self.nestrov:
                    # print(f"self.state[param]['lookahead'] requires_grad: {self.state[param]['lookahead'].requires_grad}")
                    # print(f"param requires_grad: {param.requires_grad}")
                    self.state[param]['lookahead'] = (param - momentum * self.state[param]['velocity']).requires_grad_(True)
                    if param.grad is None:
                        raise RuntimeError("param.grad is None. Ensure gradients are computed before calling step().")
                    computed_grad = torch.autograd.grad(self.state[param]['lookahead'], param, grad_outputs=param.grad, retain_graph=True, allow_unused=True)[0]
                    param.grad = computed_grad.detach()
                if epoch < self.warmup_epochs:
                    # if self.nestrov:
                    #     # velocity = momentum * velocity - lr * (scale * self.state[param]['lookahead'] + param.grad)
                    #     self.state[param]['velocity'].mul_(momentum).add_(lr * (scale * param.data + param.grad))
                    #     param.data.sub_(self.state[param]['velocity'])
                    # else:
                    # velocity = momentum * velocity + scale * param.data + param.grad
                    self.state[param]['velocity'].mul_(momentum).add_(scale * param.data + param.grad)
                    # param.data -= lr * velocity
                    param.data.sub_(lr * self.state[param]['velocity'])
                    # self.state[param]['velocity'] = velocity
                    score = torch.abs(param.grad * param.data)
                    # agg_score += score
                    self.state[param]['agg_score'] += score
                else:
                    # Check if the parameter is a weight matrix or bias vector
                    if param.dim() in [2, 4]:  # Assuming weight matrices have more than 1 dimension
                        # Custom weight update: Scale gradients before applying update
                        # print(torch.mul(param, param.grad).cpu().numpy())
                        s_W = torch.abs(param.data * param.grad)  # Move to CPU, convert to NumPy, and flatten
                        # agg_score += s_W
                        self.state[param]['agg_score'] += s_W
                        # s_W = -s_W
                        unique_values = torch.unique(s_W)
                        n_classes = min(2, len(unique_values))  # Ensure n_classes is valid
                        if n_classes > 1:
                            # # Update velocity
                            velocity_flat = self.state[param]['velocity'].view(-1)
                            param_data_flat = param.data.view(-1)
                            param_grad_flat = param.grad.data.view(-1)
                            WB_cuda_flatten = s_W.flatten()
                            WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                            WB_cuda_sorted = WB_cuda_sorted.reshape(s_W.shape)
                            # lookahead = self.state[param]['lookahead']
                            # lookahead_flat = self.state[param]['lookahead'].view(-1)
                            var = module_weights.jenks_optimization_cuda(WB_cuda_sorted)
                            var_min = var.argmin().item()
                            # print(f"Weight")
                            # print(f"var_min: {var_min}")
                            # print(f"WB_cuda_indices size: {WB_cuda_indices.size()}")

                            indices_ = WB_cuda_indices[:var_min]
                            del WB_cuda_sorted, WB_cuda_indices, var
                            torch.cuda.empty_cache()
                            param_grad_flat[indices_] = 0
                            # if self.nestrov:
                            #     # velocity_flat[indices] = momentum * velocity_flat[indices] - lr * (scale * param_data_flat[indices] + param_grad_flat[indices])
                            #     velocity_flat[indices].mul_(momentum).add_(lr * (scale * param_data_flat[indices] + param_grad_flat[indices]))
                            #     # velocity_flat[indices_] = momentum * velocity_flat[indices_] - lr * (scale * param_data_flat[indices_] + param_grad_flat[indices_])
                            #     velocity_flat[indices_].mul_(momentum).add_(lr * (scale * param_data_flat[indices_]))
                            #     # param_data_flat[indices] += velocity_flat[indices]
                            #     param_data_flat.sub_(velocity_flat)
                            #     # param_data_flat[indices_] += velocity_flat[indices_]
                            # else:
                            velocity_flat.mul_(momentum).add_(scale * param_data_flat + param_grad_flat)
                            # velocity_flat[indices_].mul_(momentum).add_(scale * param_data_flat[indices_])
                            # velocity_flat[indices_] = momentum * velocity_flat[indices_] + scale * param_data_flat[indices_]
                            param_data_flat -= lr * velocity_flat
                            # param_data_flat[indices_] -= lr * velocity_flat[indices_]

                            # Update parameters

                            param.data = param_data_flat.view(param.data.shape)
                            self.state[param]['velocity'] = velocity_flat.view(self.state[param]['velocity'].shape)
                            del param_data_flat, param_grad_flat, velocity_flat
                            torch.cuda.empty_cache()
                        else:
                            velocity = momentum * velocity + scale * param.data + param.grad
                            param.data -= lr * velocity
                            self.state[param]['velocity'] = velocity
                    else:  # Assuming bias vectors have 1 dimension
                        if self.bias:
                            s_B = torch.mul(param, param.grad)  # Move to CPU, convert to NumPy, and flatten
                            s_B = torch.abs(s_B)
                            # agg_score += s_B
                            self.state[param]['agg_score'] = s_B
                            unique_values = torch.unique(s_B)
                            n_classes = min(2, len(unique_values))  # Ensure n_classes is valid
                            # jnb = JenksNaturalBreaks(n_classes)
                            # jnb.fit(s_B)
                            # labels = jnb.labels_
                            # indices = np.where(labels == 1)[0]
                            # indices_ = np.where(labels == 0)[0]
                            B_cuda_sorted, B_cuda_indices = s_B.sort()
                            var = module_bias.jenks_optimization_biases_cuda(B_cuda_sorted)
                            var_min = var.argmin().item()
                            # print(f"Bias")
                            # print(f"var_min: {var_min}")
                            # print(f"B_cuda_indices size: {B_cuda_indices.size()}")
                            # Print the output
                            indices_ = B_cuda_indices[:var_min]
                            indices = B_cuda_indices[var_min:]
                            del B_cuda_sorted, B_cuda_indices, var
                            torch.cuda.empty_cache()
                            # lookahead = self.state[param]['lookahead']
                            # lookahead_flat = lookahead.view(-1)
                            # Update velocity
                            velocity_flat = self.state[param]['velocity'].view(-1)
                            param_data_flat = param.data.view(-1)
                            param_grad_flat = param.grad.data.view(-1)
                            param_grad_flat[indices_] = 0
                            # if self.nestrov:
                            #     # velocity_flat[indices] = momentum * velocity_flat[indices] - lr * (scale * param_data_flat[indices] + param_grad_flat[indices])
                            #     velocity_flat[indices].mul_(momentum).add_(lr * (scale * param_data_flat[indices] + param_grad_flat[indices]))
                            #     # velocity_flat[indices_] = momentum * velocity_flat[indices_] - lr * (scale * param_data_flat[indices_] + param_grad_flat[indices_])
                            #     velocity_flat[indices_].mul_(momentum).add_(lr * (scale * param_data_flat[indices_]))
                            #     # param_data_flat[indices] += velocity_flat[indices]
                            #     param_data_flat.sub_(velocity_flat)
                            # else:
                            # velocity_flat[indices] = momentum * velocity_flat[indices] + scale * param_data_flat[indices] + param_grad_flat[indices]
                            velocity_flat.mul_(momentum).add_(scale * param_data_flat + param_grad_flat)
                            # velocity_flat[indices_] = momentum * velocity_flat[indices_] + scale * param_data_flat[indices_]
                            param_data_flat -= lr * velocity_flat
                            # Update parameters
                            param.data = param_data_flat.view(param.data.shape)
                            self.state[param]['velocity'] = velocity_flat.view(self.state[param]['velocity'].shape)
                            del param_data_flat, param_grad_flat, velocity_flat
                            torch.cuda.empty_cache()
                        else:
                            self.state[param]['velocity'].mul_(momentum).add_(scale * param.data + param.grad)
                            param.data.sub_(lr * self.state[param]['velocity'])
        return loss
    def set_lr(self, lr):
        for group in self.param_groups:
            group['lr'] = lr
    def PruneWeights_Test(self, model):    
        jnb = JenksNaturalBreaks(2)
        for param in model.parameters():
            if param.dim() in [2, 4]: 
                layer = param.data.flatten()
                if 'agg_score' not in self.state[param]:
                    print("agg_score not found for param")
                    break
                score = self.state[param]['agg_score']
                # print(f"agg_score for param: {score}")
                # print(f"agg_score shape: {score.shape}")    
                WB_cuda_flatten = score.flatten()
                # print(f"WB_cuda_flatten shape: {WB_cuda_flatten.shape}")

                # Check for invalid values
                if torch.isnan(WB_cuda_flatten).any() or torch.isinf(WB_cuda_flatten).any():
                    print("Invalid values in WB_cuda_flatten")
                    continue

                WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                # print(f"WB_cuda_sorted shape: {WB_cuda_sorted.shape}")
                # print(f"WB_cuda_indices shape: {WB_cuda_indices.shape}")
                WB_cuda_sorted = WB_cuda_sorted.reshape(score.shape)

                var = module_weights.jenks_optimization_cuda(WB_cuda_sorted)
                var_min = var.argmin().item()

                # Validate var_min
                if var_min <= 0 or var_min > WB_cuda_indices.size(0):
                    print(f"Invalid var_min: {var_min}")
                    continue

                indices_ = WB_cuda_indices[:var_min]
                layer[indices_] = 0
                layer = layer.reshape(param.data.shape)
                param.data = layer
            elif param.dim() == 1 and self.bias:
                layer = param.data
                if 'agg_score' not in self.state[param]:
                    print("agg_score not found for param")
                    break
                score = self.state[param]['agg_score']
                B_cuda_sorted, B_cuda_indices = score.sort()
                var = module_bias.jenks_optimization_biases_cuda(B_cuda_sorted)
                var_min = var.argmin().item()
                # Print the output
                indices_ = B_cuda_indices[:var_min]
                layer[indices_] = 0
                param.data = layer
            else:
                print("Invalid parameter dimension")
                continue
        return model
    

def torch_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # print(f"Output shape: {output.shape}")
    # print(f"Target shape: {target.shape}")
    if target.ndim > 1:
        target = target.argmax(dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        is_correct = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # Fix applied here
        res.append(is_correct.mul_(100.0 / batch_size))
    return res




def train_one_step(net, data, label, optimizer, criterion, epoch, warmup_epochs):
    ## Check if the agg score is already defined
    # if epoch == 0:
    for group in optimizer.param_groups:
        for param in group['params']:
            if 'agg_score' not in optimizer.state[param]:
                optimizer.state[param]['agg_score'] = torch.zeros_like(param.data)
            if 'exp_avg' not in optimizer.state[param]:
                optimizer.state[param]['exp_avg'] = torch.zeros_like(param.data)
            if 'exp_avg_sq' not in optimizer.state[param]:
                optimizer.state[param]['exp_avg_sq'] = torch.zeros_like(param.data)
            if 'step' not in optimizer.state[param]:
                optimizer.state[param]['step'] = torch.tensor(0, dtype = torch.float32, device = 'cpu')
            if 'mask' not in optimizer.state[param]:
                optimizer.state[param]['mask'] = torch.ones_like(param.data, requires_grad=False)
    optimizer.zero_grad()
    pred = net(data)
    loss = criterion(pred, label)
    loss.backward()

    # to_concat_g = []
    # to_concat_v = []
    if epoch > warmup_epochs:
        for name, param in net.named_parameters():
            if param.dim() == 4:  # Convolutional layer weights (4D tensor)
                # Iterate over each kernel (slice along the output channel dimension)
                agg_sal = []
                percent_pruned_list = []
                for kernel_idx in range(param.shape[0]):
                    kernel = param[kernel_idx]  # Access the kernel (3D tensor)
                    kernel_grad = param.grad[kernel_idx]  # Access the gradient of the kernel

                    # Flatten the kernel and its gradient
                    kernel_data_flat = kernel.view(-1)
                    kernel_grad_flat = kernel_grad.view(-1)

                    # Perform the calculations from lines 386-395
                    WB_cuda_flatten = torch.abs(kernel_data_flat * kernel_grad_flat)
                    agg_sal.append(WB_cuda_flatten)
                    WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                    WB_cuda_sorted = WB_cuda_sorted.reshape(kernel.shape)
                    var = module_weights.jenks_optimization_cuda(WB_cuda_sorted)
                    var_min = var.argmin().item()
                    indices_ = WB_cuda_indices[:var_min]
                    percent_pruned = len(indices_) / len(kernel_grad_flat)
                    percent_pruned_list.append(percent_pruned)
                    param.grad[kernel_idx].view(-1)[indices_] = 0
                    optimizer.state[param]['agg_score'][kernel_idx] += WB_cuda_flatten.view(kernel.shape)
                if name and hasattr(optimizer, "layerwise_lr_stats"):
                    print(f"Layerwise scaling for {name}")
                    stats = optimizer.layerwise_lr_stats.get(name, {})
                    stats['percent_pruned'] = mean(percent_pruned_list)
                    stats['saliency_std'] = torch.std(torch.cat(agg_sal)).item()
                    optimizer.layerwise_lr_stats[name] = stats

            elif param.dim() == 2:  # Fully connected layer weights
                param_data_flat = param.data.view(-1)
                param_grad_flat = param.grad.data.view(-1)
                WB_cuda_flatten = torch.abs(param_data_flat * param_grad_flat)
                WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                WB_cuda_sorted = WB_cuda_sorted.reshape(param.data.shape)
                var = module_weights.jenks_optimization_cuda(WB_cuda_sorted)
                var_min = var.argmin().item()
                indices_ = WB_cuda_indices[:var_min]
                param.grad.view(-1)[indices_] = 0
                if name and hasattr(optimizer, "layerwise_lr_stats"):
                    print(f"Layerwise scaling for {name}")
                    stats = optimizer.layerwise_lr_stats.get(name, {})
                    percent_pruned = len(indices_) / len(param_grad_flat)
                    stats['percent_pruned'] = percent_pruned
                    stats['saliency_std'] = torch.std(WB_cuda_flatten).item()
                    optimizer.layerwise_lr_stats[name] = stats
                optimizer.state[param]['agg_score'] += WB_cuda_flatten.view(param.data.shape)

            elif param.dim() == 1:  # Bias terms
                param_data_flat = param.data.view(-1)
                param_grad_flat = param.grad.data.view(-1)
                B_cuda_flatten = torch.abs(param_data_flat * param_grad_flat)
                B_cuda_sorted, B_cuda_indices = B_cuda_flatten.sort()
                var = module_bias.jenks_optimization_biases_cuda(B_cuda_sorted)
                var_min = var.argmin().item()
                indices_ = B_cuda_indices[:var_min]
                ## Find the percent of pruned weights
                percent_pruned = len(indices_) / len(param_grad_flat)
                if name and hasattr(optimizer, "layerwise_lr_stats"):
                    print(f"Layerwise scaling for {name}")
                    stats = optimizer.layerwise_lr_stats.get(name, {})
                    stats['percent_pruned'] = percent_pruned
                    stats['saliency_std'] = torch.std(B_cuda_flatten).item()
                    optimizer.layerwise_lr_stats[name] = stats
                param.grad.view(-1)[indices_] = 0
                optimizer.state[param]['agg_score'] += param_grad_flat.view(param.data.shape)

    optimizer.step()
    acc, acc5 = torch_accuracy(pred, label, (1,5))
    return acc, acc5, loss

def train_one_step_prune(net, data, label, optimizer, criterion, epoch, warmup_epochs, prune_epochs, no_jenks = False, filter_based = True, mask = False, L2 = False,lambda_ = 0.01, debug = False, debugfile = None, jenksfile = None):
    ## Check if the agg score is already defined
    # if epoch == 0:
    for group in optimizer.param_groups:
        for param in group['params']:
            if 'agg_score' not in optimizer.state[param]:
                optimizer.state[param]['agg_score'] = torch.zeros_like(param.data)
            if 'exp_avg' not in optimizer.state[param]:
                optimizer.state[param]['exp_avg'] = torch.zeros_like(param.data)
            if 'exp_avg_sq' not in optimizer.state[param]:
                optimizer.state[param]['exp_avg_sq'] = torch.zeros_like(param.data)
            if 'step' not in optimizer.state[param]:
                optimizer.state[param]['step'] = torch.tensor(0, dtype = torch.float32, device = 'cpu')
            if 'mask' not in optimizer.state[param]:
                optimizer.state[param]['mask'] = torch.ones_like(param.data, requires_grad=False)
    optimizer.zero_grad()
    pred = net(data)
    loss = criterion(pred, label)
    if L2:
        l2_reg = sum(torch.norm(p) ** 2 for p in net.parameters())
        loss = loss + lambda_ * l2_reg
    loss.backward()
    total_params = 0
    total_pruned = 0
    # to_concat_g = []
    # to_concat_v = []

    if (epoch > warmup_epochs and epoch <= prune_epochs):
        for name, param in net.named_parameters():
            if param.dim() == 4:  # Convolutional layer weights (4D tensor)
                # Iterate over each kernel (slice along the output channel dimension)
                if filter_based:
                    agg_sal = []
                    percent_pruned_list = []
                    WB = torch.abs(param.data * param.grad)
                    mask = Conv_Mask(WB.contiguous())
                    mask = mask.to(param.device)
                    param.grad = mask * param.grad
                    WB_prime = mask * WB
                    total_params += mask.numel()
                    total_pruned += (mask.numel()-mask.sum().item())
                    percent_pruned = (mask.numel()-mask.sum().item()) / mask.numel()
                    percent_pruned_list.append(percent_pruned)
                    optimizer.state[param]['agg_score'] += WB_prime
                    if name and hasattr(optimizer, "layerwise_lr_stats"):
                        stats = optimizer.layerwise_lr_stats.get(name, {})
                        stats['percent_pruned'] = percent_pruned
                        stats['saliency_std'] = torch.std(WB).item()
                        if debug:
                            if debugfile != None:
                                with open(debugfile, 'a') as f:
                                    f.write(f"Layer Name: {name}\n")
                                    f.write(f"Percent Pruned: {stats['percent_pruned']}\n")
                                    f.write(f"Saliency Std: {stats['saliency_std']}\n")
                        optimizer.layerwise_lr_stats[name] = stats
                else:
                    ## Perform Jenks on the entire layer, not kernel by kernel
                    param_data_flat = param.data.view(-1)
                    param_grad_flat = param.grad.data.view(-1)
                    WB_cuda_flatten = torch.abs(param_data_flat * param_grad_flat)
                    WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                    WB_cuda_sorted = WB_cuda_sorted.reshape(param.data.shape)
                    WB_cuda_sorted = WB_cuda_sorted.contiguous()
                    var = module_weights.jenks_optimization_cuda(WB_cuda_sorted)
                    var_min = var.argmin().item()
                    indices_ = WB_cuda_indices[:var_min]
                    total_params += len(param_grad_flat)
                    total_pruned += len(indices_)
                    mask = torch.ones_like(param.data, requires_grad=False)
                    mask.view(-1)[indices_] = 0
                    mask = mask.to(param.device)
                    param.grad = mask * param.grad
                    WB_prime = mask * WB_cuda_flatten.view(param.data.shape)
                    WB_cuda_flatten[indices_] = 0
                    optimizer.state[param]['agg_score'] += WB_prime
                    if name and hasattr(optimizer, "layerwise_lr_stats"):
                        stats = optimizer.layerwise_lr_stats.get(name, {})
                        percent_pruned = len(indices_) / len(param_grad_flat)
                        stats['percent_pruned'] = percent_pruned
                        stats['saliency_std'] = torch.std(WB_cuda_flatten).item()
                        if debug:
                            if debugfile != None:
                                with open(debugfile, 'a') as f:
                                    f.write(f"Layer Name: {name}\n")
                                    f.write(f"Percent Pruned: {stats['percent_pruned']}\n")
                                    f.write(f"Saliency Std: {stats['saliency_std']}\n")
                        optimizer.layerwise_lr_stats[name] = stats
            elif param.dim() == 2:  # Fully connected layer weights
                param_data_flat = param.data.view(-1)
                param_grad_flat = param.grad.data.view(-1)
                WB_cuda_flatten = torch.abs(param_data_flat * param_grad_flat)
                WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                WB_cuda_sorted = WB_cuda_sorted.reshape(param.data.shape)
                var = module_weights.jenks_optimization_cuda(WB_cuda_sorted.contiguous())
                var_min = var.argmin().item()
                percent_pruned = len(indices_) / len(param_grad_flat)
                indices_ = WB_cuda_indices[:var_min]
                mask = torch.ones_like(param.data, requires_grad=False)
                mask.view(-1)[indices_] = 0
                mask = mask.to(param.device)
                param.grad = mask * param.grad
                WB_cuda_flatten = mask * WB_cuda_flatten.view(param.data.shape)
                total_params += len(param_grad_flat)
                total_pruned += len(indices_)
                optimizer.state[param]['agg_score'] += WB_cuda_flatten.view(param.data.shape)
                if name and hasattr(optimizer, "layerwise_lr_stats"):
                    stats = optimizer.layerwise_lr_stats.get(name, {})
                    percent_pruned = len(indices_) / len(param_grad_flat)
                    stats['percent_pruned'] = percent_pruned
                    stats['saliency_std'] = torch.std(WB_cuda_flatten).item()
                    if debug:
                        if debugfile != None:
                            with open(debugfile, 'a') as f:
                                f.write(f"Layer Name: {name}\n")
                                f.write(f"Percent Pruned: {stats['percent_pruned']}\n")
                                f.write(f"Length of indices_: {len(indices_)}")
                                f.write(f"Saliency Std: {stats['saliency_std']}\n")
                    optimizer.layerwise_lr_stats[name] = stats

            elif param.dim() == 1:  # Bias terms
                param_data_flat = param.data.view(-1)
                param_grad_flat = param.grad.data.view(-1)
                B_cuda_flatten = torch.abs(param_data_flat * param_grad_flat)
                B_cuda_sorted, B_cuda_indices = B_cuda_flatten.sort()
                var = module_bias.jenks_optimization_biases_cuda(B_cuda_sorted.contiguous())
                var_min = var.argmin().item()
                indices_ = B_cuda_indices[:var_min]
                # if debug:
                #     print(f"Layer Name: {name}")
                #     print(f"Var: {var}")
                #     print(f"Var Min: {var.min()}")
                #     print(f"B_cuda_indices size: {B_cuda_indices.size()}")
                #     print(f"Indices_ size: {indices_.size()}")
                param.grad.view(-1)[indices_] = 0
                optimizer.state[param]['agg_score'] += param_grad_flat.view(param.data.shape)
                percent_pruned = len(indices_) / len(param_grad_flat)
                total_params += len(param_grad_flat)
                total_pruned += len(indices_)
                if name and hasattr(optimizer, "layerwise_lr_stats"):
                    stats = optimizer.layerwise_lr_stats.get(name, {})
                    stats['percent_pruned'] = percent_pruned
                    stats['saliency_std'] = torch.std(B_cuda_flatten).item()
                    if debug:
                        if debugfile != None:
                            with open(debugfile, 'a') as f:
                                f.write(f"Layer Name: {name}\n")
                                f.write(f"Percent Pruned: {stats['percent_pruned']}\n")
                                f.write(f"Saliency Std: {stats['saliency_std']}\n")
                    optimizer.layerwise_lr_stats[name] = stats
                param.grad.view(-1)[indices_] = 0
                B_cuda_flatten[indices_] = 0
                optimizer.state[param]['agg_score'] += B_cuda_flatten.view(param.data.shape)
    if epoch > prune_epochs:
        for name, param in net.named_parameters():
            if param.dim() == 4:  # Convolutional layer weights (4D tensor)
                # Iterate over each kernel (slice along the output channel dimension)
                if filter_based:
                    WB = torch.abs(param.data * param.grad)
                    mask = Conv_Mask(WB)
                    mask = mask.to(param.device)
                    if not no_jenks:
                        param.grad = mask * param.grad
                        WB_prime = mask * WB
                    total_params += mask.numel()
                    total_pruned += (mask.numel()-mask.sum().item())
                    percent_pruned = (mask.numel()-mask.sum().item()) / mask.numel()
                    percent_pruned_list.append(percent_pruned)
                    optimizer.state[param]['agg_score'] += WB_prime
                    if name and hasattr(optimizer, "layerwise_lr_stats"):
                        stats = optimizer.layerwise_lr_stats.get(name, {})
                        stats['percent_pruned'] = percent_pruned
                        stats['saliency_std'] = torch.std(WB).item()
                        if debug:
                            if debugfile != None:
                                with open(debugfile, 'a') as f:
                                    f.write(f"Layer Name: {name}\n")
                                    f.write(f"Percent Pruned: {stats['percent_pruned']}\n")
                                    f.write(f"Saliency Std: {stats['saliency_std']}\n")
                        optimizer.layerwise_lr_stats[name] = stats
                else:
                    ## Perform Jenks on the entire layer, not kernel by kernel
                    param_data_flat = param.data.view(-1)
                    param_grad_flat = param.grad.data.view(-1)
                    WB_cuda_flatten = torch.abs(param_data_flat * param_grad_flat)
                    if not no_jenks:
                        WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                        WB_cuda_sorted = WB_cuda_sorted.reshape(param.data.shape)
                        var = module_weights.jenks_optimization_cuda(WB_cuda_sorted.contiguous())
                        var_min = var.argmin().item()
                        indices_ = WB_cuda_indices[:var_min]
                        mask = torch.ones_like(param.data, requires_grad=False)
                        mask.view(-1)[indices_] = 0
                        mask = mask.to(param.device)
                        param.grad *= mask
                        WB_cuda_flatten[indices_] = 0
                        if name and hasattr(optimizer, "layerwise_lr_stats"):
                            stats = optimizer.layerwise_lr_stats.get(name, {})
                            percent_pruned = len(indices_) / len(param_grad_flat)
                            stats['percent_pruned'] = percent_pruned
                            stats['saliency_std'] = torch.std(WB_cuda_flatten).item()
                            optimizer.layerwise_lr_stats[name] = stats
                    total_params += len(param_grad_flat)
                    total_pruned += len(indices_)
                    optimizer.state[param]['agg_score'] += WB_cuda_flatten.view(param.data.shape)
            elif param.dim() == 2:  # Fully connected layer weights
                param_data_flat = param.data.view(-1)
                param_grad_flat = param.grad.data.view(-1)
                WB_cuda_flatten = torch.abs(param_data_flat * param_grad_flat)
                if not no_jenks:
                    WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                    WB_cuda_sorted = WB_cuda_sorted.reshape(param.data.shape)
                    var = module_weights.jenks_optimization_cuda(WB_cuda_sorted.contiguous())
                    var_min = var.argmin().item()
                    indices_ = WB_cuda_indices[:var_min]
                    param.grad.view(-1)[indices_] = 0
                    total_params += len(param_grad_flat)
                    total_pruned += len(indices_)
                    WB_cuda_flatten[indices_] = 0
                    if name and hasattr(optimizer, "layerwise_lr_stats"):
                        stats = optimizer.layerwise_lr_stats.get(name, {})
                        percent_pruned = len(indices_) / len(param_grad_flat)
                        stats['percent_pruned'] = percent_pruned
                        stats['saliency_std'] = torch.std(WB_cuda_flatten).item()
                        optimizer.layerwise_lr_stats[name] = stats
                optimizer.state[param]['agg_score'] += WB_cuda_flatten.view(param.data.shape)

            elif param.dim() == 1:  # Bias terms
                param_data_flat = param.data.view(-1)
                param_grad_flat = param.grad.data.view(-1)
                B_cuda_flatten = torch.abs(param_data_flat * param_grad_flat)
                if not no_jenks:
                    B_cuda_sorted, B_cuda_indices = B_cuda_flatten.sort()
                    var = module_bias.jenks_optimization_biases_cuda(B_cuda_sorted)
                    var_min = var.argmin().item()
                    indices_ = B_cuda_indices[:var_min]
                    param.grad.view(-1)[indices_] = 0
                    total_params += len(param_grad_flat)
                    total_pruned += len(indices_)
                    B_cuda_flatten[indices_] = 0
                    if name and hasattr(optimizer, "layerwise_lr_stats"):
                        stats = optimizer.layerwise_lr_stats.get(name, {})
                        stats['percent_pruned'] = percent_pruned
                        stats['saliency_std'] = torch.std(B_cuda_flatten).item()
                        optimizer.layerwise_lr_stats[name] = stats
                optimizer.state[param]['agg_score'] += B_cuda_flatten.view(param.data.shape)
    if epoch > warmup_epochs:
        if total_params != 0:
            with open(jenksfile, 'a') as f:
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Total Params: {total_params}\n")
                f.write(f"Total Pruned: {total_pruned}\n")
                f.write(f"Percent Pruned: {total_pruned/total_params}\n")
                f.write("\n")  
        else:
            with open(jenksfile, 'a') as f:
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Total Params: {total_params}\n")
                f.write(f"Total Pruned: {total_pruned}\n")   
                f.write("\n")  
    optimizer.step()
    if epoch > prune_epochs:
        if mask:
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data * optimizer.state[param]['mask']
    acc, acc5 = torch_accuracy(pred, label, (1,5))
    return acc, acc5, loss    

def init_network(optimizer):
    for group in optimizer.param_groups:
        for param in group['params']:
            if 'agg_score' not in optimizer.state[param]:
                optimizer.state[param]['agg_score'] = torch.zeros_like(param.data)
            if 'exp_avg' not in optimizer.state[param]:
                optimizer.state[param]['exp_avg'] = torch.zeros_like(param.data)
            if 'exp_avg_sq' not in optimizer.state[param]:
                optimizer.state[param]['exp_avg_sq'] = torch.zeros_like(param.data)
            if 'step' not in optimizer.state[param]:
                optimizer.state[param]['step'] = torch.tensor(0, dtype = torch.float32, device = 'cpu')
            if 'mask' not in optimizer.state[param]:
                optimizer.state[param]['mask'] = torch.ones_like(param.data, requires_grad=False)
            if 'update_count' not in optimizer.state[param]:
                optimizer.state[param]['update_count'] = torch.zeros_like(param.data, requires_grad=False).to(param.device)

def compute_mask(param, WB, filter_based = True, bias_prune = True):
    if param.dim() == 4:
        if filter_based: 
            return Conv_Mask(WB)
        else:
            Mask, BVF = Bias_Mask(WB.view(-1))
            Mask_fin = Mask.view(param.shape)
            del Mask
            return Mask_fin, BVF
    elif param.dim() == 2:
        return Linear_Mask(WB)
    elif param.dim() == 1:
        if bias_prune:
            return Bias_Mask(WB)
        else:
            return torch.ones_like(param, requires_grad=False), 1
    else:
        raise ValueError("Unsupported parameter dimension")

def train_one_step_prune_v2(net, dataloader, optimizer, criterion, epoch, warmup_epochs, prune_epochs, no_jenks = False, bias_prune = True, filter_based = True, mask = False, L2 = False,lambda_ = 0.01, debug = False, debugfile = None, jenksfile = None, mag=False, elem_bias = False, accumulation_steps = 1):
    debug_logs = []
    jenks_logs = []
    align_check = True
    accumulation_steps = 1
    print_steps = 10
    optimizer.zero_grad()
    count = 0
    loss = 0
    acc = 0
    acc5 = 0
    device = next(net.parameters()).device
    '''We need to compute the gradient mask and apply it here, but for 
            experimentation, we need to collect the gradients and the weights before and after update
            We will then compute the dot product between them to see how much they align'''
    weights_before = {}
    weights_after = {}
    gradients_before = {}
    gradients_after = {}
    velocity = {}
    num_layers = len(list(net.parameters()))
    if elem_bias:
        optimizer.epoch = epoch
    for i, (data, label) in enumerate(dataloader):
        torch.cuda.empty_cache()
        count+=1
        start = time.time()
        data,label = data.to(device), label.to(device)
        pred = net(data)
        loss_iter = criterion(pred, label)
        if L2:
            l2_reg = sum(torch.norm(p) ** 2 for p in net.parameters())
            loss_iter = loss_iter + lambda_ * l2_reg
        loss_iter.backward()
        apply_gradient_centralization(net)
        with torch.no_grad():
            if mask and epoch>prune_epochs:
                    ## Go through all the parameters and set the pruned ones to zero
                for name, param in net.named_parameters():
                    param.data.mul_(optimizer.state[param]['mask'])
        loss += loss_iter.item()
        acc_dummy, acc5_dummy = torch_accuracy(pred, label, (1,5))
        acc += acc_dummy
        acc5 += acc5_dummy
        num_layers = len(list(net.parameters()))
        GVF = {}
        
        total_params = 0
        total_pruned = 0
        gradient_norm = 0
        gradient_norm_masked = 0
        print(f"Starting Epoch: {epoch}, Iteration: {i+1}/{len(dataloader)}")
        if warmup_epochs < epoch <= prune_epochs or epoch > prune_epochs:
            print("Pruning Step")
            if elem_bias:
                for name, param in net.named_parameters():
                    layer_pruned = 0
                    layer_params = 0
                    # Collect initial magnitude of gradient
                    # if velocity is not None:
                    #     # You can use velocity here, e.g. print or log it
                    #     pass  # Replace with your logic
                    if mag:
                        sign_WB = param.data
                    else:
                        sign_WB = param.data * param.grad
                    WB = torch.abs(sign_WB)
                    # print(WB.shape)
                    ## Generate a binary matrix, b_ij = 1 if sign_WB_ij<0 else 0
                    ## This will help us identify which weights to prune
                    decay_mask = torch.ones_like(WB, requires_grad=False)
                    decay_mask[sign_WB > 0] = 0
                    if 'bn' not in name:
                        mask_tensor, GVF_val = compute_mask(param, WB, filter_based, bias_prune)
                    else:
                        mask_tensor = torch.ones_like(param.data, requires_grad=False)
                        GVF_val = 1
                    mask_tensor = mask_tensor.to(device)
                    decay_tensor = torch.ones_like(mask_tensor, requires_grad=False, device=device)
                    decay_tensor.sub_(mask_tensor)
                    optimizer.state[param]['update_count'].add_(decay_tensor)
                    del decay_tensor
                    GVF[name] = GVF_val
                    with torch.no_grad():
                        if epoch > warmup_epochs:
                            if not no_jenks:
                                param.grad.mul_(mask_tensor)
                                gradients_after[name] = param.grad.clone().detach()
                                gradient_norm_masked += torch.norm(param.grad).item()
                                WB_prime = WB * mask_tensor
                                optimizer.state[param]['agg_score'] += WB_prime
                            else:
                                WB_prime = WB
                                optimizer.state[param]['agg_score'] += WB_prime
                        if param.dim() == 1:
                            if bias_prune:
                                total_params += mask_tensor.numel()
                                total_pruned += (mask_tensor.numel() - mask_tensor.sum().item())
                            else:
                                continue
                        elif param.dim() in [2, 4]:
                            total_params += mask_tensor.numel()
                            total_pruned += (mask_tensor.numel() - mask_tensor.sum().item())
                        layer_params += mask_tensor.numel()
                        layer_pruned += (mask_tensor.numel() - mask_tensor.sum().item())
                        # Debug stats
                        if name and hasattr(optimizer, "layerwise_lr_stats"):
                            stats = optimizer.layerwise_lr_stats.get(name, {})
                            stats['percent_pruned'] = layer_pruned / layer_params
                            stats['saliency_std'] = torch.std(WB).item()
                            optimizer.layerwise_lr_stats[name] = stats
                            # print(f"Debug is on: {debug}")
        print(f"Starting Epoch: {epoch}, Iteration: {i+1}/{len(dataloader)}")
        optimizer.step()
        optimizer.zero_grad()
        if epoch > prune_epochs and mask:
            for name, param in net.named_parameters():
                param.data.mul_(optimizer.state[param]['mask'])

    loss_avg = loss/ count
    acc_avg = acc / count
    acc5_avg = acc5 / count
    return acc_avg, acc5_avg, loss_avg




def train_one_step_prune_v2_ETF(net, dataloader, optimizer, criterion, epoch, warmup_epochs, prune_epochs, no_jenks = False, bias_prune = True, filter_based = True, mask = False, L2 = False,lambda_ = 0.01, debug = False, debugfile = None, jenksfile = None, scheduler=None):
    debug_logs = []
    jenks_logs = []
    accumulation_steps = 1
    print_steps = 10
    optimizer.zero_grad()
    count = 0
    loss = 0
    acc = 0
    acc5 = 0
    device = next(net.parameters()).device
    for i, (data, label) in enumerate(dataloader):
        torch.cuda.empty_cache()
        count+=1
        start = time.time()
        data,label = data.to(device), label.to(device)
        pred = net(data,label,training=True)
        loss_iter = criterion(pred, label)
        if L2:
            l2_reg = sum(torch.norm(p) ** 2 for p in net.parameters())
            loss_iter = loss_iter + lambda_ * l2_reg
        loss_iter.backward()
        if epoch <= warmup_epochs:
            ## Calculate the fronebius gradient norms for the optimizer layerwise_lr_stats
            for name, param in net.named_parameters():
                if param.dim() in [2,4]:
                    W_f = torch.norm(param.data, p='fro')
                    # print(f"The Weight Frobenius Norm for layer {name} is {W_f.item()}")
                    G_W_f = torch.norm(param.grad.data, p='fro')
                    # print(f"The Gradient Frobenius Norm for layer {name} is {G_W_f.item()}")
                    if name and hasattr(optimizer, "layerwise_lr_stats"):
                        stats = optimizer.layerwise_lr_stats.get(name, {})
                        stats['weight_frob_norm'] = W_f.item()
                        stats['grad_frob_norm'] = G_W_f.item()
                        optimizer.layerwise_lr_stats[name] = stats
            # if scheduler is not None:
            #     scheduler.step()
        with torch.no_grad():
            if mask and epoch>prune_epochs:
                    ## Go through all the parameters and set the pruned ones to zero
                for name, param in net.named_parameters():
                    param.data.mul_(optimizer.state[param]['mask'])
        loss += loss_iter.item()
        acc_dummy, acc5_dummy = torch_accuracy(pred, label, (1,5))
        acc += acc_dummy
        acc5 += acc5_dummy
        num_layers = len(list(net.parameters()))
        GVF = {}
        if (i+1)%accumulation_steps == 0:
            total_params = 0
            total_pruned = 0
            if warmup_epochs < epoch <= prune_epochs or epoch > prune_epochs:
                for name, param in net.named_parameters():
                    layer_pruned = 0
                    layer_params = 0
                    module = dict(net.named_modules()).get(name.rsplit('.', 1)[0], None)
                    WB = torch.abs(param.data * param.grad)
                    if (module is not None and module.do_prune):
                        mask_tensor, GVF_val = compute_mask(param, WB, filter_based, bias_prune)
                    else:
                        mask_tensor = torch.ones_like(param.data, requires_grad=False)
                        GVF_val = 1
                    mask_tensor = mask_tensor.to(device)
                    decay_tensor = torch.ones_like(mask_tensor, requires_grad=False, device=device)
                    decay_tensor.sub_(mask_tensor)
                    optimizer.state[param]['update_count'].add_(decay_tensor)
                    del decay_tensor
                    GVF[name] = GVF_val
                    with torch.no_grad():
                        if epoch > prune_epochs and mask:
                            param.data.mul_(optimizer.state[param]['mask'])

                        if epoch > warmup_epochs:
                            if not no_jenks:
                                param.grad.mul_(mask_tensor)
                                WB_prime = WB * mask_tensor
                            else:
                                WB_prime = WB

                        optimizer.state[param]['agg_score'] += WB_prime
                        if param.dim() == 1:
                            if bias_prune:
                                total_params += mask_tensor.numel()
                                total_pruned += (mask_tensor.numel() - mask_tensor.sum().item())
                            else:
                                continue
                        elif param.dim() in [2, 4]:
                            total_params += mask_tensor.numel()
                            total_pruned += (mask_tensor.numel() - mask_tensor.sum().item())
                        layer_params += mask_tensor.numel()
                        layer_pruned += (mask_tensor.numel() - mask_tensor.sum().item())
                        # Debug stats
                        if name and hasattr(optimizer, "layerwise_lr_stats"):
                            stats = optimizer.layerwise_lr_stats.get(name, {})
                            stats['percent_pruned'] = layer_pruned / layer_params
                            stats['saliency_std'] = torch.std(WB).item()
                            optimizer.layerwise_lr_stats[name] = stats
                            # print(f"Debug is on: {debug}")
                            if debug:
                                # debug_logs.append(
                                #     f"Layer Name: {name}\nPercent Pruned: {stats['percent_pruned']:.4f}\nSaliency Std: {stats['saliency_std']:.4f}\n"
                                # )
                                if isinstance(GVF[name],list):
                                    debug_logs.append(f"Layer Name: {name}\nGVF Values: {GVF[name]}\n")
                                else:
                                    debug_logs.append(f"Layer Name: {name}\nGVF Value: {GVF[name]:.4f}\n")

            optimizer.step()
            optimizer.zero_grad()

            # Logging after pruning starts
            if epoch > warmup_epochs:
                elapsed = time.time() - start
                log = (
                    f"Epoch: {epoch}\nIteration: {i}\nElapsed Time: {elapsed:.4f}\n"
                    f"Total Params: {total_params}\nTotal Pruned: {total_pruned}\n"
                )
                if total_params > 0:
                    log += f"Percent Pruned: {total_pruned / total_params:.4f}\n"
                    log += f"Accuracy: {acc.item() / count:.4f}\n"
                jenks_logs.append(log + "\n")
        if (i+1) % print_steps == 0:
            # Write logs once per epoch
            if debug and debugfile:
                # print("Debug works")
                with open(debugfile, 'a') as f:
                    f.writelines(debug_logs)
            if jenksfile and epoch > warmup_epochs:
                with open(jenksfile, 'a') as f:
                    f.writelines(jenks_logs)
            debug_logs.clear()
            jenks_logs.clear()

    loss_avg = loss/ count
    acc_avg = acc / count
    acc5_avg = acc5 / count
    return acc_avg, acc5_avg, loss_avg

def train_one_step_prune_v2_ResNet(net, dataloader, optimizer, criterion, epoch, warmup_epochs, prune_epochs, no_jenks = False, bias_prune = True, filter_based = True, mask = False, L2 = False,lambda_ = 0.01, debug = False, debugfile = None, jenksfile = None):
    debug_logs = []
    jenks_logs = []
    accumulation_steps = 1
    print_steps = 10
    optimizer.zero_grad()
    count = 0
    loss = 0
    acc = 0
    acc5 = 0
    device = next(net.parameters()).device
    cutmix = CutMix(num_classes=10)
    mixup = MixUp(num_classes=10)
    cutmix_or_mixup = RandomChoice([cutmix, mixup])
    ## Get the name of the final layer in order to skip performing Jenks on it
    final_layer_name = list(net.named_modules())[-1][0]
    for i, (data, label) in enumerate(dataloader):
        # if i % 4 == 0:
        #     data, label = cutmix_or_mixup(data, label)
        torch.cuda.empty_cache()
        count+=1
        start = time.time()
        data,label = data.to(device), label.to(device)
        pred = net(data)
        loss_iter = criterion(pred, label)
        if L2:
            l2_reg = sum(torch.norm(p) ** 2 for p in net.parameters())
            loss_iter = loss_iter + lambda_ * l2_reg
        loss_iter.backward()
        with torch.no_grad():
            if mask and epoch>prune_epochs:
                    ## Go through all the parameters and set the pruned ones to zero
                for name, param in net.named_parameters():
                    param.data.mul_(optimizer.state[param]['mask'])
        loss += loss_iter.item()
        acc_dummy, acc5_dummy = torch_accuracy(pred, label, (1,5))
        acc += acc_dummy
        acc5 += acc5_dummy
        num_layers = len(list(net.parameters()))
        GVF = {}
        if (i+1)%accumulation_steps == 0:
            total_params = 0
            total_pruned = 0
            if warmup_epochs < epoch <= prune_epochs or epoch > prune_epochs:
                for name, param in net.named_parameters():
                    layer_pruned = 0
                    layer_params = 0
                    WB = torch.abs(param.data * param.grad)
                    if 'bn' not in name or name != final_layer_name:
                        mask_tensor, GVF_val = compute_mask(param, WB, filter_based, bias_prune)
                    else:
                        mask_tensor = torch.ones_like(param.data, requires_grad=False)
                        GVF_val = 1
                    mask_tensor = mask_tensor.to(device)
                    decay_tensor = torch.ones_like(mask_tensor, requires_grad=False, device=device)
                    decay_tensor.sub_(mask_tensor)
                    optimizer.state[param]['update_count'].add_(decay_tensor)
                    del decay_tensor
                    GVF[name] = GVF_val
                    with torch.no_grad():
                        if epoch > prune_epochs and mask:
                            param.data.mul_(optimizer.state[param]['mask'])

                        if epoch > warmup_epochs:
                            if not no_jenks:
                                param.grad.mul_(mask_tensor)
                                WB_prime = WB * mask_tensor
                            else:
                                WB_prime = WB

                        optimizer.state[param]['agg_score'] += WB_prime
                        if param.dim() == 1:
                            if bias_prune:
                                total_params += mask_tensor.numel()
                                total_pruned += (mask_tensor.numel() - mask_tensor.sum().item())
                            else:
                                continue
                        elif param.dim() in [2, 4]:
                            total_params += mask_tensor.numel()
                            total_pruned += (mask_tensor.numel() - mask_tensor.sum().item())
                        layer_params += mask_tensor.numel()
                        layer_pruned += (mask_tensor.numel() - mask_tensor.sum().item())
                        # Debug stats
                        if name and hasattr(optimizer, "layerwise_lr_stats"):
                            stats = optimizer.layerwise_lr_stats.get(name, {})
                            stats['percent_pruned'] = layer_pruned / layer_params
                            stats['saliency_std'] = torch.std(WB).item()
                            optimizer.layerwise_lr_stats[name] = stats
                            # print(f"Debug is on: {debug}"

            optimizer.step()
            optimizer.zero_grad()

            # Logging after pruning starts
        #     if epoch > warmup_epochs:
        #         elapsed = time.time() - start
        #         log = (
        #             f"Epoch: {epoch}\nIteration: {i}\nElapsed Time: {elapsed:.4f}\n"
        #             f"Total Params: {total_params}\nTotal Pruned: {total_pruned}\n"
        #         )
        #         if total_params > 0:
        #             log += f"Percent Pruned: {total_pruned / total_params:.4f}\n"
        #             log += f"Accuracy: {acc.item() / count:.4f}\n"
        #         jenks_logs.append(log + "\n")
        # if (i+1) % print_steps == 0:
        #     # Write logs once per epoch
        #     if jenksfile and epoch > warmup_epochs:
        #         with open(jenksfile, 'a') as f:
        #             f.writelines(jenks_logs)
        #     debug_logs.clear()
        #     jenks_logs.clear()

    loss_avg = loss/ count
    acc_avg = acc / count
    acc5_avg = acc5 / count
    return acc_avg, acc5_avg, loss_avg

def train_one_step_prune_v2_ResNetETF(net, dataloader, optimizer, criterion, epoch, warmup_epochs, prune_epochs, no_jenks = False, bias_prune = True, filter_based = True, mask = False, L2 = False,lambda_ = 0.01, debug = False, debugfile = None, jenksfile = None):
    debug_logs = []
    jenks_logs = []
    accumulation_steps = 1
    print_steps = 10
    optimizer.zero_grad()
    count = 0
    loss = 0
    acc = 0
    acc5 = 0
    device = next(net.parameters()).device
    cutmix = CutMix(num_classes=10)
    mixup = MixUp(num_classes=10)
    cutmix_or_mixup = RandomChoice([cutmix, mixup])
    ## Get the name of the final layer in order to skip performing Jenks on it
    final_layer_name = list(net.named_modules())[-1][0]
    for i, (data, label) in enumerate(dataloader):
        # if i % 4 == 0:
        #     data, label = cutmix_or_mixup(data, label)
        torch.cuda.empty_cache()
        count+=1
        start = time.time()
        data,label = data.to(device), label.to(device)
        pred = net(data,label,training=True)
        loss_iter = criterion(pred, label)
        if L2:
            l2_reg = sum(torch.norm(p) ** 2 for p in net.parameters())
            loss_iter = loss_iter + lambda_ * l2_reg
        loss_iter.backward()
        with torch.no_grad():
            if mask and epoch>prune_epochs:
                    ## Go through all the parameters and set the pruned ones to zero
                for name, param in net.named_parameters():
                    param.data.mul_(optimizer.state[param]['mask'])
        loss += loss_iter.item()
        acc_dummy, acc5_dummy = torch_accuracy(pred, label, (1,5))
        acc += acc_dummy
        acc5 += acc5_dummy
        num_layers = len(list(net.parameters()))
        GVF = {}
        if (i+1)%accumulation_steps == 0:
            total_params = 0
            total_pruned = 0
            if warmup_epochs < epoch <= prune_epochs or epoch > prune_epochs:
                for name, param in net.named_parameters():
                    layer_pruned = 0
                    layer_params = 0
                    WB = torch.abs(param.data * param.grad)
                    module = dict(net.named_modules()).get(name.rsplit('.', 1)[0], None)
                    if module is not None and hasattr(module, 'do_prune') and module.do_prune:
                        mask_tensor, GVF_val = compute_mask(param, WB, filter_based, bias_prune)
                    else:
                        mask_tensor = torch.ones_like(param.data, requires_grad=False)
                        GVF_val = 1
                    mask_tensor = mask_tensor.to(device)
                    decay_tensor = torch.ones_like(mask_tensor, requires_grad=False, device=device)
                    decay_tensor.sub_(mask_tensor)
                    optimizer.state[param]['update_count'].add_(decay_tensor)
                    del decay_tensor
                    GVF[name] = GVF_val
                    with torch.no_grad():
                        if epoch > prune_epochs and mask:
                            param.data.mul_(optimizer.state[param]['mask'])

                        if epoch > warmup_epochs:
                            if not no_jenks:
                                param.grad.mul_(mask_tensor)
                                WB_prime = WB * mask_tensor
                            else:
                                WB_prime = WB

                        optimizer.state[param]['agg_score'] += WB_prime
                        if param.dim() == 1:
                            if bias_prune:
                                total_params += mask_tensor.numel()
                                total_pruned += (mask_tensor.numel() - mask_tensor.sum().item())
                            else:
                                continue
                        elif param.dim() in [2, 4]:
                            total_params += mask_tensor.numel()
                            total_pruned += (mask_tensor.numel() - mask_tensor.sum().item())
                        layer_params += mask_tensor.numel()
                        layer_pruned += (mask_tensor.numel() - mask_tensor.sum().item())
                        # Debug stats
                        if name and hasattr(optimizer, "layerwise_lr_stats"):
                            stats = optimizer.layerwise_lr_stats.get(name, {})
                            stats['percent_pruned'] = layer_pruned / layer_params
                            stats['saliency_std'] = torch.std(WB).item()
                            optimizer.layerwise_lr_stats[name] = stats
                            # print(f"Debug is on: {debug}"

            optimizer.step()
            optimizer.zero_grad()

            # Logging after pruning starts
        #     if epoch > warmup_epochs:
        #         elapsed = time.time() - start
        #         log = (
        #             f"Epoch: {epoch}\nIteration: {i}\nElapsed Time: {elapsed:.4f}\n"
        #             f"Total Params: {total_params}\nTotal Pruned: {total_pruned}\n"
        #         )
        #         if total_params > 0:
        #             log += f"Percent Pruned: {total_pruned / total_params:.4f}\n"
        #             log += f"Accuracy: {acc.item() / count:.4f}\n"
        #         jenks_logs.append(log + "\n")
        # if (i+1) % print_steps == 0:
        #     # Write logs once per epoch
        #     if jenksfile and epoch > warmup_epochs:
        #         with open(jenksfile, 'a') as f:
        #             f.writelines(jenks_logs)
        #     debug_logs.clear()
        #     jenks_logs.clear()

    loss_avg = loss/ count
    acc_avg = acc / count
    acc5_avg = acc5 / count
    return acc_avg, acc5_avg, loss_avg




def train_one_step_prune_global(net, dataloader, optimizer, criterion, epoch, warmup_epochs, prune_epochs, no_jenks = False, bias_prune = True, filter_based = True, mask = False, L2 = False,lambda_ = 0.01, debug = False, debugfile = None, jenksfile = None):
    debug_logs = []
    jenks_logs = []
    accumulation_steps = 1
    print_steps = 10
    optimizer.zero_grad()
    count = 0
    loss = 0
    acc = 0
    acc5 = 0
    device = next(net.parameters()).device
    for i, (data, label) in enumerate(dataloader):
        torch.cuda.empty_cache()
        count+=1
        start = time.time()
        data,label = data.to(device), label.to(device)
        pred = net(data)
        loss_iter = criterion(pred, label)
        if L2:
            l2_reg = sum(torch.norm(p) ** 2 for p in net.parameters())
            loss_iter = loss_iter + lambda_ * l2_reg
        loss_iter.backward()
        with torch.no_grad():
            if mask and epoch>prune_epochs:
                    ## Go through all the parameters and set the pruned ones to zero
                for name, param in net.named_parameters():
                    param.data.mul_(optimizer.state[param]['mask'])
        loss += loss_iter.item()
        acc_dummy, acc5_dummy = torch_accuracy(pred, label, (1,5))
        acc += acc_dummy
        acc5 += acc5_dummy
        num_layers = len(list(net.parameters()))
        GVF = {}
        if (i+1)%accumulation_steps == 0:
            total_params = 0
            total_pruned = 0
            if warmup_epochs < epoch <= prune_epochs or epoch > prune_epochs:
                to_concat_g = []
                to_concat_v = []
                for name, param in net.named_parameters():
                    to_concat_g.append(param.grad.view(-1))
                    to_concat_v.append(param.data.view(-1))
                all_g = torch.cat(to_concat_g)
                all_v = torch.cat(to_concat_v)
                metric = torch.abs(all_g * all_v)
                Net_mask, GVF = Bias_Mask(metric)
                total_params = metric.numel()
                total_pruned = (Net_mask == 0).sum().item()
                
                '''Now we need to reshape the mask to match the parameter shapes'''
                start = 0
                for name, param in net.named_parameters():
                    if param.requires_grad and param.dim() in [2, 4]:
                        numel = param.numel()
                        param_mask = Net_mask[start:start+numel].view(param.shape).to(param.device)
                        param.grad.data.mul_(param_mask)
                        # Optionally: optimizer.state[param]['mask'] = param_mask
                        start += numel
            optimizer.step()
            optimizer.zero_grad()

            # Logging after pruning starts
            if epoch > warmup_epochs:
                elapsed = time.time() - start
                log = (
                    f"Epoch: {epoch}\nIteration: {i}\nElapsed Time: {elapsed:.4f}\n"
                    f"Total Params: {total_params}\nTotal Pruned: {total_pruned}\n"
                )
                if total_params > 0:
                    log += f"Percent Pruned: {total_pruned / total_params:.4f}\n"
                    log += f"Accuracy: {acc.item() / count:.4f}\n"
                jenks_logs.append(log + "\n")
        if (i+1) % print_steps == 0:
            # Write logs once per epoch
            if debug and debugfile:
                # print("Debug works")
                with open(debugfile, 'a') as f:
                    f.writelines(debug_logs)
            if jenksfile and epoch > warmup_epochs:
                with open(jenksfile, 'a') as f:
                    f.writelines(jenks_logs)
            debug_logs.clear()
            jenks_logs.clear()

    loss_avg = loss/ count
    acc_avg = acc / count
    acc5_avg = acc5 / count
    return acc_avg, acc5_avg, loss_avg


def train_one_step_prune_globalETF(net, dataloader, optimizer, criterion, epoch, warmup_epochs, prune_epochs, no_jenks = False, bias_prune = True, filter_based = True, mask = False, L2 = False,lambda_ = 0.01, debug = False, debugfile = None, jenksfile = None):
    debug_logs = []
    jenks_logs = []
    accumulation_steps = 1
    print_steps = 10
    optimizer.zero_grad()
    count = 0
    loss = 0
    acc = 0
    acc5 = 0
    device = next(net.parameters()).device
    for i, (data, label) in enumerate(dataloader):
        torch.cuda.empty_cache()
        count+=1
        start = time.time()
        data,label = data.to(device), label.to(device)
        pred = net(data,label,training=True)
        loss_iter = criterion(pred, label)
        if L2:
            l2_reg = sum(torch.norm(p) ** 2 for p in net.parameters())
            loss_iter = loss_iter + lambda_ * l2_reg
        loss_iter.backward()
        with torch.no_grad():
            if mask and epoch>prune_epochs:
                    ## Go through all the parameters and set the pruned ones to zero
                for name, param in net.named_parameters():
                    param.data.mul_(optimizer.state[param]['mask'])
        loss += loss_iter.item()
        acc_dummy, acc5_dummy = torch_accuracy(pred, label, (1,5))
        acc += acc_dummy
        acc5 += acc5_dummy
        num_layers = len(list(net.parameters()))
        GVF = {}
        if (i+1)%accumulation_steps == 0:
            total_params = 0
            total_pruned = 0
            if warmup_epochs < epoch <= prune_epochs or epoch > prune_epochs:
                to_concat_g = []
                to_concat_v = []
                for name, param in net.named_parameters():
                    to_concat_g.append(param.grad.view(-1))
                    to_concat_v.append(param.data.view(-1))
                all_g = torch.cat(to_concat_g)
                all_v = torch.cat(to_concat_v)
                metric = torch.abs(all_g * all_v)
                Net_mask, GVF = Bias_Mask(metric)
                total_params = metric.numel()
                total_pruned = (Net_mask == 0).sum().item()
                
                '''Now we need to reshape the mask to match the parameter shapes'''
                start = 0
                for name, param in net.named_parameters():
                    if param.requires_grad and param.dim() in [2, 4]:
                        numel = param.numel()
                        param_mask = Net_mask[start:start+numel].view(param.shape).to(param.device)
                        param.grad.data.mul_(param_mask)
                        # Optionally: optimizer.state[param]['mask'] = param_mask
                        start += numel
            optimizer.step()
            optimizer.zero_grad()

            # Logging after pruning starts
            if epoch > warmup_epochs:
                elapsed = time.time() - start
                log = (
                    f"Epoch: {epoch}\nIteration: {i}\nElapsed Time: {elapsed:.4f}\n"
                    f"Total Params: {total_params}\nTotal Pruned: {total_pruned}\n"
                )
                if total_params > 0:
                    log += f"Percent Pruned: {total_pruned / total_params:.4f}\n"
                    log += f"Accuracy: {acc.item() / count:.4f}\n"
                jenks_logs.append(log + "\n")
        if (i+1) % print_steps == 0:
            # Write logs once per epoch
            if debug and debugfile:
                # print("Debug works")
                with open(debugfile, 'a') as f:
                    f.writelines(debug_logs)
            if jenksfile and epoch > warmup_epochs:
                with open(jenksfile, 'a') as f:
                    f.writelines(jenks_logs)
            debug_logs.clear()
            jenks_logs.clear()

    loss_avg = loss/ count
    acc_avg = acc / count
    acc5_avg = acc5 / count
    return acc_avg, acc5_avg, loss_avg


def Prune_Score(optimizer, kill_velocity=False, mask=False, mag_prune = False):
    ## Pass through the network and decide which weights to prune based on optimizer.state[param]['agg_score']
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.dim() == 4:  # Convolutional layer weights (4D tensor)
                # Iterate over each kernel (slice along the output channel dimension)
                agg_sal = []
                percent_pruned_list = []
                if 'agg_score' not in optimizer.state[param]:
                    print("agg_score not found for param")
                    break
                if mag_prune:
                    score = torch.abs(param.data)
                else:
                    score = optimizer.state[param]['agg_score']  # Access the agg_score for this kernel

                # Check for invalid values
                if torch.isnan(score).any() or torch.isinf(score).any():
                    print("Invalid values in score")
                    continue
                param_mask = Conv_Mask(score.contiguous())
                param_mask = param_mask.to(param.device)
                param.data = param_mask * param.data
                if param.grad is not None:
                    param.grad = param_mask * param.grad
                if hasattr(optimizer, "layerwise_lr_stats"):
                    if param.grad is not None:
                        stats = optimizer.layerwise_lr_stats.get(param, {})
                        stats['percent_pruned'] = mean(percent_pruned_list)
                        stats['saliency_std'] = torch.std(torch.cat(agg_sal)).item()
                        optimizer.layerwise_lr_stats[param] = stats    

            elif param.dim() == 2:  # Fully connected layer weights
                layer = param.data.flatten()
                if 'agg_score' not in optimizer.state[param]:
                    print("agg_score not found for param")
                    break
                if mag_prune:
                    score = layer
                else:
                    score = optimizer.state[param]['agg_score']
                WB_cuda_flatten = score.flatten()

                # Check for invalid values
                if torch.isnan(WB_cuda_flatten).any() or torch.isinf(WB_cuda_flatten).any():
                    print("Invalid values in WB_cuda_flatten")
                    continue

                WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                WB_cuda_sorted = WB_cuda_sorted.reshape(param.data.shape)
                var = module_weights.jenks_optimization_cuda(WB_cuda_sorted)
                var_min = var.argmin().item()
                indices_ = WB_cuda_indices[:var_min]
                if hasattr(optimizer, "layerwise_lr_stats"):
                    if param.grad is not None:
                        stats = optimizer.layerwise_lr_stats.get(param, {})
                        percent_pruned = len(indices_) / len(WB_cuda_flatten)
                        stats['percent_pruned'] = percent_pruned
                        stats['saliency_std'] = torch.std(WB_cuda_flatten).item()
                        optimizer.layerwise_lr_stats[param] = stats
                # Prune the layer
                layer[indices_] = 0
                if kill_velocity:
                    if 'velocity' in optimizer.state[param]:
                        optimizer.state[param]['velocity'].view(-1)[indices_] = 0
                if mask:
                    if 'mask' in optimizer.state[param]:
                        mask_dummy = torch.ones_like(layer, requires_grad=False)
                        mask_dummy[indices_] = 0
                        optimizer.state[param]['mask'] = mask_dummy.view(param.data.shape)
                    else:
                        print("Mask not found in optimizer state")
                param.data = layer.view(param.data.shape)

            elif param.dim() == 1:  # Bias terms
                layer = param.data
                if 'agg_score' not in optimizer.state[param]:
                    print("agg_score not found for param")
                    break
                if mag_prune:
                    score = layer
                else:
                    score = optimizer.state[param]['agg_score']
                B_cuda_sorted, B_cuda_indices = score.sort()
                var = module_bias.jenks_optimization_biases_cuda(B_cuda_sorted)
                var_min = var.argmin().item()
                indices_ = B_cuda_indices[:var_min]
                if hasattr(optimizer, "layerwise_lr_stats"):
                    if param.grad is not None:
                        stats = optimizer.layerwise_lr_stats.get(param, {})
                        percent_pruned = len(indices_) / len(B_cuda_sorted)
                        stats['percent_pruned'] = percent_pruned
                        stats['saliency_std'] = torch.std(score).item()
                        optimizer.layerwise_lr_stats[param] = stats
                # Prune the bias
                layer[indices_] = 0
                if kill_velocity:
                    if 'velocity' in optimizer.state[param]:
                        optimizer.state[param]['velocity'][indices_] = 0
                if mask:
                    if 'mask' in optimizer.state[param]:
                        optimizer.state[param]['mask'][indices_] = 0
                    else:
                        print("Mask not found in optimizer state")
                param.data = layer
            else:
                print("Invalid parameter dimension")
                continue

def Prune_Score_v2(optimizer, kill_velocity=False, mask=False, mag_prune = False, filter_based = True, bias_prune = True, prune_file = None):
    gc.collect()
    with torch.no_grad():
        for group in optimizer.param_groups:
            for param in group['params']:
                if mag_prune:
                    score = torch.abs(param.data)
                else:
                    score = optimizer.state[param]['agg_score']
                prune_mask, GVF = compute_mask(param, score, filter_based, bias_prune)
                print_prune_mask = prune_mask.cpu().tolist()
                if prune_file:
                    with open(prune_file, 'a') as f:
                        f.write(f"Layer: {param.shape}\n")
                        f.write(f"GVF: {GVF}\n")
                        f.write(f"Decay Count: {str(optimizer.state[param]['update_count'].cpu().tolist())}\n")
                        f.write(f"Prune Mask: {print_prune_mask}\n")
                        del print_prune_mask
                prune_mask = prune_mask.to(param.device)
                if param.grad is not None:
                    param.grad.mul_(prune_mask)
                param.data.mul_(prune_mask)
                if kill_velocity:
                    if 'velocity' in optimizer.state[param]:
                        optimizer.state[param]['velocity'].mul_(prune_mask)
                if mask:
                    if 'mask' in optimizer.state[param]:
                        optimizer.state[param]['mask'].mul_(prune_mask)
                    else:
                        print("Mask not found in optimizer state")
                if hasattr(optimizer, "layerwise_lr_stats"):
                    stats = optimizer.layerwise_lr_stats.get(param, {})
                    stats['percent_pruned'] = (prune_mask.numel() - prune_mask.sum().item()) / prune_mask.numel()
                    stats['saliency_std'] = torch.std(score).item()
                    optimizer.layerwise_lr_stats[param] = stats
                del score, prune_mask
    torch.cuda.empty_cache()


def Prune_Score_v3(net, optimizer, epoch, imp_layer_names = None, prune_epochs = None, kill_velocity=False, mask=False, mag_prune = False, filter_based = True, bias_prune = True, prune_file = None):
    gc.collect()
    with torch.no_grad():
        ''' Only prune the hidden layers
        The name of the first layer and last layer are in imp_layer_names'''
        prune_epoch = prune_epochs[0] if prune_epochs else 0
        for name, param in net.named_parameters():
            module = dict(net.named_modules()).get(name.rsplit('.', 1)[0], None)
            if (name not in imp_layer_names and epoch <= prune_epochs[-1]) or (epoch > prune_epochs[-1]):
                if mag_prune:
                    score = torch.abs(param.data)
                else:
                    score = optimizer.state[param]['agg_score']
                ## Turning off if module is not None and module.do_prune:
                if module is not None:
                    prune_mask, GVF = compute_mask(param, score, filter_based, bias_prune)
                    print_prune_mask = prune_mask.cpu().tolist()
                else:
                    prune_mask = torch.ones_like(param.data, requires_grad=False)
                    GVF = 1
                    print_prune_mask = [1] * param.numel()
                if prune_file:
                    with open(prune_file, 'a') as f:
                        f.write(f"Layer: {name}\n")
                        f.write(f"GVF: {GVF}\n")
                        f.write(f"Decay Count: {str(optimizer.state[param]['update_count'].cpu().tolist())}\n")
                        f.write(f"Prune Mask: {print_prune_mask}\n")
                        del print_prune_mask
                prune_mask = prune_mask.to(param.device)
                if param.grad is not None:
                    param.grad.mul_(prune_mask)
                param.data.mul_(prune_mask)
                if kill_velocity:
                    if 'velocity' in optimizer.state[param]:
                        optimizer.state[param]['velocity'].mul_(prune_mask)
                if mask:
                    if 'mask' in optimizer.state[param]:
                        optimizer.state[param]['mask'] = prune_mask
                    else:
                        print("Mask not found in optimizer state")
                if hasattr(optimizer, "layerwise_lr_stats"):
                    stats = optimizer.layerwise_lr_stats.get(param, {})
                    stats['percent_pruned'] = (prune_mask.numel() - prune_mask.sum().item()) / prune_mask.numel()
                    stats['saliency_std'] = torch.std(score).item()
                    optimizer.layerwise_lr_stats[param] = stats
                del score, prune_mask
    torch.cuda.empty_cache()

def Prune_Score_Reset(net, optimizer, epoch, imp_layer_names = None, prune_epochs = None, kill_velocity=False, mask=False, mag_prune = False, filter_based = True, bias_prune = True, prune_file = None):
    '''This Pruning function will generate the pruning mask, but reinitialize the weights after pruning
    This way it allows us to start training from scratch after pruning'''
    gc.collect()
    with torch.no_grad():
        ''' Only prune the hidden layers
        The name of the first layer and last layer are in imp_layer_names'''
        prune_epoch = prune_epochs[0] if prune_epochs else 0
        for name, param in net.named_parameters():
            module = dict(net.named_modules()).get(name.rsplit('.', 1)[0], None)
            if (name not in imp_layer_names and epoch <= prune_epochs[-1]) or (epoch > prune_epochs[-1]):
                if mag_prune:
                    score = torch.abs(param.data)
                else:
                    score = optimizer.state[param]['agg_score']
                if module is not None and module.do_prune:
                    prune_mask, GVF = compute_mask(param, score, filter_based, bias_prune)
                    print_prune_mask = prune_mask.cpu().tolist()
                else:
                    prune_mask = torch.ones_like(param.data, requires_grad=False)
                    GVF = 1
                    print_prune_mask = [1] * param.numel()
                if prune_file:
                    with open(prune_file, 'a') as f:
                        f.write(f"Layer: {name}\n")
                        f.write(f"GVF: {GVF}\n")
                        f.write(f"Decay Count: {str(optimizer.state[param]['update_count'].cpu().tolist())}\n")
                        f.write(f"Prune Mask: {print_prune_mask}\n")
                        del print_prune_mask
                prune_mask = prune_mask.to(param.device)
                if param.grad is not None:
                    param.grad.mul_(prune_mask)
                param.data.mul_(prune_mask)
                if kill_velocity:
                    if 'velocity' in optimizer.state[param]:
                        optimizer.state[param]['velocity'].mul_(prune_mask)
                if mask:
                    if 'mask' in optimizer.state[param]:
                        optimizer.state[param]['mask'] = prune_mask
                    else:
                        print("Mask not found in optimizer state")
                if hasattr(optimizer, "layerwise_lr_stats"):
                    stats = optimizer.layerwise_lr_stats.get(param, {})
                    stats['percent_pruned'] = (prune_mask.numel() - prune_mask.sum().item()) / prune_mask.numel()
                    stats['saliency_std'] = torch.std(score).item()
                    optimizer.layerwise_lr_stats[param] = stats
                del score, prune_mask
    torch.cuda.empty_cache()
    '''Now reinitialize the weights'''
    for name, param in net.named_parameters():
        '''Utilize xavier initialization for weights and zeros for biases'''
        if param.requires_grad and param.dim() in [2, 4]:
            if isinstance(param, torch.nn.Parameter):
                torch.nn.init.xavier_uniform_(param.data) 
            else:
                torch.nn.init.xavier_uniform_(param)
        elif param.requires_grad and param.dim() == 1:
            if isinstance(param, torch.nn.Parameter):
                torch.nn.init.zeros_(param.data)
            else:
                torch.nn.init.zeros_(param)
    for name, param in net.named_parameters():
        if mask:
            if 'mask' in optimizer.state[param]:
                param.data.mul_(optimizer.state[param]['mask'])

def Prune_Score_Global(net,optimizer, kill_velocity=False, mask=False, prune_file = None):
    gc.collect()
    with torch.no_grad():
        to_concat_v = []
        for name, param in net.named_parameters():
            to_concat_v.append(param.data.view(-1))
        all_v = torch.cat(to_concat_v)
        mask, GVF = Bias_Mask(all_v)
        '''Now we need to reshape the mask to match the parameter shapes'''
        start = 0
        for name, param in net.named_parameters():
            if param.requires_grad and param.dim() in [2, 4]:
                numel = param.numel()
                param_mask = mask[start:start+numel].view(param.shape).to(param.device)
                param.data.mul_(param_mask)
                # Optionally: optimizer.state[param]['mask'] = param_mask
                start += numel
            if param.grad is not None:
                param.grad.mul_(prune_mask)
            if kill_velocity:
                if 'velocity' in optimizer.state[param]:
                    optimizer.state[param]['velocity'].mul_(prune_mask)
            if mask:
                if 'mask' in optimizer.state[param]:
                    optimizer.state[param]['mask']=prune_mask
                else:
                    print("Mask not found in optimizer state")
            if hasattr(optimizer, "layerwise_lr_stats"):
                stats = optimizer.layerwise_lr_stats.get(param, {})
                stats['percent_pruned'] = (prune_mask.numel() - prune_mask.sum().item()) / prune_mask.numel()
                stats['saliency_std'] = torch.std(score).item()
                optimizer.layerwise_lr_stats[param] = stats
            del score, prune_mask
            if prune_file is not None:
                with open(prune_file, 'a') as f:
                    f.write(f"Layer: {name}\n")
                    f.write(f"GVF: {GVF}\n")
                    '''Calculate the percentage of pruned weights'''
                    percent_pruned = (mask.numel() - mask.sum().item()) / mask.numel()
                    f.write(f"Percent Pruned: {percent_pruned}\n")
                    f.write("\n")
    # Clear the cache to free up memory
    gc.collect()
    torch.cuda.empty_cache()



def Prune_Score_Mag(optimizer):
    ## Pass through the network and decide which weights to prune based on optimizer.state[param]['agg_score']
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.dim() in [2, 4]:   
                WB_cuda_flatten = param.data.flatten()
                # print(f"WB_cuda_flatten shape: {WB_cuda_flatten.shape}")

                # Check for invalid values
                if torch.isnan(WB_cuda_flatten).any() or torch.isinf(WB_cuda_flatten).any():
                    print("Invalid values in WB_cuda_flatten")
                    continue

                WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                # print(f"WB_cuda_sorted shape: {WB_cuda_sorted.shape}")
                # print(f"WB_cuda_indices shape: {WB_cuda_indices.shape}")
                WB_cuda_sorted = WB_cuda_sorted.reshape(param.data.shape)

                var = module_weights.jenks_optimization_cuda(WB_cuda_sorted)
                var_min = var.argmin().item()

                # Validate var_min
                if var_min <= 0 or var_min > WB_cuda_indices.size(0):
                    print(f"Invalid var_min: {var_min}")
                    continue

                indices_ = WB_cuda_indices[:var_min]
                param.data.view(-1)[indices_] = 0
            elif param.dim() == 1:
                score = param.data
                B_cuda_sorted, B_cuda_indices = score.sort()
                var = module_bias.jenks_optimization_biases_cuda(B_cuda_sorted)
                var_min = var.argmin().item()
                # Print the output
                indices_ = B_cuda_indices[:var_min]
                param.data.view(-1)[indices_] = 0
            else:
                print("Invalid parameter dimension")
                continue
        

def Prune_Score_Select(optimizer, prune_ratio = .95, kill_velocity=False, mask=False):
    '''In this function, we will still be using Jenk's natural break to find layerwise saliency scores and their break
        however, we will use the layerwise sparsity from the natural break to guide the selection of which weights to prune
        for example, say x weights pruned are from layer1.weights, and the overall pruning ratio is only .9, then we will have to prune
        (9.5/9)*x of the lowest saliency scores in layer1.weights. This will ensure if it is underpruning or overpruning, we can still have
        some semblance of control.
    
    '''
    pruned_val = []
    sorted_idx = []
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.dim() == 4:  # Convolutional layer weights (4D tensor)
                # Iterate over each kernel (slice along the output channel dimension)
                agg_sal = []
                percent_pruned_list = []
                for kernel_idx in range(param.shape[0]):
                    kernel = param.data[kernel_idx]  # Access the kernel (3D tensor)
                    if 'agg_score' not in optimizer.state[param]:
                        print("agg_score not found for param")
                        break
                    score = optimizer.state[param]['agg_score'][kernel_idx]  # Access the agg_score for this kernel
                    WB_cuda_flatten = score.flatten()
                    agg_sal.append(WB_cuda_flatten)
                    # Check for invalid values
                    if torch.isnan(WB_cuda_flatten).any() or torch.isinf(WB_cuda_flatten).any():
                        print("Invalid values in WB_cuda_flatten")
                        continue

                    WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                    WB_cuda_sorted = WB_cuda_sorted.reshape(kernel.shape)
                    var = module_weights.jenks_optimization_cuda(WB_cuda_sorted)
                    if torch.isnan(var).any() or torch.isinf(var).any() or var.numel() == 0 or var.shape[0] == 0:
                        print("Invalid values in var")
                        return

                    var_min = var.argmin().item()
                    indices_ = WB_cuda_indices[:var_min]
                    pruned_val.append(len(indices_))
                    sorted_idx.append(WB_cuda_indices)

            elif param.dim() == 2:  # Fully connected layer weights
                layer = param.data.flatten()
                if 'agg_score' not in optimizer.state[param]:
                    print("agg_score not found for param")
                    break
                score = optimizer.state[param]['agg_score']
                WB_cuda_flatten = score.flatten()

                # Check for invalid values
                if torch.isnan(WB_cuda_flatten).any() or torch.isinf(WB_cuda_flatten).any():
                    print("Invalid values in WB_cuda_flatten")
                    continue

                WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                WB_cuda_sorted = WB_cuda_sorted.reshape(param.data.shape)
                var = module_weights.jenks_optimization_cuda(WB_cuda_sorted)
                var_min = var.argmin().item()
                indices_ = WB_cuda_indices[:var_min]
                pruned_val.append(len(indices_))
                sorted_idx.append(WB_cuda_indices)

            elif param.dim() == 1:  # Bias terms
                layer = param.data
                if 'agg_score' not in optimizer.state[param]:
                    print("agg_score not found for param")
                    break
                score = optimizer.state[param]['agg_score']
                B_cuda_sorted, B_cuda_indices = score.sort()
                var = module_bias.jenks_optimization_biases_cuda(B_cuda_sorted)
                var_min = var.argmin().item()
                indices_ = B_cuda_indices[:var_min]
                pruned_val.append(len(indices_))
                sorted_idx.append(WB_cuda_indices)
            else:
                print("Invalid parameter dimension")
                continue
    # Calculate the total number of weights to prune
    pruned_weights = sum(pruned_val)
    # Calculate the total number of weights in the model
    total_weights = sum([param.numel() for group in optimizer.param_groups for param in group['params']])
    # Calculate the target number of weights to prune based on the pruning ratio
    current_prune_ratio = pruned_weights / total_weights
    target_prune_ratio = prune_ratio
    for i,value in enumerate(pruned_val):
        pruned_val[i] = int((target_prune_ratio/current_prune_ratio) * value)
    ## Pass back through the network and decide which weights to prune based on optimizer.state[param]['agg_score']
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.dim() == 4:
                # Iterate over each kernel (slice along the output channel dimension)
                for kernel_idx in range(param.shape[0]):
                    kernel = param.data[kernel_idx]
                    kernel_flat = kernel.view(-1)
                    indices_ = sorted_idx[0][:pruned_val[0]]
                    sorted_idx.pop(0)
                    pruned_val.pop(0)
                    if indices_.max() >= kernel_flat.numel() or indices_.min() < 0 or indices_.numel()>= kernel_flat.numel():
                        print(f"indices_ max: {indices_.max()}, indices_ min: {indices_.min()}")
                        print(f"Size of indices_: {indices_.size()}")
                        print(f"Invalid indices_: {indices_}")
                        continue
                    print(f"indices_ max: {indices_.max()}, indices_ min: {indices_.min()}")
                    print(f"Size of indices_: {indices_.size()}")
                    print(f"Invalid indices_: {indices_}")
                    print(f"kernel_flat size: {kernel_flat.size()}")
                    kernel_flat[indices_] = 0
                    param[kernel_idx].data = kernel_flat.view(kernel.shape)
                    if kill_velocity:
                        if 'velocity' in optimizer.state[param]:
                            optimizer.state[param]['velocity'][kernel_idx].view(-1)[indices_] = 0
                    if mask:
                        if 'mask' in optimizer.state[param]:
                            mask_dummy = torch.ones_like(kernel_flat, requires_grad=False)
                            mask_dummy[indices_] = 0
                            optimizer.state[param]['mask'][kernel_idx] = mask_dummy.view(kernel.shape)
                        else:
                            print("Mask not found in optimizer state")
            elif param.dim() == 2:
                layer = param.data.flatten()
                indices_ = sorted_idx[0][:pruned_val[0]]
                pruned_val.pop(0)
                sorted_idx.pop(0)
                layer[indices_] = 0
                param.data = layer.view(param.data.shape)
                if kill_velocity:
                    if 'velocity' in optimizer.state[param]:
                        optimizer.state[param]['velocity'].view(-1)[indices_] = 0
                if mask:
                    if 'mask' in optimizer.state[param]:
                        mask_dummy = torch.ones_like(layer, requires_grad=False)
                        mask_dummy[indices_] = 0
                        optimizer.state[param]['mask'] = mask_dummy.view(param.data.shape)
                    else:
                        print("Mask not found in optimizer state")
            elif param.dim() == 1:
                layer = param.data
                indices_ = sorted_idx[0][:pruned_val[0]]
                pruned_val.pop(0)
                sorted_idx.pop(0)
                layer[indices_] = 0
                param.data = layer
                if kill_velocity:
                    if 'velocity' in optimizer.state[param]:
                        optimizer.state[param]['velocity'][indices_] = 0
                if mask:
                    if 'mask' in optimizer.state[param]:
                        optimizer.state[param]['mask'][indices_] = 0
                    else:
                        print("Mask not found in optimizer state")
            else:
                print("Invalid parameter dimension")
                continue
            
                





class JenksSGD_Noise(Optimizer):
    def __init__(self, params, lr=0.01, scale=0.9, momentum=0.9):
        defaults = dict(lr=lr, scale=scale, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            scale = group['scale']
            momentum = group['momentum']

            for param in group['params']:
                if param.grad is None:
                    continue

                # Initialize velocity if not already done
                if 'velocity' not in self.state[param]:
                    self.state[param]['velocity'] = torch.zeros_like(param.data)

                velocity = self.state[param]['velocity']

                # Check if the parameter is a weight matrix or bias vector
                if len(param.shape) > 1:  # Assuming weight matrices have more than 1 dimension
                    # Custom weight update: Scale gradients before applying update
                    # print(torch.mul(param, param.grad).cpu().numpy())
                    s_W = torch.mul(param, param.grad)  # Move to CPU, convert to NumPy, and flatten
                    s_W = torch.abs(s_W)
                    unique_values = torch.unique(s_W)
                    n_classes = min(2, len(unique_values))  # Ensure n_classes is valid
                    if n_classes > 1:
                        # jnb = JenksNaturalBreaks(n_classes)
                        # jnb.fit(s_W)
                        # labels = jnb.labels_
                        # indices = np.where(labels == 1)[0]
                        # indices_ = np.where(labels == 0)[0]

                        # # Update velocity
                        velocity_flat = velocity.view(-1)
                        param_data_flat = param.data.view(-1)
                        param_grad_flat = param.grad.data.view(-1)
                        WB_cuda_flatten = s_W.flatten()
                        WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                        WB_cuda_sorted = WB_cuda_sorted.reshape(s_W.shape)

                        var = module_weights.jenks_optimization_cuda(WB_cuda_sorted)
                        var_min = var.argmin().item()

                        indices_ = WB_cuda_indices[:var_min]
                        indices = WB_cuda_indices[var_min:]

                        velocity_flat[indices] = momentum * velocity_flat[indices] + scale * param_data_flat[indices] + param_grad_flat[indices]
                        velocity_flat[indices_] = momentum * velocity_flat[indices_] + scale * param_data_flat[indices_]

                        # Update parameters
                        n_params = len(param_data_flat)
                        noise = torch.randn(n_params).to('cuda')
                        param_data_flat.add(noise)
                        param_data_flat[indices] -= lr * velocity_flat[indices]
                        param_data_flat[indices_] -= lr * velocity_flat[indices_]
                        param.data = param_data_flat.view(param.data.shape)
                        self.state[param]['velocity'] = velocity_flat.view(velocity.shape)
                    else:
                        velocity = momentum * velocity + scale * param.grad
                        param.data -= lr * velocity
                else:  # Assuming bias vectors have 1 dimension
                    s_B = torch.mul(param, param.grad)  # Move to CPU, convert to NumPy, and flatten
                    s_B = torch.abs(s_B)
                    unique_values = torch.unique(s_B)
                    n_classes = min(2, len(unique_values))  # Ensure n_classes is valid
                    if n_classes > 1:
                        # jnb = JenksNaturalBreaks(n_classes)
                        # jnb.fit(s_B)
                        # labels = jnb.labels_
                        # indices = np.where(labels == 1)[0]
                        # indices_ = np.where(labels == 0)[0]
                        B_cuda_sorted, B_cuda_indices = s_B.sort()
                        var = module_bias.jenks_optimization_biases_cuda(B_cuda_sorted)
                        var_min = var.argmin().item()
                        # Print the output
                        indices_ = B_cuda_indices[:var_min]
                        indices = B_cuda_indices[var_min:]
                        # Update velocity
                        velocity_flat = velocity.view(-1)
                        param_data_flat = param.data.view(-1)
                        param_grad_flat = param.grad.data.view(-1)

                        velocity_flat[indices] = momentum * velocity_flat[indices] + scale * param_data_flat[indices] + param_grad_flat[indices]
                        velocity_flat[indices_] = momentum * velocity_flat[indices_] + scale * param_data_flat[indices_]

                        # Update parameters
                        n_params = len(param_data_flat)
                        noise = torch.randn(n_params).to('cuda')
                        param_data_flat.add(noise)
                        param_data_flat[indices] -= lr * velocity_flat[indices]
                        param_data_flat[indices_] -= lr * velocity_flat[indices_]
                        param.data = param_data_flat.view(param.data.shape)
                        self.state[param]['velocity'] = velocity_flat.view(velocity.shape)
                        
                    else:
                        velocity = momentum * velocity + scale * param.grad
                        param.data -= lr * velocity
        return loss


# import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        # for group in self.param_groups:
        #     for p in group["params"]:
        #         if p.grad is None: print("Warning: gradient is None")
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


'''Now we need to also have a function in which we take the network and prune based upon magnitude of the weights.'''





def PruneWeights(model):
    # Get the weights of the model, save in different layers
    jnb = JenksNaturalBreaks(2)
    for param in model.parameters():
        layer = param.data.cpu().numpy()
        layer = layer.flatten()
        layer_abs = np.abs(layer)
        jnb.fit(layer_abs)
        labels = jnb.labels_
        indices_ = np.where(labels == 0)[0]
        indices = np.where(labels == 1)[0]
        layer[indices_] = 0
        layer = layer.reshape(param.data.shape)
        param.data = torch.from_numpy(layer)
        param = param.to('cuda')
    return model

def PruneWeights_Test(model, optimizer):
    jnb = JenksNaturalBreaks(2)
    for param in model.parameters():
        layer = param.data.cpu().numpy()
        layer = layer.flatten()
        if 'agg_score' in optimizer.state[param]:
            print(f"agg_score for param: {optimizer.state[param]['agg_score']}")
        else:
            print("agg_score not found for param")
            break
        score = optimizer.state[param]['agg_score']
        layer_abs = np.abs(score.cpu().numpy())
        jnb.fit(layer_abs)
        labels = jnb.labels_
        indices_ = np.where(labels == 0)[0]
        indices = np.where(labels == 1)[0]
        layer[indices_] = 0
        layer = layer.reshape(param.data.shape)
        param.data = torch.from_numpy(layer)
        param = param.to('cuda')
    return model

def train_one_step_prune_HPO(net, dataloader, optimizer, criterion, epoch, warmup_epochs, prune_epochs, no_jenks = False, bias_prune = True, filter_based = True, mask = False, L2 = False,lambda_ = 0.01, debug = False, debugfile = None, jenksfile = None, mag=False, elem_bias = False, accumulation_steps = 1):
    debug_logs = []
    jenks_logs = []
    align_check = True
    accumulation_steps = 1
    print_steps = 10
    optimizer.zero_grad()
    count = 0
    loss = 0
    acc = 0
    acc5 = 0
    device = next(net.parameters()).device
    '''We need to compute the gradient mask and apply it here, but for 
            experimentation, we need to collect the gradients and the weights before and after update
            We will then compute the dot product between them to see how much they align'''
    weights_before = {}
    weights_after = {}
    gradients_before = {}
    gradients_after = {}
    velocity = {}
    num_layers = len(list(net.parameters()))
    if elem_bias:
        optimizer.epoch = epoch
    # Set up MixUp/CutMix if enabled. num_classes is read from the model's final
    # Linear layer so it works for CIFAR-10/100/TinyImageNet without hardcoding.
    use_mixup = MIXUP and epoch <= MIXUP_OFF_EPOCH
    if use_mixup:
        n_classes = None
        for _m in net.modules():
            if isinstance(_m, torch.nn.Linear):
                n_classes = _m.out_features
        cutmix_or_mixup = RandomChoice([CutMix(num_classes=n_classes), MixUp(num_classes=n_classes)])
    for i, (data, label) in enumerate(dataloader):
        torch.cuda.empty_cache()
        count+=1
        start = time.time()
        data,label = data.to(device), label.to(device)
        label_hard = label  # keep integer labels for accuracy; mixup makes label soft
        if use_mixup:
            data, label = cutmix_or_mixup(data, label)
        pred = net(data)
        loss_iter = criterion(pred, label)
        if L2:
            l2_reg = sum(torch.norm(p) ** 2 for p in net.parameters())
            loss_iter = loss_iter + lambda_ * l2_reg
        loss_iter.backward()
        apply_gradient_centralization(net)
        with torch.no_grad():
            if mask and epoch>prune_epochs:
                    ## Go through all the parameters and set the pruned ones to zero
                for name, param in net.named_parameters():
                    param.data.mul_(optimizer.state[param]['mask'])
        loss += loss_iter.item()
        acc_dummy, acc5_dummy = torch_accuracy(pred, label_hard, (1,5))
        acc += acc_dummy
        acc5 += acc5_dummy
        num_layers = len(list(net.parameters()))
        GVF = {}
        
        total_params = 0
        total_pruned = 0
        gradient_norm = 0
        gradient_norm_masked = 0
        if warmup_epochs < epoch <= prune_epochs or epoch > prune_epochs:
            if elem_bias == False:
                for name, param in net.named_parameters():
                    layer_pruned = 0
                    layer_params = 0
                    # Collect initial magnitude of gradient
                    weights_before[name] = param.data.clone().detach()
                    gradient_norm += torch.norm(param.grad).item()
                    # Access momentum buffer (velocity) if it exists
                    velocity[name] = optimizer.state[param].get('momentum_buffer', None)
                    gradients_before[name] = param.grad.clone().detach()
                    # if velocity is not None:
                    #     # You can use velocity here, e.g. print or log it
                    #     pass  # Replace with your logic
                    if mag:
                        sign_WB = param.data
                    else:
                        sign_WB = param.data * param.grad
                    WB = torch.abs(sign_WB)
                    # print(WB.shape)
                    ## Generate a binary matrix, b_ij = 1 if sign_WB_ij<0 else 0
                    ## This will help us identify which weights to prune
                    decay_mask = torch.ones_like(WB, requires_grad=False)
                    decay_mask[sign_WB > 0] = 0
                    if 'bn' not in name:
                        mask_tensor, GVF_val = compute_mask(param, WB, filter_based, bias_prune)
                    else:
                        mask_tensor = torch.ones_like(param.data, requires_grad=False)
                        GVF_val = 1
                    mask_tensor = mask_tensor.to(device)
                    decay_mask = decay_mask.to(device)
                    decay_mask *= mask_tensor
                    # Find the param group for this param and get its weight_decay
                    weight_decay = 0
                    for group in optimizer.param_groups:
                        if 'weight_decay' in group:
                            weight_decay = group['weight_decay']
                            break
                    # print(decay_mask.shape)
                    # print(param.data.shape)
                    # print(weight_decay)
                    decay_mask *= weight_decay * param.data
                    optimizer.state[param]['velocity']-=decay_mask
                    decay_tensor = torch.ones_like(mask_tensor, requires_grad=False, device=device)
                    decay_tensor.sub_(mask_tensor)
                    optimizer.state[param]['update_count'].add_(decay_tensor)
                    del decay_tensor
                    GVF[name] = GVF_val
                    with torch.no_grad():
                        if epoch > warmup_epochs:
                            if not no_jenks:
                                param.grad.mul_(mask_tensor)
                                gradients_after[name] = param.grad.clone().detach()
                                gradient_norm_masked += torch.norm(param.grad).item()
                                WB_prime = WB * mask_tensor
                                optimizer.state[param]['agg_score'] += WB_prime
                            else:
                                WB_prime = WB
                                optimizer.state[param]['agg_score'] += WB_prime
                        if param.dim() == 1:
                            if bias_prune:
                                total_params += mask_tensor.numel()
                                total_pruned += (mask_tensor.numel() - mask_tensor.sum().item())
                            else:
                                continue
                        elif param.dim() in [2, 4]:
                            total_params += mask_tensor.numel()
                            total_pruned += (mask_tensor.numel() - mask_tensor.sum().item())
                        layer_params += mask_tensor.numel()
                        layer_pruned += (mask_tensor.numel() - mask_tensor.sum().item())
                        # Debug stats
                        if name and hasattr(optimizer, "layerwise_lr_stats"):
                            stats = optimizer.layerwise_lr_stats.get(name, {})
                            stats['percent_pruned'] = layer_pruned / layer_params
                            stats['saliency_std'] = torch.std(WB).item()
                            optimizer.layerwise_lr_stats[name] = stats
                            # print(f"Debug is on: {debug}")
                            if debug:
                                # debug_logs.append(
                                #     f"Layer Name: {name}\nPercent Pruned: {stats['percent_pruned']:.4f}\nSaliency Std: {stats['saliency_std']:.4f}\n"
                                # )
                                if isinstance(GVF[name],list):
                                    debug_logs.append(f"Layer Name: {name}\nGVF Values: {GVF[name]}\n")
                                else:
                                    debug_logs.append(f"Layer Name: {name}\nGVF Value: {GVF[name]:.4f}\n")

        optimizer.step()
        optimizer.zero_grad()
        if epoch > prune_epochs and mask:
            for name, param in net.named_parameters():
                param.data.mul_(optimizer.state[param]['mask'])
        ## Get the weights after the step
        if epoch > warmup_epochs:
            elapsed = time.time() - start
            log = (
                f"Epoch: {epoch}\nIteration: {i}\nElapsed Time: {elapsed:.4f}\n"
                f"Total Params: {total_params}\nTotal Pruned: {total_pruned}\n"
            )
            if total_params > 0:
                log += f"Percent Pruned: {total_pruned / total_params:.4f}\n"
                log += f"Accuracy: {acc.item() / count:.4f}\n"
                log += f"Initial Gradient Norm: {gradient_norm:.4f}\n"
                log += f"Masked Gradient Norm: {gradient_norm_masked:.4f}\n"
                jenks_logs.append(log + "\n")
        if (i+1) % print_steps == 0:
            # Write logs once per epoch
            if debug and debugfile:
                # print("Debug works")
                with open(debugfile, 'a') as f:
                    f.writelines(debug_logs)
            if jenksfile and epoch > warmup_epochs:
                with open(jenksfile, 'a') as f:
                    f.writelines(jenks_logs)
            debug_logs.clear()
            jenks_logs.clear()

    loss_avg = loss/ count
    acc_avg = acc / count
    acc5_avg = acc5 / count
    return acc_avg, acc5_avg, loss_avg


class ElementwiseMomentumSGD(Optimizer):
    def __init__(self, params, name_map, do_prune_map, lr=0.01, weight_decay=0.0, momentum=0.9, warmup_epochs=10, pruning_epochs =150, mag = False, device = 'cuda', filter_based = True, bias_prune = True):
        defaults = dict(lr=lr, weight_decay=weight_decay,momentum=momentum)
        self.mag = mag
        self.device = device
        self.filter_based = filter_based
        self.bias_prune = bias_prune
        self.name_map = name_map
        self.epoch = 0
        self.warmup_epochs = warmup_epochs
        self.do_prune_map = do_prune_map
        self.pruning_epochs = pruning_epochs
        super(ElementwiseMomentumSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        if self.epoch < self.warmup_epochs:
            for group in self.param_groups:
                lr = group['lr']
                momentum = group['momentum']
                weight_decay = group['weight_decay']
                lr_t = torch.tensor(lr, device=self.device)
                for param in group['params']:
                    if param.grad is None:
                        continue
                    # Initialize velocity if not already done
                    if 'velocity' not in self.state[param]:
                        self.state[param]['velocity'] = torch.zeros_like(param.data)
                    eig_hess = torch.norm(param.grad)**2
                    velocity = self.state[param]['velocity']
                    sal_beta = (1 - torch.sqrt(lr_t * eig_hess))**2
                    # Update velocity
                    velocity.mul_(sal_beta).add_(weight_decay * param.data + param.grad)

                    # Update parameters
                    param.data -= lr * velocity
            return loss
        elif self.epoch >= self.warmup_epochs and self.epoch <= self.pruning_epochs:
            total_params = 0
            total_pruned = 0
            for group in self.param_groups:
                lr = group['lr']
                momentum = group['momentum']
                weight_decay = group['weight_decay']
                lr_t = torch.tensor(lr, device=self.device)
                wd_t = torch.tensor(weight_decay, device=self.device)
                # print(f"Weight Decay in step: {weight_decay}")
                # print(f"Learning Rate in step: {lr}")
                layer_params = 0
                layer_pruned = 0
                for param in group['params']:
                    name = self.name_map.get(param, None)
                # Get the module name (if needed, adjust this logic to match your do_prune_map keys)
                    module_name = name.rsplit('.', 1)[0] if name and '.' in name else name
                    do_prune_layer = self.do_prune_map.get(module_name, True)
                    if do_prune_layer:
                        if param.grad is None:
                            continue
                        if self.mag:
                            sign_WB = param.data
                        else:
                            with torch.no_grad():
                                sign_WB = param.data * param.grad
                        WB = torch.abs(sign_WB)
                        # print(WB.shape)
                        ## Generate a binary matrix, b_ij = 1 if sign_WB_ij<0 else 0
                        ## This will help us identify which weights to prune
                        decay_mask = torch.ones_like(WB, requires_grad=False)
                        decay_mask[sign_WB > 0] = 0
                        if 'bn' not in self.name_map[param]:
                            mask_tensor, GVF_val = compute_mask(param, WB, self.filter_based, self.bias_prune)
                        else:
                            mask_tensor = torch.ones_like(param.data, requires_grad=False)
                            GVF_val = 1
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                        mask_tensor = mask_tensor.to(self.device)
                        decay_mask = decay_mask.to(self.device)
                        decay_mask *= mask_tensor
                        ## If there is a zero in the mask tensor, then the beta is equal to (1-\sqrt{lr*weight_decay})^2
                        ## OTherwise, beta = (1-\sqrt{lr*h})^2 where h is an approximation of the curvature
                        ## h = ||param.grad||^2_2
                        eig_hess = torch.norm(param.grad*mask_tensor)**2
                        unsal_beta = (1 - torch.sqrt(lr_t * wd_t))**2
                        sal_beta = (1 - torch.sqrt(lr_t * eig_hess))**2
                        beta_tensor = torch.ones_like(mask_tensor, requires_grad=False)
                        beta_tensor[mask_tensor==1] = sal_beta
                        beta_tensor[mask_tensor==0] = unsal_beta
                        param.grad.mul_(mask_tensor)
                        update = weight_decay * param.data + param.grad
                        if 'velocity' not in self.state[param]:
                            self.state[param]['velocity'] = torch.zeros_like(param.data)

                        velocity = self.state[param]['velocity']

                        # Update velocity
                        velocity.mul_(beta_tensor).add_(update)

                    else:
                        if param.grad is None:
                            continue
                        if 'velocity' not in self.state[param]:
                            self.state[param]['velocity'] = torch.zeros_like(param.data)
                        eig_hess = torch.norm(param.grad)**2
                        velocity = self.state[param]['velocity']
                        sal_beta = (1 - torch.sqrt(lr_t * eig_hess))**2
                        # Update velocity
                        velocity.mul_(sal_beta).add_(weight_decay * param.data + param.grad)

                    # decay_mask *= weight_decay * param.data
                    # self.state[param]['momentum_buffer']-=decay_mask
                    # Initialize velocity if not already done
                    # Update parameters
                    param.data -= lr * velocity
                    if do_prune_layer:
                        if param.dim() == 1:
                            if self.bias_prune:
                                total_params += mask_tensor.numel()
                                total_pruned += (mask_tensor.numel() - mask_tensor.sum().item())
                            else:
                                continue
                        elif param.dim() in [2, 4]:
                            total_params += mask_tensor.numel()
                            total_pruned += (mask_tensor.numel() - mask_tensor.sum().item())
                        layer_params += mask_tensor.numel()
                        layer_pruned += (mask_tensor.numel() - mask_tensor.sum().item())
                        # Debug stats
                        if self.name_map[param] and hasattr(self, "layerwise_lr_stats"):
                            stats = self.layerwise_lr_stats.get(self.name_map[param], {})
                            stats['percent_pruned'] = total_pruned / total_params
                            stats['saliency_std'] = torch.std(WB).item()
                            self.layerwise_lr_stats[self.name_map[param]] = stats

            return loss
        else:
            for group in self.param_groups:
                lr = group['lr']
                momentum = group['momentum']
                weight_decay = group['weight_decay']
                lr_t = torch.tensor(lr, device=self.device)
                for param in group['params']:
                    if param.grad is None:
                        continue
                    # Initialize velocity if not already done
                    if 'velocity' not in self.state[param]:
                        self.state[param]['velocity'] = torch.zeros_like(param.data)
                    param.grad.mul_(self.state[param]['mask'])
                    eig_hess = torch.norm(param.grad)**2
                    velocity = self.state[param]['velocity']
                    sal_beta = (1 - torch.sqrt(lr_t * eig_hess))**2
                    # Update velocity
                    velocity.mul_(sal_beta).add_(weight_decay * param.data + param.grad)

                    # Update parameters
                    param.data -= lr * velocity
            return loss
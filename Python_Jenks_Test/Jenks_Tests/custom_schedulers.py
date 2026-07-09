# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from bisect import bisect_right
# from dataset import num_iters_per_epoch
import torch
from math import sqrt
from math import log
from custom_optimizer import ElementwiseMomentumSGD

# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

def singular_value(p):
    if p.dim() < 2:
        return 1.0
    sv = sqrt(p.shape[0] / p.shape[1])
    if p.dim() == 4:
        sv /= sqrt(p.shape[2] * p.shape[3])
    return sv



class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

class WarmupMultiStepJenks(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        alpha=0.25,     # Custom scaling factor for pruning-based adjustment, last set to .5. Init .25
        beta=0.1,      # Optionally add saliency std as another factor, last set to 0. Init .1
        last_epoch=-1,
        adjustable = False,
        cosine = False
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                f"Milestones should be a list of increasing integers. Got {milestones}"
            )
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted, got {}".format(warmup_method)
            )

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.alpha = alpha
        self.beta = beta
        self.adjustable = adjustable  # Whether to adjust learning rate based on pruning/saliency
        super(WarmupMultiStepJenks, self).__init__(optimizer, last_epoch)

    def get_lr(self, epoch=None, metric=None):
        warmup_factor = 1.0
        if epoch is not None:
            self.last_epoch = epoch
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [
                base_lr
                * warmup_factor
                * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            scaled_lrs = []
            for group in self.optimizer.param_groups:
                name = group.get("name", None)
                base_lr = group['initial_lr'] if 'initial_lr' in group else group['lr']
                milestone_scale = self.gamma ** bisect_right(self.milestones, self.last_epoch)

                # Fetch param group name
                name = group.get("name", None)
                if self.adjustable:
                    if name and hasattr(self.optimizer, "layerwise_lr_stats"):
                        stats = self.optimizer.layerwise_lr_stats.get(name, {})
                        percent_pruned = stats.get('percent_pruned', 0.0)
                        saliency_std = stats.get('saliency_std', 0.0)

                        # Custom scaling logic
                        dynamic_scale = 1.0 + self.alpha * percent_pruned + self.beta * saliency_std
                        if "weight_decay" in group:
                            group["weight_decay"] *= (1-percent_pruned)**self.alpha
                else:
                    dynamic_scale = 1.0  # fallback

                scaled_lr = base_lr * warmup_factor * milestone_scale * dynamic_scale
                scaled_lrs.append(scaled_lr)


            return scaled_lrs
    def step(self, epoch=None, metric=None):
        super().step(epoch)
        

class WarmupMultiStepJenksBias(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones_weights,
        milestones_bias,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        alpha=0.5,     # Custom scaling factor for pruning-based adjustment
        beta=0.0,      # Optionally add saliency std as another factor
        last_epoch=-1,
        adjustable = True,
        cosine = False
    ):
        if not list(milestones_weights) == sorted(milestones_weights):
            raise ValueError(
                f"Milestones should be a list of increasing integers. Got {milestones_weights}"
            )
        if not list(milestones_bias) == sorted(milestones_bias):
            raise ValueError(
                f"Milestones should be a list of increasing integers. Got {milestones_bias}"
            )
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted, got {}".format(warmup_method)
            )

        self.milestones_weights = milestones_weights
        self.milestones_bias = milestones_bias
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.alpha = alpha
        self.beta = beta
        self.adjustable = adjustable  # Whether to adjust learning rate based on pruning/saliency
        super(WarmupMultiStepJenksBias, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1.0
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [
                base_lr
                * warmup_factor
                * self.gamma ** bisect_right(self.milestones_weights, self.last_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            scaled_lrs = []
        for group in self.optimizer.param_groups:
            name = group.get("name", None)
            base_lr = group['initial_lr'] if 'initial_lr' in group else group['lr']
            if (name is not None) and ("bias" in name or "bn" in name):
                milestone_scale = self.gamma ** bisect_right(self.milestones_bias, self.last_epoch)
            else:
                milestone_scale = self.gamma ** bisect_right(self.milestones_weights, self.last_epoch)

            if self.adjustable:
                if 'bias' in name or 'bn' in name:
                    continue
                else:
                    if name and hasattr(self.optimizer, "layerwise_lr_stats"):
                        stats = self.optimizer.layerwise_lr_stats.get(name, {})
                        percent_pruned = stats.get('percent_pruned', 0.0)
                        saliency_std = stats.get('saliency_std', 0.0)
                        dynamic_scale = 1.0 + self.alpha * percent_pruned + self.beta * saliency_std
                    # if "weight_decay" in group:
                    #     group["weight_decay"] *= (1-percent_pruned)**self.alpha
            else:
                dynamic_scale = 1.0  # fallback

            scaled_lr = base_lr * warmup_factor * milestone_scale * dynamic_scale
            scaled_lrs.append(scaled_lr)

        return scaled_lrs
    
class Min_OBJ_Lr_Scheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self,
                 optimizer, 
                 last_epoch=-1):
        self.a = torch.ones(len(optimizer.param_groups), device=optimizer.param_groups[0]['params'][0].device)
        self.b = torch.ones(len(optimizer.param_groups), device=optimizer.param_groups[0]['params'][0].device)
        self.init_lr = torch.zeros(len(optimizer.param_groups), device=optimizer.param_groups[0]['params'][0].device)
        for i, group in enumerate(optimizer.param_groups):
            # ensure stored initial lr is tensor on device
            self.init_lr[i] = torch.tensor(group['lr'], device=optimizer.param_groups[0]['params'][0].device, dtype=self.a.dtype)
        for i,val in enumerate(self.init_lr):
            self.a[i] = (self.init_lr[i]/(sqrt(2)-1))**2
            self.b[i] = self.init_lr[i]/(sqrt(2)-1)
        
        super(Min_OBJ_Lr_Scheduler, self).__init__(optimizer, last_epoch)
    def get_lr(self, epoch=None, metric=None):
        if epoch is not None:
            self.last_epoch = epoch
        scaled_lrs = []
        for i,group in enumerate(self.optimizer.param_groups):
            lr_t = torch.sqrt(self.b[i]**2 + self.a[i]) - self.b[i]
            # update a,b with tensor lr
            self.a[i] += lr_t**2
            self.b[i] += lr_t
            scaled_lrs.append(lr_t.item())
        return scaled_lrs

class WarmupAutoJenks(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        prune_epochs = 100,
        reset = False,
        warmup_method="linear",
        alpha=0.25,     # Custom scaling factor for pruning-based adjustment
        beta=0.1,      # Optionally add saliency std as another factor
        last_epoch=-1,
        adjustable = False,
        wd_adj = True,
        cosine = False,
        rewind_epoch = None
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                f"Milestones should be a list of increasing integers. Got {milestones}"
            )
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted, got {}".format(warmup_method)
            )

        self.rewind_epoch = rewind_epoch
        self.milestones = milestones
        self.gamma = gamma
        self.optimizer = optimizer
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.prune_epochs = prune_epochs
        self.alpha = alpha
        self.beta = beta
        self.reset = reset
        self.adjustable = adjustable  # Whether to adjust learning rate based on pruning/saliency
        self.wd_adj = wd_adj
        self.a = torch.ones(len(optimizer.param_groups), device=optimizer.param_groups[0]['params'][0].device)
        self.b = torch.ones(len(optimizer.param_groups), device=optimizer.param_groups[0]['params'][0].device)
        self.adjustable = adjustable
        size = len(self.optimizer.param_groups)
        if self.wd_adj:
            self.M = torch.zeros(size, device=self.optimizer.param_groups[0]['params'][0].device)
            self.P = torch.zeros(size, device=self.optimizer.param_groups[0]['params'][0].device)
            self.init_lr = torch.zeros_like(self.a)
        # store initial wd as tensor on same device/dtype
        self.init_wd = torch.tensor([group.get("weight_decay", 0.0) for group in self.optimizer.param_groups],
                                    device=self.optimizer.param_groups[0]['params'][0].device, dtype=self.a.dtype)
        for i, group in enumerate(self.optimizer.param_groups):
            # ensure stored initial lr is tensor on device
            self.init_lr[i] = torch.tensor(group['lr'], device=self.optimizer.param_groups[0]['params'][0].device, dtype=self.a.dtype)
        for i,(val, wd) in enumerate(zip(self.init_lr, self.init_wd)):
            self.a[i] = (self.init_lr[i]/(sqrt(2)-1))**2
            self.b[i] = self.init_lr[i]/(sqrt(2)-1)
            self.P[i] = wd/(sqrt(2)-1)
            self.M[i] = (wd/(sqrt(2)-1))**2 * self.init_lr[i]
        super(WarmupAutoJenks, self).__init__(optimizer, last_epoch)
    def reset_accumulators(self):
        for i,(val, wd) in enumerate(zip(self.init_lr, self.init_wd)):
            self.a[i] = (self.init_lr[i]/(sqrt(2)-1))**2
            self.b[i] = self.init_lr[i]/(sqrt(2)-1)
            self.P[i] = wd/(sqrt(2)-1)
            self.M[i] = (wd/(sqrt(2)-1))**2 * self.init_lr[i]
        '''Reset the learning rate and weight decay to self.init_lr and self.init_wd'''
        
    def get_lr(self, epoch=None, metric=None):
        warmup_factor = 1.0
        if epoch is not None:
            self.last_epoch = epoch
        if self.last_epoch < self.warmup_iters:
            ## get the current learning rate
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            lr = [
                base_lr
                * warmup_factor
                * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs
            ]
            return lr
        elif self.last_epoch == self.prune_epochs and self.reset:
            self.reset_accumulators()
            scaled_lrs = []
            for i,group in enumerate(self.optimizer.param_groups):
                lr_t = self.init_lr[i]
                wd = self.init_wd[i]
                # stabilize division with eps; avoid lr_t being zero
                if self.wd_adj:
                    eps = 1e-12
                    wd = torch.sqrt(self.P[i]**2 + self.M[i] / (lr_t + eps)) - self.P[i]
                    # update accumulators using tensor arithmetic
                    self.M[i] += lr_t * (wd ** 2)
                    self.P[i] += wd
                    # update param group weight decay deterministically (no cumulative multiplication)
                    group["weight_decay"] = float(wd.clamp(min=0.0).item())
                # update a,b with tensor lr
                self.a[i] += lr_t**2
                self.b[i] += lr_t
                scaled_lrs.append(lr_t.item())
            return scaled_lrs
        elif self.rewind_epoch is not None and self.last_epoch == self.rewind_epoch:
            # LR warm-restart on the post-freeze recovery tail: mask is frozen,
            # re-anneal the (already accumulated) LR/WD schedule from init so the
            # fixed sparse weights can re-converge into a better basin.
            self.reset_accumulators()
            scaled_lrs = []
            for i,group in enumerate(self.optimizer.param_groups):
                lr_t = self.init_lr[i]
                wd = self.init_wd[i]
                if self.wd_adj:
                    eps = 1e-12
                    wd = torch.sqrt(self.P[i]**2 + self.M[i] / (lr_t + eps)) - self.P[i]
                    self.M[i] += lr_t * (wd ** 2)
                    self.P[i] += wd
                    group["weight_decay"] = float(wd.clamp(min=0.0).item())
                self.a[i] += lr_t**2
                self.b[i] += lr_t
                scaled_lrs.append(lr_t.item())
            return scaled_lrs
        elif self.last_epoch >= self.warmup_iters:
            scaled_lrs = []
            for i,group in enumerate(self.optimizer.param_groups):
                lr_t = torch.sqrt(self.b[i]**2 + self.a[i]) - self.b[i]
                # stabilize division with eps; avoid lr_t being zero
                if self.wd_adj:
                    eps = 1e-12
                    wd = torch.sqrt(self.P[i]**2 + self.M[i] / (lr_t + eps)) - self.P[i]
                    # update accumulators using tensor arithmetic
                    self.M[i] += lr_t * (wd ** 2)
                    self.P[i] += wd
                    # update param group weight decay deterministically (no cumulative multiplication)
                    group["weight_decay"] = float(wd.clamp(min=0.0).item())
                # update a,b with tensor lr
                self.a[i] += lr_t**2
                self.b[i] += lr_t
                scaled_lrs.append(lr_t.item())
            return scaled_lrs
    def step(self, epoch=None, metric=None):
        super().step(epoch)


class WarmupAutoSGDJenks(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        alpha=0.25,     # Custom scaling factor for pruning-based adjustment
        beta=0.1,      # Optionally add saliency std as another factor
        last_epoch=-1,
        adjustable = False,
        wd_adj = True,
        cosine = False
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                f"Milestones should be a list of increasing integers. Got {milestones}"
            )
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted, got {}".format(warmup_method)
            )
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.alpha = alpha
        self.beta = beta
        self.adjustable = adjustable  # Whether to adjust learning rate based on pruning/saliency
        self.wd_adj = wd_adj
        count = 0
        for group in self.optimizer.param_groups:
            name = group.get("name", None)
            # if 'bias' in name or 'bn' in name:
            #     continue
            if name and hasattr(self.optimizer, "layerwise_lr_stats"):
                if 'bias' in name or 'bn' in name:
                    continue
                else:
                    count += 1
        self.weight_length = count
        size = len(optimizer.param_groups)
        self.size = size
        device = optimizer.param_groups[0]['params'][0].device
        self.a = torch.ones(size, device=device)
        self.b = torch.ones(size, device=device)
        if self.wd_adj:
            self.M = torch.zeros(size, device=device)
            self.P = torch.zeros(size, device=device)
            self.init_lr = torch.zeros_like(self.a)
        # store initial wd as tensor on same device/dtype
        self.init_wd = torch.tensor([group.get("weight_decay", 0.0) for group in optimizer.param_groups],
                                    device=device, dtype=self.a.dtype)
        for i, group in enumerate(optimizer.param_groups):
            # ensure stored initial lr is tensor on device
            self.init_lr[i] = torch.tensor(group['lr'], device=device, dtype=self.a.dtype)
        self.a *= self.init_lr**2
        self.b *= self.init_lr
        super(WarmupAutoSGDJenks, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        #For warmup epochs, we want to find the lr according to Automatic SGD
        # Namely, \eta = ln((1+\sqrt(1+4G))/2, where G is the sum of the gradient fronebius norms. Each sum term is
        # multiplied by the sqrt of the output dimension divided by the input dimension
        # For each layer the learning rate will then be divided by the number of layers, multiplied by the sqrt of the output dimension
        # divided by the input dimension and then divided by the fronebius norm of the gradient for that layer
        # Further, we will want to select the weight decay to in order to match the weight update, i.e. ||G||_F=\alpha*||W||_F
        # After warmup we will do the same as WarmupAutoJenks
        warmup_factor = 1.0
        if self.last_epoch < self.warmup_iters: 
            # Get the gradient fronebius norms from the optimizer
            G = 0.0
            for group in self.optimizer.param_groups:
                name = group.get("name", None)
                # if 'bias' in name or 'bn' in name:
                #     continue
                if name and hasattr(self.optimizer, "layerwise_lr_stats"):
                    if 'bias' in name or 'bn' in name:
                        continue
                    stats = self.optimizer.layerwise_lr_stats.get(name, {})
                    grad_frob_norm = stats.get('grad_frob_norm', 0.0)
                    singular_value = stats.get('singular_value', 1.0)
                    G += grad_frob_norm * singular_value
            G/=self.weight_length
            eta = log((1 + sqrt(1 + 4 * G)) / 2)
            # print(f"Warmup AutoSGD Jenks eta: {eta}, G: {G}")
            # Now we set the learning rate for each layer
            eps = 1e-6
            max_lr_factor = 1e3  # clamp factor relative to base lr
            weight_lr_map = {}
            # compute adaptive lrs only for weight layers (skip bias/bn)
            for group in self.optimizer.param_groups:
                name = group.get("name", None)
                base_lr = group.get("initial_lr", group.get("lr", 1e-6))
                computed_lr = float(base_lr)
                if name and hasattr(self.optimizer, "layerwise_lr_stats") and ('bias' not in name and 'bn' not in name):
                    stats = self.optimizer.layerwise_lr_stats.get(name, {})
                    grad_frob_norm = stats.get('grad_frob_norm', 0.0)
                    weight_frob_norm = stats.get('weight_frob_norm', 0.0)
                    if isinstance(grad_frob_norm, torch.Tensor):
                        grad_frob_norm = float(grad_frob_norm.item())
                    if isinstance(weight_frob_norm, torch.Tensor):
                        weight_frob_norm = float(weight_frob_norm.item())
                    if weight_frob_norm > eps:
                        wd = float(grad_frob_norm / (weight_frob_norm + eps))
                        group["weight_decay"] = max(0.0, wd)
                    if grad_frob_norm > eps:
                        scale = self.optimizer.layerwise_lr_stats.get(name, {}).get('singular_value', 1.0)
                        den = max(self.weight_length * grad_frob_norm, eps)
                        computed_lr = float(eta * scale / den)
                weight_lr_map[name] = computed_lr
            lrs = []
            for i, group in enumerate(self.optimizer.param_groups):
                name = group.get("name", None)
                base_lr = group.get("initial_lr", group.get("lr", 1e-6))
                if name is None or ('bias' in name or 'bn' in name):
                    final_lr = float(base_lr)   # keep bias/BN lr constant
                else:
                    final_lr = float(max(1e-12, min(weight_lr_map.get(name, float(base_lr)), float(base_lr) * max_lr_factor)))

                # update accumulators (do not overwrite bias/bn lr)
                lr_t = torch.tensor(final_lr, device=self.a.device, dtype=self.a.dtype)
                self.a[i] += lr_t**2
                self.b[i] += lr_t
                if self.wd_adj:
                    wd_val = group.get("weight_decay", 0.0)
                    self.M[i] += lr_t * (wd_val ** 2)
                    self.P[i] += wd_val
                lrs.append(final_lr)
            # print(f"Warmup AutoSGD Jenks lrs: {lrs}")
            return lrs
        else:
            scaled_lrs = []
            for i,group in enumerate(self.optimizer.param_groups):
                lr_t = torch.sqrt(self.b[i]**2 + self.a[i]) - self.b[i]
                # stabilize division with eps; avoid lr_t being zero
                if self.wd_adj:
                    eps = 1e-12
                    wd = torch.sqrt(self.P[i]**2 + self.M[i] / (lr_t + eps)) - self.P[i]
                    # update accumulators using tensor arithmetic
                    self.M[i] += lr_t * (wd ** 2)
                    self.P[i] += wd
                    # update param group weight decay deterministically (no cumulative multiplication)
                    group["weight_decay"] = float(wd.clamp(min=0.0).item())
                # update a,b with tensor lr
                self.a[i] += lr_t**2
                self.b[i] += lr_t
                scaled_lrs.append(lr_t.item())
            return scaled_lrs
    def step(self, epoch=None, metric=None):
        super().step(epoch)
            



class SequentialJenksScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, schedulers, milestones):
        assert len(schedulers) == len(milestones) + 1, "schedulers should be one more than milestones"
        self.schedulers = schedulers
        self.milestones = milestones
        self.current_idx = 0
        self.current_scheduler = self.schedulers[self.current_idx]
        self.last_epoch = -1
        super().__init__(optimizer)
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def step(self, epoch=None, metric=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # Switch scheduler if at milestone
        while self.current_idx < len(self.milestones) and epoch >= self.milestones[self.current_idx]:
            self.current_idx += 1
            self.current_scheduler = self.schedulers[self.current_idx]

        # Step the current scheduler
        import inspect
        params = inspect.signature(self.current_scheduler.step).parameters
        if "metrics" in params:
            self.current_scheduler.step(metric)
        elif "epoch" in params:
            self.current_scheduler.step(epoch)
        else:
            self.current_scheduler.step()

        # Update optimizer lrs
        super().step(epoch)

    def get_lr(self):
        return self.current_scheduler.get_last_lr()
        


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        final_lr,
        final_iters,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        assert final_iters > warmup_iters
        self.final_lr = final_lr
        self.final_iters = final_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = max(warmup_iters, 0)
        self.warmup_method = warmup_method
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                raise ValueError(
                    "Only 'constant' or 'linear' warmup_method accepted"
                    "got {}".format(self.warmup_method)
                )
            return [
                base_lr * warmup_factor for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr + (self.final_lr - base_lr) * (
                    1 + torch.cos(torch.tensor(self.last_epoch - self.warmup_iters) / (self.final_iters - self.warmup_iters) * 3.141592653589793)) / 2
                for base_lr in self.base_lrs
            ]


class WarmupLinearLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        final_lr,
        final_iters,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        assert final_iters > warmup_iters
        self.final_lr = final_lr
        self.final_iters = final_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = max(warmup_iters, 0)
        self.warmup_method = warmup_method
        super(WarmupLinearLR, self).__init__(optimizer, last_epoch)

    #   last_epoch == 0:            base_lr * warmup_factor
    #   last_epoch == warmup_iters: base_lr
    #   last_epoch == final_iters:  final_lr

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                raise ValueError(
                    "Only 'constant' or 'linear' warmup_method accepted"
                    "got {}".format(self.warmup_method)
                )
            return [
                base_lr
                * warmup_factor
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr - (base_lr - self.final_lr) * float(self.last_epoch - self.warmup_iters) / (
                            self.final_iters - self.warmup_iters)
                for base_lr in self.base_lrs
            ]

#   LR scheduler should work according the number of iterations
# def get_lr_scheduler(cfg, optimizer):
#     it_ep = num_iters_per_epoch(cfg)
#     if cfg.linear_final_lr is None:
#         lr_iter_boundaries = [it_ep * ep for ep in cfg.lr_epoch_boundaries]
#         return WarmupMultiStepLR(
#             optimizer, lr_iter_boundaries, cfg.lr_decay_factor,
#             warmup_factor=cfg.warmup_factor,
#             warmup_iters=cfg.warmup_epochs * it_ep,
#             warmup_method=cfg.warmup_method, )
#     else:
#         return WarmupLinearLR(optimizer, final_lr=cfg.linear_final_lr,
#                               final_iters=cfg.max_epochs * it_ep,
#                               warmup_factor=cfg.warmup_factor,
#                               warmup_iters=cfg.warmup_epochs * it_ep,
#                               warmup_method=cfg.warmup_method,)



def compute_layer_lr(base_lr, percent_pruned, std_saliency, alpha=1.0, beta=0.1):
    # alpha controls pruning sensitivity; beta controls saliency sensitivity
    scale = 1.0 + alpha * percent_pruned + beta * std_saliency
    return base_lr * scale


class LayerwiseAdaptiveLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, alpha=1.0, beta=0.1, gamma=0.1,warmup_factor=1.0 / 3,warmup_iters=500,warmup_method="linear",last_epoch=-1):
        self.optimizer = optimizer
        self.alpha = alpha
        self.beta = beta

    def step(self):
        for group in self.optimizer.param_groups:
            name = group.get("name", None)
            if name and hasattr(self.optimizer, "layerwise_lr_stats"):
                stats = self.optimizer.layerwise_lr_stats.get(name, {})
                base_lr = self.optimizer.base_lrs_by_name.get(name, group['lr'])

                # Custom scaling rule
                scale = 1.0 + self.alpha * stats.get('percent_pruned', 0.0) + self.beta * stats.get('saliency_std', 0.0)
                group['lr'] = base_lr * scale




def init_lr_weight_decay(model, learning_rate, weight_decay, bias_weight_decay=0, momentum=0.9, nestrov=False, bias_lr=False, elem_bias = False, warmup_epochs =10, prune_epoch=150):
    base_lrs = {}
    layerwise_lr_stats = {}
    params = []
    name_map = {p: n for n, p in model.named_parameters()}
    #for do_prune we need to get the named_modules and access the do_prune attribute
    do_prune_map = {n: m.do_prune if hasattr(m, 'do_prune') else False for n, m in model.named_modules()}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Set learning rate and weight decay
        lr = 2 * learning_rate if (bias_lr and 'bias' in name) else learning_rate
        wd = weight_decay
        if "bias" in name or "bn" in name or "BN" in name:
            # lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            wd = bias_weight_decay
            # print('set weight_decay_bias={} for {}'.format(wd, name))
        wd_norm = torch.norm(param.data).item()
        # Store base LR and initialize stats
        base_lrs[name] = lr
        layerwise_lr_stats[name] = {
            'percent_pruned': 0.0,
            'saliency_std': 0.0,
            'grad_frob_norm': 0.0,
            'weight_frob_norm': wd_norm,
            'singular_value': singular_value(param),
        }

        # Add to param group
        params.append({
            "params": [param],
            "lr": lr,
            "weight_decay": wd,
            "name": name  # Tag group with param name
        })

    # Create optimizer
    if elem_bias:
        optimizer = ElementwiseMomentumSGD(params, name_map, do_prune_map, lr=learning_rate, momentum=momentum, warmup_epochs=warmup_epochs,pruning_epochs=prune_epoch)
    else:
        optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, nesterov=nestrov)

    # Embed layerwise stats and base LRs directly into the optimizer
    optimizer.layerwise_lr_stats = layerwise_lr_stats
    optimizer.base_lrs_by_name = base_lrs

    return optimizer


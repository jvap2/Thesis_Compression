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
        return dict(exp_bits=3, man_bits=1)

    return dict(exp_bits=3, man_bits=0)





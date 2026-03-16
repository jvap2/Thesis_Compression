import torch
import torch.nn as nn
import copy



    

def compute_fp4_range_pruned(weight, percentile=0.01):

    w = weight.abs().view(-1)

    w = w[w > 0]   # remove pruned weights

    lower = torch.quantile(w, percentile)

    e_min = torch.floor(torch.log2(lower))

    return int(e_min)





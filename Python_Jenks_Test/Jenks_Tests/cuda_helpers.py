# Core libraries for various functionalities
import torch  # PyTorch library for deep learning
import os     # Operating system interfaces
import math   # Mathematical functions
import gzip   # Compression/decompression using gzip
import pickle # Object serialization
from jenkspy import JenksNaturalBreaks
import numpy as np
# Plotting library
import matplotlib.pyplot as plt

# For downloading files from URLs
from urllib.request import urlretrieve

# File and directory handling
from pathlib import Path

# Specific PyTorch imports
from torch import tensor  # Tensor data structure
import torch
# Computer vision libraries
import torchvision as tv  # PyTorch's computer vision library
import torchvision.transforms.functional as tvf  # Functional image transformations
from torchvision import io  # I/O operations for images and videos
import pynvml

# For loading custom CUDA extensions
from torch.utils.cpp_extension import load_inline, CUDA_HOME

# os.environ['TORCH_CUDA_ARCH_LIST'] = "8.9"

# Verify the CUDA install path 
print(CUDA_HOME)

def get_memory_free_MiB(gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.free // 1024 ** 2


def load_cuda(cuda_src, cpp_src, funcs, opt=False, verbose=False):
    """
    Load CUDA and C++ source code as a Python extension.

    This function compiles and loads CUDA and C++ source code as a Python extension,
    allowing for the use of custom CUDA kernels in Python.

    Args:
        cuda_src (str): CUDA source code as a string.
        cpp_src (str): C++ source code as a string.
        funcs (list): List of function names to be exposed from the extension.
        opt (bool, optional): Whether to enable optimization flags. Defaults to False.
        verbose (bool, optional): Whether to print verbose output during compilation. Defaults to False.

    Returns:
        module: Loaded Python extension module containing the compiled functions.
    """
    # Use load_inline to compile and load the CUDA and C++ source code
    return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs,
                       extra_cuda_cflags=["-O3","--use_fast_math","-Xcompiler", "-fPIC","--ptxas-options=-v","-gencode", "arch=compute_86,code=sm_86"] if opt else [], 
                       verbose=verbose, name=f"inline_ext_{os.getpid()}")


# Define CUDA boilerplate code and utility macros
cuda_begin = r'''
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

// Macro to check if a tensor is a CUDA tensor
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

// Macro to check if a tensor is contiguous in memory
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Macro to check both CUDA and contiguity requirements
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Utility function for ceiling division
inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}
'''

cuda_bias = cuda_begin + r'''
template <typename T>
__global__ void Jenks_Optimization_Biases(T* d_B, T* d_var, int rows){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    T mean_1,mean_2;
    if(idx < (rows)){
        // Now, we want to use the index to decide where the break is for calculation sake
        // For the first grouping, the idx will be the break point
        //We will calculate the mean of the first group
        mean_1 = 0;
        for(int i = 0; i<idx; i++){
            mean_1 += d_B[i];
        }
        if(idx != 0){
            mean_1 = mean_1/idx;
        }
        //We will calculate the mean of the second group
        mean_2 = 0;
        for(int i = idx; i<rows; i++){
            mean_2 += d_B[i];
        }
        mean_2 = mean_2/(rows-idx);
    }
    __syncthreads();
    if(idx<rows){
        T var = 0;
        for(int i = 0; i<idx; i++){
            var += (d_B[i] - mean_1)*(d_B[i] - mean_1);
        }
        for(int i = idx; i<rows; i++){
            var += (d_B[i] - mean_2)*(d_B[i] - mean_2);
        }
        d_var[idx] = var;
    }
}



torch::Tensor jenks_optimization_biases_cuda(torch::Tensor B){
    // Check input
    CHECK_INPUT(B);

    // Get dimensions
    int rows = B.size(0);

    // Allocate output tensor
    auto var = torch::empty({rows}, B.options());

    // Launch kernel
    const int threads = 256;
    const int blocks = cdiv(rows, threads);

    AT_DISPATCH_FLOATING_TYPES(B.scalar_type(), "jenks_optimization_biases_cuda", ([&] {
        Jenks_Optimization_Biases<scalar_t><<<blocks, threads>>>(B.data_ptr<scalar_t>(), var.data_ptr<scalar_t>(), rows);
    }));

    // Check for errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cudaDeviceSynchronize();

    return var;
}
'''

cuda_src = cuda_begin + r'''    

template <typename T>
__global__ void Jenks_Optimization(T* d_WB, T* d_var, int rows, int cols){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    T mean_1,mean_2;
    if(idx < (rows*(cols))){
        // Now, we want to use the index to decide where the break is for calculation sake
        // For the first grouping, the idx will be the break point
        //We will calculate the mean of the first group
        mean_1 = 0;
        for(int i = 0; i<idx; i++){
            mean_1 += d_WB[i];
        }
        if(idx != 0){
            mean_1 = mean_1/idx;
        }
        //We will calculate the mean of the second group
        mean_2 = 0;
        for(int i = idx; i<rows*(cols); i++){
            mean_2 += d_WB[i];
        }
        mean_2 = mean_2/(rows*(cols)-idx);
    }
    __syncthreads();
    if(idx<rows*(cols)){
        T var = 0;
        for(int i = 0; i<idx; i++){
            var += (d_WB[i] - mean_1)*(d_WB[i] - mean_1);
        }
        for(int i = idx; i<rows*(cols); i++){
            var += (d_WB[i] - mean_2)*(d_WB[i] - mean_2);
        }
        d_var[idx] = var;
    }
}


torch::Tensor jenks_optimization_cuda(torch::Tensor WB){
    // Check input
    CHECK_INPUT(WB);

    // Get dimensions
    int rows = WB.size(0);
    int cols = WB.size(1);

    // Allocate output tensor
    auto var = torch::empty({rows*cols}, WB.options());

    // Launch kernel
    const int threads = 256;
    const int blocks = cdiv(rows*cols, threads);
    AT_DISPATCH_FLOATING_TYPES(WB.scalar_type(), "jenks_optimization_cuda", ([&] {
        Jenks_Optimization<scalar_t><<<blocks, threads>>>(WB.data_ptr<scalar_t>(), var.data_ptr<scalar_t>(), rows, cols);
    }));

    // Check for errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cudaDeviceSynchronize();

    return var;
}


'''

## Perform for convolutional layers
cuda_conv = cuda_begin + r'''

template <typename T>
__global__ void Jenks_Optimization(T* d_WB, T* d_var, int rows, int cols, int in_channels, int out_channels){
    int in = threadIdx.x + blockIdx.x * blockDim.x;
    int out = threadIdx.y + blockIdx.y * blockDim.y;
    T mean_1,mean_2;
    if (in < in_channels && out < out_channels){
        // We will find the variance for each entry in this specific filter
        T* d_WB_filter = d_WB + in*out_channels*rows*cols + out*rows*cols;
        T* d_var_filter = d_var + in*out_channels*rows*cols + out*rows*cols;
        // Fill the indices array and then sort the weights, and move the indices around to keep track of the original indices


        for(int i = 0; i<rows*cols; i++){
            mean_1 = 0;
            for(int j = 0; j<i; j++){
                mean_1 += d_WB_filter[j];
            }
            if(i != 0){
                mean_1 = mean_1/i;
            }
            //We will calculate the mean of the second group
            mean_2 = 0;
            for(int j = i; j<rows*cols; j++){
                mean_2 += d_WB_filter[j];
            }
            mean_2 = mean_2/(rows*cols-i);
            T var = 0;
            for(int j = 0; j<i; j++){
                var += (d_WB_filter[j] - mean_1)*(d_WB_filter[j] - mean_1);
            }
            for(int j = i; j<rows*cols; j++){
                var += (d_WB_filter[j] - mean_2)*(d_WB_filter[j] - mean_2);
            }
            d_var_filter[i] = var;
        }
    }
}


torch::Tensor jenks_optimization_cuda_conv(torch::Tensor WB){
    // Check input
    CHECK_INPUT(WB);

    // Get dimensions
    int in_channels = WB.size(0);
    int out_channels = WB.size(1);
    int rows = WB.size(2);
    int cols = WB.size(3);

    // Allocate output tensor
    auto var = torch::empty({rows*cols*in_channels*out_channels}, WB.options());

    // Launch kernel
    const int threads = 16;
    const int blocks = cdiv(rows*cols, threads);

    int block_size = 16;
    dim3 blockDim2D(block_size, block_size);

    dim3 gridDim2D((in_channels + block_size - 1) / block_size, (out_channels+block_size-1)/block_size, 1);
    AT_DISPATCH_FLOATING_TYPES(WB.scalar_type(), "jenks_optimization_cuda", ([&] {
        Jenks_Optimization<scalar_t><<<gridDim2D, blockDim2D>>>(WB.data_ptr<scalar_t>(), var.data_ptr<scalar_t>(), rows, cols, in_channels, out_channels);
    }));

    // Check for errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cudaDeviceSynchronize();

    return var;
}

'''

cpp_src = "torch::Tensor jenks_optimization_cuda(torch::Tensor WB);"
cpp_bias_src = "torch::Tensor jenks_optimization_biases_cuda(torch::Tensor B);"
cpp_conv_src = "torch::Tensor jenks_optimization_cuda_conv(torch::Tensor WB);"

module_weights = load_cuda(cuda_src, cpp_src, ["jenks_optimization_cuda"], opt=True, verbose=True)
module_bias = load_cuda(cuda_bias, cpp_bias_src, ["jenks_optimization_biases_cuda"], opt=True, verbose=True)
module_conv = load_cuda(cuda_conv, cpp_conv_src, ["jenks_optimization_cuda_conv"], opt=True, verbose=True)

# Test
def vectorized_filter_mask(indices: torch.Tensor, min_indices: torch.Tensor) -> torch.Tensor:
    """
    Vectorized replacement for:
        for i in range(B):
            for j in range(C):
                arr[i, j, indices[i][j][:min_indices[i][j].item()]] = 0
    
    Parameters:
        indices (LongTensor): shape (B, C, K), sorted index positions of weights
        min_indices (LongTensor): shape (B, C), number of elements to zero per filter

    Returns:
        mask (FloatTensor): shape (B, C, K), 1s and 0s, where 0s mark pruned weights
    """
    B, C, K = indices.shape
    device = indices.device

    # Step 1: Create a range [0, 1, ..., K-1] and compare with min_indices
    range_tensor = torch.arange(K, device=device).view(1, 1, K)  # (1, 1, K)
    cutoff_mask = range_tensor < min_indices.unsqueeze(-1)       # (B, C, K), bool

    # Step 2: Create a mask filled with ones
    mask = torch.ones((B, C, K), device=device)

    # Step 3: Use scatter to zero-out the selected indices
    flat_indices = indices.view(B, C, K)
    mask.scatter_(dim=2, index=flat_indices, src=(~cutoff_mask).float())  # invert mask to zero-out

    return mask

# Fraction of surviving (kept) weights to additionally prune beyond the Jenks
# natural break. 0.0 == pure Jenks (original behavior). With >0, re-pruning a
# layer that already contains zeros compounds: the Jenks break lands at the
# zero/survivor boundary, and this shift removes the lowest alpha-fraction of
# the survivors, so iterative pruning increases sparsity each pass.
OVER_PRUNE = 0.0

# Create a random tensor
def Conv_Mask(weights_cuda):
    B, C, H, W = weights_cuda.shape
    indices = torch.zeros(B, C, H, W, dtype=torch.int64)
    weights_sorted = torch.zeros(B, C, H, W, weights_cuda.shape[3])
    # streams = [torch.cuda.Stream() for _ in range(weights_cuda.shape[0] * weights_cuda.shape[1])]
    # count = 0
    # for i in range(weights_cuda.shape[0]):
    #     for j in range(weights_cuda.shape[1]):
    #         with torch.cuda.stream(streams[count]):
    #             count += 1
    #             sorted_weights, sorted_indices = weights_cuda[i, j].view(-1).sort()
    #             weights_sorted[i,j]=sorted_weights.reshape(weights_cuda[i, j].shape)
    #             indices[i,j] =sorted_indices.reshape(weights_cuda[i, j].shape)
    flat = weights_cuda.view(B * C, -1)
    sorted_weights, sorted_indices = torch.sort(flat, dim=1)
    weights_sorted = sorted_weights.view(B, C, H, W)
    indices = sorted_indices.view(B, C, -1)
    # Flatten weights_sorted
    # Call the custom CUDA function
    # Combine weights_sorted into a single 4D tensor
    weights_sorted = weights_sorted.cuda()  # Move weights_sorted to the GPU
    weights_sorted = weights_sorted.contiguous()
    mean = weights_sorted.mean(dim=(2, 3), keepdim=True)
    var = module_conv.jenks_optimization_cuda_conv(weights_sorted)
    ## Reshape the var
    var = var.reshape(B, C, H, W)
    ## Find the minimums for each filter
    # Find the minimums for each filter
    min_values, min_indices = var.view(var.shape[0], var.shape[1], -1).min(dim=2)
    SSD_total = ((weights_sorted - mean) ** 2).sum(dim=(2, 3))
    GVF = (SSD_total - min_values) / (SSD_total + 1e-8)  # Avoid division by zero
    # Split weights_sorted based on var_min
    # Now indices holds the original indices of the sorted weights sorted on a filter basis
    # min_indices holds the minimum indices for each filter
    # Now we can use the indices to create the arr
    arr = torch.ones(B, C, H * W)
    indices_flatten = indices.view(B, C, -1)
    # for i in range(B):
    #     for j in range(C):
    #         arr[i, j, indices[i][j][:min_indices[i][j].item()]]=0
    arr = vectorized_filter_mask(indices_flatten, min_indices)
    arr = arr.view(B, C, H, W)
    del weights_sorted, indices, min_indices, min_values
    return arr, GVF.cpu().detach().numpy().tolist()

def Linear_Mask(weights_cuda):
    weights_cuda_flatten = weights_cuda.view(-1)
    mean = weights_cuda_flatten.mean()
    weights_cuda_sorted, weights_cuda_indices = weights_cuda_flatten.sort()
    SSD_total = ((weights_cuda_sorted - mean) ** 2).sum()
    # Call the custom CUDA function
    weights_cuda_sorted = weights_cuda_sorted.view(weights_cuda.shape)
    weights_cuda_sorted = weights_cuda_sorted.contiguous()
    var = module_weights.jenks_optimization_cuda(weights_cuda_sorted)
    var_min = var.argmin().item()
    if OVER_PRUNE > 0:
        n = weights_cuda_flatten.numel()
        var_min = min(var_min + int(OVER_PRUNE * (n - var_min)), n - 1)
    # Print the output
    ones = weights_cuda_indices[var_min:]
    arr = torch.zeros(weights_cuda_flatten.shape)
    arr[ones] = 1
    arr = arr.reshape(weights_cuda.shape)
    GVF = (SSD_total - var.min()) / (SSD_total + 1e-8)  # Avoid division by zero
    del weights_cuda_sorted, weights_cuda_indices, var, ones
    return arr, GVF.item()

def Bias_Mask(weights_cuda):
    weights_cuda_sorted, weights_cuda_indices = weights_cuda.sort()
    mean = weights_cuda_sorted.mean()
    SSD_total = ((weights_cuda_sorted - mean) ** 2).sum()
    weights_cuda_sorted = weights_cuda_sorted.contiguous()
    var = module_bias.jenks_optimization_biases_cuda(weights_cuda_sorted)
    var_min = var.argmin().item()
    if OVER_PRUNE > 0:
        n = weights_cuda.numel()
        var_min = min(var_min + int(OVER_PRUNE * (n - var_min)), n - 1)
    # Print the output
    # zeros = weights_cuda_indices[:var_min]
    ones = weights_cuda_indices[var_min:]
    arr = torch.zeros(weights_cuda.shape)
    arr[ones] = 1
    GVF = (SSD_total - var.min()) / (SSD_total + 1e-8)  # Avoid division by zero
    del weights_cuda_sorted, weights_cuda_indices, var, ones
    return arr, GVF.item()


# WB = torch.rand(4, 5)
# WB_cuda = WB.cuda()
# print(WB_cuda)
 
# ## Sort it

# arr = Linear_Mask(WB_cuda)
# ''' We now need to find the break point which '''

# arr_score = WB.numpy()
# arr_score_flat = arr_score.flatten()
# jnb = JenksNaturalBreaks(2)
# jnb.fit(arr_score_flat)
# print(jnb.labels_)
# labels = jnb.labels_
# indices = np.where(labels == 1)[0]
# indices_ = np.where(labels == 0)[0]

# test_arr = np.zeros(arr_score_flat.shape)
# test_arr[indices] = 1
# test_arr[indices_] = 0
# test_arr = test_arr.reshape(arr_score.shape)
# print(test_arr)

# # ''' Test they are equal'''

# print(np.allclose(arr.numpy(), test_arr, atol=1e-4))

# # Create a random tensor
# B = torch.rand(4)
# B_cuda = B.cuda()
# # Call the custom CUDA function
# arr = Bias_Mask(B_cuda)

# # Test
# arr_score = B.numpy()
# arr_score_flat = arr_score.flatten()
# jnb = JenksNaturalBreaks(2)
# jnb.fit(arr_score_flat)
# print(jnb.labels_)
# labels = jnb.labels_
# indices = np.where(labels == 1)[0]
# indices_ = np.where(labels == 0)[0]

# test_arr = np.zeros(arr_score_flat.shape)
# test_arr[indices] = 1
# test_arr[indices_] = 0

# print(test_arr)

# ''' Test they are equal'''

# print(np.allclose(arr, test_arr, atol=1e-4))


## Test how this works on convolutional layers

# layer = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
# layer_cpu = layer.cpu()
# # Get the weights
# layer_cuda = layer.cuda()
# # Get the weights
# weights = layer_cuda.weight
# weights_cuda = weights.cuda()
# arr, GVF = Conv_Mask(weights_cuda)
# print("GVF:", GVF)
# print("Shape of the mask:", arr.shape)
# print("Shape of the GVF:", GVF.shape)
# # Test


# for i in range(weights_cuda.shape[0]):
#     for j in range(weights_cuda.shape[1]):
#         arr_score = layer.weight[i,j].cpu().detach().numpy()
#         arr_score_flat = arr_score.flatten()
#         jnb = JenksNaturalBreaks(2)
#         jnb.fit(arr_score_flat)
#         # print(jnb.labels_)
#         labels = jnb.labels_
#         indices = np.where(labels == 1)[0]
#         indices_ = np.where(labels == 0)[0]
#         test_arr = np.zeros(arr_score_flat.shape)
#         test_arr[indices] = 1
#         test_arr[indices_] = 0
#         test_arr = test_arr.reshape(arr_score.shape)
#         print(test_arr)
#         print(arr[i,j].cpu().detach().numpy())

#         ''' Test they are equal'''
#         print(np.allclose(arr[i,j].cpu().numpy(), test_arr, atol=1e-4))





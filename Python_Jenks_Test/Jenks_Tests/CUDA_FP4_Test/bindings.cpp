// bindings.cpp
//
// Thin PyTorch (pybind11) wrappers around the raw CUDA launch_* functions
// declared in hadamard_kernel.cu, gf4_encode_kernel.cu,
// e2m1_fused_gemv_kernel.cu, and hessian_weight_quant_kernel.cu, so bench.py
// can drive everything from Python with torch.utils.cpp_extension.load(...).
// No kernel logic lives here - this file only validates tensor
// shapes/dtypes/contiguity and casts pointers.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// ---- forward declarations of the launch_* functions defined in the .cu files ----
void launch_hadamard_fwht(float* d_x, int rows, int M, int block_size,
                           const float* d_sign, cudaStream_t stream);
void launch_hadamard_naive(const float* d_x_in, float* d_x_out, int rows, int M, int block_size,
                            const float* d_sign, cudaStream_t stream);

void launch_gf4_encode_warp32(const float* d_x, uint8_t* d_codes, __half* d_scales,
                               int n_blocks, float clip_ratio, const float* d_mu,
                               int n_blocks_per_row, cudaStream_t stream);
void launch_gf4_encode_naive(const float* d_x, uint8_t* d_codes, __half* d_scales,
                              int n_blocks, float clip_ratio, const float* d_mu,
                              int n_blocks_per_row, cudaStream_t stream);
void launch_gf4_encode_perblock_adaptive(const float* d_x, uint8_t* d_codes, __half* d_scales,
                                          int n_blocks, const float* d_clip_candidates,
                                          int n_candidates, const float* d_mu,
                                          int n_blocks_per_row, cudaStream_t stream);
void launch_gf4_decode(const uint8_t* d_codes, const __half* d_scales, float* d_x_out,
                        int n_blocks, cudaStream_t stream);

void launch_e2m1_fused_gemv(const uint8_t* d_Wc, const __half* d_Wa, const uint8_t* d_Wb,
                             const __half* d_x, __half* d_y, int M, int K,
                             const float* d_bias_correction, cudaStream_t stream);
void launch_e2m1_naive_gemv(const uint8_t* d_Wc, const __half* d_Wa, const uint8_t* d_Wb,
                             __half* d_Wdequant_scratch, const __half* d_x, __half* d_y,
                             int M, int K, const float* d_bias_correction, cudaStream_t stream);

void launch_gf4_fused_gemv(const uint8_t* d_Wc, const __half* d_Wa,
                           const __half* d_x, __half* d_y, int M, int K,
                           const float* d_bias_correction, cudaStream_t stream);
void launch_gf4_lattice_gemv(const uint8_t* d_Wc, const __half* d_Wa,
                             const __half* d_x, __half* d_y, int M, int K,
                             const float* d_bias_correction, cudaStream_t stream);
void launch_gf4_lattice_gemv_batched(const uint8_t* d_Wc, const __half* d_Wa,
                                      const __half* d_x, __half* d_y,
                                      int M, int K, int bs,
                                      const float* d_bc, cudaStream_t stream);
void launch_gf4_naive_gemv(const uint8_t* d_Wc, const __half* d_Wa,
                           __half* d_Wdequant_scratch, const __half* d_x, __half* d_y,
                           int M, int K, const float* d_bias_correction, cudaStream_t stream);
void launch_dense_fp16_gemv(const __half* d_W, const __half* d_x, __half* d_y,
                            int M, int K, const float* d_bias_correction, cudaStream_t stream);

void launch_hessian_accumulate(const float* d_X, float* d_H_out, int n_tokens, int M,
                                int block_size, cudaStream_t stream);
void launch_hessian_damp_blocks(const float* d_H_in, float* d_H_out, int n_blocks,
                                 int block_size, float kappa, int power_iters,
                                 cudaStream_t stream);
void launch_hessian_weight_solve(const float* d_W, const float* d_H_damped,
                                  uint8_t* d_W_codes, __half* d_W_alpha, uint8_t* d_W_bias,
                                  int N, int M, int block_size, cudaStream_t stream);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// ---------------------------------------------------------------------------
// Hadamard transform (randomized, blockwise). d_sign: optional [M] +-1
// tensor (pass an empty tensor for "no sign flip" / plain Hadamard).
// ---------------------------------------------------------------------------
torch::Tensor hadamard_fwht(torch::Tensor x, int64_t block_size, torch::Tensor d_sign) {
    CHECK_CUDA(x); CHECK_CONTIG(x);
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be fp32");
    TORCH_CHECK(x.dim() == 2, "x must be [rows, M]");
    auto out = x.clone();
    int rows = out.size(0), M = out.size(1);
    const float* sign_ptr = nullptr;
    if (d_sign.numel() > 0) {
        CHECK_CUDA(d_sign); CHECK_CONTIG(d_sign);
        TORCH_CHECK(d_sign.dtype() == torch::kFloat32, "d_sign must be fp32");
        TORCH_CHECK(d_sign.numel() == M, "d_sign must have M elements");
        sign_ptr = d_sign.data_ptr<float>();
    }
    launch_hadamard_fwht(out.data_ptr<float>(), rows, M, (int)block_size, sign_ptr,
                          at::cuda::getCurrentCUDAStream());
    return out;
}

torch::Tensor hadamard_naive(torch::Tensor x, int64_t block_size, torch::Tensor d_sign) {
    CHECK_CUDA(x); CHECK_CONTIG(x);
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be fp32");
    int rows = x.size(0), M = x.size(1);
    auto out = torch::empty_like(x);
    const float* sign_ptr = nullptr;
    if (d_sign.numel() > 0) {
        CHECK_CUDA(d_sign); CHECK_CONTIG(d_sign);
        sign_ptr = d_sign.data_ptr<float>();
    }
    launch_hadamard_naive(x.data_ptr<float>(), out.data_ptr<float>(), rows, M, (int)block_size,
                          sign_ptr, at::cuda::getCurrentCUDAStream());
    return out;
}

// ---------------------------------------------------------------------------
// GF4 activation encoder (GF4_BLOCK == 32). scale = block_rms * clip_ratio.
// mu: optional [row_width] per-channel mean (pass an empty tensor to disable
// mean-centering, i.e. the original behavior). row_width must divide x's
// total element count and must itself be a multiple of 32.
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> gf4_encode(torch::Tensor x, double clip_ratio, bool fused, torch::Tensor mu) {
    CHECK_CUDA(x); CHECK_CONTIG(x);
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be fp32");
    TORCH_CHECK(x.numel() % 32 == 0, "x.numel() must be a multiple of 32");
    int n_blocks = x.numel() / 32;

    auto codes = torch::empty({n_blocks * 16}, x.options().dtype(torch::kUInt8));
    auto scales = torch::empty({n_blocks}, x.options().dtype(torch::kFloat16));

    const float* mu_ptr = nullptr;
    int n_blocks_per_row = 0;
    if (mu.numel() > 0) {
        CHECK_CUDA(mu); CHECK_CONTIG(mu);
        TORCH_CHECK(mu.dtype() == torch::kFloat32, "mu must be fp32");
        TORCH_CHECK(mu.numel() % 32 == 0, "mu.numel() (row width) must be a multiple of 32");
        mu_ptr = mu.data_ptr<float>();
        n_blocks_per_row = mu.numel() / 32;
        TORCH_CHECK(n_blocks % n_blocks_per_row == 0,
                    "total n_blocks must be a multiple of mu's n_blocks_per_row");
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    __half* scales_ptr = reinterpret_cast<__half*>(scales.data_ptr<at::Half>());
    if (fused) {
        launch_gf4_encode_warp32(x.data_ptr<float>(), codes.data_ptr<uint8_t>(), scales_ptr,
                                  n_blocks, (float)clip_ratio, mu_ptr, n_blocks_per_row, stream);
    } else {
        launch_gf4_encode_naive(x.data_ptr<float>(), codes.data_ptr<uint8_t>(), scales_ptr,
                                 n_blocks, (float)clip_ratio, mu_ptr, n_blocks_per_row, stream);
    }
    return {codes, scales};
}

// ---------------------------------------------------------------------------
// GF4 per-block ADAPTIVE-clip encoder: each block picks the clip in
// clip_candidates that minimizes its own reconstruction MSE (online, no
// calibration). Matches quantize_activations_gf4_adaptive() in bit_split.py.
// Returns {codes, scales}, decodable by gf4_decode exactly like gf4_encode's
// output - the chosen per-block clip is baked into the per-block scale.
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> gf4_encode_adaptive(torch::Tensor x, torch::Tensor clip_candidates,
                                               torch::Tensor mu) {
    CHECK_CUDA(x); CHECK_CONTIG(x);
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be fp32");
    TORCH_CHECK(x.numel() % 32 == 0, "x.numel() must be a multiple of 32");
    CHECK_CUDA(clip_candidates); CHECK_CONTIG(clip_candidates);
    TORCH_CHECK(clip_candidates.dtype() == torch::kFloat32, "clip_candidates must be fp32");
    TORCH_CHECK(clip_candidates.numel() > 0, "clip_candidates must be non-empty");
    int n_blocks = x.numel() / 32;
    int n_candidates = clip_candidates.numel();

    auto codes = torch::empty({n_blocks * 16}, x.options().dtype(torch::kUInt8));
    auto scales = torch::empty({n_blocks}, x.options().dtype(torch::kFloat16));

    const float* mu_ptr = nullptr;
    int n_blocks_per_row = 0;
    if (mu.numel() > 0) {
        CHECK_CUDA(mu); CHECK_CONTIG(mu);
        TORCH_CHECK(mu.dtype() == torch::kFloat32, "mu must be fp32");
        TORCH_CHECK(mu.numel() % 32 == 0, "mu.numel() (row width) must be a multiple of 32");
        mu_ptr = mu.data_ptr<float>();
        n_blocks_per_row = mu.numel() / 32;
        TORCH_CHECK(n_blocks % n_blocks_per_row == 0,
                    "total n_blocks must be a multiple of mu's n_blocks_per_row");
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    __half* scales_ptr = reinterpret_cast<__half*>(scales.data_ptr<at::Half>());
    launch_gf4_encode_perblock_adaptive(x.data_ptr<float>(), codes.data_ptr<uint8_t>(), scales_ptr,
                                        n_blocks, clip_candidates.data_ptr<float>(), n_candidates,
                                        mu_ptr, n_blocks_per_row, stream);
    return {codes, scales};
}

// ---------------------------------------------------------------------------
// GF4 decode: packed codes + per-block scale -> dense fp32 activations.
// Does not re-add mu (see e2m1_fused_gemv's bias_correction instead).
// ---------------------------------------------------------------------------
torch::Tensor gf4_decode(torch::Tensor codes, torch::Tensor scales, int64_t n_blocks) {
    CHECK_CUDA(codes); CHECK_CONTIG(codes);
    CHECK_CUDA(scales); CHECK_CONTIG(scales);
    TORCH_CHECK(codes.dtype() == torch::kUInt8, "codes must be uint8");
    TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be fp16");
    TORCH_CHECK(codes.numel() == n_blocks * 16, "codes must have n_blocks*16 bytes");
    TORCH_CHECK(scales.numel() == n_blocks, "scales must have n_blocks elements");

    auto x_out = torch::empty({n_blocks * 32}, codes.options().dtype(torch::kFloat32));
    launch_gf4_decode(codes.data_ptr<uint8_t>(),
                       reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
                       x_out.data_ptr<float>(), (int)n_blocks, at::cuda::getCurrentCUDAStream());
    return x_out;
}

// ---------------------------------------------------------------------------
// Fused / naive weight-only GEMV: y[M] = W[M,K] @ x[K], E2M1 weight codes.
// bias_correction: optional [M] fp32 tensor (W_had_q @ mu, precomputed on the
// calibration side), added to each row's dot product. Pass an empty tensor
// to disable (the pre-mean-centering behavior).
// ---------------------------------------------------------------------------
torch::Tensor e2m1_fused_gemv(torch::Tensor W_codes, torch::Tensor W_alpha, torch::Tensor W_bias,
                               torch::Tensor x, int64_t K, torch::Tensor bias_correction) {
    CHECK_CUDA(W_codes); CHECK_CUDA(W_alpha); CHECK_CUDA(W_bias); CHECK_CUDA(x);
    CHECK_CONTIG(W_codes); CHECK_CONTIG(W_alpha); CHECK_CONTIG(W_bias); CHECK_CONTIG(x);
    TORCH_CHECK(W_codes.dtype() == torch::kUInt8, "W_codes must be uint8");
    TORCH_CHECK(W_alpha.dtype() == torch::kFloat16, "W_alpha must be fp16");
    TORCH_CHECK(W_bias.dtype() == torch::kUInt8, "W_bias must be uint8");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be fp16");

    int M = W_codes.size(0);
    auto y = torch::empty({M}, x.options());

    const float* bc_ptr = nullptr;
    if (bias_correction.numel() > 0) {
        CHECK_CUDA(bias_correction); CHECK_CONTIG(bias_correction);
        TORCH_CHECK(bias_correction.dtype() == torch::kFloat32, "bias_correction must be fp32");
        TORCH_CHECK(bias_correction.numel() == M, "bias_correction must have M elements");
        bc_ptr = bias_correction.data_ptr<float>();
    }

    launch_e2m1_fused_gemv(
        W_codes.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(W_alpha.data_ptr<at::Half>()),
        W_bias.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        M, (int)K, bc_ptr, at::cuda::getCurrentCUDAStream());
    return y;
}

torch::Tensor e2m1_naive_gemv(torch::Tensor W_codes, torch::Tensor W_alpha, torch::Tensor W_bias,
                               torch::Tensor x, int64_t K, torch::Tensor bias_correction) {
    CHECK_CUDA(W_codes); CHECK_CUDA(W_alpha); CHECK_CUDA(W_bias); CHECK_CUDA(x);
    CHECK_CONTIG(W_codes); CHECK_CONTIG(W_alpha); CHECK_CONTIG(W_bias); CHECK_CONTIG(x);

    int M = W_codes.size(0);
    auto y = torch::empty({M}, x.options());
    // Scratch buffer for the materialized fp16 weight matrix - this allocation
    // + the write/read of it is exactly the extra HBM traffic being measured.
    auto W_dequant = torch::empty({(int64_t)M * K}, x.options());

    const float* bc_ptr = nullptr;
    if (bias_correction.numel() > 0) {
        CHECK_CUDA(bias_correction); CHECK_CONTIG(bias_correction);
        TORCH_CHECK(bias_correction.dtype() == torch::kFloat32, "bias_correction must be fp32");
        TORCH_CHECK(bias_correction.numel() == M, "bias_correction must have M elements");
        bc_ptr = bias_correction.data_ptr<float>();
    }

    launch_e2m1_naive_gemv(
        W_codes.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(W_alpha.data_ptr<at::Half>()),
        W_bias.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(W_dequant.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        M, (int)K, bc_ptr, at::cuda::getCurrentCUDAStream());
    return y;
}

// ---------------------------------------------------------------------------
// GF4 weight-only GEMV (datapath analog): y[M] = W[M,K] @ x[K], GF4 weight
// codes decoded with the 8-entry Gaussian-quantile LUT (no per-block bias,
// unlike E2M1). W_alpha is [M, K/GF4_BLOCK] fp16 per-block scale. Used to
// measure the GF4 codebook decode + fp16 datapath on silicon against the
// dense-fp16 measuring stick; see gf4_fused_gemv_kernel.cu for why the
// benchmark decodes GF4-coded WEIGHTS.
// ---------------------------------------------------------------------------
torch::Tensor gf4_fused_gemv(torch::Tensor W_codes, torch::Tensor W_alpha,
                             torch::Tensor x, int64_t K, torch::Tensor bias_correction) {
    CHECK_CUDA(W_codes); CHECK_CUDA(W_alpha); CHECK_CUDA(x);
    CHECK_CONTIG(W_codes); CHECK_CONTIG(W_alpha); CHECK_CONTIG(x);
    TORCH_CHECK(W_codes.dtype() == torch::kUInt8, "W_codes must be uint8");
    TORCH_CHECK(W_alpha.dtype() == torch::kFloat16, "W_alpha must be fp16");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be fp16");

    int M = W_codes.size(0);
    auto y = torch::empty({M}, x.options());

    const float* bc_ptr = nullptr;
    if (bias_correction.numel() > 0) {
        CHECK_CUDA(bias_correction); CHECK_CONTIG(bias_correction);
        TORCH_CHECK(bias_correction.dtype() == torch::kFloat32, "bias_correction must be fp32");
        TORCH_CHECK(bias_correction.numel() == M, "bias_correction must have M elements");
        bc_ptr = bias_correction.data_ptr<float>();
    }

    launch_gf4_fused_gemv(
        W_codes.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(W_alpha.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        M, (int)K, bc_ptr, at::cuda::getCurrentCUDAStream());
    return y;
}

// Same fused GEMV but SHIFT-AND-ADD decode for the constrained-optimal codebook
// {0,2,4,6,8,10,12,16}/16 (no arbitrary-fp16 LUT). Times against gf4_fused_gemv.
torch::Tensor gf4_lattice_gemv(torch::Tensor W_codes, torch::Tensor W_alpha,
                               torch::Tensor x, int64_t K, torch::Tensor bias_correction) {
    CHECK_CUDA(W_codes); CHECK_CUDA(W_alpha); CHECK_CUDA(x);
    CHECK_CONTIG(W_codes); CHECK_CONTIG(W_alpha); CHECK_CONTIG(x);
    int M = W_codes.size(0);
    auto y = torch::empty({M}, x.options());
    const float* bc_ptr = nullptr;
    if (bias_correction.numel() > 0) {
        CHECK_CUDA(bias_correction); CHECK_CONTIG(bias_correction);
        bc_ptr = bias_correction.data_ptr<float>();
    }
    launch_gf4_lattice_gemv(
        W_codes.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(W_alpha.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        M, (int)K, bc_ptr, at::cuda::getCurrentCUDAStream());
    return y;
}

// Batched lattice GEMV (opt-2 + opt-3): x is [bs, K], returns y [bs, M].
// Loads W once per bs tokens (instead of bs separate kernel launches).
// Automatically tiles bs into the largest chunk that fits the 47 KB smem budget.
torch::Tensor gf4_lattice_gemv_batched(torch::Tensor W_codes, torch::Tensor W_alpha,
                                        torch::Tensor x, int64_t K, int64_t bs,
                                        torch::Tensor bias_correction) {
    CHECK_CUDA(W_codes); CHECK_CUDA(W_alpha); CHECK_CUDA(x);
    CHECK_CONTIG(W_codes); CHECK_CONTIG(W_alpha); CHECK_CONTIG(x);
    TORCH_CHECK(W_codes.dtype() == torch::kUInt8, "W_codes must be uint8");
    TORCH_CHECK(W_alpha.dtype() == torch::kFloat16, "W_alpha must be fp16");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be fp16");
    TORCH_CHECK(x.numel() == bs * K, "x must have bs*K elements");
    int M = W_codes.size(0);
    auto y = torch::empty({bs, M}, x.options());
    const float* bc_ptr = nullptr;
    if (bias_correction.numel() > 0) {
        CHECK_CUDA(bias_correction); CHECK_CONTIG(bias_correction);
        TORCH_CHECK(bias_correction.dtype() == torch::kFloat32, "bias_correction must be fp32");
        TORCH_CHECK(bias_correction.numel() == M, "bias_correction must have M elements");
        bc_ptr = bias_correction.data_ptr<float>();
    }
    launch_gf4_lattice_gemv_batched(
        W_codes.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(W_alpha.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        M, (int)K, (int)bs, bc_ptr, at::cuda::getCurrentCUDAStream());
    return y;
}

torch::Tensor gf4_naive_gemv(torch::Tensor W_codes, torch::Tensor W_alpha,
                             torch::Tensor x, int64_t K, torch::Tensor bias_correction) {
    CHECK_CUDA(W_codes); CHECK_CUDA(W_alpha); CHECK_CUDA(x);
    CHECK_CONTIG(W_codes); CHECK_CONTIG(W_alpha); CHECK_CONTIG(x);

    int M = W_codes.size(0);
    auto y = torch::empty({M}, x.options());
    auto W_dequant = torch::empty({(int64_t)M * K}, x.options());

    const float* bc_ptr = nullptr;
    if (bias_correction.numel() > 0) {
        CHECK_CUDA(bias_correction); CHECK_CONTIG(bias_correction);
        TORCH_CHECK(bias_correction.dtype() == torch::kFloat32, "bias_correction must be fp32");
        TORCH_CHECK(bias_correction.numel() == M, "bias_correction must have M elements");
        bc_ptr = bias_correction.data_ptr<float>();
    }

    launch_gf4_naive_gemv(
        W_codes.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(W_alpha.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(W_dequant.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        M, (int)K, bc_ptr, at::cuda::getCurrentCUDAStream());
    return y;
}

// All-fp16 measuring stick: dense GEMV over a native fp16 weight matrix.
torch::Tensor dense_fp16_gemv(torch::Tensor W, torch::Tensor x, torch::Tensor bias_correction) {
    CHECK_CUDA(W); CHECK_CUDA(x);
    CHECK_CONTIG(W); CHECK_CONTIG(x);
    TORCH_CHECK(W.dtype() == torch::kFloat16, "W must be fp16");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be fp16");
    TORCH_CHECK(W.dim() == 2, "W must be [M, K]");

    int M = W.size(0), K = W.size(1);
    auto y = torch::empty({M}, x.options());

    const float* bc_ptr = nullptr;
    if (bias_correction.numel() > 0) {
        CHECK_CUDA(bias_correction); CHECK_CONTIG(bias_correction);
        TORCH_CHECK(bias_correction.dtype() == torch::kFloat32, "bias_correction must be fp32");
        TORCH_CHECK(bias_correction.numel() == M, "bias_correction must have M elements");
        bc_ptr = bias_correction.data_ptr<float>();
    }

    launch_dense_fp16_gemv(
        reinterpret_cast<const __half*>(W.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        M, K, bc_ptr, at::cuda::getCurrentCUDAStream());
    return y;
}

// ---------------------------------------------------------------------------
// Hessian-weighted weight quantization (v5): the three-kernel pipeline.
// ---------------------------------------------------------------------------
torch::Tensor hessian_accumulate(torch::Tensor X, int64_t block_size) {
    CHECK_CUDA(X); CHECK_CONTIG(X);
    TORCH_CHECK(X.dtype() == torch::kFloat32, "X must be fp32");
    int n_tokens = X.size(0), M = X.size(1);
    TORCH_CHECK(M % block_size == 0, "M must be divisible by block_size");
    int n_blocks = M / block_size;
    auto H = torch::empty({n_blocks, block_size, block_size}, X.options());
    launch_hessian_accumulate(X.data_ptr<float>(), H.data_ptr<float>(), n_tokens, M,
                               (int)block_size, at::cuda::getCurrentCUDAStream());
    return H;
}

torch::Tensor hessian_damp_blocks(torch::Tensor H, double kappa, int64_t power_iters) {
    CHECK_CUDA(H); CHECK_CONTIG(H);
    int n_blocks = H.size(0), block_size = H.size(1);
    auto H_out = torch::empty_like(H);
    launch_hessian_damp_blocks(H.data_ptr<float>(), H_out.data_ptr<float>(), n_blocks,
                                block_size, (float)kappa, (int)power_iters,
                                at::cuda::getCurrentCUDAStream());
    return H_out;
}

std::vector<torch::Tensor> hessian_weight_solve(torch::Tensor W, torch::Tensor H_damped) {
    CHECK_CUDA(W); CHECK_CONTIG(W);
    CHECK_CUDA(H_damped); CHECK_CONTIG(H_damped);
    int N = W.size(0), M = W.size(1);
    int n_blocks = H_damped.size(0), block_size = H_damped.size(1);
    TORCH_CHECK(M == n_blocks * block_size, "W width must equal n_blocks*block_size of H_damped");

    auto W_codes = torch::empty({N, M / 2}, W.options().dtype(torch::kUInt8));
    auto W_alpha = torch::empty({N, n_blocks}, W.options().dtype(torch::kFloat16));
    auto W_bias  = torch::empty({N, n_blocks}, W.options().dtype(torch::kUInt8));

    launch_hessian_weight_solve(
        W.data_ptr<float>(), H_damped.data_ptr<float>(),
        W_codes.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(W_alpha.data_ptr<at::Half>()),
        W_bias.data_ptr<uint8_t>(),
        N, M, block_size, at::cuda::getCurrentCUDAStream());

    return {W_codes, W_alpha, W_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hadamard_fwht", &hadamard_fwht, "Fused randomized blockwise Hadamard transform (single kernel)",
          py::arg("x"), py::arg("block_size"), py::arg("d_sign"));
    m.def("hadamard_naive", &hadamard_naive, "Naive O(N^2) dense Hadamard baseline",
          py::arg("x"), py::arg("block_size"), py::arg("d_sign"));
    m.def("gf4_encode", &gf4_encode, "GF4 activation encoder (scale = rms*clip_ratio)",
          py::arg("x"), py::arg("clip_ratio") = 2.5, py::arg("fused") = true,
          py::arg("mu") = torch::empty({0}));
    m.def("gf4_encode_adaptive", &gf4_encode_adaptive,
          "GF4 per-block adaptive-clip encoder (online per-block MSE clip search; "
          "matches quantize_activations_gf4_adaptive in bit_split.py)",
          py::arg("x"), py::arg("clip_candidates"), py::arg("mu") = torch::empty({0}));
    m.def("gf4_decode", &gf4_decode, "GF4 decode: packed codes + scale -> dense fp32 activations",
          py::arg("codes"), py::arg("scales"), py::arg("n_blocks"));
    m.def("e2m1_fused_gemv", &e2m1_fused_gemv, "Fused E2M1-dequant + GEMV (no materialized W)",
          py::arg("W_codes"), py::arg("W_alpha"), py::arg("W_bias"), py::arg("x"), py::arg("K"),
          py::arg("bias_correction") = torch::empty({0}));
    m.def("e2m1_naive_gemv", &e2m1_naive_gemv, "Baseline: dequant-to-HBM kernel + dense fp16 GEMV",
          py::arg("W_codes"), py::arg("W_alpha"), py::arg("W_bias"), py::arg("x"), py::arg("K"),
          py::arg("bias_correction") = torch::empty({0}));
    m.def("gf4_lattice_gemv", &gf4_lattice_gemv, "Fused GEMV, shift-and-add decode (opt lattice codebook)",
          py::arg("W_codes"), py::arg("W_alpha"), py::arg("x"), py::arg("K"),
          py::arg("bias_correction") = torch::empty({0}));
    m.def("gf4_lattice_gemv_batched", &gf4_lattice_gemv_batched,
          "Batched lattice GEMV: x=[bs,K], y=[bs,M]; loads W once across all bs tokens",
          py::arg("W_codes"), py::arg("W_alpha"), py::arg("x"), py::arg("K"), py::arg("bs"),
          py::arg("bias_correction") = torch::empty({0}));
    m.def("gf4_fused_gemv", &gf4_fused_gemv, "Fused GF4-dequant + GEMV (no materialized W; 8-entry LUT)",
          py::arg("W_codes"), py::arg("W_alpha"), py::arg("x"), py::arg("K"),
          py::arg("bias_correction") = torch::empty({0}));
    m.def("gf4_naive_gemv", &gf4_naive_gemv, "GF4 baseline: dequant-to-HBM kernel + dense fp16 GEMV",
          py::arg("W_codes"), py::arg("W_alpha"), py::arg("x"), py::arg("K"),
          py::arg("bias_correction") = torch::empty({0}));
    m.def("dense_fp16_gemv", &dense_fp16_gemv, "All-fp16 measuring stick: dense GEMV over native fp16 W",
          py::arg("W"), py::arg("x"), py::arg("bias_correction") = torch::empty({0}));
    m.def("hessian_accumulate", &hessian_accumulate, "H = X^T X / n_tokens, block-diagonal");
    m.def("hessian_damp_blocks", &hessian_damp_blocks,
          "Condition-number damping via shifted power iteration (approximates eigvalsh)");
    m.def("hessian_weight_solve", &hessian_weight_solve,
          "Per-(row,block) Hessian-weighted E2M1 alpha/bias solve (reconstruct_layer_fp_blockdiag_scaled_v5)");
}

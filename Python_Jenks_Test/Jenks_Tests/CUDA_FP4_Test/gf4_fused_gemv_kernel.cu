// gf4_fused_gemv_kernel.cu
//
// Fused GF4-dequant + GEMV, y[M] = W[M,K] @ x[K], as a hardware measuring
// stick for the GF4 datapath. Three paths are provided so bench.py can time
// them against each other on real silicon:
//
//   1. gf4_fused_wonly_gemv   - reads 4-bit GF4 codes, decodes on-the-fly in
//                               registers (8-entry LUT + sign + per-block
//                               scale), FMAs at fp16-input / fp32-accumulate.
//                               The dequantized W never touches HBM.
//   2. gf4_naive_gemv         - the fake-quantization protocol as two kernels:
//                               dequantize codes -> fp16 W in HBM, then a dense
//                               fp16 GEMV reads it back. Extra M*K*2 bytes of
//                               avoidable traffic.
//   3. dense_fp16_gemv        - the ALL-FP16 roofline. Weights are stored
//                               natively as fp16 in HBM (no quantization at
//                               all) and streamed once. This is the measuring
//                               stick: GF4's speedup is (dense_fp16 time) /
//                               (gf4_fused time), i.e. the pure memory-traffic
//                               win from reading 4-bit codes instead of 16-bit
//                               weights, with an identical fp16 multiplier on
//                               both sides.
//
// WHY GF4-CODED WEIGHTS HERE (a deliberate datapath analog, not the accuracy
// recipe): in the real software pipeline GF4 codes ACTIVATIONS and weights are
// quantized separately with E2M1 (see e2m1_common.cuh / gf4_common.cuh). This
// kernel instead decodes GF4-coded WEIGHTS in a W4A16 GEMV purely to measure
// the GF4 codebook's decode cost on silicon in the same weight-stationary
// structure as e2m1_fused_gemv_kernel.cu. Because the two fused kernels are
// byte-for-byte identical except for the decode LUT (GF4 needs no per-block
// exponent bias that E2M1 carries), the difference (fused_gf4 - fused_e2m1)
// isolates the 8-entry Gaussian-quantile lookup, and (dense_fp16 - fused_gf4)
// is the memory-traffic speedup. Together they corroborate, on real hardware,
// the decode-overhead and speedup that the Timeloop/Accelergy models predict
// analytically; the multiply-format cost (4-bit vs fp16) cannot be measured on
// this GPU, which has no int4 multiplier, and remains a modeling question.
//
// GF4 storage layout (W4A16, weight-only):
//   W_codes: [M, K/2] packed bytes (2 codes/byte, low nibble = even column).
//            each code = bit 3 sign, bits 2:0 magnitude index into GF4_LEVEL.
//   W_alpha: [M, K/GF4_BLOCK] fp16 per-block scale (block_rms * clip_ratio).
//   x:       [K] fp16 dense activation vector.
//   y:       [M] fp16 output.
//   bias_correction: optional [M] fp32 per-row correction (mean-centering);
//                    pass nullptr to disable. Mirrors the E2M1 kernel.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "gf4_common.cuh"

#ifndef GEMV_ROWS_PER_BLOCK
#define GEMV_ROWS_PER_BLOCK 8   // 8 warps per block -> 8 output rows per block
#endif

// GF4_BLOCK (per-block scale granularity) comes from gf4_common.cuh (default 32).

// NOTE on tuning: half2/__hmul2 arithmetic and 2-way uint4 ILP were tried and
// MEASURED SLOWER (11008x4096 cold-L2 1.70x -> 1.55x). The kernel is not
// compute-bound - at ~190-230 GB/s it is memory-latency/occupancy-bound (only
// 8 warps/block), so arithmetic changes don't help and the extra ILP registers
// cut occupancy. The scalar-decode form below (vectorized uint4 loads + fp16 x
// staging) is the fastest measured; further speedup needs a different
// parallelization (split-K / more rows in flight), not a hotter inner loop.

// ---------------------------------------------------------------------------
// 1. Fused GF4-decode GEMV: decode-on-the-fly, never materializes fp16 W.
// One warp per output row; warp-shuffle reduce to the row dot product. x is
// staged into shared memory (as fp16) once per block and reused by every warp.
//
// Vectorized: each lane loads a uint4 (128-bit = 16 packed bytes = 32 codes)
// per iteration, so a warp reads a contiguous 512-byte segment in 32 coalesced
// 128-bit transactions instead of 32 single-byte reads. Because GF4_BLOCK == 32
// and one uint4 holds exactly 32 codes whose first weight index k0 = 32*v is a
// multiple of 32, every vector falls entirely inside ONE per-block scale --
// so the scale is loaded once per uint4 (row_alpha[v]), never per code. This
// requires K % 32 == 0 (always true for GF4_BLOCK-aligned weights), so there is
// no scalar tail. Staging x as __half (not fp32) halves shared-memory use,
// lifting occupancy and keeping large-K rows (e.g. K=11008 -> 22 KB) inside the
// default 48 KB smem budget.
// ---------------------------------------------------------------------------
__global__ void gf4_fused_wonly_gemv_kernel(const uint8_t* __restrict__ W_codes,
                                            const __half* __restrict__ W_alpha,
                                            const __half* __restrict__ x,
                                            __half* __restrict__ y,
                                            int M, int K,
                                            const float* __restrict__ bias_correction) {
    extern __shared__ __half x_sh[];   // K halves, staged once per block
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    for (int k = tid; k < K; k += threads) x_sh[k] = x[k];
    __syncthreads();

    const int warp = tid / 32;
    const int lane = tid & 31;
    const int row = blockIdx.x * GEMV_ROWS_PER_BLOCK + warp;   // one warp per output row
    if (row >= M) return;

    const int vecs_per_row = K / GF4_BLOCK;   // uint4s per row == blocks per row
    const uint4* row_codes4 = reinterpret_cast<const uint4*>(W_codes + (size_t)row * (K / 2));
    const __half* row_alpha = W_alpha + (size_t)row * vecs_per_row;

    float acc = 0.0f;
    // Each lane owns every 32nd uint4; 32 lanes cover a contiguous 512-byte
    // segment per step with coalesced 128-bit loads.
    for (int v = lane; v < vecs_per_row; v += 32) {
        uint4 p = row_codes4[v];               // 16 bytes = 32 codes = 32 weights = 1 block
        const int k0 = v * GF4_BLOCK;          // first weight index (multiple of 32)
        const float a = __half2float(row_alpha[v]);   // one scale per uint4
        const uint32_t words[4] = {p.x, p.y, p.z, p.w};
#pragma unroll
        for (int wi = 0; wi < 4; ++wi) {       // 4 words x 4 bytes x 2 codes = 32 codes
            const uint32_t word = words[wi];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint8_t byte = (word >> (8 * j)) & 0xFF;
                const int k = k0 + wi * 8 + j * 2;
                const float w0 = gf4_decode_one(byte & 0xF) * a;
                const float w1 = gf4_decode_one((byte >> 4) & 0xF) * a;
                acc += w0 * __half2float(x_sh[k]) + w1 * __half2float(x_sh[k + 1]);
            }
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);
    }
    if (lane == 0) {
        const float bc = (bias_correction != nullptr) ? bias_correction[row] : 0.0f;
        y[row] = __float2half(acc + bc);
    }
}

void launch_gf4_fused_gemv(const uint8_t* d_Wc, const __half* d_Wa,
                           const __half* d_x, __half* d_y, int M, int K,
                           const float* d_bias_correction, cudaStream_t stream) {
    dim3 block(GEMV_ROWS_PER_BLOCK * 32);
    dim3 grid((M + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK);
    size_t smem = (size_t)K * sizeof(__half);   // fp16 staging
    gf4_fused_wonly_gemv_kernel<<<grid, block, smem, stream>>>(d_Wc, d_Wa, d_x, d_y, M, K, d_bias_correction);
}

// ---------------------------------------------------------------------------
// 1b. Lattice-decode variant for the CONSTRAINED-OPTIMAL codebook
// {0,2,4,6,8,10,12,16}/16. Its levels are k/16 with k = 2*idx (idx<7) or 16
// (idx==7), so decode is a SHIFT-AND-ADD (a shift plus one compare) rather than
// a lookup of arbitrary fp16 constants -- no codebook LUT touched. Identical to
// the GF4 kernel above except gf4_decode_one -> lattice_decode_one; benchmark
// the two to isolate the decode cost inside the (memory-bound) fused GEMV.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float lattice_decode_one(uint8_t code) {
    const int idx = code & 0x7;
    const int k = (idx == 7) ? 16 : (idx << 1);   // {0,2,4,6,8,10,12,16}
    const float lvl = k * 0.0625f;                // * 1/16 (exact in fp32/fp16)
    return (code & 0x8) ? -lvl : lvl;
}

__global__ void gf4_fused_lattice_gemv_kernel(const uint8_t* __restrict__ W_codes,
                                              const __half* __restrict__ W_alpha,
                                              const __half* __restrict__ x,
                                              __half* __restrict__ y,
                                              int M, int K,
                                              const float* __restrict__ bias_correction) {
    // Opt-3: x staged as float (convert once at load, use directly in FMAs).
    // smem = K*4 bytes: K=4096->16KB, K=11008->43KB, both fit in 48KB budget.
    // Named x_shf (not x_sh) to avoid extern __shared__ type conflict with
    // gf4_fused_wonly_gemv_kernel which declares x_sh as __half in same TU.
    extern __shared__ float x_shf[];
    const int tid = threadIdx.x, threads = blockDim.x;
    for (int k = tid; k < K; k += threads) x_shf[k] = __half2float(x[k]);
    __syncthreads();
    const int warp = tid / 32, lane = tid & 31;
    const int row = blockIdx.x * GEMV_ROWS_PER_BLOCK + warp;
    if (row >= M) return;
    const int vecs_per_row = K / GF4_BLOCK;
    const uint4* row_codes4 = reinterpret_cast<const uint4*>(W_codes + (size_t)row * (K / 2));
    const __half* row_alpha = W_alpha + (size_t)row * vecs_per_row;
    float acc = 0.0f;
    for (int v = lane; v < vecs_per_row; v += 32) {
        uint4 p = row_codes4[v];
        const int k0 = v * GF4_BLOCK;
        const float a = __half2float(row_alpha[v]);
        const uint32_t words[4] = {p.x, p.y, p.z, p.w};
#pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            const uint32_t word = words[wi];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint8_t byte = (word >> (8 * j)) & 0xFF;
                const int k = k0 + wi * 8 + j * 2;
                const float w0 = lattice_decode_one(byte & 0xF) * a;
                const float w1 = lattice_decode_one((byte >> 4) & 0xF) * a;
                acc += w0 * x_shf[k] + w1 * x_shf[k + 1];
            }
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);
    if (lane == 0) {
        const float bc = (bias_correction != nullptr) ? bias_correction[row] : 0.0f;
        y[row] = __float2half(acc + bc);
    }
}

void launch_gf4_lattice_gemv(const uint8_t* d_Wc, const __half* d_Wa,
                             const __half* d_x, __half* d_y, int M, int K,
                             const float* d_bias_correction, cudaStream_t stream) {
    dim3 block(GEMV_ROWS_PER_BLOCK * 32);
    dim3 grid((M + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK);
    size_t smem = (size_t)K * sizeof(float);   // float staging (opt-3)
    gf4_fused_lattice_gemv_kernel<<<grid, block, smem, stream>>>(d_Wc, d_Wa, d_x, d_y, M, K, d_bias_correction);
}

// ---------------------------------------------------------------------------
// 1c. Batched lattice GEMV (opt-2): one warp handles one W-row and accumulates
// BS separate dot products simultaneously, loading W once from HBM instead of
// BS separate kernel launches each re-reading W.
//
// x layout: [BS, K] row-major (BS tokens stacked). Each token's K halves are
// staged into shared memory at launch, then the inner decode loop accumulates
// into BS independent float registers. Smem = BS*K*sizeof(__half).
//
// Smem budget (48 KB): BS=2,K=11008->44KB OK; BS=4,K=4096->32KB OK;
// BS=4,K=11008->88KB exceeds budget (launcher splits into two BS=2 calls).
// BS=8 available for K<=3000 (K=4096->64KB overflows; launcher uses BS=4 pairs).
//
// Bank conflicts: consecutive b-values at the same k share a bank (stride K is
// a multiple of 32 for K>=4096). This is BS-way broadcast (read-only smem
// access is conflict-free on Ampere+), so no real penalty on sm86/sm89.
// ---------------------------------------------------------------------------
template<int BS>
__global__ void gf4_lattice_gemv_batched_kernel(const uint8_t* __restrict__ W_codes,
                                                const __half*  __restrict__ W_alpha,
                                                const __half*  __restrict__ x,   // [BS, K]
                                                __half*        __restrict__ y,   // [BS, M]
                                                int M, int K,
                                                const float*   __restrict__ bias_correction) {
    extern __shared__ __half x_bsh[];   // [BS, K] halves
    const int tid = threadIdx.x, threads = blockDim.x;
#pragma unroll
    for (int b = 0; b < BS; b++)
        for (int k = tid; k < K; k += threads)
            x_bsh[(size_t)b * K + k] = x[(size_t)b * K + k];
    __syncthreads();

    const int warp = tid / 32, lane = tid & 31;
    const int row = blockIdx.x * GEMV_ROWS_PER_BLOCK + warp;
    if (row >= M) return;

    const int vecs_per_row = K / GF4_BLOCK;
    const uint4* row_codes4 = reinterpret_cast<const uint4*>(W_codes + (size_t)row * (K / 2));
    const __half* row_alpha = W_alpha + (size_t)row * vecs_per_row;

    float acc[BS] = {};

    for (int v = lane; v < vecs_per_row; v += 32) {
        uint4 p = row_codes4[v];
        const int k0 = v * GF4_BLOCK;
        const float a = __half2float(row_alpha[v]);
        const uint32_t words[4] = {p.x, p.y, p.z, p.w};
#pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            const uint32_t word = words[wi];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint8_t byte = (word >> (8 * j)) & 0xFF;
                const int k = k0 + wi * 8 + j * 2;
                const float w0 = lattice_decode_one(byte & 0xF) * a;
                const float w1 = lattice_decode_one((byte >> 4) & 0xF) * a;
#pragma unroll
                for (int b = 0; b < BS; b++) {
                    acc[b] += w0 * __half2float(x_bsh[(size_t)b * K + k])
                            + w1 * __half2float(x_bsh[(size_t)b * K + k + 1]);
                }
            }
        }
    }

    const float bc = (bias_correction != nullptr) ? bias_correction[row] : 0.0f;
#pragma unroll
    for (int b = 0; b < BS; b++) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            acc[b] += __shfl_xor_sync(0xFFFFFFFF, acc[b], offset);
        if (lane == 0)
            y[(size_t)b * M + row] = __float2half(acc[b] + bc);
    }
}

// Dispatch one chunk of bs_chunk tokens (must be 1, 2, 4, or 8).
static void _dispatch_lattice_batched(const uint8_t* Wc, const __half* Wa,
                                       const __half* x, __half* y,
                                       int M, int K, int bs_chunk,
                                       const float* bc, cudaStream_t stream) {
    dim3 block(GEMV_ROWS_PER_BLOCK * 32);
    dim3 grid((M + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK);
    if (bs_chunk == 1) {
        size_t smem = (size_t)K * sizeof(float);
        gf4_fused_lattice_gemv_kernel<<<grid, block, smem, stream>>>(Wc, Wa, x, y, M, K, bc);
    } else {
        size_t smem = (size_t)bs_chunk * K * sizeof(__half);
        switch (bs_chunk) {
            case 2: gf4_lattice_gemv_batched_kernel<2><<<grid, block, smem, stream>>>(Wc, Wa, x, y, M, K, bc); break;
            case 4: gf4_lattice_gemv_batched_kernel<4><<<grid, block, smem, stream>>>(Wc, Wa, x, y, M, K, bc); break;
            case 8: gf4_lattice_gemv_batched_kernel<8><<<grid, block, smem, stream>>>(Wc, Wa, x, y, M, K, bc); break;
        }
    }
}

// Launch batched lattice GEMV for arbitrary bs. Automatically picks the largest
// BS template (8/4/2/1) that keeps smem under 47 KB, then tiles over bs.
void launch_gf4_lattice_gemv_batched(const uint8_t* d_Wc, const __half* d_Wa,
                                      const __half* d_x, __half* d_y,
                                      int M, int K, int bs,
                                      const float* d_bc, cudaStream_t stream) {
    const size_t smem_limit = 47 * 1024;
    // Largest chunk size that fits smem budget
    int max_chunk = 1;
    for (int c : {8, 4, 2}) {
        if ((size_t)c * K * sizeof(__half) <= smem_limit) { max_chunk = c; break; }
    }
    int b = 0;
    while (b < bs) {
        int rem = bs - b;
        int chunk = (rem >= max_chunk) ? max_chunk : (rem >= 4 ? 4 : (rem >= 2 ? 2 : 1));
        // clamp chunk to largest template that fits smem
        while (chunk > 1 && (size_t)chunk * K * sizeof(__half) > smem_limit) chunk >>= 1;
        _dispatch_lattice_batched(d_Wc, d_Wa,
                                   d_x + (size_t)b * K,
                                   d_y + (size_t)b * M,
                                   M, K, chunk, d_bc, stream);
        b += chunk;
    }
}

// ---------------------------------------------------------------------------
// 2. Naive baseline: materialize dequantized fp16 W (kernel 1), then a dense
// fp16 GEMV reads it back (kernel 2). The fake-quantization protocol as an
// actual two-kernel pipeline, so bench.py can measure the avoidable
// M*K*2-byte round trip as wall-clock time.
// ---------------------------------------------------------------------------
__global__ void gf4_dequantize_to_fp16_kernel(const uint8_t* __restrict__ W_codes,
                                              const __half* __restrict__ W_alpha,
                                              __half* __restrict__ W_dequant,
                                              int M, int K) {
    const int row = blockIdx.y;
    const int col_pair = blockIdx.x * blockDim.x + threadIdx.x;   // one packed byte
    const int bytes_per_row = K / 2;
    if (col_pair >= bytes_per_row) return;

    const int blocks_per_row = K / GF4_BLOCK;
    uint8_t packed = W_codes[(size_t)row * bytes_per_row + col_pair];
    int k0 = col_pair * 2, k1 = k0 + 1;
    int blk0 = k0 / GF4_BLOCK, blk1 = k1 / GF4_BLOCK;
    float a0 = __half2float(W_alpha[(size_t)row * blocks_per_row + blk0]);
    float a1 = __half2float(W_alpha[(size_t)row * blocks_per_row + blk1]);
    W_dequant[(size_t)row * K + k0] = __float2half(gf4_decode_one(packed & 0xF) * a0);
    W_dequant[(size_t)row * K + k1] = __float2half(gf4_decode_one((packed >> 4) & 0xF) * a1);
}

// ---------------------------------------------------------------------------
// 3. Dense fp16 GEMV over NATIVE fp16 weights (the all-fp16 measuring stick).
// Distinct symbol from the E2M1 file's dense kernel so both .cu link together.
// ---------------------------------------------------------------------------
__global__ void dense_fp16_gemv_kernel(const __half* __restrict__ W, const __half* __restrict__ x,
                                       __half* __restrict__ y, int M, int K,
                                       const float* __restrict__ bias_correction) {
    extern __shared__ float x_shd[];
    for (int k = threadIdx.x; k < K; k += blockDim.x) x_shd[k] = __half2float(x[k]);
    __syncthreads();

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x / 32) + warp;
    if (row >= M) return;

    const __half* row_w = W + (size_t)row * K;
    float acc = 0.0f;
    for (int k = lane; k < K; k += 32) acc += __half2float(row_w[k]) * x_shd[k];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);
    if (lane == 0) {
        const float bc = (bias_correction != nullptr) ? bias_correction[row] : 0.0f;
        y[row] = __float2half(acc + bc);
    }
}

// Naive GF4 path: dequantize kernel + dense fp16 GEMV over the scratch buffer.
void launch_gf4_naive_gemv(const uint8_t* d_Wc, const __half* d_Wa,
                           __half* d_Wdequant_scratch, const __half* d_x, __half* d_y,
                           int M, int K, const float* d_bias_correction, cudaStream_t stream) {
    // Kernel 1: materialize the dequantized fp16 weight matrix in HBM.
    {
        dim3 block(256);
        dim3 grid(((K / 2) + block.x - 1) / block.x, M);
        gf4_dequantize_to_fp16_kernel<<<grid, block, 0, stream>>>(d_Wc, d_Wa, d_Wdequant_scratch, M, K);
    }
    // Kernel 2: dense fp16 GEMV reads that buffer back from HBM.
    {
        const int rows_per_block = 8;
        dim3 block(rows_per_block * 32);
        dim3 grid((M + rows_per_block - 1) / rows_per_block);
        size_t smem = (size_t)K * sizeof(float);
        dense_fp16_gemv_kernel<<<grid, block, smem, stream>>>(d_Wdequant_scratch, d_x, d_y, M, K, d_bias_correction);
    }
}

// All-fp16 roofline: dense GEMV over a native fp16 weight matrix (no codes).
void launch_dense_fp16_gemv(const __half* d_W, const __half* d_x, __half* d_y,
                            int M, int K, const float* d_bias_correction, cudaStream_t stream) {
    const int rows_per_block = 8;
    dim3 block(rows_per_block * 32);
    dim3 grid((M + rows_per_block - 1) / rows_per_block);
    size_t smem = (size_t)K * sizeof(float);
    dense_fp16_gemv_kernel<<<grid, block, smem, stream>>>(d_W, d_x, d_y, M, K, d_bias_correction);
}

"""
Hardware-accelerated kernels for GF4 inference.

Pipeline per linear layer:
  fwht_blockwise_fast  — torch.compile-able FWHT (replaces Python FWHT loop)
  gf4_quant_triton     — Triton kernel: per-block RMS + GF4 pack to INT4
  pack_weights_int4    — offline: map dequant FP4 weights → packed INT4
  int8_linear          — torch._int_mm (INT8 tensor cores, sm80+)

INT4 encoding: sign bit (bit 3) | 3-bit level index (bits 2-0).
Two 4-bit values packed per INT8 byte: low nibble = elem 2i, high nibble = elem 2i+1.
"""

import torch
import torch.nn.functional as F  # noqa: F401 (used by callers who import *)
import triton
import triton.language as tl

# ── GF4 constants ─────────────────────────────────────────────────────────────

# Nearest-level assignment thresholds (midpoints between GF4_POS entries)
_T0, _T1, _T2, _T3, _T4, _T5, _T6 = \
    0.03980, 0.12668, 0.22830, 0.33907, 0.46018, 0.61063, 0.84812

# GF4_POS dequantisation values (positive half)
GF4_LEVELS = torch.tensor(
    [0.0, 0.0796082, 0.1737177, 0.2828685, 0.3952704, 0.5250730, 0.6961928, 1.0],
    dtype=torch.float32,
)

# ── Hadamard utilities (pure PyTorch, no CUDA extension needed) ──────────────

def generate_random_signs(M: int, block_size: int, device,  # block_size kept for API compat
                          seed: int | None = None) -> torch.Tensor:
    """Per-element ±1 sign vector for randomized Hadamard, shape [M]."""
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        return torch.randint(0, 2, (M,), device=device, generator=gen).float() * 2 - 1
    return torch.randint(0, 2, (M,), device=device).float() * 2 - 1


def fwht_blockwise(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Pure-PyTorch normalised blockwise FWHT (reference / calibration path)."""
    orig = x.shape
    M = orig[-1]
    n_blocks = M // block_size
    x = x.reshape(*orig[:-1], n_blocks, block_size)
    h = 1
    while h < block_size:
        x = x.reshape(*orig[:-1], n_blocks, block_size // (2 * h), 2, h)
        a, b = x[..., 0, :], x[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2)
        x = x.reshape(*orig[:-1], n_blocks, block_size)
        h *= 2
    return (x * (block_size ** -0.5)).reshape(orig)


# ── Fast FWHT (torch.compile-friendly, replaces the Python butterfly loop) ────

@torch.compile(fullgraph=True)
def fwht_blockwise_fast(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Normalised block-wise FWHT via torch.compile (reference fast path).
    Still launches log2(block_size) separate CUDA ops — use fwht_hadacore
    for the single-kernel L2-cached version.
    """
    orig = x.shape
    M = orig[-1]
    n_blocks = M // block_size
    x = x.reshape(*orig[:-1], n_blocks, block_size)

    h = 1
    while h < block_size:
        x = x.reshape(*orig[:-1], n_blocks, block_size // (2 * h), 2, h)
        a = x[..., 0, :]
        b = x[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2)
        x = x.reshape(*orig[:-1], n_blocks, block_size)
        h *= 2

    return (x * (block_size ** -0.5)).reshape(orig)


# ── HadaCore-style single-kernel FWHT (L2-cached butterfly stages) ────────────
#
# HadaCore (QuaRot et al., 2024) keeps all log2(N) butterfly stages inside
# a single kernel so intermediate values hit L2 cache (~3 TB/s) rather than
# DRAM (~716 GB/s).  This gives ~10-20× over torch.compile for large blocks.
#
# Implementation: one Triton program per (row, had_block).  Each program owns
# a contiguous slice of a scratch tensor; it writes/reads that slice through
# the L2 cache for all log2(N) butterfly stages before writing the final
# result.  No __syncthreads needed — each program instance touches only its
# own scratch slice exclusively.

@triton.jit
def _fwht_hadacore_kernel(
    x_ptr,       # [T, M] fp32 input
    out_ptr,     # [T, M] fp32 output  (also ping buffer)
    sc_ptr,      # [T, M] fp32 scratch (pong buffer)
    T, M,
    stride_T,
    HAD_BS: tl.constexpr,   # block size (power-of-2)
    LOG2_BS: tl.constexpr,  # log2(HAD_BS)
    NORM: tl.constexpr,     # 1 / sqrt(HAD_BS)
):
    """
    HadaCore single-kernel FWHT with ping-pong double buffering.

    Each stage reads from one buffer and writes to the other, so reads
    and writes never alias within a stage — eliminating the L1-cache
    coherence issue that occurs with in-place butterfly updates.
    The debug_barrier between stages enforces visibility across warps.
    """
    pid     = tl.program_id(0)
    n_hblks = M // HAD_BS
    row     = pid // n_hblks
    hblk    = pid %  n_hblks

    if row >= T:
        return

    base = row * stride_T + hblk * HAD_BS
    offs = tl.arange(0, HAD_BS)

    # Copy input into out_ptr (ping buffer for stage 0)
    tl.store(out_ptr + base + offs, tl.load(x_ptr + base + offs))
    tl.debug_barrier()

    # LOG2_BS butterfly stages with explicit ping-pong buffering.
    # Stage log_h reads from src and writes to dst, then they swap.
    # src always points to the buffer written by the PREVIOUS stage.
    #
    # Stage 0: src=out_ptr, dst=sc_ptr
    # Stage 1: src=sc_ptr, dst=out_ptr
    # Stage 2: src=out_ptr, dst=sc_ptr   …and so on.
    #
    # Because log_h from tl.static_range is a compile-time constant,
    # the if/else is resolved at JIT compile time (different code per stage).
    for log_h in tl.static_range(LOG2_BS):
        h         = 1 << log_h
        pair_offs = tl.arange(0, HAD_BS // 2)
        group     = pair_offs // h
        within    = pair_offs %  h
        a_offs    = group * (2 * h) + within
        b_offs    = a_offs + h

        if log_h % 2 == 0:          # even stage: out_ptr → sc_ptr
            a = tl.load(out_ptr + base + a_offs)
            b = tl.load(out_ptr + base + b_offs)
            tl.store(sc_ptr + base + a_offs, a + b)
            tl.store(sc_ptr + base + b_offs, a - b)
        else:                        # odd stage:  sc_ptr → out_ptr
            a = tl.load(sc_ptr + base + a_offs)
            b = tl.load(sc_ptr + base + b_offs)
            tl.store(out_ptr + base + a_offs, a + b)
            tl.store(out_ptr + base + b_offs, a - b)

        tl.debug_barrier()

    # Last stage index = LOG2_BS - 1.
    # Even last stage (LOG2_BS odd)  → result written to sc_ptr.
    # Odd  last stage (LOG2_BS even) → result written to out_ptr.
    if (LOG2_BS - 1) % 2 == 0:      # LOG2_BS is odd → result in sc_ptr
        result = tl.load(sc_ptr + base + offs)
    else:                            # LOG2_BS is even → result in out_ptr
        result = tl.load(out_ptr + base + offs)

    tl.store(out_ptr + base + offs, result * NORM)


def fwht_hadacore(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    HadaCore-style single-kernel blockwise FWHT.

    All log2(block_size) butterfly stages run in one Triton kernel.
    Ping-pong double buffering ensures correct inter-warp ordering
    without relying on L1 cache coherence.

    x: [..., M] fp32  (M divisible by block_size, block_size power-of-2)
    Returns same shape and dtype.
    """
    assert (block_size & (block_size - 1)) == 0
    orig = x.shape
    x2   = x.reshape(-1, orig[-1]).contiguous()
    T, M = x2.shape
    assert M % block_size == 0

    out     = torch.empty_like(x2)
    scratch = torch.empty_like(x2)
    log2_bs = block_size.bit_length() - 1

    grid = (T * (M // block_size),)
    _fwht_hadacore_kernel[grid](
        x2, out, scratch,
        T, M, x2.stride(0),
        HAD_BS  = block_size,
        LOG2_BS = log2_bs,
        NORM    = block_size ** -0.5,
    )
    return out.reshape(orig)


# ── Triton: GF4 quantise a [T, M] float16 tensor ─────────────────────────────
# Input is already in the Hadamard domain (post-FWHT, post-μ-subtract).
# Each Triton program handles one row × one GF4-block of 16 elements.

@triton.jit
def _gf4_quant_kernel(
    x_ptr,       # [T, M] fp16 input (Hadamard domain)
    idx_ptr,     # [T, M//2] int8 output packed GF4 indices
    sc_ptr,      # [T, M//GF4_BS] fp16 output block scales
    T, M,
    stride_xT, stride_xM,
    stride_iT,
    stride_sT,
    GF4_BS: tl.constexpr,   # 16
    CLIP: tl.constexpr,     # clip ratio (e.g. 2.5 stored as float)
):
    pid  = tl.program_id(0)
    n_gb = M // GF4_BS
    row  = pid // n_gb
    gb   = pid %  n_gb

    if row >= T:
        return

    offs = tl.arange(0, GF4_BS)
    x_base = row * stride_xT + gb * GF4_BS * stride_xM
    xb = tl.load(x_ptr + x_base + offs * stride_xM).to(tl.float32)

    # Per-block RMS scale
    rms   = tl.sqrt(tl.sum(xb * xb) / GF4_BS + 1e-16)
    scale = rms * CLIP
    tl.store(sc_ptr + row * stride_sT + gb, scale.to(tl.float16))

    # Nearest GF4 level — 7 threshold comparisons on |x|/scale.
    # Thresholds inlined as literals (Triton can't read module-level floats).
    xn  = tl.abs(xb) / tl.maximum(scale, 1e-8)
    xn  = tl.minimum(xn, 1.0)
    lvl = (xn > 0.03980).to(tl.int8) + (xn > 0.12668).to(tl.int8) \
        + (xn > 0.22830).to(tl.int8) + (xn > 0.33907).to(tl.int8) \
        + (xn > 0.46018).to(tl.int8) + (xn > 0.61063).to(tl.int8) \
        + (xn > 0.84812).to(tl.int8)                          # 0..7

    sign_bit = (xb < 0.0).to(tl.int8) << 3   # bit 3
    nibble   = sign_bit | lvl                  # 4-bit value in [0..15]

    # Store each nibble as a full int8 (unpacked).
    # Packing two nibbles per byte is done by gf4_pack() in Python after the
    # kernel returns — Triton 3.x lacks tl.gather/scatter for this pattern.
    tl.store(idx_ptr + row * stride_iT + gb * GF4_BS + tl.arange(0, GF4_BS),
             nibble)


def gf4_quant(x: torch.Tensor,
              clip_ratio: float = 2.5,
              gf4_block: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantise a Hadamard-domain activation tensor to GF4.

    x: [T, M] fp16/fp32  (post-FWHT, post-μ-subtract)
    Returns:
        idx:    [T, M]              int8  one nibble per byte (unpacked)
        scales: [T, M//gf4_block]  fp16  per-block scales
    """
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    T, M = x2.shape
    assert M % gf4_block == 0

    idx    = torch.empty((T, M),              dtype=torch.int8,   device=x.device)
    scales = torch.empty((T, M // gf4_block), dtype=torch.float16, device=x.device)

    grid = (T * (M // gf4_block),)
    _gf4_quant_kernel[grid](
        x2, idx, scales,
        T, M,
        x2.stride(0), x2.stride(1),
        idx.stride(0),
        scales.stride(0),
        GF4_BS = gf4_block,
        CLIP   = clip_ratio,
    )
    return idx, scales


def gf4_pack(idx: torch.Tensor) -> torch.Tensor:
    """Pack unpacked GF4 nibbles [T, M] int8 → [T, M//2] int8."""
    lo = idx[:, 0::2] & 0xF
    hi = idx[:, 1::2] & 0xF
    return (lo | (hi << 4)).to(torch.int8)


# ── Dequantise GF4 activations ────────────────────────────────────────────────

def gf4_dequant(idx: torch.Tensor,
                scales: torch.Tensor,
                gf4_block: int = 16) -> torch.Tensor:
    """
    Dequantise GF4 activations to fp16.
    idx:    [T, M]              int8  (one nibble per byte, from gf4_quant)
    scales: [T, M//gf4_block]  fp16
    Returns [T, M] fp16.
    """
    T, M = idx.shape
    cb = GF4_LEVELS.to(idx.device)

    nib  = idx.to(torch.int32)
    sign = (nib >> 3).bool()
    lvl  = nib & 0x7
    vals = cb[lvl]
    vals = torch.where(sign, -vals, vals)

    sc  = scales.float().unsqueeze(-1).expand(-1, -1, gf4_block)
    out = (vals.view(T, M // gf4_block, gf4_block) * sc).view(T, M)
    return out.to(torch.float16)


# ── INT4 weight packing ───────────────────────────────────────────────────────

def pack_weights_int4(weight_q: torch.Tensor,
                      codebook: torch.Tensor) -> torch.Tensor:
    """
    Map dequantised fp16 weight matrix to packed INT4.

    weight_q: [N, M] fp16  dequantised E2M1 values
    codebook: [8]    fp32  positive-half E2M1 codebook entries

    Returns [N, M//2] int8  (two 4-bit indices per byte).
    """
    w   = weight_q.float()
    cb  = codebook.to(w.device).float()  # [8] positive values

    sign_bit = (w < 0).to(torch.int8)
    w_abs    = w.abs()

    dist = (w_abs.unsqueeze(-1) - cb.view(1, 1, -1)).abs()   # [N, M, 8]
    idx  = dist.argmin(dim=-1).to(torch.int8)                 # [N, M]
    nibble = (sign_bit << 3) | idx                            # 4-bit encoded

    lo = nibble[:, 0::2] & 0xF
    hi = nibble[:, 1::2] & 0xF
    return (lo | (hi << 4)).to(torch.int8)                    # [N, M//2]


def unpack_weights_int4(packed: torch.Tensor,
                        codebook: torch.Tensor) -> torch.Tensor:
    """
    Unpack INT4 weights to fp32 for dequant-fused GEMM.
    packed:   [N, M//2] int8
    codebook: [8]       fp32 positive-half values
    Returns   [N, M]    fp32.
    """
    cb  = codebook.to(packed.device).float()
    lo  = (packed & 0x0F).to(torch.int32)
    hi  = ((packed >> 4) & 0x0F).to(torch.int32)
    slo = (lo >> 3).bool();  ilo = lo & 0x7
    shi = (hi >> 3).bool();  ihi = hi & 0x7

    vlo = cb[ilo];  vlo = torch.where(slo, -vlo, vlo)
    vhi = cb[ihi];  vhi = torch.where(shi, -vhi, vhi)

    N, Mp2 = packed.shape
    out = torch.empty((N, Mp2 * 2), dtype=torch.float32, device=packed.device)
    out[:, 0::2] = vlo
    out[:, 1::2] = vhi
    return out


# ── INT8 GEMM via torch._int_mm ───────────────────────────────────────────────

def int8_linear(a_i8: torch.Tensor, b_i8: torch.Tensor,
                a_scale: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
    """
    INT8 × INT8 GEMM using tensor cores (sm80+, requires T > 16).

    a_i8:    [T, M] int8   uniform-quantised activations
    b_i8:    [N, M] int8   uniform-quantised weights
    a_scale: [T]    fp32   per-row activation scale
    b_scale: [N]    fp32   per-output-channel weight scale

    Falls back to fp32 GEMM for T ≤ 16 (tensor core minimum tile constraint).
    Returns [T, N] fp16.
    """
    T = a_i8.shape[0]
    if T <= 16:
        # Tensor cores require T > 16; dequant and use fp32
        a_fp = a_i8.float() * a_scale.unsqueeze(1)   # [T, M] * [T, 1]
        b_fp = b_i8.float() * b_scale.unsqueeze(1)   # [N, M] * [N, 1]
        return torch.mm(a_fp, b_fp.t()).to(torch.float16)
    acc = torch._int_mm(a_i8, b_i8.t())                              # [T, N] int32
    return (acc.float() * a_scale.unsqueeze(1) * b_scale.unsqueeze(0)).to(torch.float16)

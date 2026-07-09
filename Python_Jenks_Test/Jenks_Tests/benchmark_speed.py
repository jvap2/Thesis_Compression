"""
Throughput benchmark: FP16 vs W4A16 vs W4A4 (GF4) for a single linear layer
and for a full model forward pass.

Run:
    python3 benchmark_speed.py
"""

import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6;8.9"

import sys, time, math
import torch
import triton
import torch.nn as nn
import torch.nn.functional as F

# Import triton_kernels directly — bypasses FP_Quantization_Experiments/__init__.py
# which would otherwise pull in bit_split.py and trigger a CUDA extension rebuild.
_tk_path = os.path.join(os.path.dirname(__file__),
                        "FP_Quantization_Experiments", "triton_kernels.py")
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("triton_kernels", _tk_path)
_tk   = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_tk)

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                               "FP_Quantization_Experiments"))
from triton_kernels import (  # noqa — direct module, no __init__ side-effects
    fwht_blockwise, fwht_blockwise_fast, fwht_hadacore,
    gf4_quant, gf4_dequant,
    pack_weights_int4, unpack_weights_int4, int8_linear,
    generate_random_signs, GF4_LEVELS,
)

def get_fp4_codebook(e_bits, m_bits, bias, device):
    """Minimal E2M1 codebook (positive half) — avoids importing bit_split."""
    import torch
    e_levels = torch.arange(0, 2**e_bits, device=device).float()
    m_levels = torch.arange(0, 2**m_bits, device=device).float()
    base = 2.0 ** (e_levels - bias)
    mant = 1.0 + m_levels / (2**m_bits)
    return (base.unsqueeze(1) * mant.unsqueeze(0)).reshape(-1)

def quantize_activations_gf4(x, block_size, clip_ratio=2.5, levels=None):
    """Reference GF4 quantise (pure PyTorch) — avoids importing bit_split."""
    import torch
    import torch.nn.functional as F
    from triton_kernels import GF4_LEVELS  # loaded via direct sys.path above
    orig_shape, orig_dtype = x.shape, x.dtype
    x2 = x.reshape(-1, orig_shape[-1]).float()
    N, K = x2.shape
    pad = (block_size - K % block_size) % block_size
    xp = F.pad(x2, (0, pad))
    xb = xp.reshape(N * (xp.shape[1] // block_size), block_size)
    rms   = xb.pow(2).mean(-1).sqrt().clamp(min=1e-8)
    scale = (rms * clip_ratio).unsqueeze(-1)
    lvls  = (GF4_LEVELS if levels is None else levels).to(x.device)
    xn    = (xb.abs() / scale).clamp(0, 1)
    dist  = (xn.unsqueeze(-1) - lvls.view(1, 1, -1)).abs()
    q_lvl = lvls[dist.argmin(-1)]
    out   = (torch.sign(xb) * scale * q_lvl).reshape(N, xp.shape[1])
    if pad: out = out[:, :K]
    return out.reshape(orig_shape).to(orig_dtype)

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# ─────────────────────────────────────────────────────────────────────────────
# Timing helper
# ─────────────────────────────────────────────────────────────────────────────

def cuda_time_ms(fn, n_warmup=10, n_iter=100):
    """Return median wall-clock time in ms for one call of fn()."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]  # median


# ─────────────────────────────────────────────────────────────────────────────
# Single linear-layer benchmark
# ─────────────────────────────────────────────────────────────────────────────

def bench_linear_layer(T: int, M: int, N: int, had_block_size: int = 256):
    """
    Compare inference paths for one W4A4 linear layer at token batch T.

    Paths:
      fp16      — standard nn.Linear in fp16
      w4a16     — fp16 acts × dequant fp16 weights (weight bandwidth halved)
      w4a4_sim  — simulated: FWHT + GF4 quant + dequant + fp16 GEMM
      w4a4_fast — FWHT (compiled) + Triton GF4 quant + fp16 GEMM with packed w
      int8      — uniform INT8 × INT8 via torch._int_mm (tensor cores)
    """
    print(f"\n── Linear layer T={T}, M={M}, N={N}, had_bs={had_block_size} ──")

    x_fp16  = torch.randn(T, M, dtype=torch.float16, device=device)
    W_fp16  = torch.randn(N, M, dtype=torch.float16, device=device) * 0.02
    D       = generate_random_signs(M, had_block_size, device, seed=42).to(device)
    mu      = torch.zeros(M, dtype=torch.float32, device=device)
    cb_pos  = get_fp4_codebook(2, 1, bias=1, device=device)  # [4] E2M1 positive half

    # Pre-pack weights for w4a4_fast
    W_packed = pack_weights_int4(W_fp16, cb_pos)   # [N, M//2] int8
    W_dequant = unpack_weights_int4(W_packed, cb_pos).to(torch.float16)  # [N, M]

    # ── fp16 baseline ────────────────────────────────────────────────────────
    t_fp16 = cuda_time_ms(lambda: F.linear(x_fp16, W_fp16))
    print(f"  fp16 baseline:  {t_fp16:.3f} ms")

    # ── w4a16: fp16 acts × dequant weights ───────────────────────────────────
    t_w4a16 = cuda_time_ms(lambda: F.linear(x_fp16, W_dequant))
    print(f"  w4a16 (dequant weight): {t_w4a16:.3f} ms  "
          f"({t_fp16/t_w4a16:.2f}× vs fp16)")

    # ── w4a4 simulated (current Python path) ─────────────────────────────────
    def sim_forward():
        x_h = fwht_blockwise(x_fp16.float() * D, had_block_size) - mu
        x_q = quantize_activations_gf4(x_h, 16, clip_ratio=2.5).to(torch.float16)
        return F.linear(x_q, W_dequant)

    t_sim = cuda_time_ms(sim_forward)
    print(f"  w4a4 simulated (Python FWHT): {t_sim:.3f} ms  "
          f"({t_fp16/t_sim:.2f}× vs fp16)")

    # ── w4a4 fast: compiled FWHT + Triton GF4 quant ──────────────────────────
    def fast_forward():
        x_h = fwht_blockwise_fast(x_fp16.float() * D, had_block_size) - mu
        idx, sc = gf4_quant(x_h, clip_ratio=2.5)
        x_dq = gf4_dequant(idx, sc)
        return F.linear(x_dq, W_dequant)

    fast_forward()  # warm up torch.compile
    t_fast = cuda_time_ms(fast_forward)
    print(f"  w4a4 fast (compile+Triton):   {t_fast:.3f} ms  "
          f"({t_fp16/t_fast:.2f}× vs fp16, {t_sim/t_fast:.2f}× vs sim)")

    # ── w4a4 HadaCore: single-kernel FWHT + Triton GF4 quant ─────────────────
    def hada_forward():
        x_h = fwht_hadacore(x_fp16.float() * D, had_block_size) - mu
        idx, sc = gf4_quant(x_h, clip_ratio=2.5)
        x_dq = gf4_dequant(idx, sc)
        return F.linear(x_dq, W_dequant)

    hada_forward()  # warm up Triton JIT
    t_hada = cuda_time_ms(hada_forward)
    print(f"  w4a4 HadaCore+Triton:         {t_hada:.3f} ms  "
          f"({t_fp16/t_hada:.2f}× vs fp16, {t_fast/t_hada:.2f}× vs fast)")

    # ── INT8 tensor core GEMM ─────────────────────────────────────────────────
    # Quantise both x and W to INT8 with per-row/per-col scales
    x_abs_max = x_fp16.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-8)  # [T,1]
    a_scale   = (x_abs_max / 127.0).squeeze(1).to(torch.float32)
    a_i8      = (x_fp16.float() / x_abs_max * 127).clamp(-127, 127).to(torch.int8)

    w_abs_max = W_fp16.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-8)  # [N,1]
    b_scale   = (w_abs_max / 127.0).squeeze(1).to(torch.float32)
    b_i8      = (W_fp16.float() / w_abs_max * 127).clamp(-127, 127).to(torch.int8)

    t_i8 = cuda_time_ms(lambda: int8_linear(a_i8, b_i8, a_scale, b_scale))
    print(f"  INT8 tensor core GEMM:        {t_i8:.3f} ms  "
          f"({t_fp16/t_i8:.2f}× vs fp16)")

    return dict(fp16=t_fp16, w4a16=t_w4a16, w4a4_sim=t_sim,
                w4a4_fast=t_fast, int8=t_i8)


# ─────────────────────────────────────────────────────────────────────────────
# Full model forward benchmark (uses calibrated checkpoint if available)
# ─────────────────────────────────────────────────────────────────────────────

def bench_model(model_name: str = "facebook/opt-125m",
                batch_sizes=(1, 8, 32)):
    print(f"\n{'='*60}")
    print(f"Full model: {model_name}")
    print(f"{'='*60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device).eval()

    text = "The quick brown fox jumps over the lazy dog"
    enc  = tokenizer([text] * 1, return_tensors="pt",
                     truncation=True, max_length=64)
    ids  = enc.input_ids.to(device)   # [1, seq_len]
    seq  = ids.shape[1]

    print(f"\nSequence length: {seq} tokens")
    print(f"{'Batch':>6}  {'fp16 ms':>10}  {'tok/s':>10}")
    print("-" * 32)

    results_fp16 = {}
    for bs in batch_sizes:
        x = ids.expand(bs, -1)
        t = cuda_time_ms(lambda: model_fp16(x), n_warmup=5, n_iter=50)
        tps = bs * seq / (t / 1000)
        results_fp16[bs] = (t, tps)
        print(f"  {bs:4d}  {t:10.2f}  {tps:10.0f}")

    # Load quantised model if checkpoint exists
    ckpt_path = os.path.join(os.path.dirname(__file__),
                             "Jenks_Tests", "quant_model_ckpt.pt")
    if not os.path.exists(ckpt_path):
        print("\nNo quant checkpoint found — skipping quantised model benchmark.")
        print(f"(Run FP_QuantNetworkTest_LLM.py first to generate {ckpt_path})")
        return

    print(f"\nLoading quantised model from {ckpt_path}")
    # Import bit_split directly to avoid __init__.py triggering CUDA JIT build
    import importlib.util as _ilu2
    _bs_spec = _ilu2.spec_from_file_location(
        "bit_split",
        os.path.join(os.path.dirname(__file__),
                     "FP_Quantization_Experiments", "bit_split.py"))
    _bs = _ilu2.module_from_spec(_bs_spec)
    _bs_spec.loader.exec_module(_bs)
    quantize_model_fp = _bs.quantize_model_fp
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    def collate(batch):
        texts = [b["text"] for b in batch if b["text"].strip()]
        if not texts:
            return None, None
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=64)
        ids_ = enc.input_ids.to(device)
        return ids_, ids_

    calib_loader = DataLoader(dataset, batch_size=8, shuffle=False,
                              collate_fn=collate)

    model_q = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device).eval()
    model_q = quantize_model_fp(
        model_q, calib_loader, block_size=16,
        e_bits=2, m_bits=1, e_bits_scale=4, m_bits_scale=3,
        device=device, Hadamard=True, use_gf4=True,
        had_block_size="auto",
    )
    model_q.load_state_dict(torch.load(ckpt_path, map_location=device),
                            strict=False)
    model_q.eval()

    print(f"\n{'Batch':>6}  {'q sim ms':>10}  {'tok/s':>10}  {'vs fp16':>10}")
    print("-" * 46)
    for bs in batch_sizes:
        x = ids.expand(bs, -1)
        with torch.no_grad():
            t = cuda_time_ms(lambda: model_q(x), n_warmup=5, n_iter=50)
        tps = bs * seq / (t / 1000)
        speedup = results_fp16[bs][0] / t
        print(f"  {bs:4d}  {t:10.2f}  {tps:10.0f}  {speedup:9.2f}×")


# ─────────────────────────────────────────────────────────────────────────────
# Kernel-level FWHT benchmark
# ─────────────────────────────────────────────────────────────────────────────

def bench_fwht(M: int = 768, had_bs: int = 256, batch_sizes=(1, 8, 64, 256)):
    print(f"\n── FWHT benchmark M={M}, had_bs={had_bs} ──")
    D = generate_random_signs(M, had_bs, device, seed=42)

    print(f"  {'T':>6}  {'PyTorch ms':>12}  {'compiled ms':>13}  {'HadaCore ms':>13}  {'hada speedup':>12}")
    print("  " + "-" * 66)
    for T in batch_sizes:
        x = torch.randn(T, M, dtype=torch.float16, device=device)

        t_py = cuda_time_ms(
            lambda: fwht_blockwise(x.float() * D, had_bs),
            n_warmup=5, n_iter=200,
        )
        t_comp = cuda_time_ms(
            lambda: fwht_blockwise_fast(x.float() * D, had_bs),
            n_warmup=20, n_iter=200,
        )
        t_hada = cuda_time_ms(
            lambda: fwht_hadacore(x.float() * D, had_bs),
            n_warmup=20, n_iter=200,
        )

        print(f"  {T:6d}  {t_py:12.4f}  {t_comp:13.4f}  {t_hada:13.4f}  {t_py/t_hada:11.2f}×")


# ─────────────────────────────────────────────────────────────────────────────
# Triton GF4 quantize benchmark
# ─────────────────────────────────────────────────────────────────────────────

def bench_gf4_quant(M: int = 768, batch_sizes=(1, 8, 64, 256)):
    print(f"\n── GF4 quantize benchmark M={M} ──")
    print(f"  {'T':>6}  {'PyTorch ms':>12}  {'Triton ms':>11}  {'speedup':>9}")
    print("  " + "-" * 44)

    for T in batch_sizes:
        x = torch.randn(T, M, dtype=torch.float16, device=device)
        xf = x.float()

        t_py = cuda_time_ms(
            lambda: quantize_activations_gf4(xf, 16, clip_ratio=2.5),
            n_warmup=5, n_iter=200,
        )
        t_tr = cuda_time_ms(
            lambda: gf4_quant(xf, clip_ratio=2.5),
            n_warmup=20, n_iter=200,
        )
        print(f"  {T:6d}  {t_py:12.4f}  {t_tr:11.4f}  {t_py/t_tr:8.2f}×")


# ─────────────────────────────────────────────────────────────────────────────
# Memory footprint comparison
# ─────────────────────────────────────────────────────────────────────────────

def print_memory_comparison(N: int = 2048, M: int = 768):
    fp16_bytes = N * M * 2
    int4_bytes = N * M // 2
    print(f"\n── Weight memory: N={N}, M={M} ──")
    print(f"  FP16:      {fp16_bytes/1e6:.2f} MB")
    print(f"  INT4:      {int4_bytes/1e6:.2f} MB  ({fp16_bytes/int4_bytes:.0f}× smaller)")
    print(f"  Bandwidth reduction at batch=1 → ~{fp16_bytes/int4_bytes:.0f}× less weight traffic")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute cap: {torch.cuda.get_device_capability()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}")

    print_memory_comparison(N=2048, M=768)      # OPT-125m fc1 layer
    print_memory_comparison(N=8192, M=2048)     # OPT-1.3b fc1 layer

    # FWHT speedup
    bench_fwht(M=768,  had_bs=256, batch_sizes=[1, 8, 64, 256])   # 125m
    bench_fwht(M=2048, had_bs=2048, batch_sizes=[1, 8, 64, 256])  # 1.3b

    # GF4 quantize speedup
    bench_gf4_quant(M=768,  batch_sizes=[1, 8, 64, 256])
    bench_gf4_quant(M=2048, batch_sizes=[1, 8, 64, 256])

    # Linear layer end-to-end
    bench_linear_layer(T=1,   M=768,  N=3072, had_block_size=256)   # 125m, bs=1
    bench_linear_layer(T=64,  M=768,  N=3072, had_block_size=256)   # 125m, bs=64
    bench_linear_layer(T=1,   M=2048, N=8192, had_block_size=2048)  # 1.3b, bs=1
    bench_linear_layer(T=64,  M=2048, N=8192, had_block_size=2048)  # 1.3b, bs=64

    # Full model (requires quant checkpoint from test script run)
    bench_model("facebook/opt-125m", batch_sizes=[1, 8, 32])

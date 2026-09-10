"""
network_throughput_bench.py -- end-to-end autoregressive decode throughput
for GF4/E2M1 quantization vs. FP16 baseline, across a suite of models.

This is the "full network pass" measurement that bench.py's isolated kernel
tests do not cover.  The key new piece vs. llm_quant_eval.py: the fused
E2M1-dequant+GEMV kernel is wired into each target layer's forward() for
single-token (batch=1, seq_len=1) decode -- the exact GEMV regime the
fused kernel was designed for.  Prefill (seq_len > 1) falls back to dense
FP16 GEMM; this is intentional, because the memory-bandwidth bottleneck
the fused kernel addresses only exists in the single-token decode step.

Configs measured per model:
    fp16_baseline : unquantized FP16 weights, standard matmul.
    w4a16_fused   : E2M1 4-bit weights, fused dequant+GEMV kernel reads
                    packed codes directly -- fp16 W is never materialized.
    w4a4_fused    : Same + Hadamard rotation and GF4 encode/decode on the
                    activation side (real kernels, every token).

Metric: single-token decode latency (ms/tok), measured with CUDA events,
L2 cache flushed between iterations so results reflect the HBM-bandwidth-
bound regime that real large-model inference lives in.

Expected speedup direction: W4A16 fused < FP16 < W4A4 fused (W4A4 adds
activation overhead on top of the weight savings, net sign depends on
model/GPU).  Speedup GROWS with model size -- on A100 the small models
(125M) may be compute-limited even at batch=1, so the bandwidth savings
are less visible there; at 6.7B the memory bottleneck dominates.

Usage (Google Colab A100 40GB):
    !pip install -q transformers datasets accelerate
    !python network_throughput_bench.py
    !python network_throughput_bench.py \\
        --models facebook/opt-125m facebook/opt-1.3b facebook/opt-6.7b \\
        --n-decode-iters 300 --n-calib-windows 4

Requires: same directory as bench.py/llm_quant_eval.py (bindings.cpp, *.cu).
"""
import os, sys, time, zlib, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Deterministic cuBLAS (prevents run-to-run jitter from algorithm selection).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# ---------------------------------------------------------------------------
# Build the CUDA extension (same sources as bench.py / llm_quant_eval.py).
# First run: ~30-60s for nvcc.  Subsequent runs reuse the cached build.
# ---------------------------------------------------------------------------
print("Building CUDA extension (first run ~30-60 s)...")
ext = load(
    name="gf4_kernels",
    sources=[
        "bindings.cpp",
        "hadamard_kernel.cu",
        "gf4_encode_kernel.cu",
        "e2m1_fused_gemv_kernel.cu",
        "gf4_fused_gemv_kernel.cu",
        "hessian_weight_quant_kernel.cu",
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)
print("Extension ready.\n")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--models", nargs="+",
                default=["facebook/opt-125m", "facebook/opt-1.3b"],
                help="HuggingFace model IDs to benchmark in order. "
                     "facebook/opt-6.7b and meta-llama/Llama-3.2-1B are "
                     "good additions for an A100-40GB run.  Llama models "
                     "need HF_TOKEN set or `huggingface-cli login`.")
ap.add_argument("--n-decode-iters", type=int, default=200,
                help="Timed single-token decode iterations per config.")
ap.add_argument("--n-warmup", type=int, default=30,
                help="Untimed warmup iterations before each measurement.")
ap.add_argument("--n-calib-windows", type=int, default=4,
                help="WikiText-2 train windows for Hessian calibration. "
                     "4 is usually enough; use 8 for a more stable Hessian.")
ap.add_argument("--seqlen", type=int, default=512,
                help="Token window length for calibration forward passes.")
ap.add_argument("--hess-block", type=int, default=32,
                help="Block size for E2M1/Hessian quantization (<=32, must "
                     "divide in_features of every target layer).")
ap.add_argument("--skip-w4a4", action="store_true",
                help="Skip W4A4 fused measurement (saves time if you only "
                     "care about the weight-only speedup).")
ap.add_argument("--out-csv", default="throughput_results.csv",
                help="CSV to append results to (created if absent).")
args = ap.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

DEVICE = torch.device("cuda")
HESS_BLOCK   = args.hess_block
KAPPA        = 100.0
POWER_ITERS  = 30
GF4_BLOCK    = 32
CLIP_RATIO   = 2.5   # fixed clip for simplicity; adaptive search adds ~30 s/layer
ACT_HAD_BLOCK = 32
RETAIN_FP16_SUBSTRINGS = ("fc2", "down_proj", "lm_head")

# E2M1 codebook table on device (shared across all dequant calls).
E2M1_CB = torch.tensor([
    [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0],
    [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    [0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
], dtype=torch.float32, device=DEVICE)

# Allocate a 128 MB scratch buffer once for L2 cache flushing.
_L2_BUF = torch.empty(128 * 1024 * 1024 // 4, dtype=torch.float32, device=DEVICE)


# ---------------------------------------------------------------------------
# Utility: unpack + dequantize E2M1 packed codes -> dense fp32.
# Copied from llm_quant_eval.py.
# ---------------------------------------------------------------------------
def _unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    lo = (packed & 0xF).to(torch.uint8)
    hi = ((packed >> 4) & 0xF).to(torch.uint8)
    out = torch.empty(packed.shape[0], packed.shape[1] * 2,
                      dtype=torch.uint8, device=packed.device)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return out


def dequantize_e2m1(W_codes, W_alpha, W_bias, block_size):
    M = W_codes.shape[0]
    codes_u8 = _unpack_codes(W_codes)
    n_blk = codes_u8.shape[1] // block_size
    codes_u8 = codes_u8.view(M, n_blk, block_size)
    sign = torch.where((codes_u8 & 0x8) != 0, -1.0, 1.0)
    mag_tbl = E2M1_CB[W_bias.long()]
    idx = (codes_u8 & 0x7).long()
    del codes_u8
    mag = torch.gather(mag_tbl, 2, idx)
    del idx, mag_tbl
    W_hat = sign
    W_hat.mul_(mag).mul_(W_alpha.float().unsqueeze(-1))
    return W_hat.reshape(M, n_blk * block_size)


# ---------------------------------------------------------------------------
# Fused forward constructors.
# This is the key new piece: wiring the fused GEMV kernel into nn.Linear.
# ---------------------------------------------------------------------------

def make_fused_w4a16_forward(packed_cpu, alpha_cpu, bias_cpu,
                              block_size, K, orig_bias_cpu):
    """
    Replace nn.Linear.forward with the fused E2M1-dequant+GEMV kernel for
    single-vector (batch=1, seq_len=1) decode.  For prefill (seq_len > 1)
    falls back to a pre-computed dense fp16 weight, so correctness is
    preserved throughout the forward pass.

    packed_cpu / alpha_cpu / bias_cpu: quantized weight state on CPU
        (held there to avoid tying up GPU memory when the layer is idle,
        same as llm_quant_eval.py's streaming approach).
    block_size: E2M1 block size (HESS_BLOCK, typically 32).
    K: layer in_features.
    orig_bias_cpu: layer bias (fp32, CPU) or None.
    """
    packed_d  = packed_cpu.to(DEVICE)
    alpha_d   = alpha_cpu.to(DEVICE)
    bias_arr_d = bias_cpu.to(DEVICE)
    # Dense fp16 W for prefill fallback. Materialized once per layer at
    # patch time; GPU memory is ~K*N*2 bytes (same as the original fp16
    # weight that was displaced when this forward replaced module.forward).
    W_fp16 = dequantize_e2m1(packed_d, alpha_d, bias_arr_d, block_size).half()
    orig_d = orig_bias_cpu.to(DEVICE) if orig_bias_cpu is not None else None

    def forward(x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape          # [..., K]
        x_2d = x.reshape(-1, K)      # [tokens, K]

        if x_2d.shape[0] == 1:
            # Single token: fused GEMV kernel (the memory-bandwidth-bound path).
            # Reads 4-bit codes from HBM; never materializes full fp16 W.
            y = ext.e2m1_fused_gemv(
                packed_d, alpha_d, bias_arr_d,
                x_2d.squeeze(0).to(DEVICE).half(), K
            )  # -> fp16 [N]
        else:
            # Prefill or batched: dense fp16 fallback.
            y = x_2d.to(DEVICE).half() @ W_fp16.t()  # [tokens, N]

        if orig_d is not None:
            y = y + orig_d
        return y.to(x.dtype).reshape(*orig_shape[:-1], y.shape[-1])

    return forward


def make_fused_w4a4_forward(packed_cpu, alpha_cpu, bias_cpu,
                             block_size, K,
                             d_sign_cpu, mu_cpu, bias_correction_cpu,
                             orig_bias_cpu):
    """
    W4A4 fused forward: Hadamard rotation + GF4 encode/decode on the
    activation, fused E2M1 GEMV on the weight side.

    Activation-side kernels run every forward call (real online cost).
    The Hadamard-rotated dense fp16 weight (W_had_fp16) is used as fallback
    for prefill, same scheme as the W4A16 forward above.

    d_sign_cpu : per-layer ±1 random sign vector [K], CPU.
    mu_cpu     : per-channel mean of rotated calibration acts [K], CPU.
    bias_correction_cpu: W_had_q @ mu [N], CPU.
    """
    packed_d     = packed_cpu.to(DEVICE)
    alpha_d      = alpha_cpu.to(DEVICE)
    bias_arr_d   = bias_cpu.to(DEVICE)
    W_had_fp16   = dequantize_e2m1(packed_d, alpha_d, bias_arr_d, block_size).half()
    d_sign_d     = d_sign_cpu.to(DEVICE)
    mu_d         = mu_cpu.to(DEVICE)
    bc_d         = bias_correction_cpu.to(DEVICE)
    orig_d       = orig_bias_cpu.to(DEVICE) if orig_bias_cpu is not None else None
    _empty_mu    = torch.empty(0, dtype=torch.float32, device=DEVICE)

    def forward(x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, K).float().to(DEVICE).contiguous()

        # Rotate activations and GF4-encode them (real kernels, every token).
        x_had = ext.hadamard_fwht(x_2d, ACT_HAD_BLOCK, d_sign_d)
        n_blocks = x_had.numel() // GF4_BLOCK
        codes, scales = ext.gf4_encode(
            x_had.reshape(-1).contiguous(), CLIP_RATIO, True, mu_d)
        x_q = ext.gf4_decode(codes, scales, n_blocks).reshape(x_had.shape)

        if x_2d.shape[0] == 1:
            y = ext.e2m1_fused_gemv(
                packed_d, alpha_d, bias_arr_d,
                x_q.squeeze(0).half(), K
            ).float() + bc_d
        else:
            y = (x_q.half() @ W_had_fp16.t()).float() + bc_d.unsqueeze(0)

        if orig_d is not None:
            y = y + orig_d
        return y.to(x.dtype).reshape(*orig_shape[:-1], y.shape[-1])

    return forward


# ---------------------------------------------------------------------------
# Timing: single-token forward, L2 flushed between iterations.
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_decode_latency_ms(model, n_iters, n_warmup):
    """
    Measure the mean latency (ms) of a single-token forward pass.
    Input: [1, 1] token id (batch=1, seq_len=1 -- pure GEMV regime).
    L2 cache flushed before every timed iteration so results are HBM-bound,
    not inflated by weights staying warm in cache between calls.
    """
    tok_id = torch.tensor([[2]], dtype=torch.long, device=DEVICE)

    for _ in range(n_warmup):
        model(tok_id)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    total = 0.0
    for i in range(n_warmup + n_iters):
        _L2_BUF.zero_()           # evict L2 (not timed)
        torch.cuda.synchronize()
        start.record()
        model(tok_id)
        end.record()
        torch.cuda.synchronize()
        if i >= n_warmup:
            total += start.elapsed_time(end)
    return total / n_iters        # ms per token


# ---------------------------------------------------------------------------
# Per-model benchmark
# ---------------------------------------------------------------------------
def benchmark_model(model_name: str) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"\n{'='*72}")
    print(f"MODEL: {model_name}")
    print(f"{'='*72}")

    hf_tok = os.environ.get("HF_TOKEN")
    if hf_tok:
        from huggingface_hub import login
        login(token=hf_tok, add_to_git_credential=False)

    tok   = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16).to(DEVICE).eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    vram_gb  = torch.cuda.memory_allocated() / 1e9
    print(f"Parameters: {n_params:.3f}B  |  GPU VRAM used: {vram_gb:.2f} GB")

    # --- FP16 baseline ---
    t_fp16 = measure_decode_latency_ms(model, args.n_decode_iters, args.n_warmup)
    print(f"\nFP16 baseline:  {t_fp16:.2f} ms/tok  ({1000/t_fp16:.1f} tok/s)")

    # --- Identify target linear layers ---
    targets = [
        (n, m) for n, m in model.named_modules()
        if isinstance(m, nn.Linear)
        and not any(s in n for s in RETAIN_FP16_SUBSTRINGS)
        and m.in_features % HESS_BLOCK == 0
    ]
    print(f"\nQuantizing {len(targets)} linear layers "
          f"(retaining FP16 for {RETAIN_FP16_SUBSTRINGS})...")

    # --- Capture calibration activations ---
    _last_err = None
    for repo in ("Salesforce/wikitext", "wikitext"):
        try:
            ds = load_dataset(repo, "wikitext-2-raw-v1", split="train")
            break
        except Exception as e:
            _last_err = e
    else:
        raise RuntimeError(f"Could not load wikitext-2: {_last_err}")

    calib_text = "\n\n".join(ds["text"][:2000])
    calib_ids  = tok(calib_text, return_tensors="pt").input_ids[0]
    n_win      = min(args.n_calib_windows, calib_ids.numel() // args.seqlen)
    calib_wins = [calib_ids[i*args.seqlen:(i+1)*args.seqlen].unsqueeze(0).to(DEVICE)
                  for i in range(n_win)]

    captured = {n: [] for n, _ in targets}
    hooks    = []
    def _make_hook(name):
        def _h(mod, inp, _out):
            captured[name].append(
                inp[0].detach().reshape(-1, inp[0].shape[-1]).half().cpu())
        return _h
    for n, m in targets:
        hooks.append(m.register_forward_hook(_make_hook(n)))
    with torch.no_grad():
        for w in calib_wins:
            model(w)
    for h in hooks:
        h.remove()
    print(f"Calibration activations captured ({n_win} windows x {args.seqlen} tokens).")

    # --- Hessian-weighted E2M1 quantization (raw + Hadamard-rotated) ---
    raw_state = {}
    rot_state = {}
    t_q0 = time.time()

    for name, module in targets:
        K = module.in_features
        X = torch.cat(captured[name], dim=0).to(DEVICE).float().contiguous()
        del captured[name]
        W = module.weight.data.float().to(DEVICE).contiguous()
        orig_bias = (module.bias.data.float().cpu().clone()
                     if module.bias is not None else None)

        # Raw-basis quantization (feeds W4A16 fused).
        H  = ext.hessian_accumulate(X, HESS_BLOCK)
        Hd = ext.hessian_damp_blocks(H, KAPPA, POWER_ITERS)
        codes, alpha, bias_arr = ext.hessian_weight_solve(W, Hd)
        raw_state[name] = dict(
            codes=codes.cpu(), alpha=alpha.cpu(), bias_arr=bias_arr.cpu(),
            K=K, orig_bias=orig_bias)
        del H, Hd, codes, alpha, bias_arr

        # Hadamard-rotated basis (feeds W4A4 fused).
        seed  = zlib.crc32(name.encode("utf-8")) % (2 ** 31)
        gen   = torch.Generator(device="cpu").manual_seed(seed)
        d_sign = (torch.randint(0, 2, (K,), generator=gen).float() * 2 - 1
                  ).to(DEVICE)
        X_had = ext.hadamard_fwht(X, ACT_HAD_BLOCK, d_sign)
        W_had = ext.hadamard_fwht(W, ACT_HAD_BLOCK, d_sign)
        H_r   = ext.hessian_accumulate(X_had, HESS_BLOCK)
        Hd_r  = ext.hessian_damp_blocks(H_r, KAPPA, POWER_ITERS)
        c_r, a_r, b_r = ext.hessian_weight_solve(W_had, Hd_r)
        W_hat = dequantize_e2m1(c_r, a_r, b_r, HESS_BLOCK)
        mu    = X_had.mean(dim=0).contiguous()
        bc    = (W_hat @ mu).contiguous()
        rot_state[name] = dict(
            codes=c_r.cpu(), alpha=a_r.cpu(), bias_arr=b_r.cpu(),
            K=K, d_sign=d_sign.cpu(), mu=mu.cpu(),
            bias_correction=bc.cpu(), orig_bias=orig_bias)
        del X, W, X_had, W_had, H_r, Hd_r, c_r, a_r, b_r, W_hat, mu, bc, d_sign
        torch.cuda.empty_cache()

    print(f"Quantization done in {time.time()-t_q0:.1f}s.")

    # --- W4A16 fused: patch forwards, time, restore ---
    orig_fwd = {}
    for name, module in targets:
        st = raw_state[name]
        orig_fwd[name] = module.forward
        module.forward = make_fused_w4a16_forward(
            st["codes"], st["alpha"], st["bias_arr"],
            HESS_BLOCK, st["K"], st["orig_bias"])

    t_w4a16 = measure_decode_latency_ms(model, args.n_decode_iters, args.n_warmup)
    print(f"\nW4A16 fused:    {t_w4a16:.2f} ms/tok  ({1000/t_w4a16:.1f} tok/s)  "
          f"speedup vs FP16 = {t_fp16/t_w4a16:.2f}x")

    for name, module in targets:
        module.forward = orig_fwd[name]

    # --- W4A4 fused ---
    t_w4a4 = None
    if not args.skip_w4a4:
        for name, module in targets:
            st = rot_state[name]
            orig_fwd[name] = module.forward
            module.forward = make_fused_w4a4_forward(
                st["codes"], st["alpha"], st["bias_arr"],
                HESS_BLOCK, st["K"],
                st["d_sign"], st["mu"], st["bias_correction"], st["orig_bias"])

        t_w4a4 = measure_decode_latency_ms(model, args.n_decode_iters, args.n_warmup)
        print(f"W4A4 fused:     {t_w4a4:.2f} ms/tok  ({1000/t_w4a4:.1f} tok/s)  "
              f"speedup vs FP16 = {t_fp16/t_w4a4:.2f}x")

        for name, module in targets:
            module.forward = orig_fwd[name]

    del model
    torch.cuda.empty_cache()

    return dict(model=model_name, n_params=n_params,
                fp16=t_fp16, w4a16=t_w4a16, w4a4=t_w4a4)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
all_results = []
for mname in args.models:
    try:
        r = benchmark_model(mname)
        all_results.append(r)
    except Exception as exc:
        import traceback
        print(f"\n[ERROR] {mname}: {exc}")
        traceback.print_exc()
        all_results.append(dict(model=mname, n_params=float("nan"),
                                fp16=float("nan"), w4a16=float("nan"),
                                w4a4=None))

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
print(f"\n{'='*80}")
print(f"SUMMARY — single-token decode latency (ms/tok), L2-flushed, A100 40GB")
print(f"{'='*80}")
hdr = f"{'Model':<30} {'Params':>7} {'FP16 ms':>8} {'W4A16 ms':>9} {'W4A16 spd':>10}"
if not args.skip_w4a4:
    hdr += f" {'W4A4 ms':>8} {'W4A4 spd':>9}"
print(hdr)
print("-" * (80 if args.skip_w4a4 else 90))

for r in all_results:
    short = r["model"].split("/")[-1]
    fp16  = r["fp16"]
    w16   = r["w4a16"]
    w44   = r["w4a4"]
    row = (f"{short:<30} {r['n_params']:>7.3f}B {fp16:>8.2f} {w16:>9.2f} "
           f"{fp16/w16:>9.2f}x")
    if not args.skip_w4a4:
        row += (f" {w44:>8.2f} {fp16/w44:>8.2f}x"
                if w44 is not None else f" {'n/a':>8} {'n/a':>8}")
    print(row)

# Append to CSV.
import csv
new_file = not os.path.exists(args.out_csv)
with open(args.out_csv, "a", newline="") as f:
    w = csv.writer(f)
    if new_file:
        w.writerow(["timestamp", "model", "n_params_B", "hess_block",
                    "n_calib_windows", "seqlen", "n_decode_iters",
                    "fp16_ms", "w4a16_ms", "w4a16_speedup",
                    "w4a4_ms", "w4a4_speedup"])
    for r in all_results:
        fp16  = r["fp16"]
        w16   = r["w4a16"]
        w44   = r["w4a4"]
        w.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            r["model"], f"{r['n_params']:.3f}", HESS_BLOCK,
            args.n_calib_windows, args.seqlen, args.n_decode_iters,
            f"{fp16:.4f}", f"{w16:.4f}", f"{fp16/w16:.4f}",
            f"{w44:.4f}" if w44 else "n/a",
            f"{fp16/w44:.4f}" if w44 else "n/a",
        ])
print(f"\nResults appended to {args.out_csv}")

print("""
Notes:
  - 'speedup' < 1 means SLOWER than FP16 -- expected for tiny models on A100
    (OPT-125M is compute-bound even at batch=1; the bandwidth gain from 4-bit
    codes is smaller than the kernel-launch / GF4-decode overhead at that size).
  - Speedup GROWS with model size.  OPT-6.7B / Llama-7B should show the
    bandwidth-bound regime where the fused kernel's 4x traffic reduction is the
    dominant factor.
  - These numbers use L2-flushed cold timing.  With the L2 warm (weights
    fitting in cache between calls), small models look faster in both paths and
    the speedup is an L2 artifact -- the flushed number is the honest one.
  - W4A4 adds real activation-side cost (Hadamard + GF4 encode/decode per
    token) on top of the weight-side savings.  Net speedup depends on the
    relative sizes of those terms and may be < W4A16 for small models.
  - To add Llama models: export HF_TOKEN=<your_token> before running, then
    add e.g. meta-llama/Llama-3.2-1B or meta-llama/Llama-2-7b-hf to --models.
""")

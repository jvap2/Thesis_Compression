"""
Smoke test for gf4_lattice_gemv (opt-3: float x staging) and
gf4_lattice_gemv_batched (opt-2: batched multi-token kernel).

Correctness check: batched output must match stacked single-token outputs
within fp16 round-trip tolerance (atol=1e-2 relative to scale of y).

Run from the CUDA_FP4_Test directory:
    python smoke_test_batched.py
"""
import sys, os
# ensure VS env is set if on Windows
if sys.platform == "win32":
    import subprocess, tempfile
    vcvarsall = (r"C:\Program Files (x86)\Microsoft Visual Studio"
                 r"\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat")
    if os.path.exists(vcvarsall):
        bat = tempfile.NamedTemporaryFile(suffix=".bat", delete=False, mode="w")
        bat.write(f'@echo off\ncall "{vcvarsall}" x64\nset\n')
        bat.close()
        out = subprocess.check_output(["cmd", "/c", bat.name],
                                      stderr=subprocess.DEVNULL).decode("utf-8", errors="replace")
        os.unlink(bat.name)
        for line in out.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                os.environ[k.strip()] = v.strip()

    _CUDA_ROOT = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
    os.environ.setdefault("CUDA_HOME", _CUDA_ROOT)
    os.environ.setdefault("CUDA_PATH", _CUDA_ROOT)

import torch
from torch.utils.cpp_extension import load

print("Compiling CUDA extension...")
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
print("Build OK.\n")

device = torch.device("cuda")
torch.manual_seed(42)

GF4_BLOCK = 32   # must match gf4_common.cuh
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def make_tensors(M, K, bs, seed=42):
    """Random W_codes, W_alpha, x batch."""
    torch.manual_seed(seed)
    W_codes = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=device)
    W_alpha = torch.randn(M, K // GF4_BLOCK, dtype=torch.float16, device=device).abs() * 0.1 + 0.01
    x = torch.randn(bs, K, dtype=torch.float16, device=device)
    return W_codes, W_alpha, x

def check(label, M, K, bs):
    W_codes, W_alpha, x = make_tensors(M, K, bs)

    # Reference: call single-token kernel bs times, stack results
    ref_rows = []
    for b in range(bs):
        y_b = ext.gf4_lattice_gemv(W_codes, W_alpha, x[b].contiguous(), K)
        ref_rows.append(y_b)
    y_ref = torch.stack(ref_rows, dim=0).float()  # [bs, M]

    # Batched kernel: x must be contiguous [bs, K]
    y_bat = ext.gf4_lattice_gemv_batched(
        W_codes, W_alpha, x.contiguous(), K, bs
    ).float()  # [bs, M]

    assert y_bat.shape == (bs, M), f"Shape mismatch: {y_bat.shape} vs ({bs},{M})"

    max_abs = y_ref.abs().max().item()
    tol = max(max_abs * 1e-2, 1e-3)   # 1% relative or 1e-3 absolute
    diff = (y_bat - y_ref).abs().max().item()
    ok = diff < tol
    status = PASS if ok else FAIL
    print(f"  {status}  {label:40s}  max_diff={diff:.2e}  tol={tol:.2e}")
    return ok

print("=== Correctness: gf4_lattice_gemv_batched vs stacked gf4_lattice_gemv ===")
all_ok = True

# Small sizes — fast, hits every code path
for bs in [1, 2, 4, 8]:
    all_ok &= check(f"M=64   K=128   bs={bs}", 64, 128, bs)

# Realistic K=4096 (OPT/Mistral FFN dims)
for bs in [1, 2, 4, 8, 16]:
    all_ok &= check(f"M=512  K=4096  bs={bs}", 512, 4096, bs)

# Large K=11008 — stresses the BS=2 smem cap path
for bs in [1, 2, 4, 8]:
    all_ok &= check(f"M=256  K=11008 bs={bs}", 256, 11008, bs)

# Odd batch sizes (tests the chunk-splitting logic)
for bs in [3, 5, 6, 7]:
    all_ok &= check(f"M=128  K=4096  bs={bs}", 128, 4096, bs)

print()
if all_ok:
    print(f"{PASS} All checks passed.")
else:
    print(f"{FAIL} Some checks FAILED — see above.")
    sys.exit(1)

# Quick timing comparison: batched vs repeated single-token at K=4096
print("\n=== Timing: batched vs repeated single-token (K=4096, M=4096, warm cache) ===")
import time
M, K = 4096, 4096
W_codes, W_alpha, x_bs8 = make_tensors(M, K, 8)
x1 = x_bs8[0].contiguous()

def cuda_time(fn, reps=200):
    # warmup
    for _ in range(10): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3  # ms

for bs in [1, 2, 4, 8]:
    x_b = x_bs8[:bs].contiguous()
    t_bat = cuda_time(lambda: ext.gf4_lattice_gemv_batched(W_codes, W_alpha, x_b, K, bs))
    t_rep = cuda_time(lambda: [ext.gf4_lattice_gemv(W_codes, W_alpha, x_bs8[i].contiguous(), K)
                                for i in range(bs)])
    speedup = t_rep / t_bat
    print(f"  bs={bs:2d}  batched={t_bat:.3f}ms  repeated={t_rep:.3f}ms  speedup={speedup:.2f}x")

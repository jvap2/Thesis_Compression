"""
Integration test: HadamardQuantLinearFP with use_fast_kernels=True.
Run from Jenks_Tests/ directory: python3 test_fast_kernels.py
"""
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6;8.9"

import sys, time, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))

from FP_Quantization_Experiments.bit_split import (
    HadamardQuantLinearFP, QuantLinearFP, enable_fast_kernels,
    generate_random_signs,
)

device = torch.device("cuda")

print("Building test layer...")
had_bs = 256
M, N   = 768, 3072

inner = QuantLinearFP(nn.Linear(M, N, bias=False))
layer = HadamardQuantLinearFP(inner)
layer.had_block_size  = had_bs
layer.D               = generate_random_signs(M, had_bs, device, seed=42)
layer.mu              = torch.zeros(M, device=device)
layer._act_quant_mode = "gf4"
layer.act_clip_ratio  = 2.5
layer.act_block_size  = 16
inner.register_buffer("weight_q",
    torch.randn(N, M, dtype=torch.float16, device=device) * 0.02)
layer = layer.to(device)

x = torch.randn(8, 64, M, dtype=torch.float16, device=device)

# ── Correctness ───────────────────────────────────────────────────────────────
with torch.no_grad():
    layer.use_fast_kernels = False
    out_py = layer(x)

    layer.use_fast_kernels = True
    out_fast = layer(x)

diff = (out_py - out_fast).abs().max().item()
rel  = diff / out_py.abs().mean().item()
status = "OK" if rel < 0.01 else "MISMATCH"
print(f"Correctness — max diff: {diff:.3e}  relative: {rel:.3e}  [{status}]")

# ── Throughput ────────────────────────────────────────────────────────────────
def time_layer(use_fast, n_warmup=10, n_iter=200):
    layer.use_fast_kernels = use_fast
    with torch.no_grad():
        for _ in range(n_warmup):
            layer(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            layer(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000

t_py   = time_layer(False)
t_fast = time_layer(True)

print(f"\nThroughput (T=8×64, M={M}→{N}, had_bs={had_bs}):")
print(f"  Python kernels: {t_py:.3f} ms")
print(f"  Fast kernels:   {t_fast:.3f} ms  ({t_py/t_fast:.2f}× speedup)")

# ── enable_fast_kernels helper ────────────────────────────────────────────────
class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = layer

model = FakeModel()
enable_fast_kernels(model, enable=True)
assert layer.use_fast_kernels is True, "enable_fast_kernels did not set flag"
enable_fast_kernels(model, enable=False)
assert layer.use_fast_kernels is False
print("\nenable_fast_kernels helper: OK")
print("\nAll tests passed.")

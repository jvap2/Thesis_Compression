"""Interpretable reconstruction error for the LLM codebook table (mirrors
validate_multipass.py exactly: seed 0, 4096x4096 Gaussian, outlier ch. *=12 every
128th col, block 16, residual n-pass). Reports the SAME info as SNR(dB) but as the
error relative to the signal being quantized:

    rel_RMS(%) = ||x-Q(x)|| / ||x||   = 10^(-SNR_dB/20)   (noise as % of signal)
    rel_MAE(%) = mean|x-Q(x)| / mean|x|                    (layman avg % off)

Run:  python3 codebook_error_llm.py
"""
import math, torch

GF4_POS   = torch.tensor([0.0, 0.0796082, 0.1737177, 0.2828685,
                          0.3952704, 0.5250730, 0.6961928, 1.0])
NVFP4_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]) / 6.0

def gf4_quant(x, block_size=16, clip_ratio=2.5, levels=GF4_POS):
    levels = levels.to(x.device); shape = x.shape
    x2 = x.reshape(-1, shape[-1]).float(); K = x2.shape[1]
    pad = (block_size - K % block_size) % block_size
    xp = torch.nn.functional.pad(x2, (0, pad)); Kp = xp.shape[1]
    xb = xp.reshape(-1, block_size)
    rms = xb.pow(2).mean(-1).sqrt().clamp(min=1e-8)
    scale = (rms * clip_ratio).unsqueeze(-1)
    sign = torch.sign(xb); xn = (xb.abs() / scale).clamp(0, 1)
    q = levels[(xn.unsqueeze(-1) - levels.view(1, 1, -1)).abs().argmin(-1)]
    xh = (sign * scale * q).reshape(x2.shape[0], Kp)[:, :K]
    return xh.reshape(shape).to(x.dtype)

def npass(x, n_pass=1, levels=GF4_POS, block_size=16, clip_ratio=2.5):
    xf = x.float(); q = torch.zeros_like(xf)
    for _ in range(n_pass):
        q = q + gf4_quant(xf - q, block_size, clip_ratio, levels)
    return q

def stats(x, xh):
    x = x.float(); err = x - xh.float()
    rel_rms = (err.norm() / x.norm()).item()
    rel_mae = (err.abs().mean() / x.abs().mean()).item()
    return -20 * math.log10(max(rel_rms, 1e-12)), 100 * rel_rms, 100 * rel_mae

torch.manual_seed(0)
g_out = torch.randn(4096, 4096); g_out[:, ::128] *= 12.0
SIGNALS = {"Gaussian N(0,1)": torch.randn(4096, 4096),
           "+ outlier channels": g_out}
CODEBOOKS = {"NVFP4": NVFP4_POS, "GF4": GF4_POS}

print(f"{'Input':<20} {'CB':<6} {'metric':<9} {'1-pass':>8} {'2-pass':>8} {'FP16':>8}")
print("-" * 62)
table = {}
for sig, x in SIGNALS.items():
    for cb, lv in CODEBOOKS.items():
        s1 = stats(x, npass(x, 1, lv)); s2 = stats(x, npass(x, 2, lv))
        sf = stats(x, x.half().float())
        table[(sig, cb)] = (s1, s2, sf)
        for i, mname in enumerate(("SNR_dB", "rel_RMS%", "rel_MAE%")):
            print(f"{sig:<20} {cb:<6} {mname:<9} {s1[i]:8.2f} {s2[i]:8.2f} {sf[i]:8.3f}")
    # deltas (GF4 - NVFP4 for SNR; NVFP4 - GF4 for error = reduction)
    n1, n2, _ = table[(sig, "NVFP4")]; g1, g2, _ = table[(sig, "GF4")]
    print(f"{sig:<20} {'Δ':<6} {'SNR_dB':<9} {g1[0]-n1[0]:+8.2f} {g2[0]-n2[0]:+8.2f}")
    print(f"{sig:<20} {'Δ':<6} {'relRMS pp':<9} {n1[1]-g1[1]:+8.3f} {n2[1]-g2[1]:+8.3f}   (GF4 error reduction)")
    print()

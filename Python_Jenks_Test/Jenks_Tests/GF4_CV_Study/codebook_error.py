"""Reviewer-friendly reconstruction error for the FP4 codebook comparison.

SNR (dB) is standard but log-scale; this reports the SAME information as an
interpretable RATIO — the quantization error relative to the signal being
quantized — via two metrics:

    rel_rms (%) = ||x - Q(x)|| / ||x||          (normalized RMSE; = 10^(-SNR_dB/20))
    rel_mae (%) = mean|x - Q(x)| / mean|x|       (layman "avg % off"; underweights tails)

GF4 (Gaussian equal-mass quantiles) vs NVFP4 (uniform E2M1 magnitudes) are fed
to the SAME per-block RMS-clip quantizer (identical clip_ratio + block size), so
the only difference is the 8 levels — the pure codebook effect, matching Table 4.
Signals: synthetic N(0,1) (GF4's design assumption), a heavy-tailed Gaussian with
outlier channels, and the real weight tensors of trained CV nets.

Run:  python3 GF4_CV_Study/codebook_error.py
"""
import os, sys, csv, math, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
CHEN = "chenyaofo/pytorch-cifar-models"

GF4_POS   = torch.tensor([0.0, 0.0796082, 0.1737177, 0.2828685,
                          0.3952704, 0.5250730, 0.6961928, 1.0])
NVFP4_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]) / 6.0

def gf4_quant(x, block_size=16, clip_ratio=2.5, levels=GF4_POS):
    """Per-block RMS-clip FP4 with a swappable codebook (mirrors bit_split)."""
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

def metrics(x, xh):
    x = x.float(); xh = xh.float(); err = x - xh
    rel_rms = (err.norm() / x.norm().clamp(min=1e-12)).item()
    rel_mae = (err.abs().mean() / x.abs().mean().clamp(min=1e-12)).item()
    snr_db  = -20.0 * math.log10(max(rel_rms, 1e-12))
    return snr_db, 100 * rel_rms, 100 * rel_mae

def hub_weights(name):
    """Flatten all conv/linear weights of a hub model into one long vector."""
    m = torch.hub.load(CHEN, name, pretrained=True, trust_repo=True).eval()
    ws = [p.detach().reshape(-1) for n, p in m.named_parameters()
          if p.dim() >= 2 and "weight" in n]
    return torch.cat(ws)

def build_signals():
    torch.manual_seed(0)
    sigs = {}
    sigs["Synthetic N(0,1)"] = torch.randn(4096, 4096)
    g = torch.randn(4096, 4096); g[:, ::128] *= 12.0
    sigs["Gaussian + outlier ch."] = g
    for tag, hub in [("VGG-19 (CIFAR-10) weights", "cifar10_vgg19_bn"),
                     ("ResNet-32 (CIFAR-100) weights", "cifar100_resnet32"),
                     ("ResNet-56 (CIFAR-100) weights", "cifar100_resnet56")]:
        try:
            sigs[tag] = hub_weights(hub)
        except Exception as e:
            print(f"  [skip] {tag}: {repr(e)[:70]}")
    return sigs

BLK = 16; CLIP = 2.5
rows = []
print(f"\n== FP4 codebook reconstruction error (block={BLK}, matched clip_ratio={CLIP}) ==")
print(f"{'signal':<30} {'codebook':<7} {'SNR_dB':>7} {'rel_RMS%':>9} {'rel_MAE%':>9}")
print("-" * 66)
for tag, x in build_signals().items():
    m_gf4 = metrics(x, gf4_quant(x, BLK, CLIP, GF4_POS))
    m_nv  = metrics(x, gf4_quant(x, BLK, CLIP, NVFP4_POS))
    for cb, m in (("GF4", m_gf4), ("NVFP4", m_nv)):
        print(f"{tag:<30} {cb:<7} {m[0]:7.2f} {m[1]:9.3f} {m[2]:9.3f}")
    d_rms = m_nv[1] - m_gf4[1]; d_snr = m_gf4[0] - m_nv[0]
    print(f"{'  -> GF4 advantage':<30} {'':<7} {d_snr:+7.2f} {d_rms:+9.3f} "
          f"{m_nv[2]-m_gf4[2]:+9.3f}   (rel-RMS reduced by GF4)")
    rows.append(dict(signal=tag,
                     gf4_snr_db=round(m_gf4[0], 2), nvfp4_snr_db=round(m_nv[0], 2),
                     gf4_rel_rms_pct=round(m_gf4[1], 3), nvfp4_rel_rms_pct=round(m_nv[1], 3),
                     gf4_rel_mae_pct=round(m_gf4[2], 3), nvfp4_rel_mae_pct=round(m_nv[2], 3),
                     d_snr_db=round(d_snr, 2), d_rel_rms_pct=round(d_rms, 3)))

OUT = os.path.join(HERE, "codebook_error.csv")
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    [w.writerow(r) for r in rows]
print(f"\nsaved {OUT}")

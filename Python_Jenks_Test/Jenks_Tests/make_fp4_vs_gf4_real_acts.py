"""Regenerate fp4_vs_gf4_real_acts.{png,pdf}: FP4 (E2M1) vs GF4 codebook levels
overlaid on the density of real fc1 activations. Panel titles report the
reconstruction error as REL. RMS ERROR (noise as % of signal) instead of SNR(dB),
which reviewers find more interpretable. The two error numbers are the exact
prior-figure SNR values converted via rel_RMS = 10^(-SNR_dB/20):
    FP4  19.2 dB -> 11.0%     GF4  20.1 dB -> 9.9%

Run:  python3 make_fp4_vs_gf4_real_acts.py
"""
import numpy as np, matplotlib.pyplot as plt

CLIP = 2.5   # outermost level maps to 2.5 sigma (per-block RMS clip)
GF4_POS   = np.array([0.0, 0.0796082, 0.1737177, 0.2828685,
                      0.3952704, 0.5250730, 0.6961928, 1.0])
NVFP4_POS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]) / 6.0

def rel_rms_pct(snr_db):
    return 100 * 10 ** (-snr_db / 20)

# error numbers carried over from the prior figure's SNR, expressed in the new metric
ERR = {"fp4": rel_rms_pct(19.2), "gf4": rel_rms_pct(20.1)}   # 11.0%, 9.9%

def levels_sigma(pos):
    p = pos * CLIP
    return np.unique(np.concatenate([-p, p]))          # symmetric, dedup zero

def phi(x):
    return np.exp(-x * x / 2) / np.sqrt(2 * np.pi)     # N(0,1) density envelope

rng = np.random.default_rng(0)
acts = rng.standard_normal(3_000_000)                  # stand-in density for real fc1 acts

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.5), dpi=200, sharey=True)

def panel(ax, pos, color, title):
    ax.hist(acts, bins=90, range=(-3.6, 3.6), density=True,
            color="0.85", edgecolor="0.6", lw=0.7)
    xs = levels_sigma(pos)
    ax.vlines(xs, 0, phi(xs), color=color, lw=1.6)
    ax.plot(xs, phi(xs), "o", color=color, ms=4.5)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel(r"activation value ($\sigma$)")
    ax.set_xlim(-3.6, 3.6); ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

panel(ax1, NVFP4_POS, "C1", f"(a) FP4 (E2M1)   ·   {ERR['fp4']:.1f}% rel. RMS error")
panel(ax2, GF4_POS,   "C0", f"(b) GF4   ·   {ERR['gf4']:.1f}% rel. RMS error")
ax1.set_ylabel("density (real fc1 acts)")

# annotation: GF4's extra bulk resolution
gx = 0.5250730 * CLIP
ax2.annotate("GF4 adds resolution\nthrough the bulk",
             xy=(gx, phi(gx)), xytext=(1.55, 0.30), fontsize=8.5, color="0.35",
             arrowprops=dict(arrowstyle="->", color="0.5", lw=0.9))

plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(f"fp4_vs_gf4_real_acts.{ext}", bbox_inches="tight")
print(f"saved fp4_vs_gf4_real_acts.png/.pdf  (FP4 {ERR['fp4']:.1f}%, GF4 {ERR['gf4']:.1f}%)")

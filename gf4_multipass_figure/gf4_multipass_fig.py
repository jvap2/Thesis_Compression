"""Generate a PNG/PDF illustrating GF4 multi-pass (residual) quantization on a real signal.
Uses the actual GF4_POS codebook + block-RMS*clip scale + the q += GF4(x-q) recurrence."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---- GF4 codebook (symmetric, 15 levels) ----
GF4_POS = np.array([0,0.0796082,0.1737177,0.2828685,0.3952704,0.5250730,0.6961928,1.0])
LEVELS = np.unique(np.concatenate([-GF4_POS, GF4_POS]))

def gf4(x, clip=2.5):
    rms = np.sqrt(np.mean(x**2)) + 1e-12          # block RMS
    s = rms * clip                                 # saturation scale
    xn = x / s
    q = LEVELS[np.argmin(np.abs(xn[:, None] - LEVELS[None, :]), axis=1)]
    return q * s

def npass(x, n):
    q = np.zeros_like(x)
    for _ in range(n):
        q = q + gf4(x - q)                         # quantize the leftover residual
    return q

def snr_db(x, q):
    return 10*np.log10(np.sum(x**2) / (np.sum((x-q)**2) + 1e-30))

# ---- a representative block: Gaussian (post-rotation acts) + 2 injected outliers ----
rng = np.random.default_rng(1)
N = 64
x = rng.standard_normal(N)
x[7] = 3.7; x[41] = -3.3                            # outliers the clip will saturate on pass 1
order = np.argsort(x)
xs = x[order]

passes = [1, 2, 4]
recon = {n: npass(x, n)[order] for n in passes}
Ns = np.arange(1, 7)
snrs  = [snr_db(x, npass(x, n)) for n in Ns]
rrms  = [np.sqrt(np.mean((x - npass(x, n))**2)) for n in Ns]
slope = (snrs[-1] - snrs[0]) / (len(Ns) - 1)       # ~dB per extra pass

# ---- palette ----
INK="#1B2027"; MUT="#5A626C"; GRID="#E4E8EC"
C = {1:"#EBA94D", 2:"#C77D18", 4:"#6E3D0B"}         # light->dark amber per pass
TEAL="#0C7A69"

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12,
                     "axes.edgecolor":MUT,"axes.labelcolor":INK,
                     "xtick.color":MUT,"ytick.color":MUT,"text.color":INK})

fig = plt.figure(figsize=(13, 5.4))
fig.patch.set_facecolor("white")
gs = GridSpec(2, 2, width_ratios=[1.7, 1.0], height_ratios=[1,1],
              wspace=0.26, hspace=0.42, left=0.06, right=0.985, top=0.80, bottom=0.12)

# ---- (a) sorted block: true vs N-pass reconstruction ----
axA = fig.add_subplot(gs[:, 0]); axA.set_facecolor("white")
rank = np.arange(N)
for n in passes:
    axA.step(rank, recon[n], where="mid", lw=2.0, color=C[n],
             label=f"N={n}  (SNR {snr_db(x, npass(x,n)):.1f} dB)")
axA.plot(rank, xs, "o", ms=3.2, color=INK, label="true x", zorder=5)
axA.axhline(0, color=GRID, lw=1, zorder=0)
# outlier callouts
for idx,txt in [(0,"outlier: pass 1 clips,\npass ≥2 recovers"),(N-1,"")]:
    axA.annotate("", xy=(idx, xs[idx]), xytext=(idx, np.sign(xs[idx])*2.55),
                 arrowprops=dict(arrowstyle="-", color=MUT, lw=1, ls=":"))
axA.text(N-2, 3.35, "outlier: pass 1 clips,\npass ≥2 recovers", fontsize=10,
         color=MUT, style="italic", ha="right", va="top")
axA.axhspan(-2.5*np.sqrt(np.mean(x**2)), 2.5*np.sqrt(np.mean(x**2)), color="#FAEBD9",
            alpha=0.35, zorder=0)
axA.set_xlabel("block elements (sorted by value)")
axA.set_ylabel("value")
axA.set_title("(a) Residual passes refine the reconstruction", fontsize=12.5,
              fontweight="bold", loc="left", color=INK)
axA.legend(frameon=False, fontsize=10.5, loc="lower right")
axA.margins(x=0.01)
for s in ("top","right"): axA.spines[s].set_visible(False)

# ---- (b) SNR vs passes ----
axB = fig.add_subplot(gs[0, 1]); axB.set_facecolor("white")
axB.plot(Ns, snrs, "-o", color="#C77D18", lw=2.2, ms=6)
axB.set_xlabel("number of passes N"); axB.set_ylabel("SNR (dB)")
axB.set_title(f"(b) SNR climbs ~{slope:.1f} dB / pass", fontsize=12, fontweight="bold",
              loc="left", color=INK)
axB.grid(True, color=GRID, lw=0.8)
axB.set_xticks(Ns)
for s in ("top","right"): axB.spines[s].set_visible(False)

# ---- (c) residual RMS vs passes (log) ----
axC = fig.add_subplot(gs[1, 1]); axC.set_facecolor("white")
axC.semilogy(Ns, rrms, "-o", color=TEAL, lw=2.2, ms=6)
axC.set_xlabel("number of passes N"); axC.set_ylabel("residual RMS")
axC.set_title("(c) Leftover error decays geometrically", fontsize=12, fontweight="bold",
              loc="left", color=INK)
axC.grid(True, which="both", color=GRID, lw=0.8)
axC.set_xticks(Ns)
for s in ("top","right"): axC.spines[s].set_visible(False)

fig.suptitle(r"GF4 multi-pass residual quantization:   $q \leftarrow q + \mathrm{GF4}(x - q)$",
             fontsize=16, fontweight="bold", color=INK, y=0.965)
fig.text(0.5, 0.885, "the 4-bit GF4 grid is re-used every pass — no added precision, no second datapath",
         ha="center", fontsize=11.5, color=MUT, style="italic")

out="/tmp/claude-241606/-home-jvap2-Desktop-Code-TIME-ML/f78e01f8-60ca-47f1-b0e2-eb3a6c07d272/scratchpad/gf4_multipass"
fig.savefig(out+".png", dpi=200, facecolor="white")
fig.savefig(out+".pdf", facecolor="white")
print("SNR per pass:", [f"{s:.2f}" for s in snrs])
print("resid RMS   :", [f"{r:.4f}" for r in rrms])
print("saved", out+".png/.pdf")

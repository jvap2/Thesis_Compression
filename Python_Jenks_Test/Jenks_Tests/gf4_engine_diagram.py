"""Render the GF4-Engine block diagram (preview PNG). Paper uses the TikZ version.
Front-end = Kronecker-tiled FULL-ROW FWHT (local B-butterflies -> corner-turn ->
cross-block butterflies), adders-only. Block-diagonal Hadamard collapses at scale
(opt-1.3b A16 -> 7405), so the transform must be full-row; the Kronecker tiling
keeps the compute local without sacrificing correctness."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, (axA, axB) = plt.subplots(2, 1, figsize=(12.5, 9.4),
                               gridspec_kw={"height_ratios": [0.82, 1.18]})
plt.subplots_adjust(hspace=0.28)

BLUE, GREEN, ORANGE, RED, ACC, GREY = "#cfe3f7", "#d6efd6", "#ffe4c4", "#f7cccc", "#cdeccd", "#e8e8e8"

def box(ax, x, y, w, h, label, color, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.3, edgecolor="#333", facecolor=color))
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")

def harrow(ax, x1, x2, y, label=None):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                                 mutation_scale=15, linewidth=1.5, color="#333"))
    if label:
        ax.text((x1+x2)/2, y + 0.22, label, ha="center", va="bottom",
                fontsize=8.5, color="#245", style="italic")

# ── Panel A: the quantize datapath (all horizontal, no crossing labels) ──────
axA.set_title("(a) GF4 datapath — full-row FWHT front-end fused into FP4 quantize",
              fontsize=13, fontweight="bold", loc="left")
yc = 1.5
box(axA, 0.2,  1.0, 1.5, 1.4, "SRAM\n4-bit acts", BLUE, fs=9.5)
box(axA, 2.4,  0.9, 2.0, 1.6, "FWHT\nfront-end\n(full-row,\ntiled, adders)", GREEN, fs=9.5, bold=True)
box(axA, 5.1,  0.9, 2.2, 1.6, "RMS scale +\ncodebook LUT\n$16\\times$E4M3", ORANGE, fs=9.5)
box(axA, 8.0,  1.05, 1.5, 1.3, "int4\nMAC", RED, fs=11, bold=True)
box(axA, 10.2, 0.9, 2.0, 1.6, "wide\naccumulator\n24-bit psum", ACC, fs=9.5, bold=True)
harrow(axA, 1.7,  2.4,  yc+0.2, "acts")
harrow(axA, 4.4,  5.1,  yc+0.2, "rotated")
harrow(axA, 7.3,  8.0,  yc+0.2, "decoded")
harrow(axA, 9.5,  10.2, yc+0.2)
# multi-pass loop under MAC<-accumulator
axA.add_patch(FancyArrowPatch((11.2, 0.9), (8.75, 0.55), connectionstyle="arc3,rad=0.28",
              arrowstyle="-|>", mutation_scale=13, linewidth=1.5, color="#b30", linestyle="--"))
axA.text(9.6, 0.16, "multi-pass residual  $Q \\leftarrow Q+\\mathrm{GF4}(x-Q)$",
         ha="center", fontsize=9, color="#b30")
axA.text(6.2, 0.15, "scale is Parseval-parallel ($\\|Hx\\|=\\|x\\|$)",
         ha="center", fontsize=8.3, color="#666")
axA.set_xlim(0, 12.5); axA.set_ylim(0, 2.9); axA.axis("off")

# ── Panel B: Kronecker-tiled full-row FWHT (extra vertical headroom) ──────────
axB.set_title("(b) FWHT front-end: full-row $H_M = H_G \\otimes H_B$ as tiled butterflies "
              "— block-diagonal collapses at scale",
              fontsize=12.5, fontweight="bold", loc="left")
box(axB, 0.2, 2.1, 1.4, 1.1, "$x$ : $M$\n$=G\\times B$", GREY, fs=10)
# stage 1: G local B-butterflies (3 stacked, generous gaps)
s1x, s1w = 2.1, 1.7
for i, yy in enumerate([3.35, 2.55, 1.75, 0.95]):
    lbl = "$\\vdots$" if i == 3 else f"$H_B$  block {i}"
    box(axB, s1x, yy, s1w, 0.6, lbl, GREEN, fs=9)
axB.text(s1x + s1w/2, 4.35, "Stage 1: $G$ local\n$B$-wide butterflies",
         ha="center", va="bottom", fontsize=9, fontweight="bold")
# corner turn
box(axB, 4.35, 2.0, 1.5, 1.3, "corner-turn\n(transpose)", BLUE, fs=9.5)
# stage 2: B cross-block G-butterflies
s2x, s2w = 6.5, 1.7
for i, yy in enumerate([3.35, 2.55, 1.75, 0.95]):
    lbl = "$\\vdots$" if i == 3 else f"$H_G$  mix {i}"
    box(axB, s2x, yy, s2w, 0.6, lbl, GREEN, fs=9)
axB.text(s2x + s2w/2, 4.35, "Stage 2: $B$ cross-block\n$G$-wide butterflies",
         ha="center", va="bottom", fontsize=9, fontweight="bold")
box(axB, 8.7, 2.1, 1.5, 1.1, "rotated\n$xH$", ORANGE, fs=10)
# arrows along the mid row (y~2.6)
for x1, x2 in [(1.6, 2.1), (s1x+s1w, 4.35), (5.85, s2x), (s2x+s2w, 8.7)]:
    axB.add_patch(FancyArrowPatch((x1, 2.6), (x2, 2.6), arrowstyle="-|>",
                  mutation_scale=14, linewidth=1.4, color="#333"))
# butterfly inset (well separated on the right)
bx, by = 10.9, 2.15
axB.text(bx+0.6, 3.5, "butterfly node", ha="center", fontsize=9, fontweight="bold")
axB.plot([bx, bx+1.2], [by+1.0, by+0.1], color="#333", lw=1.1)
axB.plot([bx, bx+1.2], [by+0.1, by+1.0], color="#333", lw=1.1)
axB.plot([bx, bx+1.2], [by+1.0, by+1.0], color="#333", lw=1.1)
axB.plot([bx, bx+1.2], [by+0.1, by+0.1], color="#333", lw=1.1)
axB.text(bx-0.12, by+1.0, "$a$", ha="right", va="center", fontsize=9.5)
axB.text(bx-0.12, by+0.1, "$b$", ha="right", va="center", fontsize=9.5)
axB.text(bx+1.32, by+1.0, "$a{+}b$", ha="left", va="center", fontsize=9.5)
axB.text(bx+1.32, by+0.1, "$a{-}b$", ha="left", va="center", fontsize=9.5)
axB.text(bx+0.6, by-0.55, "add/sub only ($\\pm1$),\nno multiplier", ha="center", fontsize=8.3, color="#b30")
axB.text(5.2, 0.25, "$\\log_2 M$ stages  •  adders only  •  full-row decorrelation "
         "(correct at scale)  •  reuses one small $B$-butterfly tile",
         ha="center", fontsize=9, color="#333")
axB.set_xlim(0, 12.5); axB.set_ylim(0, 4.7); axB.axis("off")

out = "/home/jvap2/Desktop/Code/TIME_ML/Python_Jenks_Test/Jenks_Tests/gf4_engine.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)

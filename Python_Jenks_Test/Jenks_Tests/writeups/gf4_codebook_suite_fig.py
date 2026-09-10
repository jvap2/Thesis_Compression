"""
gf4_codebook_suite_fig.py -- two-panel figure for the codebook-design result.
(a) dPPL vs model scale, 3 codebooks over 8 models / 3 families (W4A4): the
    optimized codebooks hug the GF4 baseline (iso-accuracy), rounding sits above.
(b) the solved-per-model codebook: near-identical integer levels across every
    model -> one universal lattice codebook (the GANQ-per-layer-LUT distinction).

Okabe-Ito categorical palette (CVD-safe by construction); color = codebook,
marker = family. Light-theme figure for the dissertation.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- data (W4A4, mean over 20-40 windows; dPPL vs exact GF4) ----------------
models = ["OPT-125m","OPT-1.3b","OPT-2.7b","OPT-6.7b","OPT-13b",
          "Mistral-7B","Qwen2.5-7B","Qwen2.5-14B"]
params = np.array([0.125, 1.3, 2.7, 6.7, 13.0, 7.24, 7.6, 14.7])       # billions
family = ["OPT","OPT","OPT","OPT","OPT","Mistral","Qwen","Qwen"]
dppl = {
    "round_Q1.4": np.array([-0.232, 0.090, 0.074, 0.135,-0.003, 0.050, 0.039, 0.065]),
    "opt_Q1.4":   np.array([-0.537,-0.087, 0.061, 0.111,-0.032,-0.019, 0.006, 0.018]),
    "opt_pop2":   np.array([-0.676,-0.142, 0.044, 0.125,-0.031,-0.015,-0.010, 0.003]),
}
# solved-per-model codebook (interior levels x16); all share {0,2,4,6,8,10,*,16}
solved = {m: [0,2,4,6,8,10,13,16] for m in models}
solved["Qwen2.5-7B"] = [0,2,4,6,8,10,12,16]     # the single one-step exception

# Okabe-Ito
C_ROUND, C_OPT, C_POP2 = "#D55E00", "#009E73", "#0072B2"
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d9"
MARK = {"OPT":"o", "Mistral":"D", "Qwen":"s"}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
                     "font.family": "DejaVu Sans"})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.4, 3.4), gridspec_kw={"width_ratios":[1.35,1]})

# ---- (a) dPPL vs scale -----------------------------------------------------
axA.axhspan(-0.8, 0, color=C_POP2, alpha=0.045, zorder=0)      # "beats GF4" region
axA.axhline(0, color=INK, lw=1.4, ls=(0,(4,3)), zorder=2)
for cb, col in (("round_Q1.4",C_ROUND),("opt_Q1.4",C_OPT),("opt_pop2",C_POP2)):
    y = dppl[cb]
    for i in range(len(models)):
        axA.plot(params[i], y[i], MARK[family[i]], color=col, ms=7, mec="white",
                 mew=0.8, zorder=4)
    axA.axhline(y.mean(), color=col, lw=1.4, ls=":", alpha=0.9, zorder=3)   # mean
axA.set_xscale("log")
axA.set_xlabel("Model size (billion parameters, log scale)")
axA.set_ylabel(r"$\Delta$PPL vs. hand-tuned GF4")
axA.set_ylim(-0.8, 0.25)
axA.set_xticks([0.125,1,3,7,14]); axA.set_xticklabels(["0.1","1","3","7","14"])
axA.grid(axis="y", color=GRID, lw=0.6, zorder=1)
axA.text(0.108, 0.012, "GF4 baseline", color=INK, fontsize=7.5, va="bottom")
axA.text(0.13, -0.77, "beats GF4", color=C_POP2, fontsize=7.5, style="italic", va="bottom")
axA.set_title("(a)  Accuracy vs. scale", fontsize=9.5, loc="left", color=INK)

leg_cb = [Line2D([],[],color=C_ROUND,marker="o",ms=6,ls=":",label="round  Q1.4"),
          Line2D([],[],color=C_OPT,  marker="o",ms=6,ls=":",label="opt  Q1.4"),
          Line2D([],[],color=C_POP2, marker="o",ms=6,ls=":",label="opt  +popcount$\\leq$2")]
leg_fam = [Line2D([],[],color=MUTED,marker="o",ms=6,ls="none",label="OPT"),
           Line2D([],[],color=MUTED,marker="D",ms=6,ls="none",label="Mistral"),
           Line2D([],[],color=MUTED,marker="s",ms=6,ls="none",label="Qwen")]
l1 = axA.legend(handles=leg_cb, loc="lower right", fontsize=7.3, frameon=False,
                title="codebook (dotted = mean)", title_fontsize=7.3)
axA.add_artist(l1)
axA.legend(handles=leg_fam, loc="upper left", fontsize=7.3, frameon=False,
           ncol=3, columnspacing=1.0, handletextpad=0.2)

# ---- (b) universality: solved codebook per model (solved_opt levels) -------
qi = models.index("Qwen2.5-7B")
for yi, m in enumerate(models):
    ks = solved[m]
    axB.plot(ks, [yi]*len(ks), "-", color=GRID, lw=1.0, zorder=1)
    for k in ks:
        is_dev = (m == "Qwen2.5-7B" and k == 12)     # the single one-step exception
        axB.plot(k, yi, "o", color=(C_ROUND if is_dev else C_POP2), ms=6.5,
                 mec="white", mew=0.8, zorder=3)
# annotation in the clear whitespace BELOW the grid, arrow straight up the
# otherwise-empty column k=12 (only the Qwen2.5-7B dot lives there).
axB.annotate("only exception:\nQwen2.5-7B ($13\\!\\to\\!12$)", xy=(12, qi),
             xytext=(12, len(models)+0.85), fontsize=7.3, color=INK,
             ha="center", va="top",
             arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
axB.set_yticks(range(len(models))); axB.set_yticklabels(models, fontsize=8)
axB.set_xticks([0,2,4,6,8,10,12,13,16])
axB.set_xticklabels(["0","2","4","6","8","10","12","13","16"], fontsize=7.5)
axB.set_xlabel(r"codebook level ($k$, on the $k/16$ lattice)")
axB.set_xlim(-1, 17); axB.set_ylim(-0.6, len(models)+2.3)
axB.invert_yaxis()
axB.grid(axis="x", color=GRID, lw=0.5, zorder=0)
for s in ("top","right"): axB.spines[s].set_visible(False)
axB.set_title("(b)  Solved codebook is universal", fontsize=9.5, loc="left", color=INK)

fig.tight_layout(w_pad=2.0)
out = __file__.rsplit("/",1)[0] + "/gf4_codebook_suite.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", out)
print("means: round %.3f  opt %.3f  pop2 %.3f" %
      (dppl["round_Q1.4"].mean(), dppl["opt_Q1.4"].mean(), dppl["opt_pop2"].mean()))

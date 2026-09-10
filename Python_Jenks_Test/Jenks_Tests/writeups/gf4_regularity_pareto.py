"""Accuracy vs decode-area Pareto for the GF4 codebook-regularity study.
Joins the PPL sweep (CUDA_FP4_Test/gf4_regularity_results.csv) with the RTL
decode-area sweep (rtl_prototype/synth_out/regularity/decode_area.csv) and
plots dPPL (vs exact GF4) against decode-area savings."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/Desktop/Code/TIME_ML/Python_Jenks_Test/Jenks_Tests")
PPL = os.path.join(BASE, "CUDA_FP4_Test", "gf4_regularity_results.csv")
AREA = os.path.join(BASE, "rtl_prototype", "synth_out", "regularity", "decode_area.csv")
OUT = os.path.join(BASE, "writeups", "gf4_regularity_pareto.png")

ppl = {r["codebook"]: r for r in csv.DictReader(open(PPL))}
area = {r["codebook"]: r for r in csv.DictReader(open(AREA))}

order = ["exact", "q1b5", "q1b4", "q1b3", "pow2"]
label = {"exact": "exact GF4 (Q1.7)", "q1b5": "Q1.5", "q1b4": "Q1.4",
         "q1b3": "Q1.3", "pow2": "pow2"}
xs, ys, names = [], [], []
for k in order:
    if k in ppl and k in area:
        xs.append(float(area[k]["area_savings_pct"]))
        ys.append(float(ppl[k]["dppl_exact"]))
        names.append(label[k])

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.plot(xs, ys, "-o", color="#1f77b4", lw=1.5, ms=7, zorder=3)
for x, y, n in zip(xs, ys, names):
    dy = 0.6 if n != "Q1.4" else -1.4
    ax.annotate(n, (x, y), textcoords="offset points", xytext=(6, 8 if y < 5 else -12),
                fontsize=9)
# highlight the knee
ki = names.index("Q1.4")
ax.scatter([xs[ki]], [ys[ki]], s=180, facecolors="none", edgecolors="#d62728", lw=2, zorder=4)
ax.axhline(0, color="grey", lw=0.6, ls=":")
ax.set_xlabel("Decode-ROM area savings vs. exact GF4 (%)")
ax.set_ylabel(r"$\Delta$PPL vs. exact GF4 (WikiText-2, OPT-125M)")
ax.set_title("GF4 codebook regularity: accuracy vs. decode area")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=160)
print(f"wrote {OUT}")
for n, x, y in zip(names, xs, ys):
    print(f"  {n:18s} area -{x:5.1f}%   dPPL {y:+7.3f}")

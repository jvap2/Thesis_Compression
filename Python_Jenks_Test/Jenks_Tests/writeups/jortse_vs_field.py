#!/usr/bin/env python3
"""JORTsE vs Benchmarks: accuracy, sparsity, total iterations.

- Benchmark (external methods only: GSM, DNS, DENNC, Li2016, SNIP, GRaSP, ProbMask)
  drawn as a grey min-max ribbon per category.
- JORTsE drawn as two operating-point lines (moderate / high sparsity, colors unchanged),
  with the region between them shaded blue ("JORTsE range"). Lines coincide where only one
  JORTsE result exists, so the blue band has zero width there.
- x-axis: biggest -> smallest. TinyImageNet/ResNet-32 included.
Outputs individual figures and one stacked 3-panel figure (PNG 300 dpi + PDF).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT = "/home/jvap2/Desktop/Code/TIME_ML/Python_Jenks_Test/Jenks_Tests/writeups/"

# x-axis: biggest -> smallest
cats = ["TinyImageNet\nResNet-32", "CIFAR-100\nResNet-32", "CIFAR-100\nVGG-19",
        "CIFAR-10\nResNet-32", "CIFAR-10\nVGG-19", "CIFAR-10\nDenseNet-40",
        "CIFAR-10\nResNet56", "MNIST\nLeNet-300", "MNIST\nLeNet-5"]
x = np.arange(len(cats))

# ---- field (external methods only) min/max per category ----
acc_field = [(49.57, 57.25), (54.81, 74.09), (58.46, 74.48), (87.51, 94.96), (92.05, 94.00),
             (93.90, 94.07), (93.06, 94.10), (97.98, 98.18), (98.59, 99.22)]
spa_field = [(90, 98), (90, 98), (90, 99.5), (90, 98), (90, 98),
             (85, 90), (13.70, 90), (95.89, 98.34), (93.09, 99.67)]
tot_field = [(234600, 234600), (58800, 62560), (58800, 62560), (58800, 62560), (58800, 62560),
             (468600, 468600), (859100, 859100), (25000, 93600), (16000, 93600)]

# ---- JORTsE: line a (moderate sparsity), line b (high sparsity); b==a if only one result ----
acc_a = [51.12, 72.49, 70.67, 93.55, 94.11, 93.86, 93.10, 97.68, 98.86]
acc_b = [51.12, 67.70, 65.45, 92.95, 90.79, 93.86, 93.10, 97.68, 98.86]
spa_a = [89.40, 92.126, 92.42, 86.57, 90.32, 90.06, 89.18, 97.86, 99.50]
spa_b = [89.40, 95.20, 98.29, 95.30, 99.51, 90.06, 89.18, 97.86, 99.50]
tot_a = [78200, 58450, 98000, 75000, 98000, 156400, 195500, 30550, 30550]
tot_b = [78200, 58450, 98000, 58450, 98000, 156400, 195500, 30550, 30550]

FIELD = "#9aa0a6"
JBAND = "#1f4e79"          # blue JORTsE range fill
CA, CB = "#c0392b", "#1f4e79"

def field_ribbon(ax, band):
    lo = np.array([b[0] for b in band]); hi = np.array([b[1] for b in band])
    ax.fill_between(x, lo, hi, color=FIELD, alpha=0.45, zorder=1, label="Benchmark range")
    ax.plot(x, lo, color=FIELD, lw=1.0, zorder=2)
    ax.plot(x, hi, color=FIELD, lw=1.0, zorder=2)

def jortse(ax, ya, yb):
    ya = np.array(ya); yb = np.array(yb)
    ax.fill_between(x, ya, yb, color=JBAND, alpha=0.20, zorder=3, label="JORTsE range")
    ax.plot(x, ya, color=CA, lw=1.8, marker="o", ms=6, zorder=4, label="JORTsE (moderate sparsity)")
    ax.plot(x, yb, color=CB, lw=1.8, ls="--", marker="s", ms=5, zorder=4, label="JORTsE (high sparsity)")

def panel(ax, band, ya, yb, ylabel, title, logy=False, show_x=True, legend_loc="best"):
    field_ribbon(ax, band); jortse(ax, ya, yb)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12, pad=8)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)
    ax.set_xlim(-0.4, len(cats) - 0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(cats if show_x else [""] * len(cats), fontsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}"))
    if legend_loc:
        ax.legend(fontsize=8.5, framealpha=0.9, loc=legend_loc)

def save(fig, name):
    fig.savefig(OUT + name + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT + name + ".pdf", bbox_inches="tight")
    print("wrote", OUT + name + ".png / .pdf")

# ---- individual figures ----
specs = [
    (acc_field, acc_a, acc_b, "Top-1 accuracy (%)", "JORTsE vs Benchmarks: Accuracy", False, "lower right", "fig_accuracy"),
    (spa_field, spa_a, spa_b, "Sparsity (%)", "JORTsE vs Benchmarks: Sparsity", False, "lower left", "fig_sparsity"),
    (tot_field, tot_a, tot_b, "Total training iterations", "JORTsE vs Benchmarks: Training cost", True, "upper right", "fig_total_iters"),
]
for band, ya, yb, yl, ti, logy, loc, name in specs:
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    panel(ax, band, ya, yb, yl, ti, logy=logy, show_x=True, legend_loc=loc)
    fig.tight_layout(); save(fig, name)

# ---- stacked 3-panel figure ----
fig, axes = plt.subplots(3, 1, figsize=(9.5, 12), sharex=True)
panel(axes[0], acc_field, acc_a, acc_b, "Top-1 accuracy (%)", "Accuracy", show_x=False, legend_loc="lower right")
panel(axes[1], spa_field, spa_a, spa_b, "Sparsity (%)", "Sparsity", show_x=False, legend_loc="lower left")
panel(axes[2], tot_field, tot_a, tot_b, "Total training iterations", "Training cost", logy=True, show_x=True, legend_loc="upper right")
fig.suptitle("JORTsE vs Benchmarks", fontsize=14, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.99])
save(fig, "fig_jortse_vs_field_stacked")

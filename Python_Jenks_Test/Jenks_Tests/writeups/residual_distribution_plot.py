"""
Residual-distribution diagnostic for the GF4 W4A4 paper.

Hypothesis (user): down-proj WEIGHTS are heavier-tailed; the input ACTIVATIONS
are closer to Gaussian. We visualize, for one down-proj layer of OPT-125m:
  (col 1) raw element distribution        (weight vs activation)
  (col 2) after randomized-Hadamard       (should Gaussianize -> lower kurtosis)
  (col 3) 1st-pass quantization residual   (weight: NVFP4;  act: GF4), rotated domain
  (col 4) residual RMS decay across passes (Lemma-2 geometric-decay check)

Standardized (zero mean, unit std) so shapes are comparable; log-y to expose
tails; excess kurtosis annotated (Gaussian = 0, heavy-tailed > 0).

CPU-only (small model; leaves the GPU for the ResNet run). Saves PNG + PDF.
"""
import os, math, torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kurtosis  # Fisher (excess): Gaussian -> 0

torch.manual_seed(0)
DEV = "cpu"
BLOCK, CLIP = 16, 2.5

# ---- codebooks + kernels (verbatim from multipass_vs_retention notebook) ----
GF4_POS = torch.tensor([0.0, 0.0796082, 0.1737177, 0.2828685,
                        0.3952704, 0.5250730, 0.6961928, 1.0])
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

def fwht(x):
    n = x.shape[-1]; orig = x.shape
    x = x.reshape(-1, n).clone(); h = 1
    while h < n:
        x = x.view(x.shape[0], n // (2*h), 2, h)
        a = x[:, :, 0, :].clone(); b = x[:, :, 1, :].clone()
        x[:, :, 0, :] = a + b; x[:, :, 1, :] = a - b
        x = x.reshape(x.shape[0], n); h *= 2
    return (x / math.sqrt(n)).reshape(orig)

def rotate(x, D, Kpad):
    K = x.shape[-1]
    if Kpad != K: x = F.pad(x, (0, Kpad - K))
    return fwht(x * D)

def to_e4m3(x):
    x = x.float().clamp(min=2.0**-9, max=448.0)
    e = torch.floor(torch.log2(x)).clamp(min=-6.0); s = 2.0**e
    return (torch.round((x/s)*8)/8 * s).clamp(max=448.0)

def quant_nvfp4(W, block=BLOCK):
    shape = W.shape; x = W.reshape(-1, block).float()
    absmax = x.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    scale = to_e4m3(absmax/6.0); xn = x/scale
    sign = xn.sign(); mag = xn.abs().clamp(max=6.0); q = E2M1
    idx = (mag.unsqueeze(-1) - q.view(1,1,-1)).abs().argmin(-1)
    return (sign*q[idx]*scale).reshape(shape).to(W.dtype)

def gf4(x, block=BLOCK, clip=CLIP):
    shape = x.shape; K = shape[-1]; x2 = x.reshape(-1, K).float()
    pad = (block - K % block) % block; xp = F.pad(x2, (0, pad)); Kp = xp.shape[1]
    xb = xp.reshape(-1, block)
    rms = xb.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-8); scale = rms*clip
    sign = xb.sign(); xn = (xb.abs()/scale).clamp(0, 1); L = GF4_POS
    q = L[(xn.unsqueeze(-1) - L.view(1,1,-1)).abs().argmin(-1)]
    xh = (sign*scale*q).reshape(x2.shape[0], Kp)[:, :K]
    return xh.reshape(shape).to(x.dtype)

def npass_residuals(x, quant, n, block=BLOCK):
    """Return list of residual tensors r_0=x, r_1, ..., r_n under `quant`."""
    xf = x.float(); q = torch.zeros_like(xf); res = [xf.clone()]
    for _ in range(n):
        q = q + quant(xf - q); res.append((xf - q).clone())
    return res

def std1(t):  # standardize to zero mean, unit std, flatten to numpy
    t = t.float().flatten(); return ((t - t.mean())/t.std().clamp(min=1e-9)).numpy()

# ---- grab one down-proj (fc2) weight + its input activation ------------------
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
MODEL = "facebook/opt-125m"
LAYER = 6  # middle decoder block
print(f"loading {MODEL} (CPU) ...")
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(DEV).eval()

fc2_name = f"model.decoder.layers.{LAYER}.fc2"
fc2 = dict(model.named_modules())[fc2_name]
W = fc2.weight.data.clone()  # [d_model, d_ff] = [768, 3072]

captured = {}
def hook(mod, inp, out): captured["x"] = inp[0].detach().clone()
h = fc2.register_forward_hook(hook)
test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
ids = tok("\n\n".join(test["text"][:200]), return_tensors="pt").input_ids[:, :2048]
with torch.no_grad(): model(ids)
h.remove()
X = captured["x"].reshape(-1, W.shape[1])  # [tokens, d_ff] activations into fc2
print(f"  weight {tuple(W.shape)}  activation {tuple(X.shape)}")

# rotated-domain versions (fold D into W offline; apply to acts online)
Kpad = 1 << (W.shape[1]-1).bit_length()
g = torch.Generator().manual_seed(Kpad)
D = (torch.randint(0,2,(Kpad,),generator=g).float()*2-1)
Wr = rotate(W.float(), D, Kpad)
Xr = rotate(X.float(), D, Kpad)

# 1st-pass residuals in the rotated domain
Wres1  = Wr - quant_nvfp4(Wr)          # weight, NVFP4
Xres1  = Xr - gf4(Xr)                  # activation, GF4 (pipeline default)
Xres1n = Xr - quant_nvfp4(Xr)          # activation, NVFP4 (counterfactual: weight format on acts)

# residual RMS decay across passes
NP = 5
Wres  = npass_residuals(Wr, quant_nvfp4, NP)
Xres  = npass_residuals(Xr, gf4,         NP)
Xresn = npass_residuals(Xr, quant_nvfp4, NP)   # activation under NVFP4
def rms(t): return t.float().pow(2).mean().sqrt().item()
Wdecay  = [20*math.log10(rms(Wres[0])/max(rms(r),1e-12))  for r in Wres]
Xdecay  = [20*math.log10(rms(Xres[0])/max(rms(r),1e-12))  for r in Xres]
Xdecayn = [20*math.log10(rms(Xresn[0])/max(rms(r),1e-12)) for r in Xresn]

# ---- plot (POST-HADAMARD focus) --------------------------------------------
from scipy.stats import probplot
INK="#1B2027"; WCOL="#B4610F"; ACOL="#0C7A69"; RAWC="#9AA1AA"
Wres2 = Wres[2]; Xres2 = Xres[2]   # 2-pass residuals for the residual-shape panel
fig, ax = plt.subplots(2, 4, figsize=(16, 7.2))
fig.suptitle(f"Down-proj (fc2) of {MODEL}, layer {LAYER} — POST-HADAMARD distributions "
             f"(what the pipeline actually quantizes)", fontsize=13, fontweight="bold", y=0.98)

def hist_panel(a, data, color, label, title, raw=None):
    d = data; k = kurtosis(d, fisher=True)
    bins = np.linspace(-6, 6, 160)
    if raw is not None:   # faint raw (pre-rotation) shadow for contrast
        a.hist(raw, bins=bins, density=True, color=RAWC, alpha=0.35,
               label=f"raw (kurt={kurtosis(raw,fisher=True):.1f})")
    a.hist(d, bins=bins, density=True, color=color, alpha=0.70,
           label=f"{label} (kurt={k:.2f})")
    xs = np.linspace(-6, 6, 400)
    a.plot(xs, np.exp(-xs**2/2)/math.sqrt(2*math.pi), "--", color=INK, lw=1.2, label="N(0,1)")
    a.set_yscale("log"); a.set_ylim(1e-5, 1); a.set_xlim(-6, 6)
    a.set_title(title, fontsize=10.5); a.legend(fontsize=8, loc="upper right"); a.grid(alpha=0.15)

def qq_panel(a, data, color, title):
    (osm, osr), (sl, ic, r) = probplot(data, dist="norm")
    a.plot(osm, osr, ".", color=color, ms=2, alpha=0.5)
    lim = [min(osm.min(),osr.min()), max(osm.max(),osr.max())]
    a.plot(lim, lim, "--", color=INK, lw=1.0)
    a.set_title(title, fontsize=10.5); a.set_xlabel("normal quantile", fontsize=8)
    a.set_ylabel("sample quantile", fontsize=8); a.grid(alpha=0.15)

# Row 0 = rotated WEIGHT, Row 1 = rotated ACTIVATION
hist_panel(ax[0,0], std1(Wr), WCOL, "rotated weight", "Rotated weight vs N(0,1)", raw=std1(W))
qq_panel  (ax[0,1], std1(Wr), WCOL, "Rotated weight — normal QQ")
hist_panel(ax[0,2], std1(Wres1), WCOL, "1-pass resid", "Weight residual (NVFP4), post-H")
hist_panel(ax[1,0], std1(Xr), ACOL, "rotated activation", "Rotated activation vs N(0,1)", raw=std1(X))
qq_panel  (ax[1,1], std1(Xr), ACOL, "Rotated activation — normal QQ")
hist_panel(ax[1,2], std1(Xres1), ACOL, "1-pass resid", "Activation residual (GF4), post-H")

# decay panel (span both rows, col 4)
ax[0,3].remove(); ax[1,3].remove()
axd = fig.add_subplot(1, 4, 4)
passes = list(range(NP+1))
axd.plot(passes, Wdecay,  "o-",  color=WCOL, label="weight · NVFP4 — SATURATES")
axd.plot(passes, Xdecay,  "s-",  color=ACOL, label="activation · GF4 — keeps decaying")
axd.plot(passes, Xdecayn, "^--", color=INK,  label="activation · NVFP4 (counterfactual)")
axd.set_xlabel("residual passes K"); axd.set_ylabel("reconstruction SNR (dB)")
axd.set_title("Post-H residual decay (Lemma 2)", fontsize=10.5)
axd.legend(fontsize=8.5); axd.grid(alpha=0.2)
axd.text(0.03, 0.03, "NVFP4 floors: E4M3-quantized scale\nGF4: float RMS scale",
         transform=axd.transAxes, fontsize=8, style="italic", color=INK)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(os.path.dirname(__file__), "residual_distributions")
plt.savefig(out+".png", dpi=150); plt.savefig(out+".pdf")
print(f"saved {out}.png / .pdf")

# ---- print the numbers so they're captured in the log too -------------------
print("\n== excess kurtosis (Gaussian=0; higher = heavier tails) ==")
for nm, t in [("raw weight", W), ("raw activation", X),
              ("rotated weight", Wr), ("rotated activation", Xr),
              ("W 1-pass residual (NVFP4)", Wres1),
              ("A 1-pass residual (GF4)",   Xres1),
              ("A 1-pass residual (NVFP4)", Xres1n)]:
    print(f"  {nm:28s}: {kurtosis(t.float().flatten().numpy(), fisher=True):8.2f}")
print("\n== residual SNR (dB) by pass ==")
print("  weight     · NVFP4:", ", ".join(f"{v:.1f}" for v in Wdecay))
print("  activation · GF4  :", ", ".join(f"{v:.1f}" for v in Xdecay))
print("  activation · NVFP4:", ", ".join(f"{v:.1f}" for v in Xdecayn))

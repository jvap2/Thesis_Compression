"""
Why GF4 beats NVFP4 on post-rotation (near-Gaussian) data: the granularity
argument, measured.

Mechanism under test: NVFP4's block scale is ABSMAX-based (scale=absmax/6), so a
single heavy value in a block stretches the whole codebook and coarsens the BULK
step by the block dynamic range kappa=absmax/RMS. GF4's RMS-based scale
(scale=RMS*2.5) is outlier-robust, so its bulk step is kappa-independent.
Prediction: bulk quantization error grows ~kappa^2 for NVFP4, flat for GF4;
they tie at kappa ~ 2.5 (clean Gaussian block of 16) and diverge above it.

Two experiments:
  (A) synthetic: 16-sample blocks = 15 x N(0,1) + 1 injected outlier at kappa*RMS,
      sweep kappa; measure BULK RMSE (error on the 15 Gaussian entries only).
  (B) real: rotated OPT-125m fc2 activation blocks, bin by their own kappa.
Saves a 3-panel figure + prints the numbers.
"""
import os, math, torch, numpy as np
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
BLOCK = 16
GF4_POS = torch.tensor([0.0, 0.0796082, 0.1737177, 0.2828685,
                        0.3952704, 0.5250730, 0.6961928, 1.0])
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

def to_e4m3(x):
    x = x.float().clamp(min=2.0**-9, max=448.0)
    e = torch.floor(torch.log2(x)).clamp(min=-6.0); s = 2.0**e
    return (torch.round((x/s)*8)/8 * s).clamp(max=448.0)

def gf4_block(xb, clip=2.5):
    rms = xb.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-8); scale = rms*clip
    sign = xb.sign(); xn = (xb.abs()/scale).clamp(0, 1)
    q = GF4_POS[(xn.unsqueeze(-1) - GF4_POS.view(1,1,-1)).abs().argmin(-1)]
    return sign*scale*q

def nvfp4_block(xb):
    absmax = xb.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    scale = to_e4m3(absmax/6.0); xn = xb/scale
    sign = xn.sign(); mag = xn.abs().clamp(max=6.0)
    q = E2M1[(mag.unsqueeze(-1) - E2M1.view(1,1,-1)).abs().argmin(-1)]
    return sign*q*scale

# ---- (A) synthetic kappa sweep ---------------------------------------------
def bulk_rmse_vs_kappa(kappas, nblk=4000):
    gf, nv = [], []
    for kap in kappas:
        g = torch.randn(nblk, BLOCK)
        g = g / g.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-8)  # unit RMS
        b = g.clone(); b[:, 0] = kap * b.sign()[:, 0].clamp(min=1)      # inject outlier at col 0
        b[:, 0] = kap  # deterministic +kappa outlier (in RMS units)
        qg, qn = gf4_block(b), nvfp4_block(b)
        # BULK error = error on the non-outlier 15 entries
        eg = (b[:, 1:] - qg[:, 1:]).pow(2).mean().sqrt().item()
        en = (b[:, 1:] - qn[:, 1:]).pow(2).mean().sqrt().item()
        gf.append(eg); nv.append(en)
    return np.array(gf), np.array(nv)

kappas = np.array([2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0, 20.0])
gf_e, nv_e = bulk_rmse_vs_kappa(kappas)
print("== (A) synthetic BULK RMSE vs block dynamic range kappa=absmax/RMS ==")
print("  kappa :", ", ".join(f"{k:5.1f}" for k in kappas))
print("  GF4   :", ", ".join(f"{v:.4f}" for v in gf_e))
print("  NVFP4 :", ", ".join(f"{v:.4f}" for v in nv_e))
print("  ratio :", ", ".join(f"{n/g:5.2f}" for g, n in zip(gf_e, nv_e)))

# ---- (B) real rotated fc2 activation, binned by per-block kappa --------------
def real_blocks():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=False)
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=torch.float32).eval()
    cap = {}
    fc2 = dict(model.named_modules())["model.decoder.layers.6.fc2"]
    h = fc2.register_forward_hook(lambda m,i,o: cap.__setitem__("x", i[0].detach()))
    test = load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    ids = tok("\n\n".join(test["text"][:200]), return_tensors="pt").input_ids[:, :2048]
    with torch.no_grad(): model(ids)
    h.remove()
    X = cap["x"].reshape(-1, fc2.weight.shape[1])
    # rotate
    K = X.shape[-1]; Kpad = 1 << (K-1).bit_length()
    g = torch.Generator().manual_seed(Kpad); D = (torch.randint(0,2,(Kpad,),generator=g).float()*2-1)
    def fwht(x):
        n=x.shape[-1]; x=x.reshape(-1,n).clone(); h=1
        while h<n:
            x=x.view(x.shape[0],n//(2*h),2,h); a=x[:,:,0,:].clone(); b=x[:,:,1,:].clone()
            x[:,:,0,:]=a+b; x[:,:,1,:]=a-b; x=x.reshape(x.shape[0],n); h*=2
        return x/math.sqrt(n)
    Xr = fwht(F.pad(X, (0, Kpad-K)) * D)
    return Xr.reshape(-1, BLOCK)

try:
    xb = real_blocks()
    kap = (xb.abs().amax(-1) / xb.pow(2).mean(-1).sqrt().clamp(min=1e-8))
    qg, qn = gf4_block(xb), nvfp4_block(xb)
    # per-block bulk error (drop the argmax entry of each block)
    amx = xb.abs().argmax(-1)
    mask = torch.ones_like(xb, dtype=torch.bool); mask[torch.arange(len(xb)), amx] = False
    eg = ((xb-qg)*mask).pow(2).sum(-1).sqrt() / math.sqrt(BLOCK-1)
    en = ((xb-qn)*mask).pow(2).sum(-1).sqrt() / math.sqrt(BLOCK-1)
    bins = [ (2,3),(3,4),(4,5),(5,7),(7,10),(10,100) ]
    print("\n== (B) real rotated fc2 blocks, BULK RMSE by kappa bin ==")
    print(f"  kappa distribution: median {kap.median():.2f}, 90pct {kap.quantile(0.9):.2f}, max {kap.max():.2f}")
    rk, rg, rn = [], [], []
    for lo,hi in bins:
        m = (kap>=lo)&(kap<hi)
        if m.sum()<20: continue
        rk.append((lo+hi)/2); rg.append(eg[m].mean().item()); rn.append(en[m].mean().item())
        print(f"  kappa[{lo:2d},{hi:3d}) n={int(m.sum()):6d}  GF4 {eg[m].mean():.4f}  NVFP4 {en[m].mean():.4f}  ratio {en[m].mean()/eg[m].mean():.2f}")
    have_real = True
except Exception as e:
    print("real-block step skipped:", type(e).__name__, str(e)[:80]); have_real=False

# ---- figure -----------------------------------------------------------------
INK="#1B2027"; GCOL="#0C7A69"; NCOL="#B4610F"
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

# panel 1: synthetic bulk RMSE vs kappa (log-log => slope = power law)
ax[0].loglog(kappas, gf_e, "s-", color=GCOL, label="GF4 (RMS scale)")
ax[0].loglog(kappas, nv_e, "o-", color=NCOL, label="NVFP4 (absmax scale)")
ax[0].loglog(kappas, nv_e[1]*(kappas/kappas[1])**2 * 0 + nv_e[1]*(kappas/kappas[1]), ":", color=INK, lw=1,
             label="slope-1 (∝κ) guide")
ax[0].set_xlabel("block dynamic range κ = absmax/RMS"); ax[0].set_ylabel("BULK RMSE (non-outlier entries)")
ax[0].set_title("(A) synthetic: NVFP4 bulk error grows with κ"); ax[0].legend(fontsize=8.5); ax[0].grid(alpha=0.2, which="both")

# panel 2: codebook level placement over Gaussian, at a heavy-block scale
xs = np.linspace(0, 4, 500)
ax[1].plot(xs, np.exp(-xs**2/2)/math.sqrt(2*math.pi)*2, color=INK, lw=1.2, label="|N(0,1)| density")
kap_demo = 6.0
gf_lvls = (GF4_POS*2.5).numpy()                        # RMS units (RMS=1)
nv_lvls = (E2M1 * (kap_demo/6.0)).numpy()              # absmax=kap_demo*RMS
for L in gf_lvls: ax[1].axvline(L, color=GCOL, lw=1.1, alpha=0.8)
for L in nv_lvls: ax[1].axvline(L, color=NCOL, lw=1.1, alpha=0.6, ls="--")
ax[1].axvline(gf_lvls[0], color=GCOL, lw=1.1, alpha=0.8, label="GF4 levels")
ax[1].axvline(nv_lvls[0], color=NCOL, lw=1.1, alpha=0.6, ls="--", label=f"NVFP4 levels (κ={kap_demo:.0f})")
ax[1].set_xlim(0, 4); ax[1].set_xlabel("magnitude (RMS units)"); ax[1].set_ylabel("density")
ax[1].set_title("(B) level placement on a heavy block:\nNVFP4 levels flee to the outlier, bulk starves")
ax[1].legend(fontsize=8.5); ax[1].grid(alpha=0.15)

# panel 3: real blocks bulk RMSE by kappa bin
if have_real and rk:
    ax[2].plot(rk, rg, "s-", color=GCOL, label="GF4")
    ax[2].plot(rk, rn, "o-", color=NCOL, label="NVFP4")
    ax[2].set_xlabel("block κ bin center"); ax[2].set_ylabel("BULK RMSE")
    ax[2].set_title("(C) real rotated fc2 acts:\nGF4 wins on high-κ blocks"); ax[2].legend(fontsize=9); ax[2].grid(alpha=0.2)
else:
    ax[2].text(0.5,0.5,"real-block panel skipped", ha="center", transform=ax[2].transAxes)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "gf4_vs_nvfp4_granularity")
plt.savefig(out+".png", dpi=150); plt.savefig(out+".pdf")
print(f"\nsaved {out}.png / .pdf")

"""
validate_multipass.py — does multi-pass residual GF4 earn the "no FP16 unit"
claim?  i.e. does running the OUTLIER-retention layers (down_proj/fc2/lm_head)
at N FP4 passes recover the accuracy of keeping them in FP16?

This is the empirical backing for the hardware model in hw_sim_gf4.py, where the
minimal "GF4 multipass" engine has NO FP16 multiplier — precision on the outlier
layers comes from pass count alone (~+9 dB SNR per pass: 1 pass ≈ 3 eff. bits,
2 ≈ 6 (≈A16), 3-4 enough to recover FP16-retention accuracy).  Self-contained
(inline GF4, no bit_split import).

Two modes:

  python3 validate_multipass.py recon
      LOCAL, no GPU.  Reconstruction SNR of 1/2/3/4-pass residual GF4 on Gaussian
      weights (and a real weight if a .pt path is given).  Shows the precision
      ladder and how close 4-pass gets to FP16.

  python3 validate_multipass.py ppl facebook/opt-1.3b
      GPU.  WikiText-2 perplexity with all weights GF4 fake-quantized, sweeping
      the OUTLIER layers over {FP16-retain, 1, 2, 4 passes}.  The claim holds if
      4-pass PPL ~= FP16-retain PPL.  Needs torch + transformers + datasets.
"""
import sys, math
import torch

GF4_POS = torch.tensor([0.0, 0.0796082, 0.1737177, 0.2828685,
                        0.3952704, 0.5250730, 0.6961928, 1.0])

# layers single-pass FP4 collapses on — the retention set (== iso_energy _mlp_skip)
OUTLIER_SUBSTR = ("down_proj", "fc2", "dense_4h_to_h", "lm_head", "embed_out")


def gf4_quant(x, block_size=16, clip_ratio=2.5, levels=GF4_POS):
    """Single-pass Gaussian-optimal FP4 (mirrors bit_split.quantize_activations_gf4)."""
    levels = levels.to(x.device)
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).float()
    N, K = x2.shape
    pad = (block_size - K % block_size) % block_size
    xp = torch.nn.functional.pad(x2, (0, pad))
    Kp = xp.shape[1]
    xb = xp.reshape(-1, block_size)
    rms = xb.pow(2).mean(-1).sqrt().clamp(min=1e-8)
    scale = (rms * clip_ratio).unsqueeze(-1)
    sign = torch.sign(xb)
    xn = (xb.abs() / scale).clamp(0, 1)
    q = levels[(xn.unsqueeze(-1) - levels.view(1, 1, -1)).abs().argmin(-1)]
    xh = (sign * scale * q).reshape(N, Kp)[:, :K]
    return xh.reshape(shape).to(x.dtype)


def npass(x, block_size=16, n_pass=2, clip_ratio=2.5, levels=GF4_POS):
    """N-stage residual GF4 — accumulate n_pass FP4 codes (the multi-pass mode)."""
    xf = x.float()
    q = torch.zeros_like(xf)
    for _ in range(n_pass):
        q = q + gf4_quant(xf - q, block_size, clip_ratio, levels)
    return q.to(x.dtype)


def snr_db(x, xh):
    return 20.0 * math.log10((x.float().norm() / (x.float() - xh.float()).norm()).item())


# ── recon mode (local) ───────────────────────────────────────────────────────
def run_recon(weight_path=None):
    torch.manual_seed(0)
    print("== multi-pass residual GF4: reconstruction SNR ==")
    print("  (each pass re-quantizes the leftover residual; ~6 dB ≈ 1 effective bit)\n")
    tensors = {"Gaussian N(0,1) [4096x4096]": torch.randn(4096, 4096)}
    # add a heavy-tailed tensor (outlier-like) and an optional real weight
    g = torch.randn(4096, 4096); g[:, ::128] *= 12.0   # inject outlier channels
    tensors["Gaussian + outlier channels"] = g
    if weight_path:
        W = torch.load(weight_path, map_location="cpu")
        tensors[f"real weight {tuple(W.shape)}"] = W.float()
    for name, W in tensors.items():
        print(f"  {name}")
        prev = 0.0
        for n in (1, 2, 3, 4):
            s = snr_db(W, npass(W, 16, n))
            bits = (s - 1.76) / 6.02
            print(f"    {n}-pass : SNR {s:6.2f} dB  (~{bits:4.1f} eff. bits)  "
                  f"{'+%.1f dB vs prev'%(s-prev) if prev else ''}")
            prev = s
        # FP16 reference error (store-as-fp16 round trip)
        s16 = snr_db(W, W.half().float())
        print(f"    FP16   : SNR {s16:6.2f} dB  (reference: keep in FP16)\n")
    print("  Read: if 4-pass SNR >= the 2-pass (residual-GF4 ~A16) level and keeps")
    print("  climbing ~linearly, the outlier layers don't need an FP16 unit — the")
    print("  accumulator's pass count delivers the precision.")


# ── ppl mode (GPU) ───────────────────────────────────────────────────────────
def run_ppl(model_id, n_passes=(1, 2, 4), seqlen=2048, nsamp=40):
    import torch as T
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    dev = "cuda" if T.cuda.is_available() else "cpu"
    print(f"== {model_id}: WikiText-2 PPL, outlier layers at FP16 vs N-pass ==")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids

    def is_outlier(name):
        return any(s in name for s in OUTLIER_SUBSTR)

    def build(outlier_passes):
        """Fresh model; fake-quant every Linear weight. Bulk=1-pass GF4.
        Outliers: FP16 if outlier_passes is None, else N-pass residual GF4."""
        m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=T.float16).to(dev).eval()
        with T.no_grad():
            for name, mod in m.named_modules():
                if not isinstance(mod, T.nn.Linear):
                    continue
                if is_outlier(name):
                    if outlier_passes is None:
                        continue                       # keep FP16 (retention baseline)
                    mod.weight.copy_(npass(mod.weight.data, 16, outlier_passes).half())
                else:
                    mod.weight.copy_(gf4_quant(mod.weight.data, 16).half())  # bulk W4
        return m

    @T.no_grad()
    def ppl(m):
        n = enc.numel() // seqlen
        nll, cnt = 0.0, 0
        for i in range(min(n, nsamp)):
            ids = enc[:, i*seqlen:(i+1)*seqlen].to(dev)
            out = m(ids, labels=ids)
            nll += out.loss.float().item() * (seqlen - 1)
            cnt += seqlen - 1
        return math.exp(nll / cnt)

    rows = [("FP16-retain (baseline)", None)] + [(f"{p}-pass residual", p) for p in n_passes]
    print(f"  {'outlier treatment':28s} {'PPL':>8s}   {'Δ vs FP16-retain':>16s}")
    base = None
    for label, p in rows:
        v = ppl(build(p))
        if base is None:
            base = v
        print(f"  {label:28s} {v:8.3f}   {('%+.3f'%(v-base)) if p is not None else 'baseline':>16s}")
        T.cuda.empty_cache()
    print("\n  Claim holds if the 4-pass row ≈ the FP16-retain baseline: the outlier")
    print("  layers can run on the FP4 array (4 passes) with no dedicated FP16 unit.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "recon"
    if mode == "recon":
        run_recon(sys.argv[2] if len(sys.argv) > 2 else None)
    elif mode == "ppl":
        run_ppl(sys.argv[2] if len(sys.argv) > 2 else "facebook/opt-1.3b")
    else:
        sys.exit("usage: validate_multipass.py [recon [weight.pt] | ppl <model_id>]")

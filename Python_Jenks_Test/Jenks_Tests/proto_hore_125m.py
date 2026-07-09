"""
Prototype: a NOVEL, strictly SINGLE-TERM (genuine single-GEMM W4A4) activation
fix that improves on SmoothQuant — Hessian/Output-aware Rotated Equalization
(HORE).

Diagnosis (from OPT-125m / Llama-2-7B):
  * GF4 blocks are PER-TOKEN (each token's 16-channel block gets its own RMS
    scale), so the ~54% single-term error on well-behaved layers is the genuine
    4-bit-on-Gaussian floor, NOT cross-token scale sharing.
  * Mean subtraction (module.mu, the affine shift) is ALREADY on in the gf4
    baseline.
  * The residual's win is a 2nd FP4 term => effectively ~8-bit => not true W4A4.

Why SmoothQuant is the wrong tool here:
  SmoothQuant applies a per-channel diagonal scale chosen from activation
  MAGNITUDE.  A diagonal scale is only the optimal correction family when the
  weight Hessian is diagonal.  In the raw basis it is not, so SmoothQuant is
  mis-specified; and it is blind to how much each channel matters for the
  OUTPUT.

HORE (novel):
  After the Hadamard rotation the cross-channel Hessian is approximately
  diagonal (incoherence) — exactly the regime where a per-channel scale IS the
  right family.  We choose that scale in closed form to reduce the OUTPUT error
  Sum_j ||w_j||^2 E[eps_j^2], i.e. weighted by each rotated channel's weight-
  column norm, not just its activation magnitude:

      s_j  ∝  sqrt( sigma_j / ||w_had[:,j]|| )        (mean-normalized to 1)

  Folded exactly into the (already-rotated) weights:  (x/s) @ diag(s) W = x@W.
  This redistributes the 4 bits of per-block dynamic range toward the channels
  that move the output most.  Still ONE FP4 activation term, one GEMM, fully
  static — genuine W4A4.

Compares on OPT-125m / WikiText-2 (non-overlapping 2048):
    A16, gf4(1-term), gf4_adaptive(1-term), gf4+HORE(1-term),
    gf4+HORE+adaptive(1-term),  residual(2-term, reference ceiling)
"""
import os, math, gc, random
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from FP_Quantization_Experiments import (
    quantize_model_fp, enable_fast_kernels, act_quant_mode,
)
from FP_Quantization_Experiments.bit_split import _rotate

MODEL   = "facebook/opt-125m"
DEV     = "cuda"
SEQLEN  = 2048
BS      = 16            # FP4 scale block_size (matches driver)
E_B, M_B, E_S, M_S = 2, 1, 4, 3
NCALIB  = 16
HORE_P  = 0.5          # exponent: s_j ∝ (sigma_j/||w_j||)^HORE_P
random.seed(0); torch.manual_seed(0)


# ── data ──────────────────────────────────────────────────────────────────────
def tok_corpus(tok, ds):
    return tok("\n\n".join(ds["text"]), return_tensors="pt",
               add_special_tokens=False).input_ids.squeeze(0)

def ppl_eval(model, ids, seqlen=SEQLEN, bs=2):
    model.eval()
    n = ids.size(0) // seqlen
    chunks = ids[:n * seqlen].view(n, seqlen)
    nll, ntok = 0.0, 0
    with torch.inference_mode():
        for i in range(0, n, bs):
            batch = chunks[i:i + bs].to(DEV)
            logits = model(batch, use_cache=False).logits
            sl = logits[:, :-1, :].contiguous(); lb = batch[:, 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), lb.view(-1))
            if not (torch.isnan(loss) or torch.isinf(loss)):
                b = batch.size(0)
                nll += loss.item() * b * (seqlen - 1); ntok += b * (seqlen - 1)
            del logits, sl, lb, batch
    return math.exp(nll / ntok) if ntok else float("inf")


# ── NOVEL: Hessian/Output-aware Rotated Equalization (single-term) ────────────
@torch.no_grad()
def install_hore(model, calib, block_size, device, num_batches=8, p=HORE_P):
    """Collect per-(rotated)channel activation RMS, combine with weight-column
    norm, install as h_smooth_scale and pre-multiply weight_q (exact fold)."""
    # Phase A — rotated-domain per-channel second moment of the CENTERED act.
    #   sigma_j^2 = E[(x_had - mu)_j^2]   (mu already subtracted in forward)
    acc, cnt, handles = {}, {}, []
    for name, module in model.named_modules():
        if type(module).__name__ != "HadamardQuantLinearFP":
            continue
        if module.D is None or module.had_block_size is None:
            continue
        D_l, hbs = module.D, module.had_block_size
        mu = module.mu

        def mk(n, D_ref, h, mu_ref):
            def hook(mod, inp, out):
                x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                xh = _rotate(x, D_ref, h)                       # [T, P]
                if mu_ref is not None:
                    xh = xh - mu_ref.to(xh.device, xh.dtype)
                s2 = xh.pow(2).sum(dim=0)                       # [P]
                if n not in acc:
                    acc[n] = s2; cnt[n] = xh.shape[0]
                else:
                    acc[n] += s2; cnt[n] += xh.shape[0]
            return hook
        handles.append(module.register_forward_hook(mk(name, D_l, hbs, mu)))

    model.eval(); run = 0
    for batch in calib:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        model(x.to(device)); run += 1
        if run >= num_batches:
            break
    for h in handles:
        h.remove()

    # Phase B — build s_j, install, fold into weights.
    n_done = 0
    for name, module in model.named_modules():
        if type(module).__name__ != "HadamardQuantLinearFP":
            continue
        if name not in acc or module.weight_q is None:
            continue
        sigma = (acc[name] / max(cnt[name], 1)).sqrt().clamp(min=1e-8)   # [P]
        Wq    = module.weight_q.to(device).float()                       # [N, P]
        wnorm = Wq.pow(2).sum(dim=0).sqrt().clamp(min=1e-8)              # [P]

        s = (sigma / wnorm).pow(p)
        s = (s / s.mean().clamp(min=1e-8)).clamp(min=1e-3, max=1e3)      # mean≈1
        s = s.to(Wq.dtype)

        # exact fold: (x/s) @ diag(s) Wq == x @ Wq   (h_smooth divides x in fwd)
        module.weight_q = (Wq * s.unsqueeze(0)).to(module.inner.linear.weight.dtype)
        module.h_smooth_scale = s
        n_done += 1
    print(f"  HORE installed on {n_done} layers (p={p})")


# ── build ─────────────────────────────────────────────────────────────────────
def build(model, calib):
    return quantize_model_fp(
        model, calib, block_size=BS, e_bits=E_B, m_bits=M_B,
        e_bits_scale=E_S, m_bits_scale=M_S, device=DEV,
        use_HG=False, use_Hessian=False, use_adap=False, use_forward=False,
        Hadamard=True, joint=False, preshift=False, decompose=False,
        had_block_size="auto", use_gf4=True)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test  = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids   = tok_corpus(tok, train)
    test_ids = tok_corpus(tok, test)
    calib = [ids[(s := random.randint(0, ids.size(0) - SEQLEN - 1)):s + SEQLEN].unsqueeze(0)
             for _ in range(NCALIB)]

    out = {}

    # ── stock weights (clean-Hessian, mu/shift already on) ────────────────────
    print("\n========== BUILD A: stock weights (single-term baselines) ==========")
    mA = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV)
    mA = build(mA, calib); enable_fast_kernels(mA, enable=True)
    with act_quant_mode(mA, mode=None):           out["A16"]          = ppl_eval(mA, test_ids)
    with act_quant_mode(mA, mode="gf4"):          out["gf4"]          = ppl_eval(mA, test_ids)
    with act_quant_mode(mA, mode="gf4_adaptive"): out["gf4_adaptive"] = ppl_eval(mA, test_ids)
    with act_quant_mode(mA, mode="gf4_residual"): out["resid_ref"]    = ppl_eval(mA, test_ids)
    print("A:", {k: round(out[k], 3) for k in ("A16", "gf4", "gf4_adaptive", "resid_ref")})
    del mA; gc.collect(); torch.cuda.empty_cache()

    # ── HORE weights (novel single-term) ──────────────────────────────────────
    print("\n========== BUILD B: + HORE (novel single-term) ==========")
    mB = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV)
    mB = build(mB, calib)
    # NOTE: install HORE BEFORE enabling fast kernels — h_smooth_scale is honored
    # by the PyTorch path; fast Triton GF4 path quantizes whatever forward feeds.
    install_hore(mB, calib, BS, DEV)
    enable_fast_kernels(mB, enable=True)
    with act_quant_mode(mB, mode="gf4"):          out["gf4_hore"]      = ppl_eval(mB, test_ids)
    with act_quant_mode(mB, mode="gf4_adaptive"): out["gf4_hore_adap"] = ppl_eval(mB, test_ids)
    del mB; gc.collect(); torch.cuda.empty_cache()

    print("\n================= RESULTS (OPT-125m, WikiText-2 nonoverlap-2048) =================")
    order = ["A16", "gf4", "gf4_adaptive", "gf4_hore", "gf4_hore_adap", "resid_ref"]
    for k in order:
        tag = "  (2-term ref)" if k == "resid_ref" else ("  (W-only ceil)" if k == "A16" else "")
        print(f"  {k:16s} {out.get(k, float('nan')):8.3f}{tag}")
    if "gf4" in out and "gf4_hore" in out:
        gap0 = out["gf4"] - out["resid_ref"]
        gap1 = out["gf4_hore"] - out["resid_ref"]
        print(f"\n  single-term gap to residual:  gf4 {gap0:+.3f}  ->  gf4_hore {gap1:+.3f}")
        if gap0 > 0:
            print(f"  HORE closed {100*(gap0-gap1)/gap0:.0f}% of the single-term->residual gap")


if __name__ == "__main__":
    main()

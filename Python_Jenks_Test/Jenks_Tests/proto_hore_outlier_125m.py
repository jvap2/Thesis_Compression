"""
Step 2 of 2 (outlier-targeted): on top of global HORE (best p from
proto_hore_sweep_125m.py), give the OUTLIER layers extra single-term budget:
  (a) a stronger HORE exponent p_out, and
  (b) a per-layer LEARNED GF4 codebook calibrated on the POST-HORE rotated
      activations (heavy-tail aware), where clean layers stay on GF4_POS.

Why only outlier layers: clean (attn / MLP-in) layers sit at the 4-bit
Gaussian floor after Hadamard — no static per-channel scale or codebook can
beat it.  The exploitable structure lives on down_proj/fc2 and lm_head
(massive-activation channels, act_max ~1e3).  We spend the extra machinery
exactly there.  Still ONE FP4 activation term, one GEMM, fully static — genuine
W4A4.

IMPORTANT — composition correctness:
  The stock calibrate_gf4_learned_levels() does D->mu->block-norm but does NOT
  apply Step-3.5 h_smooth_scale, so it would fit the codebook to the PRE-HORE
  distribution.  We therefore calibrate levels with our own hook that mirrors
  the forward exactly: rotate -> mu -> /h_smooth_scale -> block-normalize.

Outlier selection is data-driven: per layer we score the rotated-domain
channel-variance concentration  score = sigma.max() / sigma.median()  (collected
during HORE install).  Layers above OUTLIER_Q quantile (plus any fc2/down_proj/
lm_head by name) get the treatment.
"""
import math, gc, random, copy
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from FP_Quantization_Experiments import (
    quantize_model_fp, enable_fast_kernels, act_quant_mode,
)
from FP_Quantization_Experiments.bit_split import _rotate, optimize_gf4_levels, GF4_POS

import os, sys
# CLI/env overrides so the same prototype runs the scaling ladder:
#   python3 proto_hore_outlier_125m.py [MODEL] [BEST_P] [P_OUT]
MODEL   = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("HORE_MODEL", "facebook/opt-125m")
DEV     = "cuda"
SEQLEN  = 2048
BS      = 16
E_B, M_B, E_S, M_S = 2, 1, 4, 3
NCALIB  = 16

BEST_P     = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25  # global HORE exponent (from sweep)
P_OUT      = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5   # stronger exponent on outlier layers
OUTLIER_Q  = 0.80    # layers above this sigma-concentration quantile = outlier
NAME_OUT   = ("fc2", "down_proj", "lm_head")
N_LVL_STEPS = 400
random.seed(0); torch.manual_seed(0)
print(f"[config] MODEL={MODEL}  BEST_P={BEST_P}  P_OUT={P_OUT}")


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


@torch.no_grad()
def _collect_sigma(model, calib, device, num_batches=8):
    """Rotated-domain per-channel RMS of the centered activation, per layer."""
    acc, cnt, handles = {}, {}, []
    for name, module in model.named_modules():
        if type(module).__name__ != "HadamardQuantLinearFP":
            continue
        if module.D is None or module.had_block_size is None:
            continue
        D_l, hbs, mu = module.D, module.had_block_size, module.mu

        def mk(n, D_ref, h, mu_ref):
            def hook(mod, inp, out):
                x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                xh = _rotate(x, D_ref, h)
                if mu_ref is not None:
                    xh = xh - mu_ref.to(xh.device, xh.dtype)
                s2 = xh.pow(2).sum(dim=0)
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

    sigma = {n: (acc[n] / max(cnt[n], 1)).sqrt().clamp(min=1e-8) for n in acc}
    return sigma


@torch.no_grad()
def install_hore_targeted(model, sigma, device, p_global, p_out, outlier_set):
    """Per-layer HORE: exponent p_out on outlier layers, p_global elsewhere."""
    n_done, n_out = 0, 0
    for name, module in model.named_modules():
        if type(module).__name__ != "HadamardQuantLinearFP":
            continue
        if name not in sigma or module.weight_q is None:
            continue
        p = p_out if name in outlier_set else p_global
        sig = sigma[name]
        Wq    = module.weight_q.to(device).float()
        wnorm = Wq.pow(2).sum(dim=0).sqrt().clamp(min=1e-8)
        s = (sig / wnorm).pow(p)
        s = (s / s.mean().clamp(min=1e-8)).clamp(min=1e-3, max=1e3).to(Wq.dtype)
        module.weight_q = (Wq * s.unsqueeze(0)).to(module.inner.linear.weight.dtype)
        module.h_smooth_scale = s
        n_done += 1
        n_out += int(name in outlier_set)
    print(f"  targeted HORE: {n_done} layers ({n_out} outlier @ p={p_out}, "
          f"rest @ p={p_global})")


def calib_levels_posthore(model, calib, device, names, block_size,
                          num_batches=8, n_steps=N_LVL_STEPS, max_samples=8192):
    """Learn a per-layer GF4 codebook on the POST-HORE block-normalized act.
    Mirrors forward Steps 2,3,3.5 exactly (rotate -> mu -> /h_smooth_scale).
    NOTE: not @torch.no_grad — optimize_gf4_levels needs autograd; only the
    activation-collection loop is wrapped in no_grad."""
    bank, handles = {n: [] for n in names}, []
    for name, module in model.named_modules():
        if name not in names:
            continue
        D_l, hbs, mu = module.D, module.had_block_size, module.mu
        hsm, clip = module.h_smooth_scale, \
            (module.act_clip_ratio if module.act_clip_ratio is not None else 2.5)

        def mk(n, D_ref, h, mu_ref, s_ref, c):
            def hook(mod, inp, out):
                x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                xh = _rotate(x, D_ref, h)
                if mu_ref is not None:
                    xh = xh - mu_ref.to(xh.device, xh.dtype)
                if s_ref is not None:
                    xh = xh / s_ref.to(xh.device, xh.dtype)
                K = xh.shape[1]
                pad = (block_size - K % block_size) % block_size
                xb = F.pad(xh, (0, pad)).reshape(-1, block_size)
                rms = xb.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
                xn = (xb.abs() / (rms * c)).clamp(0.0, 1.0)
                bank[n].append(xn.cpu())
            return hook
        handles.append(model.get_submodule(name)
                       .register_forward_hook(mk(name, D_l, hbs, mu, hsm, clip)))

    model.eval(); run = 0
    with torch.no_grad():
        for batch in calib:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            model(x.to(device)); run += 1
            if run >= num_batches:
                break
    for h in handles:
        h.remove()

    for name in names:
        if not bank[name]:
            continue
        samples = torch.cat(bank[name], dim=0).float()
        if samples.shape[0] > max_samples:
            samples = samples[torch.randperm(samples.shape[0])[:max_samples]]
        learned = optimize_gf4_levels(samples.to(device), n_steps=n_steps)  # needs grad
        model.get_submodule(name).gf4_levels = learned.to(device)
        print(f"  learned levels [{name}]: {[round(v,3) for v in learned.tolist()]}")


def build(model, calib):
    return quantize_model_fp(
        model, calib, block_size=BS, e_bits=E_B, m_bits=M_B,
        e_bits_scale=E_S, m_bits_scale=M_S, device=DEV,
        use_HG=False, use_Hessian=False, use_adap=False, use_forward=False,
        Hadamard=True, joint=False, preshift=False, decompose=False,
        had_block_size="auto", use_gf4=True)


def pick_outliers(sigma, q, name_out):
    scores = {n: (s.max() / s.median().clamp(min=1e-8)).item()
              for n, s in sigma.items()}
    vals = torch.tensor(list(scores.values()))
    thr = torch.quantile(vals, q).item()
    chosen = {n for n, v in scores.items()
              if v >= thr or any(k in n for k in name_out)}
    top = sorted(scores.items(), key=lambda kv: -kv[1])[:8]
    print(f"  outlier score thr (q={q}) = {thr:.1f}; {len(chosen)} layers chosen")
    print("  top-8 by sigma-concentration:")
    for n, v in top:
        print(f"    {v:8.1f}  {'<<' if n in chosen else '  '} {n}")
    return chosen


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test  = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids   = tok_corpus(tok, train)
    test_ids = tok_corpus(tok, test)
    calib = [ids[(s := random.randint(0, ids.size(0) - SEQLEN - 1)):s + SEQLEN].unsqueeze(0)
             for _ in range(NCALIB)]

    out = {}

    # ── ONE calibrated build; every arm is a deepcopy of it so the stochastic ─
    # ── v5 weight_q is IDENTICAL across arms (isolates the HORE effect from ───
    # ── calibration noise — the confound seen in proto_hore_sweep).  ──────────
    print("\n========== Calibrating base model (once) ==========")
    mBase = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV)
    mBase = build(mBase, calib)
    sigma = _collect_sigma(mBase, calib, DEV)
    outset = pick_outliers(sigma, OUTLIER_Q, NAME_OUT)

    # arm 1 — stock single-term baselines + residual reference (shared weights)
    print("\n========== ARM A: stock baselines (shared build) ==========")
    mA = copy.deepcopy(mBase); enable_fast_kernels(mA, enable=True)
    with act_quant_mode(mA, mode=None):           out["A16"]       = ppl_eval(mA, test_ids)
    with act_quant_mode(mA, mode="gf4_adaptive"): out["gf4_adap"]  = ppl_eval(mA, test_ids)
    with act_quant_mode(mA, mode="gf4_residual"): out["resid_ref"] = ppl_eval(mA, test_ids)
    print("A:", {k: round(out[k], 3) for k in ("A16", "gf4_adap", "resid_ref")})
    del mA; gc.collect(); torch.cuda.empty_cache()

    # arm L — adaptive + learned codebook on OUTLIER layers only (NO HORE).
    #   Isolates the codebook lever; the sweep showed HORE is redundant with
    #   the adaptive per-block clip search, so we test the codebook on its own.
    print("\n========== ARM L: adaptive + learned levels on outliers (no HORE) ==========")
    mL = copy.deepcopy(mBase); enable_fast_kernels(mL, enable=True)
    calib_levels_posthore(mL, calib, DEV, outset, BS)   # hsm is None here -> pre-HORE == forward
    with act_quant_mode(mL, mode="gf4_adaptive"): out["lvl_only"] = ppl_eval(mL, test_ids)
    del mL; gc.collect(); torch.cuda.empty_cache()

    # arm B — outlier-only HORE (p_out on outliers, BEST_P elsewhere) + learned
    #   codebook calibrated POST-HORE. Tests whether outlier HORE adds anything
    #   on top of the codebook.
    print("\n========== ARM B: outlier HORE (p_out={}) + post-HORE levels ==========".format(P_OUT))
    mB = copy.deepcopy(mBase)
    install_hore_targeted(mB, sigma, DEV, BEST_P, P_OUT, outset)
    enable_fast_kernels(mB, enable=True)
    with act_quant_mode(mB, mode="gf4_adaptive"): out["hore_tgt"] = ppl_eval(mB, test_ids)
    print("\n  + post-HORE learned codebook on outlier layers ...")
    calib_levels_posthore(mB, calib, DEV, outset, BS)
    with act_quant_mode(mB, mode="gf4_adaptive"): out["hore_tgt_lvl"] = ppl_eval(mB, test_ids)
    del mB, mBase; gc.collect(); torch.cuda.empty_cache()

    print(f"\n========= RESULTS ({MODEL}, WikiText-2 nonoverlap-2048) =========")
    order = ["A16", "gf4_adap", "lvl_only", "hore_tgt", "hore_tgt_lvl", "resid_ref"]
    labels = {"A16": "(W-only ceil)", "gf4_adap": "(1-term, stock adaptive)",
              "lvl_only": "(1-term, +learned lvls, no HORE)",
              "hore_tgt": "(1-term, outlier HORE)",
              "hore_tgt_lvl": "(1-term, outlier HORE +learned lvls)",
              "resid_ref": "(2-term ref)"}
    for k in order:
        print(f"  {k:14s} {out.get(k, float('nan')):8.3f}  {labels[k]}")
    g0 = out["gf4_adap"] - out["resid_ref"]
    for k in ("lvl_only", "hore_tgt_lvl"):
        gk = out[k] - out["resid_ref"]
        if g0 > 0:
            print(f"  gap to residual: gf4_adap {g0:+.3f} -> {k} {gk:+.3f} "
                  f"({100*(g0-gk)/g0:.0f}% closed)")


if __name__ == "__main__":
    main()

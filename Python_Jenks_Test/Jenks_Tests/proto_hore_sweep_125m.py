"""
Step 1 of 2 (P-sweep): find the best HORE exponent p before adding the
outlier-targeted learned-levels machinery (proto_hore_outlier_125m.py).

HORE scale:  s_j ∝ (sigma_j / ||w_had[:,j]||)^p   (mean-normalized to 1)
  p=0   -> s≡1 (no equalization, == plain gf4)
  p=0.5 -> sqrt redistribution (current default)
  p=1.0 -> full output-variance equalization

Single-GEMM, static, folded exactly into the rotated weights. Strictly 1-term.
Baselines (A16, gf4, gf4_adaptive, resid_ref) are computed once on the stock
build; each p gets a FRESH build because install_hore mutates weight_q in place.
"""
import math, gc, random
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from FP_Quantization_Experiments import (
    quantize_model_fp, enable_fast_kernels, act_quant_mode,
)
from FP_Quantization_Experiments.bit_split import _rotate

MODEL   = "facebook/opt-125m"
DEV     = "cuda"
SEQLEN  = 2048
BS      = 16
E_B, M_B, E_S, M_S = 2, 1, 4, 3
NCALIB  = 16
P_GRID  = [0.25, 0.5, 0.75, 1.0]
random.seed(0); torch.manual_seed(0)


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
def install_hore(model, calib, device, num_batches=8, p=0.5):
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

    n_done = 0
    for name, module in model.named_modules():
        if type(module).__name__ != "HadamardQuantLinearFP":
            continue
        if name not in acc or module.weight_q is None:
            continue
        sigma = (acc[name] / max(cnt[name], 1)).sqrt().clamp(min=1e-8)
        Wq    = module.weight_q.to(device).float()
        wnorm = Wq.pow(2).sum(dim=0).sqrt().clamp(min=1e-8)
        s = (sigma / wnorm).pow(p)
        s = (s / s.mean().clamp(min=1e-8)).clamp(min=1e-3, max=1e3)
        s = s.to(Wq.dtype)
        module.weight_q = (Wq * s.unsqueeze(0)).to(module.inner.linear.weight.dtype)
        module.h_smooth_scale = s
        n_done += 1
    print(f"  HORE installed on {n_done} layers (p={p})")


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

    print("\n========== BUILD A: stock single-term baselines ==========")
    mA = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV)
    mA = build(mA, calib); enable_fast_kernels(mA, enable=True)
    with act_quant_mode(mA, mode=None):           out["A16"]          = ppl_eval(mA, test_ids)
    with act_quant_mode(mA, mode="gf4"):          out["gf4"]          = ppl_eval(mA, test_ids)
    with act_quant_mode(mA, mode="gf4_adaptive"): out["gf4_adaptive"] = ppl_eval(mA, test_ids)
    with act_quant_mode(mA, mode="gf4_residual"): out["resid_ref"]    = ppl_eval(mA, test_ids)
    print("A:", {k: round(out[k], 3) for k in ("A16", "gf4", "gf4_adaptive", "resid_ref")})
    del mA; gc.collect(); torch.cuda.empty_cache()

    for p in P_GRID:
        print(f"\n========== BUILD: + HORE (p={p}) ==========")
        mB = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV)
        mB = build(mB, calib)
        install_hore(mB, calib, DEV, p=p)
        enable_fast_kernels(mB, enable=True)
        with act_quant_mode(mB, mode="gf4"):          out[f"hore_p{p}"]      = ppl_eval(mB, test_ids)
        with act_quant_mode(mB, mode="gf4_adaptive"): out[f"hore_adap_p{p}"] = ppl_eval(mB, test_ids)
        print(f"  p={p}: hore {out[f'hore_p{p}']:.3f}  hore_adap {out[f'hore_adap_p{p}']:.3f}")
        del mB; gc.collect(); torch.cuda.empty_cache()

    print("\n================= P-SWEEP RESULTS (OPT-125m, WikiText-2 nonoverlap-2048) =================")
    print(f"  {'A16':16s} {out['A16']:8.3f}  (W-only ceil)")
    print(f"  {'gf4':16s} {out['gf4']:8.3f}")
    print(f"  {'gf4_adaptive':16s} {out['gf4_adaptive']:8.3f}")
    print(f"  {'resid_ref':16s} {out['resid_ref']:8.3f}  (2-term ref)")
    print("  ----- single-term HORE sweep -----")
    best_p, best_v = None, float("inf")
    for p in P_GRID:
        v, va = out[f"hore_p{p}"], out[f"hore_adap_p{p}"]
        print(f"  p={p:<5}  hore {v:8.3f}   hore_adap {va:8.3f}")
        if va < best_v:
            best_v, best_p = va, p
    print(f"\n  best single-term: p={best_p}  hore_adap {best_v:.3f}  "
          f"(gap to residual {best_v - out['resid_ref']:+.3f})")


if __name__ == "__main__":
    main()

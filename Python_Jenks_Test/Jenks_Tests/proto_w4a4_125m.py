"""
Prototype: can a SINGLE GF4 activation term (genuine single-GEMM W4A4) match the
2-term residual, if we re-solve the weights against the JOINT Hessian?

Baseline weights are already quantized against the clean Hadamard-domain Hessian
H_clean = E[x_had x_had^T]  (reconstruct_layer_fp_blockdiag_scaled_v5).
The residual's only advantage is a 2nd FP4 activation term — which is per-token
and cannot be folded into static weights.

This prototype instead re-solves each layer's weights against the JOINT Hessian
    H_joint = E[ x_hat_had  x_hat_had^T ],   x_hat = GF4(x_had)   (single term)
so the weights compensate the single-term activation-quant error IN EXPECTATION
— the residual's job, absorbed statically into the weights. Still one A4 GEMM.

Compares, on OPT-125m / WikiText-2 (non-overlapping 2048):
    A16, gf4(1-term), residual(2-term)              [stock weights]
    gf4(1-term), residual(2-term)                   [joint-Hessian weights]
"""
import os, math, gc, random
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from FP_Quantization_Experiments import (
    quantize_model_fp, enable_fast_kernels, act_quant_mode,
)
from FP_Quantization_Experiments.bit_split import (
    _rotate, quantize_activations_gf4, compute_hessian_blocks,
    reconstruct_layer_fp_blockdiag_scaled_v5,
)

MODEL   = "facebook/opt-125m"
DEV     = "cuda"
SEQLEN  = 2048
BS      = 16            # FP4 scale block_size (matches driver)
E_B, M_B, E_S, M_S = 2, 1, 4, 3
NCALIB  = 16
random.seed(0); torch.manual_seed(0)


# ── data ────────────────────────────────────────────────────────────────────
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
            if torch.isnan(loss) or torch.isinf(loss):
                del logits, sl, lb, batch; continue
            b = batch.size(0)
            nll += loss.item() * b * (seqlen - 1); ntok += b * (seqlen - 1)
            del logits, sl, lb, batch
    return math.exp(nll / ntok) if ntok else float("inf")


# ── the new bit: re-solve weights against the joint (single-GF4) Hessian ──────
@torch.no_grad()
def recalibrate_weights_joint_gf4(model, calib, block_size, device,
                                  e_bits, m_bits, e_bits_scale, m_bits_scale,
                                  num_batches=8):
    # Phase A — joint Hessian H = E[x_hat_had x_hat_had^T], x_hat = GF4(x_had)
    H_data, handles = {}, []
    for name, module in model.named_modules():
        if type(module).__name__ != "HadamardQuantLinearFP":
            continue
        if module.D is None or module.had_block_size is None:
            continue
        D_l, hbs = module.D, module.had_block_size
        inner = module.inner.linear
        clip = module.act_clip_ratio if getattr(module, "act_clip_ratio", None) else 2.5

        def mk(n, D_ref, h, inner_mod, clip_r):
            def hook(mod, inp, out):
                x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                x_had = _rotate(x, D_ref, h)
                x_hat = quantize_activations_gf4(x_had, block_size, clip_ratio=clip_r)
                Hb = compute_hessian_blocks(x_hat, inner_mod, block_size)
                if Hb is None:
                    return
                if n not in H_data:
                    H_data[n] = Hb
                else:
                    for i in range(len(Hb)):
                        H_data[n][i] += Hb[i]
            return hook
        handles.append(module.register_forward_hook(mk(name, D_l, hbs, inner, clip)))

    model.eval()
    run = 0
    for batch in calib:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        model(x.to(device)); run += 1
        if run >= num_batches:
            break
    for h in handles:
        h.remove()
    H_joint = {n: [b.float() / run for b in blks] for n, blks in H_data.items()}
    print(f"  joint Hessian collected for {len(H_joint)} layers over {run} batches")

    # Phase B — re-solve each layer's weights with H_joint, install, fix bias
    for name, module in model.named_modules():
        if type(module).__name__ != "HadamardQuantLinearFP":
            continue
        if name not in H_joint or module.weight_q is None:
            continue
        D, hbs = module.D, module.had_block_size
        W_mat = module.inner.linear.weight.data.to(device).float()
        W_had = _rotate(W_mat, D, hbs)                         # [N, P]
        old_wq = module.weight_q.to(device).float()            # [N, P]

        tmp = nn.Linear(W_had.shape[1], W_had.shape[0], bias=False, device=device)
        tmp.weight.data = W_had
        res = reconstruct_layer_fp_blockdiag_scaled_v5(
            tmp, H_joint[name], block_size,
            e_bits, m_bits, e_bits_scale, m_bits_scale, device)
        new_wq = res["weight_q"].reshape(W_had.shape).to(device).float()

        # bias compensation: bias_corr = W_had_q @ mu  → update by the delta
        if getattr(module, "mu", None) is not None:
            mu = module.mu.to(device).float()
            delta = (new_wq - old_wq) @ mu                     # [N]
            if getattr(module, "bias_correction", None) is not None:
                module.bias_correction = (
                    module.bias_correction.to(device).float() + delta
                ).to(module.weight_q.dtype)
            elif module.inner.linear.bias is not None:
                module.inner.linear.bias.data = (
                    module.inner.linear.bias.data.float() + delta
                ).to(module.inner.linear.bias.dtype)

        module.weight_q = new_wq.to(module.inner.linear.weight.dtype)
        del tmp, res, W_had, old_wq, new_wq
        torch.cuda.empty_cache()


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

    # ── stock weights (clean-Hessian) ────────────────────────────────────────
    print("\n========== BUILD A: stock (clean-Hessian) weights ==========")
    mA = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV)
    mA = build(mA, calib); enable_fast_kernels(mA, enable=True)
    with act_quant_mode(mA, mode=None):          out["A16"]          = ppl_eval(mA, test_ids)
    with act_quant_mode(mA, mode="gf4"):         out["gf4_clean"]    = ppl_eval(mA, test_ids)
    with act_quant_mode(mA, mode="gf4_residual"):out["resid_clean"]  = ppl_eval(mA, test_ids)
    print("A:", {k: out[k] for k in ("A16", "gf4_clean", "resid_clean")})
    del mA; gc.collect(); torch.cuda.empty_cache()

    # ── joint-Hessian weights ────────────────────────────────────────────────
    print("\n========== BUILD B: joint-Hessian (single-GF4-aware) weights ==========")
    mB = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV)
    mB = build(mB, calib)
    print("  re-solving weights against joint single-GF4 Hessian...")
    recalibrate_weights_joint_gf4(mB, calib, BS, DEV, E_B, M_B, E_S, M_S)
    enable_fast_kernels(mB, enable=True)
    with act_quant_mode(mB, mode="gf4"):         out["gf4_joint"]    = ppl_eval(mB, test_ids)
    with act_quant_mode(mB, mode="gf4_residual"):out["resid_joint"]  = ppl_eval(mB, test_ids)
    del mB; gc.collect(); torch.cuda.empty_cache()

    print("\n================= RESULTS (OPT-125m, WikiText-2 nonoverlap-2048) =================")
    for k in ("A16", "gf4_clean", "resid_clean", "gf4_joint", "resid_joint"):
        print(f"  {k:14s} {out.get(k, float('nan')):8.3f}")
    print("\n  KEY: does gf4_joint (single-term, genuine W4A4) close the gap to resid_clean?")
    if "gf4_joint" in out and "gf4_clean" in out:
        print(f"  gf4: clean {out['gf4_clean']:.3f} -> joint {out['gf4_joint']:.3f}")
        print(f"  residual reference (clean): {out['resid_clean']:.3f}")


if __name__ == "__main__":
    main()

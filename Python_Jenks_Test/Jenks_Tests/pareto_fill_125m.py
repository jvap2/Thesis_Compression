"""
Fill the remaining hw_sim_gf4.py Pareto rows with MEASURED PPL on OPT-125m
(WikiText-2 nonoverlap-2048): FP16, FP8 (W8A8), MXFP4 (per-32 E2M1).

NVFP4 / GF4 / residual / A16 already measured in iso_energy_125m.py.  Kept on the
SAME eval harness (seqlen 2048 nonoverlap, bs=2) so numbers are comparable.
Each measurement is independent + guarded so one failure doesn't drop the rest.
"""
import math, gc, random
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from FP_Quantization_Experiments import quantize_model_fp, enable_fast_kernels, act_quant_mode

MODEL  = "facebook/opt-125m"
DEV    = "cuda"
SEQLEN = 2048
BS     = 16
NCALIB = 16
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


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test  = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids   = tok_corpus(tok, train)
    test_ids = tok_corpus(tok, test)
    calib = [ids[(s := random.randint(0, ids.size(0) - SEQLEN - 1)):s + SEQLEN].unsqueeze(0)
             for _ in range(NCALIB)]
    out = {}

    # ── FP16 (W16A16, no quant) — the 1.0x Pareto anchor ──────────────────────
    try:
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV)
        out["FP16"] = ppl_eval(m, test_ids)
        print(f"FP16 (W16A16):       {out['FP16']:.3f}")
        del m; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print("FP16 FAILED:", e)

    # ── MXFP4 (per-32 E2M1) on a GF4 build: inject E2M1 levels + block_size 32 ─
    #   vs NVFP4 (per-16) this isolates the microscale-block-size penalty.
    try:
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV)
        m = quantize_model_fp(m, calib, block_size=BS, e_bits=2, m_bits=1,
                              e_bits_scale=4, m_bits_scale=3, device=DEV,
                              use_HG=False, use_Hessian=False, use_adap=False,
                              use_forward=False, Hadamard=True, joint=False,
                              preshift=False, decompose=False, had_block_size="auto",
                              use_gf4=True)
        enable_fast_kernels(m, enable=True)
        E2M1 = (torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=DEV) / 6.0)
        for mod in m.modules():
            if type(mod).__name__ == "HadamardQuantLinearFP" and mod.weight_q is not None:
                mod.gf4_levels = E2M1
                mod.act_block_size = 32          # MXFP4 per-32 microscale block
        with act_quant_mode(m, mode="gf4_adaptive"):
            out["MXFP4"] = ppl_eval(m, test_ids)
        print(f"MXFP4 (E2M1,per-32): {out['MXFP4']:.3f}")
        del m; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print("MXFP4 FAILED:", e)

    # ── FP8 (W8A8, E4M3) — separate build, no Hadamard/GF4 ────────────────────
    try:
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV)
        m = quantize_model_fp(m, calib, block_size=BS, e_bits=4, m_bits=3,
                              e_bits_scale=4, m_bits_scale=3, device=DEV,
                              use_HG=False, use_Hessian=False, use_adap=False,
                              use_forward=False, Hadamard=False, joint=False,
                              preshift=False, decompose=False, had_block_size=None,
                              use_gf4=False)
        with act_quant_mode(m, mode="nvfp4"):    # E4M3 act path (e_bits=4,m_bits=3)
            out["FP8-E4M3"] = ppl_eval(m, test_ids)
        print(f"FP8-E4M3 (W8A8):     {out['FP8-E4M3']:.3f}")
        del m; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print("FP8 FAILED:", repr(e))

    print("\n===== PARETO-FILL RESULTS (OPT-125m, WikiText-2 nonoverlap-2048) =====")
    for k in ("FP16", "FP8-E4M3", "MXFP4"):
        v = out.get(k)
        print(f"  {k:12s} {('%.3f'%v) if v is not None else 'FAILED'}")
    print("\n  paste into hw_sim_gf4.py PPL['opt-125m']:")
    for k in ("FP16", "FP8-E4M3", "MXFP4"):
        if out.get(k) is not None:
            print(f'        "{k}": {out[k]:.2f},')


if __name__ == "__main__":
    main()

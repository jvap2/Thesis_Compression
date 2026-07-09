"""
Iso-energy accuracy baselines for the hw_sim_gf4.py Pareto.

GF4 1-term has IDENTICAL modeled energy/BPV to NVFP4 (both 4.5 bpv, 0.08x FP16),
so the whole pitch is "same pJ, better PPL".  This fills the missing PPL points.

Fair iso-EVERYTHING comparison: ONE Hadamard build; every arm shares the same
rotation, the same per-16 block, and the same E4M3 scale precision.  The ONLY
difference is the activation codebook:
    "e2m1"  -> HadamardQuantLinearFP else-branch = standard E2M1 ladder
               {0,.5,1,1.5,2,3,4,6} + E4M3 per-16 scale  == NVFP4 (+Hadamard).
    "gf4"   -> our NF4-style Gaussian-quantile LUT, fixed clip.
    "gf4_adaptive" -> our LUT + per-block clip search.
    "gf4_residual" -> two FP4 terms (8-bit effective, 2x MACs).
Because E2M1 here ALSO gets the Hadamard, it is a STRONG/generous NVFP4 baseline
(plain NVFP4 has no rotation and would be worse) — any GF4 win is purely the
Gaussian codebook.
"""
import os
# Reduce CUDA fragmentation -> avoids edge-of-memory OOMs on 24GB GPUs (L4) for
# 6.7B+ models. Must be set before torch initializes CUDA. PyTorch 2.9 renamed
# PYTORCH_CUDA_ALLOC_CONF -> PYTORCH_ALLOC_CONF; set both for old+new torch.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys, math, gc, random
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from FP_Quantization_Experiments import quantize_model_fp, enable_fast_kernels, act_quant_mode
from FP_Quantization_Experiments.bit_split import evaluate_ppl_offload

MODEL  = sys.argv[1] if len(sys.argv) > 1 else "facebook/opt-125m"
DEV    = "cuda"
SEQLEN = 2048
# Calibration sequence length: shorter than eval to bound peak activation memory
# on memory-constrained GPUs (the calibration forward, not the model, is the
# OOM peak for 6.7B on 24GB). Env override: CALIB_SEQLEN. 1024 keeps 16k calib
# tokens, ample for the Hessian/scale statistics.
CALIB_SEQLEN = int(os.environ.get("CALIB_SEQLEN", SEQLEN))
BS     = 16
E_B, M_B, E_S, M_S = 2, 1, 4, 3
NCALIB = int(os.environ.get("NCALIB", 16))   # calibration samples; lower = faster (env override)
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


def _mlp_skip(model_name):
    """Keep the MLP-output (massive-activation) layer in FP16 — quantizing it
    collapses single-term W4A4 (OPT-6.7b). Per-architecture name:
      Llama/Mistral SwiGLU -> down_proj ; OPT/GPT -> fc2 ; Pythia/NeoX -> dense_4h_to_h."""
    mn = model_name.lower()
    if "pythia" in mn or "neox" in mn:   return ("dense_4h_to_h",)
    if "llama" in mn or "mistral" in mn: return ("down_proj",)
    if "opt" in mn or "gpt" in mn:       return ("fc2",)
    return ("down_proj", "fc2", "dense_4h_to_h")   # safe default: all variants


# RETAIN=0 disables outlier-layer retention (the without-retention ablation).
RETAIN = os.environ.get("RETAIN", "1") != "0"

# Hadamard block size: "auto" (full-row padded Hadamard — best decorrelation, the
# default that carries the main results) or an integer (e.g. 16, 32, 64) for the
# hardware-local block-diagonal Hadamard. Smaller blocks are cheaper/local in
# silicon but decorrelate less; this override lets us probe whether the GF4
# codebook buys back accuracy at a hardware-friendly block size. Env: HAD_BS.
_hb    = os.environ.get("HAD_BS", "auto")
HAD_BS = _hb if _hb == "auto" else int(_hb)

# LEAN=1 frees each layer's original weight during weight-quant so the GPU never
# holds originals + weight_q together (~2x model). Halves the resident weight-quant
# peak so opt-6.7b / llama-2-7b fit a 24GB L4 (no A100 needed). Numerically
# identical to LEAN=0; just skips an orig-vs-quant sanity print.
LEAN = os.environ.get("LEAN", "0") != "0"

# OFFLOAD=1 quantizes the model one decoder block at a time (GPTQ/QuaRot style):
# the model lives in CPU RAM and only one block sits on the GPU during weight-quant,
# so 13B/30B/70B fit a 24GB card given enough system RAM. NOT byte-identical to the
# in-GPU path (uses sequential error feedback) — validate parity on opt-125m/1.3b.
# The model is loaded on CPU when OFFLOAD=1; for eval it is moved back to the GPU
# (fine for models that fit; big-model eval offload is a separate step).
OFFLOAD = os.environ.get("OFFLOAD", "0") != "0"


def build(model, calib):
    skip = _mlp_skip(MODEL) if RETAIN else ()
    return quantize_model_fp(
        model, calib, block_size=BS, e_bits=E_B, m_bits=M_B,
        e_bits_scale=E_S, m_bits_scale=M_S, device=DEV,
        use_HG=False, use_Hessian=False, use_adap=False, use_forward=False,
        Hadamard=True, joint=False, preshift=False, decompose=False,
        had_block_size=HAD_BS, use_gf4=True, extra_skip_patterns=skip,
        lean=LEAN, offload=OFFLOAD)


def main():
    skip = _mlp_skip(MODEL) if RETAIN else ()
    print(f"[iso-energy] MODEL={MODEL}  RETAIN={RETAIN}  HAD_BS={HAD_BS}  "
          f"LEAN={LEAN}  OFFLOAD={OFFLOAD}  (skip {skip} -> FP16)")
    tok = AutoTokenizer.from_pretrained(MODEL)
    train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    test  = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids   = tok_corpus(tok, train)
    test_ids = tok_corpus(tok, test)
    calib = [ids[(s := random.randint(0, ids.size(0) - CALIB_SEQLEN - 1)):s + CALIB_SEQLEN].unsqueeze(0)
             for _ in range(NCALIB)]

    # OFFLOAD keeps the model on CPU during calibration (blocks stream to the GPU
    # one at a time); otherwise the whole model goes to the GPU up front.
    # low_cpu_mem_usage streams checkpoint shards into the model instead of
    # building a full second copy in RAM — keeps the load peak at ~1x model size
    # (critical for offloading 13B+ on a 32GB box).
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    if not OFFLOAD:
        m = m.to(DEV)
    m = build(m, calib)
    USE_EVAL_OFFLOAD = False
    if OFFLOAD:
        # Decide whether the calibrated model fits the GPU for eval. Fake-quant
        # weights are ~2 bytes/param, so ~13B is the ceiling on 24GB. If it won't
        # fit (or EVAL_OFFLOAD=1), evaluate with block-major streaming (model stays
        # on CPU, one block on the GPU at a time). EVAL_OFFLOAD=0 forces in-GPU.
        mbytes = sum(t.numel() * t.element_size()
                     for t in list(m.parameters()) + list(m.buffers()))
        _free, _total = torch.cuda.mem_get_info()
        _ev = os.environ.get("EVAL_OFFLOAD", "auto")
        USE_EVAL_OFFLOAD = (_ev == "1") or (_ev == "auto" and mbytes > 0.70 * _total)
        print(f"[iso-energy] eval: model={mbytes/2**30:.1f}GiB  gpu={_total/2**30:.1f}GiB"
              f"  -> {'STREAMING offload' if USE_EVAL_OFFLOAD else 'in-GPU'}")
        if not USE_EVAL_OFFLOAD:
            m = m.to(DEV)          # fits — move whole model to GPU for eval
    enable_fast_kernels(m, enable=True)
    gc.collect(); torch.cuda.empty_cache()   # free calibration scratch before eval

    def EVAL():
        """Perplexity on the WikiText-2 test set under the currently-set act_quant
        mode — streaming (model on CPU) when it won't fit the GPU, else in-GPU."""
        return (evaluate_ppl_offload(m, test_ids, DEV, SEQLEN)
                if USE_EVAL_OFFLOAD else ppl_eval(m, test_ids))

    # E2M1 ladder normalized to [0,1] (positive half) — the FIXED Blackwell FP4
    # codebook.  Injected as gf4_levels so it runs through the IDENTICAL GF4
    # machinery (rms*clip scale, per-16 block, Hadamard); only level positions
    # differ from our Gaussian-quantile GF4_POS.
    E2M1_POS = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=DEV) / 6.0

    def set_levels(model, levels):
        n = 0
        for mod in model.modules():
            if type(mod).__name__ == "HadamardQuantLinearFP" and mod.weight_q is not None:
                mod.gf4_levels = levels.to(DEV) if levels is not None else None
                n += 1
        return n

    out = {}
    with act_quant_mode(m, mode=None):           out["A16 (W4A16)"]      = EVAL()

    def set_actblock(model, b):
        for mod in model.modules():
            if type(mod).__name__ == "HadamardQuantLinearFP" and mod.weight_q is not None:
                mod.act_block_size = b

    # --- E2M1/NVFP4 codebook (iso-cost): inject E2M1 levels, use GF4 machinery
    set_levels(m, E2M1_POS)
    with act_quant_mode(m, mode="gf4"):          out["E2M1/NVFP4 fixed"] = EVAL()
    with act_quant_mode(m, mode="gf4_adaptive"): out["E2M1 + adaptive"]  = EVAL()
    set_actblock(m, 32)   # MXFP4 = E2M1 + per-32 microscale block (same build)
    with act_quant_mode(m, mode="gf4_adaptive"): out["MXFP4 (per-32)"]   = EVAL()
    set_actblock(m, BS); set_levels(m, None)   # restore per-16 + Gaussian GF4_POS

    # --- our Gaussian GF4 codebook
    with act_quant_mode(m, mode="gf4"):          out["GF4 1-term"]        = EVAL()
    with act_quant_mode(m, mode="gf4_adaptive"): out["GF4 adaptive"]      = EVAL()
    with act_quant_mode(m, mode="gf4_residual"): out["residual-GF4 (2t)"] = EVAL()

    # --- Learned GF4 levels (LEARNED=1; off by default — adds calibration memory
    #     and was ~null vs GF4_POS on tested models). Calibrated per-layer, then
    #     evaluated through the standard gf4 path.
    out["GF4 learned"] = float("nan")
    if os.environ.get("LEARNED", "0") == "1":
        try:
            from FP_Quantization_Experiments.bit_split import calibrate_gf4_learned_levels
            set_levels(m, None)
            calibrate_gf4_learned_levels(m, calib, DEV, BS, num_batches=4)
            with act_quant_mode(m, mode="gf4"): out["GF4 learned"] = EVAL()
            set_levels(m, None)   # clear learned per-layer levels
            gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [learned] skipped: {e}")

    print(f"\n===== ISO-ENERGY PPL ({MODEL}, WikiText-2 nonoverlap-2048) =====")
    print("  (all arms: Hadamard + rms*clip scale + per-16 block; only codebook differs)")
    order = ["A16 (W4A16)", "E2M1/NVFP4 fixed", "E2M1 + adaptive", "MXFP4 (per-32)",
             "GF4 1-term", "GF4 adaptive", "residual-GF4 (2t)"]
    for k in order:
        print(f"  {k:22s} {out[k]:8.3f}")
    print("\n  --- codebook A/B at IDENTICAL hardware cost (4.5 bpv, 0.08x FP16 energy) ---")
    print(f"  fixed clip:   E2M1 {out['E2M1/NVFP4 fixed']:.3f}  vs  GF4 {out['GF4 1-term']:.3f}"
          f"   -> GF4 {out['E2M1/NVFP4 fixed']-out['GF4 1-term']:+.3f}")
    print(f"  adaptive:     E2M1 {out['E2M1 + adaptive']:.3f}  vs  GF4 {out['GF4 adaptive']:.3f}"
          f"   -> GF4 {out['E2M1 + adaptive']-out['GF4 adaptive']:+.3f}")

    # --- structured results row (one CSV accumulating every model/retention run,
    #     so the paper table is assembled from a single consistent protocol) -----
    import csv as _csv, datetime as _dt
    res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iso_results.csv")
    cols = ["timestamp", "model", "retain", "calib_seqlen", "had_bs", "offload",
            "A16", "NVFP4_fixed", "NVFP4_adap", "MXFP4",
            "GF4", "GF4_adap", "GF4_learned", "residual"]
    row = {"timestamp": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "model": MODEL, "retain": int(RETAIN), "calib_seqlen": CALIB_SEQLEN,
           "had_bs": HAD_BS, "offload": int(OFFLOAD),
           "A16": out["A16 (W4A16)"], "NVFP4_fixed": out["E2M1/NVFP4 fixed"],
           "NVFP4_adap": out["E2M1 + adaptive"], "MXFP4": out["MXFP4 (per-32)"],
           "GF4": out["GF4 1-term"], "GF4_adap": out["GF4 adaptive"],
           "GF4_learned": out["GF4 learned"], "residual": out["residual-GF4 (2t)"]}
    _new = not os.path.exists(res_path)
    # Migrate pre-had_bs CSVs: insert the column (existing rows = "auto") so the
    # header matches and older full-row runs stay labelled correctly.
    if not _new:
        with open(res_path, newline="") as f:
            old = list(_csv.reader(f))
        if old and "had_bs" not in old[0]:
            i = old[0].index("calib_seqlen") + 1 if "calib_seqlen" in old[0] else len(old[0])
            mig = [old[0][:i] + ["had_bs"] + old[0][i:]]
            mig += [r[:i] + ["auto"] + r[i:] for r in old[1:]]
            with open(res_path, "w", newline="") as f:
                _csv.writer(f).writerows(mig)
        # Migrate pre-offload CSVs: insert the column (existing rows = "0" = in-GPU).
        with open(res_path, newline="") as f:
            old = list(_csv.reader(f))
        if old and "offload" not in old[0]:
            i = old[0].index("had_bs") + 1 if "had_bs" in old[0] else len(old[0])
            mig = [old[0][:i] + ["offload"] + old[0][i:]]
            mig += [r[:i] + ["0"] + r[i:] for r in old[1:]]
            with open(res_path, "w", newline="") as f:
                _csv.writer(f).writerows(mig)
    with open(res_path, "a", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        if _new: w.writeheader()
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"[results] appended to {res_path}")

    # --- WIRE-BACK: export actual quantized shapes + PPL for the HW sim ---------
    import json as _json
    sc = {}
    for mod in m.modules():
        if type(mod).__name__ == "HadamardQuantLinearFP" and mod.weight_q is not None:
            N, K = (int(x) for x in mod.weight_q.shape)
            sc[(K, N)] = sc.get((K, N), 0) + 1
    cfg = m.config
    export = {"model": MODEL, "seqlen": SEQLEN, "block_size": BS,
              "skipped_fp16": list(_mlp_skip(MODEL)),
              "config": {"hidden_size": getattr(cfg, "hidden_size", None),
                         "intermediate_size": getattr(cfg, "intermediate_size",
                                                      getattr(cfg, "ffn_dim", None)),
                         "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
                         "vocab_size": getattr(cfg, "vocab_size", None)},
              "ppl": {"A16 (W4A16)": out["A16 (W4A16)"], "E2M1/NVFP4": out["E2M1 + adaptive"],
                      "GF4 adaptive": out["GF4 adaptive"], "residual-GF4": out["residual-GF4 (2t)"]},
              "shapes": [{"K": k, "N": n, "count": c} for (k, n), c in sorted(sc.items())]}
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "timeloop_gf4", "models",
                      MODEL.split("/")[-1] + "_hwsim.json")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, "w") as f:
        _json.dump(export, f, indent=2)
    print(f"\n[hwsim] wrote {fn}")


if __name__ == "__main__":
    main()

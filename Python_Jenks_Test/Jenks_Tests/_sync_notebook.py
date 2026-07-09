"""
Sync Colab/FP_Quant.ipynb to the current core .py files and add a large-model
iso-energy experiment cell.

  cell 3  <- FP_Quantization_Experiments/bit_split.py   (relative imports commented)
  cell 4  <- FP_QuantNetworkTest_LLM.py                 (package imports commented)
  cell 5  <- NEW: iso-energy GF4-vs-NVFP4 + residual Pareto, parameterized by MODEL

The Colab adaptation is: the code is INLINED across cells, so every
`from .x import ...` / `from FP_Quantization_Experiments import ...` must be
commented (the names already exist globally from earlier cells).
"""
import json, re, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "Colab", "FP_Quant.ipynb")


def comment_pkg_imports(src: str) -> str:
    """Comment relative/package imports (single-line and multi-line blocks)."""
    out, in_block = [], False
    pat = re.compile(r"^(\s*)(from\s+\.|from\s+FP_Quantization_Experiments\b|import\s+FP_Quantization_Experiments\b)")
    for line in src.split("\n"):
        if in_block:
            out.append("# " + line if not line.lstrip().startswith("#") else line)
            if ")" in line:
                in_block = False
            continue
        if pat.match(line):
            out.append("# " + line if not line.lstrip().startswith("#") else line)
            # multi-line import block?  open paren without close on same line
            if "(" in line and ")" not in line:
                in_block = True
        else:
            out.append(line)
    return "\n".join(out)


def as_source(text: str):
    """Notebook 'source' = list of lines each ending in \\n (except maybe last)."""
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


EXPERIMENT_CELL = r'''# ============================================================================
# ISO-ENERGY CODEBOOK A/B + W4A4 PARETO   (large-model port of iso_energy_125m.py)
# Runs on the code inlined in the cells above. Set MODEL to any HF causal LM.
#   A16          : 4-bit weights, fp16 acts        (weight-only ceiling)
#   E2M1/NVFP4   : E2M1 ladder + adaptive clip      (Blackwell-native FP4 codebook)
#   GF4 adaptive : Gaussian-quantile + adaptive clip (OURS, true 1-term W4A4)
#   residual-GF4 : two FP4 terms (~8-bit effective)  (OURS, 2-term)
# All arms share ONE Hadamard build (identical weights) -> only the activation
# codebook/terms differ -> isolates the codebook at IDENTICAL hardware cost.
# Finding on OPT-125m/1.3b: GF4-adaptive beats E2M1/NVFP4 by ~1 ppl iso-cost;
# residual-GF4 ~= A16.  This cell re-runs that A/B on a larger network.
# ============================================================================
import math, gc, random
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL   = "meta-llama/Llama-2-7b-hf"   # <-- set your model (e.g. facebook/opt-2.7b)
SEQLEN  = 2048
BS      = 16                            # FP4 scale block
E_B, M_B, E_S, M_S = 2, 1, 4, 3         # E2M1 weights, E4M3 scale
NCALIB  = 16
EVAL_BS = 1                             # lower for big models to fit memory
DEV     = "cuda"
random.seed(0); torch.manual_seed(0)

def _tok_corpus(tok, ds):
    return tok("\n\n".join(ds["text"]), return_tensors="pt",
               add_special_tokens=False).input_ids.squeeze(0)

def ppl_eval(model, ids, seqlen=SEQLEN, bs=EVAL_BS):
    model.eval(); n = ids.size(0) // seqlen
    chunks = ids[:n*seqlen].view(n, seqlen); nll = 0.0; ntok = 0
    with torch.inference_mode():
        for i in range(0, n, bs):
            batch = chunks[i:i+bs].to(DEV)
            logits = model(batch, use_cache=False).logits
            sl = logits[:, :-1, :].contiguous(); lb = batch[:, 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), lb.view(-1))
            if not (torch.isnan(loss) or torch.isinf(loss)):
                b = batch.size(0); nll += loss.item()*b*(seqlen-1); ntok += b*(seqlen-1)
            del logits, sl, lb, batch
    return math.exp(nll/ntok) if ntok else float("inf")

# ── Get a quantized model `m` WITHOUT re-quantizing when possible ────────────
# Priority:
#   1) REUSE the DRIVER cell's model (it builds a global `quant_model`) — this
#      saves the most compute: run the driver cell, then this cell reuses it.
#   2) REUSE a build this cell already made (re-running this cell is free).
#   3) Otherwise build fresh.
# To save compute, run EITHER the driver cell OR this one (not both).
# To FORCE a fresh build: `del quant_model, m` (or change MODEL) before running.
_build_key = (MODEL, BS, E_B, M_B, E_S, M_S)
if "quant_model" in globals() and globals().get("quant_model") is not None:
    m = quant_model
    MODEL = globals().get("model_name", MODEL)            # match the driver's model
    tok = globals().get("tokenizer", None) or AutoTokenizer.from_pretrained(MODEL)
    print(f"[reuse] using `quant_model` from the cell-4 driver: {MODEL} -- NO re-quantization")
elif globals().get("_GF4_BUILD_KEY") == _build_key and "m" in globals() and getattr(m, "_gf4_ready", False):
    tok = globals().get("tok", None) or AutoTokenizer.from_pretrained(MODEL)
    print(f"[reuse] {MODEL} already quantized in this cell -- skipping rebuild")
else:
    tok = AutoTokenizer.from_pretrained(MODEL)
    _train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    _ids   = _tok_corpus(tok, _train)
    calib  = [_ids[(s:=random.randint(0, _ids.size(0)-SEQLEN-1)):s+SEQLEN].unsqueeze(0)
              for _ in range(NCALIB)]
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV)
    m = quantize_model_fp(m, calib, block_size=BS, e_bits=E_B, m_bits=M_B,
            e_bits_scale=E_S, m_bits_scale=M_S, device=DEV,
            use_HG=False, use_Hessian=False, use_adap=False, use_forward=False,
            Hadamard=True, joint=False, preshift=False, decompose=False,
            had_block_size="auto", use_gf4=True)
    enable_fast_kernels(m, enable=True)
    m._gf4_ready = True
    globals()["_GF4_BUILD_KEY"] = _build_key

# WikiText test split for PPL (moved to Salesforce/wikitext on new huggingface_hub)
test_ids = _tok_corpus(tok, load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test"))

# E2M1 ladder injected as gf4_levels -> runs the NVFP4 codebook through the
# IDENTICAL GF4 machinery (rms*clip scale, per-16 block, Hadamard).
E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=DEV) / 6.0
def set_levels(model, levels):
    for mod in model.modules():
        if type(mod).__name__ == "HadamardQuantLinearFP" and getattr(mod, "weight_q", None) is not None:
            mod.gf4_levels = (levels.to(DEV) if levels is not None else None)

out = {}
with act_quant_mode(m, mode=None):           out["A16 (W4A16)"]  = ppl_eval(m, test_ids)
set_levels(m, E2M1)
with act_quant_mode(m, mode="gf4_adaptive"): out["E2M1/NVFP4"]   = ppl_eval(m, test_ids)
set_levels(m, None)
with act_quant_mode(m, mode="gf4_adaptive"): out["GF4 adaptive"] = ppl_eval(m, test_ids)
with act_quant_mode(m, mode="gf4_residual"): out["residual-GF4"] = ppl_eval(m, test_ids)

print(f"\n===== ISO-ENERGY ({MODEL}, WikiText-2 nonoverlap-{SEQLEN}) =====")
for k in ["A16 (W4A16)", "E2M1/NVFP4", "GF4 adaptive", "residual-GF4"]:
    print(f"  {k:16s} {out[k]:8.3f}")
d = out["E2M1/NVFP4"] - out["GF4 adaptive"]
print(f"\n  GF4-adaptive vs E2M1/NVFP4 at IDENTICAL hw cost: {d:+.3f} ppl "
      f"({'GF4 better' if d > 0 else 'E2M1 better'})")
print(f"  residual-GF4 vs A16: {out['residual-GF4']-out['A16 (W4A16)']:+.3f} ppl")

# ── WIRE-BACK: export ACTUAL linear shapes + config + PPL for local HW sim ────
# Captures every quantized layer's (in=K, out=N) dims with multiplicity, so the
# Timeloop problem generator + hw_sim_gf4.py here can model THIS network exactly
# (architecture-agnostic: handles Llama GQA / gated MLP / OPT alike).  Download
# the json and drop it in  timeloop_gf4/models/  locally, then run:
#     python3 gen_problems.py && ./run.sh && python3 postprocess.py
import json as _json
_sc = {}
for _mod in m.modules():
    if type(_mod).__name__ == "HadamardQuantLinearFP" and getattr(_mod, "weight_q", None) is not None:
        _N, _K = (int(x) for x in _mod.weight_q.shape)   # weight is [out, in]
        _sc[(_K, _N)] = _sc.get((_K, _N), 0) + 1
_cfg = m.config
_export = {
    "model": MODEL, "seqlen": SEQLEN, "block_size": BS,
    "config": {"hidden_size": getattr(_cfg, "hidden_size", None),
               "intermediate_size": getattr(_cfg, "intermediate_size", None),
               "num_hidden_layers": getattr(_cfg, "num_hidden_layers", None),
               "vocab_size": getattr(_cfg, "vocab_size", None)},
    "ppl": out,
    "shapes": [{"K": k, "N": n, "count": c} for (k, n), c in sorted(_sc.items())],
}
_fn = MODEL.split("/")[-1] + "_hwsim.json"
with open(_fn, "w") as _f:
    _json.dump(_export, _f, indent=2)
try:                                  # auto-download in Colab
    from google.colab import files as _gfiles
    _gfiles.download(_fn)
except Exception:
    pass
print(f"\n[hwsim] wrote {_fn}  (download -> timeloop_gf4/models/ locally)")
print(_json.dumps(_export, indent=2))   # also printed so you can copy from output
'''


SETUP_MD = """## FP4 / GF4 quantization — Colab runbook

**1. Runtime → Change runtime type → A100 GPU (High-RAM).** OPT-6.7B / Llama-7B
will OOM on a T4/L4. The setup cell below asserts a GPU and warns if it isn't an A100.

**2. Run order:** the SETUP cell, then the module cells (triton / bit_split /
helpers), then **either** the *driver* cell (full results table) **or** the
*iso-energy* cell (GF4-vs-NVFP4 + hwsim export) — not both (each quantizes once;
the iso-energy cell reuses the driver's model if you ran it).

**3. Model:** set it in the SETUP cell (`os.environ["MODEL"]`, used by the driver)
and/or the `MODEL=` line in the iso-energy cell.

**4. Gated models (Llama):** uncomment the `login()` line in setup. OPT/GPT need no token."""

SETUP_CODE = r'''# ============================================================================
# COLAB SETUP — RUN ME FIRST.  Needs an A100 (Runtime -> Change runtime type).
# ============================================================================
import os
# CUDA arch for the JIT-compiled kernels (A100=8.0). Set before torch/extension import.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6;8.9")
import subprocess, torch

# 1) GPU check — require a GPU, recommend A100 for 6.7B/7B.
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"
assert torch.cuda.is_available(), "No GPU! Runtime -> Change runtime type -> A100 GPU."
mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu}  ({mem_gb:.0f} GB)")
if "A100" not in gpu:
    print(f"  WARNING: {gpu} is not an A100. OPT-6.7B / Llama-7B will likely OOM — "
          "switch to A100, or use a smaller MODEL (125m/1.3b/2.7b) and EVAL_BS=1.")

# 2) Dependencies (Colab ships torch; add the rest). Safe to re-run.
subprocess.run("pip -q install -U transformers datasets accelerate triton".split(), check=False)

# 3) Hugging Face login — ONLY for gated models (Llama). OPT/GPT need NO token.
#    Uncomment, run, and paste a token from https://huggingface.co/settings/tokens :
# from huggingface_hub import login; login()

# 4) Model for the DRIVER cell (it reads os.environ["MODEL"]).
#    keys: 125m  1.3b  2.7b  6.7b  13b  |  llama-1b  llama-3b  llama-7b
os.environ["MODEL"] = "6.7b"          # <-- SET ME
print("Driver MODEL =", os.environ["MODEL"],
      "| (the iso-energy cell has its own MODEL= variable)")
'''


def main():
    nb = json.load(open(NB))
    # Idempotent: strip any previously-inserted setup cells, operate on the
    # canonical 6-cell base (modules 0,1,2; bit_split 3; driver 4; experiment 5),
    # then re-prepend fresh setup cells.
    cells = [c for c in nb["cells"]
             if c.get("metadata", {}).get("gf4_role") != "setup"]
    assert len(cells) >= 5, f"expected >=5 base cells, got {len(cells)}"

    bit_split = open(os.path.join(HERE, "FP_Quantization_Experiments", "bit_split.py")).read()
    driver    = open(os.path.join(HERE, "FP_QuantNetworkTest_LLM.py")).read()

    cells[3]["source"] = as_source(comment_pkg_imports(bit_split))
    cells[3]["outputs"] = []; cells[3]["execution_count"] = None
    cells[4]["source"] = as_source(comment_pkg_imports(driver))
    cells[4]["outputs"] = []; cells[4]["execution_count"] = None

    exp = {"cell_type": "code", "metadata": {}, "execution_count": None,
           "outputs": [], "source": as_source(EXPERIMENT_CELL)}
    if len(cells) == 5:
        cells.append(exp)
    else:
        cells[5] = exp
    base = cells[:6]

    setup_md = {"cell_type": "markdown", "metadata": {"gf4_role": "setup"},
                "source": as_source(SETUP_MD)}
    setup_code = {"cell_type": "code", "metadata": {"gf4_role": "setup"},
                  "execution_count": None, "outputs": [], "source": as_source(SETUP_CODE)}
    nb["cells"] = [setup_md, setup_code] + base

    json.dump(nb, open(NB, "w"), indent=1)
    print(f"wrote {NB}: {len(nb['cells'])} cells (2 setup + 6 base)")


if __name__ == "__main__":
    main()

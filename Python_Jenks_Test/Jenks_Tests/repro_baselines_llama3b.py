"""
Faithful RTN + GPTQ W4A16 baselines for Llama-3.2-3B, evaluated under the SAME
WikiText-2 protocol as the main pipeline (non-overlapping 2048-token chunks,
mean NLL exponentiated).  Pure PyTorch, runs in the main env (no extra deps).

  - FP16 baseline       : sanity check (expect ~7.82)
  - RTN  W4 g128        : per-group asymmetric INT4 round-to-nearest
  - GPTQ W4 g128        : the published algorithm — sequential, block-by-block,
                          Hessian-weighted column-wise OBQ (Frantar et al. 2023),
                          using quantized activations to drive each layer's
                          Hessian (the faithful version, not per-layer-isolated).

Weight-only: weights are fake-quantized (quantize->dequantize to fp16) and the
model runs normally; perplexity matches a packed-INT4 model.
"""
import os, math, time, gc, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_NAME = "meta-llama/Llama-3.2-3B"
DEVICE     = "cuda"
SEQLEN     = 2048
GROUP      = 128
BITS       = 4
NSAMPLES   = 128
EVAL_BS    = 2
SEED       = 0
DEBUG_MSE  = True   # print per-layer RTN vs GPTQ reconstruction MSE
torch.manual_seed(SEED); random.seed(SEED)

# ── Eval (verbatim protocol from FP_QuantNetworkTest_LLM.py) ────────────────
def tokenize_corpus(tok, dataset, add_special_tokens=True):
    full_text = "\n\n".join(dataset["text"])
    return tok(full_text, return_tensors="pt",
               add_special_tokens=add_special_tokens).input_ids.squeeze(0)

def chunk_nll(model, chunk):
    with torch.inference_mode():
        logits = model(chunk, use_cache=False).logits
        sl = logits[:, :-1, :].contiguous(); lb = chunk[:, 1:].contiguous()
        del logits
        loss = F.cross_entropy(sl.view(-1, sl.size(-1)), lb.view(-1), reduction="mean")
        del sl, lb
    return None if (torch.isnan(loss) or torch.isinf(loss)) else loss

def ppl_eval(model, input_ids, seq_len=SEQLEN, batch_size=EVAL_BS):
    model.eval()
    n = input_ids.size(0) // seq_len
    chunks = input_ids[:n * seq_len].view(n, seq_len)
    nll_sum, ntok = 0.0, 0
    for i in range(0, n, batch_size):
        batch = chunks[i:i + batch_size].to(DEVICE)
        loss = chunk_nll(model, batch); del batch
        if loss is None: continue
        b = min(batch_size, n - i)
        nll_sum += loss.item() * b * (seq_len - 1); ntok += b * (seq_len - 1)
    return math.exp(nll_sum / ntok) if ntok else float("inf")

# ── Quant primitives ────────────────────────────────────────────────────────
def find_params(w, bits=BITS):
    """Per-row asymmetric scale/zero for a [rows, g] column group."""
    maxq = 2 ** bits - 1
    xmax = torch.clamp(w.max(1, keepdim=True).values, min=0)
    xmin = torch.clamp(w.min(1, keepdim=True).values, max=0)
    scale = torch.clamp((xmax - xmin) / maxq, min=1e-8)
    zero  = torch.round(-xmin / scale)
    return scale, zero, maxq

def fake_quant(w, scale, zero, maxq):
    q = torch.clamp(torch.round(w / scale) + zero, 0, maxq)
    return scale * (q - zero)

def layer_linears(layer):
    return [(n, m) for n, m in layer.named_modules() if isinstance(m, nn.Linear)]

# ── RTN ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def quantize_rtn(model, group=GROUP):
    for layer in model.model.layers:
        for _, lin in layer_linears(layer):
            W = lin.weight.data.float(); cols = W.shape[1]; Q = torch.empty_like(W)
            for g0 in range(0, cols, group):
                g1 = min(g0 + group, cols)
                s, z, mq = find_params(W[:, g0:g1])
                Q[:, g0:g1] = fake_quant(W[:, g0:g1], s, z, mq)
            lin.weight.data = Q.to(lin.weight.dtype)

# ── GPTQ column-wise OBQ (per linear, given Hessian) ────────────────────────
@torch.no_grad()
def gptq_quant_weight(W, H, bits=BITS, group=GROUP, percdamp=0.01):
    W = W.clone().float(); rows, cols = W.shape
    H = H.clone().to(W.device).float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1; W[:, dead] = 0
    damp = percdamp * torch.mean(torch.diag(H))
    di = torch.arange(cols, device=W.device); H[di, di] += damp
    H = torch.linalg.cholesky(H); H = torch.cholesky_inverse(H)
    Hinv = torch.linalg.cholesky(H, upper=True)
    Q = torch.zeros_like(W); bs = 128; scale = zero = mq = None
    for i1 in range(0, cols, bs):
        i2 = min(i1 + bs, cols)
        W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1); Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]; d = Hinv1[i, i]
            if (i1 + i) % group == 0:
                scale, zero, mq = find_params(W[:, (i1 + i):min(i1 + i + group, cols)], bits)
            q = fake_quant(w.unsqueeze(1), scale, zero, mq).flatten()
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q

# ── Block-sequential GPTQ (the faithful algorithm) ──────────────────────────
# Quantize decoder block-by-block.  Each block's Hessians are built from the
# inputs PRODUCED BY THE ALREADY-QUANTIZED PREVIOUS BLOCKS, then after the
# block is quantized we push the calibration set through the *quantized* block
# to get the next block's inputs.  This is what makes GPTQ beat RTN — the
# clean-input (non-sequential) variant optimizes the wrong curvature and is
# actually worse than RTN.
@torch.no_grad()
def quantize_gptq(model, calib):
    layers = model.model.layers

    # 1) Capture the input (and per-call kwargs) to the first decoder block.
    class _Caught(Exception):
        pass
    inps, kwargs_cache = [], []
    class Catcher(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, hidden_states, **kwargs):
            inps.append(hidden_states.detach().cpu())
            kwargs_cache.append(kwargs)
            raise _Caught
    layers[0] = Catcher(layers[0])
    for s in calib:
        try:
            model(s.to(DEVICE), use_cache=False)
        except _Caught:
            pass
    layers[0] = layers[0].m
    gc.collect(); torch.cuda.empty_cache()

    # 2) Walk the blocks in order.
    for li, layer in enumerate(layers):
        named = layer_linears(layer)
        Hs = {name: torch.zeros(m.in_features, m.in_features,
                                dtype=torch.float32, device=DEVICE)
              for name, m in named}
        nsmp = {name: 0 for name, _ in named}
        hooks = []
        def mk(name):
            def hook(mod, inp, out):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                Hs[name] += x.t() @ x; nsmp[name] += x.shape[0]
            return hook
        for name, m in named:
            hooks.append(m.register_forward_hook(mk(name)))
        for j, inp in enumerate(inps):
            layer(inp.to(DEVICE), **kwargs_cache[j])
        for h in hooks:
            h.remove()

        for name, m in named:
            W0 = m.weight.data.float()
            Q = gptq_quant_weight(W0, Hs[name])
            if DEBUG_MSE:
                cols = W0.shape[1]; Qr = torch.empty_like(W0)
                for g0 in range(0, cols, GROUP):
                    g1 = min(g0 + GROUP, cols)
                    s, z, mq = find_params(W0[:, g0:g1])
                    Qr[:, g0:g1] = fake_quant(W0[:, g0:g1], s, z, mq)
                rtn_mse  = (W0 - Qr).pow(2).mean().item()
                gptq_mse = (W0 - Q).pow(2).mean().item()
                flag = "  <-- GPTQ WORSE" if gptq_mse > 1.5 * rtn_mse else ""
                print(f"    L{li:2d} {name:10s} {tuple(W0.shape)!s:14s} "
                      f"rtn={rtn_mse:.3e} gptq={gptq_mse:.3e}{flag}", flush=True)
            m.weight.data = Q.to(m.weight.dtype)
            Hs[name] = None
        del Hs; gc.collect(); torch.cuda.empty_cache()

        # 3) Forward calib through the now-quantized block → next block's inputs.
        for j, inp in enumerate(inps):
            out = layer(inp.to(DEVICE), **kwargs_cache[j])
            inps[j] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
        print(f"  GPTQ block {li+1}/{len(layers)} done", flush=True)
        gc.collect(); torch.cuda.empty_cache()

# ── Driver ──────────────────────────────────────────────────────────────────
def load_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)

def make_calib(tok, train, n=NSAMPLES, seqlen=SEQLEN):
    ids = tokenize_corpus(tok, train, add_special_tokens=False); total = ids.size(0)
    return [ids[(s := random.randint(0, total - seqlen - 1)):s + seqlen].unsqueeze(0)
            for _ in range(n)]

def main():
    print(f"=== Faithful RTN / GPTQ baselines for {MODEL_NAME} ===", flush=True)
    tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
    test  = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test_ids = tokenize_corpus(tok, test, add_special_tokens=False)
    res = {}

    t = time.time(); m = load_model()
    res["FP16"] = ppl_eval(m, test_ids)
    print(f"FP16  PPL = {res['FP16']:.4f}  ({time.time()-t:.0f}s)", flush=True)
    del m; gc.collect(); torch.cuda.empty_cache()

    t = time.time(); m = load_model(); quantize_rtn(m)
    res["RTN-W4g128"] = ppl_eval(m, test_ids)
    print(f"RTN   PPL = {res['RTN-W4g128']:.4f}  ({time.time()-t:.0f}s)", flush=True)
    del m; gc.collect(); torch.cuda.empty_cache()

    t = time.time(); m = load_model(); calib = make_calib(tok, train)
    quantize_gptq(m, calib)
    res["GPTQ-W4g128"] = ppl_eval(m, test_ids)
    print(f"GPTQ  PPL = {res['GPTQ-W4g128']:.4f}  ({time.time()-t:.0f}s)", flush=True)
    del m; gc.collect(); torch.cuda.empty_cache()

    print("\n=== RESULTS (WikiText-2, non-overlapping 2048) ===", flush=True)
    base = res["FP16"]
    for k, v in res.items():
        d = "" if k == "FP16" else f"  (+{v-base:.2f})"
        print(f"  {k:14s} {v:8.3f}{d}", flush=True)
    if res["GPTQ-W4g128"] >= res["RTN-W4g128"]:
        print("  WARNING: GPTQ did not beat RTN — check implementation.", flush=True)
    else:
        print("  sanity OK: FP16 < GPTQ < RTN", flush=True)

if __name__ == "__main__":
    main()

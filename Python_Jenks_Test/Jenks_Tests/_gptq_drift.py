import torch, torch.nn as nn, random
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from repro_baselines_llama3b import (gptq_quant_weight, find_params, fake_quant,
                                     GROUP, tokenize_corpus, layer_linears)
DEV="cuda"; random.seed(0); torch.manual_seed(0)
tok=AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
m=AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B",torch_dtype=torch.float16).to(DEV)
tr=load_dataset("wikitext","wikitext-2-raw-v1",split="train")
ids=tokenize_corpus(tok,tr,add_special_tokens=False)
calib=[ids[(s:=random.randint(0,ids.size(0)-2049)):s+2048].unsqueeze(0) for _ in range(64)]
layers=m.model.layers
class C(Exception):pass
inps=[]; kw=[]
class Catch(nn.Module):
    def __init__(s,mm):super().__init__();s.m=mm
    def forward(s,hs,**k): inps.append(hs.detach().cpu()); kw.append(k); raise C
layers[0]=Catch(layers[0])
for s in calib:
    try: m(s.to(DEV),use_cache=False)
    except C: pass
layers[0]=layers[0].m
ref=[t.clone() for t in inps]          # fp16 reference inputs to block0 (same)
def rel(a,b): 
    a=a.float();b=b.float(); return (a-b).pow(2).mean().sqrt().item()/(b.pow(2).mean().sqrt().item()+1e-9)
for li,layer in enumerate(layers):
    named=layer_linears(layer)
    Hs={n:torch.zeros(mm.in_features,mm.in_features,device=DEV) for n,mm in named}
    hk=[]
    def mk(n):
        def f(mod,i,o): x=i[0].detach().reshape(-1,i[0].shape[-1]).float(); Hs[n]+=x.t()@x
        return f
    for n,mm in named: hk.append(mm.register_forward_hook(mk(n)))
    for j,inp in enumerate(inps): layer(inp.to(DEV),**kw[j])
    for h in hk: h.remove()
    for n,mm in named:
        Q=gptq_quant_weight(mm.weight.data.float(),Hs[n]); mm.weight.data=Q.to(mm.weight.dtype)
    Hs=None
    # advance quantized inps AND fp16 ref, measure drift at block output
    drift=0.0
    for j in range(len(inps)):
        oq=layer(inps[j].to(DEV),**kw[j]); oq=(oq[0] if isinstance(oq,tuple) else oq).detach()
        # fp16 ref: need fp16 weights — recompute via saved? Instead track ref separately:
        inps[j]=oq.cpu()
    # fp16 ref path: forward ref through a PRISTINE copy of this layer's fp16 weights is gone.
    print(f"block {li:2d}: mean|quant-out|={sum(t.float().abs().mean().item() for t in inps)/len(inps):.4f}",flush=True)

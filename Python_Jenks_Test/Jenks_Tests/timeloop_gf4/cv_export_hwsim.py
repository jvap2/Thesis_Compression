"""Emit DENSE (density=1) conv/linear-as-GEMM hwsim JSONs for the GF4 CV-study nets,
so the Timeloop GF4-Engine can price their energy/area. Unlike export_all_hwsim.py
(which exports the SPARSE harvested nets for the pruning study), these are the full
dense models — GF4 is a PRECISION story, not a sparsity one, so density=1 everywhere.
Mapping: C=Cin*R*S, K=Cout, N=P*Q (a conv = GEMM). Writes models/cv_<key>_hwsim.json.
Run from Jenks_Tests:  python3 timeloop_gf4/cv_export_hwsim.py"""
import torch, torch.nn as nn, torch.nn.functional as F, json, os, sys
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
os.chdir(ROOT); sys.path.insert(0, ROOT)
CHEN="chenyaofo/pytorch-cifar-models"

def hooked_shapes(model, hw, in_ch=3):
    recs=[]; hooks=[]
    def mk(m):
        def h(m,inp,out):
            if isinstance(m,nn.Conv2d):
                Cin,R,S=m.in_channels,m.kernel_size[0],m.kernel_size[1]
                P,Q=out.shape[-2],out.shape[-1]; K=m.out_channels
                recs.append(dict(name=f"L{len(recs)}", K=Cin*R*S, N=K, batchN=int(P*Q), count=1,
                                 density=1.0, macs=int(Cin*R*S*K*P*Q)))
            elif isinstance(m,nn.Linear):
                recs.append(dict(name=f"L{len(recs)}", K=m.in_features, N=m.out_features, batchN=1, count=1,
                                 density=1.0, macs=int(m.in_features*m.out_features)))
        return h
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.Linear)): hooks.append(m.register_forward_hook(mk(m)))
    model.eval()
    with torch.no_grad(): model(torch.randn(1,in_ch,hw,hw))
    for h in hooks: h.remove()
    return recs

def lenet300():
    return nn.Sequential(nn.Flatten(), nn.Linear(784,300), nn.ReLU(),
                         nn.Linear(300,100), nn.ReLU(), nn.Linear(100,10))
def lenet5():
    from rcnet import create_lenet5; return create_lenet5()
def densenet40():
    from densenet import create_densenet40; return create_densenet40()
def hub(name):
    try:    return torch.hub.load(CHEN, name, pretrained=False, trust_repo=True)
    except Exception: return torch.hub.load(CHEN, name, pretrained=True, trust_repo=True)

# (key, builder, input_hw, in_ch). cifar100 conv shapes == cifar10 (only final FC
# differs 10 vs 100 -> negligible MACs), so one profile per architecture suffices.
NETS=[
  ("cv_resnet32",   lambda: hub("cifar10_resnet32"),   32, 3),
  ("cv_resnet56",   lambda: hub("cifar10_resnet56"),   32, 3),
  ("cv_vgg19",      lambda: hub("cifar10_vgg19_bn"),   32, 3),
  ("cv_densenet40", densenet40,                        32, 3),
  ("cv_lenet5",     lenet5,                            28, 1),
  ("cv_lenet300",   lenet300,                          28, 1),
]

def main():
    os.makedirs(os.path.join(HERE,"models"), exist_ok=True); done=0
    for key, build, hw, inch in NETS:
        try:
            recs=hooked_shapes(build(), hw, inch)
            exp=dict(model=key, source="dense", input_hw=hw,
                     note="conv/linear as GEMM: C=Cin*R*S, K=Cout, N=P*Q; density=1 (dense, GF4 precision study)",
                     shapes=recs, total_macs=sum(r["macs"] for r in recs),
                     effective_macs_sparse=sum(r["macs"] for r in recs))
            json.dump(exp, open(os.path.join(HERE,"models",f"{key}_hwsim.json"),"w"), indent=2)
            print(f"{key:16s} {len(recs):3d} layers  {exp['total_macs']/1e6:8.2f}M MACs")
            done+=1
        except Exception as e:
            print(f"{key:16s} ERR: {repr(e)[:80]}")
    print(f"\nwrote {done} dense CV hwsim JSONs to timeloop_gf4/models/")

if __name__=="__main__": main()

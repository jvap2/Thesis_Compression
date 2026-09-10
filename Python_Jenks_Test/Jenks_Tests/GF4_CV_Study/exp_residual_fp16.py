"""Residual-stream precision experiment (ResNet, additive skips).

Our W4A4 already keeps the skip in fp16 by construction: in a BasicBlock the add is
`out += identity` with identity=x un-quantized (4-bit conv products accumulate into the
fp16 psum, and the skip just joins that psum). This script measures the COST of the
alternative — a truly uniform 4-bit activation datapath where the residual stream is
also 4-bit — by GF4-quantizing each residual block's INPUT (which feeds both conv1 and
the identity skip). conv1 re-quantizes its own input either way, so the delta isolates
the skip-quantization cost. Expectation: forcing 4-bit skips hurts, proving the fp16
skip (0.2% of activation traffic on ResNet-56) is worth keeping.

W4A4(skip fp16)  vs  W4A4(skip 4-bit).  Run:  BLK=9 python3 GF4_CV_Study/exp_residual_fp16.py"""
import sys, os, copy, csv, torch
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0, ROOT); os.chdir(HERE)
DATA=os.path.join(ROOT,"datasets"); BLK=int(os.environ.get("BLK","9"))
ABLK=int(os.environ.get("ABLK","16"))          # activation block size (NVFP4-style)
from FP_Quantization_Experiments import quantize_model_fp
from FP_Quantization_Experiments.bit_split import set_act_quant, quantize_activations_gf4_adaptive
from torchvision import datasets, transforms
DEV="cuda" if torch.cuda.is_available() else "cpu"
CHEN="chenyaofo/pytorch-cifar-models"
NORM={"cifar10":([0.4914,0.4822,0.4465],[0.2470,0.2435,0.2616]),
      "cifar100":([0.5071,0.4865,0.4409],[0.2673,0.2564,0.2762])}
def loader(ds, bs=256):
    m,s=NORM[ds]; tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(m,s)])
    D=datasets.CIFAR100 if ds=="cifar100" else datasets.CIFAR10
    return torch.utils.data.DataLoader(D(DATA,False,tf,download=False),bs,shuffle=False,num_workers=4)
def top1(m, ld):
    m.eval().to(DEV); c=t=0
    with torch.no_grad():
        for x,y in ld:
            x,y=x.to(DEV),y.to(DEV); c+=(m(x).argmax(1)==y).sum().item(); t+=y.numel()
    return 100*c/t

def add_skip_quant(model):
    """forward_pre_hook on every residual block: GF4-quantize the block input, so the
    identity skip becomes 4-bit (uniform-A4 datapath). Returns handles to remove."""
    hs=[]
    def pre(mod, inp):
        x=inp[0]
        xq=quantize_activations_gf4_adaptive(x, ABLK, levels=None)
        return (xq,)+tuple(inp[1:])
    for mod in model.modules():
        if type(mod).__name__ in ("BasicBlock","Bottleneck"):
            hs.append(mod.register_forward_pre_hook(pre))
    return hs

JOBS=[("ResNet-32","cifar10","cifar10_resnet32"), ("ResNet-56","cifar10","cifar10_resnet56"),
      ("ResNet-32","cifar100","cifar100_resnet32"), ("ResNet-56","cifar100","cifar100_resnet56")]
rows=[]
for name, ds, hub in JOBS:
    ld=loader(ds); model=torch.hub.load(CHEN,hub,pretrained=True,trust_repo=True).to(DEV).eval()
    base=top1(model, ld)
    qm=quantize_model_fp(copy.deepcopy(model), ld, block_size=BLK, e_bits=2, m_bits=1,
                         e_bits_scale=4, m_bits_scale=3, device=DEV,
                         use_HG=False, use_Hessian=False, use_adap=True, use_forward=False, Hadamard=False)
    set_act_quant(qm, "gf4_adaptive")
    a_fp16=top1(qm, ld)                              # skip fp16 (current)
    hs=add_skip_quant(qm); a_4b=top1(qm, ld); [h.remove() for h in hs]   # skip 4-bit
    print(f"{name:10s} {ds:9s}  FP32 {base:5.2f}  W4A4/skip-fp16 {a_fp16:5.2f}  W4A4/skip-4bit {a_4b:5.2f}  (skip cost {a_fp16-a_4b:+.2f})", flush=True)
    rows.append(dict(model=name, dataset=ds, block_size=BLK, fp32=round(base,2),
                     w4a4_skip_fp16=round(a_fp16,2), w4a4_skip_4bit=round(a_4b,2),
                     skip_fp16_gain=round(a_fp16-a_4b,2)))
OUT="gf4_cv_residual_fp16.csv"
with open(OUT,"w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
print(f"\nsaved {OUT}")

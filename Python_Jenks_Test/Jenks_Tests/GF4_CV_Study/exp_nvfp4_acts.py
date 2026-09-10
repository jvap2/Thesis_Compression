"""GF4 vs NVFP4 activations, identical E2M1 weights (apples-to-apples W4A4).

Weights are quantized once (E2M1 adaptive-mesh, per-input-channel BLK=9) exactly as
the headline W4A4 run. Then the SAME quantized model is evaluated with two activation
codebooks, toggled non-destructively via set_act_quant:
  - GF4  : Gaussian-optimal codebook (quantize_activations_gf4_adaptive)
  - NVFP4: uniform E2M1 + E4M3 per-16 block scale (the plain-4-bit datapath our
           energy baseline prices). In the conv/linear path this is the generic
           quantize_activations with e_bits_scale=4,m_bits_scale=3 -> nvfp4 branch.
This isolates the codebook: same weights, same block scale, only the 8 FP4 levels differ.

Run:  BLK=9 python3 GF4_CV_Study/exp_nvfp4_acts.py"""
import sys, os, copy, csv, time, torch, torch.nn as nn, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0, ROOT); os.chdir(HERE)
DATA=os.path.join(ROOT,"datasets"); BLK=int(os.environ.get("BLK","9"))
from FP_Quantization_Experiments import quantize_model_fp
from FP_Quantization_Experiments.bit_split import set_act_quant
from torchvision import datasets, transforms
DEV="cuda" if torch.cuda.is_available() else "cpu"
CHEN="chenyaofo/pytorch-cifar-models"
NORM={"cifar10":([0.4914,0.4822,0.4465],[0.2470,0.2435,0.2616]),
      "cifar100":([0.5071,0.4865,0.4409],[0.2673,0.2564,0.2762]),
      "mnist":([0.1307],[0.3081])}
def loaders(ds, bs=256):
    m,s=NORM[ds]; tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(m,s)])
    if ds=="cifar10":  te=datasets.CIFAR10(DATA,False,tf,download=False); tr=datasets.CIFAR10(DATA,True,tf,download=False)
    elif ds=="cifar100": te=datasets.CIFAR100(DATA,False,tf,download=False); tr=datasets.CIFAR100(DATA,True,tf,download=False)
    else: te=datasets.MNIST(DATA,False,tf,download=False); tr=datasets.MNIST(DATA,True,tf,download=False)
    L=lambda d,sh: torch.utils.data.DataLoader(d,batch_size=bs,shuffle=sh,num_workers=4)
    return L(tr,True), L(te,False)
def top1(m, loader):
    m.eval().to(DEV); c=t=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(DEV),y.to(DEV); c+=(m(x).argmax(1)==y).sum().item(); t+=y.numel()
    return 100*c/t
def train_mnist(model, epochs=3):
    tr,te=loaders("mnist"); model.to(DEV).train()
    opt=torch.optim.Adam(model.parameters(),1e-3)
    for _ in range(epochs):
        for x,y in tr:
            x,y=x.to(DEV),y.to(DEV); opt.zero_grad(); F.cross_entropy(model(x),y).backward(); opt.step()
    return model
def lenet300():
    return nn.Sequential(nn.Flatten(), nn.Linear(784,300), nn.ReLU(),
                         nn.Linear(300,100), nn.ReLU(), nn.Linear(100,10))
def lenet5():
    from rcnet import create_lenet5; return create_lenet5()
def densenet40():
    from densenet import create_densenet40
    m=create_densenet40(); sd=torch.load("densenet40_cifar10.pth", map_location="cpu")
    m.load_state_dict(sd); return m
def build(spec):
    kind,arg=spec
    if kind=="hub":  return torch.hub.load(CHEN, arg, pretrained=True, trust_repo=True)
    if kind=="lenet300": return train_mnist(lenet300())
    if kind=="lenet5":   return train_mnist(lenet5())
    if kind=="densenet40": return densenet40()

JOBS=[
  ("ResNet-32","cifar10",  ("hub","cifar10_resnet32")),
  ("ResNet-56","cifar10",  ("hub","cifar10_resnet56")),
  ("VGG-19","cifar10",     ("hub","cifar10_vgg19_bn")),
  ("DenseNet-40","cifar10",("densenet40",None)),
  ("ResNet-32","cifar100", ("hub","cifar100_resnet32")),
  ("ResNet-56","cifar100", ("hub","cifar100_resnet56")),
  ("VGG-19","cifar100",    ("hub","cifar100_vgg19_bn")),
  ("LeNet-5","mnist",      ("lenet5",None)),
  ("LeNet-300","mnist",    ("lenet300",None)),
]
rows=[]
for name, ds, spec in JOBS:
    try:
        t0=time.time(); model=build(spec).to(DEV).eval(); tr,te=loaders(ds)
        base=top1(model, te)
        qm=quantize_model_fp(copy.deepcopy(model), te, block_size=BLK, e_bits=2, m_bits=1,
                             e_bits_scale=4, m_bits_scale=3, device=DEV,
                             use_HG=False, use_Hessian=False, use_adap=True, use_forward=False, Hadamard=False)
        w4a16=top1(qm, te)                       # act None
        set_act_quant(qm, "gf4_adaptive"); a_gf4=top1(qm, te)
        set_act_quant(qm, "nvfp4");        a_nv =top1(qm, te)
        r=dict(model=name, dataset=ds, fp32=round(base,2), w4a16=round(w4a16,2),
               w4a4_gf4=round(a_gf4,2), w4a4_nvfp4=round(a_nv,2),
               gf4_minus_nvfp4=round(a_gf4-a_nv,2), secs=round(time.time()-t0))
        print(f"{name:10s} {ds:9s}  FP32 {base:5.2f}  W4A16 {w4a16:5.2f}  "
              f"W4A4-GF4 {a_gf4:5.2f}  W4A4-NVFP4 {a_nv:5.2f}  (GF4-NVFP4 {a_gf4-a_nv:+.2f})", flush=True)
    except Exception as e:
        r=dict(model=name, dataset=ds, fp32="", w4a16="", w4a4_gf4="",
               w4a4_nvfp4="ERR: "+repr(e)[:60], gf4_minus_nvfp4="", secs="")
        print(f"{name:10s} {ds:9s}  ERROR: {repr(e)[:80]}", flush=True)
    rows.append(r); r["block_size"]=BLK
OUT=f"gf4_vs_nvfp4_acts_blk{BLK}.csv"
with open(OUT,"w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
print(f"\nsaved {OUT}")

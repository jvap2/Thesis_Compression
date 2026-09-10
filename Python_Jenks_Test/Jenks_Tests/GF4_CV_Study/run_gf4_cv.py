"""GF4 post-training quantization sweep on pretrained CV networks.
Baseline (FP32) vs GF4 W4A4 (E2M1 weights + GF4-adaptive activations, block 16).
Pretrained CIFAR models from chenyaofo/pytorch-cifar-models (torch.hub); LeNets
trained on MNIST here. Writes gf4_cv_results.csv."""
import sys, os, copy, csv, time, torch, torch.nn as nn, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0, ROOT); os.chdir(HERE)
DATA=os.path.join(ROOT,"datasets")   # cached CIFAR/MNIST live in Jenks_Tests/datasets
BLK=int(os.environ.get("BLK","16"))  # 16 = per-block; >=4608 = per-filter (one scale/output channel)
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
    try:
        from rcnet import create_lenet5; return create_lenet5()
    except Exception:
        class L5(nn.Module):
            def __init__(s):
                super().__init__(); s.stem=nn.Module(); s.stem.conv1=nn.Conv2d(1,20,5); s.stem.conv2=nn.Conv2d(20,50,5)
                s.linear1=nn.Linear(50*16,500); s.linear2=nn.Linear(500,10)
            def forward(s,x):
                x=F.max_pool2d(F.relu(s.stem.conv1(x)),2); x=F.max_pool2d(F.relu(s.stem.conv2(x)),2)
                return s.linear2(F.relu(s.linear1(x.flatten(1))))
        return L5()

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
def params(m): return sum(p.numel() for p in m.parameters())

rows=[]
for name, ds, spec in JOBS:
    try:
        t0=time.time(); model=build(spec).to(DEV).eval(); tr,te=loaders(ds)
        p=params(model); base=top1(model, te)
        qm=quantize_model_fp(copy.deepcopy(model), te, block_size=BLK, e_bits=2, m_bits=1,
                             e_bits_scale=4, m_bits_scale=3, device=DEV,
                             use_HG=False, use_Hessian=False, use_adap=True, use_forward=False, Hadamard=False)
        gf4w=top1(qm, te)                                 # W4A16: act_quant_mode is None right after quant
        set_act_quant(qm, "gf4_adaptive")                 # turn on GF4-adaptive activations
        gf4=top1(qm, te)                                  # W4A4
        r=dict(model=name, dataset=ds, params_M=round(p/1e6,3),
               size_fp32_MB=round(p*4/1e6,2), size_int4_MB=round(p*0.5/1e6,3),
               baseline_top1=round(base,2), gf4_w4a16_top1=round(gf4w,2), drop_w4a16=round(base-gf4w,2),
               gf4_w4a4_top1=round(gf4,2), drop_w4a4=round(base-gf4,2), secs=round(time.time()-t0))
        print(f"{name:10s} {ds:9s}  FP32 {base:5.2f}%  W4A16 {gf4w:5.2f}% ({base-gf4w:+.2f})  W4A4 {gf4:5.2f}% ({base-gf4:+.2f})  ({p/1e6:.2f}M)")
    except Exception as e:
        r=dict(model=name, dataset=ds, params_M="", size_fp32_MB="", size_int4_MB="",
               baseline_top1="", gf4_w4a16_top1="", drop_w4a16="", gf4_w4a4_top1="",
               drop_w4a4="ERR: "+repr(e)[:60], secs="")
        print(f"{name:10s} {ds:9s}  ERROR: {repr(e)[:80]}")
    rows.append(r)
OUT=f"gf4_cv_results_blk{BLK}.csv"
for r in rows: r["block_size"]=BLK
with open(OUT,"w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
print(f"\nsaved {OUT}  (block_size={BLK}, {'per-filter' if BLK>=4608 else 'per-block'})")

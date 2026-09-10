"""Smoke test: does GF4 W4A4 PTQ work on a CIFAR-100 ViT-B/16 (85.9M, LayerNorm acts)?
Step 1 verifies baseline accuracy is sane (catches input-size / normalization mismatch),
THEN quantizes and compares GF4 vs matched-clip NVFP4 activations (codebook isolation),
same protocol as exp_codebook_ablation.py. If baseline is garbage, stop and diagnose.

Run:  BLK=9 python3 GF4_CV_Study/smoke_vit.py"""
import sys, os, copy, time, torch, torch.nn as nn
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0, ROOT); os.chdir(HERE)
DATA=os.path.join(ROOT,"datasets"); BLK=int(os.environ.get("BLK","9"))
from FP_Quantization_Experiments import quantize_model_fp
from FP_Quantization_Experiments import bit_split as BS
from torchvision import datasets, transforms
DEV="cuda" if torch.cuda.is_available() else "cpu"
CHEN="chenyaofo/pytorch-cifar-models"
NVFP4_POS = torch.tensor([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0])/6.0
M,S=[0.5071,0.4865,0.4409],[0.2673,0.2564,0.2762]

def loader(bs=128, size=None):
    tfs=[transforms.ToTensor(), transforms.Normalize(M,S)]
    if size: tfs=[transforms.Resize(size)]+tfs
    tf=transforms.Compose(tfs)
    te=datasets.CIFAR100(DATA,False,tf,download=False)
    return torch.utils.data.DataLoader(te,bs,shuffle=False,num_workers=4)
def top1(m, ld, lim=None):
    m.eval().to(DEV); c=t=0
    with torch.no_grad():
        for i,(x,y) in enumerate(ld):
            x,y=x.to(DEV),y.to(DEV); c+=(m(x).argmax(1)==y).sum().item(); t+=y.numel()
            if lim and i>=lim: break
    return 100*c/t
def set_codebook(model, levels):
    lv=levels.to(DEV)
    for mod in model.modules():
        if isinstance(mod,(BS.QuantLinearFP,BS.QuantConv2dFP,BS.QuantConv1dFP)): mod.gf4_levels=lv
    BS.set_act_quant(model,"gf4_adaptive")

print("loading cifar100_vit_b16 (pretrained)...", flush=True)
model=torch.hub.load(CHEN,"cifar100_vit_b16",pretrained=True,trust_repo=True).to(DEV).eval()
# Try native 32px first; if baseline is garbage, retry at 224.
for size in (None, 224):
    ld=loader(size=size)
    acc=top1(model, ld, lim=20)   # quick partial-eval sanity
    print(f"  baseline top1 (partial, size={size or 32}) = {acc:.2f}%", flush=True)
    if acc>40: break
else:
    print("  [ABORT] baseline accuracy garbage at both sizes — input pipeline mismatch, not quantizing.", flush=True)
    sys.exit(0)
ld=loader(size=size); base=top1(model, ld)
print(f"  FULL baseline top1 (size={size or 32}) = {base:.2f}%   -> quantizing", flush=True)
t0=time.time()
qm=quantize_model_fp(copy.deepcopy(model), ld, block_size=BLK, e_bits=2, m_bits=1,
                     e_bits_scale=4, m_bits_scale=3, device=DEV,
                     use_HG=False, use_Hessian=False, use_adap=True, use_forward=False, Hadamard=False)
w4a16=top1(qm, ld)
set_codebook(qm, BS.GF4_POS); a_gf4=top1(qm, ld)
set_codebook(qm, NVFP4_POS);  a_nv =top1(qm, ld)
print(f"\nViT-B/16 CIFAR-100  FP32 {base:.2f}  W4A16 {w4a16:.2f}  "
      f"A4-GF4 {a_gf4:.2f}  A4-NVFP4(clip) {a_nv:.2f}  (codebook {a_gf4-a_nv:+.2f})  [{time.time()-t0:.0f}s]", flush=True)

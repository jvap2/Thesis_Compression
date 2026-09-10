"""GF4 vs matched-clip NVFP4 codebook on TinyImageNet (200 classes) — the hardest task
in the suite, extending the difficulty axis of the codebook ablation. Dense VGG-19 and
ResNet-32 checkpoints from Best_Results_HPO (the same nets as the sparse TinyImageNet
rows). Same protocol as exp_codebook_ablation.py: weights E2M1 adaptive (per-input-channel
BLK=9), then GF4 vs uniform-E2M1 levels under an identical adaptive clip. Crop size (56 vs
64) is auto-detected by baseline accuracy to match each checkpoint's training regime.

Run:  BLK=9 python3 GF4_CV_Study/exp_tinyimagenet.py"""
import sys, os, copy, csv, glob, time, torch, torch.nn as nn
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0, ROOT); os.chdir(ROOT)                 # run from Jenks_Tests (dataset paths are relative)
BLK=int(os.environ.get("BLK","9"))
from FP_Quantization_Experiments import quantize_model_fp
from FP_Quantization_Experiments import bit_split as BS
from harvest_all import _vgg, VGG_CFG, resnet32_c
from utils import TinyImageNetDataset
from torchvision import transforms
DEV="cuda" if torch.cuda.is_available() else "cpu"
NVFP4_POS=torch.tensor([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0])/6.0
TROOT="./datasets/tiny-imagenet-200"
NORM=transforms.Normalize(mean=[0.48024578664982126,0.44807218089384643,0.3975477478649648],
                          std=[0.2769864069088257,0.26906448510256,0.282081906210584])
id_dict={}
for i,line in enumerate(open(f"{TROOT}/wnids.txt")): id_dict[line.strip()]=i

def val_loader(crop, bs=128):
    tfs=[transforms.CenterCrop(crop),transforms.ToTensor(),NORM] if crop<64 else [transforms.ToTensor(),NORM]
    ds=TinyImageNetDataset(root=TROOT, id=id_dict, transform=transforms.Compose(tfs), train=False)
    return torch.utils.data.DataLoader(ds,bs,shuffle=False,num_workers=6)
def top1(m, ld, lim=None):
    m.eval().to(DEV); c=t=0
    with torch.no_grad():
        for i,(x,y) in enumerate(ld):
            x,y=x.to(DEV),y.to(DEV); c+=(m(x).argmax(1)==y).sum().item(); t+=y.numel()
            if lim and i>=lim: break
    return 100*c/t
def load_sd(ck):
    sd=torch.load(ck,map_location="cpu",weights_only=False)
    return sd["state_dict"] if isinstance(sd,dict) and "state_dict" in sd else sd
def build_vgg(ck):
    m=_vgg(VGG_CFG,200,512); m.load_state_dict(load_sd(ck),strict=True); return m
def build_resnet(ck):
    m=resnet32_c(200); m.load_state_dict(load_sd(ck),strict=True); return m
def set_codebook(model, levels):
    lv=levels.to(DEV)
    for mod in model.modules():
        if isinstance(mod,(BS.QuantLinearFP,BS.QuantConv2dFP,BS.QuantConv1dFP)): mod.gf4_levels=lv
    BS.set_act_quant(model,"gf4_adaptive")

def find(pat):
    g=sorted(glob.glob(pat)); return g[0] if g else None
JOBS=[
  ("VGG-19","tinyimagenet", build_vgg,   find("Best_Results_HPO/VGG-19/TinyImageNet/*vgg19.pth")),
  ("ResNet-32","tinyimagenet", build_resnet, find("Best_Results_HPO/ResNet32/TinyImageNet/*ResNet_32.pth")),
]
rows=[]
for name, ds, build, ck in JOBS:
    try:
        assert ck, f"no checkpoint for {name}"
        t0=time.time(); model=build(ck).to(DEV).eval()
        # auto-detect crop (56 vs 64) by partial baseline accuracy
        best=None
        for crop in (64,56):
            a=top1(model, val_loader(crop), lim=15)
            print(f"  {name}: crop {crop} partial-acc {a:.2f}%", flush=True)
            if best is None or a>best[1]: best=(crop,a)
        crop=best[0]; ld=val_loader(crop); base=top1(model, ld)
        print(f"  {name}: using crop {crop}, FULL baseline {base:.2f}% -> quantizing", flush=True)
        qm=quantize_model_fp(copy.deepcopy(model), ld, block_size=BLK, e_bits=2, m_bits=1,
                             e_bits_scale=4, m_bits_scale=3, device=DEV,
                             use_HG=False, use_Hessian=False, use_adap=True, use_forward=False, Hadamard=False)
        w4a16=top1(qm, ld)
        set_codebook(qm, BS.GF4_POS); a_gf4=top1(qm, ld)
        set_codebook(qm, NVFP4_POS);  a_nv =top1(qm, ld)
        r=dict(model=name, dataset=ds, crop=crop, fp32=round(base,2), w4a16=round(w4a16,2),
               a4_gf4=round(a_gf4,2), a4_nvfp4_matchedclip=round(a_nv,2),
               codebook_gain=round(a_gf4-a_nv,2), secs=round(time.time()-t0))
        print(f"{name:10s} TinyImageNet  FP32 {base:5.2f}  W4A16 {w4a16:5.2f}  "
              f"A4-GF4 {a_gf4:5.2f}  A4-NVFP4(clip) {a_nv:5.2f}  (codebook {a_gf4-a_nv:+.2f})", flush=True)
    except Exception as e:
        r=dict(model=name, dataset=ds, crop="", fp32="", w4a16="", a4_gf4="",
               a4_nvfp4_matchedclip="ERR: "+repr(e)[:70], codebook_gain="", secs="")
        print(f"{name:10s} TinyImageNet  ERROR: {repr(e)[:90]}", flush=True)
    rows.append(r); r["block_size"]=BLK
OUT="GF4_CV_Study/gf4_vs_nvfp4_tinyimagenet_blk9.csv"
with open(OUT,"w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
print(f"\nsaved {OUT}")

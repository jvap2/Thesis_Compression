"""
V:N:M Phase-1 accuracy prototype (pure PyTorch, no Spatha kernel).

Takes a trained VGG-19 CIFAR-10 checkpoint (the repo's are GSM-pruned ~90% — there is
no dense VGG checkpoint), imposes the V:N:M sparsity PATTERN (fake sparsity via
vnm_sparsity.py), fine-tunes with the mask re-applied each step, and measures how much
accuracy survives the vector-structured constraint. Answers: can our GSM-pruned net be
converted to the hardware-mappable V:N:M form at matched sparsity without losing accuracy?
(The CNN question VENOM never answered — they only did transformers.)

Reports: base(GSM) acc -> masked acc (pre-finetune) -> best acc (post-finetune), + sparsity.
Env-parametrized so we can sweep N:M:tileM. Results appended to vnm_phase1_results.tsv.
"""
import os, time, glob
import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vnm_sparsity import vnm_mask_any

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ── config (env-overridable) ─────────────────────────────────────────────────
N       = int(os.environ.get("VNM_N", "2"))
M       = int(os.environ.get("VNM_M", "16"))       # 2:16 = 87.5% sparse; 2:20 -> 90%
TILEM   = int(os.environ.get("VNM_TILEM", "8"))
EPOCHS  = int(os.environ.get("VNM_EPOCHS", "20"))
LR      = float(os.environ.get("VNM_LR", "0.01"))
CKPT    = os.environ.get("VNM_CKPT", "models/best_2026-06-09_13-46-27_cifar10_vgg19.pth")
NAME    = os.environ.get("VNM_NAME", f"vnm_vgg19_n{N}_m{M}_v{TILEM}")
SKIP_FIRST_LAST = os.environ.get("VNM_SKIP_FIRST_LAST", "0") == "1"  # keep 1st conv + classifier dense
SP = os.environ.get("VNM_SP", ".")
RESULTS = os.path.join(SP, "vnm_phase1_results.tsv")
print(f"[VNM-P1] N={N} M={M} tileM={TILEM} EPOCHS={EPOCHS} LR={LR} "
      f"skip_first_last={SKIP_FIRST_LAST} ckpt={CKPT} name={NAME}", flush=True)

# ── model (matches harvest_channels_vgg.py) ──────────────────────────────────
CFG = [64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']
def make(cfg, in_ch=3):
    layers=[]; c=in_ch
    for v in cfg:
        if v=='M': layers+=[nn.MaxPool2d(2,2)]
        else: layers+=[nn.Conv2d(c,v,3,padding=1,bias=False), nn.BatchNorm2d(v), nn.ReLU(True)]; c=v
    return nn.Sequential(*layers)
class VGG(nn.Module):
    def __init__(s, cfg, nc, last_ch):
        super().__init__(); s.feature=make(cfg); s.avgpool=nn.AdaptiveAvgPool2d((1,1)); s.classifier=nn.Linear(last_ch,nc)
    def forward(s,x): return s.classifier(torch.flatten(s.avgpool(s.feature(x)),1))

# ── data ─────────────────────────────────────────────────────────────────────
means=[0.4914,0.4822,0.4465]; stds=[0.2470,0.2435,0.2616]
norm=transforms.Normalize(means,stds)
train_tf=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),norm])
val_tf=transforms.Compose([transforms.ToTensor(),norm])
train_ds=datasets.CIFAR10("./datasets",train=True,download=True,transform=train_tf)
val_ds  =datasets.CIFAR10("./datasets",train=False,download=True,transform=val_tf)
train_dl=DataLoader(train_ds,128,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True)
val_dl  =DataLoader(val_ds,256,shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True)

@torch.no_grad()
def evaluate(model):
    model.eval(); correct=total=0
    for x,y in val_dl:
        x,y=x.to(DEV),y.to(DEV)
        correct+=(model(x).argmax(1)==y).sum().item(); total+=y.numel()
    return correct/total

def global_sparsity(model):
    nz=tot=0
    for _,p in model.named_parameters():
        if p.dim() in (2,4): nz+=(p.data!=0).sum().item(); tot+=p.numel()
    return 1-nz/tot

# ── load dense checkpoint ────────────────────────────────────────────────────
sd=torch.load(CKPT,map_location='cpu',weights_only=False)
if isinstance(sd,dict) and 'state_dict' in sd: sd=sd['state_dict']
nc=sd['classifier.weight'].shape[0]
model=VGG(CFG,nc,512).to(DEV); model.load_state_dict(sd,strict=True)
base_acc=evaluate(model)
print(f"[VNM-P1] checkpoint baseline acc (GSM-pruned) = {base_acc:.4f}", flush=True)

# ── build + apply V:N:M masks ────────────────────────────────────────────────
prunable=[(n,p) for n,p in model.named_parameters() if p.dim() in (2,4)]
first_conv_name=prunable[0][0]; classifier_name='classifier.weight'
masks={}
with torch.no_grad():
    for n,p in prunable:
        if SKIP_FIRST_LAST and n in (first_conv_name, classifier_name):
            continue
        mask=vnm_mask_any(p.data, N, M, TILEM).to(DEV)
        p.data.mul_(mask); masks[n]=mask
sp_after=global_sparsity(model)
masked_acc=evaluate(model)
print(f"[VNM-P1] after V:N:M mask: sparsity={sp_after:.4f}  acc(pre-finetune)={masked_acc:.4f}", flush=True)

# ── fine-tune with mask re-applied each step ─────────────────────────────────
opt=torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=5e-4,nesterov=True)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
lossf=nn.CrossEntropyLoss()
best=masked_acc
for ep in range(EPOCHS):
    model.train(); t0=time.time()
    for x,y in train_dl:
        x,y=x.to(DEV),y.to(DEV)
        opt.zero_grad(); loss=lossf(model(x),y); loss.backward(); opt.step()
        with torch.no_grad():                    # re-impose the V:N:M mask (straight-through)
            for n,p in model.named_parameters():
                if n in masks: p.data.mul_(masks[n])
    sched.step()
    acc=evaluate(model); best=max(best,acc)
    print(f"[VNM-P1] epoch {ep+1}/{EPOCHS} val={acc:.4f} best={best:.4f} "
          f"({time.time()-t0:.1f}s)", flush=True)

sp_final=global_sparsity(model)
print(f"[VNM-P1] DONE  base(GSM)={base_acc:.4f}  masked_preFT={masked_acc:.4f}  "
      f"best_postFT={best:.4f}  sparsity={sp_final:.4f}", flush=True)
newf=not os.path.exists(RESULTS)
with open(RESULTS,"a") as f:
    if newf: f.write("name\tn\tm\ttileM\tbase_acc\tmasked_preFT\tbest_postFT\tsparsity\tdrop_vs_base\n")
    f.write(f"{NAME}\t{N}\t{M}\t{TILEM}\t{base_acc:.4f}\t{masked_acc:.4f}\t{best:.4f}\t{sp_final:.4f}\t{base_acc-best:.4f}\n")

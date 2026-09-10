"""Train DenseNet-40 (growth 12) on CIFAR-10 for a dense FP32 baseline to GF4-quantize.
Not on chenyaofo, so trained here. ~100 epochs -> ~93% (plenty for a PTQ baseline)."""
import sys, os, time, torch, torch.nn as nn, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0, ROOT); os.chdir(HERE)
DATA=os.path.join(ROOT,"datasets")
from densenet import create_densenet40
from torchvision import datasets, transforms
DEV="cuda" if torch.cuda.is_available() else "cpu"
EPOCHS=int(os.environ.get("EPOCHS","100"))

mean=[0.4914,0.4822,0.4465]; std=[0.2470,0.2435,0.2616]
tr_tf=transforms.Compose([transforms.RandomCrop(32,padding=4), transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(), transforms.Normalize(mean,std)])
te_tf=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])
tr=datasets.CIFAR10(DATA,True,tr_tf,download=False); te=datasets.CIFAR10(DATA,False,te_tf,download=False)
trl=torch.utils.data.DataLoader(tr,128,shuffle=True,num_workers=6,drop_last=False)
tel=torch.utils.data.DataLoader(te,256,shuffle=False,num_workers=4)

m=create_densenet40().to(DEV)
opt=torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
def evalacc():
    m.eval(); c=t=0
    with torch.no_grad():
        for x,y in tel:
            x,y=x.to(DEV),y.to(DEV); c+=(m(x).argmax(1)==y).sum().item(); t+=y.numel()
    return 100*c/t
best=0.0; t0=time.time()
for ep in range(EPOCHS):
    m.train()
    for x,y in trl:
        x,y=x.to(DEV),y.to(DEV); opt.zero_grad(); F.cross_entropy(m(x),y).backward(); opt.step()
    sch.step()
    if ep%5==0 or ep==EPOCHS-1:
        a=evalacc(); best=max(best,a)
        print(f"epoch {ep:3d}  test {a:.2f}%  best {best:.2f}%  ({(time.time()-t0)/60:.1f} min)", flush=True)
        torch.save(m.state_dict(), "densenet40_cifar10.pth")
print(f"DONE  final best {best:.2f}%  saved densenet40_cifar10.pth", flush=True)

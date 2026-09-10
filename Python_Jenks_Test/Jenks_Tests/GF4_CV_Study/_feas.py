import sys, os, copy, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from FP_Quantization_Experiments import quantize_model_fp
from torchvision import datasets, transforms
DEV="cuda" if torch.cuda.is_available() else "cpu"
model=torch.hub.load('chenyaofo/pytorch-cifar-models','cifar10_resnet32',pretrained=True,trust_repo=True).to(DEV).eval()
norm=transforms.Normalize([0.4914,0.4822,0.4465],[0.2470,0.2435,0.2616])
test=datasets.CIFAR10("datasets",train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),norm]))
loader=torch.utils.data.DataLoader(test,batch_size=256,shuffle=False,num_workers=4)
def top1(m):
    m.eval(); c=t=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(DEV),y.to(DEV); c+=(m(x).argmax(1)==y).sum().item(); t+=y.numel()
    return 100*c/t
base=top1(model); print(f"baseline top1: {base:.2f}%")
qm=quantize_model_fp(copy.deepcopy(model),loader,block_size=16,e_bits=2,m_bits=1,e_bits_scale=4,m_bits_scale=3,device=DEV,use_HG=False,use_Hessian=False,use_adap=True,use_forward=False,Hadamard=False)
g=top1(qm); print(f"GF4 top1: {g:.2f}%   drop: {base-g:.2f}%")

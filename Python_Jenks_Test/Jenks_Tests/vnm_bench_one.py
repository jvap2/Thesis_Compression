"""Benchmark ONE conv shape: dense (cuDNN fp16) vs Conv2dVNM. Isolated per-process so a
CUDA fault in one shape can't abort the sweep. Args: in out H k batch tileM n m"""
import sys, os, torch, torch.nn as nn
sys.path.insert(0,"/home/jvap2/Desktop/Code/TIME_ML/Python_Jenks_Test/Jenks_Tests")
os.environ.setdefault("VENOM_DIR","/tmp/venom_probe/end2end")
from conv2d_vnm import Conv2dVNM, build_spatha
from torch.utils.cpp_extension import load
inc,outc,H,k,batch,tileM,n,m = [int(a) for a in sys.argv[1:9]]
dev="cuda"; torch.backends.cudnn.benchmark=True; torch.manual_seed(0)
BS="/tmp/venom_probe/end2end/spatha_mod/block_sparse"
if tileM==64:
    spatha=load(name="spatha_v64",sources=[f"{BS}/api/spatha.cu"],
        extra_include_paths=[BS,f"{BS}/api",f"{BS}/spmm",f"{BS}/common"],
        extra_cuda_cflags=["-arch=sm_89","-DV_64","-std=c++17"],verbose=False)
else:
    spatha=build_spatha()
def t_ms(fn,x,it=60,warm=20):
    for _ in range(warm): fn(x)
    torch.cuda.synchronize(); a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
    for _ in range(it): fn(x)
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/it
pad=((outc+tileM-1)//tileM)*tileM
x=torch.randn(batch,inc,H,H,device=dev,dtype=torch.half)
dense=nn.Conv2d(inc,outc,k,padding=k//2,bias=True).to(dev).half()
td=t_ms(lambda z: dense(z), x)
convp=nn.Conv2d(inc,pad,k,padding=k//2,bias=True).to(dev)
vnm=Conv2dVNM(convp,n,m,tileM,spatha,dev=dev)
tv=t_ms(lambda z: vnm(z), x)
padnote = "" if pad==outc else f"pad{outc}->{pad}"
print(f"RESULT\t{inc}->{outc}\t{H}x{H}\tk{k}\tB{batch}\tv{tileM}\t{td:.3f}\t{tv:.3f}\t{td/tv:.2f}\t{padnote}")

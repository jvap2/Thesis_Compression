"""Builds JORTsE_ImageNet_Colab.ipynb — pretrained ResNet-50 / VGG-16-BN + short JORTsE
(warmup -> auto-Jenks -> prune -> EMA) on HuggingFace ImageNet-1k, for Colab Pro+ A100.
Run:  python build_jortse_imagenet_notebook.py   (produces the .ipynb next to it)"""
import json, os

def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
def code(s): return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                     "source": s.strip("\n").splitlines(keepends=True)}

cells = [
md("""# JORTsE on ImageNet — pretrained ResNet-50 / VGG-16-BN (Colab Pro+ A100)

Short **JORTsE** run (warmup → auto-Jenks LR/WD/momentum → prune → EMA finetune) starting from
**torchvision pretrained** weights, on **HuggingFace `imagenet-1k`**. Pretrained init matches how
GSM (our baseline) ran ImageNet, so the comparison is fair.

**Runtime:** set to **A100** + **High-RAM**, and turn on **background execution** (Pro+).
Data streams/loads from HF (your token); every epoch's best + best-EMA checkpoint is saved to Drive.
"""),

code("""
# 1) Mount Drive, clone the JORTsE repo, install deps
from google.colab import drive; drive.mount('/content/drive')
import os, subprocess, sys
REPO = "/content/Thesis_Compression"
if not os.path.exists(REPO):
    !git clone --depth 1 https://github.com/jvap2/Thesis_Compression.git {REPO}
JENKS = f"{REPO}/Python_Jenks_Test/Jenks_Tests"
os.chdir(JENKS); sys.path.insert(0, JENKS)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")   # A100
!pip -q install backpack-for-pytorch torchmetrics datasets huggingface_hub pillow jenkspy pynvml ninja
print("cwd:", os.getcwd())
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
print("\n--- DISK MOUNTS (find the ~368 GB scratch mount, set SCRATCH_DIR to it in cell 2) ---")
!df -h | grep -vE "tmpfs|udev"
"""),

code("""
# 2) CONFIG  — SHORTENED JORTsE schedule, but the SAME 4-stage FLOW as DenseNet40_CIFAR10_rewind_ema.py:
#    warmup -> auto-Jenks LR/WD/momentum -> ITERATIVE prune (@0.4*EPOCHS) -> EMA re-anneal.
#    (Starts from PRETRAINED weights, so far fewer epochs are needed than CIFAR-from-scratch's 400.)
HF_TOKEN     = ""          # <-- paste your HuggingFace token (needs imagenet-1k license accepted)
ARCH         = "resnet50"  # "resnet50" | "vgg16"
# GSM (NeurIPS'19, Table 3) ResNet-50 targets: 75% sparsity (4x, top-1 75.33) / 80% (5x, 74.30).
PRUNE_RATIO  = 0.75        # match GSM's headline ResNet-50 point (4x). Stretch: 0.80 (5x).
EPOCHS       = 40          # SHORT + pretrained (GSM used 60 from a trained base; 40 fits one A100 session)
WARMUP_EPOCHS= 3           # short warmup (DenseNet40 was 10 for a 400-ep from-scratch run)
PRUNE_EPOCH  = int(0.4 * EPOCHS)   # =16 @40 — same 0.4*EPOCHS prune-start ratio as DenseNet40
ONE_SHOT     = False       # ITERATIVE prune to target (the DenseNet40 winning recipe), NOT one-shot
PRUNE_BETWEEN= 2           # iterative prune step spacing (DenseNet40 used 5 over 400 ep -> 2 here)
# LR-decay milestones scaled from DenseNet40's [25,50,75,125,200,300,400,500] by EPOCHS/400
GSM_LR_BOUNDARIES = sorted({max(WARMUP_EPOCHS+1, round(b*EPOCHS/400)) for b in [25,50,75,125,200,300,400,500]})
BATCH_SIZE   = 256         # A100 (GSM used 64); auto-Jenks warmup absorbs the larger-batch LR
import os
NUM_WORKERS  = os.cpu_count() or 8   # use ALL Colab vCPUs — CPU JPEG-decode is the bottleneck
STREAMING    = False       # False = download+cache to local disk (faster epochs, needs disk);
                           # True  = stream from HF (no disk, slower)
# ImageNet-1k parquet is ~155 GB. HF caches to the BOOT disk by default (won't use scratch on its own),
# so point the cache at your big scratch mount. Check mounts first with:  !df -h
SCRATCH_DIR  = "/content/scratch"   # <-- SET to the 368 GB scratch mount from `df -h`
import os
os.environ["HF_HOME"]          = f"{SCRATCH_DIR}/hf"
os.environ["HF_DATASETS_CACHE"] = f"{SCRATCH_DIR}/hf/datasets"
os.environ["HF_HUB_CACHE"]      = f"{SCRATCH_DIR}/hf/hub"
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
CACHE_DIR = os.environ["HF_DATASETS_CACHE"]
# Seeds. lr/wd matter as auto-Jenks starting points; MOMENTUM IS AUTO-DERIVED per-parameter
# (ElementwiseMomentumSGD uses sal_beta=(1-sqrt(lr*||grad||^2))^2), so MOM is inert — value doesn't matter.
LR, WD, MOM  = 1e-3, 1e-4, 0.9   # lr 1e-3 (GSM ResNet-50 base), wd 1e-4; MOM unused (auto momentum)
LABEL_SMOOTH = 0.1
DRIVE_DIR    = "/content/drive/MyDrive/JORTsE_ImageNet"
import os; os.makedirs(DRIVE_DIR, exist_ok=True)
# DATA SOURCE toggle (one line): "hf" works out of the box; "wds"/"dali" use the 256px shards
# built once by ImageNet256_Prep_and_DALI.ipynb (fast + no 155GB re-download per session).
#   hf   = HuggingFace + PIL full-res decode (~370 img/s)
#   wds  = 256px WebDataset shards, cheap CPU decode (~3-5x faster)
#   dali = 256px shards + DALI nvJPEG GPU decode (fastest; sanity-check labels once, see prep notebook)
DATA_SOURCE = "hf"
SHARD_DRIVE = f"{DRIVE_DIR}/imagenet256_wds"     # persistent shards on Drive (from the prep notebook)
SHARD_LOCAL = f"{SCRATCH_DIR}/imagenet256_wds"   # fast working copy staged onto scratch
NUM_CLASSES  = 1000
assert HF_TOKEN, "Paste your HuggingFace token in HF_TOKEN"
# WALL-CLOCK: ~5-8 min/epoch on an A100 => 40 epochs ~ 4-5 h. warmup/prune-start/milestones
# auto-scale with EPOCHS, so the DenseNet40 flow shape holds if you change it.
"""),

code("""
# 3) Data loaders — DATA_SOURCE (cell 2) toggles: "hf" | "wds" | "dali". Yields (img[B,3,224,224], label[B] int64).
import torch, warnings, math, glob, shutil, subprocess, sys
from torchvision import transforms
from tqdm.auto import tqdm
from PIL import ImageFile
warnings.filterwarnings("ignore", message="Corrupt EXIF data")
ImageFile.LOAD_TRUNCATED_IMAGES = True
def _pip(pkgs): subprocess.run([sys.executable,"-m","pip","install","-q",*pkgs.split()], check=False)
_extra = dict(persistent_workers=(NUM_WORKERS > 0), prefetch_factor=4) if NUM_WORKERS > 0 else {}
norm  = transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
tf_tr = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(), norm])
N_TRAIN, N_VAL = 1281167, 50000

class TqdmLoader:                        # transparent wrapper: live per-batch bar inside the repo's loop
    def __init__(self, loader, desc): self.loader, self.desc = loader, desc
    def __iter__(self): return iter(tqdm(self.loader, desc=self.desc, leave=False, dynamic_ncols=True))
    def __len__(self):  return len(self.loader)
    def __getattr__(self, n): return getattr(self.loader, n)

def _stage_shards():                     # Drive -> scratch (fast local reads); asserts prep was run
    import os; os.makedirs(SHARD_LOCAL, exist_ok=True)
    for t in glob.glob(f"{SHARD_DRIVE}/*.tar"):
        d = f"{SHARD_LOCAL}/{os.path.basename(t)}"
        if not os.path.exists(d): shutil.copy2(t, d)
    tr = sorted(glob.glob(f"{SHARD_LOCAL}/train-*.tar")); va = sorted(glob.glob(f"{SHARD_LOCAL}/validation-*.tar"))
    assert tr and va, f"no shards in {SHARD_DRIVE} — run ImageNet256_Prep_and_DALI.ipynb (Prep A) first"
    return tr, va

if DATA_SOURCE == "hf":
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from huggingface_hub import login; login(HF_TOKEN)
    tf_val = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm])
    HF_DATASET = "ILSVRC/imagenet-1k"
    ds_tr  = load_dataset(HF_DATASET, split="train",      token=HF_TOKEN, streaming=STREAMING, cache_dir=CACHE_DIR)
    ds_val = load_dataset(HF_DATASET, split="validation", token=HF_TOKEN, streaming=STREAMING, cache_dir=CACHE_DIR)
    if STREAMING: ds_tr = ds_tr.shuffle(seed=0, buffer_size=10000)
    def _col(batch, tf):
        xs = torch.stack([tf(b["image"].convert("RGB")) for b in batch])
        ys = torch.tensor([b["label"] for b in batch]); return xs, ys
    train_dataloader = TqdmLoader(DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=(not STREAMING),
                        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=lambda b:_col(b,tf_tr), **_extra), "train")
    val_dataloader   = TqdmLoader(DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=lambda b:_col(b,tf_val), **_extra), "val")

elif DATA_SOURCE == "wds":
    _pip("webdataset"); import webdataset as wds
    from torch.utils.data import DataLoader
    tr_sh, va_sh = _stage_shards()
    tf_val = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), norm])   # shards already 256px
    def _mk(sh, tf, train):
        ds = wds.WebDataset(sh, shardshuffle=train, nodesplitter=wds.split_by_node)
        if train: ds = ds.shuffle(2000)
        return ds.decode("pil").to_tuple("jpg","cls").map_tuple(tf, lambda c: int(c))
    def _col(b):
        xs = torch.stack([x for x,_ in b]); ys = torch.tensor([y for _,y in b]); return xs, ys
    class _WL:
        def __init__(self, ds, n, desc): self.ds, self.n, self.desc = ds, n, desc
        def __len__(self): return math.ceil(self.n / BATCH_SIZE)
        def __iter__(self):
            dl = DataLoader(self.ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=_col,
                            pin_memory=True, prefetch_factor=4)
            return iter(tqdm(dl, total=len(self), desc=self.desc, leave=False, dynamic_ncols=True))
    train_dataloader = _WL(_mk(tr_sh, tf_tr, True),  N_TRAIN, "train")
    val_dataloader   = _WL(_mk(va_sh, tf_val, False), N_VAL,  "val")

elif DATA_SOURCE == "dali":
    _pip("nvidia-dali-cuda120")
    from nvidia.dali import pipeline_def, fn, types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    tr_sh, va_sh = _stage_shards()
    MEAN=[0.485*255,0.456*255,0.406*255]; STD=[0.229*255,0.224*255,0.225*255]
    @pipeline_def
    def _pipe(shards, train):
        img, cls = fn.readers.webdataset(paths=shards, ext=["jpg","cls"], missing_component_behavior="error",
                                         random_shuffle=train, name="Reader", pad_last_batch=True)
        im = fn.decoders.image(img, device="mixed", output_type=types.RGB)   # nvJPEG GPU decode
        if train:
            im = fn.random_resized_crop(im, size=[224,224], device="gpu"); mir = fn.random.coin_flip(probability=0.5)
        else:
            im = fn.resize(im, resize_shorter=256, device="gpu"); im = fn.crop(im, crop=[224,224], device="gpu"); mir = 0
        im = fn.crop_mirror_normalize(im, dtype=types.FLOAT, output_layout="CHW", mean=MEAN, std=STD, mirror=mir)
        cls = fn.pad(cls, fill_value=0)   # ragged ascii label bytes -> dense [B, Lmax] so as_tensor() works
        return im, cls
    class _DL:
        def __init__(self, sh, n, train, desc):
            self.n, self.desc = n, desc
            p = _pipe(sh, train, batch_size=BATCH_SIZE, num_threads=8, device_id=0); p.build()
            self.it = DALIGenericIterator(p, ["data","lb"], reader_name="Reader",
                        last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)
        def __len__(self): return math.ceil(self.n / BATCH_SIZE)
        def __iter__(self):
            for b in tqdm(self.it, total=len(self), desc=self.desc, leave=False, dynamic_ncols=True):
                x = b[0]["data"]; raw = b[0]["lb"].cpu().numpy()
                ys = [int(bytes(r[r != 0]).decode() or 0) for r in raw]
                yield x, torch.tensor(ys, device=x.device, dtype=torch.long)
    train_dataloader = _DL(tr_sh, N_TRAIN, True,  "train")
    val_dataloader   = _DL(va_sh, N_VAL,  False, "val")
else:
    raise ValueError(f"DATA_SOURCE must be 'hf' | 'wds' | 'dali', got {DATA_SOURCE!r}")

print(f"loaders ready: source={DATA_SOURCE}, workers={NUM_WORKERS}, batches/epoch~{len(train_dataloader)}")
"""),

code("""
# 4) Pretrained model + JORTsE prune flags
import torch, torch.nn as nn
device = "cuda"
if ARCH == "resnet50":
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
else:
    from torchvision.models import vgg16_bn, VGG16_BN_Weights
    model = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
for _, mmod in model.named_modules():          # mark prunable layers (Conv2d/Linear)
    mmod.do_prune = isinstance(mmod, (nn.Conv2d, nn.Linear))
model = model.to(device)
total_prunable = sum(p.numel() for p in model.parameters() if p.dim() in (2, 4))
print(f"{ARCH}: {total_prunable/1e6:.1f}M prunable params")
"""),

code("""
# 5) Optimizer (auto LR/WD/momentum) + WarmupAutoJenks scheduler + metrics  (mirrors ResNet50_ImageNet.py)
from custom_schedulers import init_lr_weight_decay, WarmupAutoJenks
from custom_optimizer import init_network
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

loss_fn   = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
optimizer = init_lr_weight_decay(model, LR, WD, bias_weight_decay=WD, momentum=MOM, nestrov=False,
                                 bias_lr=True, elem_bias=True, warmup_epochs=WARMUP_EPOCHS, prune_epoch=PRUNE_EPOCH)
init_network(optimizer)
# DenseNet40 scheduler: WarmupAutoJenks, warmup_factor=1/2, reset=False (no rewind on iterative path)
gsm_lr_boundaries = GSM_LR_BOUNDARIES
scheduler = WarmupAutoJenks(optimizer, milestones=gsm_lr_boundaries, warmup_factor=1/2,
                            warmup_iters=WARMUP_EPOCHS, prune_epochs=PRUNE_EPOCH, reset=False)
accuracy    = Accuracy(task='multiclass', num_classes=NUM_CLASSES).to(device)
top5accuracy= MulticlassAccuracy(num_classes=NUM_CLASSES, top_k=5).to(device)
writer = SummaryWriter(log_dir=os.path.join(DRIVE_DIR, "tb"))
ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
def L(tag): return os.path.join(DRIVE_DIR, f"{ARCH}_{tag}_{ts}.txt")
print("optimizer + scheduler ready; milestones", gsm_lr_boundaries)
"""),

code("""
# 5b) FAST Jenks: replace the O(N^2) per-break CUDA kernel with an O(N) prefix-sum version.
# The original launches one thread per weight and each thread re-scans the whole array (O(N^2),
# uncoalesced) — the JORTs bottleneck once pruning starts. This computes the identical per-break
# within-group variance via cumulative sums; VALIDATED to give the same argmin/break (fp32 & fp64).
# Rebinds the names inside custom_optimizer so every call site uses it (no repo edit, no recompile).
import custom_optimizer, torch
def _jenks_var_fast(w_sorted):
    w = w_sorted.contiguous(); N = w.numel()
    if N < 2: return torch.zeros(N, device=w.device, dtype=w.dtype)
    S = torch.cumsum(w, 0); Q = torch.cumsum(w*w, 0)
    Stot, Qtot = S[-1], Q[-1]
    s1 = torch.empty_like(w); q1 = torch.empty_like(w)
    s1[0] = 0; q1[0] = 0; s1[1:] = S[:-1]; q1[1:] = Q[:-1]
    idx = torch.arange(N, device=w.device, dtype=w.dtype)
    n1 = idx.clamp(min=1); n2 = (N - idx).clamp(min=1)
    s2 = Stot - s1; q2 = Qtot - q1
    var1 = q1 - s1*s1/n1; var1[0] = 0
    var2 = q2 - s2*s2/n2
    return var1 + var2
class _FastJenksW: jenks_optimization_cuda        = staticmethod(_jenks_var_fast)
class _FastJenksB: jenks_optimization_biases_cuda = staticmethod(_jenks_var_fast)
custom_optimizer.module_weights = _FastJenksW()
custom_optimizer.module_bias    = _FastJenksB()
print("Jenks break: O(N^2) CUDA kernel -> O(N) prefix-sum (rebound in custom_optimizer)")

# The training loop calls torch.cuda.empty_cache() EVERY batch (custom_optimizer ~L2351), which
# releases cached GPU memory and forces a cudaMalloc + sync each step -> big per-step stall.
# On an 80GB A100 it's unnecessary; make it a no-op so the GPU isn't bubble-stalled every step.
torch.cuda.empty_cache = lambda: None
print("per-step empty_cache() neutralized (removes per-batch GPU stall)")

# bf16 autocast on the FORWARD only: runs conv/matmul on the A100 tensor cores (speedup) while
# weights, the elementwise optimizer, BatchNorm, and Jenks all stay fp32 (stability). Pure fp16
# (model.half) would destabilize BN, the sal_beta momentum, and the Jenks cumsum -> wrong masks.
_orig_forward = model.forward
def _bf16_forward(*a, **k):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return _orig_forward(*a, **k)
model.forward = _bf16_forward
print("bf16 autocast on forward (tensor cores); weights/optimizer/Jenks stay fp32")
"""),

code("""
# 6) Run JORTsE  (warmup -> auto-Jenks -> prune @PRUNE_EPOCH -> EMA re-anneal). Checkpoints -> Drive.
from training_loop import train_val_loop_HPO, train_val_loop_GVF
os.chdir(DRIVE_DIR)   # best_*.pth / best_ema_*.pth are written to cwd
# training_loop writes to a couple of hardcoded RELATIVE paths (checkpoints + a legacy log);
# create them under DRIVE_DIR so the per-epoch writes don't crash.
os.makedirs("models", exist_ok=True)
os.makedirs("LeNet300_100_MNIST_output", exist_ok=True)
EXPERIMENT_NAME = f"jortse_imagenet_{ARCH}_sp{int(PRUNE_RATIO*100)}"

# ============================ GVF-GATED ABLATION LEVERS ============================
# Set GVF_GATED=True to run the overnight experiment: every epoch (from GVF_CHECK_FROM, incl.
# before training) each prunable layer's MAGNITUDE Jenks GVF is measured; any layer whose GVF
# meets the criterion is pruned by magnitude and FROZEN (mask fixed forever), optimistically
# forcing the remaining layers to compensate. Once global sparsity >= PRUNE_RATIO, pruning stops,
# LR/WD rewinds, and EMA runs the recovery tail. Per-layer (epoch,layer,gvf,keep,pruned,frozen)
# is logged to the *_gvf_*.csv so you can plot GVF trajectories and pick the criterion.
GVF_GATED      = True     # <-- turn the ablation on
GVF_THRESH     = 0.64     # GVF criterion; ~0.64 = unimodal-Gaussian baseline. Lower => prune more
                          #     layers / reach target faster; raise => stricter. Ignored if adaptive.
GVF_ADAPTIVE   = True     # if True: ignore GVF_THRESH; each epoch prune the HIGH-GVF group
                          #     (meta-Jenks on the per-layer GVF vector) for guaranteed progressive
                          #     staggering -> reaches target + auto-protects low-GVF fc/downsamples
GVF_CHECK_FROM = 0        # first epoch to check (0 = even before any training)
GVF_EMA_FAILSAFE = True   # if target sparsity is never reached, force-freeze late (EPOCHS-delay-3)
                          #     so EMA still seeds + harvests a tail (guarantees a usable EMA result)
# ==================================================================================

if GVF_GATED:
    train_val_loop_GVF(
        model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler, accuracy, top5accuracy,
        writer, device, EXPERIMENT_NAME, ARCH, ts,
        train_filename=L("train"), val_filename=L("val"), log_filename=L("log"),
        sparsity_filename=L("sparsity"), prune_filename=L("prune"),
        gvf_csv=L("gvf").replace(".txt", ".csv"),
        EPOCHS=EPOCHS, target_sparsity=PRUNE_RATIO, gvf_thresh=GVF_THRESH, gvf_adaptive=GVF_ADAPTIVE,
        use_ema=True, ema_decay=0.95, ema_delay=5, rewind_at_freeze=True,
        check_from_epoch=GVF_CHECK_FROM, use_bf16=False,   # model.forward already autocasts (cell 5b)
        ema_failsafe=GVF_EMA_FAILSAFE, ema_failsafe_tail=3)
else:
    train_val_loop_HPO(
        model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler, accuracy, top5accuracy,
        writer, device,
        EXPERIMENT_NAME, ARCH, ts,          # experiment_name, model_name, timestamp (positional)
        train_filename=L("train"), val_filename=L("val"), log_filename=L("log"),
        sparsity_filename=L("sparsity"), prune_filename=L("prune"), debug_filename=L("debug"),
        jenks_filename=L("jenks"),
        prune_count=0, one_update=True, EPOCHS=EPOCHS, sparsity=0.0,
        prune_epoch_list=[PRUNE_EPOCH, PRUNE_EPOCH + 100], prune_epoch=PRUNE_EPOCH, prune_between=PRUNE_BETWEEN,
        prune_ratio=PRUNE_RATIO, one_shot=ONE_SHOT, mask=True,
        mag_prune=True, bias_prune=False, kill_velocity=False,
        l2=False, lambda_=0, warmup_epochs=WARMUP_EPOCHS, min_epochs=EPOCHS, elem_bias=True,
        use_ema=True, ema_decay=0.95,
        rewind_at_freeze=True)   # warm-restart LR/WD to init at the (one-shot-like) prune so the sparse
                                 # net recovers at ~peak LR instead of the decayed ~5e-4 that plateaus it
print("JORTsE done. checkpoints + logs in", DRIVE_DIR)
"""),

code("""
# 7) Final eval — load best EMA checkpoint, report top-1/top-5 + achieved sparsity
import glob, torch
ckpts = sorted(glob.glob(os.path.join(DRIVE_DIR, "best_ema_*.pth")) or
               glob.glob(os.path.join(DRIVE_DIR, "best_*.pth")), key=os.path.getmtime)
assert ckpts, "no checkpoint found"
model.load_state_dict(torch.load(ckpts[-1], map_location=device)); model.eval()
nz = sum((p != 0).sum().item() for p in model.parameters() if p.dim() in (2,4))
tot= sum(p.numel() for p in model.parameters() if p.dim() in (2,4))
c = t = c5 = 0
with torch.no_grad():
    for x, y in val_dataloader:
        x, y = x.to(device), y.to(device); out = model(x)
        c  += (out.argmax(1) == y).sum().item()
        c5 += (out.topk(5,1).indices == y[:,None]).any(1).sum().item(); t += y.numel()
print(f"[{ARCH}]  top1={100*c/t:.2f}%  top5={100*c5/t:.2f}%  sparsity={1-nz/tot:.3f}  ({ckpts[-1]})")
"""),

md("""## Tuning / notes
- **Fastest first result:** `STREAMING=True` runs with zero disk setup (slower epochs). For the fast lane set
  `STREAMING=False` (downloads+caches ImageNet-1k to the A100 local disk, ~5–8 min/epoch).
- **If the A100 starves** (low GPU util): raise `NUM_WORKERS`, or switch the loader to NVIDIA DALI / FFCV.
- **Sparsity sweep:** the GSM ResNet-50 comparison points are `PRUNE_RATIO` = **0.75** (4×, GSM 75.33 top-1) and
  **0.80** (5×, GSM 74.30); each is one JORTsE run (auto-Jenks = no HPO sweep).
- **VGG-16:** set `ARCH="vgg16"`; heavier classifier, drop `BATCH_SIZE` if OOM.
- **Resuming after a disconnect:** re-run cells 1–5, then load the latest Drive checkpoint before cell 6, or just relaunch.
- **Hyperparameter seeds:** `lr=1e-3` (GSM's ResNet-50 base lr) and `wd=1e-4` seed the auto-Jenks LR/WD schedule.
  **Momentum is automated** — `ElementwiseMomentumSGD` derives it per-parameter from local curvature
  (`sal_beta=(1-sqrt(lr*||grad||^2))^2`), so the `MOM` seed is inert and its value doesn't matter.
- GSM's own budget was 60 epochs (lr 1e-3/1e-4/1e-5 × 40/10/10) from a trained base; `EPOCHS=40` is the fast lane,
  bump toward 60 for a strict head-to-head.
"""),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU",
      "colab": {"provenance": [], "machine_shape": "hm"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "JORTsE_ImageNet_Colab.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, "(%d cells)" % len(cells))

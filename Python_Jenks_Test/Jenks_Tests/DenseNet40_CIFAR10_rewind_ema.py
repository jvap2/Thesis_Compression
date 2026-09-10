from models import ResNet56
import torch
torch.set_float32_matmul_precision('highest')
from custom_optimizer import JenksSGD,PruneWeights, JenksSGD_Noise, SAM, JenksSGD_Test, PruneWeights_Test, train_one_step, Prune_Score_Mag
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy
from datetime import datetime
import os
import torch.nn as nn
from networks import LeNet5V1,alexnet,lenet5v1
from torch.autograd.functional import hessian
from functions import hutchinson_trace_hmp,rademacher

from functions import exact_trace
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from time import time
from cuda_helpers import get_memory_free_MiB
from custom_optimizer import Prune_Score,train_one_step_prune,Prune_Score_Select, train_one_step_prune_v2, init_network, Prune_Score_v2
from custom_schedulers import WarmupMultiStepLR, init_lr_weight_decay,WarmupMultiStepJenks, WarmupAutoJenks, WarmupAutoSGDJenks
from densenet import create_densenet40
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy
from datetime import datetime
import os
import torch.nn as nn
# from networks import LeNet5V1,alexnet,lenet5v1
from torch.autograd.functional import hessian
# from functions import hutchinson_trace_hmp,rademacher
from torch.autograd import profiler as prof
from torch import compile
from training_loop import train_val_loop, train_val_loop_HPO, train_val_loopETF
from models import LeNet5V1ETFModel, Args,densenet40ETF, AGD_init_weights

print(torch.cuda.is_available())

one_shot = False        # ITERATIVE to prune_ratio (the ResNet32@95 WINNING recipe): gentle multi-step
prune_ratio = .9        # @90% target; iterative stop + late-EMA freeze trigger
# --- winning recipe ported from ResNet32@95 (plain 91.48 / EMA 92.95, 0 NaN) ---
import custom_optimizer, cuda_helpers
custom_optimizer.GC_ENABLED = False      # GC OFF: at high sparsity its per-filter zero-sum starves the
                                         # surviving weights (mask-aware GC held vs NaN but capped acc).
custom_optimizer.GRAD_CLIP_NORM = 5.0    # clip gradient MAGNITUDE to kill the frozen-tail NaN blow-up.
cuda_helpers.OVER_PRUNE = 0.0            # iterative Jenks compounds to prune_ratio (no per-step over-prune)
torch.cuda.empty_cache()
# train_val_dataset = datasets.MNIST(root="./datasets/", train=True, download=True)
# test_dataset = datasets.MNIST(root="./datasets/", train=False, download=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

kill_velocity = True
train_lr_decay_factor = 0.25

gsm_lr_base_value = 1e-2
gsm_lr_boundaries = [25, 50, 75, 125, 200, 300, 400, 500]
gsm_momentum = 0.99
gsm_max_epochs = 280
mask = True

CIFAR10_PATH = "./datasets"
means = [0.4918687901200927, 0.49185976472299225, 0.4918583862227116]
stds  = [0.24697121702736, 0.24696766978537033, 0.2469719877121087]


normalize_1 = transforms.Normalize((.5, .5, .5), (.25, .25, .25))
normalize_4 = transforms.Normalize(
    mean=means,
    std=stds,
)
# train_val_dataset = datasets.CIFAR10(root="./datasets/", train=True, download=True, transform=transforms.ToTensor())
train_dataset = datasets.CIFAR10(root = CIFAR10_PATH, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.Pad(padding=(4, 4, 4, 4)),
                                   transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                                   transforms.RandomCrop(32),
                                   transforms.RandomHorizontalFlip(),
                                #    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                #    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                                   transforms.ToTensor(),
                                   # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   normalize_4
                               ]))
# train_dataset = datasets.CIFAR10(root = CIFAR10_PATH, train=True, download=True,
#                                 transform=transforms.Compose([
#                                 transforms.Pad(padding=(4, 4, 4, 4)),
#                                 transforms.RandomCrop(32),
#                                 transforms.RandomHorizontalFlip(),
#                                 transforms.ToTensor(),
#                                 # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                                 normalize_4
#                             ]))
fin_val_dataset = datasets.CIFAR10(CIFAR10_PATH, train=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    normalize_4
                                ]))





BATCH_SIZE = 128
AGD = True
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset=fin_val_dataset, batch_size=BATCH_SIZE, shuffle=True)
# model_lenet5v1 = LeNet5V1()
decl_ETF = False
if decl_ETF:
    model = densenet40ETF(decl_ETF=True, args=Args(inference=False), device=device).to(device)
else:
    model = create_densenet40()

if AGD:
    AGD_init_weights(model)
# model = torch.compile(model, mode="reduce-overhead", backend="inductor")
model = model.to(device)
print(model)  
min_epochs = 400
label_smoothing = 0.0
loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
momentum = 0.9
learning_rate = 1e-2
weight_decay = .875e-4
warmup_epochs = 10
warmup_iters = warmup_epochs * len(train_dataloader)
nestrov = True
params = []
gamma = .375
bias_lr = True
bias_wd = 5e-5
prune_epoch = 160   # prune EARLY (0.4*EPOCHS, mirrors ResNet56 200/500) so the rewind has ~240 epochs to
                    # re-anneal, vs the HPO best's late prune@350 that left only 50 (and undershot to 84.5%)
optimizer = init_lr_weight_decay(model, learning_rate, weight_decay,bias_weight_decay=bias_wd, momentum=momentum, nestrov=nestrov, bias_lr=bias_lr, elem_bias = True, warmup_epochs=warmup_epochs, prune_epoch=prune_epoch)
init_network(optimizer)
# scheduler = WarmupMultiStepLR(optimizer, milestones=[80, 120, 140], warmup_factor=0.1, warmup_iters=10, warmup_method="linear")
# scheduler = WarmupMultiStepJenks(optimizer, milestones=gsm_lr_boundaries,gamma=gamma, warmup_factor=0.1, warmup_iters=warmup_epochs, warmup_method="linear", adjustable=False)
# REWIND: reset the auto-Jenks LR/WD accumulators back to INITIAL at the prune epoch
# (a,b,P,M -> init in WarmupAutoJenks.reset_accumulators), so the pruned subnetwork
# re-anneals from a high LR ("fresh training"). Mirrors ResNet56_CIFAR10_protectskip.
scheduler = WarmupAutoJenks(optimizer,milestones=gsm_lr_boundaries, warmup_factor=1/2, warmup_iters=warmup_epochs,
                            prune_epochs=prune_epoch, reset=False)   # NO rewind: it destabilizes the iterative path
accuracy = Accuracy(task='multiclass', num_classes=10)
top5accuracy = MulticlassAccuracy(num_classes=10, top_k=5)
 ## Check the number of parameters in the model vs number of trainiable parameters
bias_prune = False

# Experiment tracking
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_name = "CIFAR10"
model_name = "DenseNet40"
log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
writer = SummaryWriter(log_dir)

# device-agnostic setup
print(f"Using {device} device")
accuracy = accuracy.to(device)
top5accuracy = top5accuracy.to(device)
os.makedirs("models", exist_ok=True)
train_loss, train_acc = 0.0, 0.0
train_top5acc = 0.0
count = 0
original_magnitude = sum(torch.norm(p)**2 for p in model.parameters())
lambda_ = 0


train_dir = "DenseNet40_CIFAR10_output/"
os.makedirs(train_dir, exist_ok=True)  # Create directory if it doesn't exist
name =  "SGD_Agg_REWIND_EMA"
EPOCHS = 400
log_filename = os.path.join(train_dir, f"log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
train_filename = os.path.join(train_dir, f"training_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
trace_filename = os.path.join(train_dir, f"trace_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
sparsity_filename = os.path.join(train_dir, f"sparisty_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
trace_val_filename = os.path.join(train_dir, f"sparisty_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
val_filename = os.path.join(train_dir,f"validation_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
debug_filename = os.path.join(train_dir,f"debug_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
jenks_filename = os.path.join(train_dir,f"jenks_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
prune_filename = os.path.join(train_dir,f"prune_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
master_count = 0
epoch = 0
no_jenks =False
l2 = False
mag_prune = True
prune_between = 10
with open(log_filename,"a") as f:
    print(f"Starting Learning rate: {learning_rate}", file=f)
    print(f"Momentum: {momentum}", file=f)
    print(f"Weight decay: {weight_decay}", file=f)
    print(f"Batch size: {BATCH_SIZE}", file=f)
    print(f"Epochs: {EPOCHS}", file=f)
    print(f"Epoch to start pruning: {prune_epoch}", file=f)
    print(f"Warmup epochs: {warmup_epochs}", file=f)
    if nestrov:
        print(f"Opt type is Nesterov", file=f)
    else:
        print(f"Opt type is Jenks SGD", file=f)
    if kill_velocity:
        print(f"Velocity is killed", file=f)
    else:
        print(f"Velocity is not killed", file=f)
    if mask:
        print(f"Mask is used", file=f)
    else:
        print(f"Mask is not used", file=f)
    if bias_lr:
        print(f"Bias LR is used", file=f)
    else:
        print(f"Bias LR is not used", file=f)
    if no_jenks:
        print(f"No Jenks is used", file=f)
    else:
        print(f"Jenks is used", file=f)
    if mag_prune:
        print(f"Mag prune is used", file=f)
    else:
        print(f"Mag prune is not used", file=f)
prune_count = 0
sparsity = 0.0
one_update = True
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters in the model: {total_params}")
total_pruned_params = sum(p.numel() for p in model.parameters() if p.dim() in [2, 4])
print(f"Total prunebale parameters in the model: {total_pruned_params}")
prune_epoch_list = [prune_epoch, prune_epoch+100]
# Run the training and validation loop
weight_turnoff = True

if not decl_ETF:
    if weight_turnoff:
        train_val_loop_HPO(model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler, accuracy, top5accuracy, writer, device,
                    experiment_name, model_name, timestamp,
                    train_filename=train_filename, val_filename=val_filename, log_filename=log_filename,
                    sparsity_filename=sparsity_filename, prune_filename=trace_filename, debug_filename=debug_filename,
                    jenks_filename=jenks_filename,
                    prune_count=prune_count, one_update=one_update, EPOCHS=EPOCHS, sparsity=sparsity,
                    prune_epoch_list=prune_epoch_list, prune_epoch=prune_epoch, prune_between=5,
                    prune_ratio=prune_ratio, one_shot=one_shot, mask=mask,
                    mag_prune=mag_prune, bias_prune=bias_prune, kill_velocity=kill_velocity,
                    l2=l2, lambda_=lambda_, warmup_epochs=warmup_epochs, min_epochs=min_epochs, elem_bias=True,
                    use_ema=True, ema_decay=0.95)   # EMA: epoch-granularity weight avg, best_ema_*.pth on EMA val acc
    else:
        train_val_loop(model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler, accuracy, top5accuracy, writer, device,
                    experiment_name, model_name, timestamp,
                    train_filename=train_filename, val_filename=val_filename, log_filename=log_filename,
                    sparsity_filename=sparsity_filename, prune_filename=trace_filename, debug_filename=debug_filename,
                    jenks_filename=jenks_filename,
                    prune_count=prune_count, one_update=one_update, EPOCHS=EPOCHS, sparsity=sparsity,
                    prune_epoch_list=prune_epoch_list, prune_epoch=prune_epoch, prune_between=5,
                    prune_ratio=prune_ratio, one_shot=one_shot, mask=mask,
                    mag_prune=mag_prune, bias_prune=bias_prune, kill_velocity=kill_velocity,
                    l2=l2, lambda_=lambda_, warmup_epochs=warmup_epochs, min_epochs=min_epochs)
else:
    train_val_loopETF(model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler, accuracy, top5accuracy, writer, device,
                experiment_name, model_name, timestamp,
                train_filename, val_filename, log_filename, sparsity_filename, prune_filename, debug_filename, jenks_filename,
                prune_count=prune_count, one_update=one_update, EPOCHS=EPOCHS, sparsity=sparsity,
                prune_epoch_list=prune_epoch_list, prune_epoch=prune_epoch, prune_between=prune_between,
                prune_ratio=prune_ratio, one_shot=one_shot, mask=mask, mag_prune=mag_prune,
                bias_prune=bias_prune, kill_velocity=kill_velocity, l2=l2, lambda_=lambda_, warmup_epochs=warmup_epochs,
                min_epochs=min_epochs)

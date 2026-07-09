from models import ResNet56
import torch
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
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
from custom_optimizer import Prune_Score,train_one_step_prune,Prune_Score_Select, train_one_step_prune_v2, init_network, Prune_Score_v2, ElementwiseMomentumSGD
from custom_schedulers import WarmupMultiStepLR, init_lr_weight_decay,WarmupMultiStepJenks, WarmupCosineLR,WarmupMultiStepJenksBias, SequentialJenksScheduler, WarmupAutoJenks
from rcnet import create_RC56
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
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
from training_loop import train_val_loop, train_val_loop_scheduler, train_val_loop_ResNet, train_val_loop_ResNet_scheduler,train_val_loop_ResNet_scheduler_v2,train_val_loop_ResNet_scheduler_ETF, train_val_loop_HPO, train_val_loopETF
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, SequentialLR, ExponentialLR
from resnet import resnet56, resnet32
from models import ResNet56ETF, Args, AGD_init_weights
from rcnet import create_ResNet18, create_ResNet32
from utils import calculate_normalisation_params, RandomContrast, RandomGamma, TinyImageNetDataset
print(torch.cuda.is_available())

one_shot = False        # iterative layerwise Jenks: re-prune every 5 epochs from prune_epoch
prune_ratio = 0.95      # cutoff: stop pruning once sparsity reaches this, then recover (CIFAR-100: 90% to keep capacity)
import cuda_helpers
cuda_helpers.OVER_PRUNE = 0.0    # pure Jenks (~89% sparsity); accuracy push, no extra cut
torch.cuda.empty_cache()
# train_val_dataset = datasets.MNIST(root="./datasets/", train=True, download=True)
# test_dataset = datasets.MNIST(root="./datasets/", train=False, download=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

kill_velocity = True
train_lr_decay_factor = 0.25

gsm_lr_base_value = 1e-2
# gsm_lr_boundaries = [90, 260, 400]
# gsm_lr_boundaries = [90, 200, 300, 400, 500]
gsm_lr_boundaries = [40]
gsm_bias_lr_boundaries = [240, 440]
gsm_momentum = 0.99
gsm_max_epochs = 280
mask = True
decl_ETF = False
TinyImageNet_PATH = "./datasets/tiny-imagenet-200/"
CIFAR10_PATH = "./datasets"
datasets_name = 'cifar100'  # 'cifar10' , 'cifar100', 'tiny_imagenet'
if datasets_name == 'tiny_imagenet':
    num_classes = 200
    test = True
    id_dict = {}
    for i, line in enumerate(open('./datasets/tiny-imagenet-200/wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    normalize = transforms.Normalize(mean=[0.48024578664982126, 0.44807218089384643, 0.3975477478649648], 
                                std=[0.2769864069088257, 0.26906448510256, 0.282081906210584])
    if test:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(56),  # 56x56 random crop each time
            transforms.RandomHorizontalFlip(p=0.5),

            # translation + rotation (affine)
            transforms.RandomAffine(
                degrees=15,        # rotation range
                translate=(0.1, 0.1)  # up to ±10% translation
            ),

            transforms.ToTensor(),

            # the paper applies these randomly, so probability 0.5 each is normal
            transforms.RandomApply([RandomContrast((0.9, 1.08))], p=0.5),
            transforms.RandomApply([RandomGamma((0.9, 1.08))], p=0.5),
            # Tiny ImageNet normalization if used
            normalize,
        ])
        val_transforms = transforms.Compose([
            transforms.CenterCrop(56),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transforms = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                normalize,
            ])
        val_transforms = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
    root = TinyImageNet_PATH
    train_dataset = TinyImageNetDataset(root=root, id=id_dict, transform=train_transforms, train=True)
    val_dataset = TinyImageNetDataset(root=root, id=id_dict, transform=val_transforms, train=False)
if datasets_name == 'cifar10':
    num_classes = 10
    means = [0.4918687901200927, 0.49185976472299225, 0.4918583862227116]
    stds  = [0.24697121702736, 0.24696766978537033, 0.2469719877121087]

    normalize_4 = transforms.Normalize(
        mean=means,
        std=stds,
    )
    CIFAR10_PATH = "./datasets"
    train_dataset = datasets.CIFAR10(root = CIFAR10_PATH, train=True, download=True,
                                transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                                # transforms.RandomEqualize(),
                                transforms.ToTensor(),
                                normalize_4,
                                transforms.RandomErasing(p=0.25),
                            ]))
    val_dataset = datasets.CIFAR10(CIFAR10_PATH, train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize_4
                                    ]))
elif datasets_name == 'cifar100':
    num_classes = 100
    normalize_3 = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761],
    )
    CIFAR100_PATH = "./datasets"
    train_dataset = datasets.CIFAR100(root = CIFAR100_PATH, train=True, download=True,
                                transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                                transforms.ToTensor(),
                                normalize_3,
                                transforms.RandomErasing(p=0.25),
                            ]))
    val_dataset = datasets.CIFAR100(CIFAR100_PATH, train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize_3
                                    ]))


''' From AutoAugment:  the most commonly picked
transformations on CIFAR-10 are Equalize, AutoContrast,
Color, and Brightness '''
reset= False
depth = 32

if depth == 18:
    model = create_ResNet18(num_classes=num_classes)
elif depth == 32:
    model = resnet32(num_classes=num_classes, test_firstlayer=False, test_lastlayer=False)
BATCH_SIZE = 350
AGD = False
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)
gain = 1
xavier = True

if AGD:
    AGD_init_weights(model)
# for k, v in model.named_parameters():
#     if v.dim() in [2, 4]:
#         if xavier:
#             torch.nn.init.xavier_uniform_(v, gain=gain)
#             # print('init {} as xavier_uniform'.format(k))
#         else:
#             continue
#     if 'bias' in k and 'bn' not in k.lower():
#         torch.nn.init.zeros_(v)
        # print('init {} as zero'.format(k))
# model = torch.compile(model, mode="reduce-overhead", backend="inductor")
model = model.to(device)
# print(model)  
min_epochs = 350
label_smoothing = 0.1
loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
momentum = 0.99
learning_rate = 5e-2
weight_decay = 5e-4
bias_weight_decay = 2e-4
warmup_epochs = 10
nestrov = True
params = []
bias_lr = True
prune_epoch =  175

optimizer = init_lr_weight_decay(model, learning_rate, weight_decay,bias_weight_decay=bias_weight_decay, momentum=momentum, nestrov=nestrov, bias_lr=bias_lr, elem_bias = True, warmup_epochs=warmup_epochs, prune_epoch=prune_epoch)
init_network(optimizer)
EPOCHS = 350
# MixUp/CutMix to gain accuracy headroom; off for final 20 epochs (clean fine-tune).
import custom_optimizer
custom_optimizer.MIXUP = True    # on for CIFAR-100: hard 100-class task, +3.9% on VGG; not capacity-starved at 90%
custom_optimizer.MIXUP_OFF_EPOCH = EPOCHS - 20
# scheduler = WarmupMultiStepLR(optimizer, milestones=[80, 120, 140], warmup_factor=0.1, warmup_iters=10, warmup_method="linear")
adj = False
schedule = True
# scheduler = WarmupMultiStepJenksBias(optimizer, milestones_weights=gsm_lr_boundaries, milestones_bias=gsm_bias_lr_boundaries, warmup_factor=0.1, warmup_iters=warmup_epochs, warmup_method="linear", adjustable=False)
gamma = .875
warmup_epochs_2 = 10
two_schedulers = False
rewind_epoch = None    # LR warm-restart disabled: gave no lift on CIFAR-10 (92.882% vs 92.880%)
scheduler = WarmupAutoJenks(optimizer,milestones=gsm_lr_boundaries, warmup_factor=1/2, warmup_iters=warmup_epochs, prune_epochs=prune_epoch, reset=reset, rewind_epoch=rewind_epoch)
accuracy = Accuracy(task='multiclass', num_classes=num_classes)
top5accuracy = MulticlassAccuracy(num_classes=num_classes, top_k=5)
 ## Check the number of parameters in the model vs number of trainiable parameters
bias_prune = False

# Experiment tracking
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_name = datasets_name
model_name = f"ResNet_{depth}"
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


train_dir = model_name + f"_{datasets_name}_Output"
os.makedirs(train_dir, exist_ok=True)  # Create directory if it doesn't exist
name =  "SGD_Agg"
log_filename = os.path.join(train_dir, f"log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
train_filename = os.path.join(train_dir, f"training_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
trace_filename = os.path.join(train_dir, f"trace_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
sparsity_filename = os.path.join(train_dir, f"sparisty_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
trace_val_filename = os.path.join(train_dir, f"sparisty_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
val_filename = os.path.join(train_dir,f"validation_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
test_filename = os.path.join(train_dir,f"test_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
debug_filename = os.path.join(train_dir,f"debug_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
jenks_filename = os.path.join(train_dir,f"jenks_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
prune_filename = os.path.join(train_dir,f"prune_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
master_count = 0
epoch = 0
no_jenks =True
l2 = False
mag_prune = True
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
    if label_smoothing > 0:
        print(f"Label smoothing is used with value: {label_smoothing}", file=f)
    else:
        print(f"Label smoothing is not used", file=f)
prune_count = 0
sparsity = 0.0
one_update = True
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters in the model: {total_params}")
total_pruned_params = sum(p.numel() for p in model.parameters() if p.dim() in [2, 4])
print(f"Total prunebale parameters in the model: {total_pruned_params}")
prune_epoch_list = [prune_epoch]
prune_between = 25
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
                    l2=l2, lambda_=lambda_, warmup_epochs=warmup_epochs, min_epochs=min_epochs, elem_bias = True, weight_reset=reset)
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


# if not schedule:
#     if decl_ETF:
#         train_val_loop_ResNet_scheduler_ETF(model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler, accuracy, top5accuracy, writer, device,
#                     experiment_name, model_name, timestamp,
#                     train_filename=train_filename, val_filename=val_filename, log_filename=log_filename,
#                     sparsity_filename=sparsity_filename, prune_filename=trace_filename, debug_filename=debug_filename,
#                     jenks_filename=jenks_filename,
#                     prune_count=prune_count, one_update=one_update, EPOCHS=EPOCHS, sparsity=sparsity,
#                     prune_epoch_list=prune_epoch_list, prune_epoch=prune_epoch, prune_between=prune_between,
#                     prune_ratio=prune_ratio, one_shot=one_shot, mask=mask,
#                     mag_prune=mag_prune, bias_prune=bias_prune, kill_velocity=kill_velocity,
#                     l2=l2, lambda_=lambda_, warmup_epochs=warmup_epochs,warmup_epochs_2=warmup_epochs_2, min_epochs=min_epochs)
#     else:
#         train_val_loop_ResNet(model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler, accuracy, top5accuracy, writer, device,
#                     experiment_name, model_name, timestamp,
#                     train_filename=train_filename, val_filename=val_filename, log_filename=log_filename,
#                     sparsity_filename=sparsity_filename, prune_filename=trace_filename, debug_filename=debug_filename,
#                     jenks_filename=jenks_filename,
#                     prune_count=prune_count, one_update=one_update, EPOCHS=EPOCHS, sparsity=sparsity,
#                     prune_epoch_list=prune_epoch_list, prune_epoch=prune_epoch, prune_between=prune_between,
#                     prune_ratio=prune_ratio, one_shot=one_shot, mask=mask,
#                     mag_prune=mag_prune, bias_prune=bias_prune, kill_velocity=kill_velocity,
#                     l2=l2, lambda_=lambda_, warmup_epochs=warmup_epochs, warmup_epochs_2=warmup_epochs_2, min_epochs=min_epochs)
# if schedule:
#     if not two_schedulers:
#         if decl_ETF:
#             train_val_loop_ResNet_scheduler_ETF(model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler_ResNet, accuracy, top5accuracy, writer, device,
#                     experiment_name, model_name, timestamp,
#                     train_filename=train_filename, val_filename=val_filename, log_filename=log_filename,
#                     sparsity_filename=sparsity_filename, prune_filename=trace_filename, debug_filename=debug_filename,
#                     jenks_filename=jenks_filename,
#                     prune_count=prune_count, one_update=one_update, EPOCHS=EPOCHS, sparsity=sparsity,
#                     prune_epoch_list=prune_epoch_list, prune_epoch=prune_epoch, prune_between=prune_between,
#                     prune_ratio=prune_ratio, one_shot=one_shot, mask=mask,
#                     mag_prune=mag_prune, bias_prune=bias_prune, kill_velocity=kill_velocity,
#                     l2=l2, lambda_=lambda_, warmup_epochs=warmup_epochs,warmup_epochs_2=warmup_epochs_2, min_epochs=min_epochs)
#         else:
#             train_val_loop_ResNet_scheduler(model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler_ResNet, accuracy, top5accuracy, writer, device,
#                     experiment_name, model_name, timestamp,
#                     train_filename=train_filename, val_filename=val_filename, log_filename=log_filename,
#                     sparsity_filename=sparsity_filename, prune_filename=trace_filename, debug_filename=debug_filename,
#                     jenks_filename=jenks_filename,
#                     prune_count=prune_count, one_update=one_update, EPOCHS=EPOCHS, sparsity=sparsity,
#                     prune_epoch_list=prune_epoch_list, prune_epoch=prune_epoch, prune_between=prune_between,
#                     prune_ratio=prune_ratio, one_shot=one_shot, mask=mask,
#                     mag_prune=mag_prune, bias_prune=bias_prune, kill_velocity=kill_velocity,
#                     l2=l2, lambda_=lambda_, warmup_epochs=warmup_epochs,warmup_epochs_2=warmup_epochs_2, min_epochs=min_epochs)
#     else:
#         train_val_loop_ResNet_scheduler_v2(model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler_ResNet, scheduler_double, accuracy, top5accuracy, writer, device,
#                 experiment_name, model_name, timestamp,
#                 train_filename=train_filename, val_filename=val_filename, log_filename=log_filename,
#                 sparsity_filename=sparsity_filename, prune_filename=trace_filename, debug_filename=debug_filename,
#                 jenks_filename=jenks_filename,
#                 prune_count=prune_count, one_update=one_update, EPOCHS=EPOCHS, sparsity=sparsity,
#                 prune_epoch_list=prune_epoch_list, prune_epoch=prune_epoch, prune_between=prune_between,
#                 prune_ratio=prune_ratio, one_shot=one_shot, mask=mask,
#                 mag_prune=mag_prune, bias_prune=bias_prune, kill_velocity=kill_velocity,
#                 l2=l2, lambda_=lambda_, warmup_epochs=warmup_epochs,warmup_epochs_2=warmup_epochs_2, min_epochs=min_epochs)

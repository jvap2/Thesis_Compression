from FP_Quantization_Experiments import brecq_quantize_exp_fp, brecq_quantize_exp_fp_scale, quantize_model_fp, QuantConv2dFP, QuantLinearFP
from torchvision import datasets, transforms
from utils import RandomContrast, RandomGamma, TinyImageNetDataset
from Quantization_Experiments.utils import QuantNetwork
import brevitas
from torch import nn
from brevitas.nn import QuantLinear
from densenet import create_densenet40
from rcnet import create_lenet5
from resnet import resnet56 , resnet32
from models import vgg19
import torch
import os
networks = ["LeNet5", "LeNet300", "DenseNet40", "ResNet56", "VGG19", "ResNet32"]
data = ["MNIST", "CIFAR10", "CIFAR100", "tiny_imagenet"]
geometry = True
batch_size = 512
bitwidth = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = networks[3]
data = data[1]
precision_config = {
    "first": (4, 3),
    "last": (4, 3),
    "default": (3, 0),
    "conv": (3, 1),
}
if net == "LeNet5":
    reg_model = create_lenet5()
    model_to_quantize = create_lenet5()
    saved_dict = "Best_Results_HPO/LeNet5/best_2025-12-08_15-03-41_MNIST_LeNet5.pth"
    datasets_name = "MNIST"
    csv_file = "LeNet5_data_quant.csv"
elif net == "LeNet300":
    reg_model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=784, out_features=300),
                nn.ReLU(),
                nn.Linear(in_features=300, out_features=100),
                nn.ReLU(),
                nn.Linear(in_features=100, out_features=10),
            )
    model_to_quantize= nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=784, out_features=300),
                nn.ReLU(),
                nn.Linear(in_features=300, out_features=100),
                nn.ReLU(),
                nn.Linear(in_features=100, out_features=10),
            )
    saved_dict = "Best_Results_HPO/LeNet300/best_2025-12-09_18-58-19_MNIST_LeNet300.pth"
    datasets_name = "MNIST"
    csv_file = "LeNet300_data_quant.csv"
elif net == "DenseNet40":
    reg_model = create_densenet40()
    model_to_quantize = create_densenet40()
    saved_dict = "Best_Results_HPO/DenseNet40/best_2025-11-16_18-44-04_CIFAR10_DenseNet40.pth"
    datasets_name = "cifar10"
    csv_file = "DenseNet40_data_quant.csv"
elif net == "ResNet56":
    reg_model = torch.hub.load(
        'chenyaofo/pytorch-cifar-models',
        'cifar10_resnet56',
        pretrained=True
    )
    model_to_quantize= torch.hub.load(
        'chenyaofo/pytorch-cifar-models',
        'cifar10_resnet56',
        pretrained=True
    )
    # saved_dict = "Best_Results_HPO/ResNet56/best_2025-11-16_18-44-00_CIFAR10_ResNet56.pth"
    datasets_name = "cifar10"
    csv_file = "ResNet56_data_quant.csv"
elif net == "ResNet32":
    if data == "CIFAR10":
        num_classes = 10
        datasets_name = "cifar10"
        saved_dict = "Best_Results_HPO/ResNet32/CIFAR-10/best_2025-12-05_00-10-54_cifar10_ResNet_32.pth"
        csv_file = "ResNet32_CIFAR10_data_quant.csv"
    elif data == "CIFAR100":
        num_classes = 100
        datasets_name = "cifar100"
        saved_dict = "Best_Results_HPO/ResNet32/CIFAR-100/86_Sparsity/best_2025-12-03_10-14-41_cifar100_ResNet_32.pth"
        csv_file = "ResNet32_CIFAR100_data_quant.csv"
    elif data == "tiny_imagenet":
        num_classes = 200   
        datasets_name = "tiny_imagenet"
        saved_dict = "Best_Results_HPO/ResNet32/TinyImageNet/best_2025-12-15_10-04-01_tiny_imagenet_ResNet_32.pth"
        csv_file = "ResNet32_TinyImageNet_data_quant.csv"
    reg_model = resnet32(num_classes=num_classes)
    model_to_quantize= resnet32(num_classes=num_classes)
elif net == "VGG19":
    if data == "CIFAR10":
        datasets_name = "cifar10"
        saved_dict = "Best_Results_HPO/VGG-19/CIFAR-10/best_2025-11-25_06-06-15_cifar10_vgg19.pth"
        csv_file = "VGG19_C10_data_quant.csv"
    elif data == "CIFAR100":
        datasets_name = "cifar100"
        saved_dict = "Best_Results_HPO/VGG-19/CIFAR-100/98_sparsity/best_2025-11-23_20-24-12_cifar100_vgg19.pth"
        csv_file = "VGG19_CIFAR100_data_quant.csv"
    elif data == "tiny_imagenet":
        datasets_name = "tiny_imagenet"
        saved_dict = "Best_Results_HPO/VGG-19/TinyImageNet/best_2025-12-31_11-17-06_tiny_imagenet_vgg19.pth"
        csv_file = "VGG19_TinyImageNet_data_quant.csv"
    reg_model = vgg19(dataset=datasets_name)
    model_to_quantize = vgg19(dataset=datasets_name)
folder_name = f"FP_Quantization_Experiments/{net}_{data}/{bitwidth}_bit"
import os
os.makedirs(folder_name, exist_ok=True)
bitwidth_filename = f"{folder_name}/{net}_{data}_bitwidths.txt"
if geometry==False:
    bitwidth_geometry_filename = f"{folder_name}/{net}_{data}_bitwidths_after_BRECQ_{batch_size}.txt"
else:
    bitwidth_geometry_filename = f"{folder_name}/{net}_{data}_bitwidths_after_GAQ_{batch_size}.txt"
accuracy_filename = f"{folder_name}/{net}_{data}_accuracy.txt"
if geometry==False:
    accuracy_geometry_filename = f"{folder_name}/{net}_{data}_accuracy_after_BRECQ_{batch_size}.txt"
else:
    accuracy_geometry_filename = f"{folder_name}/{net}_{data}_accuracy_after_GAQ_{batch_size}.txt"
theta_filename = f"{folder_name}/{net}_{data}_theta_values.txt"
pruned_filename = f"{folder_name}/{net}_{data}_pruned_weights.txt"
visual_filename = f"{folder_name}/{net}_{data}_data_visuals.png"
reg_model.to(device=device)
if net !="ResNet56":
    reg_model.load_state_dict(torch.load(saved_dict))
    model_to_quantize.load_state_dict(torch.load(saved_dict))
TinyImageNet_PATH = "./datasets/tiny-imagenet-200/"
CIFAR10_PATH = "./datasets"  # 'cifar10' , 'cifar100', 'tiny_imagenet'
if datasets_name == 'tiny_imagenet':
    num_classes = 200
    if net == "VGG19":
        test = False
    else:
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
                            ]))
    val_dataset = datasets.CIFAR100(CIFAR100_PATH, train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize_3
                                    ]))
elif datasets_name == 'MNIST':
    num_classes = 10
    train_dataset =  datasets.MNIST("./datasets/", train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))]))
    val_dataset= datasets.MNIST("./datasets/", train=False, transform=transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    
import torch
from torch.utils.data import DataLoader
''' Evaluate the inference accuracy of the quantized model '''
def TestNetwork(model, val_dataset, filepath=accuracy_filename):
    batch_size = 128
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the quantized model on the test set: {accuracy:.2f}%', file=open(filepath, "w"))


'''Now we want to look at the weight histograms in each layer to see how the quantization is working. We can use matplotlib to plot the histograms.'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data = pd.DataFrame(columns=["name", "sparsity"])
log_dynamic_ranges = []
def plot_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, (brevitas.nn.QuantConv2d, brevitas.nn.QuantLinear)):
            param = module.quant_weight()
            weights = param.detach().cpu().tensor.numpy().flatten()
            ''' Drop the weights that are zero'''
            print(f"Before: zeros={np.sum(weights == 0)}")
            weights_no = param.tensor.flatten().numel()
            '''Check the number of unique weights'''
            unique_weights = len(np.unique(weights))
            print(f"Unique weights: {unique_weights}")
            sparsity = np.sum(weights == 0) / weights_no
            data.loc[len(data)] = [name, sparsity]
            weights = weights[weights != 0]
            max_weight = np.max(np.abs(weights))
            min_weight = np.min(np.abs(weights))
            log_dynamic_range = np.log10(max_weight) - np.log10(min_weight)
            log_dynamic_ranges.append((name, log_dynamic_range))

            print(f"After: zeros={np.sum(weights == 0)}")
            plt.figure(figsize=(10, 5))
            plt.hist(weights, bins=100)
            plt.title(f'Weight Histogram of {name}')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.savefig(f"{folder_name}/{name}_weights_histogram.png")
            plt.close()
            print(f"File exists after save: {os.path.exists(f'{folder_name}/{name}_weights_histogram.png')}")
# Save the sparsity data to a CSV file
data.to_csv(f"{folder_name}/{csv_file}", index=False)
plt.figure(figsize=(10, 5))
## plot the log dynamic ranges
names = [item[0] for item in log_dynamic_ranges]
ranges = [item[1] for item in log_dynamic_ranges]
plt.bar(names, ranges)
plt.xticks(rotation=90)
plt.title(f'Log Dynamic Ranges for {net} on {data}')
plt.xlabel('Layer Name')
plt.ylabel('Log10 Dynamic Range')
plt.savefig(f"{folder_name}/log_dynamic_ranges_{net}.png")

## Save the log dynamic ranges to a CSV file
log_data = pd.DataFrame(log_dynamic_ranges, columns=["name", "log_dynamic_range"])
log_data.to_csv(f"{folder_name}/log_dynamic_ranges_{net}.csv", index=False)


''' Now we want to apply the geometry-aware quantization to see if it improves the accuracy. Use a smaller calibration batch size to reduce memory.'''
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
'''We need to get the mask from pruning from the reg model to multiply it with the quantized weights to make sure we are only quantizing the non-zero weights. We can use the state_dict of the reg model to get the masks.'''
'''Iterate through the layers and set the mask to one if a value is non-zero and zero if it is zero. Then we can apply the geometry-aware quantization to the quantized model using the masks.'''
mask = {}
for name, module in reg_model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        weight_mask = (module.weight != 0).float()
        mask[name] = weight_mask
'''Put the mask on the same device as the model'''
device = next(reg_model.parameters()).device
for name in mask:
    mask[name] = mask[name].to(device)
import torch
torch.cuda.empty_cache()
import torch
import torch.nn as nn

def diagnose_forward_pass(model, data_loader, device='cuda'):
    """Hook every layer and print activation statistics."""
    hooks = []
    activation_stats = {}

    def make_hook(name):
        def hook(module, input, output):
            out = output.detach().float()
            activation_stats[name] = {
                'mean_abs': out.abs().mean().item(),
                'max_abs':  out.abs().max().item(),
                'std':      out.std().item(),
                'has_nan':  torch.isnan(out).any().item(),
                'has_inf':  torch.isinf(out).any().item(),
            }
        return hook

    # Register hooks on every module that produces an output
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        batch = next(iter(data_loader))
        x = batch[0][:4].to(device)  # just 4 samples
        out = model(x)

    for h in hooks:
        h.remove()

    print(f"\nFinal output: min={out.min().item():.4e}, max={out.max().item():.4e}, mean={out.abs().mean().item():.4e}")
    print(f"\n{'Layer':<45} {'mean_abs':>12} {'max_abs':>12} {'std':>12} {'nan':>6} {'inf':>6}")
    print("-" * 95)
    for name, stats in activation_stats.items():
        flag = " <<<" if stats['mean_abs'] < 1e-6 or stats['has_nan'] or stats['has_inf'] else ""
        print(f"{name:<45} {stats['mean_abs']:>12.4e} {stats['max_abs']:>12.4e} "
              f"{stats['std']:>12.4e} {str(stats['has_nan']):>6} {str(stats['has_inf']):>6}{flag}")



def diagnose_quantization(model, device='cuda'):
    for name, module in model.named_modules():
        if hasattr(module, 'weight_q') and module.weight_q is not None:
            wq = module.weight_q
            wo = None
            if hasattr(module, 'linear'):
                wo = module.linear.weight.data
            elif hasattr(module, 'conv'):
                wo = module.conv.weight.data

            print(f"\n=== {name} ===")
            print(f"  weight_q shape     : {wq.shape}")
            print(f"  weight_q min/max   : {wq.min().item():.4e} / {wq.max().item():.4e}")
            print(f"  weight_q mean abs  : {wq.abs().mean().item():.4e}")
            print(f"  weight_q % zeros   : {(wq == 0).float().mean().item()*100:.1f}%")
            print(f"  weight_q has nan   : {torch.isnan(wq).any().item()}")
            print(f"  weight_q has inf   : {torch.isinf(wq).any().item()}")

            if wo is not None:
                print(f"  original min/max   : {wo.min().item():.4e} / {wo.max().item():.4e}")
                print(f"  original mean abs  : {wo.abs().mean().item():.4e}")
                print(f"  original % zeros   : {(wo == 0).float().mean().item()*100:.1f}%")
def recalibrate_batchnorm(model, data_loader, device='cuda', num_batches=50):
    """
    Reset and recompute BatchNorm running stats using the quantized weights.
    Call this AFTER quantize_model_fp() and BEFORE evaluation.
    """
    model.to(device)

    # 1. Reset all BN stats to neutral starting point
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            module.reset_running_stats()
            module.momentum = 0.1  # standard momentum
            module.train()         # enable running stat updates

    # 2. Freeze everything else — we only want BN stats to update
    for name, module in model.named_modules():
        if not isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()

    # 3. Forward pass — no gradients needed, just accumulate BN stats
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            x = x.to(device)
            model(x)

    # 4. Switch everything back to eval
    model.eval()
    print(f"BatchNorm recalibrated over {min(num_batches, len(data_loader))} batches.")
    return model
def evaluate(model, data_loader, device):
    model.eval().to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"Accuracy: {100.*correct/total:.2f}%  ({correct}/{total})")
evaluate(reg_model, val_dataloader, device)
# quant_model = geometry_aware_rounding_BRECQ(reg_model, val_dataloader, device=device, name=net, bitwidth=bitwidth)
# quant_model = brecq_quantize(model=reg_model, calibration_loader=val_dataloader, name=net,bitwidth=bitwidth, geometry = geometry)
# quant_model = brecq_quantize_exp_fp(model=reg_model, calibration_loader=val_dataloader, name=net,bitwidth=bitwidth, geometry = geometry, batch_size=batch_size)
# quant_model = brecq_quantize_exp_fp_scale(model=reg_model, calibration_loader=val_dataloader, name=net,bitwidth=bitwidth, geometry = geometry, batch_size=batch_size)
import copy
# model_to_quantize = copy.deepcopy(reg_model)
quant_model = quantize_model_fp(model_to_quantize,val_dataloader, block_size=128,e_bits=2,m_bits=1,e_bits_scale=4,m_bits_scale=3, device = device, use_HG=False, use_Hessian=True)
# quant_model = quantize_net_fixed(model_to_quantize,val_dataloader,block_size=64, mbits_weight=1, ebits_weight=2, mbits_scale=3, ebits_scale=4)

# quant_model = recalibrate_batchnorm(quant_model, train_dataloader, device=device, num_batches=50)
TestNetwork(quant_model, val_dataset, filepath=accuracy_geometry_filename)
'''Print out the model details after geometry-aware quantization to see if there are any changes in bitwidths'''
with open(bitwidth_geometry_filename, "a") as f:
    for name, module in quant_model.named_modules():

        if hasattr(module, "weight_q") and module.weight_q is not None:
            print(f"\n{name} (QUANTIZED):", file=f)
            print(module.weight_q, file=f)

        elif isinstance(module, nn.Conv2d):
            print(f"\n{name} (ORIGINAL CONV):", file=f)
            print(module.weight.data, file=f)

        elif isinstance(module, nn.Linear):
            print(f"\n{name} (ORIGINAL LINEAR):", file=f)
            print(module.weight.data, file=f)
# diagnose_quantization(quant_model)
# diagnose_forward_pass(quant_model, val_dataloader, device)

# ============================================================
# TEST 1: Does the ORIGINAL unquantized model work?
# Expected: should give your baseline accuracy (e.g. 85-93%)
# ============================================================


print("=== TEST 1: Original model ===")
evaluate(reg_model, val_dataloader, device)

# ============================================================
# TEST 2: Does quantized model WITH weight_q replaced by 
#         original weights give correct accuracy?
# This isolates whether the problem is in the forward() path
# or in the quantized weights themselves.
# ============================================================
print("\n=== TEST 2: Quant model structure, original weights ===")
for module in quant_model.modules():
    if isinstance(module, QuantLinearFP):
        module.weight_q = module.linear.weight.data.clone()
    elif isinstance(module, QuantConv2dFP):
        module.weight_q = module.conv.weight.data.clone()

evaluate(quant_model, val_dataloader, device)

# ============================================================
# TEST 3: Are quantized weights actually being used in forward()?
# Print weight_q vs what forward() sees
# ============================================================
print("\n=== TEST 3: Weight identity check ===")
for name, module in quant_model.named_modules():
    if isinstance(module, QuantConv2dFP):
        wq = module.weight_q
        wc = module.conv.weight.data
        print(f"{name}: weight_q mean={wq.abs().mean():.4e}, "
              f"conv.weight mean={wc.abs().mean():.4e}, "
              f"same={torch.allclose(wq, wc)}")
        break  # just check first layer

# ============================================================
# TEST A: How is reg_model actually structured?
# ============================================================
# print("=== reg_model structure ===")
# print(reg_model)

# ============================================================
# TEST B: What do the original weights look like?
# ============================================================
print("\n=== reg_model weight stats ===")
for name, module in reg_model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        w = module.weight.data
        print(f"{name:<40} mean_abs={w.abs().mean():.4e}  "
              f"min={w.min():.4e}  max={w.max():.4e}  "
              f"zeros={100.*(w==0).float().mean():.1f}%")

# ============================================================
# TEST C: Does a single forward pass produce sane output?
# ============================================================
print("\n=== Single batch forward pass ===")
reg_model.eval().to(device)
with torch.no_grad():
    x, y = next(iter(val_dataloader))
    x = x.to(device)
    out = reg_model(x)
    out_quant = quant_model(x)
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output min  : {out.min().item():.4e}")
    print(f"Output max  : {out.max().item():.4e}")
    print(f"Output mean : {out.abs().mean().item():.4e}")
    print(f"Predicted classes: {out.argmax(dim=1)[:20].tolist()}")
    print(f"Predicted classes: {out_quant.argmax(dim=1)[:20].tolist()}")
    print(f"True classes:      {y[:20].tolist()}")


# 1) Original and quantized layer outputs
with torch.no_grad():
    x, _ = next(iter(val_dataloader))
    x = x.to(device)
    first_layer = list(reg_model.children())[0]
    quant_data = list(quant_model.children())[0]
    # original model (fp32)
    y_fp32 = first_layer(x)

    # quantized weights, same input
    y_fp4  = quant_data(x)

    rel_err = (y_fp4 - y_fp32).pow(2).sum() / (y_fp32.pow(2).sum() + 1e-8)
    print("conv1 relative output MSE:", rel_err.item())


# def collect_layer_outputs(model, x, device):
#     """
#     Runs a forward pass and returns an ordered dict:
#     {module_name: output_tensor_detached}
#     """
#     outputs = {}
#     hooks = []

#     # Hook function factory to capture 'name'
#     def make_hook(name):
#         def hook(module, inp, out):
#             # Store as float32 on CPU to simplify MSE computation
#             outputs[name] = out.detach().to('cpu', dtype=torch.float32)
#         return hook

#     # Register hooks on all modules that produce activations
#     for name, module in model.named_modules():
#         # Often you skip the top-level module "" itself
#         if len(list(module.children())) == 0:  # leaf modules only
#             h = module.register_forward_hook(make_hook(name))
#             hooks.append(h)

#     # Run forward once
#     with torch.no_grad():
#         _ = model(x.to(device))

#     # Remove hooks
#     for h in hooks:
#         h.remove()

#     return outputs

# # Get one batch
# with torch.no_grad():
#     x, _ = next(iter(val_dataloader))

# # Collect outputs
# fp32_outs = collect_layer_outputs(reg_model.eval(), x, device)
# fp4_outs  = collect_layer_outputs(quant_model.eval(), x, device)

# # Compute per-layer relative MSE
# rel_mse_per_layer = {}
# for name, name_q in zip(fp32_outs.keys(), fp4_outs.keys()):
#     y_fp32 = fp32_outs[name]
#     y_fp4  = fp4_outs[name_q]
#     if y_fp32.shape != y_fp4.shape:
#         print(f"Shape mismatch at {name} and {name_q}: {y_fp32.shape} vs {y_fp4.shape}")
#         continue
#     num = (y_fp4 - y_fp32).pow(2).sum()
#     den = (y_fp32.pow(2).sum() + 1e-8)
#     rel_mse = (num / den).item()
#     rel_mse_per_layer[name] = rel_mse

# # Print sorted by error
# for name, err in sorted(rel_mse_per_layer.items(), key=lambda kv: kv[1], reverse=True):
#     print(f"{name:40s}  rel MSE = {err:.4f}")


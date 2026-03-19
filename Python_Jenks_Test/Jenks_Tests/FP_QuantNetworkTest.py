from FP_Quantization_Experiments import brecq_quantize_exp_fp
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
batch_size = 1024
bitwidth = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = networks[-1]
data = data[2]
precision_config = {
    "first": (4, 3),
    "last": (4, 3),
    "default": (3, 0),
    "conv": (3, 1),
}
if net == "LeNet5":
    reg_model = create_lenet5()
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
    saved_dict = "Best_Results_HPO/LeNet300/best_2025-12-09_18-58-19_MNIST_LeNet300.pth"
    datasets_name = "MNIST"
    csv_file = "LeNet300_data_quant.csv"
elif net == "DenseNet40":
    reg_model = create_densenet40()
    saved_dict = "Best_Results_HPO/DenseNet40/best_2025-11-16_18-44-04_CIFAR10_DenseNet40.pth"
    datasets_name = "cifar10"
    csv_file = "DenseNet40_data_quant.csv"
elif net == "ResNet56":
    reg_model = resnet56()
    saved_dict = "Best_Results_HPO/ResNet56/best_2025-11-16_18-44-00_CIFAR10_ResNet56.pth"
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
reg_model.load_state_dict(torch.load(saved_dict))
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
# quant_model = geometry_aware_rounding_BRECQ(reg_model, val_dataloader, device=device, name=net, bitwidth=bitwidth)
# quant_model = brecq_quantize(model=reg_model, calibration_loader=val_dataloader, name=net,bitwidth=bitwidth, geometry = geometry)
quant_model = brecq_quantize_exp_fp(model=reg_model, calibration_loader=val_dataloader, name=net,bitwidth=bitwidth, geometry = geometry, batch_size=batch_size)
TestNetwork(quant_model, val_dataset, filepath=accuracy_geometry_filename)
'''Print out the model details after geometry-aware quantization to see if there are any changes in bitwidths'''
for name, module in quant_model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        print(f"Quantized Weights of {name}: {module.weight.data}", file=open(bitwidth_geometry_filename, "a"))
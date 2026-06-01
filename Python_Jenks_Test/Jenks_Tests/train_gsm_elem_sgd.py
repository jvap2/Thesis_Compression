"""
train_gsm_elem_sgd.py

Train classic CV models from scratch using Global Sparse Momentum (GSM)
gradient masking combined with ElementwiseMomentumSGD.

GSM (Global Sparse Momentum):
  After each backward pass, compute |grad × weight| for every weight and bias
  in the network globally. Keep only the top `nonzero_ratio` fraction of
  gradients — the rest are zeroed before the optimizer step. This lets the
  network learn which weights are most salient while simultaneously imposing
  sparsity, without a separate pruning phase.

ElementwiseMomentumSGD:
  A custom Jenks-Natural-Breaks-based optimizer that additionally applies
  per-layer elementwise saliency masking inside its step(), giving a
  second level of gradient filtering on top of the global GSM mask.

Usage:
    python train_gsm_elem_sgd.py                         # all models
    python train_gsm_elem_sgd.py --model ResNet56        # one model
    python train_gsm_elem_sgd.py --model VGG19/CIFAR10
"""

import os
# Must be set before any CUDA extension is JIT-compiled.
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6;8.9"

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from custom_optimizer import ElementwiseMomentumSGD, train_one_step_gsm, torch_accuracy
from harness.model_registry import MODEL_REGISTRY


# ── Experiment table ──────────────────────────────────────────────────────────
# nonzero_ratio : fraction of (global) gradients kept by GSM
# warmup        : epochs before Jenks masking activates in ElementwiseMomentumSGD
# prune_end     : last epoch of active pruning (mask frozen afterwards)
EXPERIMENTS = {
    "LeNet5": dict(
        epochs=130, lr=1e-2, weight_decay=1e-4, momentum=0.9,
        warmup=5,  prune_end=125, nonzero_ratio=0.1, batch_size=128,
    ),
    "LeNet300": dict(
        epochs=80,  lr=1e-2, weight_decay=1e-4, momentum=0.9,
        warmup=5,  prune_end=75,  nonzero_ratio=0.1, batch_size=128,
    ),
    "DenseNet40": dict(
        epochs=400, lr=0.1,  weight_decay=1e-4, momentum=0.9,
        warmup=10, prune_end=390, nonzero_ratio=0.1, batch_size=64,
    ),
    "ResNet56": dict(
        epochs=400, lr=0.1,  weight_decay=1e-4, momentum=0.9,
        warmup=10, prune_end=390, nonzero_ratio=0.1, batch_size=128,
    ),
    "ResNet32/CIFAR10": dict(
        epochs=300, lr=0.1,  weight_decay=1e-4, momentum=0.9,
        warmup=10, prune_end=290, nonzero_ratio=0.1, batch_size=128,
    ),
    "ResNet32/CIFAR100_86": dict(
        epochs=300, lr=0.1,  weight_decay=1e-4, momentum=0.9,
        warmup=10, prune_end=290, nonzero_ratio=0.1, batch_size=128,
    ),
    "ResNet32/TinyImageNet": dict(
        epochs=200, lr=0.1,  weight_decay=1e-4, momentum=0.9,
        warmup=10, prune_end=190, nonzero_ratio=0.1, batch_size=64,
    ),
    "VGG19/CIFAR10": dict(
        epochs=300, lr=0.1,  weight_decay=1e-4, momentum=0.9,
        warmup=10, prune_end=290, nonzero_ratio=0.1, batch_size=64,
    ),
    "VGG19/CIFAR100_98": dict(
        epochs=300, lr=0.1,  weight_decay=1e-4, momentum=0.9,
        warmup=10, prune_end=290, nonzero_ratio=0.1, batch_size=64,
    ),
    "VGG19/CIFAR100_90": dict(
        epochs=300, lr=0.1,  weight_decay=1e-4, momentum=0.9,
        warmup=10, prune_end=290, nonzero_ratio=0.1, batch_size=64,
    ),
    "VGG19/TinyImageNet": dict(
        epochs=300, lr=0.1,  weight_decay=1e-4, momentum=0.9,
        warmup=10, prune_end=290, nonzero_ratio=0.1, batch_size=32,
    ),
}


# ── Dataset builders ──────────────────────────────────────────────────────────

def build_dataloaders(dataset: str, batch_size: int, data_root: str = "./datasets"):
    """Return (train_loader, val_loader) for the given dataset name."""
    if dataset == "MNIST":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST(data_root, train=True,  download=True, transform=tfm)
        val_ds   = datasets.MNIST(data_root, train=False, download=True, transform=tfm)

    elif dataset == "cifar10":
        train_tfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ])
        val_tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ])
        train_ds = datasets.CIFAR10(data_root, train=True,  download=True, transform=train_tfm)
        val_ds   = datasets.CIFAR10(data_root, train=False, download=True, transform=val_tfm)

    elif dataset == "cifar100":
        train_tfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        ])
        val_tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        ])
        train_ds = datasets.CIFAR100(data_root, train=True,  download=True, transform=train_tfm)
        val_ds   = datasets.CIFAR100(data_root, train=False, download=True, transform=val_tfm)

    elif dataset == "tiny_imagenet":
        train_tfm = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        val_tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_root = os.path.join(data_root, "TinyImageNet", "Train")
        val_root   = os.path.join(data_root, "TinyImageNet", "Val")
        train_ds = datasets.ImageFolder(train_root, transform=train_tfm)
        val_ds   = datasets.ImageFolder(val_root,   transform=val_tfm)

    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, val_loader


# ── Optimizer factory ─────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: dict, device: torch.device) -> ElementwiseMomentumSGD:
    """
    Build an ElementwiseMomentumSGD for `model`.

    name_map     : param tensor → its named_parameters() key
    do_prune_map : module name → True iff Jenks masking should be applied there
                   (Conv2d and Linear get pruned; BatchNorm and others do not)
    """
    name_map = {p: n for n, p in model.named_parameters() if p.requires_grad}

    do_prune_map = {}
    for mod_name, mod in model.named_modules():
        do_prune_map[mod_name] = isinstance(mod, (nn.Conv2d, nn.Linear))

    trainable = [p for p in model.parameters() if p.requires_grad]

    return ElementwiseMomentumSGD(
        trainable,
        name_map=name_map,
        do_prune_map=do_prune_map,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        momentum=cfg["momentum"],
        warmup_epochs=cfg["warmup"],
        pruning_epochs=cfg["prune_end"],
        device=str(device),
        filter_based=True,   # Conv layers use filter-based masking
        bias_prune=False,    # leave biases unmasked
    )


# ── Validation helper ─────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = total_acc = total_acc5 = count = 0.0
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        pred = model(data)
        loss = criterion(pred, label)
        acc_res = torch_accuracy(pred, label, (1, 5))
        total_loss += loss.item()
        total_acc  += acc_res[0].item()
        total_acc5 += acc_res[1].item()
        count += 1
    return total_loss / count, total_acc / count, total_acc5 / count


# ── Per-model sparsity snapshot ───────────────────────────────────────────────

def measure_sparsity(model: nn.Module) -> float:
    total = zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += (p == 0).sum().item()
    return zeros / total if total > 0 else 0.0


# ── Main experiment runner ────────────────────────────────────────────────────

def run_experiment(key: str, device: torch.device, out_dir: Path, data_root: str = "./datasets") -> dict | None:
    if key not in EXPERIMENTS:
        print(f"  [skip] No experiment config for {key!r}")
        return None
    if key not in MODEL_REGISTRY:
        print(f"  [skip] {key!r} not in MODEL_REGISTRY")
        return None

    cfg   = EXPERIMENTS[key]
    entry = MODEL_REGISTRY[key]

    print(f"\n{'='*62}")
    print(f"  Model   : {key}")
    print(f"  Dataset : {entry.dataset}   Classes: {entry.num_classes}")
    print(f"  Epochs  : {cfg['epochs']}   LR: {cfg['lr']}   "
          f"GSM ratio: {cfg['nonzero_ratio']}")
    print(f"  Warmup  : {cfg['warmup']} ep   Prune phase ends: {cfg['prune_end']} ep")
    print(f"{'='*62}")

    # ── Model initialised from scratch (no saved weights) ─────────────────
    model = entry.constructor().to(device)

    # ── Dataloaders ───────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(entry.dataset, cfg["batch_size"], data_root)

    # ── Optimizer + cosine LR scheduler ──────────────────────────────────
    optimizer = build_optimizer(model, cfg, device)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    # ── Log files ─────────────────────────────────────────────────────────
    safe_key   = key.replace("/", "_")
    ts         = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prefix     = out_dir / f"{safe_key}_{ts}"

    train_log = open(f"{prefix}_train.csv", "w", newline="")
    val_log   = open(f"{prefix}_val.csv",   "w", newline="")
    t_writer  = csv.writer(train_log)
    v_writer  = csv.writer(val_log)
    t_writer.writerow(["epoch", "train_loss", "train_acc1", "train_acc5", "sparsity"])
    v_writer.writerow(["epoch", "val_loss",   "val_acc1",   "val_acc5"])

    best_val_acc = 0.0
    best_path    = out_dir / f"{safe_key}_{ts}_best.pth"

    for epoch in range(cfg["epochs"]):
        # Tell ElementwiseMomentumSGD which phase we're in
        optimizer.epoch = epoch
        model.train()

        ep_loss = ep_acc = ep_acc5 = 0.0
        n_batches = 0
        t0 = time.time()

        for data, label in train_loader:
            data, label = data.to(device), label.to(device)

            # ── GSM masking + ElementwiseMomentumSGD step ─────────────────
            # train_one_step_gsm:
            #   1. forward + backward
            #   2. globally rank |grad × weight|; zero bottom (1-ratio) gradients
            #   3. calls optimizer.step() → ElementwiseMomentumSGD applies
            #      its own Jenks-based elementwise masking and adaptive momentum
            #   4. calls optimizer.zero_grad()
            acc_res, acc5_res, loss = train_one_step_gsm(
                model, data, label, optimizer, criterion,
                nonzero_ratio=cfg["nonzero_ratio"],
            )
            # ──────────────────────────────────────────────────────────────

            ep_loss += loss.item()
            ep_acc  += acc_res.item()
            ep_acc5 += acc5_res.item()
            n_batches += 1

        scheduler.step()

        ep_loss /= n_batches
        ep_acc  /= n_batches
        ep_acc5 /= n_batches
        sparsity = measure_sparsity(model)

        t_writer.writerow([epoch + 1, f"{ep_loss:.4f}",
                           f"{ep_acc:.2f}", f"{ep_acc5:.2f}",
                           f"{sparsity:.4f}"])
        train_log.flush()

        # Validation
        val_loss, val_acc, val_acc5 = validate(model, val_loader, criterion, device)
        v_writer.writerow([epoch + 1, f"{val_loss:.4f}",
                           f"{val_acc:.2f}", f"{val_acc5:.2f}"])
        val_log.flush()

        elapsed = time.time() - t0
        print(
            f"  ep {epoch + 1:3d}/{cfg['epochs']}  "
            f"loss={ep_loss:.4f}  train={ep_acc:.1f}%  "
            f"val={val_acc:.1f}%  sparse={sparsity*100:.1f}%  "
            f"({elapsed:.0f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    train_log.close()
    val_log.close()

    final_sparsity = measure_sparsity(model)
    print(f"\n  Best val acc : {best_val_acc:.2f}%")
    print(f"  Final sparsity: {final_sparsity * 100:.1f}%")
    print(f"  Best checkpoint: {best_path}")

    return {
        "model":         key,
        "dataset":       entry.dataset,
        "epochs":        cfg["epochs"],
        "nonzero_ratio": cfg["nonzero_ratio"],
        "best_val_acc":  f"{best_val_acc:.2f}",
        "final_sparsity":f"{final_sparsity:.4f}",
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train CV models from scratch with GSM + ElementwiseMomentumSGD"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Registry key of the model to train (e.g. 'ResNet56', 'VGG19/CIFAR10'). "
             "Omit to run all experiments sequentially.",
    )
    parser.add_argument(
        "--out_dir", type=str, default="gsm_elem_sgd_results",
        help="Directory for logs and checkpoints.",
    )
    parser.add_argument(
        "--data_root", type=str, default="./datasets",
        help="Root directory for datasets.",
    )
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"Device : {device}")
    print(f"Output : {out_dir}")

    keys = [args.model] if args.model else list(EXPERIMENTS.keys())

    summary_rows = []
    for key in keys:
        result = run_experiment(key, device, out_dir, args.data_root)
        if result:
            summary_rows.append(result)

    if summary_rows:
        summary_path = out_dir / f"summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        fieldnames   = ["model", "dataset", "epochs", "nonzero_ratio",
                        "best_val_acc", "final_sparsity"]
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()

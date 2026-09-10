"""Run V:N:M auto-hybrid across ALL Best_Results_HPO networks (VGG-19, ResNet32/56,
DenseNet40, LeNet5) over their datasets (CIFAR-10/100 @32, TinyImageNet @64, MNIST @28)
and report the guaranteed (>=1x) whole-network speedup for each. fp16, RTX 4080."""
import os, sys, copy, argparse
import torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vnm_autohybrid import auto_hybrid, load_spatha, _t_ms, _build_vgg19
DEV = "cuda"

# (tag, arch, ckpt, num_classes, input_hw, in_ch)
MANIFEST = [
  ("VGG19  CIFAR10   @32", "vgg19", "Best_Results_HPO/VGG-19/CIFAR-10/90_sparsity/best_2026-06-09_13-46-27_cifar10_vgg19.pth", 10, 32, 3),
  ("VGG19  CIFAR100  @32", "vgg19", "Best_Results_HPO/VGG-19/CIFAR-100/90_sparsity/best_2026-06-10_10-37-39_cifar100_vgg19.pth", 100, 32, 3),
  ("VGG19  TinyImgNet@64", "vgg19", "Best_Results_HPO/VGG-19/TinyImageNet/best_2025-12-31_11-17-06_tiny_imagenet_vgg19.pth", 200, 64, 3),
  ("ResNet32 CIFAR10 @32", "resnet32", "Best_Results_HPO/ResNet32/CIFAR-10/95_sparsity/best_2026-06-11_09-09-44_cifar10_ResNet_32.pth", 10, 32, 3),
  ("ResNet32 CIFAR100@32", "resnet32", "Best_Results_HPO/ResNet32/CIFAR-100/86_Sparsity/best_2025-12-03_10-14-41_cifar100_ResNet_32.pth", 100, 32, 3),
  ("ResNet32 TinyImg @64", "resnet32", "Best_Results_HPO/ResNet32/TinyImageNet/best_2025-12-15_10-04-01_tiny_imagenet_ResNet_32.pth", 200, 64, 3),
  ("ResNet56 CIFAR10 @32", "resnet56", "Best_Results_HPO/ResNet56/best_2025-11-16_18-44-00_CIFAR10_ResNet56.pth", 10, 32, 3),
  ("DenseNet40 CIFAR10@32","densenet40", "Best_Results_HPO/DenseNet40/best_2025-11-16_18-44-04_CIFAR10_DenseNet40.pth", 10, 32, 3),
  ("LeNet5  MNIST    @28", "lenet5", "Best_Results_HPO/LeNet5/best_2025-12-08_15-03-41_MNIST_LeNet5.pth", 10, 28, 1),
]


def build_model(arch, nc):
    if arch == "vgg19":
        return _build_vgg19(nc)
    if arch == "resnet32":
        from resnet import resnet32
        return resnet32(num_classes=nc)
    if arch == "resnet56":
        from resnet import resnet56
        return resnet56()
    if arch == "densenet40":
        from densenet import create_densenet40
        return create_densenet40()
    if arch == "lenet5":
        from networks import LeNet5V1
        return LeNet5V1()
    raise ValueError(arch)


def run_one(tag, arch, ckpt, nc, hw, in_ch, spatha, tileM, B, margin):
    try:
        model = build_model(arch, nc)
    except Exception as e:
        print(f"\n===== {tag} ===== BUILD FAIL: {str(e)[:70]}"); return (tag, "build-fail", 0, 0, float('nan'))
    try:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
    except Exception as e:
        print(f"  [{tag}] ckpt warn: {str(e)[:60]}")
    model = model.to(DEV).half().eval()
    x = torch.randn(B, in_ch, hw, hw, device=DEV, dtype=torch.half)
    try:
        dense = copy.deepcopy(model).eval()
        print(f"\n===== {tag}  (B{B}, tileM{tileM}) =====")
        hybrid, rep = auto_hybrid(model, x, spatha, tileM=tileM, margin=margin, verbose=False)
        nf = sum(1 for r in rep if "FUSED" in r[7]); ncv = len(rep)
        td = _t_ms(lambda z: dense(z), x); th = _t_ms(lambda z: hybrid(z), x)
        sp = td/th
        print(f"  fused {nf}/{ncv} convs | WHOLE-NET dense={td:.3f}ms hybrid={th:.3f}ms => {sp:.2f}x")
        return (tag, f"{nf}/{ncv}", td, th, sp)
    except Exception as e:
        print(f"  [{tag}] RUN FAIL: {str(e)[:80]}"); return (tag, "run-fail", 0, 0, float('nan'))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128); ap.add_argument("--tileM", type=int, default=64)
    ap.add_argument("--margin", type=float, default=1.15)
    args = ap.parse_args()
    spatha = load_spatha(args.tileM)
    rows = [run_one(*m, spatha, args.tileM, args.batch, args.margin) for m in MANIFEST]
    print("\n===== HPO SUITE SUMMARY (tileM={}, B{}, margin{}) =====".format(args.tileM, args.batch, args.margin))
    print(f"{'network':22s} {'fused':>8} {'dense_ms':>9} {'hybrid_ms':>10} {'speedup':>8}")
    for tag, fu, td, th, sp in rows:
        s = f"{sp:.2f}x" if sp==sp else "n/a"
        print(f"{tag:22s} {fu:>8} {td:>9.3f} {th:>10.3f} {s:>8}")

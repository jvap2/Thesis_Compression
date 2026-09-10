"""Run the auto-hybrid V:N:M fusion across our saved sparse networks and report the
guaranteed (>=1x) whole-network speedup each one gets. CIFAR-10, fp16, RTX 4080."""
import os, sys, copy, argparse
import torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vnm_autohybrid import auto_hybrid, load_spatha, _t_ms, _build_vgg19
DEV = "cuda"


def build(name):
    if name == "vgg19":
        from vnm_autohybrid import _build_vgg19
        return _build_vgg19(), "models/best_2026-06-09_13-46-27_cifar10_vgg19.pth"
    if name == "resnet56":
        from resnet import resnet56
        return resnet56(), "models/best_2026-08-18_22-18-26_CIFAR10_ResNet56.pth"
    if name == "resnet32":
        from resnet import resnet32
        return resnet32(), "models/best_2026-06-11_09-09-44_cifar10_ResNet_32.pth"
    if name == "densenet40":
        from densenet import create_densenet40
        return create_densenet40(), "Best_Results_HPO/DenseNet40/best_2025-11-16_18-44-04_CIFAR10_DenseNet40.pth"
    raise ValueError(name)


def run_one(name, spatha, tileM, B, margin):
    model, ckpt = build(name)
    try:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
    except Exception as e:
        print(f"  [{name}] ckpt load warn: {str(e)[:60]}")
    model = model.to(DEV).half().eval()
    x = torch.randn(B, 3, 32, 32, device=DEV, dtype=torch.half)
    dense = copy.deepcopy(model).eval()
    print(f"\n===== {name}  (B{B}, tileM{tileM}) =====")
    hybrid, rep = auto_hybrid(model, x, spatha, tileM=tileM, margin=margin, verbose=False)
    n_fused = sum(1 for r in rep if "FUSED" in r[7])
    n_conv = len(rep)
    td = _t_ms(lambda z: dense(z), x); th = _t_ms(lambda z: hybrid(z), x)
    print(f"  fused {n_fused}/{n_conv} convs | WHOLE-NET dense={td:.3f}ms hybrid={th:.3f}ms "
          f"=> {td/th:.2f}x")
    # show the fused layers + their per-layer wins
    wins = [(r[0], r[5]/r[6]) for r in rep if "FUSED" in r[7]]
    if wins:
        print("   fused layers:", ", ".join(f"{n.split('.')[-2] if '.' in n else n}:{s:.2f}x"
                                            for n, s in wins[:12]))
    return name, n_fused, n_conv, td, th, td/th


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", default="vgg19,resnet56,densenet40")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--tileM", type=int, default=64)
    ap.add_argument("--margin", type=float, default=1.0)
    args = ap.parse_args()
    spatha = load_spatha(args.tileM)
    rows = []
    for net in args.nets.split(","):
        try:
            rows.append(run_one(net.strip(), spatha, args.tileM, args.batch, args.margin))
        except Exception as e:
            print(f"  [{net}] FAILED: {str(e)[:80]}")
    print("\n===== SUITE SUMMARY (tileM={}, B{}) =====".format(args.tileM, args.batch))
    print(f"{'network':12s} {'fused':>8} {'dense_ms':>9} {'hybrid_ms':>10} {'speedup':>8}")
    for name, nf, nc, td, th, sp in rows:
        print(f"{name:12s} {nf:>3}/{nc:<4} {td:>9.3f} {th:>10.3f} {sp:>7.2f}x")

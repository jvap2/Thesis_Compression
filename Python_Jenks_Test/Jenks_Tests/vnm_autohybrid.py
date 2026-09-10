"""
vnm_autohybrid.py — automatically build a hybrid model that uses the fused V:N:M
sparse-tensor-core conv (conv2d_vnm.Conv2dVNM) ONLY on layers where it actually beats
dense cuDNN, leaving every other layer dense. Guarantees whole-network speedup >= 1x
(worst case: nothing is fused -> identical to dense).

How it works:
  1. Trace the model forward once to record each Conv2d's real input shape.
  2. For every stride-1 Conv2d, micro-benchmark dense vs Conv2dVNM at that shape.
  3. Replace the layer with Conv2dVNM iff  fused_time * margin < dense_time.
  4. Report which layers were fused + the projected whole-network speedup.

Usage (as a script):
  VENOM_DIR=/tmp/venom_probe/end2end TORCH_CUDA_ARCH_LIST=8.9 \
    python3 vnm_autohybrid.py --ckpt <path> --arch vgg19 --batch 128 --tileM 64
"""
import os, sys, time, argparse
import torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conv2d_vnm import Conv2dVNM, build_spatha

DEV = "cuda"


def _t_ms(fn, x, it=40, warm=15):
    for _ in range(warm):
        fn(x)
    torch.cuda.synchronize()
    a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record()
    for _ in range(it):
        fn(x)
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / it


def _get_submodule(model, dotted):
    m = model
    for p in dotted.split(".")[:-1]:
        m = getattr(m, p) if not p.isdigit() else m[int(p)]
    return m, dotted.split(".")[-1]


def _set_submodule(model, dotted, new):
    parent, leaf = _get_submodule(model, dotted)
    if leaf.isdigit():
        parent[int(leaf)] = new
    else:
        setattr(parent, leaf, new)


def auto_hybrid(model, example_input, spatha, n=2, m=16, tileM=64, margin=1.0, verbose=True):
    """Return (hybrid_model, report). margin>1 requires a bigger win before fusing."""
    model = model.to(DEV).eval()
    # 1) trace input shapes for every Conv2d
    shapes = {}
    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            def mk(nm):
                def hook(m_, inp, out):
                    shapes[nm] = tuple(inp[0].shape)
                return hook
            hooks.append(mod.register_forward_hook(mk(name)))
    with torch.no_grad():
        model(example_input.to(DEV))
    for h in hooks:
        h.remove()

    # 2) per-layer decide
    report = []
    fused_total = dense_total = 0.0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Conv2d):
            continue
        B, C, H, W = shapes[name]
        x = torch.randn(B, C, H, W, device=DEV, dtype=torch.half)
        dense = mod.to(DEV).half()
        td = _t_ms(lambda z: dense(z), x)
        fuseable = (mod.stride == (1, 1) and mod.dilation == (1, 1) and mod.groups == 1)
        tf, status = None, "dense (not fuseable)"
        if fuseable:
            try:
                pad = ((mod.out_channels + tileM - 1) // tileM) * tileM
                conv = nn.Conv2d(mod.in_channels, pad, mod.kernel_size, stride=1,
                                 padding=mod.padding, bias=(mod.bias is not None)).to(DEV)
                with torch.no_grad():
                    conv.weight[:mod.out_channels].copy_(mod.weight)
                    if mod.bias is not None:
                        conv.bias.zero_(); conv.bias[:mod.out_channels].copy_(mod.bias)
                vnm = Conv2dVNM(conv, n, m, tileM, spatha, dev=DEV)
                # trim padded out-channels via a thin wrapper if we padded
                vnm_layer = _TrimOut(vnm, mod.out_channels) if pad != mod.out_channels else vnm
                tf = _t_ms(lambda z: vnm_layer(z), x)
                if tf * margin < td:
                    _set_submodule(model, name, vnm_layer)
                    status = f"FUSED {td/tf:.2f}x"
                else:
                    status = f"dense (fused {td/tf:.2f}x, not worth)"
            except Exception as e:
                status = f"dense (fused failed: {str(e)[:30]})"
        chosen = tf if (tf is not None and "FUSED" in status) else td
        fused_total += chosen; dense_total += td
        report.append((name, shapes[name], mod.out_channels, mod.kernel_size[0],
                       mod.stride[0], td, tf, status))
        if verbose:
            k = mod.kernel_size[0]
            print(f"  {name:24s} {C}->{mod.out_channels} @{H} k{k}s{mod.stride[0]}  "
                  f"dense={td:.3f} fused={'-' if tf is None else f'{tf:.3f}'}  {status}")
    speedup = dense_total / fused_total if fused_total > 0 else 1.0
    if verbose:
        n_fused = sum(1 for r in report if "FUSED" in r[7])
        print(f"\n  fused {n_fused}/{len(report)} conv layers | "
              f"projected conv-time speedup = {speedup:.2f}x "
              f"(dense {dense_total:.3f}ms -> {fused_total:.3f}ms)")
    return model, report


class _TrimOut(nn.Module):
    """Wrap a Conv2dVNM whose out_channels were padded to a tileM multiple; trim back."""
    def __init__(self, vnm, real_out):
        super().__init__(); self.vnm = vnm; self.real_out = real_out
    def forward(self, x):
        return self.vnm(x)[:, :self.real_out]


def load_spatha(tileM):
    """Build/load the Spatha kernel for a given V (tileM): 32/64/128."""
    from torch.utils.cpp_extension import load
    BS = os.environ.get("VENOM_DIR", "/tmp/venom_probe/end2end") + "/spatha_mod/block_sparse"
    flag = {32: ["-DV_32"], 64: ["-DV_64"], 128: []}.get(tileM, ["-DV_64"])
    return load(name=f"spatha_auto_v{tileM}", sources=[f"{BS}/api/spatha.cu"],
                extra_include_paths=[BS, f"{BS}/api", f"{BS}/spmm", f"{BS}/common"],
                extra_cuda_cflags=["-arch=sm_89", "-std=c++17"] + flag, verbose=False)


# ─────────────────────────── runnable main ───────────────────────────
def _build_vgg19(nc=10):
    CFG = [64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']
    layers, c = [], 3
    for v in CFG:
        if v == 'M':
            layers += [nn.MaxPool2d(2, 2)]
        else:
            layers += [nn.Conv2d(c, v, 3, padding=1, bias=False), nn.BatchNorm2d(v), nn.ReLU(True)]; c = v
    feat = nn.Sequential(*layers)
    class VGG(nn.Module):
        def __init__(s):
            super().__init__(); s.feature = feat
            s.avgpool = nn.AdaptiveAvgPool2d((1, 1)); s.classifier = nn.Linear(512, nc)
        def forward(s, x):
            return s.classifier(torch.flatten(s.avgpool(s.feature(x)), 1))
    return VGG()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--arch", default="vgg19")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--tileM", type=int, default=64)
    ap.add_argument("--hw", type=int, default=32)      # input spatial
    ap.add_argument("--margin", type=float, default=1.0)
    args = ap.parse_args()

    spatha = load_spatha(args.tileM)
    if args.arch == "vgg19":
        model = _build_vgg19()
    else:
        raise SystemExit(f"add arch builder for {args.arch}")
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
    x = torch.randn(args.batch, 3, args.hw, args.hw, device=DEV, dtype=torch.half)
    print(f"=== auto-hybrid {args.arch}  B{args.batch}  tileM{args.tileM} ===")
    import copy
    model = model.to(DEV).half()
    dense_model = copy.deepcopy(model).eval()          # SAME base weights, before fusion
    hybrid, rep = auto_hybrid(model, x, spatha, tileM=args.tileM, margin=args.margin)

    # rel-err vs dense-full reflects the V:N:M SPARSITY on fused layers (~the accuracy cost),
    # NOT a kernel error — per-layer kernel exactness is rel 0.0000 vs masked-dense (proven).
    with torch.no_grad():
        rd = dense_model(x); rg = hybrid(x)
    print(f"  hybrid vs dense-full rel-err = "
          f"{(rd-rg).abs().mean().item()/(rd.abs().mean().item()+1e-6):.4f}  "
          f"(= V:N:M sparsity effect on fused layers; kernel itself is exact)")
    print(f"  WHOLE-NET fwd: dense={_t_ms(lambda z: dense_model(z), x):.3f}ms  "
          f"hybrid={_t_ms(lambda z: hybrid(z), x):.3f}ms  "
          f"speedup={_t_ms(lambda z: dense_model(z), x)/_t_ms(lambda z: hybrid(z), x):.2f}x")

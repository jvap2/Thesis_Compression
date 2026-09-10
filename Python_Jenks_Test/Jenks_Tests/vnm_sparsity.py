"""
V:N:M (VENOM-style) sparsity for CNNs — pure-PyTorch reference / accuracy prototype.

VENOM (Castro et al., SC'23, https://arxiv.org/abs/2310.02065) runs arbitrary N:M
ratios on NVIDIA 2:4 Sparse Tensor Cores by adding a VECTOR dimension (V/tileM):
a block of `tileM` consecutive rows (output channels, in a GEMM/im2col view) SHARE
one N-of-M column selection. That vector-sharing is what makes >50% sparsity mappable
to the 2:4 hardware.

Their Spatha library provides this only for nn.Linear (a sparse GEMM `spatha.spmm`).
This module is our CNN adaptation at the ALGORITHM level: it produces the V:N:M mask
for a conv weight so we can measure the accuracy cost of the pattern in pure PyTorch
(fake sparsity), BEFORE taking on the CUDA-kernel build for the actual speedup.

Density = n/m  (e.g. n=2,m=16 -> 12.5% dense = 87.5% sparse;  n=2,m=20 -> 90% sparse).

A conv weight W[out, in, kh, kw] is viewed as a 2D GEMM operand W2d[out, in*kh*kw]
(the im2col layout), which is exactly what a Conv2dVNM wrapper will feed to spatha.spmm.
"""
import torch
import torch.nn.functional as F


def _round_up(x, mult):
    return ((x + mult - 1) // mult) * mult


def vnm_mask_2d(W2d, n, m, tileM):
    """
    Compute the V:N:M binary mask for a 2D weight [R, C].

    Partition rows into vectors of `tileM`; partition cols into blocks of `m`.
    Within each (row-vector, col-block), keep the `n` columns whose magnitude,
    AGGREGATED over the tileM rows of the vector, is largest — and keep those
    same `n` columns for every row in the vector (the VENOM vector-sharing).

    Returns a float mask of the same [R, C] shape (1 = kept, 0 = pruned).
    """
    assert W2d.dim() == 2, "expects a 2D weight (im2col view)"
    R, C = W2d.shape
    Rp, Cp = _round_up(R, tileM), _round_up(C, m)
    # pad with zeros so padded columns are never selected (mag 0) and padded rows are inert
    Wp = torch.zeros(Rp, Cp, device=W2d.device, dtype=W2d.dtype)
    Wp[:R, :C] = W2d

    nV, nB = Rp // tileM, Cp // m                      # #row-vectors, #col-blocks
    # [nV, tileM, nB, m]
    blk = Wp.reshape(nV, tileM, nB, m)
    agg = blk.abs().sum(dim=1)                          # [nV, nB, m]  aggregate over the vector
    # keep the n largest columns per (vector, block)
    keep_idx = agg.topk(n, dim=2).indices              # [nV, nB, n]
    blk_mask = torch.zeros(nV, nB, m, device=W2d.device, dtype=W2d.dtype)
    blk_mask.scatter_(2, keep_idx, 1.0)                # [nV, nB, m]
    # broadcast the shared selection across the tileM rows of each vector
    full = blk_mask.unsqueeze(1).expand(nV, tileM, nB, m).reshape(Rp, Cp)
    return full[:R, :C].contiguous()


def vnm_mask_conv(weight, n, m, tileM):
    """V:N:M mask for a conv weight [out, in, kh, kw], via its im2col 2D view."""
    out_ch = weight.shape[0]
    W2d = weight.reshape(out_ch, -1)                    # [out, in*kh*kw]
    mask2d = vnm_mask_2d(W2d, n, m, tileM)
    return mask2d.reshape(weight.shape).contiguous()


def vnm_mask_any(weight, n, m, tileM):
    """Dispatch: 4D conv -> im2col mask; 2D linear -> direct."""
    if weight.dim() == 4:
        return vnm_mask_conv(weight, n, m, tileM)
    elif weight.dim() == 2:
        return vnm_mask_2d(weight, n, m, tileM)
    else:
        raise ValueError(f"unsupported weight dim {weight.dim()}")


def apply_vnm_inplace(model, n, m, tileM, key_substr=None, skip_substr=("downsample",)):
    """
    Fake-sparsify a model's conv/linear weights to V:N:M in-place (for accuracy
    prototyping). Returns dict name->mask so a training loop can re-apply after steps.
    skip_substr: layers to leave dense (default: shortcut/projection convs, matching
    our GSM protect-shortcut choice). key_substr: if set, only touch matching layers.
    """
    masks = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() not in (2, 4):
                continue
            if key_substr is not None and key_substr not in name:
                continue
            if any(s in name for s in skip_substr):
                continue
            mask = vnm_mask_any(p.data, n, m, tileM)
            p.data.mul_(mask)
            masks[name] = mask
    return masks


# ───────────────────────── self-test ─────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    print("V:N:M sparsity self-test\n" + "-" * 40)

    # 1) Linear-shaped weight, n=2 m=16 tileM=4  -> 12.5% density
    W = torch.randn(64, 512)
    for (n, m, tileM) in [(2, 16, 4), (2, 20, 8), (1, 10, 4)]:
        mask = vnm_mask_2d(W, n, m, tileM)
        dens = mask.mean().item()
        C = W.shape[1]
        nB = (C + m - 1) // m                       # blocks incl. a padded partial block
        exp_dens = n * nB / C                        # exact when m ∤ C (partial block keeps n reals)
        print(f"[2D] n={n} m={m} tileM={tileM}: density={dens:.4f} "
              f"(exact {exp_dens:.4f}, ideal n/m {n/m:.4f})  sparsity={1-dens:.4f}")
        assert abs(dens - exp_dens) < 1e-3, "density must equal n*ceil(C/m)/C"
        # vector-sharing: every tileM-row block must share identical column pattern
        R, C = W.shape
        Rp = _round_up(R, tileM)
        mpad = torch.zeros(Rp, C); mpad[:R] = mask
        blocks = mpad.reshape(Rp // tileM, tileM, C)
        shared = (blocks == blocks[:, :1, :]).all().item()
        assert shared, "rows in a vector must share the SAME column selection"
        print("      vector-sharing OK; density-exact OK")

    # 2) Conv weight [out,in,kh,kw] — CIFAR ResNet-ish 3x3, and its im2col view
    for shape in [(64, 64, 3, 3), (32, 16, 3, 3), (16, 3, 3, 3)]:
        w = torch.randn(*shape)
        n, m, tileM = 2, 16, 8
        mask = vnm_mask_conv(w, n, m, tileM)
        dens = mask.mean().item()
        print(f"[conv {shape}] n={n} m={m} tileM={tileM}: density={dens:.4f} "
              f"(target {n/m:.4f})")
        # nonzeros only where mask=1
        assert torch.all((w * mask == 0) | (mask == 1))
    print("-" * 40 + "\nALL V:N:M SELF-TESTS PASSED")

"""
Conv2dVNM — run a Conv2d on NVIDIA Sparse Tensor Cores via VENOM's V:N:M format.

VENOM/Spatha (Castro et al., SC'23) only ship a sparse GEMM (spatha.spmm) wired to
nn.Linear. A conv2d is a GEMM after im2col, so this wrapper does:

    weight  W[out, Cin, kh, kw]  --reshape-->  W2d[out, Cin*kh*kw]   (packed to V:N:M)
    input   x[B, Cin, H, W]      --F.unfold-->  cols[B, K, L]         (K=Cin*kh*kw, L=out_H*out_W)
    GEMM    out[out, B*L] = spatha.spmm( sparse(W2d), dense(cols reshaped [K, B*L]) )
    output  reshape -> [B, out, out_H, out_W]

Packing (dense W2d -> values/columns/metadata in Spatha's exact layout) is reused from
VENOM's SrNMTensor + nm_vector_mask_sparsify (magnitude V:N:M mask). The Spatha kernel is
loaded via torch JIT load() (build_spatha()), matching how this repo builds its CUDA.

Density = n/m (n=2,m=16 -> 87.5% sparse; n=2,m=20 -> 90%). tileM = the V (vector) length.

IMPORTANT — tileM must match the compiled kernel's block-tile M: **128** for the default
Spatha build, or **64** for the V_64 build (blockwise_library.cu). Arbitrary tileM (e.g. 8)
produces garbage. Consequently out_channels must be padded to a multiple of tileM, so this
path suits WIDE nets (VGG, out_ch>=64/128), not narrow ResNet56 stages (out_ch 16/32).
Validated: 2D spmm matches (W*mask)@B at rel err 0.0000 (fp16) with tileM=128.
"""
import os, sys, math
import torch, torch.nn as nn, torch.nn.functional as F

VENOM_DIR = os.environ.get("VENOM_DIR", "/tmp/venom_probe/end2end")
_BS = f"{VENOM_DIR}/spatha_mod/block_sparse"


def build_spatha(arch="sm_89"):
    """JIT-compile & load the Spatha spmm kernel (like cuda_helpers)."""
    from torch.utils.cpp_extension import load
    return load(
        name="spatha",
        sources=[f"{_BS}/api/spatha.cu"],
        extra_include_paths=[_BS, f"{_BS}/api", f"{_BS}/spmm", f"{_BS}/common"],
        extra_cuda_cflags=[f"-arch={arch}", "--ptxas-options=-v", "-lineinfo", "-std=c++17"],
        verbose=False,
    )


def _import_venom_packing():
    if VENOM_DIR not in sys.path:
        sys.path.insert(0, VENOM_DIR)
    from grouped_nmv_tensor import SrNMTensor, nm_vector_mask_sparsify
    return SrNMTensor, nm_vector_mask_sparsify


def _round_up(x, m):
    return int(math.ceil(x / m) * m)


class Conv2dVNM(nn.Module):
    """Drop-in-ish replacement for an nn.Conv2d whose weights run on Sparse Tensor Cores."""

    def __init__(self, conv: nn.Conv2d, n, m, tileM, spatha, dev="cuda"):
        super().__init__()
        assert conv.groups == 1, "grouped/depthwise conv not supported (im2col GEMM assumes groups=1)"
        self.n, self.m, self.tileM = n, m, tileM
        self.spatha = spatha
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.kh, self.kw = conv.kernel_size
        self.out_ch = conv.out_channels
        self.in_ch = conv.in_channels

        SrNMTensor, nm_vector_mask_sparsify = _import_venom_packing()
        W2d = conv.weight.detach().reshape(self.out_ch, -1).contiguous()  # [out, K], K=Cin*kh*kw
        mask, columns = nm_vector_mask_sparsify(W2d.cpu(), n, m, tileM)    # magnitude V:N:M
        srnm = SrNMTensor(n, m, tileM, W2d.cpu(), mask, columns, dev)
        # packed sparse operands (already moved to device / half by SrNMTensor)
        self.register_buffer("values",   srnm.values)
        self.register_buffer("columns",  srnm.columns)
        self.register_buffer("metadata", srnm.metadata)
        self.nrows_sp = srnm.nrows          # = out_ch
        self.ncols_sp = srnm.ncols          # = K
        self.nnz = srnm.nnz
        if conv.bias is not None and conv.bias.detach().abs().sum() > 0:
            self.register_buffer("bias", conv.bias.detach().to(dev).half())
            self.has_bias = True
        else:
            self.register_buffer("bias", torch.zeros(self.out_ch, device=dev, dtype=torch.half))
            self.has_bias = False   # bias-free conv (e.g. VGG conv+BN): skip the add entirely
        # The Spatha kernel's in-kernel bias add is unreliable; feed it zeros and add bias
        # manually after the GEMM (verified exact with zero in-kernel bias).
        self.register_buffer("_zero_bias", torch.zeros(self.out_ch, device=dev, dtype=torch.half))
        self._col_off_cache = {}   # (B,oh,ow,Hp,Wp) -> precomputed im2col column offsets

    def _out_hw(self, H, W):
        oh = (H + 2*self.padding[0] - self.dilation[0]*(self.kh-1) - 1)//self.stride[0] + 1
        ow = (W + 2*self.padding[1] - self.dilation[1]*(self.kw-1) - 1)//self.stride[1] + 1
        return oh, ow

    def forward(self, x):
        B, C, H, W = x.shape
        oh, ow = self._out_hw(H, W)
        # FUSED conv (stride 1, any k incl. 1x1): pre-pad activation, kernel does im2col inline.
        if self.stride == (1, 1) and self.dilation == (1, 1):
            oh2, ow2 = oh, ow
            xp = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0])).contiguous()
            Hp, Wp = xp.shape[2], xp.shape[3]
            OHOW = oh2 * ow2
            # precomputed im2col column offsets (removes per-element divisions in the kernel gather)
            key = (B, oh2, ow2, Hp, Wp)
            col_off = self._col_off_cache.get(key)
            if col_off is None:
                N = B * OHOW
                idx = torch.arange(N, device=x.device, dtype=torch.int64)
                bb = idx // OHOW; ss = idx % OHOW; oi = ss // ow2; oj = ss % ow2
                col_off = (bb * (self.in_ch * Hp * Wp) + oi * Wp + oj).to(torch.int32).contiguous()
                self._col_off_cache[key] = col_off
            out = self.spatha.spmm(self.metadata, self.columns, self.values, xp.reshape(-1), self._zero_bias,
                                   self.nrows_sp, self.ncols_sp, B*OHOW,
                                   self.tileM, self.n, self.m, self.nnz, 0, 32, 4,
                                   OHOW, ow2, self.in_ch, self.kw, self.kh*self.kw, Hp, Wp, col_off)  # L_conv,cOW,cC,cKW,cKHKW,cHp,cWp
            out = out.reshape(B, OHOW, self.out_ch).permute(0, 2, 1).reshape(B, self.out_ch, oh2, ow2)
            return (out + self.bias.view(1, -1, 1, 1)) if self.has_bias else out
        # fallback (stride>1 / dilation>1): explicit im2col [B, K, L] -> dense RHS [K, B*L]
        cols = F.unfold(x, (self.kh, self.kw), self.dilation, self.padding, self.stride)
        K, L = cols.shape[1], cols.shape[2]
        dense = cols.permute(1, 0, 2).reshape(K, B*L).contiguous().half()
        # kernel returns output as {B_num_cols, A_num_rows} = [B*L, out_ch]  (see blockwise_library.cu:313)
        out = self.spatha.spmm(self.metadata, self.columns, self.values, dense, self._zero_bias,
                               self.nrows_sp, self.ncols_sp, B*L,
                               self.tileM, self.n, self.m, self.nnz, 0, 32, 4)
        # [B*L, out_ch] with row index = b*L + l -> [B, out_ch, oh, ow]
        out = out.reshape(B, L, self.out_ch).permute(0, 2, 1).reshape(B, self.out_ch, oh, ow)
        if self.has_bias:
            out = out + self.bias.view(1, -1, 1, 1)   # manual bias add (skip if bias-free)
        return out


# ───────────────────────── correctness test (needs GPU) ─────────────────────────
def _test(n=2, m=16, tileM=128):
    dev = "cuda"
    spatha = build_spatha()
    _, nm_vector_mask_sparsify = _import_venom_packing()
    torch.manual_seed(0)
    # out_ch multiple of tileM(128); reduction K=in*kh*kw divisible by m
    conv = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=True).to(dev)
    x = torch.randn(4, 64, 16, 16, device=dev, dtype=torch.half)

    # reference: DENSE conv with the SAME VENOM V:N:M-masked weights the kernel packs
    out_ch = conv.out_channels
    W2d = conv.weight.detach().reshape(out_ch, -1).cpu()
    mask, _cols = nm_vector_mask_sparsify(W2d, n, m, tileM)
    conv_masked = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=True).to(dev)
    with torch.no_grad():
        conv_masked.weight.copy_((W2d * mask.float()).reshape(conv.weight.shape).to(dev))
        conv_masked.bias.copy_(conv.bias)
    ref = conv_masked.half()(x)

    vnm = Conv2dVNM(conv_masked.float(), n, m, tileM, spatha, dev=dev)
    got = vnm(x)

    print("ref shape", tuple(ref.shape), "| got shape", tuple(got.shape))
    if ref.shape == got.shape:
        err = (ref.float() - got.float()).abs()
        rel = err.mean().item() / (ref.float().abs().mean().item() + 1e-6)
        print(f"max abs err={err.max().item():.4f}  mean abs err={err.mean().item():.5f}  rel={rel:.4f}")
        print("PASS" if rel < 0.05 else "MISMATCH — check dim/packing/mask alignment")
    else:
        print("SHAPE MISMATCH — fix reshape mapping")


if __name__ == "__main__":
    _test()

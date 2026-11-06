"""Experimental fused CUDA linear algebra for existing GGUF block formats.

Weights are normally decoded inside each matrix-multiplication tile. For large
Q8 matrix multiplications, a temporary half-precision weight matrix allows the
device's optimized dense multiplication; it is released after that operation.
This module is imported only when the experimental fused path is enabled.
"""
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

# The bundled Windows Triton launcher contains initializers not accepted by
# MSVC in C11 mode. Its bundled Tiny C compiler accepts them. Respect an
# explicit compiler choice and do not alter the user's system environment.
if os.name == 'nt' and 'CC' not in os.environ:
    bundled_cc = Path(triton.__file__).resolve().parent / 'runtime' / 'tcc' / 'tcc.exe'
    if bundled_cc.is_file():
        os.environ['CC'] = str(bundled_cc)


@triton.jit
def _unpack_q8(W, D, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < COUNT
    base = (i // 32) * 34
    lo = tl.load(W + base, mask=valid, other=0).to(tl.uint16)
    hi = tl.load(W + base + 1, mask=valid, other=0).to(tl.uint16)
    scale = (lo | (hi << 8)).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    code = tl.load(W + base + 2 + i % 32, mask=valid, other=0).to(tl.int8, bitcast=True)
    tl.store(D + i, scale * code.to(tl.float32), mask=valid)


@triton.jit
def _linear_kernel(X, W, BIAS, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   KIND: tl.constexpr, HAS_BIAS: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k0 = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for start in range(0, K, BK):
        k = start + k0
        x = tl.load(X + m[:, None] * K + k[None, :],
                    mask=(m[:, None] < M) & (k[None, :] < K), other=0)
        if KIND == 4 or KIND == 8:
            size: tl.constexpr = 18 if KIND == 4 else 34
            base = (n[None, :] * (K // 32) + k[:, None] // 32) * size
            valid = (n[None, :] < N) & (k[:, None] < K)
            lo = tl.load(W + base, mask=valid, other=0).to(tl.uint16)
            hi = tl.load(W + base + 1, mask=valid, other=0).to(tl.uint16)
            scale = (lo | (hi << 8)).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            if KIND == 4:
                offset = k[:, None] % 32
                byte = tl.load(W + base + 2 + offset % 16, mask=valid, other=0)
                code = ((byte >> ((offset // 16) * 4)) & 15).to(tl.int32) - 8
            else:
                byte = tl.load(W + base + 2 + k[:, None] % 32, mask=valid, other=0)
                code = byte.to(tl.int8, bitcast=True).to(tl.int32)
            weight = scale * code.to(tl.float32)
        else:
            base = (n[None, :] * (K // 256) + k[:, None] // 256) * 144
            valid = (n[None, :] < N) & (k[:, None] < K)
            d0 = tl.load(W + base, mask=valid, other=0).to(tl.uint16)
            d1 = tl.load(W + base + 1, mask=valid, other=0).to(tl.uint16)
            m0 = tl.load(W + base + 2, mask=valid, other=0).to(tl.uint16)
            m1 = tl.load(W + base + 3, mask=valid, other=0).to(tl.uint16)
            d = (d0 | (d1 << 8)).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            dmin = (m0 | (m1 << 8)).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            group = (k[:, None] % 256) // 32
            j = group % 4
            a = tl.load(W + base + 4 + j, mask=valid, other=0)
            b = tl.load(W + base + 8 + j, mask=valid, other=0)
            c = tl.load(W + base + 12 + j, mask=valid, other=0)
            scale = tl.where(group < 4, a & 63, (c & 15) | ((a >> 2) & 48))
            minimum = tl.where(group < 4, b & 63, (c >> 4) | ((b >> 2) & 48))
            byte = tl.load(W + base + 16 + (group // 2) * 32 + k[:, None] % 32,
                           mask=valid, other=0)
            code = (byte >> ((group % 2) * 4)) & 15
            weight = d * scale.to(tl.float32) * code.to(tl.float32) - dmin * minimum.to(tl.float32)
        acc += tl.dot(x, weight.to(x.dtype))
    if HAS_BIAS:
        bias = tl.load(BIAS + n, mask=n < N, other=0).to(tl.float32)
        # PyTorch's reduced-precision linear path rounds the matrix product
        # before adding bias. Preserve that boundary rather than fusing the
        # addition into the higher-precision accumulator.
        acc = acc.to(Y.dtype.element_ty).to(tl.float32) + bias[None, :]
    tl.store(Y + m[:, None] * N + n[None, :], acc,
             mask=(m[:, None] < M) & (n[None, :] < N))


def fused_linear(value, packed, shape, kind, bias=None):
    """Multiply FP16/BF16 activations by Q4_0, Q8_0 or Q4_K weights on CUDA."""
    if (not value.is_cuda or not packed.is_cuda or packed.dtype != torch.uint8
            or value.device != packed.device or value.dtype not in (torch.float16, torch.bfloat16)
            or len(shape) != 2 or value.shape[-1] != shape[1] or kind not in (4, 8, 44)):
        raise ValueError('unsupported fused linear input')
    if torch.is_grad_enabled() and (value.requires_grad or (bias is not None and bias.requires_grad)):
        raise ValueError('fused linear supports inference only')
    n, k = shape
    group, size = (256, 144) if kind == 44 else (32, 18 if kind == 4 else 34)
    if n < 1 or k < 1 or k % group or packed.numel() != n * (k // group) * size:
        raise ValueError('invalid packed linear dimensions')
    if bias is not None and (bias.shape != (n,) or bias.device != value.device or bias.dtype != value.dtype):
        raise ValueError('invalid fused linear bias')
    x = value.reshape(-1, k).contiguous()
    m = x.shape[0]
    if not m:
        return torch.empty((*value.shape[:-1], n), device=value.device, dtype=value.dtype)
    if kind == 8 and value.dtype == torch.float16 and m >= 4096 and n >= 1024 and k >= 1024:
        # The scratch matrix is scoped to this call; packed weights stay resident.
        # This trades transient memory for optimized dense-matmul throughput.
        dense = torch.empty((n, k), device=value.device, dtype=value.dtype)
        _unpack_q8[(triton.cdiv(n * k, 1024),)](packed, dense, n * k, 1024)
        return torch.nn.functional.linear(x, dense, bias).reshape(*value.shape[:-1], n)
    out = torch.empty((m, n), device=value.device, dtype=value.dtype)
    # Wider Q8 projections can reuse each decoded weight tile across more rows.
    # Keep narrower projections and short sequences on the smaller tiles.
    block_m = (128 if kind == 8 and m >= 4096 and (n >= 3072 or k >= 4096)
               else 64 if kind == 8 and m >= 4096 else 32)
    _linear_kernel[(triton.cdiv(m, block_m), triton.cdiv(n, 64))](
        x, packed, bias if bias is not None else packed, out, m, n, k, kind, bias is not None,
        block_m, 64, 32, num_warps=4)
    return out.reshape(*value.shape[:-1], n)

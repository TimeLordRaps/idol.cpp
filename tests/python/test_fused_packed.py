"""Opt-in CUDA checks against a float32-accumulating packed-weight reference."""
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

if os.environ.get('IDOL_TEST_GPU') != '1' or any(
        importlib.util.find_spec(name) is None for name in ('torch', 'gguf', 'triton', 'numpy')):
    print('skipped: set IDOL_TEST_GPU=1 with torch, gguf, numpy and Triton installed')
    sys.exit(77)

import gguf
import numpy as np
import torch

if not torch.cuda.is_available():
    print('skipped: no CUDA device')
    sys.exit(77)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from idol_packed import PackedLinear, enable_fused_packed
from idol_triton import fused_linear

torch.backends.cuda.matmul.allow_tf32 = False
rng = np.random.default_rng(144)
formats = ((gguf.GGMLQuantizationType.Q4_0, 4), (gguf.GGMLQuantizationType.Q8_0, 8),
           (gguf.GGMLQuantizationType.Q4_K, 44))
for kind, code in formats:
    shapes = [(1, 7, 256), (41, 71, 256), (5, 33, 512)]
    if kind == gguf.GGMLQuantizationType.Q8_0:
        shapes.extend(((4097, 71, 256), (4097, 3072, 32), (4097, 1024, 1024)))
    for rows, columns, width in shapes:
        if kind == gguf.GGMLQuantizationType.Q4_K:
            data = rng.integers(0, 256, size=(columns, width // 256 * 144), dtype=np.uint8)
            blocks = data.reshape(-1, 144)
            blocks[:, :4] = np.array([0, 60, 0, 56], dtype=np.uint8)
        else:
            weights = rng.normal(0, 0.25, (columns, width)).astype(np.float32)
            data = gguf.quantize(weights, kind)
        tensor = SimpleNamespace(shape=np.array([width, columns]), tensor_type=kind, data=data)
        layer = PackedLinear(tensor, None).cuda()
        for dtype in (torch.float16, torch.bfloat16):
            # A strided input also exercises the contiguous-copy boundary.
            base = torch.randn(rows, width * 2, device='cuda', dtype=dtype)
            value = base[:, ::2]
            for bias in (None, torch.randn(columns, device='cuda', dtype=dtype)):
                output = fused_linear(value, layer.packed, layer.shape, code, bias)
                layer.bias = torch.nn.Parameter(bias, requires_grad=False) if bias is not None else None
                assert enable_fused_packed(layer) == 1
                torch.testing.assert_close(layer(value), output, rtol=0, atol=0)
                dense = layer.dequantize(dtype)
                reference = (value.float() @ dense.float().T).to(dtype)
                if bias is not None:
                    reference = reference + bias
                delta = output.float() - reference.float()
                relative = delta.norm() / reference.float().norm().clamp_min(1e-12)
                if not torch.isfinite(output).all() or relative >= 0.001:
                    raise AssertionError(f'{kind.name} {dtype} relative error {float(relative)}')
            del base, value
        del layer
torch.cuda.synchronize()
print('fused packed matrix checks passed')

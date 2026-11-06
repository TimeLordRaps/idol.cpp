"""Exercise DINOv3 conversion metadata and all available block formats."""
import argparse
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in ('numpy', 'gguf', 'safetensors')):
    print('skipped: needs numpy, gguf, and safetensors')
    sys.exit(77)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import gguf
import numpy as np
from safetensors.numpy import save_file
from export_dinov3_gguf import export, parse_layers, runtime_bfloat16_storage
from export_full_gguf import digest

suffixes = ('attention.q_proj.weight', 'attention.k_proj.weight', 'attention.v_proj.weight',
            'attention.o_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight')
matrix = np.linspace(-1, 1, 256, dtype=np.float32).reshape(1, 256)
weights = {f'layer.{index}.{suffix}': matrix for index in range(32) for suffix in suffixes}
weights.update({f'other.{index}': np.array([index], dtype=np.float32) for index in range(391)})
assert len(weights) == 615
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    source = root / 'source.safetensors'
    save_file(weights, str(source))
    library = os.environ.get('IDOL_GGML_LIB')
    tiers = ('Q8_0', 'Q6_K', 'Q4_K', 'Q2_K') if library else ('Q8_0',)
    for tier in tiers:
        target = root / (tier + '.gguf')
        export(source, target, 'claimed-revision', tier, library)
        reader = gguf.GGUFReader(target)
        assert reader.get_field('general.architecture').contents() == 'dinov3'
        assert reader.get_field('general.quantization_version').contents() == gguf.GGML_QUANT_VERSION
        assert reader.get_field('source.revision_claimed').contents() == 'claimed-revision'
        assert reader.get_field('quantization.exporter.sha256').contents() == digest(Path(export.__code__.co_filename))
        if tier == 'Q8_0':
            assert reader.get_field('quantization.ggml.sha256') is None
        else:
            assert reader.get_field('quantization.ggml.sha256').contents() == digest(Path(library))
        selected = [tensor for tensor in reader.tensors if tensor.tensor_type == getattr(gguf.GGMLQuantizationType, tier)]
        assert len(reader.tensors) == 615 and len(selected) == 224
        reconstructed = gguf.dequantize(selected[0].data, selected[0].tensor_type).reshape(matrix.shape)
        assert np.isfinite(reconstructed).all()
        del selected, reader
        try:
            export(source, target, 'claimed-revision', tier, library)
        except FileExistsError:
            pass
        else:
            raise AssertionError('existing encoder file was overwritten')
    if importlib.util.find_spec('torch') is not None:
        import torch
        assert parse_layers('0-1,31') == (0, 1, 31)
        for invalid in ('-1', '0-32', '0-2,2', '3-1'):
            try:
                parse_layers(invalid)
            except argparse.ArgumentTypeError:
                pass
            else:
                raise AssertionError('invalid preserved layer selection accepted: ' + invalid)
        tiny = np.array([1e-30, 1.2345], dtype=np.float32)
        stored = runtime_bfloat16_storage(tiny)
        assert stored.dtype == np.uint8
        assert torch.equal(torch.from_numpy(gguf.dequantize(stored, gguf.GGMLQuantizationType.BF16).copy()).to(torch.bfloat16),
                           torch.from_numpy(tiny).to(torch.bfloat16))
        mixed = root / 'mixed.gguf'
        export(source, mixed, 'claimed-revision', 'Q8_0', preserve_layers=(0, 31))
        reader = gguf.GGUFReader(mixed)
        assert reader.get_field('quantization.preserved_layers').contents() == '[0, 31]'
        tensors = {tensor.name: tensor for tensor in reader.tensors}
        assert sum(t.tensor_type == gguf.GGMLQuantizationType.Q8_0 for t in tensors.values()) == 210
        for name in ('layer.0.attention.q_proj.weight', 'layer.31.mlp.down_proj.weight'):
            tensor = tensors[name]
            assert tensor.tensor_type == gguf.GGMLQuantizationType.BF16
            decoded = gguf.dequantize(tensor.data, tensor.tensor_type).reshape(matrix.shape).copy()
            assert torch.equal(torch.from_numpy(decoded).to(torch.bfloat16), torch.from_numpy(matrix).to(torch.bfloat16))
        assert tensors['layer.1.attention.q_proj.weight'].tensor_type == gguf.GGMLQuantizationType.Q8_0
        del tensor, decoded, reader, tensors
    else:
        print('mixed-precision DINOv3 tests skipped: torch is unavailable')
    if not library:
        print('K-block encoder export round trips skipped: IDOL_GGML_LIB is unset')
print('DINOv3 encoder export round trips passed')

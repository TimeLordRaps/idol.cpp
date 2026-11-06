"""Convert locally obtained official DINOv3 ViT-H/16+ weights to GGUF.

The original DINOv3 terms still apply. This tool does not grant redistribution
rights or upload the resulting file. Nonlinear tensors stay at source precision.
"""
import argparse
import importlib.metadata
import json
import os
import re
import tempfile
from pathlib import Path

import gguf
import numpy as np
from safetensors import safe_open

from export_full_gguf import GGMLQuantizer, digest

LINEAR = re.compile(r'^layer\.(?:[0-9]|[12][0-9]|3[01])\.(?:attention\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.weight$')


def runtime_bfloat16_storage(value):
    """Encode the exact original runtime bfloat16 values in GGUF form."""
    import torch
    original = torch.from_numpy(np.array(value, copy=True)).to(torch.bfloat16)
    if not bool(torch.isfinite(original).all()):
        raise ValueError('source cannot be represented in runtime bfloat16')
    encoded = gguf.quantize(original.float().cpu().numpy(), gguf.GGMLQuantizationType.BF16)
    restored = torch.from_numpy(gguf.dequantize(encoded, gguf.GGMLQuantizationType.BF16).copy()).to(torch.bfloat16)
    if not torch.equal(restored, original):
        raise ValueError('bfloat16 encoding changed the runtime value')
    return encoded


def parse_layers(value):
    if not value:
        return ()
    selected = []
    try:
        for part in value.split(','):
            bounds = part.split('-')
            if len(bounds) == 1:
                selected.append(int(bounds[0]))
            elif len(bounds) == 2:
                first, last = map(int, bounds)
                if first > last:
                    raise ValueError('reversed range')
                selected.extend(range(first, last + 1))
            else:
                raise ValueError('invalid range')
    except ValueError as exc:
        raise argparse.ArgumentTypeError('expected layer numbers or ranges such as 0-7,24-31') from exc
    if len(set(selected)) != len(selected) or any(index < 0 or index > 31 for index in selected):
        raise argparse.ArgumentTypeError('layer numbers must be unique and between 0 and 31')
    return tuple(sorted(selected))


def export(source, output, revision, tier='Q8_0', ggml_lib=None, preserve_layers=()):
    source, output = Path(source), Path(output)
    if not revision:
        raise ValueError('an exact source revision is required')
    if tier not in ('Q8_0', 'Q6_K', 'Q4_K', 'Q2_K'):
        raise ValueError('unsupported encoder quantization tier')
    if tier != 'Q8_0' and ggml_lib is None:
        raise ValueError('K-block tiers require an explicitly supplied GGML library')
    preserve_layers = tuple(preserve_layers)
    if (len(set(preserve_layers)) != len(preserve_layers)
            or any(type(index) is not int or index < 0 or index > 31 for index in preserve_layers)):
        raise ValueError('invalid original-precision DINOv3 layers')
    preserve_layers = tuple(sorted(preserve_layers))
    quantizer = GGMLQuantizer(ggml_lib) if tier != 'Q8_0' else None
    kind = getattr(gguf.GGMLQuantizationType, tier)
    block_size, _ = gguf.GGML_QUANT_SIZES[kind]
    original_hash = digest(source)
    exporter_hash = digest(Path(__file__))
    library_hash = digest(Path(ggml_lib)) if quantizer is not None else None
    try:
        gguf_version = importlib.metadata.version('gguf')
    except importlib.metadata.PackageNotFoundError:
        gguf_version = 'unknown'
    if output.exists():
        raise FileExistsError(output)
    partial = output.with_name(output.name + '.partial')
    with partial.open('xb'):
        pass
    writer = None
    old_tempdir = tempfile.tempdir
    try:
        tempfile.tempdir = str(output.parent.resolve())
        writer = gguf.GGUFWriter(partial, arch='dinov3', use_temp_file=True)
        writer.add_name('DINOv3 ViT-H/16+ ' + tier + (' mixed precision' if preserve_layers else ''))
        writer.add_string('source.repository_expected', 'facebook/dinov3-vith16plus-pretrain-lvd1689m')
        writer.add_string('source.revision_claimed', revision)
        writer.add_string('source.sha256', original_hash)
        writer.add_string('quantization.exporter.sha256', exporter_hash)
        writer.add_string('quantization.gguf_version', gguf_version)
        if library_hash is not None:
            writer.add_string('quantization.ggml.sha256', library_hash)
        writer.add_string('quantization.policy', tier + ' eligible transformer linear matrices except preserved layers; preserved matrices reproduce original runtime bfloat16; other tensors at source precision')
        writer.add_string('quantization.preserved_layers', json.dumps(preserve_layers))
        writer.add_uint32('general.quantization_version', gguf.GGML_QUANT_VERSION)
        count = quantized = 0
        with safe_open(str(source), framework='np') as tensors:
            for name in sorted(tensors.keys()):
                value = np.asarray(tensors.get_tensor(name))
                if value.dtype.kind != 'f' or not np.isfinite(value).all():
                    raise ValueError('unsupported or non-finite source tensor: ' + name)
                if LINEAR.fullmatch(name):
                    if value.ndim != 2 or value.shape[-1] % block_size:
                        raise ValueError('unexpected linear shape: ' + name)
                    if int(name.split('.')[1]) in preserve_layers:
                        writer.add_tensor(name, runtime_bfloat16_storage(value), raw_dtype=gguf.GGMLQuantizationType.BF16)
                    else:
                        value = (gguf.quantize(value.astype(np.float32, copy=False), kind)
                                 if quantizer is None else quantizer.quantize(value, kind))
                        writer.add_tensor(name, value, raw_dtype=kind)
                        quantized += 1
                else:
                    writer.add_tensor(name, value)
                count += 1
        if count != 615 or quantized != 224 - 7 * len(preserve_layers):
            raise ValueError(f'unexpected source architecture: {count} tensors, {quantized} linear matrices')
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        writer = None
        if (digest(source) != original_hash or digest(Path(__file__)) != exporter_hash
                or (library_hash is not None and digest(Path(ggml_lib)) != library_hash)):
            raise RuntimeError('source weights or conversion tools changed during conversion')
        reader = gguf.GGUFReader(partial)
        if len(reader.tensors) != count or len({x.name for x in reader.tensors}) != count:
            raise RuntimeError('output tensor count or names are invalid')
        del reader
        os.link(partial, output)
        partial.unlink()
        return digest(output)
    finally:
        if writer is not None:
            writer.close()
        tempfile.tempdir = old_tempdir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='original gated model.safetensors')
    parser.add_argument('--out', type=Path, required=True, help='new local output file')
    parser.add_argument('--revision', required=True, help='exact source revision')
    parser.add_argument('--tier', choices=('Q8_0', 'Q6_K', 'Q4_K', 'Q2_K'), default='Q8_0')
    parser.add_argument('--ggml-lib', type=Path, help='local GGML shared library required for K-block tiers')
    parser.add_argument('--preserve-layers', type=parse_layers, default=(),
                        help='original runtime-precision layer numbers or ranges, for example 0-7,24-31')
    args = parser.parse_args()
    print('sha256', export(args.source, args.out, args.revision, args.tier, args.ggml_lib, args.preserve_layers))


if __name__ == '__main__':
    main()

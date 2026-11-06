"""Encode Gaussian-decoder matrices using recorded input covariances.

The archive contains standard GGML K-block codes and source identifiers. Its
identifiers bind bytes, but do not prove that the calibration inputs represent
future images or independently establish how the recorded covariances arose.
"""
import argparse
import json
import math
import os
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open

import export_full_gguf
from export_full_gguf import GGMLQuantizer, digest


def encode_matrix(value, covariance, quantizer, kind, damping, device, importance=False):
    """Apply blockwise error compensation and return standard GGML codes."""
    value = np.ascontiguousarray(value, dtype=np.float32)
    if (value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0 or value.shape[1] % 256
            or not np.isfinite(value).all() or not isinstance(covariance, torch.Tensor)
            or covariance.shape != (value.shape[1], value.shape[1]) or not torch.isfinite(covariance).all()):
        raise ValueError('invalid source matrix or covariance')
    covariance = covariance.to(device=device, dtype=torch.float32).clone()
    diagonal = torch.diagonal(covariance).detach().cpu().numpy().copy() if importance else None
    if importance and (not np.isfinite(diagonal).all() or np.any(diagonal < 0)):
        raise ValueError('invalid covariance diagonal')
    adjustment = damping * torch.diagonal(covariance).mean()
    if not bool(torch.isfinite(adjustment)) or adjustment <= 0:
        raise ValueError('invalid covariance damping')
    covariance.diagonal().add_(adjustment)
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(covariance))
    work = torch.from_numpy(value.copy()).to(device)
    blocks = []
    for start in range(0, value.shape[1], 256):
        end = start + 256
        chunk = np.ascontiguousarray(work[:, start:end].cpu().numpy(), dtype=np.float32)
        weights = None
        if importance:
            weights = diagonal[start:end].astype(np.float32)
            mean = weights.mean()
            weights = (np.ones_like(weights) if mean <= 0 else np.maximum(weights / mean, 1e-6))
            weights = np.ascontiguousarray(weights, dtype=np.float32)
        code = quantizer.quantize(chunk, kind, weights)
        restored = gguf.dequantize(code, kind).reshape(chunk.shape)
        error = work[:, start:end] - torch.from_numpy(np.array(restored, copy=True)).to(device)
        blocks.append(code)
        if end < value.shape[1]:
            gain = torch.linalg.solve(inverse[start:end, start:end], inverse[start:end, end:])
            work[:, end:] -= error @ gain
            inverse[end:, end:] -= inverse[end:, start:end] @ gain
        if not torch.isfinite(work[:, end:]).all():
            raise ValueError('non-finite compensated matrix')
    return np.ascontiguousarray(np.concatenate(blocks, axis=1))


def create(source, moments, moments_report, library, output, quant_type='Q2_K', damping=0.3, device='cpu',
           checkpoint=None, importance=False):
    source, moments, moments_report, library, output = map(Path, (source, moments, moments_report, library, output))
    if quant_type not in ('Q2_K', 'Q4_K', 'Q6_K') or not math.isfinite(damping) or damping <= 0 or type(importance) is not bool:
        raise ValueError('invalid quantization type or damping')
    if output.exists() or output.with_name(output.name + '.partial').exists():
        raise FileExistsError(output)
    inputs = {'source_sha256': digest(source), 'moments_sha256': digest(moments),
              'moments_report_sha256': digest(moments_report), 'ggml_sha256': digest(library),
              'tool_sha256': digest(Path(__file__)), 'quantizer_sha256': digest(Path(export_full_gguf.__file__))}
    recorded = json.loads(moments_report.read_text(encoding='utf-8'))
    if (recorded.get('status') != 'complete' or recorded.get('hashes_unchanged_after_run') is not True
            or recorded.get('hessians_sha256') != inputs['moments_sha256']
            or recorded.get('inputs', {}).get('gaussian_weights') != inputs['source_sha256']):
        raise ValueError('covariance report does not bind the source model and moments')
    quantizer = GGMLQuantizer(library)
    kind = getattr(gguf.GGMLQuantizationType, quant_type)
    covariances = torch.load(moments, map_location='cpu', weights_only=True, mmap=True)
    if not isinstance(covariances, dict):
        raise ValueError('invalid covariance archive')
    encoded = {}
    with torch.no_grad(), safe_open(str(source), framework='np') as weights:
        names = sorted(name for name in weights.keys() if name.endswith('weight')
                       and len(weights.get_slice(name).get_shape()) == 2
                       and all(dimension > 0 for dimension in weights.get_slice(name).get_shape())
                       and weights.get_slice(name).get_shape()[-1] % 256 == 0)
        if not names:
            raise ValueError('no eligible Gaussian matrices')
        for name in names:
            if checkpoint is not None:
                checkpoint()
            try:
                encoded[name] = encode_matrix(weights.get_tensor(name), covariances.get('gs.' + name),
                                              quantizer, kind, damping, device, importance)
            except ValueError as exc:
                raise ValueError(f'{exc}: {name}') from exc
            print('encoded', name, flush=True)
    final = {'source_sha256': digest(source), 'moments_sha256': digest(moments),
             'moments_report_sha256': digest(moments_report), 'ggml_sha256': digest(library),
             'tool_sha256': digest(Path(__file__)), 'quantizer_sha256': digest(Path(export_full_gguf.__file__))}
    if final != inputs:
        raise RuntimeError('calibration inputs changed during encoding')
    metadata = {'version': 1, 'quant_type': quant_type, 'damping': damping,
                'importance': 'covariance_diagonal' if importance else 'none', **inputs}
    encoded['__metadata__'] = np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8)
    partial = output.with_name(output.name + '.partial')
    with partial.open('xb') as stream:
        np.savez(stream, **encoded)
    os.link(partial, output)
    partial.unlink()
    return {'encoded_sha256': digest(output), 'eligible_matrices': len(names), **metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='original Gaussian decoder safetensors')
    parser.add_argument('--moments', type=Path, required=True, help='recorded input covariance archive')
    parser.add_argument('--moments-report', type=Path, required=True)
    parser.add_argument('--ggml-lib', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--type', choices=('Q2_K', 'Q4_K', 'Q6_K'), default='Q2_K')
    parser.add_argument('--damping', type=float, default=0.3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--importance', action='store_true', help='weight each quantization block by its covariance diagonal')
    args = parser.parse_args()
    print(json.dumps(create(args.source, args.moments, args.moments_report, args.ggml_lib,
                            args.out, args.type, args.damping, args.device, importance=args.importance), indent=2))


if __name__ == '__main__':
    main()

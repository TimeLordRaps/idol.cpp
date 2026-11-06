"""Encode denoiser block matrices using explicitly recorded input covariances.

The covariance report must bind the source weights and moments. Its digests
identify those inputs but do not establish their origin or representativeness.
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

import calibrate_gaussian_covariance
import export_full_gguf
from calibrate_gaussian_covariance import encode_matrix
from export_full_gguf import GGMLQuantizer, digest


def create(source, moments, moments_report, library, output, quant_type='Q8_0', damping=0.3, device='cpu',
           checkpoint=None, importance=False):
    source, moments, moments_report, library, output = map(Path, (source, moments, moments_report, library, output))
    if (quant_type not in ('Q2_K', 'Q4_K', 'Q6_K', 'Q8_0') or not math.isfinite(damping)
            or damping <= 0 or type(importance) is not bool):
        raise ValueError('invalid denoiser quantization type or damping')
    if output.exists() or output.with_name(output.name + '.partial').exists():
        raise FileExistsError(output)
    inputs = {'source_sha256': digest(source), 'moments_sha256': digest(moments),
              'moments_report_sha256': digest(moments_report), 'ggml_sha256': digest(library),
              'tool_sha256': digest(Path(__file__)), 'helper_sha256': digest(Path(calibrate_gaussian_covariance.__file__)),
              'quantizer_sha256': digest(Path(export_full_gguf.__file__))}
    recorded = json.loads(moments_report.read_text(encoding='utf-8'))
    if recorded.get('status') == 'complete' and recorded.get('hashes_unchanged_after_run') is True:
        report_inputs = recorded.get('inputs', {})
        source_digest = report_inputs.get('denoiser_source')
        reference_model = report_inputs.get('model')
        complete = True
    else:
        report_inputs = recorded.get('inputs_sha256', {})
        source_digest = report_inputs.get('source_sha256')
        reference_model = report_inputs.get('model_sha256')
        complete = {'plain', 'damp_0.3'} <= set(recorded.get('variants', {}))
    if (recorded.get('hessians_sha256') != inputs['moments_sha256']
            or source_digest != inputs['source_sha256'] or not complete):
        raise ValueError('denoiser covariance report does not bind completed source and moments')
    if (not isinstance(reference_model, str) or len(reference_model) != 64
            or any(c not in '0123456789abcdef' for c in reference_model)):
        raise ValueError('invalid covariance reference model digest')
    covariances = torch.load(moments, map_location='cpu', weights_only=True, mmap=True)
    if not isinstance(covariances, dict):
        raise ValueError('invalid denoiser covariance archive')
    quantizer = GGMLQuantizer(library)
    kind = getattr(gguf.GGMLQuantizationType, quant_type)
    encoded = {}
    with torch.no_grad(), safe_open(str(source), framework='np') as weights:
        names = sorted(name for name in weights.keys() if name.startswith('blocks.') and name.endswith('weight')
                       and len(weights.get_slice(name).get_shape()) == 2
                       and all(dimension > 0 for dimension in weights.get_slice(name).get_shape())
                       and weights.get_slice(name).get_shape()[-1] % 256 == 0)
        if not names or set(covariances) != set(names):
            raise ValueError('denoiser covariance tensor names do not match eligible matrices')
        for name in names:
            if checkpoint is not None:
                checkpoint()
            try:
                encoded[name] = encode_matrix(weights.get_tensor(name), covariances[name], quantizer, kind, damping, device, importance)
            except ValueError as exc:
                raise ValueError(f'{exc}: {name}') from exc
            print('encoded', name, flush=True)
    final = {'source_sha256': digest(source), 'moments_sha256': digest(moments),
             'moments_report_sha256': digest(moments_report), 'ggml_sha256': digest(library),
             'tool_sha256': digest(Path(__file__)), 'helper_sha256': digest(Path(calibrate_gaussian_covariance.__file__)),
             'quantizer_sha256': digest(Path(export_full_gguf.__file__))}
    if final != inputs:
        raise RuntimeError('denoiser calibration inputs changed during encoding')
    metadata = {'version': 1, 'component': 'denoiser', 'quant_type': quant_type, 'damping': damping,
                'importance': 'covariance_diagonal' if importance else 'none',
                'reference_model_sha256': reference_model, **inputs}
    encoded['__metadata__'] = np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8)
    partial = output.with_name(output.name + '.partial')
    with partial.open('xb') as stream:
        np.savez(stream, **encoded)
    os.link(partial, output)
    partial.unlink()
    return {'encoded_sha256': digest(output), 'eligible_matrices': len(names), **metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='original denoiser safetensors')
    parser.add_argument('--moments', type=Path, required=True)
    parser.add_argument('--moments-report', type=Path, required=True)
    parser.add_argument('--ggml-lib', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--type', choices=('Q2_K', 'Q4_K', 'Q6_K', 'Q8_0'), default='Q8_0')
    parser.add_argument('--damping', type=float, default=0.3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--importance', action='store_true', help='weight each quantization block by its covariance diagonal')
    args = parser.parse_args()
    print(json.dumps(create(args.source, args.moments, args.moments_report, args.ggml_lib,
                            args.out, args.type, args.damping, args.device, importance=args.importance), indent=2))


if __name__ == '__main__':
    main()

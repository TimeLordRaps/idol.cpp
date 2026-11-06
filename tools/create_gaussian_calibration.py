"""Build an experimental Gaussian quantization plan from recorded input moments.

The reports must bind the moment file and source Gaussian weights by digest.
Those bindings do not independently prove how choices were produced or whether
calibration inputs are representative. Only the input-channel diagonal is kept.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from export_full_gguf import digest


def group_for(name):
    parts = name.split('.')
    return '.'.join(parts[:2]) if parts[0] == 'blocks' else name


def create(source, moments_path, moments_report_path, selection_path, quant_type, output):
    source, moments_path = Path(source), Path(moments_path)
    moments_report_path, selection_path, output = map(Path, (moments_report_path, selection_path, output))
    if quant_type not in ('Q2_K', 'Q4_K', 'Q6_K'):
        raise ValueError('unsupported Gaussian quantization type')
    inputs = {'source': digest(source), 'moments': digest(moments_path),
              'moments_report': digest(moments_report_path), 'selection': digest(selection_path)}
    report = json.loads(moments_report_path.read_text(encoding='utf-8'))
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if (report.get('status') != 'complete' or report.get('hashes_unchanged_after_run') is not True
            or report.get('hessians_sha256') != inputs['moments']
            or report.get('inputs', {}).get('gaussian_weights') != inputs['source']):
        raise ValueError('calibration report does not bind the moments and Gaussian source')
    selection_inputs = selection.get('inputs_sha256', {})
    if (quant_type not in selection.get('scope', '') or selection.get('status') != 'complete'
            or selection.get('hashes_unchanged_after_run') is not True
            or selection_inputs.get('gaussian') != inputs['source']
            or selection_inputs.get('moments') != inputs['moments']
            or selection_inputs.get('moments_report') != inputs['moments_report']):
        raise ValueError('selection report does not bind the requested type, source, and moments')
    choices = selection.get('selection', {}).get('choices')
    if not isinstance(choices, dict) or any(not isinstance(v, dict) or type(v.get('weighted')) is not bool
                                            for v in choices.values()):
        raise ValueError('invalid Gaussian matrix selection')
    groups = {name: value['weighted'] for name, value in choices.items()}
    moments = torch.load(moments_path, map_location='cpu', weights_only=True)
    vectors = {}
    expected_groups = set()
    eligible = []
    with safe_open(str(source), framework='np') as weights:
        for name in sorted(weights.keys()):
            shape = tuple(weights.get_slice(name).get_shape())
            if not name.endswith('weight') or len(shape) != 2 or shape[-1] % 256:
                continue
            group = group_for(name)
            expected_groups.add(group)
            eligible.append(name)
            if group not in groups:
                raise ValueError('selection is missing group: ' + group)
            if groups[group]:
                matrix = moments.get('gs.' + name)
                if matrix is None or tuple(matrix.shape) != (shape[-1], shape[-1]):
                    raise ValueError('calibration moments are missing or have wrong shape: ' + name)
                diagonal = matrix.diag().numpy().astype(np.float32)
                if not np.isfinite(diagonal).all() or np.any(diagonal < 0) or diagonal.mean() <= 0:
                    raise ValueError('invalid calibration moments: ' + name)
                vectors[name] = np.ascontiguousarray(np.maximum(diagonal / diagonal.mean(), 1e-6), dtype=np.float32)
    if not eligible or set(groups) != expected_groups:
        raise ValueError('selection groups do not match eligible Gaussian matrices')
    metadata = {'version': 1, 'quant_type': quant_type, 'gaussian_source_sha256': inputs['source'],
                'moments_sha256': inputs['moments'], 'moments_report_sha256': inputs['moments_report'],
                'selection_sha256': inputs['selection'], 'groups': groups, 'eligible': eligible,
                'selection_provenance': 'The selection report binds source and moments; its derivation is not independently verified by this plan.'}
    payload = dict(vectors)
    payload['__metadata__'] = np.frombuffer(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode('utf-8'), dtype=np.uint8)
    partial = output.with_name(output.name + '.partial')
    if output.exists() or partial.exists():
        raise FileExistsError(output)
    claimed = False
    try:
        with partial.open('xb') as stream:
            claimed = True
            np.savez_compressed(stream, **payload)
        if inputs != {'source': digest(source), 'moments': digest(moments_path),
                      'moments_report': digest(moments_report_path), 'selection': digest(selection_path)}:
            raise RuntimeError('calibration inputs changed during plan creation')
        os.link(partial, output)
    finally:
        if claimed and partial.exists():
            partial.unlink()
    return {'plan_sha256': digest(output), 'eligible_matrices': len(eligible),
            'weighted_matrices': len(vectors), 'metadata': metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='original decoder_gs.safetensors')
    parser.add_argument('--moments', type=Path, required=True)
    parser.add_argument('--moments-report', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--type', choices=('Q2_K', 'Q4_K', 'Q6_K'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = create(args.source, args.moments, args.moments_report, args.selection, args.type, args.out)
    print(json.dumps({key: result[key] for key in ('plan_sha256', 'eligible_matrices', 'weighted_matrices')}, indent=2))


if __name__ == '__main__':
    main()

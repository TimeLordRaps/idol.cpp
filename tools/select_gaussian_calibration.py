"""Select experimental Gaussian K-block encodings using separate latent cases.

The first latent files are held out; only the following calibration latents
influence the greedy block selection. The inputs' digests are recorded, but
their origin and representativeness must be established separately.
"""
import argparse
import json
import math
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open

from eval_construction import available_bytes
from export_full_gguf import GGMLQuantizer, digest, _gaussian_group
from quant_idol import latent_files, load_models, read_latent, sample_anchors


def run(args):
    if (args.holdout < 1 or args.calib < 1 or args.anchors < 1 or args.threads < 1 or args.seed < 0
            or args.seed + args.holdout + args.calib - 1 >= 2**64
            or not math.isfinite(args.min_available_gb) or args.min_available_gb < 0):
        raise ValueError('invalid case, thread, or memory settings')
    paths = latent_files(args.latents)
    if len(paths) < args.holdout + args.calib:
        raise ValueError('not enough byte-disjoint latent cases')
    chosen = paths[:args.holdout + args.calib]
    source = args.weights / 'decoder_gs.safetensors'
    inputs = {'config': args.weights / 'config.json', 'octree': args.weights / 'decoder.safetensors',
              'gaussian': source, 'moments': args.moments, 'moments_report': args.moments_report,
              'ggml_lib': args.ggml_lib, 'tool': Path(__file__),
              'quantizer': Path(__file__).with_name('export_full_gguf.py'),
              'model_adapter': Path(__file__).with_name('quant_idol.py')}
    inputs.update({f'case_{i}': path for i, path in enumerate(chosen)})
    inputs.update({'sdk/' + p.relative_to(args.sdk).as_posix(): p for p in sorted((args.sdk / 'idol').rglob('*.py'))})
    if not any(key.startswith('sdk/') for key in inputs):
        raise ValueError('missing SDK source files')
    hashes = {name: digest(path) for name, path in inputs.items()}
    case_hashes = [hashes[f'case_{i}'] for i in range(len(chosen))]
    if len(set(case_hashes)) != len(case_hashes):
        raise ValueError('calibration and held-out latents must be byte-disjoint')
    report = json.loads(args.moments_report.read_text(encoding='utf-8'))
    expected_training = [report.get('inputs', {}).get(f'calibration_{i}') for i in range(args.calib)]
    if (report.get('status') != 'complete' or report.get('hashes_unchanged_after_run') is not True
            or report.get('hessians_sha256') != hashes['moments']
            or report.get('inputs', {}).get('gaussian_weights') != hashes['gaussian']
            or report.get('calibration_count') != args.calib
            or expected_training != case_hashes[args.holdout:]):
        raise ValueError('moments report does not bind the selected source and calibration latents')
    if args.out.exists():
        raise FileExistsError(args.out)
    if available_bytes() < args.min_available_gb * 1_000_000_000:
        raise RuntimeError('insufficient available host memory')
    torch.set_num_threads(args.threads)
    moments = torch.load(args.moments, map_location='cpu', weights_only=True)
    quantizer = GGMLQuantizer(args.ggml_lib)
    kind = getattr(gguf.GGMLQuantizationType, args.type)
    encoded = {'baseline': {}, 'weighted': {}}
    with safe_open(str(source), framework='np') as weights:
        for name in sorted(weights.keys()):
            shape = tuple(weights.get_slice(name).get_shape())
            if not name.endswith('weight') or len(shape) != 2 or shape[-1] % 256:
                continue
            if available_bytes() < args.min_available_gb * 1_000_000_000:
                raise RuntimeError('available host memory fell below the requested guard')
            matrix = moments.get('gs.' + name)
            if matrix is None or tuple(matrix.shape) != (shape[-1], shape[-1]):
                raise ValueError('missing input moments: ' + name)
            diagonal = matrix.diag().numpy().astype(np.float32)
            if not np.isfinite(diagonal).all() or np.any(diagonal < 0) or diagonal.mean() <= 0:
                raise ValueError('invalid input moments: ' + name)
            importance = np.ascontiguousarray(np.maximum(diagonal / diagonal.mean(), 1e-6), dtype=np.float32)
            values = np.ascontiguousarray(weights.get_tensor(name), dtype=np.float32)
            encoded['baseline'][name] = (quantizer.quantize(values, kind), shape)
            encoded['weighted'][name] = (quantizer.quantize(values, kind, importance), shape)
    del moments
    if not encoded['baseline']:
        raise ValueError('no eligible Gaussian matrices')
    octree, gaussian = load_models(args.weights, args.sdk)
    references = []
    with torch.no_grad():
        for index, path in enumerate(chosen):
            if available_bytes() < args.min_available_gb * 1_000_000_000:
                raise RuntimeError('available host memory fell below the requested guard')
            cond = read_latent(path)
            anchors = sample_anchors(octree, cond, args.anchors, args.seed + index)
            features = gaussian(anchors, cond)
            rep = gaussian.to_representation(anchors, features)[0]
            references.append((cond, anchors, features.clone(), rep.xyz.clone(), rep.features_dc.clone(), rep.opacity.clone()))
            print(f'reference case {index + 1}/{len(chosen)}', flush=True)
        del octree
        params = dict(gaussian.named_parameters())

        def apply(method, names):
            for name in names:
                packed, shape = encoded[method][name]
                values = gguf.dequantize(packed, kind).reshape(shape)
                params[name].copy_(torch.from_numpy(np.array(values, copy=True)))

        def errors(index):
            cond, anchors, reference, xyz, color, opacity = references[index]
            actual = gaussian(anchors, cond)
            rep = gaussian.to_representation(anchors, actual)[0]
            norm = torch.linalg.vector_norm(reference).item()
            if norm <= 0:
                raise ValueError('zero reference feature norm')
            return {'relative_feature_l2': torch.linalg.vector_norm(actual - reference).item() / norm,
                    'xyz_rmse': torch.mean((rep.xyz - xyz).square()).sqrt().item(),
                    'color_feature_mae': torch.mean(torch.abs(rep.features_dc - color)).item(),
                    'opacity_mae': torch.mean(torch.abs(rep.opacity - opacity)).item()}

        names = sorted(encoded['baseline'])
        apply('baseline', names)
        baseline = [errors(i) for i in range(len(chosen))]
        training = range(args.holdout, len(chosen))
        current = sum(baseline[i]['relative_feature_l2'] for i in training) / args.calib
        initial = current
        groups = {}
        for name in names:
            groups.setdefault(_gaussian_group(name), []).append(name)
        choices = {}
        for group, group_names in sorted(groups.items()):
            apply('weighted', group_names)
            candidate = sum(errors(i)['relative_feature_l2'] for i in training) / args.calib
            accepted = candidate < current
            if accepted:
                current = candidate
            else:
                apply('baseline', group_names)
            choices[group] = {'weighted': accepted, 'calibration_score': current, 'candidate_score': candidate}
            print('selected', group, 'weighted' if accepted else 'baseline', flush=True)
        selected = [errors(i) for i in range(len(chosen))]
    if hashes != {name: digest(path) for name, path in inputs.items()}:
        raise RuntimeError('selection inputs changed during the run')
    result = {'status': 'complete', 'scope': f'CPU Gaussian decoder {args.type} greedy group selection; '
              'held-out latents are excluded from selection. Latent origin and representativeness are not established.',
              'inputs_sha256': hashes, 'holdout_count': args.holdout, 'calibration_count': args.calib,
              'anchors': args.anchors, 'seed': args.seed, 'threads': args.threads, 'torch_version': str(torch.__version__),
              'matrix_count': len(names), 'hashes_unchanged_after_run': True,
              'selection': {'baseline_calibration_mean_relative_error': initial,
                            'final_calibration_mean_relative_error': current, 'choices': choices},
              'cases': [{'role': 'holdout' if i < args.holdout else 'calibration',
                         'latent_sha256': case_hashes[i], 'baseline': baseline[i], 'selected': selected[i]}
                        for i in range(len(chosen))]}
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('weights', 'sdk', 'latents', 'moments', 'moments-report', 'ggml-lib', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--type', choices=('Q2_K', 'Q4_K', 'Q6_K'), required=True)
    parser.add_argument('--holdout', type=int, default=1)
    parser.add_argument('--calib', type=int, default=2)
    parser.add_argument('--anchors', type=int, default=256)
    parser.add_argument('--seed', type=int, default=5000)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--min-available-gb', type=float, default=8.0)
    args = parser.parse_args()
    result = run(args)
    print('wrote', args.out, 'with', result['matrix_count'], 'quantized matrices', flush=True)


if __name__ == '__main__':
    main()

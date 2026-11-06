"""Select Q4 Gaussian groups over a Q2 baseline using separate latent cases.

Selection uses only training latents. Evaluation latents are reported but do
not influence group choices. Their origin and representativeness are external.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors.torch import load_file

from eval_construction import available_bytes
from export_full_gguf import _gaussian_group, digest, load_gaussian_codes


def checkpoint():
    """Optional shared-device guard hook."""


def hold_model(module, name):
    """Optional shared-device guard hook."""
    return module


def read_case(path):
    path = Path(path)
    value = np.load(path, allow_pickle=False) if path.suffix.lower() == '.npy' else np.fromfile(path, dtype='<f4')
    if value.dtype != np.float32 or value.size == 0 or value.size % 16 or not np.isfinite(value).all():
        raise ValueError('invalid latent values: ' + path.name)
    value = np.ascontiguousarray(value.reshape(1, -1, 16), dtype='<f4')
    return value, hashlib.sha256(value.tobytes()).hexdigest()


def run(args):
    if (not args.train_latent or not args.eval_latent or args.anchors < 1 or args.threads < 1
            or args.max_groups < 1 or args.seed < 0 or args.seed >= 2**64
            or not math.isfinite(args.min_available_gb) or args.min_available_gb < 0):
        raise ValueError('invalid latent split, group limit, or resource settings')
    if args.out.exists():
        raise FileExistsError(args.out)
    cases = [('training', Path(path)) for path in args.train_latent] + [('evaluation', Path(path)) for path in args.eval_latent]
    inputs = {'config': args.weights / 'config.json', 'octree': args.weights / 'decoder.safetensors',
              'gaussian': args.weights / 'decoder_gs.safetensors', 'q2_codes': args.q2_codes, 'q4_codes': args.q4_codes,
              'tool': Path(__file__), 'exporter': Path(__file__).with_name('export_full_gguf.py')}
    sdk_files = sorted((args.sdk / 'idol').rglob('*.py'))
    if not sdk_files:
        raise ValueError('missing software development kit source files')
    inputs.update({'sdk/' + path.relative_to(args.sdk).as_posix(): path for path in sdk_files})
    inputs.update({f'case_{i}': path for i, (_, path) in enumerate(cases)})
    before = {name: digest(path) for name, path in inputs.items()}
    data = [read_case(path) for _, path in cases]
    if len({identity for _, identity in data}) != len(data):
        raise ValueError('training and evaluation latents must be byte-disjoint after decoding')
    if available_bytes() < args.min_available_gb * 1_000_000_000:
        raise RuntimeError('insufficient available host memory')
    metadata2, codes2 = load_gaussian_codes(args.q2_codes, inputs['gaussian'], before['gaussian'], 'Q2_K')
    metadata4, codes4 = load_gaussian_codes(args.q4_codes, inputs['gaussian'], before['gaussian'], 'Q4_K')
    if set(codes2) != set(codes4):
        raise ValueError('encoded Gaussian tensor names differ')
    if str(args.sdk) not in sys.path:
        sys.path.insert(0, str(args.sdk))
    from idol.models import OctreeProbabilityFixedlenDecoder, GaussianFixedlenDecoder
    config = json.loads(inputs['config'].read_text(encoding='utf-8'))
    level = config.get('max_voxel_level', 8)
    if not isinstance(level, int) or level < 1:
        raise ValueError('invalid octree level')
    torch.set_num_threads(args.threads)
    octree = OctreeProbabilityFixedlenDecoder(**config['decoder']['args'])
    octree.load_state_dict(load_file(str(inputs['octree'])), strict=True)
    octree = hold_model(octree.eval().requires_grad_(False).to(args.device), 'octree')
    gaussian = GaussianFixedlenDecoder(**config['decoder_gs']['args'])
    gaussian.load_state_dict(load_file(str(inputs['gaussian'])), strict=True)
    gaussian = hold_model(gaussian.eval().requires_grad_(False).to(args.device), 'Gaussian decoder')
    parameters = dict(gaussian.named_parameters())
    names = sorted(codes2)
    if any(name not in parameters for name in names):
        raise ValueError('encoded Gaussian weights do not match runtime parameters')
    decoded = {}
    for bits, codes in ((2, codes2), (4, codes4)):
        kind = getattr(gguf.GGMLQuantizationType, f'Q{bits}_K')
        decoded[bits] = {name: gguf.dequantize(codes[name], kind).reshape(parameters[name].shape).copy() for name in names}
    groups = sorted({_gaussian_group(name) for name in names})
    with torch.no_grad():
        references = []
        for value, _ in data:
            checkpoint()
            cond = torch.from_numpy(value).to(args.device)
            torch.manual_seed(args.seed)
            anchors = OctreeProbabilityFixedlenDecoder.sample(octree, cond, num_points=args.anchors, level=level)
            references.append((cond, anchors, gaussian(anchors, cond).clone()))
        def apply(group, bits):
            for name in names:
                if _gaussian_group(name) == group:
                    parameters[name].copy_(torch.from_numpy(decoded[bits][name]))
        def score(indices):
            result = []
            for index in indices:
                checkpoint()
                cond, anchors, reference = references[index]
                denominator = torch.linalg.vector_norm(reference)
                if denominator <= 0 or not bool(torch.isfinite(denominator)):
                    raise ValueError('invalid reference feature norm')
                error = torch.linalg.vector_norm(gaussian(anchors, cond) - reference) / denominator
                if not bool(torch.isfinite(error)):
                    raise ValueError('non-finite feature error')
                result.append(float(error))
            return result
        for group in groups:
            apply(group, 2)
        train = range(len(args.train_latent))
        evaluation = range(len(args.train_latent), len(cases))
        baseline = {'training': score(train), 'evaluation': score(evaluation)}
        best_mean = sum(baseline['training']) / len(args.train_latent)
        selected = []
        remaining = set(groups)
        for _ in range(min(args.max_groups, len(groups))):
            if available_bytes() < args.min_available_gb * 1_000_000_000:
                raise RuntimeError('available host memory fell below the requested guard')
            best, candidates = None, {}
            for group in sorted(remaining):
                apply(group, 4)
                values = score(train)
                mean = sum(values) / len(values)
                candidates[group] = mean
                apply(group, 2)
                if best is None or mean < best[0]:
                    best = (mean, group)
            if best is None or best[0] >= best_mean:
                break
            best_mean, group = best
            apply(group, 4)
            remaining.remove(group)
            entry = {'group': group, 'training': score(train), 'evaluation': score(evaluation),
                     'candidate_training_means': candidates}
            selected.append(entry)
            print('selected', len(selected), group, flush=True)
    if before != {name: digest(path) for name, path in inputs.items()}:
        raise RuntimeError('mixed-precision selection inputs changed during the run')
    report = {'status': 'complete', 'scope': __doc__, 'inputs_sha256': before, 'hashes_unchanged_after_run': True,
              'q2_moments_sha256': metadata2['moments_sha256'], 'q4_moments_sha256': metadata4['moments_sha256'],
              'cases': [{'role': role, 'name': path.name, 'file_sha256': before[f'case_{i}'],
                         'latent_sha256': data[i][1]} for i, (role, path) in enumerate(cases)],
              'anchors': args.anchors, 'seed': args.seed, 'device': args.device, 'threads': args.threads,
              'torch_version': str(torch.__version__), 'matrix_count': len(names), 'baseline': baseline, 'selected': selected}
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('weights', 'sdk', 'q2-codes', 'q4-codes', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--train-latent', type=Path, action='append', default=[])
    parser.add_argument('--eval-latent', type=Path, action='append', default=[])
    parser.add_argument('--anchors', type=int, default=1024)
    parser.add_argument('--max-groups', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--min-available-gb', type=float, default=8.0)
    args = parser.parse_args()
    report = run(args)
    print('selected', len(report['selected']), 'groups', flush=True)


if __name__ == '__main__':
    main()

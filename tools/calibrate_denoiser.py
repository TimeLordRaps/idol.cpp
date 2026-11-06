"""Record denoiser input covariances from explicitly supplied images.

This uses the original runtime denoiser and separately obtained image encoders.
Recorded hashes bind input bytes; they do not establish image representativeness.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
from safetensors import safe_open

import idol_gguf
from export_full_gguf import digest
from idol_gguf import IdolGGUF


def snapshot_files(root, prefix):
    root = Path(root)
    files = [root / 'config.json', *sorted(root.glob('*.safetensors'))]
    if not files[0].is_file() or len(files) < 2 or not all(path.is_file() for path in files):
        raise ValueError(f'{prefix} requires a local configuration and safetensors weights')
    return {prefix + '/' + path.name: path for path in files}


def verify_original_denoiser(module, source):
    """Compare every loaded runtime value with original source at runtime dtype."""
    current = module.state_dict()
    with safe_open(str(source), framework='pt', device='cpu') as weights:
        if set(current) != set(weights.keys()):
            raise ValueError('denoiser tensor names do not match the original source')
        for name, value in current.items():
            original = weights.get_tensor(name)
            if tuple(original.shape) != tuple(value.shape) or not torch.equal(value.detach().cpu(), original.to(value.dtype)):
                raise ValueError('loaded denoiser differs from the original runtime: ' + name)


def collect(model_path, denoiser_source, sdk, dino, flux, cases, out_dir, *, device='cuda', steps=10,
            seed=42, cfg_strength=3.0, rescale_t=3.0, samples_per_forward=64, threads=4, stage_encoders=False):
    from PIL import Image
    import idol_generate as run
    model_path, denoiser_source, sdk, dino, flux, out_dir = map(Path, (model_path, denoiser_source, sdk, dino, flux, out_dir))
    if (not cases or steps < 1 or threads < 1 or not 1 <= samples_per_forward <= 4096
            or seed < 0 or seed >= 2**64 or rescale_t <= 0
            or not math.isfinite(cfg_strength) or not math.isfinite(rescale_t)):
        raise ValueError('invalid calibration cases or sampling parameters')
    if out_dir.exists():
        raise FileExistsError(out_dir)
    cases = [(mode, Path(path)) for mode, path in cases]
    if any(mode not in ('real', 'synthetic') or not path.is_file() for mode, path in cases):
        raise ValueError('invalid calibration image or mode')
    sdk_files = sorted((sdk / 'idol').rglob('*.py'))
    if not sdk_files:
        raise ValueError('missing software development kit sources')
    paths = {'model': model_path, 'denoiser_source': denoiser_source, 'tool': Path(__file__),
             'inference_tool': Path(run.__file__), 'loader_tool': Path(idol_gguf.__file__)}
    paths.update(snapshot_files(dino, 'dino'))
    paths.update(snapshot_files(flux / 'vae', 'flux_vae'))
    paths.update({'sdk/' + path.relative_to(sdk).as_posix(): path for path in sdk_files})
    paths.update({f'image_{i}': path for i, (_, path) in enumerate(cases)})
    before = {name: digest(path) for name, path in paths.items()}
    image_hashes = [before[f'image_{i}'] for i in range(len(cases))]
    if len(set(image_hashes)) != len(image_hashes):
        raise ValueError('calibration images must be byte-distinct')
    if str(sdk) not in sys.path:
        sys.path.insert(0, str(sdk))
    torch.set_num_threads(threads)
    model = IdolGGUF(model_path)
    out_dir.mkdir(parents=True, exist_ok=False)
    pipe = run.make_pipeline(model, device, str(dino), str(flux))
    verify_original_denoiser(pipe.denoiser, denoiser_source)
    names = sorted(name for name, parameter in pipe.denoiser.named_parameters()
                   if name.startswith('blocks.') and name.endswith('weight') and parameter.ndim == 2
                   and parameter.shape[-1] > 0 and parameter.shape[-1] % 256 == 0)
    if not names:
        raise ValueError('no eligible denoiser matrices')
    moments, counts, hooks = {}, {}, []
    for name in names:
        module = pipe.denoiser.get_submodule(name[:-7])
        if not isinstance(module, torch.nn.Linear):
            raise ValueError('eligible denoiser weight is not linear: ' + name)
        def capture(_module, inputs, name=name):
            x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            indices = torch.linspace(0, x.shape[0]-1, min(samples_per_forward, x.shape[0]), device=x.device).long()
            x = x[indices]
            gram = x.T @ x
            moments[name] = moments.get(name, 0) + gram
            counts[name] = counts.get(name, 0) + len(x)
        hooks.append(module.register_forward_pre_hook(capture))
    try:
        with torch.no_grad():
            for index, (mode, path) in enumerate(cases):
                run.checkpoint()
                run.image_to_latent(pipe, Image.open(path), mode, seed, steps, cfg_strength, rescale_t, stage_encoders)
                print(f'calibration image {index+1}/{len(cases)}', flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    if set(moments) != set(names) or any(counts[name] < 1 for name in names):
        raise RuntimeError('missing denoiser covariance samples')
    matrices = {name: (moments[name] / counts[name]).cpu() for name in names}
    if any(not torch.isfinite(matrix).all() for matrix in matrices.values()):
        raise RuntimeError('non-finite denoiser covariance')
    after = {name: digest(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError('calibration inputs changed during collection')
    moments_path = out_dir / 'hessians.pt'
    with moments_path.open('xb') as stream:
        torch.save(matrices, stream)
    report = {'status': 'complete', 'scope': 'Original-runtime denoiser input covariances sampled from explicitly supplied images; no held-out accuracy claim.',
              'inputs': before, 'cases': [{'mode': mode, 'name': path.name, 'sha256': image_hashes[i]} for i, (mode, path) in enumerate(cases)],
              'hessians_sha256': digest(moments_path), 'hashes_unchanged_after_run': True,
              'counts': counts, 'steps': steps, 'seed': seed, 'samples_per_forward': samples_per_forward,
              'cfg_strength': cfg_strength, 'rescale_t': rescale_t, 'device': device, 'threads': threads,
              'stage_encoders': stage_encoders, 'torch_version': str(torch.__version__)}
    with (out_dir / 'report.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='complete Idol GGUF with original-runtime denoiser')
    parser.add_argument('--denoiser-source', type=Path, required=True, help='original denoiser safetensors')
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--dino', type=Path, required=True, help='local original DINOv3 snapshot')
    parser.add_argument('--flux-vae', type=Path, required=True, help='local original FLUX.2 snapshot containing vae/')
    parser.add_argument('--real-image', type=Path, action='append', default=[])
    parser.add_argument('--synthetic-image', type=Path, action='append', default=[])
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cfg-strength', type=float, default=3.0)
    parser.add_argument('--rescale-t', type=float, default=3.0)
    parser.add_argument('--samples-per-forward', type=int, default=64)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--stage-encoders', action='store_true')
    args = parser.parse_args()
    cases = [('real', path) for path in args.real_image] + [('synthetic', path) for path in args.synthetic_image]
    report = collect(args.model, args.denoiser_source, args.sdk, args.dino, args.flux_vae, cases, args.out_dir,
                     device=args.device, steps=args.steps, seed=args.seed, cfg_strength=args.cfg_strength,
                     rescale_t=args.rescale_t, samples_per_forward=args.samples_per_forward, threads=args.threads,
                     stage_encoders=args.stage_encoders)
    print('wrote covariance archive', report['hessians_sha256'], flush=True)


if __name__ == '__main__':
    main()

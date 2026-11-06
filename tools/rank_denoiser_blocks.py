"""Measure single-block denoiser sensitivity on supplied calibration images.

Every other block remains at original runtime precision. Scores compare latent
outputs with the original runtime, not with ground-truth 3D reconstruction.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open

import calibrate_denoiser
import export_full_gguf
import idol_gguf
from calibrate_denoiser import snapshot_files, verify_original_denoiser
from export_full_gguf import GGMLQuantizer, digest, load_denoiser_codes
from idol_gguf import IdolGGUF


def rank(model_path, denoiser_source, sdk, dino, flux, images, library, output, *, quant_type='Q4_K',
         device='cuda', steps=10, seed=42, cfg_strength=3.0, rescale_t=3.0, threads=4, encoded=None):
    from PIL import Image
    import idol_generate as run
    from tensordict import TensorDict

    model_path, denoiser_source, sdk, dino, flux, library, output = map(
        Path, (model_path, denoiser_source, sdk, dino, flux, library, output))
    images = [(mode, Path(path)) for mode, path in images]
    if (not images or any(mode not in ('real', 'synthetic') or not path.is_file() for mode, path in images)
            or quant_type not in ('Q2_K', 'Q4_K', 'Q6_K', 'Q8_0') or steps < 1 or threads < 1
            or seed < 0 or seed >= 2**64 or rescale_t <= 0
            or not math.isfinite(cfg_strength) or not math.isfinite(rescale_t)):
        raise ValueError('invalid ranking images, quantization, or sampling settings')
    if output.exists():
        raise FileExistsError(output)
    sdk_files = sorted((sdk / 'idol').rglob('*.py'))
    if not sdk_files:
        raise ValueError('missing software development kit sources')
    paths = {'model': model_path, 'denoiser_source': denoiser_source, 'ggml_lib': library,
             'tool': Path(__file__), 'inference_tool': Path(run.__file__),
             'calibration_tool': Path(calibrate_denoiser.__file__), 'quantizer_tool': Path(export_full_gguf.__file__),
             'loader_tool': Path(idol_gguf.__file__)}
    if encoded is not None:
        paths['encoded'] = Path(encoded)
    paths.update(snapshot_files(dino, 'dino'))
    paths.update(snapshot_files(flux / 'vae', 'flux_vae'))
    paths.update({'sdk/' + path.relative_to(sdk).as_posix(): path for path in sdk_files})
    paths.update({f'image_{i}': path for i, (_, path) in enumerate(images)})
    before = {name: digest(path) for name, path in paths.items()}
    image_hashes = [before[f'image_{i}'] for i in range(len(images))]
    if len(set(image_hashes)) != len(image_hashes):
        raise ValueError('ranking images must be byte-distinct')
    if str(sdk) not in sys.path:
        sys.path.insert(0, str(sdk))
    from idol.sampler import sample_flow
    torch.set_num_threads(threads)
    model = IdolGGUF(model_path)
    quantizer = GGMLQuantizer(library) if encoded is None else None
    kind = getattr(gguf.GGMLQuantizationType, quant_type)
    codes = None
    if encoded is not None:
        metadata, codes = load_denoiser_codes(encoded, denoiser_source, before['denoiser_source'])
        if metadata['quant_type'] != quant_type or metadata['reference_model_sha256'] != before['model']:
            raise ValueError('encoded denoiser archive does not match ranking inputs')
    pipe = run.make_pipeline(model, device, str(dino), str(flux))
    verify_original_denoiser(pipe.denoiser, denoiser_source)
    cases = []
    with torch.no_grad():
        for mode, path in images:
            run.checkpoint()
            reference, prepared, _ = run.image_to_latent(pipe, Image.open(path), mode, seed, steps, cfg_strength, rescale_t)
            pixels = torch.from_numpy(np.array(prepared)).permute(2, 0, 1).float().div(255)[None].to(pipe.device)
            torch.manual_seed(seed)
            cond = pipe.image_encoder(pixels)
            noise = TensorDict({'latent': torch.randn(1, pipe.denoiser.q_token_length, pipe.denoiser.in_channels, device=pipe.device),
                                'camera': torch.randn(1, 1, pipe.denoiser.cam_channels, device=pipe.device)},
                               batch_size=1, device=pipe.device)
            cases.append((cond, cond.apply(torch.zeros_like), noise, reference))
        def denoise(case):
            run.checkpoint()
            cond, negative, noise, _ = case
            with pipe._sampling_settings():
                sample = sample_flow(pipe.denoiser, noise.clone(), cond=cond, neg_cond=negative, steps=steps,
                                     rescale_t=rescale_t, cfg_strength=cfg_strength, verbose=False, batch_cfg=False)
            return sample['latent'].detach().cpu().float().numpy()
        for case in cases:
            if not np.array_equal(denoise(case), case[3]):
                raise RuntimeError('cached denoising differs from the image pipeline')
        pipe.image_encoder.dinov3.to('cpu')
        pipe.image_encoder.flux2_vae.to('cpu')
        pipe.background_remover.to('cpu')
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()
        parameters = dict(pipe.denoiser.named_parameters())
        eligible = sorted(name for name, value in parameters.items() if name.startswith('blocks.') and name.endswith('weight')
                          and value.ndim == 2 and value.shape[-1] > 0 and value.shape[-1] % 256 == 0)
        if not eligible or any(len(name.split('.')) < 3 or not name.split('.')[1].isdigit() for name in eligible):
            raise ValueError('unexpected denoiser block names')
        if codes is not None and set(codes) != set(eligible):
            raise ValueError('encoded denoiser archive does not cover all eligible blocks')
        groups = sorted({int(name.split('.')[1]) for name in eligible})
        scores = {}
        with safe_open(str(denoiser_source), framework='np') as weights:
            for block in groups:
                names = [name for name in eligible if int(name.split('.')[1]) == block]
                original = {name: parameters[name].detach().clone() for name in names}
                try:
                    for name in names:
                        run.checkpoint()
                        value = np.ascontiguousarray(weights.get_tensor(name), dtype=np.float32)
                        packed = quantizer.quantize(value, kind) if codes is None else codes[name]
                        restored = gguf.dequantize(packed, kind).reshape(value.shape)
                        parameters[name].copy_(torch.from_numpy(np.array(restored, copy=True)))
                    errors = {}
                    for index, case in enumerate(cases):
                        actual, reference = denoise(case).astype(np.float64), case[3].astype(np.float64)
                        norm = np.linalg.norm(reference)
                        if norm <= 0 or not np.isfinite(norm):
                            raise ValueError('invalid reference latent norm')
                        errors[f'case_{index}'] = float(np.linalg.norm(actual-reference)/norm)
                    scores[str(block)] = errors
                    print('ranked block', block, errors, flush=True)
                finally:
                    for name in names:
                        parameters[name].copy_(original[name])
    if before != {name: digest(path) for name, path in paths.items()}:
        raise RuntimeError('denoiser ranking inputs changed during the run')
    report = {'status': 'complete', 'scope': __doc__, 'source_sha256': before['denoiser_source'],
              'model_sha256': before['model'], 'ggml_sha256': before['ggml_lib'], 'inputs_sha256': before,
              'hashes_unchanged_after_run': True, 'quant_type': quant_type,
              'quantization_source': 'provided_encoded_archive' if codes is not None else 'on_demand', 'steps': steps, 'seed': seed,
              'cfg_strength': cfg_strength, 'rescale_t': rescale_t, 'device': device, 'threads': threads,
              'torch_version': str(torch.__version__), 'cases': [{'mode': mode, 'name': path.name, 'sha256': image_hashes[i]}
                                                              for i, (mode, path) in enumerate(images)], 'blocks': scores}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('model', 'denoiser-source', 'sdk', 'dino', 'flux-vae', 'ggml-lib', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--real-image', type=Path, action='append', default=[])
    parser.add_argument('--synthetic-image', type=Path, action='append', default=[])
    parser.add_argument('--type', choices=('Q2_K', 'Q4_K', 'Q6_K', 'Q8_0'), default='Q4_K')
    parser.add_argument('--encoded', type=Path, help='optional source-bound denoiser code archive to rank directly')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cfg-strength', type=float, default=3.0)
    parser.add_argument('--rescale-t', type=float, default=3.0)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    cases = [('real', path) for path in args.real_image] + [('synthetic', path) for path in args.synthetic_image]
    result = rank(args.model, args.denoiser_source, args.sdk, args.dino, args.flux_vae, cases, args.ggml_lib, args.out,
                  quant_type=args.type, device=args.device, steps=args.steps, seed=args.seed,
                  cfg_strength=args.cfg_strength, rescale_t=args.rescale_t, threads=args.threads, encoded=args.encoded)
    print('ranked', len(result['blocks']), 'blocks', flush=True)


if __name__ == '__main__':
    main()

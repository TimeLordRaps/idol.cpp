"""Generate a Gaussian point cloud from an image using Idol GGUF and the native decoder.

The upstream Idol software development kit supplies image preprocessing, image encoders and denoising. The
optional packed denoiser keeps supported linear weights compressed; the experimental fused path multiplies
supported NVIDIA graphics tensors directly and expands other calls. The native decoder uses packed quantized weights.
The two external image encoders are obtained under their own licenses and access terms.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gguf_to_decoder import materialize
from idol_gguf import IdolGGUF, validate_runtime_half_storage


def checkpoint():
    """Optional shared-device guard hook; a host application may replace this callback."""


def hold_model(module, name):
    """Optional shared-device guard hook for moving models off the device when needed."""
    return module


def checkpoint_before_forward(_module, _args):
    # A pre-hook must return None: returning a guard's status would replace
    # the positional arguments supplied to the model.
    checkpoint()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def decoder_cache(model, path):
    path = Path(path)
    model_hash = digest(model.path)
    record = path / "idol-cache.json"
    if path.exists():
        if not record.is_file():
            raise ValueError("decoder cache exists without a completed identity record")
        existing = json.loads(record.read_text(encoding="utf-8"))
        expected = {"version": 1, "model_sha256": model_hash,
                    "manifest_sha256": digest(path / "manifest.txt"), "weights_sha256": digest(path / "weights.bin")}
        if existing != expected:
            raise ValueError("decoder cache does not match this model or its recorded bytes")
        return path
    materialize(model, path)
    identity = {"version": 1, "model_sha256": model_hash,
                "manifest_sha256": digest(path / "manifest.txt"), "weights_sha256": digest(path / "weights.bin")}
    with record.open("x", encoding="utf-8") as stream:
        json.dump(identity, stream, indent=2)
    return path


def encoder_configs_requiring_access(encoders, dinov3, flux_vae, dino_gguf, flux_vae_gguf):
    dino = dinov3 or encoders['dinov3']
    flux = flux_vae or encoders['flux2_vae']
    configs = ((dinov3 if dino_gguf else dino), (flux_vae if flux_vae_gguf else flux))
    return [value for value in configs if value and not os.path.isdir(value)]


def validate_encoder_choices(idol_only, dino_gguf, flux_vae_gguf):
    """Do not silently substitute quantized image encoders in Idol-only mode."""
    if idol_only and (dino_gguf is not None or flux_vae_gguf is not None):
        raise ValueError('--idol-only cannot be combined with quantized image-encoder files')


def make_pipeline(model, device, dinov3, flux_vae, packed_denoiser=False, dino_gguf=None, flux_vae_gguf=None,
                  fused_kernels=False, cudnn_attention=False):
    from idol.conditioning import ImageEncoder, require_encoder_access
    from idol.models import BiRefNet, LatentSeqMMFlowModel
    from idol.pipeline import IdolPipeline

    pipeline = IdolPipeline.__new__(IdolPipeline)
    pipeline.config = model.config
    pipeline.device = torch.device(device)
    if fused_kernels and (pipeline.device.type != 'cuda' or not torch.cuda.is_available()):
        raise ValueError('fused kernels require an available CUDA device')
    encoders = model.config["image_encoders"]
    dino = dinov3 or encoders["dinov3"]
    flux = flux_vae or encoders["flux2_vae"]
    # A supplied encoder file does not require access to the original weights.
    # Explicit configuration overrides may themselves be gated repositories.
    gated = encoder_configs_requiring_access(encoders, dinov3, flux_vae, dino_gguf, flux_vae_gguf)
    if gated:
        require_encoder_access(gated)
    checkpoint()
    if packed_denoiser or fused_kernels:
        from accelerate import init_empty_weights
        with init_empty_weights():
            denoiser = LatentSeqMMFlowModel(**model.config["denoiser"]["args"])
        validate_runtime_half_storage(model, denoiser)
        denoiser = model.load_module_packed('denoiser', denoiser, include_half=True).eval().requires_grad_(False)
    else:
        denoiser = LatentSeqMMFlowModel(**model.config["denoiser"]["args"])
        validate_runtime_half_storage(model, denoiser)
        denoiser = model.load_module("denoiser", denoiser).eval().requires_grad_(False)
    pipeline.denoiser = hold_model(denoiser.to(pipeline.device), "denoiser")
    pipeline.denoiser.register_forward_pre_hook(checkpoint_before_forward)
    checkpoint()
    background = BiRefNet()
    pipeline.background_remover = model.load_module("background", background).eval().requires_grad_(False).to(pipeline.device)
    if pipeline.device.type == "cuda":
        pipeline.background_remover.half()
    pipeline.background_remover = hold_model(pipeline.background_remover, "background remover")
    checkpoint()
    if dino_gguf or flux_vae_gguf:
        from diffusers.models import AutoencoderKLFlux2
        from idol_encoder_gguf import build_dino, build_flux_vae
        pipeline.image_encoder = ImageEncoder.__new__(ImageEncoder)
        pipeline.image_encoder.device = pipeline.device
        if dino_gguf:
            dino_module = build_dino(dinov3, dino_gguf)
        else:
            from transformers import AutoModel
            dino_module = AutoModel.from_pretrained(dino, torch_dtype=torch.bfloat16).eval().requires_grad_(False)
        pipeline.image_encoder.dinov3 = hold_model(dino_module.to(pipeline.device), "image encoder")
        checkpoint()
        vae_module = (build_flux_vae(flux_vae, flux_vae_gguf) if flux_vae_gguf else
                      AutoencoderKLFlux2.from_pretrained(flux, subfolder='vae', torch_dtype=torch.bfloat16)
                      .eval().requires_grad_(False))
        pipeline.image_encoder.flux2_vae = hold_model(vae_module.to(pipeline.device), "image autoencoder")
    else:
        pipeline.image_encoder = ImageEncoder(dino, flux, dinov3_dtype="bf16", device=pipeline.device)
        pipeline.image_encoder.dinov3 = hold_model(pipeline.image_encoder.dinov3, "image encoder")
        pipeline.image_encoder.flux2_vae = hold_model(pipeline.image_encoder.flux2_vae, "image autoencoder")
    if fused_kernels:
        from idol_packed import enable_fused_packed
        count = enable_fused_packed(pipeline.denoiser) + enable_fused_packed(pipeline.image_encoder.dinov3)
        if not count:
            raise ValueError('fused kernels require supported packed denoiser or DINOv3 matrices')
    pipeline.optimize = bool(cudnn_attention and pipeline.device.type == 'cuda')
    return pipeline


@torch.no_grad()
def image_to_latent(pipeline, image, mode, seed, steps, cfg_strength, rescale_t, stage_encoders=False,
                    batch_guidance=False):
    from tensordict import TensorDict
    from idol.camera import decode_camera
    from idol.sampler import sample_flow

    staging = stage_encoders and pipeline.device.type == 'cuda'
    checkpoint()
    if staging:
        # The denoiser is not needed until image conditioning is complete.
        pipeline.denoiser.to('cpu')
        pipeline.background_remover.to(pipeline.device)
        torch.cuda.empty_cache()
    elif getattr(pipeline, '_image_models_parked', False):
        pipeline.background_remover.to(pipeline.device)
        pipeline.image_encoder.dinov3.to(pipeline.device)
        pipeline.image_encoder.flux2_vae.to(pipeline.device)
        pipeline._image_models_parked = False
    prepared = pipeline.preprocess(image, mode=mode)
    if staging:
        pipeline.background_remover.to('cpu')
        pipeline.image_encoder.dinov3.to(pipeline.device)
        pipeline.image_encoder.flux2_vae.to(pipeline.device)
        torch.cuda.empty_cache()
    pixels = torch.from_numpy(np.array(prepared)).permute(2, 0, 1).float().div(255.0)[None].to(pipeline.device)
    torch.manual_seed(seed)
    checkpoint()
    cond = pipeline.image_encoder(pixels)
    if staging:
        pipeline.image_encoder.dinov3.to('cpu')
        pipeline.image_encoder.flux2_vae.to('cpu')
        torch.cuda.empty_cache()
        pipeline.denoiser.to(pipeline.device)
        pipeline._image_models_parked = True
    noise = TensorDict({"latent": torch.randn(1, pipeline.denoiser.q_token_length, pipeline.denoiser.in_channels,
                                              device=pipeline.device),
                        "camera": torch.randn(1, 1, pipeline.denoiser.cam_channels, device=pipeline.device)},
                       batch_size=1, device=pipeline.device)
    with pipeline._sampling_settings():
        sample = sample_flow(pipeline.denoiser, noise, cond=cond, neg_cond=cond.apply(torch.zeros_like),
                             steps=steps, rescale_t=rescale_t, cfg_strength=cfg_strength, verbose=False,
                             batch_cfg=(pipeline.optimize or batch_guidance))
    latent = sample["latent"].detach().cpu().float().numpy()
    if not np.isfinite(latent).all():
        raise ValueError("denoiser produced non-finite latent values")
    camera = decode_camera(sample["camera"][0, 0])
    if not all(np.isfinite(value) for value in camera.values()):
        raise ValueError('denoiser produced a non-finite camera')
    return latent, prepared, camera


def validate_gaussian_cloud(cloud, expected_count):
    if cloud.xyz.ndim != 2 or tuple(cloud.xyz.shape) != (expected_count, 3):
        raise ValueError('GPU decoder produced an unexpected Gaussian count')
    for name in ('xyz', 'features_dc', 'scaling', 'rotation', 'opacity'):
        if not bool(torch.isfinite(getattr(cloud, name)).all()):
            raise ValueError('GPU decoder produced non-finite ' + name)


def load_gpu_decoders(model, fused_kernels=False):
    """Load the decoders, avoiding dense intermediate weights for fused inference."""
    from idol.models import OctreeProbabilityFixedlenDecoder, GaussianFixedlenDecoder
    from idol_packed import replace_quantized_linear_layers, enable_fused_packed
    checkpoint()
    if fused_kernels:
        from accelerate import init_empty_weights
        with init_empty_weights():
            octree = OctreeProbabilityFixedlenDecoder(**model.config['decoder']['args'])
            gaussian = GaussianFixedlenDecoder(**model.config['decoder_gs']['args'])
        octree = model.load_module_packed('octree', octree)
        gaussian = model.load_module_packed('gaussian', gaussian)
        enable_fused_packed(octree)
        enable_fused_packed(gaussian)
    else:
        octree = model.load_module('octree', OctreeProbabilityFixedlenDecoder(**model.config['decoder']['args']))
        gaussian = model.load_module('gaussian', GaussianFixedlenDecoder(**model.config['decoder_gs']['args']))
        replace_quantized_linear_layers(gaussian, model.tensors['gaussian'])
    return octree.eval().requires_grad_(False), gaussian.eval().requires_grad_(False)


@torch.no_grad()
def decode_gpu(model, latent, device, seed, num_gaussians, output, fused_kernels=False):
    """Use the upstream mixed-precision GPU decoders with packed Gaussian weights."""
    from idol.models import OctreeProbabilityFixedlenDecoder

    octree, gaussian = load_gpu_decoders(model, fused_kernels)
    octree = hold_model(octree.to(device), 'octree decoder')
    checkpoint()
    gaussian = hold_model(gaussian.eval().requires_grad_(False).to(device), 'Gaussian decoder')
    cond = torch.from_numpy(np.ascontiguousarray(latent, dtype=np.float32)).to(device)
    torch.manual_seed(seed)
    per_anchor = gaussian.rep_config['num_gaussians']
    if num_gaussians % per_anchor:
        raise ValueError('requested Gaussian count is not divisible by the model per-anchor count')
    checkpoint()
    anchors = OctreeProbabilityFixedlenDecoder.sample(octree, cond, num_points=num_gaussians // per_anchor,
                                                      level=model.config['max_voxel_level'])
    checkpoint()
    output_cloud = gaussian.to_representation(anchors, gaussian(anchors, cond))[0]
    validate_gaussian_cloud(output_cloud, num_gaussians)
    output_cloud.save_ply(str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path, help="local Idol GGUF file")
    source.add_argument("--repo", help="Hugging Face model repository containing the GGUF file")
    parser.add_argument("--file", help="model filename within --repo (required with --repo)")
    parser.add_argument("--revision", default="main", help="revision within --repo")
    parser.add_argument("--sdk", type=Path, help="upstream Idol SDK checkout, if not installed")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new output PLY file")
    parser.add_argument("--latent-out", type=Path, help="optionally save the generated float32 latent to a new file")
    parser.add_argument("--decoder", type=Path, help="built idol_decode executable")
    parser.add_argument("--decoder-cache", type=Path, help="reusable native decoder cache; created if absent")
    parser.add_argument("--decoder-backend", choices=("native", "gpu"), default="native",
                        help="native CPU decoder or experimental upstream mixed-precision GPU decoder")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dino", help="local image-encoder directory or repository override")
    parser.add_argument("--flux-vae", help="local image-encoder directory or repository override")
    parser.add_argument('--idol-only', action='store_true',
                        help='use original-precision image encoders; this is the default without encoder files')
    parser.add_argument("--dino-gguf", type=Path, help="experimental compatible DINOv3 GGUF weights")
    parser.add_argument("--flux-vae-gguf", type=Path, help="experimental compatible FLUX.2 autoencoder GGUF weights")
    parser.add_argument("--packed-denoiser", action="store_true", help="keep supported quantized denoiser linear weights packed on the device; unpack per call")
    parser.add_argument('--fused-kernels', action='store_true',
                        help='experimental CUDA packed-weight multiplication; implies --packed-denoiser')
    parser.add_argument('--stage-encoders', action='store_true', help='move image models off the GPU before denoising to reduce peak allocation')
    parser.add_argument("--mode", choices=("real", "synthetic"), default="synthetic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-strength", type=float, default=3.0)
    parser.add_argument("--rescale-t", type=float, default=3.0)
    parser.add_argument("--gaussians", type=int, default=262144)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    try:
        validate_encoder_choices(args.idol_only, args.dino_gguf, args.flux_vae_gguf)
    except ValueError as error:
        parser.error(str(error))
    if args.repo and not args.file:
        parser.error("--file is required with --repo")
    if args.sdk:
        sys.path.insert(0, str(args.sdk))
    image_output = args.out.with_name(args.out.stem + "_input.png")
    camera_output = args.out.with_name(args.out.stem + "_camera.json")
    destinations = [args.out, image_output, camera_output] + ([args.latent_out] if args.latent_out else [])
    if (not args.image.is_file() or any(path.exists() for path in (args.out, image_output, camera_output)) or
            len({path.resolve() for path in destinations}) != len(destinations) or
            (args.latent_out and (args.latent_out.exists() or not args.latent_out.parent.is_dir())) or
            args.out.suffix.lower() != ".ply" or not args.out.parent.is_dir() or args.steps < 1 or
            args.threads < 1 or args.seed < 0 or args.seed >= 2**64 or args.gaussians < 32 or args.gaussians > 1048576 or
            args.gaussians % 32 or args.rescale_t <= 0 or not np.isfinite([args.cfg_strength, args.rescale_t]).all()):
        parser.error("invalid input, output, sampling or thread parameters")
    decoder = str(args.decoder) if args.decoder else shutil.which("idol_decode")
    if args.decoder_backend == 'native' and (not decoder or not Path(decoder).is_file()):
        parser.error("build idol_decode and pass --decoder or add it to PATH")
    if args.decoder_backend == 'gpu' and (torch.device(args.device).type != 'cuda' or not torch.cuda.is_available()):
        parser.error('the GPU decoder requires an available CUDA device')
    if args.fused_kernels and (torch.device(args.device).type != 'cuda' or not torch.cuda.is_available()):
        parser.error('fused kernels require an available CUDA device')
    if args.repo:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(args.repo, args.file, revision=args.revision))
    else:
        path = args.model
    model = IdolGGUF(path)
    with tempfile.TemporaryDirectory(prefix="idol-inference-") as temporary:
        cache = (decoder_cache(model, args.decoder_cache or (Path(temporary) / "decoder"))
                 if args.decoder_backend == 'native' else None)
        pipeline = make_pipeline(model, args.device, args.dino, args.flux_vae, args.packed_denoiser,
                                 args.dino_gguf, args.flux_vae_gguf, args.fused_kernels)
        latent, prepared, camera = image_to_latent(pipeline, Image.open(args.image), args.mode, args.seed, args.steps,
                                                  args.cfg_strength, args.rescale_t, args.stage_encoders)
        del pipeline
        latent_path = Path(temporary) / "latent.f32"
        np.ascontiguousarray(latent, dtype="<f4").tofile(latent_path)
        # Both programs write different artifacts; refuse to overwrite an existing point cloud.
        temporary_ply = Path(temporary) / "result.ply"
        if args.decoder_backend == 'gpu':
            decode_gpu(model, latent, args.device, args.seed, args.gaussians, temporary_ply, args.fused_kernels)
        else:
            command = [decoder, str(cache), "--latent", str(latent_path), "--out", str(temporary_ply),
                       "--gaussians", str(args.gaussians), "--level", str(model.config["max_voxel_level"]),
                       "--seed", str(args.seed), "--threads", str(args.threads)]
            subprocess.run(command, check=True, env=dict(os.environ, IDOL_QUANT_KERNEL="float"))
        # Exclusive creation prevents overwriting a file created while inference was running.
        with args.out.open("xb") as stream, temporary_ply.open("rb") as source_ply:
            shutil.copyfileobj(source_ply, stream)
        with image_output.open("xb") as stream:
            prepared.save(stream, format="PNG")
        with camera_output.open("x", encoding="utf-8") as stream:
            json.dump(camera, stream, indent=2, allow_nan=False)
        if args.latent_out:
            with args.latent_out.open("xb") as stream, latent_path.open("rb") as saved_latent:
                shutil.copyfileobj(saved_latent, stream)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()

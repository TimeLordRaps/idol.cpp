"""Build the Idol pipeline without ever holding its models together on the GPU.

`idol_generate.make_pipeline` moves the denoiser, background remover and both image encoders onto the
device while it builds, and only afterwards does staged inference park the ones it is not using. The
sum of those models is the peak, and it is reached before the first image is processed. Here every
model is built on the CPU. Staged inference (`image_to_latent` with `stage_encoders=True`) then moves
each stage's models to the device just before use and back afterwards, so peak GPU memory is that of
the largest stage, not of all of them.

The image encoders also run one at a time (`SequentialImageEncoder`). Use this only with staged inference
(`stage_encoders=True`, as `IdolSession` does unless `resident_gpu` is set); a pipeline built here keeps its
models on the CPU until a stage needs them.

Only device placement changes: the same modules, dtypes and operations run in the same order, so
results are unchanged. This lives apart from `idol_generate.py` and `idol_session.py` because the
published benchmark receipts bind those files' exact bytes.
"""
from contextlib import contextmanager

import torch

import idol_generate as run


class SequentialImageEncoder:
    """Run the two image encoders one after the other, so only one is on the device while it works.

    `ImageEncoder.__call__` runs DINOv3 and then the FLUX.2 autoencoder, both resident. The autoencoder's
    full-resolution activations are large, so DINOv3's weights are moved off the device before it runs.
    The operations, their order and the random draw in the autoencoder are those of `ImageEncoder`, so
    the features are unchanged. Other attributes (`dinov3`, `flux2_vae`, `device`) are the wrapped encoder's."""

    def __init__(self, encoder):
        self._encoder = encoder

    def __getattr__(self, name):
        return getattr(self._encoder, name)

    @torch.no_grad()
    def __call__(self, images):
        from tensordict import TensorDict
        encoder = self._encoder
        images = images.to(encoder.device)
        encoder.dinov3.to(encoder.device)
        feature1 = encoder._dinov3_tokens(images)
        encoder.dinov3.to('cpu')
        torch.cuda.empty_cache()
        encoder.flux2_vae.to(encoder.device)
        feature2 = encoder._flux2_tokens(images)
        return TensorDict({'feature1': feature1, 'feature2': feature2}, batch_size=feature1.shape[0])


def make_staged_pipeline(model, device, dinov3, flux_vae, packed_denoiser=False, dino_gguf=None,
                         flux_vae_gguf=None, fused_kernels=False, cudnn_attention=False):
    """Same arguments and result as `idol_generate.make_pipeline`, with every model left on the CPU."""
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('staged construction requires an available CUDA device')
    # make_pipeline places models on the device it is given, so build on the CPU and retarget afterwards.
    # Fused kernels need the packed loader, which make_pipeline selects for packed_denoiser as well.
    pipeline = _build_on_cpu(model, dinov3, flux_vae, packed_denoiser or fused_kernels, dino_gguf, flux_vae_gguf)
    pipeline.device = device
    pipeline.image_encoder.device = device
    pipeline.background_remover.half()  # make_pipeline halves it on CUDA; the CPU build leaves it in float32
    if fused_kernels:
        from idol_packed import enable_fused_packed
        count = enable_fused_packed(pipeline.denoiser) + enable_fused_packed(pipeline.image_encoder.dinov3)
        if not count:
            raise ValueError('fused kernels require supported packed denoiser or DINOv3 matrices')
    pipeline.optimize = bool(cudnn_attention)
    pipeline.image_encoder = SequentialImageEncoder(pipeline.image_encoder)
    return pipeline


def _build_on_cpu(model, dinov3, flux_vae, packed, dino_gguf, flux_vae_gguf):
    return _original_make_pipeline(model, 'cpu', dinov3, flux_vae, packed, dino_gguf, flux_vae_gguf, False, False)


_original_make_pipeline = run.make_pipeline


@contextmanager
def staged_construction():
    """Make `idol_generate.make_pipeline` build on the CPU for the duration of the block.

    `IdolSession` and the command-line tool look `make_pipeline` up when they build, so wrapping the
    construction is enough; nothing is changed permanently."""
    previous = run.make_pipeline
    run.make_pipeline = make_staged_pipeline
    try:
        yield
    finally:
        run.make_pipeline = previous

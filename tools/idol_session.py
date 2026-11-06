"""Reuse an Idol image pipeline and graphics-device decoder across generations.

By default, models are staged through host memory between phases to limit peak
graphics memory. The optional resident mode keeps them on the device instead.
This interface is sequential: it uses PyTorch's process-wide random generator,
so concurrent inference in the same process is not supported. It requires a
graphics processing unit (GPU) with Compute Unified Device Architecture (CUDA)
support.
"""
import math
import shutil
import tempfile
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import idol_generate as run
from idol_gguf import IdolGGUF


_PROCESS_INFERENCE_LOCK = threading.Lock()


class IdolSession:
    """Load a full model once and generate sequential point clouds on the GPU."""

    def __init__(self, model, *, device='cuda', dino=None, flux_vae=None,
                 packed_denoiser=False, dino_gguf=None, flux_vae_gguf=None, fused_kernels=False,
                 resident_gpu=False, batch_guidance=False, cudnn_attention=False):
        self.device = torch.device(device)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('a reusable GPU session requires an available CUDA device')
        self._lock = threading.Lock()
        self._closed = False
        self._failed = False
        self._decoders_registered = False
        self.resident_gpu = resident_gpu
        self.batch_guidance = batch_guidance
        self.model = IdolGGUF(Path(model))
        self.pipeline = run.make_pipeline(self.model, self.device, dino, flux_vae,
                                          packed_denoiser, dino_gguf, flux_vae_gguf, fused_kernels,
                                          cudnn_attention)
        run.checkpoint()
        self.octree, self.gaussian = run.load_gpu_decoders(self.model, fused_kernels)

    @torch.no_grad()
    def generate(self, image, output, *, mode='real', seed=42, steps=50,
                 cfg_strength=3.0, rescale_t=3.0, gaussians=262144):
        """Write a new Polygon File Format (PLY) point cloud and return its data.

        The result contains the latent, preprocessed image, and camera. The
        output must not exist. A failed inference invalidates the session;
        create a new session before trying again.
        """
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('a session cannot generate concurrently')
        if not _PROCESS_INFERENCE_LOCK.acquire(blocking=False):
            self._lock.release()
            raise RuntimeError('reusable sessions cannot generate concurrently in one process')
        started = False
        try:
            if self._closed or self._failed:
                raise RuntimeError('session is closed or invalid after a failed inference')
            image, output = Path(image), Path(output)
            per_anchor = self.gaussian.rep_config['num_gaussians']
            if (not image.is_file() or output.exists() or not output.parent.is_dir()
                    or output.suffix.lower() != '.ply' or image.resolve() == output.resolve()
                    or mode not in ('real', 'synthetic') or not isinstance(seed, int) or isinstance(seed, bool)
                    or not 0 <= seed < 2**64 or not isinstance(steps, int) or isinstance(steps, bool) or steps < 1
                    or not isinstance(gaussians, int) or isinstance(gaussians, bool)
                    or not per_anchor <= gaussians <= 1048576 or gaussians % per_anchor
                    or not math.isfinite(cfg_strength) or not math.isfinite(rescale_t) or rescale_t <= 0):
                raise ValueError('invalid image, output, sampling, or Gaussian count')
            started = True
            with Image.open(image) as source:
                latent, prepared, camera = run.image_to_latent(
                    self.pipeline, source, mode, seed, steps, cfg_strength, rescale_t,
                    not self.resident_gpu, self.batch_guidance)
            # Keep the next denoising pass and decoder weights from sharing the
            # device; the image encoders are already parked by staged inference.
            if not self.resident_gpu:
                self.pipeline.denoiser.to('cpu')
                torch.cuda.empty_cache()
            run.checkpoint()
            try:
                self.octree.to(self.device)
                self.gaussian.to(self.device)
                if not self._decoders_registered:
                    self.octree = run.hold_model(self.octree, 'reusable octree decoder')
                    self.gaussian = run.hold_model(self.gaussian, 'reusable Gaussian decoder')
                    self._decoders_registered = True
                run.checkpoint()
                cond = torch.from_numpy(np.ascontiguousarray(latent, dtype=np.float32)).to(self.device)
                torch.manual_seed(seed)
                from idol.models import OctreeProbabilityFixedlenDecoder
                anchors = OctreeProbabilityFixedlenDecoder.sample(
                    self.octree, cond, num_points=gaussians // per_anchor,
                    level=self.model.config['max_voxel_level'])
                run.checkpoint()
                cloud = self.gaussian.to_representation(anchors, self.gaussian(anchors, cond))[0]
                run.validate_gaussian_cloud(cloud, gaussians)
                with tempfile.TemporaryDirectory(prefix='idol-session-') as temporary:
                    point_cloud = Path(temporary) / 'result.ply'
                    cloud.save_ply(str(point_cloud))
                    with output.open('xb') as destination, point_cloud.open('rb') as source:
                        shutil.copyfileobj(source, destination)
            finally:
                if not self.resident_gpu:
                    self.octree.to('cpu')
                    self.gaussian.to('cpu')
                    torch.cuda.empty_cache()
            return {'latent': latent, 'preprocessed_image': prepared, 'camera': camera}
        except Exception:
            if started:
                self._failed = True
            raise
        finally:
            _PROCESS_INFERENCE_LOCK.release()
            self._lock.release()

    def close(self):
        """Release this session's references after all inference has stopped."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('cannot close a running session')
        try:
            self._closed = True
            try:
                if self.pipeline is not None and not getattr(self, 'resident_gpu', False):
                    for module in (self.pipeline.denoiser, self.pipeline.background_remover,
                                   self.pipeline.image_encoder.dinov3, self.pipeline.image_encoder.flux2_vae):
                        module.to('cpu')
                if not getattr(self, 'resident_gpu', False):
                    for module in (self.octree, self.gaussian):
                        if module is not None:
                            module.to('cpu')
            finally:
                self.pipeline = None
                self.octree = None
                self.gaussian = None
                self.model = None
                torch.cuda.empty_cache()
        finally:
            self._lock.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

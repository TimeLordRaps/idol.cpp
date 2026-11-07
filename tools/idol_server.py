#!/usr/bin/env python3
"""Serve the reusable Idol image-to-3D pipeline and a local browser interface."""
import argparse
import base64
import binascii
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from idol_memory import available_mib, keep_idol_loaded, plan_idol

MAX_IMAGE_BYTES = 20 * 1024 * 1024


def result_file(directory, identifier, suffix):
    if len(identifier) != 32 or any(c not in '0123456789abcdef' for c in identifier):
        raise FileNotFoundError('Result not found')
    path = Path(directory) / (identifier + suffix)
    if not path.is_file():
        raise FileNotFoundError('Result not found')
    return path


class BusyError(Exception):
    pass


def default_mode_for_source(source):
    return 'synthetic' if source == 'Prompt' else 'real'


def image_from_prompt(endpoint, prompt):
    """Use a separately operated image server with the stable-diffusion.cpp API."""
    if not endpoint:
        raise ValueError('Text prompts require a local --image-config or a running --image-server')
    if not prompt or not prompt.strip() or len(prompt) > 4000:
        raise ValueError('Prompt must contain between 1 and 4000 characters')
    request = Request(endpoint.rstrip('/') + '/v1/images/generations',
                      data=json.dumps({'prompt': prompt, 'n': 1, 'size': '1024x1024',
                                       'output_format': 'png'}).encode(),
                      headers={'Content-Type': 'application/json'}, method='POST')
    with urlopen(request, timeout=600) as response:
        raw = response.read(MAX_IMAGE_BYTES * 2 + 4096)
    try:
        item = json.loads(raw)['data'][0]
        encoded = item['b64_json']
        image = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError, KeyError, IndexError, binascii.Error) as exc:
        raise ValueError('Image server returned an invalid image response') from exc
    if not image or len(image) > MAX_IMAGE_BYTES:
        raise ValueError('Generated image is empty or exceeds 20 MiB')
    info = item.get('generation_info')
    models = info.get('models') if isinstance(info, dict) else None
    if not isinstance(models, dict):
        raise ValueError('Image service did not report its model files for this image')
    models = {key: value for key, value in models.items()
              if isinstance(key, str) and isinstance(value, str) and value.strip()
              and len(key) <= 128 and len(value) <= 1024}
    model = models.get('diffusion_model') or models.get('model')
    if not model or model.strip().casefold() in {'unknown', 'none', 'n/a', 'default', 'sd-cpp-local'}:
        raise ValueError('Image service did not identify the image-generation model for this image')
    return image, model, models


class Runtime:
    """Hold one Idol session and, optionally, an image generator.

    `resident_gpu`, `batch_guidance` and `cudnn_attention` left as None are chosen from the graphics memory
    that is free when the session is built (`idol_memory.plan_idol`): everything resident and fast when it
    fits, otherwise staged through the device. A value that is set is used as given. `vram_budget_gib`
    states how much graphics memory this program may count on instead of the amount measured free."""

    def __init__(self, model, output_dir, *, image_server=None, device='cuda',
                 packed_denoiser=False, fused_kernels=False, resident_gpu=None, session_factory=None,
                 image_generator=None, batch_guidance=None, cudnn_attention=None,
                 dino_gguf=None, flux_vae_gguf=None, mesh_executable=None,
                 vram_budget_gib=None, memory_probe=None):
        self.model = str(model)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_server = image_server
        self.image_generator = image_generator
        self.device = device
        self.packed_denoiser = packed_denoiser
        self.fused_kernels = fused_kernels
        self.resident_gpu = resident_gpu
        self.batch_guidance = batch_guidance
        self.cudnn_attention = cudnn_attention
        self.dino_gguf = dino_gguf
        self.flux_vae_gguf = flux_vae_gguf
        self.session_factory = session_factory
        self.vram_budget_gib = vram_budget_gib
        self.memory_probe = memory_probe or (lambda: available_mib(self.device, self.vram_budget_gib))
        self.memory_mode = None
        self.session = None
        self.lock = threading.Lock()
        from idol_mesh_tools import find_mesh_executable
        self.mesh_executable = find_mesh_executable(mesh_executable)

    def _session_settings(self):
        """(resident, batch guidance, cuDNN attention) for the next session, and why."""
        if self.resident_gpu is not None:
            # An explicit mode is honored exactly: the speed options stay off unless they were asked for.
            return self.resident_gpu, bool(self.batch_guidance), bool(self.cudnn_attention), 'set explicitly'
        plan = plan_idol(self.memory_probe())
        batch = plan.batch_guidance if self.batch_guidance is None else self.batch_guidance
        cudnn = plan.cudnn_attention if self.cudnn_attention is None else self.cudnn_attention
        return plan.resident, batch, cudnn, plan.reason

    def _get_session(self):
        if self.session is None:
            resident, batch_guidance, cudnn_attention, reason = self._session_settings()
            self.memory_mode = 'resident' if resident else 'staged'
            print(f'Idol memory mode: {self.memory_mode} ({reason})', file=sys.stderr, flush=True)
            factory = self.session_factory
            construction = nullcontext()
            if factory is None:
                from idol_session import IdolSession
                factory = IdolSession
                if not resident:
                    # Staged inference moves each model to the GPU only while it is used; build them on the CPU
                    # so they are never all on the GPU at once.
                    from idol_staging import staged_construction
                    construction = staged_construction()
            with construction:
                self.session = factory(self.model, device=self.device,
                                       packed_denoiser=self.packed_denoiser,
                                       fused_kernels=self.fused_kernels,
                                       resident_gpu=resident, batch_guidance=batch_guidance,
                                       cudnn_attention=cudnn_attention,
                                       dino_gguf=self.dino_gguf, flux_vae_gguf=self.flux_vae_gguf)
        return self.session

    def _release_session_for_image_generator(self):
        """Free the session before a prompt's image is generated, unless both comfortably fit together.

        With a stated budget the session's own use cannot be separated from it, so it is always released."""
        if self.session is None:
            return
        if self.vram_budget_gib is None and keep_idol_loaded(self.memory_probe()):
            return
        previous, self.session, self.memory_mode = self.session, None, None
        previous.close()

    def preload(self):
        with self.lock:
            self._get_session()

    def run(self, *, image=None, prompt=None, mode='real', seed=42, steps=50,
            gaussians=262144):
        if not self.lock.acquire(blocking=False):
            raise BusyError('A reconstruction is already running; try again when it finishes')
        try:
            if bool(image) == bool(prompt):
                raise ValueError('Supply exactly one image or prompt')
            if mode not in ('real', 'synthetic'):
                raise ValueError('Mode must be real or synthetic')
            if not 0 <= seed < 2**64 or not 1 <= steps <= 1000:
                raise ValueError('Seed or step count is out of range')
            if not 32 <= gaussians <= 1048576 or gaussians % 32:
                raise ValueError('Gaussian count must be a multiple of 32 between 32 and 1048576')
            model, model_components = None, None
            if prompt:
                if self.image_generator:
                    self._release_session_for_image_generator()
                    image, model, model_components = self.image_generator.generate(prompt, seed)
                else:
                    image, model, model_components = image_from_prompt(self.image_server, prompt)
            if not image or len(image) > MAX_IMAGE_BYTES:
                raise ValueError('Image is empty or exceeds 20 MiB')
            from PIL import Image, UnidentifiedImageError
            import io
            try:
                with Image.open(io.BytesIO(image)) as check:
                    check.verify()
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                raise ValueError('Input is not a valid supported image') from exc
            with tempfile.TemporaryDirectory(prefix='idol-server-input-') as temporary:
                source = Path(temporary) / 'input.image'
                source.write_bytes(image)
                output = self.output_dir / (uuid.uuid4().hex + '.ply')
                self._get_session()
                try:
                    self.session.generate(source, output, mode=mode, seed=seed,
                                          steps=steps, gaussians=gaussians)
                except Exception:
                    # IdolSession explicitly invalidates itself after failed inference.
                    failed_session, self.session = self.session, None
                    try:
                        failed_session.close()
                    except Exception:
                        pass
                    raise
                metadata = {'input': 'generated_image' if prompt else 'uploaded_image',
                            'image_sha256': hashlib.sha256(image).hexdigest(),
                            'image_model': model, 'image_model_components': model_components,
                            'image_model_source': ('local runner invocation' if self.image_generator else
                                                   'image service generation response') if prompt else None,
                            'image_model_license': self.image_generator.license_notice if prompt and self.image_generator else None}
                if prompt:
                    displayed = output.with_suffix('.input.png')
                    with Image.open(io.BytesIO(image)) as generated:
                        generated.save(displayed, format='PNG')
                    metadata['display_image_sha256'] = hashlib.sha256(displayed.read_bytes()).hexdigest()
                with output.with_suffix('.metadata.json').open('x', encoding='utf-8') as file:
                    json.dump(metadata, file, indent=2, ensure_ascii=False)
                return output
        finally:
            self.lock.release()

    def mesh(self, result_id, options):
        from idol_mesh_tools import extract_mesh
        if not self.lock.acquire(blocking=False):
            raise BusyError('A reconstruction or mesh operation is already running')
        try:
            return extract_mesh(result_file(self.output_dir, result_id, '.ply'), self.output_dir,
                                self.mesh_executable, options)
        finally:
            self.lock.release()

    def compare_mesh(self, mesh_id, reference, samples=20000, seed=23):
        from idol_mesh_tools import compare_mesh_files
        if not self.lock.acquire(blocking=False):
            raise BusyError('A reconstruction or mesh operation is already running')
        try:
            candidate = result_file(self.output_dir, mesh_id, '.obj')
            return compare_mesh_files(reference, candidate, self.output_dir, samples, seed)
        finally:
            self.lock.release()

    def mesh_uploaded(self, source, options):
        from idol_mesh_tools import extract_mesh
        if not self.lock.acquire(blocking=False):
            raise BusyError('A reconstruction or mesh operation is already running')
        try:
            return extract_mesh(source, self.output_dir, self.mesh_executable, options)
        finally:
            self.lock.release()

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None


def create_app(runtime, *, ui=True):
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse
    from pydantic import BaseModel, Field
    from idol_mesh_tools import MAX_MESH_BYTES, MAX_SPLAT_BYTES, MeshOptions

    class PromptRequest(BaseModel):
        prompt: str = Field(min_length=1, max_length=4000)
        mode: str = 'synthetic'
        seed: int = 42
        steps: int = 50
        gaussians: int = 262144

    class MeshRequest(BaseModel):
        resolution: int = Field(default=176, ge=32, le=256)
        sigma: float = Field(default=1.5, gt=0, le=8)
        iso: float = Field(default=0.2, gt=0, le=1)
        min_piece: float = Field(default=0.05, ge=0, le=1)

    app = FastAPI(title='idol-server')

    def execute(**kwargs):
        try:
            return runtime.run(**kwargs)
        except BusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def mesh_operation(function, *args):
        try:
            return function(*args)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail='Result not found') from exc
        except BusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def model_response(path):
        metadata = json.loads(path.with_suffix('.metadata.json').read_text(encoding='utf-8'))
        headers = {'X-Idol-Metadata': f'/v1/reconstructions/{path.stem}/metadata'}
        if metadata['image_model']:
            headers['X-Idol-Image-Model'] = quote(metadata['image_model'], safe='')
            headers['X-Idol-Input'] = f'/v1/reconstructions/{path.stem}/input'
        return FileResponse(path, media_type='application/octet-stream', filename=path.name, headers=headers)

    @app.get('/health')
    def health():
        return {'status': 'busy' if runtime.lock.locked() else 'ready',
                'model_loaded': runtime.session is not None,
                'memory_mode': runtime.memory_mode,
                'text_to_image_configured': bool(runtime.image_server or runtime.image_generator),
                'image_model': runtime.image_generator.label if runtime.image_generator else None,
                'mesh_generation_available': runtime.mesh_executable is not None}

    @app.post('/v1/reconstructions/{result_id}/mesh')
    def generate_mesh(result_id: str, request: MeshRequest):
        path = mesh_operation(runtime.mesh, result_id, MeshOptions(**request.model_dump()))
        return FileResponse(path, media_type='text/plain', filename=path.name,
                            headers={'X-Idol-Mesh-Metadata': f'/v1/meshes/{path.stem}/metadata'})

    @app.post('/v1/meshes')
    def uploaded_mesh(splats: UploadFile = File(...), resolution: int = Form(176), sigma: float = Form(1.5),
                      iso: float = Form(0.2), min_piece: float = Form(0.05)):
        data = splats.file.read(MAX_SPLAT_BYTES + 1)
        if not data or len(data) > MAX_SPLAT_BYTES:
            raise HTTPException(status_code=400, detail='Gaussian point cloud is empty or exceeds 80 MiB')
        with tempfile.TemporaryDirectory(prefix='idol-splats-') as directory:
            source = Path(directory) / 'uploaded.ply'
            source.write_bytes(data)
            path = mesh_operation(runtime.mesh_uploaded, source, MeshOptions(resolution, sigma, iso, min_piece))
        return FileResponse(path, media_type='text/plain', filename=path.name,
                            headers={'X-Idol-Mesh-Metadata': f'/v1/meshes/{path.stem}/metadata'})

    @app.get('/v1/meshes/{mesh_id}/metadata')
    def mesh_metadata(mesh_id: str):
        path = mesh_operation(result_file, runtime.output_dir, mesh_id, '.mesh.json')
        return json.loads(path.read_text(encoding='utf-8'))

    @app.post('/v1/meshes/{mesh_id}/comparisons')
    def compare_mesh(mesh_id: str, reference: UploadFile = File(...), samples: int = Form(20000),
                     seed: int = Form(23)):
        data = reference.file.read(MAX_MESH_BYTES + 1)
        if not data or len(data) > MAX_MESH_BYTES:
            raise HTTPException(status_code=400, detail='Reference mesh is empty or exceeds 32 MiB')
        with tempfile.TemporaryDirectory(prefix='idol-reference-') as directory:
            path = Path(directory) / 'reference.obj'
            path.write_bytes(data)
            report, _ = mesh_operation(runtime.compare_mesh, mesh_id, path, samples, seed)
        return report

    @app.post('/v1/reconstructions')
    def reconstruct(image: UploadFile = File(...), mode: str = Form('real'),
                    seed: int = Form(42), steps: int = Form(50),
                    gaussians: int = Form(262144)):
        data = image.file.read(MAX_IMAGE_BYTES + 1)
        path = execute(image=data, mode=mode, seed=seed, steps=steps, gaussians=gaussians)
        return model_response(path)

    @app.post('/v1/text-to-3d')
    def text_to_3d(request: PromptRequest):
        path = execute(prompt=request.prompt, mode=request.mode, seed=request.seed,
                       steps=request.steps, gaussians=request.gaussians)
        return model_response(path)

    @app.get('/v1/reconstructions/{result_id}/metadata')
    def result_metadata(result_id: str):
        if len(result_id) != 32 or any(char not in '0123456789abcdef' for char in result_id):
            raise HTTPException(status_code=404, detail='Result not found')
        path = runtime.output_dir / (result_id + '.metadata.json')
        if not path.is_file():
            raise HTTPException(status_code=404, detail='Result not found')
        return json.loads(path.read_text(encoding='utf-8'))

    @app.get('/v1/reconstructions/{result_id}/input')
    def generated_input(result_id: str):
        if len(result_id) != 32 or any(char not in '0123456789abcdef' for char in result_id):
            raise HTTPException(status_code=404, detail='Result not found')
        path = runtime.output_dir / (result_id + '.input.png')
        if not path.is_file():
            raise HTTPException(status_code=404, detail='Generated input not found')
        return FileResponse(path, media_type='image/png')

    if ui:
        import gradio as gr

        def generate(image, prompt, source, mode, seed, steps, gaussians):
            try:
                data = None
                if source == 'Image':
                    if not image:
                        raise ValueError('Choose an image first')
                    with open(image, 'rb') as handle:
                        data = handle.read(MAX_IMAGE_BYTES + 1)
                started = time.monotonic()
                path = runtime.run(image=data, prompt=prompt if source == 'Prompt' else None,
                                   mode=mode, seed=int(seed), steps=int(steps),
                                   gaussians=int(gaussians))
                metadata = json.loads(path.with_suffix('.metadata.json').read_text(encoding='utf-8'))
                model = metadata['image_model']
                model_label = ((f"Local image model: {model}" if runtime.image_generator else
                                f"Image service reported: {model}") if model else
                               'Uploaded image; no image-generation model was used')
                encoder = (metadata['image_model_components'] or {}).get('text_encoder')
                if model and encoder:
                    model_label += f' | text encoder: {encoder}'
                if model and metadata['image_model_license']:
                    model_label += f" — {metadata['image_model_license']}"
                generated = str(path.with_suffix('.input.png')) if source == 'Prompt' else gr.update()
                return (str(path), f'Generated in {time.monotonic() - started:.1f} seconds. Saved to {path.name}.',
                        model_label, generated, path.stem, None, None, None, '', None, None, None)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        def generate_prompt(prompt, seed, steps, gaussians):
            return generate(None, prompt, 'Prompt', 'synthetic', seed, steps, gaussians)

        def mesh_from_result(result_id, saved_splats, resolution, sigma, iso, min_piece):
            try:
                options = MeshOptions(int(resolution), float(sigma), float(iso), float(min_piece))
                if saved_splats:
                    path = runtime.mesh_uploaded(saved_splats, options)
                elif result_id:
                    path = runtime.mesh(result_id, options)
                else:
                    raise ValueError('Generate a reconstruction or upload a saved Gaussian point cloud first')
                info = json.loads(path.with_suffix('.mesh.json').read_text(encoding='utf-8'))
                return (str(path.with_suffix('.glb')), str(path), path.stem,
                        f"{info['vertices']:,} vertices, {info['triangles']:,} triangles in {info['seconds']:.2f} seconds.",
                        None, None)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        def compare_with_reference(mesh_id, reference):
            try:
                if not mesh_id or not reference:
                    raise ValueError('Generate a mesh and upload a reference OBJ mesh first')
                if Path(reference).suffix.lower() != '.obj':
                    raise ValueError('Upload a Wavefront OBJ reference mesh')
                report, path = runtime.compare_mesh(mesh_id, reference)
                measured = report['measurements']
                surface, diagonal = measured['sampled_point_to_triangle'], measured['reference_bounding_box_diagonal']
                summary = {'Mean distance (% of reference diagonal)': 100 * surface['symmetric_mean_distance'] / diagonal,
                           '95th percentile distance (% of reference diagonal)': 100 * surface['symmetric_p95_distance'] / diagonal,
                           'Maximum sampled distance (% of reference diagonal)': 100 * surface['maximum_sampled_distance'] / diagonal}
                for fraction, score in surface['f_scores_by_reference_diagonal_fraction'].items():
                    summary[f'F1 (%) within {100 * float(fraction):g}% of reference diagonal'] = 100 * score['f1']
                summary['Samples per mesh'] = measured['surface_samples_per_mesh']
                summary['Reference / candidate triangles'] = [measured['reference_faces'], measured['candidate_faces']]
                summary['Extraction settings match (null means unknown)'] = report['extraction_settings_match']
                return {key: round(value, 4) if isinstance(value, float) else value
                        for key, value in summary.items()}, str(path)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        def preview_reference(reference):
            try:
                if not reference or Path(reference).suffix.lower() != '.obj':
                    raise ValueError('Upload a Wavefront OBJ reference mesh')
                from idol_mesh_tools import mesh_preview
                return str(mesh_preview(reference, runtime.output_dir)), reference, None, None
            except Exception as exc:
                gr.Warning(str(exc))
                return None, None, None, None

        with gr.Blocks(title='Idol 3D', analytics_enabled=False) as demo:
            gr.Markdown('# Idol 3D\nReconstruct a 3D Gaussian point cloud from an image. '
                        'For prompts, a configured image generator creates the input image first.')
            reconstruction_id, mesh_id, reference_path = gr.State(), gr.State(), gr.State()
            with gr.Row():
                with gr.Column():
                    source = gr.Radio(['Image', 'Prompt'], value='Image', label='Input')
                    image = gr.Image(type='filepath', sources=['upload', 'webcam'], label='Image')
                    prompt = gr.Textbox(label='Prompt', placeholder=(
                        'Describe the object to create' if runtime.image_generator or runtime.image_server
                        else 'Configure an image generator to use prompts'))
                    mode = gr.Radio(['real', 'synthetic'], value='real', label='Image type')
                    with gr.Accordion('Sampling', open=False):
                        seed = gr.Number(value=42, precision=0, label='Seed')
                        steps = gr.Slider(1, 100, value=50, step=1, label='Steps')
                        gaussians = gr.Slider(32768, 1048576, value=262144, step=32768,
                                              label='Gaussian points')
                    button = gr.Button('Generate 3D', variant='primary')
                with gr.Column():
                    with gr.Tabs():
                        with gr.Tab('Gaussian point cloud'):
                            result = gr.Model3D(label='3D reconstruction', height=600)
                        with gr.Tab('Mesh'):
                            mesh_view = gr.Model3D(label='Generated mesh', height=450)
                            mesh_download = gr.File(label='Download mesh (.obj)', interactive=False)
                            gr.Markdown('Extract untextured geometry from the current Gaussian point cloud. '
                                        'Use identical settings for the original and quantized models.')
                            saved_splats = gr.File(label='Saved Gaussian point cloud (optional .ply)', file_types=['.ply'],
                                                   elem_id='saved-splats-upload')
                            with gr.Accordion('Mesh extraction settings', open=False):
                                resolution = gr.Slider(32, 256, value=176, step=8, label='Grid resolution')
                                sigma = gr.Slider(0.25, 8, value=1.5, step=0.25, label='Blur sigma (voxels)')
                                iso = gr.Slider(0.01, 1, value=0.2, step=0.01, label='Density threshold fraction')
                                min_piece = gr.Slider(0, 1, value=0.05, step=0.01, label='Smallest component fraction')
                            mesh_button = gr.Button('Generate mesh', interactive=runtime.mesh_executable is not None)
                            mesh_status = gr.Textbox(label='Mesh status', interactive=False,
                                value='' if runtime.mesh_executable else 'Build idol_mesh or pass --mesh-executable to enable extraction.')
                            reference_mesh = gr.Model3D(label='Reference mesh (upload .obj)', height=300,
                                                        elem_id='mesh-reference')
                            compare_button = gr.Button('Compare mesh with reference')
                            gr.Markdown('Sampled points are compared with the other mesh’s closest triangles, without alignment. '
                                        'Distances use model coordinates. F1 is the harmonic mean of precision and recall '
                                        'within 0.1%, 0.5% and 1% of the reference bounding-box diagonal. '
                                        'These comparisons do not measure color or ground-truth reconstruction accuracy.')
                            comparison = gr.JSON(label='Mesh comparison', elem_id='mesh-comparison')
                            comparison_download = gr.File(label='Download comparison report', interactive=False)
                    image_model = gr.Textbox(label='Image-generation model', interactive=False,
                        value=(runtime.image_generator.label if runtime.image_generator else
                               'Waiting for a generated image' if runtime.image_server else 'None configured'))
                    status = gr.Textbox(label='Status', interactive=False)
            inputs = [image, prompt, source, mode, seed, steps, gaussians]
            outputs = [result, status, image_model, image, reconstruction_id, mesh_view, mesh_download,
                       mesh_id, mesh_status, comparison, comparison_download, saved_splats]
            button.click(generate, inputs, outputs, concurrency_limit=1,
                         concurrency_id='reconstruction')
            source.change(default_mode_for_source, source, mode, queue=False)
            prompt.submit(generate_prompt, [prompt, seed, steps, gaussians],
                          outputs, concurrency_limit=1, concurrency_id='reconstruction')
            mesh_button.click(mesh_from_result, [reconstruction_id, saved_splats, resolution, sigma, iso, min_piece],
                              [mesh_view, mesh_download, mesh_id, mesh_status, comparison, comparison_download],
                              concurrency_limit=1, concurrency_id='reconstruction')
            reference_mesh.upload(preview_reference, reference_mesh,
                                  [reference_mesh, reference_path, comparison, comparison_download],
                                  concurrency_limit=1, concurrency_id='reconstruction')
            reference_mesh.clear(lambda: (None, None, None),
                                 outputs=[reference_path, comparison, comparison_download], queue=False)
            for event in (saved_splats.upload, saved_splats.clear):
                event(lambda: (None, None, None, 'Generate mesh for the selected source.', None, None),
                      outputs=[mesh_view, mesh_download, mesh_id, mesh_status, comparison, comparison_download],
                      queue=False)
            compare_button.click(compare_with_reference, [mesh_id, reference_path],
                                 [comparison, comparison_download], concurrency_limit=1, concurrency_id='reconstruction')
        app = gr.mount_gradio_app(app, demo, path='/',
                                 allowed_paths=[str(runtime.output_dir.resolve())])
    return app


def main():
    parser = argparse.ArgumentParser(description='Run the Idol API and browser interface')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--model', help='Path to the complete Idol GGUF model')
    source.add_argument('--repo', help='Hugging Face repository containing the model')
    parser.add_argument('--file', help='Model filename in --repo')
    parser.add_argument('--revision', default='main', help='Repository revision (default: main)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8127)
    parser.add_argument('--output-dir', default='idol-outputs')
    parser.add_argument('--mesh-executable', type=Path, help='Native idol_mesh executable (otherwise found in PATH or build/)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--image-server', help='Base URL of a separate compatible text-to-image server')
    parser.add_argument('--image-config', help='Local image-generator configuration JSON file')
    parser.add_argument('--dino-gguf', type=Path, help='Compatible DINOv3 encoder GGUF file')
    parser.add_argument('--flux-vae-gguf', type=Path, help='Compatible FLUX.2 autoencoder GGUF file')
    parser.add_argument('--preload', action='store_true', help='Load the model before accepting requests')
    parser.add_argument('--packed-denoiser', action='store_true')
    parser.add_argument('--fused-kernels', action='store_true')
    parser.add_argument('--memory-mode', choices=('auto', 'resident', 'staged'), default='auto',
                        help='auto (default) measures free graphics memory: with enough, every model stays on the '
                             'device with batched guidance and cuDNN attention (fastest); otherwise models are built '
                             'on the CPU and visit the device one stage at a time. resident and staged force a mode.')
    parser.add_argument('--resident-gpu', dest='memory_mode', action='store_const', const='resident',
                        help='Same as --memory-mode resident: keep models on the graphics device between pipeline phases')
    parser.add_argument('--batch-guidance', action=argparse.BooleanOptionalAction, default=None,
                        help='Batch the conditional and unconditional denoising passes (default: on in resident auto mode)')
    parser.add_argument('--cudnn-attention', action=argparse.BooleanOptionalAction, default=None,
                        help='Use cuDNN attention and TF32 matmuls while sampling (default: on in resident auto mode)')
    parser.add_argument('--vram-budget-gib', type=float,
                        help='Graphics memory, in GiB, this program may count on, instead of the amount measured free. '
                             'Use it to leave room for other programs, or when the device cannot be measured')
    args = parser.parse_args()
    if args.vram_budget_gib is not None and not 0.5 <= args.vram_budget_gib <= 1024:
        parser.error('--vram-budget-gib must be between 0.5 and 1024')
    if args.repo and not args.file:
        parser.error('--file is required with --repo')
    if args.model and args.file:
        parser.error('--file requires --repo')
    if args.model and not Path(args.model).is_file():
        parser.error('--model must name an existing model file')
    if any(path is not None and not path.is_file() for path in (args.dino_gguf, args.flux_vae_gguf)):
        parser.error('image-encoder GGUF paths must name existing files')
    if args.mesh_executable is not None and not args.mesh_executable.is_file():
        parser.error('--mesh-executable must name an existing idol_mesh executable')
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    if args.image_server:
        parsed = urlsplit(args.image_server)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.query or parsed.fragment:
            parser.error('--image-server must be an HTTP or HTTPS base URL')
    if args.image_server and args.image_config:
        parser.error('--image-server and --image-config are mutually exclusive')
    if not args.image_server and not args.image_config:
        from setup_iris import default_config_path
        default_config = default_config_path()
        if default_config.is_file():
            args.image_config = str(default_config)
    image_generator = None
    if args.image_config:
        from idol_image import LocalImageGenerator
        image_generator = LocalImageGenerator(
            args.image_config, memory_probe=lambda: available_mib(args.device, args.vram_budget_gib))
    if args.repo:
        from huggingface_hub import hf_hub_download
        args.model = hf_hub_download(args.repo, args.file, revision=args.revision)
    import uvicorn
    runtime = Runtime(args.model, args.output_dir, image_server=args.image_server,
                      device=args.device, packed_denoiser=args.packed_denoiser,
                      fused_kernels=args.fused_kernels,
                      resident_gpu={'auto': None, 'resident': True, 'staged': False}[args.memory_mode],
                      image_generator=image_generator, batch_guidance=args.batch_guidance,
                      cudnn_attention=args.cudnn_attention, dino_gguf=args.dino_gguf,
                      flux_vae_gguf=args.flux_vae_gguf, mesh_executable=args.mesh_executable,
                      vram_budget_gib=args.vram_budget_gib)
    try:
        if args.preload:
            runtime.preload()
        uvicorn.run(create_app(runtime), host=args.host, port=args.port)
    finally:
        runtime.close()


if __name__ == '__main__':
    main()

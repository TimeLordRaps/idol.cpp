import base64
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from idol_server import Runtime, create_app, default_mode_for_source, main
from idol_image import LocalImageGenerator
from idol_mesh_tools import MeshOptions, extract_mesh


def image_bytes():
    stream = io.BytesIO()
    Image.new('RGB', (2, 2), 'red').save(stream, format='PNG')
    return stream.getvalue()


class FakeSession:
    instances = []

    def __init__(self, model, **kwargs):
        self.calls = []
        self.options = kwargs
        self.closed = False
        self.instances.append(self)

    def generate(self, image, output, **kwargs):
        self.calls.append((Path(image).read_bytes(), kwargs))
        Path(output).write_bytes(b'ply\nmock result\n')

    def close(self):
        self.closed = True


class ImageHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.server.received_path = self.path
        self.server.received = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        item = {'b64_json': base64.b64encode(image_bytes()).decode()}
        if getattr(self.server, 'report_model', True):
            item['generation_info'] = {'models': {'diffusion_model': getattr(self.server, 'model_name', 'flux-example-q8.gguf'),
                                                  'vae': 'vae-example.safetensors'}}
        body = json.dumps({'data': [item]}).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class ServerTests(unittest.TestCase):
    def setUp(self):
        FakeSession.instances.clear()
        self.temp = tempfile.TemporaryDirectory()
        self.runtime = Runtime('model.gguf', self.temp.name, session_factory=FakeSession, memory_probe=lambda: None,
                               host_probe=lambda: None)
        self.client = TestClient(create_app(self.runtime, ui=False))

    def tearDown(self):
        self.client.close()
        self.runtime.close()
        self.temp.cleanup()

    def test_image_requests_reuse_session_and_save_distinct_outputs(self):
        self.assertFalse(self.client.get('/health').json()['model_loaded'])
        for seed in (42, 43):
            response = self.client.post('/v1/reconstructions', files={'image': ('x.png', image_bytes(), 'image/png')},
                                        data={'seed': str(seed), 'steps': '5'})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.content, b'ply\nmock result\n')
        self.assertEqual(len(FakeSession.instances), 1)
        self.assertEqual([c[1]['seed'] for c in FakeSession.instances[0].calls], [42, 43])
        self.assertEqual(len(list(Path(self.temp.name).glob('*.ply'))), 2)
        self.assertTrue(self.client.get('/health').json()['model_loaded'])

    def test_preload_reuses_session(self):
        self.runtime.preload()
        self.assertTrue(self.client.get('/health').json()['model_loaded'])
        self.runtime.run(image=image_bytes())
        self.assertEqual(len(FakeSession.instances), 1)

    def test_staged_sessions_are_built_on_the_cpu_and_resident_ones_are_not(self):
        events = []

        class Session:
            def __init__(self, model, **kwargs):
                events.append(('session', kwargs['resident_gpu']))

        @contextlib.contextmanager
        def staged_construction():
            events.append(('enter',))
            try:
                yield
            finally:
                events.append(('exit',))

        modules = {'idol_session': SimpleNamespace(IdolSession=Session),
                   'idol_staging': SimpleNamespace(staged_construction=staged_construction)}
        with patch.dict(sys.modules, modules):
            Runtime('model.gguf', self.temp.name, resident_gpu=False).preload()
            self.assertEqual(events, [('enter',), ('session', False), ('exit',)])
            events.clear()
            Runtime('model.gguf', self.temp.name, resident_gpu=True).preload()
            self.assertEqual(events, [('session', True)])

    def test_encoder_files_are_forwarded_to_session(self):
        self.runtime.dino_gguf = Path('dino.gguf')
        self.runtime.flux_vae_gguf = Path('flux.gguf')
        self.runtime.preload()
        options = FakeSession.instances[0].options
        self.assertEqual(options['dino_gguf'], Path('dino.gguf'))
        self.assertEqual(options['flux_vae_gguf'], Path('flux.gguf'))

    def test_mesh_validation_and_missing_result(self):
        identifier = 'a' * 32
        self.assertEqual(self.client.post(f'/v1/reconstructions/{identifier}/mesh', json={}).status_code, 404)
        self.assertEqual(self.client.post(f'/v1/reconstructions/{identifier}/mesh',
                                         json={'resolution': 10000}).status_code, 422)
        for options in (MeshOptions(sigma=float('nan')), MeshOptions(resolution=176.5), MeshOptions(iso=0)):
            with self.assertRaises(ValueError):
                options.validate()

    def test_mesh_failure_preserves_source_and_leaves_no_mesh(self):
        source = Path(self.temp.name) / ('a' * 32 + '.ply')
        source.write_bytes(b'preserved source')
        with patch('idol_mesh_tools.subprocess.run', return_value=type('Result', (), {
                'returncode': 1, 'stderr': 'extraction failed', 'stdout': ''})()):
            with self.assertRaisesRegex(ValueError, 'extraction failed'):
                extract_mesh(source, self.temp.name, source, MeshOptions())
        self.assertEqual(source.read_bytes(), b'preserved source')
        self.assertFalse(list(Path(self.temp.name).glob('*.obj')))

    @unittest.skipUnless(os.environ.get('IDOL_MESH_EXE'), 'set IDOL_MESH_EXE for native mesh integration')
    def test_native_mesh_and_comparison_api(self):
        import struct
        root = Path(self.temp.name)
        source = root / ('a' * 32 + '.ply')
        points = [(x / 50, y / 50, z / 50, 8.) for x in range(-10, 11)
                  for y in range(-10, 11) for z in range(-10, 11) if x*x + y*y + z*z <= 100]
        header = ('ply\nformat binary_little_endian 1.0\nelement vertex ' + str(len(points)) +
                  '\nproperty float x\nproperty float y\nproperty float z\nproperty float opacity\nend_header\n')
        source.write_bytes(header.encode() + b''.join(struct.pack('<ffff', *p) for p in points))
        self.runtime.mesh_executable = Path(os.environ['IDOL_MESH_EXE'])
        response = self.client.post(f'/v1/reconstructions/{source.stem}/mesh', json={'resolution': 64})
        self.assertEqual(response.status_code, 200, response.text)
        metadata_url = response.headers['x-idol-mesh-metadata']
        info = self.client.get(metadata_url).json()
        self.assertEqual(info['source_ply_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(info['mesh_sha256'], hashlib.sha256(response.content).hexdigest())
        self.assertGreater(info['triangles'], 0)
        mesh_id = metadata_url.split('/')[3]
        comparison = self.client.post(f'/v1/meshes/{mesh_id}/comparisons',
                                     files={'reference': ('reference.obj', response.content, 'text/plain')})
        self.assertEqual(comparison.status_code, 200, comparison.text)
        measurements = comparison.json()['measurements']
        self.assertEqual(measurements['symmetric_mean_sample_distance'], 0)
        self.assertEqual(measurements['surface_f_scores_by_reference_diagonal_fraction']['0.01']['f1'], 1)
        self.assertLess(measurements['sampled_point_to_triangle']['symmetric_mean_distance'], 1e-9)
        uploaded = self.client.post('/v1/meshes', files={'splats': ('saved.ply', source.read_bytes())},
                                    data={'resolution': '64'})
        self.assertEqual(uploaded.status_code, 200, uploaded.text)
        self.assertEqual(uploaded.content, response.content)
        bad_reference = self.client.post(f'/v1/meshes/{mesh_id}/comparisons',
                                        files={'reference': ('bad.obj', b'invalid mesh')})
        self.assertEqual(bad_reference.status_code, 400)

    def test_local_image_generation_releases_existing_idol_session(self):
        self.runtime.preload()
        previous = self.runtime.session
        class FakeLocal:
            label = 'fixture-image-Q8_0.gguf — fixture license'
            license_notice = 'fixture license'
            def generate(self, prompt, seed):
                assert previous.closed
                return image_bytes(), 'fixture-image-Q8_0.gguf', {'text_encoder': 'qwen.gguf'}
        self.runtime.image_generator = FakeLocal()
        response = self.client.post('/v1/text-to-3d', json={'prompt': 'a chair'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(FakeSession.instances), 2)
        metadata = self.client.get(response.headers['x-idol-metadata']).json()
        self.assertEqual(metadata['image_model'], 'fixture-image-Q8_0.gguf')
        self.assertEqual(metadata['image_model_source'], 'local runner invocation')
        self.assertEqual(metadata['image_model_license'], 'fixture license')

    def test_validation_and_busy_request(self):
        invalid = self.client.post('/v1/reconstructions', files={'image': ('x.png', b'not image', 'image/png')})
        self.assertEqual(invalid.status_code, 400)
        invalid = self.client.post('/v1/text-to-3d', json={'prompt': 'a chair'})
        self.assertEqual(invalid.status_code, 400)
        self.runtime.lock.acquire()
        try:
            busy = self.client.post('/v1/reconstructions', files={'image': ('x.png', image_bytes(), 'image/png')})
            self.assertEqual(busy.status_code, 409)
        finally:
            self.runtime.lock.release()

    def test_prompt_uses_configured_image_service(self):
        service = HTTPServer(('127.0.0.1', 0), ImageHandler)
        thread = threading.Thread(target=service.serve_forever, daemon=True)
        thread.start()
        try:
            self.runtime.image_server = f'http://127.0.0.1:{service.server_port}'
            response = self.client.post('/v1/text-to-3d', json={'prompt': 'a red chair', 'steps': 5})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(service.received_path, '/v1/images/generations')
            self.assertEqual(service.received['prompt'], 'a red chair')
            self.assertEqual(FakeSession.instances[0].calls[0][0], image_bytes())
            self.assertEqual(response.headers['x-idol-image-model'], 'flux-example-q8.gguf')
            metadata = self.client.get(response.headers['x-idol-metadata']).json()
            self.assertEqual(metadata['image_model'], 'flux-example-q8.gguf')
            self.assertEqual(metadata['image_model_components']['vae'], 'vae-example.safetensors')
            self.assertEqual(metadata['image_model_source'], 'image service generation response')
            self.assertEqual(metadata['image_sha256'], hashlib.sha256(image_bytes()).hexdigest())
            generated = self.client.get(response.headers['x-idol-input'])
            self.assertEqual(generated.status_code, 200)
            self.assertEqual(metadata['display_image_sha256'], hashlib.sha256(generated.content).hexdigest())
            with Image.open(io.BytesIO(generated.content)) as visible:
                self.assertEqual(visible.convert('RGB').getpixel((0, 0)), (255, 0, 0))
            service.report_model = False
            unidentified = self.client.post('/v1/text-to-3d', json={'prompt': 'another chair'})
            self.assertEqual(unidentified.status_code, 400)
            self.assertIn('did not report', unidentified.json()['detail'])
            self.assertEqual(len(FakeSession.instances[0].calls), 1)
            service.report_model = True
            service.model_name = 'sd-cpp-local'
            alias = self.client.post('/v1/text-to-3d', json={'prompt': 'another chair'})
            self.assertEqual(alias.status_code, 400)
            self.assertIn('did not identify', alias.json()['detail'])
            self.assertEqual(len(FakeSession.instances[0].calls), 1)
        finally:
            service.shutdown()
            service.server_close()
            thread.join()

    def test_prompt_defaults_to_generated_image_preprocessing(self):
        self.assertEqual(default_mode_for_source('Prompt'), 'synthetic')
        self.assertEqual(default_mode_for_source('Image'), 'real')

    def test_failed_session_is_closed_and_recreated(self):
        class BrokenOnce(FakeSession):
            def generate(self, *args, **kwargs):
                if len(self.instances) == 1:
                    raise RuntimeError('inference failed')
                return super().generate(*args, **kwargs)

        self.runtime.session_factory = BrokenOnce
        with self.assertRaisesRegex(RuntimeError, 'inference failed'):
            self.runtime.run(image=image_bytes())
        self.assertTrue(BrokenOnce.instances[0].closed)
        self.assertIsNone(self.runtime.session)
        self.assertTrue(self.runtime.run(image=image_bytes()).exists())

    def test_browser_interface_is_served(self):
        with TestClient(create_app(self.runtime)) as client:
            response = client.get('/')
            self.assertEqual(response.status_code, 200)
            self.assertIn('Idol 3D', response.text)
            self.assertEqual(client.get('/health').status_code, 200)

    def test_cli_downloads_selected_repository_revision(self):
        argv = ['idol-server', '--repo', 'example/idol', '--file', 'model.gguf',
                '--revision', 'abc123', '--output-dir', self.temp.name]
        with patch.object(sys, 'argv', argv), \
             patch('huggingface_hub.hf_hub_download', return_value='cached-model.gguf') as download, \
             patch('idol_server.create_app', return_value=object()), \
             patch('uvicorn.run') as serve:
            main()
        download.assert_called_once_with('example/idol', 'model.gguf', revision='abc123')
        self.assertEqual(serve.call_args.kwargs['port'], 8127)

    def run_cli(self, *options):
        model = Path(self.temp.name) / 'model.gguf'
        model.write_bytes(b'x')
        argv = ['idol-server', '--model', str(model), '--output-dir', self.temp.name,
                '--image-server', 'http://127.0.0.1:1', *options]
        with patch.object(sys, 'argv', argv),              patch('idol_server.Runtime') as runtime,              patch('idol_server.create_app', return_value=object()),              patch('uvicorn.run'):
            main()
        return runtime.call_args.kwargs

    def test_cli_memory_modes_map_to_where_the_models_wait(self):
        expected = {'auto': (None, None), 'resident': (True, None), 'staged': (False, False), 'lean': (False, True)}
        for mode, (resident, lean) in expected.items():
            options = self.run_cli('--memory-mode', mode)
            self.assertEqual((options['resident_gpu'], options['lean']), (resident, lean), mode)
        self.assertEqual((self.run_cli('--resident-gpu')['resident_gpu']), True)

    def test_cli_passes_the_lean_directory_and_checks_it(self):
        self.assertEqual(self.run_cli('--memory-mode', 'lean', '--lean-directory', self.temp.name)['lean_directory'],
                         Path(self.temp.name))
        self.assertIsNone(self.run_cli()['lean_directory'])
        with self.assertRaises(SystemExit), patch('sys.stderr', io.StringIO()):
            self.run_cli('--lean-directory', str(Path(self.temp.name) / 'missing'))

    def test_cli_refuses_lean_with_packed_or_fused_models(self):
        for option in ('--packed-denoiser', '--fused-kernels'):
            with self.assertRaises(SystemExit), patch('sys.stderr', io.StringIO()):
                self.run_cli('--memory-mode', 'lean', option)
            self.run_cli('--memory-mode', 'staged', option)  # the same options are fine in memory

    def test_local_runner_uses_configured_files_and_releases_process(self):
        root = Path(self.temp.name)
        names = ('sd-cli.exe', 'fixture-image-Q8_0.gguf', 'qwen.safetensors', 'vae.safetensors')
        for name in names:
            (root / name).write_bytes(b'fixture')
        config = {'version': 1, 'engine': 'stable-diffusion.cpp', 'executable': str(root / names[0]),
                  'diffusion_model': str(root / names[1]), 'text_encoder': str(root / names[2]),
                  'vae': str(root / names[3]), 'license_notice': 'fixture license'}
        path = root / 'config.json'
        path.write_text(json.dumps(config))
        generator = LocalImageGenerator(path, memory_probe=lambda: None)
        self.assertIn(names[1], generator.label)
        self.assertIn(names[2], generator.label)
        def fake_run(args, **kwargs):
            Path(args[args.index('--output') + 1]).write_bytes(image_bytes())
            self.assertEqual(args[args.index('--diffusion-model') + 1], str(root / names[1]))
            self.assertEqual(args[args.index('--llm') + 1], str(root / names[2]))
            self.assertEqual(args[args.index('--vae') + 1], str(root / names[3]))
            self.assertEqual(args[args.index('--seed') + 1], '42')
            self.assertEqual(float(args[args.index('--max-vram') + 1]), 16.0)
            return type('Result', (), {'returncode': 0, 'stdout': b''})()
        with patch('idol_image.subprocess.run', side_effect=fake_run):
            image, model, components = generator.generate('a chair', 42)
        self.assertEqual(image, image_bytes())
        self.assertEqual(model, names[1])
        self.assertEqual(components['text_encoder'], names[2])
        config.pop('vae')
        path.write_text(json.dumps(config))
        generator = LocalImageGenerator(path, memory_probe=lambda: None)
        def fake_iris(args, **kwargs):
            self.assertNotIn('--vae', args)
            Path(args[args.index('--output') + 1]).write_bytes(image_bytes())
            return type('Result', (), {'returncode': 0, 'stdout': b''})()
        with patch('idol_image.subprocess.run', side_effect=fake_iris):
            _, _, components = generator.generate('a chair', 42)
        self.assertNotIn('vae', components)
        config['max_vram_gib'] = 2
        path.write_text(json.dumps(config))
        generator = LocalImageGenerator(path)
        def fake_budget(args, **kwargs):
            self.assertEqual(float(args[args.index('--max-vram') + 1]), 2.0)
            Path(args[args.index('--output') + 1]).write_bytes(image_bytes())
            return type('Result', (), {'returncode': 0, 'stdout': b''})()
        with patch('idol_image.subprocess.run', side_effect=fake_budget):
            generator.generate('a chair', 42)


class LeanModeTests(unittest.TestCase):
    """Staged sessions keep the waiting models on disk when the computer lacks the memory to hold them."""

    def setUp(self):
        FakeSession.instances.clear()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def runtime(self, host, disk=10**6, gpu=3500, **kwargs):
        kwargs.setdefault('session_factory', FakeSession)
        runtime = Runtime('model.gguf', self.temp.name, memory_probe=lambda: gpu,
                          host_probe=lambda: host, disk_probe=lambda: disk, **kwargs)
        self.addCleanup(runtime.close)
        return runtime

    def test_little_free_memory_with_room_on_disk_chooses_lean(self):
        from idol_memory import HOST_HEADROOM_MIB, IDOL_STAGED_HOST_MIB
        runtime = self.runtime(IDOL_STAGED_HOST_MIB + HOST_HEADROOM_MIB - 1)
        runtime.preload()
        self.assertEqual(runtime.memory_mode, 'lean')
        self.assertIs(FakeSession.instances[-1].options['resident_gpu'], False)

    def test_enough_unknown_or_unmeasurable_memory_keeps_the_models_in_memory(self):
        from idol_memory import HOST_HEADROOM_MIB, IDOL_STAGED_HOST_MIB
        for host in (IDOL_STAGED_HOST_MIB + HOST_HEADROOM_MIB, 10**6, None):
            runtime = self.runtime(host)
            runtime.preload()
            self.assertEqual(runtime.memory_mode, 'staged', host)

    def test_too_little_disk_space_keeps_the_models_in_memory(self):
        from idol_memory import DISK_HEADROOM_MIB, LEAN_CACHE_MIB
        runtime = self.runtime(1000, disk=LEAN_CACHE_MIB + DISK_HEADROOM_MIB - 1)
        runtime.preload()
        self.assertEqual(runtime.memory_mode, 'staged')
        runtime = self.runtime(1000, disk=LEAN_CACHE_MIB + DISK_HEADROOM_MIB)
        runtime.preload()
        self.assertEqual(runtime.memory_mode, 'lean')
        runtime = self.runtime(1000, disk=None)  # unmeasurable: the temporary directory is normally roomy enough
        runtime.preload()
        self.assertEqual(runtime.memory_mode, 'lean')

    def test_a_stated_choice_is_used_without_measuring(self):
        calls = []
        runtime = self.runtime(10**6, lean=True)
        runtime.host_probe = lambda: calls.append('host')
        runtime.preload()
        self.assertEqual(runtime.memory_mode, 'lean')
        runtime = self.runtime(1, lean=False)
        runtime.host_probe = lambda: calls.append('host')
        runtime.preload()
        self.assertEqual(runtime.memory_mode, 'staged')
        self.assertEqual(calls, [])

    def test_resident_sessions_have_no_waiting_models_to_keep_on_disk(self):
        runtime = self.runtime(1, gpu=10**6)
        runtime.host_probe = lambda: self.fail('memory of the computer does not matter to a resident session')
        runtime.preload()
        self.assertEqual(runtime.memory_mode, 'resident')
        runtime = self.runtime(1, resident_gpu=True, lean=True)
        runtime.preload()
        self.assertEqual(runtime.memory_mode, 'resident')

    def test_packed_and_fused_models_cannot_be_lean(self):
        for option in ('packed_denoiser', 'fused_kernels'):
            runtime = self.runtime(1, **{option: True})
            runtime.preload()
            self.assertEqual(runtime.memory_mode, 'staged', 'chosen automatically, they stay in memory')
            runtime = self.runtime(1, lean=True, **{option: True})
            with self.assertRaisesRegex(ValueError, 'does not support'):
                runtime.preload()
            self.assertIsNone(runtime.session)

    def test_health_reports_the_lean_mode(self):
        runtime = self.runtime(1)
        client = TestClient(create_app(runtime, ui=False))
        self.addCleanup(client.close)
        runtime.preload()
        self.assertEqual(client.get('/health').json()['memory_mode'], 'lean')

    def test_a_lean_session_is_built_in_lean_construction_and_its_files_removed_after_it(self):
        events = []

        class Cache:
            def close(self):
                events.append('cache closed')

        class Session:
            def __init__(self, model, **kwargs):
                events.append('session built')

            def close(self):
                events.append('session closed')

        @contextlib.contextmanager
        def lean_construction(directory=None):
            events.append(('enter', directory))
            yield Cache()
            events.append('exit')

        @contextlib.contextmanager
        def staged_construction():
            self.fail('a lean session is not built by staged construction as well')
            yield

        modules = {'idol_session': SimpleNamespace(IdolSession=Session),
                   'idol_lowram': SimpleNamespace(lean_construction=lean_construction),
                   'idol_staging': SimpleNamespace(staged_construction=staged_construction)}
        with patch.dict(sys.modules, modules):
            runtime = Runtime('model.gguf', self.temp.name, lean=True, lean_directory='cache-dir',
                              memory_probe=lambda: 3500)
            runtime.preload()
            self.assertEqual(events, [('enter', 'cache-dir'), 'session built', 'exit'])
            events.clear()
            runtime.close()
            self.assertEqual(events, ['session closed', 'cache closed'], 'the models are read until the session ends')
            self.assertIsNone(runtime.session)
            self.assertIsNone(runtime.lean_cache)
            self.assertIsNone(runtime.memory_mode)
            runtime.close()  # closing twice is harmless
            self.assertEqual(events, ['session closed', 'cache closed'])

    def test_a_failed_generation_removes_the_files_too(self):
        class Cache:
            closed = False

            def close(self):
                self.closed = True

        class Failing(FakeSession):
            def generate(self, image, output, **kwargs):
                raise RuntimeError('inference failed')

        runtime = self.runtime(1, session_factory=Failing)
        runtime.preload()
        cache = runtime.lean_cache = Cache()
        failed = runtime.session
        with self.assertRaisesRegex(RuntimeError, 'inference failed'):
            runtime.run(image=image_bytes())
        self.assertTrue(failed.closed)
        self.assertTrue(cache.closed, 'files of a failed session are not left behind')
        self.assertIsNone(runtime.session)
        self.assertIsNone(runtime.lean_cache)

    def test_the_files_are_removed_even_when_the_session_fails_to_close(self):
        class Cache:
            closed = False

            def close(self):
                self.closed = True

        class Stuck(FakeSession):
            def close(self):
                raise RuntimeError('cannot close')

        runtime = self.runtime(1, session_factory=Stuck)
        runtime.preload()
        cache = runtime.lean_cache = Cache()
        with self.assertRaisesRegex(RuntimeError, 'cannot close'):
            runtime.close()
        self.assertTrue(cache.closed)
        self.assertIsNone(runtime.session)

    def test_releasing_the_session_for_the_image_generator_removes_the_files_too(self):
        class Cache:
            closed = False

            def close(self):
                self.closed = True

        cache = Cache()
        runtime = self.runtime(1)
        runtime.preload()
        runtime.lean_cache = cache
        first = runtime.session
        runtime._release_session_for_image_generator()
        self.assertTrue(first.closed)
        self.assertTrue(cache.closed)
        self.assertIsNone(runtime.lean_cache)


class AutomaticMemoryTests(unittest.TestCase):
    """The memory mode and the image generator's budget follow the free graphics memory."""

    def setUp(self):
        FakeSession.instances.clear()
        self.temp = tempfile.TemporaryDirectory()
        self.probed = []

    def tearDown(self):
        self.temp.cleanup()

    def runtime(self, free, **kwargs):
        def probe():
            self.probed.append(free)
            return free
        kwargs.setdefault('session_factory', FakeSession)
        kwargs.setdefault('host_probe', lambda: None)
        return Runtime('model.gguf', self.temp.name, memory_probe=probe, **kwargs)

    def options(self, runtime):
        runtime.preload()
        options = FakeSession.instances[-1].options
        return options['resident_gpu'], options['batch_guidance'], options['cudnn_attention']

    def test_plenty_of_memory_keeps_everything_resident_and_fast(self):
        from idol_memory import HEADROOM_MIB, IDOL_RESIDENT_MIB
        runtime = self.runtime(IDOL_RESIDENT_MIB + HEADROOM_MIB)
        self.assertEqual(self.options(runtime), (True, True, True))
        self.assertEqual(runtime.memory_mode, 'resident')
        runtime.close()

    def test_short_or_unknown_memory_stages_the_models(self):
        from idol_memory import HEADROOM_MIB, IDOL_RESIDENT_MIB
        for free in (IDOL_RESIDENT_MIB + HEADROOM_MIB - 1, 3500, None):
            runtime = self.runtime(free)
            self.assertEqual(self.options(runtime), (False, False, False), free)
            self.assertEqual(runtime.memory_mode, 'staged')
            runtime.close()

    def test_a_stated_mode_is_used_as_given_without_measuring(self):
        for resident in (True, False):
            runtime = self.runtime(10**6, resident_gpu=resident)
            self.assertEqual(self.options(runtime), (resident, False, False))
            runtime.close()
        self.assertEqual(self.probed, [])

    def test_a_stated_speed_option_overrides_the_automatic_one(self):
        runtime = self.runtime(10**6, batch_guidance=False)
        self.assertEqual(self.options(runtime), (True, False, True))
        runtime.close()
        runtime = self.runtime(100, cudnn_attention=True)
        self.assertEqual(self.options(runtime), (False, False, True))
        runtime.close()

    def test_health_reports_the_mode(self):
        runtime = self.runtime(10**6)
        client = TestClient(create_app(runtime, ui=False))
        self.assertIsNone(client.get('/health').json()['memory_mode'])
        runtime.preload()
        self.assertEqual(client.get('/health').json()['memory_mode'], 'resident')
        client.close()
        runtime.close()

    def prompt_runtime(self, free, **kwargs):
        class Generator:
            license_notice = ''
            label = 'generator'
            prompts = []

            def generate(self, prompt, seed):
                self.prompts.append(prompt)
                return image_bytes(), 'iris.gguf', {'diffusion_model': 'iris.gguf'}

        return self.runtime(free, image_generator=Generator(), **kwargs)

    def test_the_session_stays_loaded_for_a_prompt_when_the_generator_fits_beside_it(self):
        from idol_memory import HEADROOM_MIB, IRIS_FULL_MIB
        runtime = self.prompt_runtime(IRIS_FULL_MIB + HEADROOM_MIB)
        runtime.preload()
        first = runtime.session
        runtime.run(prompt='a chair')
        self.assertIs(runtime.session, first)
        self.assertFalse(first.closed)
        self.assertEqual(len(FakeSession.instances), 1)
        runtime.close()

    def test_the_session_is_released_for_a_prompt_when_memory_is_short(self):
        from idol_memory import HEADROOM_MIB, IRIS_FULL_MIB
        runtime = self.prompt_runtime(IRIS_FULL_MIB + HEADROOM_MIB - 1)
        runtime.preload()
        first = runtime.session
        runtime.run(prompt='a chair')
        self.assertTrue(first.closed)
        self.assertEqual(len(FakeSession.instances), 2)
        self.assertIsNot(runtime.session, first)
        runtime.close()

    def test_a_stated_budget_always_releases_the_session(self):
        runtime = self.prompt_runtime(10**6, vram_budget_gib=16)
        runtime.preload()
        first = runtime.session
        runtime.run(prompt='a chair')
        self.assertTrue(first.closed)
        runtime.close()

    def test_image_budget_follows_the_free_memory_unless_configured(self):
        import idol_memory
        config = {'version': 1, 'engine': 'stable-diffusion.cpp'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('sd-cli', 'iris.gguf', 'qwen.gguf'):
                (root / name).write_bytes(b'x')
            config.update(executable=str(root / 'sd-cli'), diffusion_model=str(root / 'iris.gguf'),
                          text_encoder=str(root / 'qwen.gguf'))
            path = root / 'config.json'

            def budget(extra, free):
                path.write_text(json.dumps({**config, **extra}))
                generator = LocalImageGenerator(path, memory_probe=lambda: free)
                seen = []

                def fake_run(args, **kwargs):
                    seen.append(float(args[args.index('--max-vram') + 1]))
                    Path(args[args.index('--output') + 1]).write_bytes(image_bytes())
                    return type('Result', (), {'returncode': 0, 'stdout': b''})()
                with patch('idol_image.subprocess.run', side_effect=fake_run):
                    generator.generate('a chair', 1)
                self.assertEqual(generator.last_budget_gib, seen[0])
                return seen[0]

            full = idol_memory.IRIS_FULL_MIB + idol_memory.HEADROOM_MIB
            self.assertEqual(budget({}, full), 16.0)
            self.assertEqual(budget({'max_vram_gib': 'auto'}, full), 16.0)
            self.assertEqual(budget({}, None), 16.0)
            small = budget({}, 4096)
            self.assertEqual(small, idol_memory.iris_budget_gib(4096))
            self.assertLess(small, 16.0)
            self.assertEqual(budget({'max_vram_gib': 3}, 10**6), 3.0, 'a configured number is never overridden')
            for bad in (0, 100, 'low', float('nan')):
                path.write_text(json.dumps({**config, 'max_vram_gib': bad}))
                with self.assertRaises(ValueError):
                    LocalImageGenerator(path)

    def test_image_weights_are_memory_mapped_when_system_memory_is_short_unless_configured(self):
        import idol_memory
        config = {'version': 1, 'engine': 'stable-diffusion.cpp'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('sd-cli', 'iris.gguf', 'qwen.gguf'):
                (root / name).write_bytes(b'x')
            config.update(executable=str(root / 'sd-cli'), diffusion_model=str(root / 'iris.gguf'),
                          text_encoder=str(root / 'qwen.gguf'), max_vram_gib=3)
            path = root / 'config.json'

            def mapped(extra, host_free):
                path.write_text(json.dumps({**config, **extra}))
                generator = LocalImageGenerator(path, memory_probe=lambda: 10**6, host_probe=lambda: host_free)
                seen = []

                def fake_run(args, **kwargs):
                    seen.append(list(args))
                    Path(args[args.index('--output') + 1]).write_bytes(image_bytes())
                    return type('Result', (), {'returncode': 0, 'stdout': b''})()
                with patch('idol_image.subprocess.run', side_effect=fake_run):
                    generator.generate('a chair', 1)
                self.assertEqual(generator.last_mmap, '--mmap' in seen[0])
                self.assertEqual(seen[0].count('--mmap'), int(generator.last_mmap))
                return generator.last_mmap

            need = idol_memory.iris_host_mib(3) + idol_memory.HOST_HEADROOM_MIB
            self.assertFalse(mapped({}, need))
            self.assertTrue(mapped({}, need - 1))
            self.assertTrue(mapped({'mmap': 'auto'}, need - 1))
            self.assertFalse(mapped({}, None), 'unmeasurable memory keeps the normal loading')
            self.assertTrue(mapped({'mmap': True}, 10**6), 'a configured choice is never overridden')
            self.assertFalse(mapped({'mmap': False}, 0), 'a configured choice is never overridden')
            for bad in ('yes', 1, 0, None, 'on'):
                path.write_text(json.dumps({**config, 'mmap': bad}))
                with self.assertRaises(ValueError):
                    LocalImageGenerator(path)


if __name__ == '__main__':
    unittest.main()

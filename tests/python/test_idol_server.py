import base64
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import unittest
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
        self.runtime = Runtime('model.gguf', self.temp.name, session_factory=FakeSession)
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
        generator = LocalImageGenerator(path)
        self.assertIn(names[1], generator.label)
        self.assertIn(names[2], generator.label)
        def fake_run(args, **kwargs):
            Path(args[args.index('--output') + 1]).write_bytes(image_bytes())
            self.assertEqual(args[args.index('--diffusion-model') + 1], str(root / names[1]))
            self.assertEqual(args[args.index('--llm') + 1], str(root / names[2]))
            self.assertEqual(args[args.index('--vae') + 1], str(root / names[3]))
            self.assertEqual(args[args.index('--seed') + 1], '42')
            return type('Result', (), {'returncode': 0, 'stdout': b''})()
        with patch('idol_image.subprocess.run', side_effect=fake_run):
            image, model, components = generator.generate('a chair', 42)
        self.assertEqual(image, image_bytes())
        self.assertEqual(model, names[1])
        self.assertEqual(components['text_encoder'], names[2])
        config.pop('vae')
        path.write_text(json.dumps(config))
        generator = LocalImageGenerator(path)
        def fake_iris(args, **kwargs):
            self.assertNotIn('--vae', args)
            Path(args[args.index('--output') + 1]).write_bytes(image_bytes())
            return type('Result', (), {'returncode': 0, 'stdout': b''})()
        with patch('idol_image.subprocess.run', side_effect=fake_iris):
            _, _, components = generator.generate('a chair', 42)
        self.assertNotIn('vae', components)


if __name__ == '__main__':
    unittest.main()

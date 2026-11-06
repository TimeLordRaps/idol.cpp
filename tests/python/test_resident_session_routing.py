"""Exercise session staging decisions without allocating model or GPU memory."""
import contextlib
import importlib.util
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class ResidentRoutingTests(unittest.TestCase):
    def test_generation_and_close_respect_resident_mode(self):
        moves, stages = [], []

        class Module:
            rep_config = {'num_gaussians': 32}
            def __init__(self, name):
                self.name = name
            def to(self, device):
                moves.append((self.name, device))
                return self
            def __call__(self, *args):
                return object()
            def to_representation(self, *args):
                return [SimpleNamespace(save_ply=lambda path: Path(path).write_bytes(b'ply\n'))]

        def image_to_latent(*args):
            stages.append((args[-2], args[-1]))
            return object(), None, {}

        torch = ModuleType('torch')
        torch.no_grad = lambda: (lambda function: function)
        torch.from_numpy = lambda value: SimpleNamespace(to=lambda device: object())
        torch.manual_seed = lambda seed: None
        torch.cuda = SimpleNamespace(empty_cache=lambda: None)
        numpy = ModuleType('numpy')
        numpy.float32 = object()
        numpy.ascontiguousarray = lambda value, dtype: value
        pillow = ModuleType('PIL')
        pillow.Image = SimpleNamespace(open=lambda path: contextlib.nullcontext(object()))
        run = ModuleType('idol_generate')
        run.image_to_latent = image_to_latent
        run.checkpoint = lambda: None
        run.hold_model = lambda model, name: model
        run.validate_gaussian_cloud = lambda cloud, count: None
        gguf = ModuleType('idol_gguf')
        gguf.IdolGGUF = object
        models = ModuleType('idol.models')
        models.OctreeProbabilityFixedlenDecoder = SimpleNamespace(sample=lambda *args, **kwargs: object())
        idol = ModuleType('idol')
        idol.models = models
        replacements = {'torch': torch, 'numpy': numpy, 'PIL': pillow, 'idol_generate': run,
                        'idol_gguf': gguf, 'idol': idol, 'idol.models': models}
        path = Path(__file__).resolve().parents[2] / 'tools' / 'idol_session.py'
        with patch.dict(sys.modules, replacements), tempfile.TemporaryDirectory() as directory:
            spec = importlib.util.spec_from_file_location('isolated_idol_session', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            image = Path(directory) / 'image.png'
            image.write_bytes(b'fixture')
            for resident in (False, True):
                moves.clear()
                session = module.IdolSession.__new__(module.IdolSession)
                session._lock = threading.Lock()
                session._closed = session._failed = session._decoders_registered = False
                session.resident_gpu = resident
                session.batch_guidance = True
                session.device = 'cuda'
                session.pipeline = SimpleNamespace(denoiser=Module('denoiser'),
                    background_remover=Module('background'),
                    image_encoder=SimpleNamespace(dinov3=Module('dino'), flux2_vae=Module('vae')))
                session.octree, session.gaussian = Module('octree'), Module('gaussian')
                session.model = SimpleNamespace(config={'max_voxel_level': 3})
                output = Path(directory) / f'result-{resident}.ply'
                session.generate(image, output, gaussians=32)
                self.assertEqual(output.read_bytes(), b'ply\n')
                self.assertEqual(stages[-1], (not resident, True))
                self.assertIn(('octree', 'cuda'), moves)
                self.assertIn(('gaussian', 'cuda'), moves)
                self.assertEqual(('denoiser', 'cpu') in moves, not resident)
                self.assertEqual(('octree', 'cpu') in moves, not resident)
                moves.clear()
                session.close()
                self.assertEqual(bool(moves), not resident)
                self.assertIsNone(session.pipeline)


if __name__ == '__main__':
    unittest.main()

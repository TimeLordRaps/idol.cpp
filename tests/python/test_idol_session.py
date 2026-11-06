"""Check reusable-session boundaries without requiring a GPU or model weights."""
import importlib.util
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if any(importlib.util.find_spec(name) is None for name in ('numpy', 'torch', 'gguf', 'PIL')):
    print('skipped: needs numpy, torch, gguf and pillow')
    sys.exit(77)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from idol_session import IdolSession, _PROCESS_INFERENCE_LOCK
import idol_generate as run


def rejected(call, exception):
    try:
        call()
    except exception:
        return
    raise AssertionError('expected ' + exception.__name__)


run.validate_encoder_choices(True, None, None)
run.validate_encoder_choices(False, Path('dino.gguf'), Path('vae.gguf'))
rejected(lambda: run.validate_encoder_choices(True, Path('dino.gguf'), None), ValueError)
rejected(lambda: run.validate_encoder_choices(True, None, Path('vae.gguf')), ValueError)


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    image = root / 'input.png'
    image.write_bytes(b'input bytes')
    existing = root / 'existing.ply'
    existing.write_bytes(b'preserve this output')
    session = IdolSession.__new__(IdolSession)
    session._lock = threading.Lock()
    session._closed = False
    session._failed = False
    session.gaussian = SimpleNamespace(rep_config={'num_gaussians': 32}, to=lambda device: None)
    session.pipeline = session.octree = session.model = None
    rejected(lambda: session.generate(image, existing), ValueError)
    assert existing.read_bytes() == b'preserve this output'
    rejected(lambda: session.generate(image, root / 'new.ply', seed=-1), ValueError)
    rejected(lambda: session.generate(image, root / 'new.ply', steps=0), ValueError)
    rejected(lambda: session.generate(image, root / 'new.ply', gaussians=33), ValueError)
    rejected(lambda: session.generate(image, root / 'new.ply', mode='unknown'), ValueError)
    assert not session._failed
    assert session._lock.acquire(blocking=False)
    rejected(lambda: session.generate(image, root / 'new.ply'), RuntimeError)
    rejected(session.close, RuntimeError)
    session._lock.release()
    assert _PROCESS_INFERENCE_LOCK.acquire(blocking=False)
    rejected(lambda: session.generate(image, root / 'new.ply'), RuntimeError)
    _PROCESS_INFERENCE_LOCK.release()
    assert session._lock.acquire(blocking=False)
    session._lock.release()
    session.close()
    rejected(lambda: session.generate(image, root / 'new.ply'), RuntimeError)

    moved = []
    class Module:
        def __init__(self, name):
            self.name = name
        def to(self, device):
            moved.append((self.name, device))
            return self
    session = IdolSession.__new__(IdolSession)
    session._lock = threading.Lock()
    session._closed = False
    session._failed = False
    session.pipeline = SimpleNamespace(denoiser=Module('denoiser'), background_remover=Module('background'),
        image_encoder=SimpleNamespace(dinov3=Module('dino'), flux2_vae=Module('vae')))
    session.octree, session.gaussian, session.model = Module('octree'), Module('gaussian'), object()
    session.close()
    assert moved == [(name, 'cpu') for name in ('denoiser', 'background', 'dino', 'vae', 'octree', 'gaussian')]
    assert session.pipeline is None and session.octree is None and session.gaussian is None and session.model is None

# Both public inference entry points share this loader.  The fused mode must
# start from empty modules and never call the dense-first loading path.
if importlib.util.find_spec('accelerate') is not None:
    import torch
    class TinyDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
    class FakeWeights:
        config = {'decoder': {'args': {}}, 'decoder_gs': {'args': {}}}
        def __init__(self):
            self.loaded = []
        def load_module_packed(self, name, module):
            assert all(p.is_meta for p in module.parameters())
            self.loaded.append(name)
            return module.to_empty(device='cpu')
        def load_module(self, name, module):
            raise AssertionError('fused loading must not materialize dense initial weights')
    fake_models = SimpleNamespace(OctreeProbabilityFixedlenDecoder=TinyDecoder,
                                  GaussianFixedlenDecoder=TinyDecoder)
    weights = FakeWeights()
    with patch.dict(sys.modules, {'idol': SimpleNamespace(models=fake_models), 'idol.models': fake_models}), \
            patch.object(run, 'checkpoint'):
        octree, gaussian = run.load_gpu_decoders(weights, True)
    assert weights.loaded == ['octree', 'gaussian']
    assert not octree.training and not gaussian.training
    assert not any(p.requires_grad for module in (octree, gaussian) for p in module.parameters())
print('reusable-session boundary checks passed')

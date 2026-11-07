"""Check staged pipeline construction and sequential image encoding without models or a GPU."""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

if importlib.util.find_spec('torch') is None:
    print('skipped: needs torch')
    sys.exit(77)
import torch

TOOLS = Path(__file__).resolve().parents[2] / 'tools'


class FakeTensorDict(dict):
    def __init__(self, values, batch_size):
        super().__init__(values)
        self.batch_size = batch_size


def load_module(original):
    run = ModuleType('idol_generate')
    run.make_pipeline = original
    replacements = {'idol_generate': run, 'tensordict': SimpleNamespace(TensorDict=FakeTensorDict)}
    with patch.dict(sys.modules, replacements):
        spec = importlib.util.spec_from_file_location('isolated_idol_staging', TOOLS / 'idol_staging.py')
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module, run


class Recorder:
    def __init__(self, events, name):
        self.events, self.name = events, name

    def to(self, device):
        self.events.append((self.name + '.to', str(device)))
        return self

    def half(self):
        self.events.append((self.name + '.half',))
        return self


class StagingTests(unittest.TestCase):
    def build(self, **options):
        events, calls = [], []

        def original(*args):
            calls.append(args)
            return SimpleNamespace(
                device=torch.device('cpu'), denoiser=Recorder(events, 'denoiser'),
                background_remover=Recorder(events, 'background'),
                image_encoder=SimpleNamespace(device=torch.device('cpu'), dinov3=Recorder(events, 'dinov3'),
                                              flux2_vae=Recorder(events, 'vae')))

        module, run = load_module(original)
        packed = SimpleNamespace(enable_fused_packed=lambda model: options.get('fused_count', 1))
        with patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.dict(sys.modules, {'idol_packed': packed}):
            pipeline = module.make_staged_pipeline(
                'model', 'cuda', None, None, packed_denoiser=options.get('packed', False),
                fused_kernels=options.get('fused', False), cudnn_attention=options.get('cudnn', False))
        return module, pipeline, calls, events

    def test_models_are_built_on_the_cpu_then_retargeted(self):
        module, pipeline, calls, events = self.build()
        # The wrapped builder is always asked for the CPU, never for the CUDA device.
        self.assertEqual(calls, [('model', 'cpu', None, None, False, None, None, False, False)])
        self.assertEqual(pipeline.device, torch.device('cuda'))
        self.assertIsInstance(pipeline.image_encoder, module.SequentialImageEncoder)
        self.assertEqual(pipeline.image_encoder.device, torch.device('cuda'))
        self.assertIn(('background.half',), events)
        self.assertFalse([event for event in events if event[0].endswith('.to')], 'nothing is moved at build time')
        self.assertFalse(pipeline.optimize)

    def test_options_reach_the_pipeline(self):
        _, pipeline, calls, _ = self.build(cudnn=True)
        self.assertTrue(pipeline.optimize)
        _, _, calls, _ = self.build(fused=True)
        self.assertTrue(calls[0][4], 'fused kernels use the packed loader')
        _, _, calls, _ = self.build(packed=True)
        self.assertTrue(calls[0][4])
        with self.assertRaisesRegex(ValueError, 'fused kernels require'):
            self.build(fused=True, fused_count=0)

    def test_requires_cuda(self):
        module, _ = load_module(lambda *args: None)
        with patch.object(torch.cuda, 'is_available', return_value=False), \
                self.assertRaisesRegex(ValueError, 'CUDA'):
            module.make_staged_pipeline('model', 'cuda', None, None)
        with patch.object(torch.cuda, 'is_available', return_value=True), \
                self.assertRaisesRegex(ValueError, 'CUDA'):
            module.make_staged_pipeline('model', 'cpu', None, None)

    def test_encoders_run_one_at_a_time_in_the_original_order(self):
        events = []

        class Encoder:
            device = torch.device('cpu')
            dinov3 = Recorder(events, 'dinov3')
            flux2_vae = Recorder(events, 'vae')

            def _dinov3_tokens(self, images):
                events.append(('dinov3.run',))
                return torch.zeros(1, 2)

            def _flux2_tokens(self, images):
                events.append(('vae.run',))
                return torch.ones(1, 3)

        module, _ = load_module(lambda *args: None)
        wrapped = module.SequentialImageEncoder(Encoder())
        with patch.dict(sys.modules, {'tensordict': SimpleNamespace(TensorDict=FakeTensorDict)}):
            result = wrapped(torch.zeros(1, 3, 4, 4))
        self.assertEqual(events, [('dinov3.to', 'cpu'), ('dinov3.run',), ('dinov3.to', 'cpu'),
                                  ('vae.to', 'cpu'), ('vae.run',)])
        self.assertEqual(set(result), {'feature1', 'feature2'})
        self.assertEqual(result['feature1'].shape, (1, 2))
        self.assertEqual(result['feature2'].shape, (1, 3))
        self.assertIs(wrapped.dinov3, Encoder.dinov3, 'other attributes are the wrapped encoder\'s')

    def test_encoder_to_device_ordering(self):
        """The first encoder leaves the device before the second one is moved onto it."""
        events = []
        device = torch.device('meta')  # any non-CPU target; the fake modules only record the request

        class Encoder:
            dinov3 = Recorder(events, 'dinov3')
            flux2_vae = Recorder(events, 'vae')

            def __init__(self):
                self.device = device

            def _dinov3_tokens(self, images):
                return torch.zeros(1, 2)

            def _flux2_tokens(self, images):
                return torch.zeros(1, 2)

        module, _ = load_module(lambda *args: None)
        with patch.object(torch.Tensor, 'to', lambda self, *args, **kwargs: self),                 patch.dict(sys.modules, {'tensordict': SimpleNamespace(TensorDict=FakeTensorDict)}):
            module.SequentialImageEncoder(Encoder())(torch.zeros(1, 3, 4, 4))
        order = [event for event in events if event[0].endswith('.to')]
        self.assertEqual(order, [('dinov3.to', 'meta'), ('dinov3.to', 'cpu'), ('vae.to', 'meta')])

    def test_construction_context_restores_the_builder(self):
        module, run = load_module(lambda *args: None)
        original = run.make_pipeline
        with module.staged_construction():
            self.assertIs(run.make_pipeline, module.make_staged_pipeline)
        self.assertIs(run.make_pipeline, original)
        with self.assertRaises(RuntimeError), module.staged_construction():
            raise RuntimeError('construction failed')
        self.assertIs(run.make_pipeline, original)


if __name__ == '__main__':
    unittest.main()

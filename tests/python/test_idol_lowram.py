"""Check that models parked on disk come back exactly, and that construction parks them, without a GPU."""
import contextlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

if importlib.util.find_spec('torch') is None or importlib.util.find_spec('safetensors') is None:
    print('skipped: needs torch and safetensors')
    sys.exit(77)
import torch

TOOLS = Path(__file__).resolve().parents[2] / 'tools'


def load_module():
    run = ModuleType('idol_generate')
    run.hold_model = lambda module, name: module
    run.load_gpu_decoders = lambda model, fused_kernels=False: (torch.nn.Linear(2, 2), torch.nn.Linear(3, 3))
    staging = ModuleType('idol_staging')
    staging.events = []

    @contextlib.contextmanager
    def staged_construction():
        staging.events.append('enter')
        try:
            yield
        finally:
            staging.events.append('exit')

    staging.staged_construction = staged_construction
    with patch.dict(sys.modules, {'idol_generate': run, 'idol_staging': staging}):
        spec = importlib.util.spec_from_file_location('isolated_idol_lowram', TOOLS / 'idol_lowram.py')
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module, run, staging


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(4, 3)
        self.norm = torch.nn.LayerNorm(3)
        self.second = torch.nn.Linear(3, 4, bias=False)
        self.register_buffer('table', torch.arange(6, dtype=torch.float32).view(2, 3), persistent=False)
        self.register_buffer('steps', torch.tensor([7], dtype=torch.int64))

    def forward(self, values):
        return self.second(self.norm(self.first(values))) + self.table.sum()


def snapshot(module):
    return {name: tensor.detach().clone() for name, tensor in
            list(module.named_parameters()) + list(module.named_buffers())}


class DiskStageTests(unittest.TestCase):
    def setUp(self):
        self.module, self.generate, self.staging = load_module()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        torch.manual_seed(0)

    def stage(self, model=None, name='tiny'):
        model = model or Tiny()
        return model, self.module.DiskStage(model, Path(self.directory.name) / f'{name}.safetensors', name)

    def test_parking_frees_the_tensors_and_loading_restores_them_exactly(self):
        model = Tiny()
        before, expected = snapshot(model), model(torch.ones(1, 4))
        model, stage = self.stage(model)
        self.assertTrue(all(tensor.device.type == 'meta' for tensor in snapshot(model).values()))
        self.assertIsNone(stage.device)
        stage.load(torch.device('cpu'))
        after = snapshot(model)
        self.assertEqual(set(before), set(after), 'non-persistent buffers are kept too')
        for name in before:
            self.assertTrue(torch.equal(before[name], after[name]), name)
            self.assertEqual(before[name].dtype, after[name].dtype, name)
        self.assertTrue(torch.equal(model(torch.ones(1, 4)), expected))

    def test_to_cpu_parks_and_to_a_device_loads(self):
        model, stage = self.stage()
        with patch.object(stage, 'load') as load:
            self.assertIs(model.to('cuda'), model)
            load.assert_called_once_with(torch.device('cuda'))
        stage.load(torch.device('cpu'))
        self.assertEqual(stage.device, torch.device('cpu'))
        self.assertIs(model.to('cpu'), model)
        self.assertIsNone(stage.device)
        self.assertEqual(model.first.weight.device.type, 'meta')

    def test_loading_twice_and_parking_twice_are_harmless(self):
        model, stage = self.stage()
        stage.park()
        stage.load(torch.device('cpu'))
        first = model.first.weight
        stage.load(torch.device('cpu'))
        self.assertIs(model.first.weight, first, 'a model that is already placed is not read again')
        stage.park()
        stage.park()
        self.assertEqual(model.first.weight.device.type, 'meta')

    def test_requires_grad_survives_a_round_trip(self):
        model = Tiny().eval().requires_grad_(False)
        model, stage = self.stage(model)
        stage.load(torch.device('cpu'))
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
        self.assertFalse(model.training)

    def test_dtype_changed_while_parked_is_applied_by_the_same_rounding(self):
        model = Tiny()
        expected = {name: tensor.half() if tensor.is_floating_point() else tensor
                    for name, tensor in snapshot(model).items()}
        model, stage = self.stage(model)
        model.half()  # as staged construction does to the background remover after building it
        stage.load(torch.device('cpu'))
        for name, tensor in snapshot(model).items():
            self.assertEqual(tensor.dtype, expected[name].dtype, name)
            self.assertTrue(torch.equal(tensor, expected[name]), name)

    def test_other_conversions_are_refused_while_parked_and_passed_on_when_placed(self):
        model, stage = self.stage()
        with self.assertRaisesRegex(RuntimeError, 'parked on disk'):
            model.to(torch.float16)
        stage.load(torch.device('cpu'))
        model.to(torch.float64)
        self.assertEqual(model.first.weight.dtype, torch.float64)

    def test_tensors_that_share_memory_can_be_stored(self):
        model = Tiny()
        model.register_buffer('alias', model.table.view(-1)[:3], persistent=False)
        before = snapshot(model)
        model, stage = self.stage(model)
        stage.load(torch.device('cpu'))
        self.assertTrue(torch.equal(before['alias'], model.alias))
        self.assertTrue(torch.equal(before['table'], model.table))

    def test_tied_parameters_are_stored_once(self):
        model = Tiny()
        model.second = torch.nn.Linear(3, 4, bias=False)
        model.second.weight = torch.nn.Parameter(model.first.weight.t().detach().clone())
        model.alias = model.second  # the same module reachable twice
        before = snapshot(model)
        model, stage = self.stage(model)
        stage.load(torch.device('cpu'))
        self.assertTrue(torch.equal(before['second.weight'], model.second.weight))

    def test_tensors_larger_than_the_staging_buffer_are_read_in_pieces(self):
        model = Tiny()
        before = snapshot(model)
        model, stage = self.stage(model)
        header = self.module._read_header(stage.path)
        self.assertEqual(sorted(header[1]), sorted(before))
        targets = {name: torch.empty_like(tensor) for name, tensor in before.items()}
        with patch.object(self.module, '_STAGING_BYTES', 8), patch.object(self.module, '_staging', {}):  # tiny pieces, alternating buffers
            self.module._restore_in_order(stage.path, header, targets, through_buffer=True)
        for name in before:
            self.assertTrue(torch.equal(before[name], targets[name]), name)

    @unittest.skipUnless(torch.cuda.is_available(), 'needs a CUDA device')
    def test_a_model_restored_to_the_graphics_device_equals_the_original(self):
        model = Tiny()
        before = snapshot(model)
        expected = model(torch.ones(1, 4))
        model, stage = self.stage(model)
        for pieces in (8, 4096):  # many small pieces through alternating buffers, and whole tensors
            with patch.object(self.module, '_STAGING_BYTES', pieces), patch.object(self.module, '_staging', {}):
                model.to('cuda')
                self.assertEqual(model.first.weight.device.type, 'cuda')
                for name, tensor in snapshot(model).items():
                    self.assertTrue(torch.equal(before[name], tensor.cpu()), name)
                self.assertTrue(torch.allclose(model(torch.ones(1, 4, device='cuda')).cpu(), expected, atol=1e-4))  # a different device sums in another order
                model.to('cpu')
                self.assertEqual(model.first.weight.device.type, 'meta')

    def test_the_reader_agrees_with_the_safetensors_library(self):
        model = Tiny()
        before = snapshot(model)
        model, stage = self.stage(model)
        targets = {name: torch.empty_like(tensor) for name, tensor in before.items()}
        self.module._restore_each(stage.path, targets)
        for name in before:
            self.assertTrue(torch.equal(before[name], targets[name]), name)

    def test_a_truncated_or_foreign_file_is_rejected_not_half_restored(self):
        model, stage = self.stage()
        data = stage.path.read_bytes()
        stage.path.write_bytes(data[:-3])
        with self.assertRaisesRegex(EOFError, 'ends inside'):
            stage.load(torch.device('cpu'))
        stage.path.write_bytes(b'not a model file at all')
        with self.assertRaisesRegex(ValueError, 'not a model file'):
            stage.load(torch.device('cpu'))
        stage.path.write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'not a model file'):
            stage.load(torch.device('cpu'))

    def test_a_file_whose_shapes_differ_is_rejected(self):
        model, stage = self.stage()
        replacement = Tiny()
        replacement.first = torch.nn.Linear(4, 5)
        from safetensors.torch import save_file
        save_file({name: tensor.detach().contiguous() for name, tensor in
                   list(replacement.named_parameters()) + list(replacement.named_buffers())}, str(stage.path))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            stage.load(torch.device('cpu'))

    def test_a_file_that_does_not_match_the_model_is_rejected(self):
        model, stage = self.stage()
        model.extra = torch.nn.Linear(1, 1)
        with self.assertRaisesRegex(ValueError, 'does not match'):
            stage.load(torch.device('cpu'))

    def test_a_model_that_is_already_parked_is_rejected(self):
        model = Tiny().to('meta')
        with self.assertRaisesRegex(ValueError, 'already parked'):
            self.stage(model)


class LeanCacheTests(unittest.TestCase):
    def setUp(self):
        self.module, self.generate, self.staging = load_module()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def test_adopt_parks_the_model_and_close_removes_the_files(self):
        cache = self.module.LeanCache(self.directory.name)
        model = Tiny()
        self.assertIs(cache.adopt(model, 'tiny'), model)
        self.assertEqual(model.first.weight.device.type, 'meta')
        self.assertTrue(list(cache.path.glob('*.safetensors')))
        self.assertGreater(cache.size_bytes, 0)
        cache.close()
        self.assertFalse(cache.path.exists())
        cache.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            cache.adopt(Tiny())

    def test_the_cache_lives_in_the_directory_it_is_given(self):
        cache = self.module.LeanCache(self.directory.name)
        self.addCleanup(cache.close)
        self.assertEqual(cache.path.parent, Path(self.directory.name))

    def test_construction_parks_models_as_they_are_held_and_restores_the_hooks(self):
        original_hold, original_decoders = self.generate.hold_model, self.generate.load_gpu_decoders
        if True:
            with self.module.lean_construction(self.directory.name) as cache:
                self.assertEqual(self.staging.events, ['enter'])
                held = self.generate.hold_model(Tiny(), 'denoiser')
                self.assertEqual(held.first.weight.device.type, 'meta')
                octree, gaussian = self.generate.load_gpu_decoders(object())
                self.assertTrue(all(layer.weight.device.type == 'meta' for layer in (octree, gaussian)))
                self.assertEqual(len(cache.stages), 3)
            self.assertEqual(self.staging.events, ['enter', 'exit'])
            self.assertIs(self.generate.hold_model, original_hold)
            self.assertIs(self.generate.load_gpu_decoders, original_decoders)
            self.assertTrue(cache.path.exists(), 'the cache outlives construction; the session still needs it')
            cache.close()

    def test_failed_construction_removes_the_cache_and_the_hooks(self):
        original_hold = self.generate.hold_model
        seen = []
        if True:
            with self.assertRaisesRegex(RuntimeError, 'broken'):
                with self.module.lean_construction(self.directory.name) as cache:
                    seen.append(cache.path)
                    self.generate.hold_model(Tiny(), 'denoiser')
                    raise RuntimeError('broken')
        self.assertFalse(seen[0].exists())
        self.assertIs(self.generate.hold_model, original_hold)
        self.assertEqual(self.staging.events, ['enter', 'exit'])

    def test_fused_kernels_are_not_supported(self):
        if True:
            with self.module.lean_construction(self.directory.name) as cache:
                self.addCleanup(cache.close)
                with self.assertRaisesRegex(ValueError, 'fused'):
                    self.generate.load_gpu_decoders(object(), True)


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.caches = []

    def load(self):
        module, run, _staging = load_module()

        @contextlib.contextmanager
        def recording(directory=None):
            with module.lean_construction(self.directory.name) as cache:
                self.caches.append(cache)
                yield cache

        lowram = ModuleType('idol_lowram')
        lowram.lean_construction = recording
        with patch.dict(sys.modules, {'idol_generate': run, 'idol_lowram': lowram}):
            spec = importlib.util.spec_from_file_location('isolated_idol_generate_lean', TOOLS / 'idol_generate_lean.py')
            assert spec is not None and spec.loader is not None
            command = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(command)
        return command, run

    def test_it_runs_the_generator_inside_lean_construction_and_removes_the_files(self):
        command, run = self.load()
        seen = {}

        def main():
            seen['argv'] = list(sys.argv)
            seen['model'] = run.hold_model(Tiny(), 'denoiser')  # the generator holds a model: it is parked on disk
            seen['files'] = list(self.caches[0].path.glob('*.safetensors'))
        run.main = main
        with patch.object(sys, 'argv', ['idol_generate_lean.py', '--model', 'm.gguf']):
            command.main()
        self.assertEqual(seen['argv'], ['idol_generate_lean.py', '--model', 'm.gguf', '--stage-encoders'])
        self.assertEqual(seen['model'].first.weight.device.type, 'meta')
        self.assertEqual(len(seen['files']), 1)
        self.assertFalse(self.caches[0].path.exists(), 'the files are removed when the run ends')

    def test_the_files_are_removed_when_the_generator_fails(self):
        command, run = self.load()

        def main():
            run.hold_model(Tiny(), 'denoiser')
            raise RuntimeError('out of memory')
        run.main = main
        with patch.object(sys, 'argv', ['x']):
            with self.assertRaisesRegex(RuntimeError, 'out of memory'):
                command.main()
        self.assertFalse(self.caches[0].path.exists())

    def test_the_stage_option_is_not_added_twice(self):
        command, run = self.load()
        seen = []
        run.main = lambda: seen.append(list(sys.argv))
        with patch.object(sys, 'argv', ['x', '--stage-encoders']):
            command.main()
        self.assertEqual(seen, [['x', '--stage-encoders']])

    def test_options_that_keep_packed_weights_are_refused_before_anything_is_built(self):
        for option in ('--packed-denoiser', '--fused-kernels'):
            command, run = self.load()
            run.main = lambda: self.fail('the generator must not start')
            with patch.object(sys, 'argv', ['x', option]):
                with self.assertRaises(SystemExit) as stopped:
                    command.main()
            self.assertIn('lean mode', str(stopped.exception))
        self.assertEqual(self.caches, [])


if __name__ == '__main__':
    unittest.main()

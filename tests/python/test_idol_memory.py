"""Check free-memory detection and the choices made from it, without a GPU."""
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import idol_memory as memory

SMI = '0, NVIDIA GeForce RTX 4090, 24564, 13359\n1, NVIDIA GeForce GTX 1650, 4096, 3500\n'


def smi_runner(text='', returncode=0, error=None, calls=None):
    def run(command, **kwargs):
        if calls is not None:
            calls.append(command)
        if error:
            raise error
        return SimpleNamespace(returncode=returncode, stdout=text)
    return run


def query(device='cuda', **kwargs):
    kwargs.setdefault('which', lambda name: 'nvidia-smi')
    kwargs.setdefault('environ', {})
    kwargs.setdefault('torch_probe', lambda index: None)
    return memory.query_gpu_memory(device, **kwargs)


class ParsingTests(unittest.TestCase):
    def test_reads_the_requested_device(self):
        found = memory.parse_nvidia_smi(SMI, 0)
        self.assertEqual((found.name, found.total_mib, found.free_mib, found.source),
                         ('NVIDIA GeForce RTX 4090', 24564, 13359, 'nvidia-smi'))
        self.assertEqual(memory.parse_nvidia_smi(SMI, 1).free_mib, 3500)
        self.assertIsNone(memory.parse_nvidia_smi(SMI, 2))

    def test_names_may_contain_commas(self):
        found = memory.parse_nvidia_smi('0, NVIDIA RTX, Ada Generation, 8192, 100\n', 0)
        self.assertEqual((found.name, found.total_mib, found.free_mib), ('NVIDIA RTX, Ada Generation', 8192, 100))

    def test_rejects_unusable_output(self):
        for text in ('', 'garbage', '0, GPU, [N/A], [N/A]', '0, GPU, 100, 200', '0, GPU, 0, 0', '0, GPU, 100, -1'):
            self.assertIsNone(memory.parse_nvidia_smi(text, 0), text)

    def test_device_names(self):
        self.assertEqual([memory.device_index(d) for d in ('cuda', 'CUDA:2', ' cuda:0 ', 'cpu', 'cuda:x', 'mps')],
                         [0, 2, 0, None, None, None])


class QueryTests(unittest.TestCase):
    def test_measures_with_nvidia_smi_in_a_separate_process(self):
        calls = []
        found = query(run=smi_runner(SMI, calls=calls))
        self.assertEqual(found.free_mib, 13359)
        self.assertIn('--query-gpu=index,name,memory.total,memory.free', calls[0])
        self.assertEqual(query('cuda:1', run=smi_runner(SMI)).free_mib, 3500)

    def test_visible_devices_remap_the_ordinal(self):
        self.assertEqual(query(run=smi_runner(SMI), environ={'CUDA_VISIBLE_DEVICES': '1'}).free_mib, 3500)
        self.assertEqual(query('cuda:1', run=smi_runner(SMI), environ={'CUDA_VISIBLE_DEVICES': '1,0'}).free_mib, 13359)

    def test_falls_back_to_torch_when_nvidia_smi_cannot_answer(self):
        torch_answer = memory.GpuMemory('torch device', 8000, 6000, 'torch')
        probe = lambda index: torch_answer if index == 0 else None
        for runner in (smi_runner(error=OSError('missing')), smi_runner(error=subprocess.TimeoutExpired('smi', 15)),
                       smi_runner('', returncode=9), smi_runner('not csv'), smi_runner(SMI, returncode=1)):
            self.assertIs(query(run=runner, torch_probe=probe), torch_answer)
        self.assertIs(query(run=smi_runner(SMI), which=lambda name: None, torch_probe=probe), torch_answer)
        # A visible-device list of identifiers cannot be mapped to nvidia-smi ordinals, so PyTorch answers.
        self.assertIs(query(run=smi_runner(SMI), environ={'CUDA_VISIBLE_DEVICES': 'GPU-abc'}, torch_probe=probe),
                      torch_answer)

    def test_unmeasurable_devices_give_none(self):
        self.assertIsNone(query('cpu', run=smi_runner(SMI)))
        self.assertIsNone(query('cuda:5', run=smi_runner(SMI)))

    def test_budget_is_a_ceiling_on_the_measurement(self):
        kwargs = dict(run=smi_runner(SMI), which=lambda name: 'nvidia-smi', environ={}, torch_probe=lambda index: None)
        self.assertEqual(memory.available_mib('cuda', **kwargs), 13359)
        self.assertEqual(memory.available_mib('cuda', 4, **kwargs), 4096)
        self.assertEqual(memory.available_mib('cuda', 64, **kwargs), 13359)
        none = dict(kwargs, run=smi_runner(error=OSError('missing')))
        self.assertIsNone(memory.available_mib('cuda', **none))
        self.assertEqual(memory.available_mib('cuda', 4, **none), 4096, 'a stated budget is taken at its word')
        for bad in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                memory.available_mib('cuda', bad, **kwargs)


class PlanningTests(unittest.TestCase):
    def test_idol_stays_resident_only_when_it_fits_with_headroom(self):
        need = memory.IDOL_RESIDENT_MIB + memory.HEADROOM_MIB
        plan = memory.plan_idol(need)
        self.assertEqual((plan.resident, plan.batch_guidance, plan.cudnn_attention), (True, True, True))
        plan = memory.plan_idol(need - 1)
        self.assertEqual((plan.resident, plan.batch_guidance, plan.cudnn_attention), (False, False, False))
        self.assertTrue(memory.plan_idol(10**6).resident)

    def test_idol_stages_when_memory_is_unknown_or_short(self):
        self.assertFalse(memory.plan_idol(None).resident)
        self.assertIn('could not be measured', memory.plan_idol(None).reason)
        enough = memory.IDOL_STAGED_MIB
        self.assertNotIn('run out of memory', memory.plan_idol(enough).reason)
        self.assertIn('run out of memory', memory.plan_idol(enough - 1).reason)
        self.assertFalse(memory.plan_idol(enough - 1).resident, 'staging is still tried when it may not fit')
        self.assertFalse(memory.plan_idol(0).resident)

    def test_image_budget_is_the_largest_that_fits(self):
        full = memory.IRIS_FULL_MIB + memory.HEADROOM_MIB
        self.assertEqual(memory.iris_budget_gib(full), 16.0)
        self.assertEqual(memory.iris_budget_gib(10**6), 16.0)
        self.assertEqual(memory.iris_budget_gib(None), 16.0, 'unknown memory keeps the previous default')
        below = memory.iris_budget_gib(full - 1)
        self.assertLess(below, 16.0)
        budgets = [memory.iris_budget_gib(mib) for mib in range(0, full, 256)]
        self.assertEqual(budgets, sorted(budgets), 'more memory never gives a smaller budget')
        self.assertEqual(budgets[0], memory.IRIS_MIN_BUDGET_GIB)
        peaks = dict(memory.IRIS_BUDGET_PEAKS_MIB)
        for mib in range(0, full, 64):
            budget = memory.iris_budget_gib(mib)
            if budget > memory.IRIS_MIN_BUDGET_GIB:  # the smallest budget is used even when it may not fit
                self.assertLessEqual(peaks[budget] + memory.HEADROOM_MIB, mib, f'{budget} GiB budget does not fit in {mib} MiB')
        # The largest budget whose measured peak fits is chosen, not the largest that is merely smaller.
        self.assertEqual(memory.iris_budget_gib(peaks[3.0] + memory.HEADROOM_MIB), 3.0)
        self.assertEqual(memory.iris_budget_gib(peaks[3.0] + memory.HEADROOM_MIB - 1), 2.0)
        self.assertEqual(memory.iris_budget_gib(peaks[6.0] + memory.HEADROOM_MIB), 6.0)

    def test_session_is_kept_only_when_the_image_generator_fits_beside_it(self):
        need = memory.IRIS_FULL_MIB + memory.HEADROOM_MIB
        self.assertTrue(memory.keep_idol_loaded(need))
        self.assertFalse(memory.keep_idol_loaded(need - 1))
        self.assertFalse(memory.keep_idol_loaded(None))


if __name__ == '__main__':
    unittest.main()

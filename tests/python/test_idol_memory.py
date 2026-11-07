"""Check free-memory detection and the choices made from it, without a GPU."""
import subprocess
import sys
import tempfile
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

    def test_waiting_models_go_to_disk_only_when_memory_is_short_and_disk_is_roomy(self):
        need = memory.IDOL_STAGED_HOST_MIB + memory.HOST_HEADROOM_MIB
        space = memory.LEAN_CACHE_MIB + memory.DISK_HEADROOM_MIB
        self.assertFalse(memory.plan_host(need, space).lean)
        self.assertFalse(memory.plan_host(10**6, 0).lean, 'plenty of memory needs no disk')
        self.assertTrue(memory.plan_host(need - 1, space).lean)
        self.assertTrue(memory.plan_host(need - 1, None).lean, 'unmeasurable disk space is not a reason to refuse')
        refused = memory.plan_host(need - 1, space - 1)
        self.assertFalse(refused.lean)
        self.assertIn('may run out of memory', refused.reason)
        unknown = memory.plan_host(None, space)
        self.assertFalse(unknown.lean)
        self.assertIn('could not be measured', unknown.reason)

    def test_lean_is_chosen_for_every_amount_of_memory_below_the_threshold(self):
        need = memory.IDOL_STAGED_HOST_MIB + memory.HOST_HEADROOM_MIB
        choices = [memory.plan_host(mib, 10**6).lean for mib in range(0, need + 512, 128)]
        self.assertEqual(choices, sorted(choices, reverse=True), 'more memory never switches the disk on')
        self.assertTrue(choices[0])
        self.assertFalse(choices[-1])

    def test_a_lean_plan_warns_when_even_it_may_not_fit(self):
        self.assertIn('may run out of memory', memory.plan_host(memory.IDOL_LEAN_HOST_MIB - 1, 10**6).reason)
        self.assertNotIn('may run out of memory', memory.plan_host(memory.IDOL_LEAN_HOST_MIB, 10**6).reason)

    def test_the_measurements_are_ordered(self):
        self.assertLess(memory.IDOL_LEAN_HOST_MIB, memory.IDOL_STAGED_HOST_MIB)

    def test_the_image_generator_host_figures_rise_as_the_budget_falls_and_mapping_lowers_them(self):
        for peaks in (memory.IRIS_HOST_PEAKS_MIB, memory.IRIS_MMAP_HOST_PEAKS_MIB):
            budgets = [budget for budget, _ in peaks]
            self.assertEqual(budgets, sorted(set(budgets)), 'budgets are listed once, in order')
        self.assertEqual(memory.iris_host_mib(16.0), 1093)
        self.assertEqual(memory.iris_host_mib(3.0), 6846)
        self.assertEqual(memory.iris_host_mib(5.0), memory.iris_host_mib(4.0), 'between two budgets: the lower one')
        self.assertEqual(memory.iris_host_mib(1.0), memory.iris_host_mib(2.0), 'below the smallest: the smallest')
        beyond_the_plateau = [memory.iris_host_mib(budget) for budget in (4, 5, 6, 7, 8, 12, 16, 32)]
        self.assertEqual(beyond_the_plateau, sorted(beyond_the_plateau, reverse=True),
                         'from 4 GiB up, a larger budget never needs more')
        self.assertEqual(memory.iris_mmap_host_mib(3.0), 1198)
        self.assertIsNone(memory.iris_mmap_host_mib(4.5), 'mapping was not measured above 4 GiB')
        for budget, peak in memory.IRIS_MMAP_HOST_PEAKS_MIB:
            self.assertLess(peak, memory.iris_host_mib(budget))

    def test_the_image_generator_maps_its_weights_only_when_memory_is_short(self):
        for budget in (2.0, 3.0, 4.0, 6.0, 16.0):
            need = memory.iris_host_mib(budget) + memory.HOST_HEADROOM_MIB
            self.assertFalse(memory.plan_image_host(budget, need).mmap, budget)
            self.assertFalse(memory.plan_image_host(budget, 10**6).mmap, budget)
            short = memory.plan_image_host(budget, need - 1)
            self.assertTrue(short.mmap, budget)
            self.assertIn('memory-mapping', short.reason)
        unknown = memory.plan_image_host(3.0, None)
        self.assertFalse(unknown.mmap)
        self.assertIn('could not be measured', unknown.reason)

    def test_mapping_is_chosen_for_every_amount_of_memory_below_the_threshold(self):
        need = memory.iris_host_mib(3.0) + memory.HOST_HEADROOM_MIB
        choices = [memory.plan_image_host(3.0, mib).mmap for mib in range(0, need + 512, 128)]
        self.assertEqual(choices, sorted(choices, reverse=True), 'more memory never switches mapping on')
        self.assertTrue(choices[0])
        self.assertFalse(choices[-1])

    def test_a_mapping_plan_says_when_even_mapping_may_not_fit_or_was_not_measured(self):
        self.assertIn('may run out of memory', memory.plan_image_host(3.0, memory.iris_mmap_host_mib(3.0) - 1).reason)
        self.assertNotIn('may run out of memory', memory.plan_image_host(3.0, memory.iris_mmap_host_mib(3.0)).reason)
        self.assertIn('not measured', memory.plan_image_host(6.0, 100).reason)

    def test_free_disk_space_is_measured_where_the_files_go(self):
        calls = []

        def usage(path):
            calls.append(path)
            return SimpleNamespace(free=5 * 2**30)
        self.assertEqual(memory.free_disk_mib('/cache', usage=usage), 5120)
        self.assertEqual(calls, ['/cache'])
        self.assertEqual(memory.free_disk_mib(usage=usage), 5120)
        self.assertEqual(calls[-1], tempfile.gettempdir())

        def broken(_path):
            raise FileNotFoundError('gone')
        self.assertIsNone(memory.free_disk_mib('/missing', usage=broken))

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


MEMINFO = 'MemTotal:        8000000 kB\nMemFree:          100000 kB\nMemAvailable:    3000000 kB\nBuffers: 1 kB\n'


class HostMemoryTests(unittest.TestCase):
    def test_meminfo_gives_total_and_available_in_mib(self):
        self.assertEqual(memory.parse_meminfo(MEMINFO), (7812, 2929))
        for text in ('', 'MemTotal: 100 kB\n', 'MemAvailable: 100 kB\n', 'MemTotal: x kB\nMemAvailable: 1 kB\n',
                     'MemTotal: 0 kB\nMemAvailable: 0 kB\n'):
            self.assertIsNone(memory.parse_meminfo(text), text)
        self.assertEqual(memory.parse_meminfo('MemTotal: 1024 kB\nMemAvailable: 4096 kB\n'), (1, 1),
                         'available is never more than the total')

    def test_linux_reads_proc_meminfo(self):
        files = {'/proc/meminfo': MEMINFO}
        found = memory.query_host_memory(platform='linux', read_text=files.get)
        self.assertEqual((found.total_mib, found.available_mib, found.source), (7812, 2929, '/proc/meminfo'))

    def test_a_container_limit_below_the_computers_memory_applies(self):
        files = {'/proc/meminfo': MEMINFO, '/sys/fs/cgroup/memory.max': str(4096 * 2**20) + '\n',
                 '/sys/fs/cgroup/memory.current': str(3000 * 2**20) + '\n'}
        found = memory.query_host_memory(platform='linux', read_text=files.get)
        self.assertEqual((found.total_mib, found.available_mib), (4096, 1096))
        self.assertIn('cgroup', found.source)
        files['/sys/fs/cgroup/memory.max'] = 'max\n'  # no limit
        self.assertEqual(memory.query_host_memory(platform='linux', read_text=files.get).available_mib, 2929)
        files['/sys/fs/cgroup/memory.max'] = str(10**13)  # a limit above the computer's memory changes nothing
        files['/sys/fs/cgroup/memory.current'] = '0'
        self.assertEqual(memory.query_host_memory(platform='linux', read_text=files.get).available_mib, 2929)

    def test_unreadable_or_unsupported_systems_give_none(self):
        self.assertIsNone(memory.query_host_memory(platform='linux', read_text=lambda _path: None))
        self.assertIsNone(memory.query_host_memory(platform='darwin'))
        self.assertIsNone(memory.query_host_memory(platform='win32', windows_reader=lambda: None))

        def broken():
            raise OSError('no such library')
        self.assertIsNone(memory.query_host_memory(platform='win32', windows_reader=broken))

    def test_windows_uses_the_system_call(self):
        found = memory.query_host_memory(platform='win32', windows_reader=lambda: (8192, 3000))
        self.assertEqual((found.total_mib, found.available_mib, found.source), (8192, 3000, 'GlobalMemoryStatusEx'))

    def test_this_computer_is_measurable_when_it_is_windows_or_linux(self):
        found = memory.query_host_memory()
        if sys.platform == 'win32' or sys.platform.startswith('linux'):
            self.assertIsNotNone(found)
            self.assertGreater(found.total_mib, 0)
            self.assertLessEqual(found.available_mib, found.total_mib)


if __name__ == '__main__':
    unittest.main()

"""Sample one Windows process tree and total NVIDIA device memory in bytes.

The NVIDIA Management Library (NVML) device counter includes other applications.
Process working sets can share pages; their sum is not unique physical memory.
Private committed bytes include private allocations backed by RAM or paging.
Sampling can miss shorter peaks. This is not a minimum-memory certification.
"""
import ctypes
import json
import os
import statistics
import threading
import time
from pathlib import Path

import psutil


class NvidiaMemory:
    def __init__(self, index=0):
        self.library = ctypes.CDLL('nvml.dll' if os.name == 'nt' else 'libnvidia-ml.so.1')
        self.check(self.library.nvmlInit_v2())
        self.handle = ctypes.c_void_p()
        self.check(self.library.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.handle)))
        self.name = self.string('nvmlDeviceGetName', self.handle)
        self.uuid = self.string('nvmlDeviceGetUUID', self.handle)
        self.driver = self.string('nvmlSystemGetDriverVersion')

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError('NVML returned error ' + str(code))

    def string(self, function, *args):
        result = ctypes.create_string_buffer(128)
        self.check(getattr(self.library, function)(*args, result, len(result)))
        return result.value.decode()

    def read(self):
        class Memory(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in ('total', 'free', 'used')]
        result = Memory()
        self.check(self.library.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(result)))
        return dict(total=result.total, free=result.free, used=result.used)

    def close(self):
        self.check(self.library.nvmlShutdown())


def summarize_samples(samples):
    if not samples:
        raise ValueError('No memory samples')
    gaps = [b['seconds'] - a['seconds'] for a, b in zip(samples, samples[1:])]
    phases = {}
    for row in samples:
        phases.setdefault(row['phase'], []).append(row)
    def summarize(rows):
        return {
            'samples': len(rows),
            'device_used_min_bytes': min(x['device']['used'] for x in rows),
            'device_used_peak_bytes': max(x['device']['used'] for x in rows),
            'process_tree_working_set_peak_bytes': max(x['working_set_bytes'] for x in rows),
            'process_tree_private_commit_peak_bytes': max(x['private_commit_bytes'] for x in rows),
            'system_available_min_bytes': min(x['system_available_bytes'] for x in rows),
        }
    return {
        'overall': summarize(samples),
        'phases': {name: summarize(rows) for name, rows in phases.items()},
        'observed_sample_gap_seconds': {
            'median': statistics.median(gaps) if gaps else None,
            'maximum': max(gaps) if gaps else None,
        },
    }


class MemorySampler:
    def __init__(self, directory, *, root_pid=None, interval=.1, gpu_index=0):
        if os.name != 'nt':
            raise RuntimeError('Private committed-byte measurements here require Windows')
        if interval <= 0:
            raise ValueError('Sample interval must be positive')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stream = (self.directory / 'memory-samples.jsonl').open('x', encoding='utf-8')
        self.root = psutil.Process(root_pid or os.getpid())
        self.tracked = {self.root.pid: self.root.create_time()}
        self.process_names = {}
        self.gpu = NvidiaMemory(gpu_index)
        self.interval = interval
        self.phase = 'idle'
        self.samples = []
        self.errors = []
        self.ended = threading.Event()
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def snapshot(self):
        processes = []
        try:
            # One recursive discovery per sample: each call walks the system
            # process table, so repeating it for every child distorts timings.
            for child in self.root.children(recursive=True):
                try:
                    self.tracked[child.pid] = child.create_time()
                except psutil.NoSuchProcess:
                    continue
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as error:
            self.errors.append(str(error))
        for pid, created in list(self.tracked.items()):
            try:
                process = psutil.Process(pid)
                if process.create_time() != created:
                    continue
                memory = process.memory_info()
                self.process_names[str(pid)] = process.name()
                processes.append({'pid': pid, 'created': created,
                    'working_set_bytes': memory.rss, 'private_commit_bytes': memory.private})
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied as error:
                self.errors.append(str(error))
        row = {'seconds': time.monotonic()-self.started, 'phase': self.phase,
            'device': self.gpu.read(), 'system_available_bytes': psutil.virtual_memory().available,
            'processes': processes,
            'working_set_bytes': sum(x['working_set_bytes'] for x in processes),
            'private_commit_bytes': sum(x['private_commit_bytes'] for x in processes)}
        self.samples.append(row)
        self.stream.write(json.dumps(row, separators=(',', ':'))+'\n')

    def _loop(self):
        while not self.ended.is_set():
            start = time.monotonic()
            try:
                self.snapshot()
            except Exception as error:
                self.errors.append(repr(error))
            self.ended.wait(max(0, self.interval-(time.monotonic()-start)))

    def start(self):
        self.snapshot()
        self.thread.start()
        return self

    def stop(self):
        self.ended.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('Memory sampling thread did not stop')
        self.snapshot()
        self.stream.close()
        self.gpu.close()
        report = dict(summarize_samples(self.samples),
            status='PASS' if not self.errors else 'INCOMPLETE', errors=self.errors,
            requested_interval_seconds=self.interval, root_pid=self.root.pid,
            process_names=self.process_names,
            device={'name':self.gpu.name,'uuid':self.gpu.uuid,'driver':self.gpu.driver},
            method=__doc__,
            gpu_scope='nvmlDeviceGetMemoryInfo v1 used bytes for the whole device, including unrelated applications; no baseline subtraction or process-attribution claim.',
            host_scope='Maximum simultaneous sum across the root and observed descendants, including browser and image-generator children. Working sets can count shared pages more than once. Private commit is not resident RAM.',
            limitation='Finite sampling can miss shorter peaks or children that start and exit between samples. Process access failures invalidate a complete measurement.')
        (self.directory/'memory-summary.json').write_text(json.dumps(report,indent=2)+'\n')
        return report

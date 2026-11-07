"""Measure free graphics memory and choose the fastest configuration that fits in it.

Two things trade speed for memory in this project, and both can be sized from what the graphics device has
free right now:

* Idol itself runs either resident (every model stays on the device, batched guidance and cuDNN attention
  on; fastest) or staged (every model is built on the CPU and visits the device one stage at a time; slowest
  by roughly a fifth, about a third of the memory).
* The optional image generator takes a `--max-vram` budget. A budget at or above its full size keeps every
  weight resident (fastest); a smaller one streams weights and cuts the graph into segments, which is
  three to four times slower but fits in a few GiB. Weights it keeps off the device stay in host memory (about
  6.8 GB at budgets of 4 GiB and below) unless it memory-maps them, which it is told to do when the free system
  memory is short.

The figures below are whole-process peaks measured on an RTX 4090 (see `docs/data/memory-tiers.json`). They
are estimates: a different driver, resolution or model file moves them, so every choice keeps a headroom for
the display, other programs and that error. Only NVIDIA devices are measured; elsewhere the caller keeps the
conservative default and the user can state a budget with `--vram-budget-gib`.
"""
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

# Whole-process peaks in MiB, measured (docs/data/memory-tiers.json, docs/data/idol-staged-memory.json,
# docs/data/iris-memory-budget.json).
IDOL_RESIDENT_MIB = 5976  # resident, batched guidance and cuDNN attention on (5,622 with fused kernels)
IDOL_STAGED_MIB = 3016  # staged, built on the CPU, GPU decoders (2,754 under a 2.5 GiB allocator cap)
IRIS_FULL_MIB = 6710  # every weight resident: the peak at a budget of 8 GiB and above
# (budget in GiB, peak in MiB) of the image generator below its full size, as a non-decreasing envelope. The
# peak does not follow the budget linearly: budgets of 4 to 6 GiB all peak near 4.4 GiB.
IRIS_BUDGET_PEAKS_MIB = ((2.0, 2272), (3.0, 3287), (4.0, 4342), (5.0, 4402), (6.0, 4402))
IRIS_MIN_BUDGET_GIB = 2.0  # a budget of 1 GiB fails
IRIS_MAX_BUDGET_GIB = 16.0
HEADROOM_MIB = 512
# Host memory (RAM) of the Idol process, in MiB: the peak of its private working set (resident pages that are not
# memory-mapped model files, which the system can drop and read again) over building a session and generating once.
# Measured on Windows with PyTorch 2.10 (docs/data/idol-lowram.json); a smaller number is not possible without
# changing what PyTorch itself holds.
IDOL_STAGED_HOST_MIB = 5183  # staged: the waiting models sit in memory
IDOL_LEAN_HOST_MIB = 3286  # lean: the waiting models sit in files
LEAN_CACHE_MIB = 3704  # disk space those files take
HOST_HEADROOM_MIB = 1024  # the operating system and other programs
DISK_HEADROOM_MIB = 1024
# Host memory of the image generator, in MiB: the peak private working set of the runner's process at each
# `--max-vram` budget (budget in GiB, MiB), with its weights loaded normally and with `--mmap`
# (docs/data/text-prompt-host-memory.json). Weights that do not fit in the graphics budget stay in private host
# memory, so the figure rises as the budget falls; memory-mapped weights are file pages the system can drop and
# read again, so they are not private. With mapping, budgets above 4 GiB were not measured.
IRIS_HOST_PEAKS_MIB = ((2.0, 6737), (3.0, 6846), (4.0, 6898), (6.0, 3547), (16.0, 1093))
IRIS_MMAP_HOST_PEAKS_MIB = ((2.0, 2397), (3.0, 1198), (4.0, 1425))


@dataclass(frozen=True)
class GpuMemory:
    name: str
    total_mib: int
    free_mib: int
    source: str


@dataclass(frozen=True)
class IdolPlan:
    resident: bool
    batch_guidance: bool
    cudnn_attention: bool
    reason: str


def device_index(device):
    """The CUDA ordinal named by 'cuda' or 'cuda:N', else None (CPU or something we cannot measure)."""
    text = str(device).strip().lower()
    if text == 'cuda':
        return 0
    if text.startswith('cuda:') and text[5:].isdigit():
        return int(text[5:])
    return None


def _physical_index(index, environ):
    """Map a logical CUDA ordinal to the ordinal nvidia-smi reports, honoring CUDA_VISIBLE_DEVICES."""
    visible = environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None or not visible.strip():
        return index
    entries = [entry.strip() for entry in visible.split(',') if entry.strip()]
    if index >= len(entries) or not entries[index].isdigit():
        return None
    return int(entries[index])


def parse_nvidia_smi(text, index):
    """Parse `--query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits` output."""
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) < 4 or not parts[0].isdigit() or int(parts[0]) != index:
            continue
        try:
            total, free = int(float(parts[-2])), int(float(parts[-1]))
        except ValueError:
            return None
        if total <= 0 or not 0 <= free <= total:
            return None
        return GpuMemory(', '.join(parts[1:-2]), total, free, 'nvidia-smi')
    return None


def _from_torch(index):
    try:
        import torch
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            return None
        free, total = torch.cuda.mem_get_info(index)
        return GpuMemory(torch.cuda.get_device_name(index), int(total) // 2**20, int(free) // 2**20, 'torch')
    except Exception:
        return None


def query_gpu_memory(device='cuda', *, run=subprocess.run, which=shutil.which, environ=None, torch_probe=_from_torch):
    """Free and total memory of an NVIDIA device, or None when it cannot be measured.

    `nvidia-smi` is asked first because it runs in its own process and so does not create a CUDA context (which
    would itself take a few hundred MiB from the budget being measured). PyTorch is the fallback."""
    environ = os.environ if environ is None else environ
    index = device_index(device)
    if index is None:
        return None
    physical = _physical_index(index, environ)
    executable = which('nvidia-smi')
    if executable and physical is not None:
        try:
            result = run([executable, '--query-gpu=index,name,memory.total,memory.free', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, timeout=15, check=False)
            if result.returncode == 0:
                found = parse_nvidia_smi(result.stdout, physical)
                if found is not None:
                    return found
        except (OSError, subprocess.SubprocessError):
            pass
    return torch_probe(index)


@dataclass(frozen=True)
class HostMemory:
    total_mib: int
    available_mib: int
    source: str


def parse_meminfo(text):
    """`(total, available)` in MiB from Linux `/proc/meminfo` text, or None."""
    values = {}
    for line in text.splitlines():
        name, _, rest = line.partition(':')
        fields = rest.split()
        if name in ('MemTotal', 'MemAvailable') and len(fields) >= 1 and fields[0].isdigit():
            values[name] = int(fields[0]) // 1024  # the file is in kB
    if 'MemTotal' not in values or 'MemAvailable' not in values or values['MemTotal'] <= 0:
        return None
    return values['MemTotal'], min(values['MemAvailable'], values['MemTotal'])


def _read_text(path):
    try:
        with open(path, encoding='ascii') as stream:
            return stream.read()
    except (OSError, UnicodeDecodeError):
        return None


def _windows_host_memory():
    import ctypes

    class Status(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong), ('total', ctypes.c_ulonglong),
                    ('available', ctypes.c_ulonglong), ('total_page_file', ctypes.c_ulonglong),
                    ('available_page_file', ctypes.c_ulonglong), ('total_virtual', ctypes.c_ulonglong),
                    ('available_virtual', ctypes.c_ulonglong), ('available_extended', ctypes.c_ulonglong)]

    status = Status()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) or not status.total:
        return None
    return status.total // 2**20, status.available // 2**20


def query_host_memory(*, platform=sys.platform, read_text=_read_text, windows_reader=_windows_host_memory):
    """Physical memory of this computer and how much is free to use, or None when it cannot be measured.

    "Available" counts memory the system can take back (file cache) as free, so a memory-mapped model file
    does not make the computer look full. Inside a Linux container the cgroup limit applies when it is lower."""
    try:
        if platform == 'win32':
            found, source = windows_reader(), 'GlobalMemoryStatusEx'
        elif platform.startswith('linux'):
            text = read_text('/proc/meminfo')
            found, source = (parse_meminfo(text) if text else None), '/proc/meminfo'
            limit, used = read_text('/sys/fs/cgroup/memory.max'), read_text('/sys/fs/cgroup/memory.current')
            if found and limit and used and limit.strip().isdigit() and used.strip().isdigit():
                room = max(0, (int(limit) - int(used)) // 2**20)
                found = (min(found[0], int(limit) // 2**20), min(found[1], room))
                source += ' and the cgroup limit'
        else:
            return None
    except (OSError, ValueError, AttributeError):
        return None
    return HostMemory(found[0], found[1], source) if found else None


def available_mib(device='cuda', budget_gib=None, **kwargs):
    """Graphics memory this program may use, in MiB, or None when unknown.

    The smaller of what the device reports free and the user's stated budget. With no measurement, a stated
    budget is taken at its word."""
    if budget_gib is not None and (not math.isfinite(budget_gib) or budget_gib <= 0):
        raise ValueError('The graphics-memory budget must be a positive number of GiB')
    stated = None if budget_gib is None else int(budget_gib * 1024)
    measured = query_gpu_memory(device, **kwargs)
    if measured is None:
        return stated
    return measured.free_mib if stated is None else min(measured.free_mib, stated)


def plan_idol(available):
    """Pick Idol's mode for `available` MiB of free graphics memory (None: unknown)."""
    if available is None:
        return IdolPlan(False, False, False, 'free graphics memory could not be measured; using the low-memory mode')
    if available >= IDOL_RESIDENT_MIB + HEADROOM_MIB:
        return IdolPlan(True, True, True, f'{available} MiB free is enough to keep every model resident')
    reason = f'{available} MiB free; staging the models through the device'
    if available < IDOL_STAGED_MIB:
        # Staging is the smallest configuration there is, so it is still tried; the estimate may be pessimistic.
        reason += f' (about {IDOL_STAGED_MIB} MiB were needed when measured, so this may run out of memory)'
    return IdolPlan(False, False, False, reason)


@dataclass(frozen=True)
class HostPlan:
    lean: bool
    reason: str


def available_host_mib(**kwargs):
    """Memory this computer has free to use, in MiB, or None when it cannot be measured."""
    found = query_host_memory(**kwargs)
    return None if found is None else found.available_mib


def free_disk_mib(directory=None, *, usage=shutil.disk_usage):
    """Free disk space where `directory` (default: the system temporary directory) is, in MiB, or None."""
    try:
        return usage(directory or tempfile.gettempdir()).free // 2**20
    except OSError:
        return None


def plan_host(available, disk_free):
    """Pick where a staged Idol session keeps the models that are waiting, from the free memory and disk space.

    In memory (the default) is faster. On disk ("lean") is chosen only when the computer lacks the memory and
    has the space, because it adds about fifteen seconds to every generation."""
    if available is None:
        return HostPlan(False, 'free system memory could not be measured; keeping the waiting models in memory')
    need = IDOL_STAGED_HOST_MIB + HOST_HEADROOM_MIB
    if available >= need:
        return HostPlan(False, f'{available} MiB of system memory is free, enough to keep the waiting models in it')
    space = LEAN_CACHE_MIB + DISK_HEADROOM_MIB
    if disk_free is not None and disk_free < space:
        return HostPlan(False, f'only {available} MiB of system memory is free (about {need} MiB are needed to keep the '
                               f'models in it), but the {disk_free} MiB of free disk space is too little to keep '
                               f'them in files ({space} MiB); this may run out of memory')
    reason = f'{available} MiB of system memory is free; keeping the waiting models in files on disk'
    if available < IDOL_LEAN_HOST_MIB:
        reason += f' (about {IDOL_LEAN_HOST_MIB} MiB were needed when measured, so this may run out of memory)'
    return HostPlan(True, reason)


def iris_budget_gib(available):
    """The largest image-generator budget that fits in `available` MiB (None: unknown, so the full budget)."""
    if available is None:
        return IRIS_MAX_BUDGET_GIB
    usable = available - HEADROOM_MIB
    if usable >= IRIS_FULL_MIB:
        return IRIS_MAX_BUDGET_GIB
    return max((budget for budget, peak in IRIS_BUDGET_PEAKS_MIB if peak <= usable), default=IRIS_MIN_BUDGET_GIB)


def _host_peak(peaks, budget_gib):
    """The figure measured at the largest budget not above `budget_gib` (the first one below the smallest budget).

    A budget between two measured ones takes the lower one's figure, which is the larger of the two, so the
    estimate errs high."""
    return max(((budget, peak) for budget, peak in peaks if budget <= budget_gib), default=peaks[0])[1]


def iris_host_mib(budget_gib):
    """Host memory the image generator needed at `budget_gib` with its weights loaded normally, in MiB."""
    return _host_peak(IRIS_HOST_PEAKS_MIB, budget_gib)


def iris_mmap_host_mib(budget_gib):
    """The same with `--mmap`, or None for a budget above the largest one measured with it."""
    if budget_gib > IRIS_MMAP_HOST_PEAKS_MIB[-1][0]:
        return None
    return _host_peak(IRIS_MMAP_HOST_PEAKS_MIB, budget_gib)


@dataclass(frozen=True)
class ImageHostPlan:
    mmap: bool
    reason: str


def plan_image_host(budget_gib, available):
    """Decide whether the image generator should memory-map its weights, from the free system memory.

    Loading them normally is the default; mapping is chosen only when the computer lacks the memory the
    generator held at this budget, because the mapped pages are read back from disk when the system needs the
    memory elsewhere. `available` is the free system memory in MiB (None: unknown, so load normally)."""
    if available is None:
        return ImageHostPlan(False, 'free system memory could not be measured; loading the weights normally')
    need = iris_host_mib(budget_gib)
    if available >= need + HOST_HEADROOM_MIB:
        return ImageHostPlan(False, f'{available} MiB of system memory is free, enough for the weights it holds at '
                                    f'a {budget_gib:g} GiB budget (about {need} MiB)')
    mapped = iris_mmap_host_mib(budget_gib)
    reason = (f'{available} MiB of system memory is free but the generator held about {need} MiB at a '
              f'{budget_gib:g} GiB budget; memory-mapping its weights')
    if mapped is None:
        reason += ' (its private memory with mapping was not measured at this budget)'
    elif available < mapped:
        reason += f' (about {mapped} MiB were needed when measured, so this may run out of memory)'
    return ImageHostPlan(True, reason)


def keep_idol_loaded(available):
    """Whether the image generator can run at full size beside a loaded Idol session.

    `available` is the memory free while the session is still loaded. If so, there is no need to release and
    reload the session around every prompt."""
    return available is not None and available >= IRIS_FULL_MIB + HEADROOM_MIB

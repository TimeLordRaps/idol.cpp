"""Measure free graphics memory and choose the fastest configuration that fits in it.

Two things trade speed for memory in this project, and both can be sized from what the graphics device has
free right now:

* Idol itself runs either resident (every model stays on the device, batched guidance and cuDNN attention
  on; fastest) or staged (every model is built on the CPU and visits the device one stage at a time; slowest
  by roughly a fifth, about a third of the memory).
* The optional image generator takes a `--max-vram` budget. A budget at or above its full size keeps every
  weight resident (fastest); a smaller one streams weights and cuts the graph into segments, which is
  three to four times slower but fits in a few GiB.

The figures below are whole-process peaks measured on an RTX 4090 (see `docs/data/memory-tiers.json`). They
are estimates: a different driver, resolution or model file moves them, so every choice keeps a headroom for
the display, other programs and that error. Only NVIDIA devices are measured; elsewhere the caller keeps the
conservative default and the user can state a budget with `--vram-budget-gib`.
"""
import math
import os
import shutil
import subprocess
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


def iris_budget_gib(available):
    """The largest image-generator budget that fits in `available` MiB (None: unknown, so the full budget)."""
    if available is None:
        return IRIS_MAX_BUDGET_GIB
    usable = available - HEADROOM_MIB
    if usable >= IRIS_FULL_MIB:
        return IRIS_MAX_BUDGET_GIB
    return max((budget for budget, peak in IRIS_BUDGET_PEAKS_MIB if peak <= usable), default=IRIS_MIN_BUDGET_GIB)


def keep_idol_loaded(available):
    """Whether the image generator can run at full size beside a loaded Idol session.

    `available` is the memory free while the session is still loaded. If so, there is no need to release and
    reload the session around every prompt."""
    return available is not None and available >= IRIS_FULL_MIB + HEADROOM_MIB

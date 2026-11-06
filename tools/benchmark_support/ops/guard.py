"""Minimal benchmark admission guard, independent of the workstation's scheduler.

Checks memory, commit headroom on Windows, and an allocator cap. It does not
coordinate other applications' priorities or stop their processes. A scheduler-aware
adapter can be supplied with --guard-root when sharing a busy device. Historical
workstation measurements used a separate cooperative scheduler adapter.
"""
import contextlib
import ctypes
import os
import subprocess
import tempfile
import time
from pathlib import Path
import psutil


def ram_free_gb():
    return psutil.virtual_memory().available/1e9


def foreign_claim():
    return None


def claim_over(category):
    return None


def _windows_commit_gb():
    if os.name!='nt': return None
    size=ctypes.c_size_t
    class Info(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong)]+[(name,size) for name in (
            'CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable',
            'SystemCache','KernelTotal','KernelPaged','KernelNonpaged','PageSize')]+[
                ('HandleCount',ctypes.c_ulong),('ProcessCount',ctypes.c_ulong),('ThreadCount',ctypes.c_ulong)]
    info=Info(); info.cb=ctypes.sizeof(info)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info),info.cb):
        raise RuntimeError('Cannot inspect Windows commit headroom')
    return (info.CommitLimit-info.CommitTotal)*info.PageSize/1e9


def _device():
    result=subprocess.run(['nvidia-smi','--query-gpu=memory.free,memory.total,temperature.gpu',
                           '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15,
                          creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),check=True)
    free,total,temperature=[float(x.strip()) for x in result.stdout.splitlines()[0].split(',')]
    return free*1024**2/1e9,total*1024**2/1e9,temperature


class _Thermal:
    def __init__(self): self.checked=0
    def checkpoint(self):
        now=time.monotonic()
        if now-self.checked<5: return
        self.checked=now
        temperature=_device()[2]
        if temperature>=80:
            while _device()[2]>70:
                time.sleep(5)


@contextlib.contextmanager
def guarded(name,vram_gb=0,ram_gb=4,wait_s=0,category=None):
    if os.environ.get('CUDA_VISIBLE_DEVICES','') not in ('','0'):
        raise ValueError('This standalone adapter supports physical GPU 0; supply a scheduler adapter for another device')
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in name):
        raise ValueError('Invalid benchmark lock name')
    directory=Path(tempfile.gettempdir())/'idol-benchmark-locks'
    directory.mkdir(exist_ok=True)
    lock=(directory/(name+'.lock')).open('a+b')
    try:
        lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline=time.monotonic()+wait_s
        while True:
            free,total,_=_device()
            commit=_windows_commit_gb()
            if free>=vram_gb+3.2 and ram_free_gb()>=ram_gb and (commit is None or commit>=ram_gb+6.44):
                break
            if time.monotonic()>=deadline:
                raise RuntimeError('Insufficient free device memory, system memory, or commit headroom')
            time.sleep(5)
        import torch
        torch.cuda.set_per_process_memory_fraction(min(0.8,(free-3.2)/total))
        value=type('Reservation',(),{})(); value.gov=_Thermal()
        yield value
    finally:
        lock.close()

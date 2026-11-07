"""Keep the Idol models on disk, not in host memory, while they are not on the GPU.

Staged inference (`idol_staging.py`) already keeps only one stage's models on the GPU. The models that are
waiting still sit in host memory, dequantized, which is most of what the process needs: several gigabytes
on top of PyTorch itself. On a machine with little RAM that is the limit, not the graphics card.

Here each model is written once to a file as soon as it is built, and its tensors are dropped (the module
is moved to the `meta` device, which keeps its structure and no data). Placing a model on the device
reads the file once, from start to end, through two small pinned buffers and copies each piece straight to
the device, so the host never holds more than those buffers of the model that is loading. Placing it back
on the CPU, which is how staged inference parks a model, drops the tensors again.

Only where the bytes wait changes. Tensors are stored and restored exactly as they were built (a tensor
whose stored dtype differs from the module's, such as the background remover that staged construction
converts to half precision after building, is converted by the same rounding as that conversion), and the
same modules run the same operations in the same order, so results equal those of staged inference.

Use `lean_construction()` in place of `staged_construction()`. It needs disk space about equal to the
dequantized models (see `LeanCache.size_bytes`). Packed and fused loading are not supported here: those
keep quantized matrices in a custom layout that this file format does not hold.

This lives apart from `idol_generate.py` and `idol_session.py` because the published benchmark receipts
bind those files' exact bytes.
"""
import atexit
import contextlib
import json
import shutil
import struct
import sys
import tempfile
import threading
import types
from pathlib import Path

import torch

import idol_generate as run
from idol_staging import staged_construction


def _named_tensors(module):
    """Every parameter and buffer by name, including buffers that a state dictionary leaves out."""
    tensors = dict(module.named_parameters())
    tensors.update(module.named_buffers())
    return {name: tensor for name, tensor in tensors.items() if tensor is not None}


def _device_argument(args, kwargs):
    """The device a `Module.to` call names, or None when it is some other kind of conversion."""
    value = kwargs.get('device', args[0] if args else None)
    if isinstance(value, (str, torch.device, int)) and len(args) <= 1 and set(kwargs) <= {'device'}:
        return torch.device(value)
    return None


_DTYPES = {'F64': torch.float64, 'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16,
           'I64': torch.int64, 'I32': torch.int32, 'I16': torch.int16, 'I8': torch.int8, 'U8': torch.uint8,
           'BOOL': torch.bool}
_STAGING_BYTES = 16 * 2**20  # a read goes through a buffer this size when the destination is a device
_staging = {}
_staging_lock = threading.Lock()


def _read_header(path):
    """(offset of the data, {name: (dtype, shape, begin, end)}) from a safetensors file's header."""
    try:
        with open(path, 'rb') as stream:
            (length,) = struct.unpack('<Q', stream.read(8))
            if length > Path(path).stat().st_size - 8:
                raise ValueError('the header is longer than the file')
            table = json.loads(stream.read(length))
        table.pop('__metadata__', None)
        return 8 + length, {name: (entry['dtype'], tuple(entry['shape']), *entry['data_offsets'])
                            for name, entry in table.items()}
    except (struct.error, ValueError, KeyError, TypeError, AttributeError) as error:
        raise ValueError(f'{path} is not a model file written here: {error}') from error


def _staging_buffers():
    """Two pinned buffers, kept for the next restore (pinned memory is returned to the system only at exit)."""
    with _staging_lock:
        if 'buffers' not in _staging:
            _staging['buffers'] = [torch.empty(_STAGING_BYTES, dtype=torch.uint8, pin_memory=torch.cuda.is_available())
                                   for _ in range(2)]
        return _staging['buffers']


def _restore_each(path, targets):
    """Restore with the safetensors library; slower for models of many small tensors, but format-independent."""
    from safetensors import safe_open
    with safe_open(str(path), framework='pt', device='cpu') as source:
        for name, target in targets.items():
            target.copy_(source.get_tensor(name))


def _restore_in_order(path, header, targets, label='', through_buffer=False):
    """Copy the file's tensors into `targets` reading it once from start to end.

    A tensor stored in the type and layout its target has is read in pieces through two pinned buffers straight
    to the destination, one being read into while the other is copied to the device; anything else (a type
    that was converted while the model was parked) is read whole and converted by `copy_`. Opening the file
    once is what makes this fast: `safe_open(...).get_tensor` costs milliseconds per tensor, which dominates
    models with thousands of them."""
    offset, entries = header
    buffers = _staging_buffers()
    windows = [memoryview(buffer.numpy()) for buffer in buffers]
    pending = [None, None]  # the copy that is still reading each buffer
    slot = 0
    try:
        with open(path, 'rb', buffering=0) as stream:
            for name, (code, shape, begin, end) in sorted(entries.items(), key=lambda item: item[1][2]):
                target = targets[name]
                stored = _DTYPES.get(code)
                if stored is None or tuple(target.shape) != shape:
                    raise ValueError(f'{label or "model"} file does not match the model: {name}')
                size = end - begin
                if size != target.numel() * stored.itemsize:
                    raise ValueError(f'{label or "model"} file does not match the model: {name}')
                stream.seek(offset + begin)
                if stored == target.dtype and target.is_contiguous():
                    flat = target.detach().view(-1).view(torch.uint8)
                    if target.device.type == 'cpu' and not through_buffer:
                        _read_exactly(stream, memoryview(flat.numpy()), size, name)
                        continue
                    for done in range(0, size, _STAGING_BYTES):
                        count = min(_STAGING_BYTES, size - done)
                        if pending[slot] is not None:
                            pending[slot].synchronize()
                            pending[slot] = None
                        _read_exactly(stream, windows[slot], count, name)
                        if target.device.type == 'cuda':
                            with torch.cuda.device(target.device):  # the copy and its event belong to one stream
                                flat[done:done + count].copy_(buffers[slot][:count], non_blocking=True)
                                pending[slot] = torch.cuda.Event()
                                pending[slot].record()
                        else:
                            flat[done:done + count].copy_(buffers[slot][:count])
                        slot ^= 1
                else:
                    whole = torch.empty(shape, dtype=stored)
                    _read_exactly(stream, memoryview(whole.view(-1).view(torch.uint8).numpy()), size, name)
                    target.copy_(whole)
    finally:
        for event in pending:  # a buffer must not be reused, or freed, while a copy still reads it
            if event is not None:
                event.synchronize()


def _read_exactly(stream, destination, count, name):
    if count and stream.readinto(destination[:count]) != count:
        raise EOFError(f'the file ends inside {name}')


class DiskStage:
    """One model whose tensors live in a file whenever it is not on a device.

    After construction the module is parked: its tensors are on the `meta` device. `module.to(device)`
    restores them from the file, and `module.to('cpu')` parks it again, which is the whole interface the
    staged pipeline uses. Every other kind of conversion is passed to `torch.nn.Module.to` while the
    model is restored and refused while it is parked."""

    def __init__(self, module, path, name=''):
        self.module = module
        self.path = Path(path)
        self.name = name
        self.device = None
        self._write()
        # A model arrives with its tensors in memory (on the CPU when staged construction built it).
        self.device = next((tensor.device for tensor in _named_tensors(module).values()), torch.device('cpu'))
        self._install()
        self.park()

    def _write(self):
        from safetensors.torch import save_file
        stored, claimed = {}, set()
        for name, tensor in _named_tensors(self.module).items():
            if tensor.device.type == 'meta':
                raise ValueError(f'{self.name or "model"} is already parked: {name}')
            tensor = tensor.detach().contiguous()
            pointer = tensor.untyped_storage().data_ptr(), tensor.storage_offset()
            # safetensors refuses tensors that share memory; copy only those.
            stored[name] = tensor.clone() if pointer in claimed else tensor
            claimed.add(pointer)
        save_file(stored, str(self.path))

    def _install(self):
        stage = self

        def to(module, *args, **kwargs):
            target = _device_argument(args, kwargs)
            if target is None:
                if stage.device is None:
                    raise RuntimeError(f'{stage.name or "model"} is parked on disk; place it on a device first')
                return torch.nn.Module.to(module, *args, **kwargs)
            if target.type == 'cpu':
                stage.park()
            else:
                stage.load(target)
            return module

        self.module.to = types.MethodType(to, self.module)

    def park(self):
        """Drop the tensors; the file keeps them."""
        if self.device is not None:
            torch.nn.Module.to(self.module, 'meta')
            self.device = None

    def load(self, device):
        """Put the tensors on `device`, reading them from the file one at a time."""
        device = torch.device(device)
        if self.device is not None:
            if self.device != device:
                torch.nn.Module.to(self.module, device)
                self.device = device
            return
        header = _read_header(self.path)
        targets = _named_tensors(self.module)
        if set(header[1]) != set(targets):
            raise ValueError(f'{self.name or "model"} file does not match the model')
        self.module.to_empty(device=device)
        targets = _named_tensors(self.module)
        with torch.no_grad():
            if sys.byteorder == 'little':
                _restore_in_order(self.path, header, targets, self.name)
            else:
                _restore_each(self.path, targets)
        self.device = device


class LeanCache:
    """The directory that holds the parked models of one session; remove it with `close()`."""

    def __init__(self, directory=None):
        parent = None if directory is None else str(directory)
        self.path = Path(tempfile.mkdtemp(prefix='idol-lean-', dir=parent))
        self.stages = []
        self._closed = False
        atexit.register(self.close)

    def adopt(self, module, name=''):
        """Write `module` to the cache, free its tensors and return it."""
        if self._closed:
            raise RuntimeError('the cache is closed')
        stage = DiskStage(module, self.path / f'{len(self.stages):02d}.safetensors', name)
        self.stages.append(stage)
        return module

    @property
    def size_bytes(self):
        return sum(stage.path.stat().st_size for stage in self.stages if stage.path.exists())

    def close(self):
        if not self._closed:
            self._closed = True
            for stage in self.stages:
                stage.park()
            shutil.rmtree(self.path, ignore_errors=True)


@contextlib.contextmanager
def lean_construction(directory=None):
    """Build the Idol pipeline on the CPU and park every model on disk as soon as it is built.

    Use it exactly as `staged_construction()`, around the construction of an `IdolSession`, and keep
    the yielded cache until the session is closed. Peak host memory while building is that of the
    largest single model; afterwards it is that of the process without models."""
    cache = LeanCache(directory)
    previous_hold, previous_decoders = run.hold_model, run.load_gpu_decoders

    def hold(module, name):
        return cache.adopt(module, name)

    def load_decoders(model, fused_kernels=False):
        if fused_kernels:
            raise ValueError('lean construction does not support fused kernels')
        octree, gaussian = previous_decoders(model, fused_kernels)
        return cache.adopt(octree, 'octree decoder'), cache.adopt(gaussian, 'Gaussian decoder')

    run.hold_model, run.load_gpu_decoders = hold, load_decoders
    try:
        with staged_construction():
            yield cache
    except BaseException:
        cache.close()
        raise
    finally:
        run.hold_model, run.load_gpu_decoders = previous_hold, previous_decoders

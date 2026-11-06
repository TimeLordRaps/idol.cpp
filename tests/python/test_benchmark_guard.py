"""Check the standalone benchmark admission adapter without allocating GPU memory."""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/benchmark_support'))
from ops import guard

fractions=[]
fake=SimpleNamespace(cuda=SimpleNamespace(set_per_process_memory_fraction=fractions.append))
with tempfile.TemporaryDirectory() as directory, \
     patch.object(guard.tempfile,'gettempdir',return_value=directory), \
     patch.object(guard,'_device',return_value=(20.,24.,30.)), \
     patch.object(guard,'ram_free_gb',return_value=32.), \
     patch.object(guard,'_windows_commit_gb',return_value=64.), \
     patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'0'}), \
     patch.dict(sys.modules,{'torch':fake}):
    with guard.guarded('test',vram_gb=10,ram_gb=8.5):
        try:
            with guard.guarded('test',vram_gb=10,ram_gb=8.5):
                raise AssertionError('Concurrent worker admitted')
        except OSError:
            pass
    assert abs(fractions[0]-.7)<1e-12
    with patch.object(guard,'_windows_commit_gb',return_value=1.):
        try:
            with guard.guarded('low-commit',vram_gb=10,ram_gb=8.5):
                raise AssertionError('Low-commit worker admitted')
        except RuntimeError as exc:
            assert 'Insufficient' in str(exc)
print('Standalone admission lock, memory refusal, and allocator-cap checks passed')

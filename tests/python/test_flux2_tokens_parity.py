"""Compare the native token transformation with Idol's PyTorch operations."""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in ('numpy', 'torch')):
    print('skipped: needs numpy and PyTorch')
    sys.exit(77)
import numpy as np
import torch

batch, channels, height, width = 2, 3, 4, 6
index = torch.arange(batch * channels * height * width).view(batch, channels, height, width)
mean = (index.remainder(17).float() - 8) * 0.125
logvar = ((index * 7).remainder(19).float() - 9) * 0.125
noise = ((index * 3).remainder(13).float() - 6) * 0.25
latents = mean + torch.exp(0.5 * logvar.clamp(-30, 20)) * noise
latents = latents.view(batch, channels, height // 2, 2, width // 2, 2).permute(0, 1, 3, 5, 2, 4)
latents = latents.reshape(batch, channels * 4, height // 2, width // 2)
running_mean = (torch.arange(channels * 4).float() - 6) * 0.1
running_variance = (torch.arange(channels * 4).float() + 1) * 0.25
tokens = ((latents - running_mean.view(1, -1, 1, 1)) /
          torch.sqrt(running_variance.view(1, -1, 1, 1) + 1e-5)).flatten(2).transpose(1, 2)
tokens = torch.cat((torch.zeros(batch, 5, channels * 4), tokens), dim=1).contiguous()
with tempfile.TemporaryDirectory() as directory:
    result = Path(directory) / 'tokens.f32'
    subprocess.run([sys.argv[1], str(result)], check=True)
    native = np.fromfile(result, dtype=np.float32).reshape(tokens.shape)
    np.testing.assert_allclose(native, tokens.numpy(), rtol=2e-6, atol=2e-6)
print('native FLUX.2 token transformation matches the float32 PyTorch reference')

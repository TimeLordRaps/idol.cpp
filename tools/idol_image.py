"""Run an explicitly configured, local stable-diffusion.cpp image generator."""
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from idol_memory import available_mib, iris_budget_gib


class LocalImageGenerator:
    """Invoke a named local model, releasing its process before Idol starts."""

    def __init__(self, config_path, *, run_process=None, memory_probe=None):
        config = json.loads(Path(config_path).read_text(encoding='utf-8'))
        if config.get('version') != 1 or config.get('engine') != 'stable-diffusion.cpp':
            raise ValueError('Unsupported image-generator configuration')
        self.executable = Path(config['executable'])
        self.diffusion_model = Path(config['diffusion_model'])
        self.text_encoder = Path(config['text_encoder'])
        self.vae = Path(config['vae']) if config.get('vae') else None
        for path in (self.executable, self.diffusion_model, self.text_encoder, self.vae):
            if path is None:
                continue
            if not path.is_absolute() or not path.is_file():
                raise ValueError(f'Image-generator file not found: {path}')
        self.license_notice = str(config.get('license_notice', '')).strip()
        self._run_process = run_process
        self.steps = int(config.get('steps', 30))
        self.cfg = float(config.get('cfg', 5.0))
        # A number fixes the runner's graphics-memory budget; "auto" (the default) sizes it for every image from
        # the memory free at that moment, so a large device keeps every weight resident and a small one streams.
        configured = config.get('max_vram_gib', 'auto')
        self.max_vram = None if configured == 'auto' else float(configured)
        self.memory_probe = memory_probe
        self.last_budget_gib = None
        self.sampling_method = str(config.get('sampling_method', 'euler'))
        self.width, self.height = int(config.get('width', 1024)), int(config.get('height', 1024))
        if not 1 <= self.steps <= 100 or not math.isfinite(self.cfg) or not 0 <= self.cfg <= 20:
            raise ValueError('Invalid image-generation sampling settings')
        if self.max_vram is not None and (not math.isfinite(self.max_vram) or not 1 <= self.max_vram <= 64):
            raise ValueError('Invalid image-generator graphics-memory budget')
        if self.sampling_method not in {'euler', 'euler_a', 'heun', 'dpm2', 'dpm++2m', 'dpm++2s_a'}:
            raise ValueError('Unsupported image-generation sampling method')
        if any(value < 256 or value > 2048 or value % 16 for value in (self.width, self.height)):
            raise ValueError('Image dimensions must be multiples of 16 between 256 and 2048')

    @property
    def label(self):
        return (f'{self.diffusion_model.name} | text encoder: {self.text_encoder.name}'
                + (f' — {self.license_notice}' if self.license_notice else ''))

    def budget_gib(self):
        """The runner's `--max-vram` for the next image: the configured number, or the largest that fits now."""
        if self.max_vram is not None:
            return self.max_vram
        available = self.memory_probe() if self.memory_probe else available_mib()
        budget = iris_budget_gib(available)
        print(f'Image generator budget: {budget:g} GiB'
              + (f' ({available} MiB of graphics memory free)' if available is not None else ' (free memory unknown)'),
              file=sys.stderr, flush=True)
        return budget

    def generate(self, prompt, seed):
        if not prompt or not prompt.strip() or len(prompt) > 4000:
            raise ValueError('Prompt must contain between 1 and 4000 characters')
        if not 0 <= seed < 2**63:
            raise ValueError('Local image generation requires a seed below 2^63')
        with tempfile.TemporaryDirectory(prefix='idol-image-generator-') as directory:
            output = Path(directory) / 'image.png'
            args = [str(self.executable), '--diffusion-model', str(self.diffusion_model),
                    '--llm', str(self.text_encoder)]
            if self.vae is not None:
                args.extend(['--vae', str(self.vae)])
            budget = self.budget_gib()
            self.last_budget_gib = budget
            args.extend(['--prompt', prompt, '--output', str(output), '--seed', str(seed),
                    '--steps', str(self.steps), '--cfg-scale', str(self.cfg),
                    '--sampling-method', self.sampling_method, '--width', str(self.width), '--height', str(self.height),
                    '--threads', '4', '--max-vram', str(budget)])
            runner = self._run_process or subprocess.run
            result = runner(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=1200, check=False)
            if result.returncode != 0 or not output.is_file():
                raise RuntimeError('Local image generator failed (exit code '
                                   f'{result.returncode}): {result.stdout.decode(errors="replace")[-2000:]}')
            image = output.read_bytes()
        components = {'diffusion_model': self.diffusion_model.name,
                      'text_encoder': self.text_encoder.name}
        if self.vae is not None:
            components['vae'] = self.vae.name
        return image, self.diffusion_model.name, components

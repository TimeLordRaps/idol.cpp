"""Download the tested Iris models and write a local image-generator configuration."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

IRIS = ('webmp3/Sakura-Iris-3B-GGUF', 'Sakura-Iris-3B-Q4_K_M-3.14GiB.gguf',
        '14ec1a1c5f6d442167c9622700809744dcce8094',
        'a65ca4a773ca9433166d1a3b20ba8f4e66a3193e7198264f23ffd2528a7678a3')
QWEN = ('Qwen/Qwen3-VL-4B-Instruct-GGUF', 'Qwen3VL-4B-Instruct-Q4_K_M.gguf',
        '1cd86afb9a95c410a6038ab3b40d8b578c892266',
        '66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a')


def default_config_path():
    root = (Path(os.environ['LOCALAPPDATA']) if os.environ.get('LOCALAPPDATA') else
            Path(os.environ.get('XDG_CONFIG_HOME', str(Path.home() / '.config'))))
    return root / 'idol.cpp' / 'image-generator.json'


def checked_file(path, expected):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f'Not a regular file: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f'Model digest does not match the tested file: {path}')
    return path


def prepare(runner, config, *, image_model=None, text_encoder=None, max_vram_gib=None, mmap=None):
    if mmap is not None and mmap != 'auto' and not isinstance(mmap, bool):
        raise ValueError('The mmap setting must be "auto", true or false')
    if max_vram_gib not in (None, 'auto') and (
            isinstance(max_vram_gib, bool) or not isinstance(max_vram_gib, (int, float))
            or not math.isfinite(max_vram_gib) or not 1 <= max_vram_gib <= 64):
        raise ValueError('The graphics-memory budget must be "auto" or between 1 and 64 GiB')
    runner = Path(runner).expanduser().resolve(strict=True)
    if not runner.is_file() or (os.name != 'nt' and not os.access(runner, os.X_OK)):
        raise ValueError(f'The stable-diffusion.cpp runner is not executable: {runner}')
    config = Path(config).expanduser().absolute()
    if config.exists():
        raise FileExistsError(f'Configuration already exists; preserving it: {config}')

    def resolve(spec, supplied):
        if supplied is None:
            from huggingface_hub import hf_hub_download
            supplied = hf_hub_download(repo_id=spec[0], filename=spec[1], revision=spec[2])
        return checked_file(supplied, spec[3])

    image = resolve(IRIS, image_model)
    encoder = resolve(QWEN, text_encoder)
    settings = {'version': 1, 'engine': 'stable-diffusion.cpp', 'executable': str(runner),
                'diffusion_model': str(image), 'text_encoder': str(encoder),
                'license_notice': 'Iris-3B and Qwen3-VL-4B: Apache-2.0',
                'steps': 30, 'cfg': 3.0, 'sampling_method': 'dpm++2m',
                'width': 1024, 'height': 1024}
    if max_vram_gib is not None:
        settings['max_vram_gib'] = max_vram_gib
    if mmap is not None:
        settings['mmap'] = mmap
    config.parent.mkdir(parents=True, exist_ok=True)
    with config.open('x', encoding='utf-8') as stream:
        json.dump(settings, stream, indent=2)
        stream.write('\n')
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', required=True, help='Path to a compatible stable-diffusion.cpp sd-cli executable')
    parser.add_argument('--config', default=default_config_path(), help='New configuration path; existing files are preserved')
    parser.add_argument('--image-model', help='Use an existing tested Iris file instead of downloading it')
    parser.add_argument('--text-encoder', help='Use an existing tested Qwen file instead of downloading it')
    parser.add_argument('--max-vram-gib', type=lambda text: text if text == 'auto' else float(text),
                        help='Graphics-memory budget in GiB for the runner, or "auto" (the default when omitted) to size '
                             'it from the free graphics memory for every image. A small budget makes the runner load '
                             'weights on demand and cut the graph into segments, which lowers peak memory and slows generation')
    parser.add_argument('--mmap', choices=('auto', 'on', 'off'),
                        help='Whether the runner memory-maps its weights: "auto" (the default when omitted) does so only '
                             'when the computer lacks the system memory the runner holds at its graphics budget. Mapped '
                             'weights are file pages the system can drop and read again, so they lower the memory the '
                             'runner holds but are read from disk when memory is short')
    args = parser.parse_args()
    print('Image generator configuration:', prepare(args.runner, args.config,
          image_model=args.image_model, text_encoder=args.text_encoder, max_vram_gib=args.max_vram_gib,
          mmap={None: None, 'auto': 'auto', 'on': True, 'off': False}[args.mmap]))


if __name__ == '__main__':
    main()

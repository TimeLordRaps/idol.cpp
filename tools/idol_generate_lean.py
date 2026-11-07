"""Run `idol_generate.py` with the pipeline built on the CPU and its waiting models kept on disk.

Takes the same arguments as `idol_generate.py`, except `--packed-denoiser` and `--fused-kernels`. Like
`idol_generate_staged.py` it builds every model on the CPU and runs one stage at a time, but each model is
written to a temporary file as soon as it is built and dropped from memory, so the host holds one model at a
time instead of all of them. It needs about 3.7 GB of disk space for the files, which are removed on exit. See
`idol_lowram.py`.
"""
import sys

import idol_generate as run
from idol_lowram import lean_construction

UNSUPPORTED = ('--packed-denoiser', '--fused-kernels')


def main():
    for option in UNSUPPORTED:
        if option in sys.argv:
            raise SystemExit(f'{option} keeps quantized matrices in memory and cannot be combined with lean mode')
    if '--stage-encoders' not in sys.argv:
        sys.argv.append('--stage-encoders')
    with lean_construction() as cache:
        try:
            run.main()
        finally:
            cache.close()


if __name__ == '__main__':
    main()

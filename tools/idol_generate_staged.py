"""Run `idol_generate.py` with staged encoders and a pipeline that is built on the CPU.

Takes the same arguments as `idol_generate.py`. The usual `--stage-encoders` option only parks models
after they are all on the GPU; this builds them on the CPU first, so peak GPU memory is that of the
largest stage. See `idol_staging.py`.
"""
import sys

import idol_generate as run
from idol_staging import staged_construction


def main():
    if '--stage-encoders' not in sys.argv:
        sys.argv.append('--stage-encoders')
    with staged_construction():
        run.main()


if __name__ == '__main__':
    main()

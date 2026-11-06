"""Retain sensitive denoiser blocks at original runtime precision.

The block ranking is supplied as evidence and must bind the source model.
Its values and representativeness are not independently established here.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from export_full_gguf import digest, load_denoiser_codes


def create(source, encoded, ranking, output, blocks=8):
    source, encoded, ranking, output = map(Path, (source, encoded, ranking, output))
    if type(blocks) is not int or blocks < 0:
        raise ValueError('invalid preserved block count')
    partial = output.with_name(output.name + '.partial')
    if output.exists() or partial.exists():
        raise FileExistsError(output)
    paths = {'source_sha256': source, 'parent_codes_sha256': encoded, 'ranking_sha256': ranking,
             'mix_tool_sha256': Path(__file__)}
    before = {name: digest(path) for name, path in paths.items()}
    metadata, codes = load_denoiser_codes(encoded, source, before['source_sha256'])
    if metadata['quant_type'] not in ('Q2_K', 'Q4_K', 'Q6_K', 'Q8_0'):
        raise ValueError('parent denoiser archive must use a single encoded type')
    indices = set()
    for name in codes:
        parts = name.split('.')
        if len(parts) < 3 or not parts[1].isdigit():
            raise ValueError('invalid denoiser block tensor name')
        indices.add(int(parts[1]))
    rank_data = json.loads(ranking.read_text(encoding='utf-8'))
    scores = rank_data.get('blocks')
    if (rank_data.get('status') not in (None, 'complete')
            or (rank_data.get('status') == 'complete' and rank_data.get('hashes_unchanged_after_run') is not True)
            or rank_data.get('source_sha256') != before['source_sha256']
            or rank_data.get('model_sha256') != metadata['reference_model_sha256']
            or (rank_data.get('quantization_source') == 'provided_encoded_archive'
                and rank_data.get('inputs_sha256', {}).get('encoded') != before['parent_codes_sha256'])
            or not isinstance(scores, dict)
            or set(scores) != {str(index) for index in indices} or blocks >= len(indices)):
        raise ValueError('ranking does not bind the source model and eligible blocks')
    for values in scores.values():
        if (not isinstance(values, dict) or not values
                or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values.values())):
            raise ValueError('invalid denoiser sensitivity score')
    order = sorted(indices, key=lambda index: (-max(scores[str(index)].values()), index))
    preserved = sorted(order[:blocks])
    remaining = {name: value for name, value in codes.items() if int(name.split('.')[1]) not in preserved}
    if before != {name: digest(path) for name, path in paths.items()}:
        raise RuntimeError('mixed denoiser inputs changed during creation')
    mixed = {**metadata, **before, 'quant_type': metadata['quant_type'] + '+F16', 'preserved_blocks': preserved,
             'ranking_order': order, 'selection_policy': 'descending maximum reported block error'}
    remaining['__metadata__'] = np.frombuffer(json.dumps(mixed, sort_keys=True).encode(), dtype=np.uint8)
    with partial.open('xb') as stream:
        np.savez(stream, **remaining)
    os.link(partial, output)
    partial.unlink()
    return {'encoded_sha256': digest(output), 'encoded_matrix_count': len(remaining)-1, **mixed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('source', 'encoded', 'ranking', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--blocks', type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(create(args.source, args.encoded, args.ranking, args.out, args.blocks), indent=2))


if __name__ == '__main__':
    main()

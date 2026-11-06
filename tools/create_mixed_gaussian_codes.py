"""Combine source-bound Q2 and Q4 Gaussian codes using a recorded selection.

This validates identities and tensor formats; it does not independently prove
that the selection report was calculated as claimed or generalizes to images.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from export_full_gguf import _gaussian_group, digest, load_gaussian_codes


def create(source, q2, q4, selection, output, groups=8):
    source, q2, q4, selection, output = map(Path, (source, q2, q4, selection, output))
    if type(groups) is not int or groups < 0:
        raise ValueError('invalid selected group count')
    partial = output.with_name(output.name + '.partial')
    if output.exists() or partial.exists():
        raise FileExistsError(output)
    paths = {'source_sha256': source, 'q2_codes_sha256': q2, 'q4_codes_sha256': q4,
             'selection_sha256': selection, 'tool_sha256': Path(__file__)}
    before = {name: digest(path) for name, path in paths.items()}
    selection_data = json.loads(selection.read_text(encoding='utf-8'))
    recorded = selection_data.get('inputs_sha256', {})
    if (selection_data.get('status') != 'complete' or selection_data.get('hashes_unchanged_after_run') is not True
            or recorded.get('gaussian') != before['source_sha256'] or recorded.get('q2_codes') != before['q2_codes_sha256']
            or recorded.get('q4_codes') != before['q4_codes_sha256']):
        raise ValueError('selection report does not bind the encoded archives and source')
    metadata2, codes2 = load_gaussian_codes(q2, source, before['source_sha256'], 'Q2_K')
    metadata4, codes4 = load_gaussian_codes(q4, source, before['source_sha256'], 'Q4_K')
    if set(codes2) != set(codes4):
        raise ValueError('encoded Gaussian tensor names differ')
    entries = selection_data.get('selected')
    if not isinstance(entries, list) or groups > len(entries) or any(not isinstance(entry, dict) for entry in entries):
        raise ValueError('invalid selection report or group count')
    chosen = [entry.get('group') for entry in entries[:groups]]
    valid = {_gaussian_group(name) for name in codes2}
    if (any(not isinstance(group, str) for group in chosen) or len(set(chosen)) != len(chosen)
            or not set(chosen) <= valid):
        raise ValueError('invalid or duplicate selected groups')
    q4_names = sorted(name for name in codes2 if _gaussian_group(name) in chosen)
    encoded = {name: codes4[name] if name in q4_names else codes2[name] for name in sorted(codes2)}
    if before != {name: digest(path) for name, path in paths.items()}:
        raise RuntimeError('mixed Gaussian inputs changed during creation')
    metadata = {'version': 1, 'quant_type': 'Q2_K+Q4_K', 'q4_groups': chosen, 'q4_names': q4_names,
                'q2_moments_sha256': metadata2['moments_sha256'], 'q4_moments_sha256': metadata4['moments_sha256'],
                'q2_importance': metadata2.get('importance', 'none'), 'q4_importance': metadata4.get('importance', 'none'), **before}
    encoded['__metadata__'] = np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8)
    with partial.open('xb') as stream:
        np.savez(stream, **encoded)
    os.link(partial, output)
    partial.unlink()
    return {'encoded_sha256': digest(output), 'matrix_count': len(codes2), 'q4_matrix_count': len(q4_names), **metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('source', 'q2', 'q4', 'selection', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--groups', type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(create(args.source, args.q2, args.q4, args.selection, args.out, args.groups), indent=2))


if __name__ == '__main__':
    main()

"""Check retained mixed-plan exports against their assignments and a float decoder export.

This checks structure, byte ranges, packed budgets, unchanged tensors and current file hashes. It does not prove
calibration-cache provenance, correctness of the scoring code, or visual quality.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def manifest(path):
    metadata, tensors = [], {}
    for line in path.read_text(encoding="ascii").splitlines():
        row = line.split()
        if not row:
            continue
        if row[0] in ("decoders", "rep", "case"):
            metadata.append(line)
            continue
        if row[0] == "qtensor":
            if len(row) != 8:
                raise ValueError("malformed quantized tensor record")
            _, name, out, inp, bits, group, offset, size = row
            entry = {"shape": (int(out), int(inp)), "bits": int(bits), "group": int(group),
                     "offset": int(offset), "size": int(size)}
        else:
            name, rank = row[0], int(row[1])
            if rank < 0 or len(row) != 4 + rank:
                raise ValueError("malformed tensor record")
            entry = {"shape": tuple(map(int, row[2:2 + rank])), "offset": int(row[2 + rank]),
                     "size": int(row[3 + rank])}
        if name in tensors:
            raise ValueError(f"duplicate tensor {name}")
        tensors[name] = entry
    return metadata, tensors


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(original, exports, result_file):
    result = json.loads(result_file.read_text(encoding="utf-8"))
    rows = result["points"]
    require(bool(rows), "no evaluated candidates")
    base_meta, base = manifest(original / "manifest.txt")
    require(not any("bits" in value for value in base.values()), "original must be a float export")
    checks = []
    for row in rows:
        target = float(row["target_bpw"])
        require(math.isfinite(target) and target > 0, "invalid target")
        folder = exports / f"mixed_{str(target).replace('.', '_')}_{row['method']}"
        meta, tensors = manifest(folder / "manifest.txt")
        require(meta == base_meta and set(tensors) == set(base), f"metadata or tensor names differ: {folder.name}")
        assignments = row["assignments"]
        require(set(assignments) == {n for n, t in tensors.items() if "bits" in t}, "assignment names differ")
        data = folder / "weights.bin"
        file_size = data.stat().st_size
        intervals, packed_bytes, quantized_weights = [], 0, 0
        with (original / "weights.bin").open("rb") as source, data.open("rb") as current:
            for name, tensor in tensors.items():
                before = base[name]
                offset, size = tensor["offset"], tensor["size"]
                require(tensor["shape"] == before["shape"] and offset >= 0 and offset % 64 == 0 and
                        size >= 0 and offset + size <= file_size, f"invalid tensor range or shape: {name}")
                intervals.append((offset, offset + size))
                if "bits" in tensor:
                    out, inp = tensor["shape"]
                    bits, group = tensor["bits"], tensor["group"]
                    require(bits in (2, 3, 4, 5, 6, 8) and group > 0 and group % 32 == 0 and inp % group == 0 and
                            assignments[name] == [bits, group], f"invalid quantization assignment: {name}")
                    require(size == ((out + 15) // 16) * (inp // group) * (64 + 2 * group * bits),
                            f"invalid packed size: {name}")
                    packed_bytes += size
                    quantized_weights += out * inp
                else:
                    require(size == before["size"], f"dense tensor size differs: {name}")
                    source.seek(before["offset"])
                    current.seek(offset)
                    remaining = size
                    while remaining:
                        count = min(remaining, 4 * 1024 * 1024)
                        require(source.read(count) == current.read(count), f"dense tensor bytes differ: {name}")
                        remaining -= count
        intervals.sort()
        require(intervals and all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])) and
                intervals[-1][1] == file_size, f"overlap or trailing bytes: {folder.name}")
        require(quantized_weights > 0, "no quantized weights")
        actual = 8 * packed_bytes / quantized_weights
        require(packed_bytes == row["packed_bytes"] and quantized_weights == row["quantized_weights"] and
                abs(actual - row["actual_bpw"]) < 1e-10 and actual <= target, "packed budget mismatch")
        require(row["cases_count"] > 0 and all(math.isfinite(row[k]) and row[k] >= 0 for k in
                ("oct_rel", "gs_rel", "oct_rel_max", "gs_rel_max")), "invalid recorded scores")
        checks.append({"folder": folder.name, "target_bits_per_quantized_weight": target,
                       "actual_bits_per_quantized_weight": actual, "quantized_matrices": len(assignments),
                       "weight_file_bytes": file_size, "manifest_sha256": digest(folder / "manifest.txt"),
                       "weights_sha256": digest(data)})
    return {"status": "structural checks passed", "float_manifest_sha256": digest(original / "manifest.txt"),
            "float_weights_sha256": digest(original / "weights.bin"), "results_sha256": digest(result_file),
            "limitations": ["Current hashes do not bind prior calibration caches to their source inputs.",
                            "Structural checks do not establish quantized numerical correctness or visual quality."],
            "candidates": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.original, args.exports, args.results)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"structural checks passed for {len(report['candidates'])} candidates")


if __name__ == "__main__":
    main()

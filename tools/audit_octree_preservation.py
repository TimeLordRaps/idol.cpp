"""Check that a quantized export retains every octree tensor from a float export byte for byte.

This checks the current files, not the provenance of prior calibration or the visual quality of their outputs.
"""

import argparse
import json
from pathlib import Path

from audit_idol_exports import digest, manifest, require


def audit(original, candidate):
    meta, before = manifest(original / "manifest.txt")
    got_meta, after = manifest(candidate / "manifest.txt")
    require(meta == got_meta and set(before) == set(after), "metadata or tensor names differ")
    require(not any("bits" in item for item in before.values()), "original must be a float export")
    source_file, target_file = original / "weights.bin", candidate / "weights.bin"
    source_size, target_size = source_file.stat().st_size, target_file.stat().st_size
    intervals, octree_tensors, octree_bytes, gaussian_quantized = [], 0, 0, 0
    with source_file.open("rb") as source, target_file.open("rb") as target:
        for name, item in after.items():
            old = before[name]
            offset, size = item["offset"], item["size"]
            require(item["shape"] == old["shape"] and 0 <= offset and offset % 64 == 0 and size >= 0 and
                    offset + size <= target_size and 0 <= old["offset"] and
                    old["offset"] + old["size"] <= source_size, f"invalid shape or range: {name}")
            intervals.append((offset, offset + size))
            if name.startswith("oct."):
                require("bits" not in item and size == old["size"], f"octree tensor is not preserved: {name}")
                source.seek(old["offset"])
                target.seek(offset)
                remaining = size
                while remaining:
                    count = min(remaining, 4 * 1024 * 1024)
                    require(source.read(count) == target.read(count), f"octree tensor bytes differ: {name}")
                    remaining -= count
                octree_tensors += 1
                octree_bytes += size
            elif "bits" in item:
                out, inp = item["shape"]
                bits, group = item["bits"], item["group"]
                require(bits in (2, 3, 4, 5, 6, 8) and group > 0 and group % 32 == 0 and inp % group == 0 and
                        size == ((out + 15) // 16) * (inp // group) * (64 + 2 * group * bits),
                        f"invalid packed tensor: {name}")
                if name.startswith("gs."):
                    gaussian_quantized += 1
    intervals.sort()
    require(intervals and all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])) and
            intervals[-1][1] == target_size, "overlap or trailing bytes")
    require(octree_tensors > 0 and gaussian_quantized > 0, "missing octree tensors or quantized Gaussian matrices")
    return {"status": "octree tensor bytes match", "octree_tensors": octree_tensors,
            "octree_tensor_bytes": octree_bytes, "quantized_gaussian_matrices": gaussian_quantized,
            "original_manifest_sha256": digest(original / "manifest.txt"), "original_weights_sha256": digest(source_file),
            "candidate_manifest_sha256": digest(candidate / "manifest.txt"), "candidate_weights_sha256": digest(target_file),
            "limitations": ["Current byte equality does not establish prior calibration-cache provenance.",
                            "This check does not establish full-model visual quality or hardware-independent determinism."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.original, args.candidate)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"{report['octree_tensors']} octree tensors match; {report['quantized_gaussian_matrices']} Gaussian matrices quantized")


if __name__ == "__main__":
    main()

"""Check that float-export tensor bytes match the two source safetensors files.

Three generated frequency tensors do not appear in the source weight files and are listed separately. This checks
identity of the source tensor values, not the correctness of the decoder architecture or generated frequencies.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from audit_idol_exports import digest, manifest


def audit(export, weights):
    _, records = manifest(export / "manifest.txt")
    if any("bits" in item for item in records.values()):
        raise ValueError("source export must be float32")
    expected_extras = {"oct.pos_embedder.freqs", "oct.l_embedder.freqs", "gs.pos_embedder.freqs"}
    observed = set()
    source = np.memmap(export / "weights.bin", dtype=np.uint8, mode="r")
    for prefix, filename in (("oct.", "decoder.safetensors"), ("gs.", "decoder_gs.safetensors")):
        with safe_open(str(weights / filename), framework="np") as packed:
            for name in packed.keys():
                full = prefix + name
                if full not in records:
                    raise ValueError("source tensor missing from export: " + full)
                item = records[full]
                values = np.asarray(packed.get_tensor(name), dtype="<f4")
                size, offset = item["size"], item["offset"]
                if (tuple(values.shape) != item["shape"] or values.nbytes != size or offset < 0 or
                        offset + size > source.size or values.tobytes(order="C") != source[offset:offset + size].tobytes()):
                    raise ValueError("source tensor differs from export: " + full)
                observed.add(full)
    if set(records) - observed != expected_extras:
        raise ValueError("unexpected generated or missing tensor names")
    return {"matched_source_tensors": len(observed), "generated_frequency_tensors_not_checked": sorted(expected_extras)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    files = {"export_manifest": args.export / "manifest.txt", "export_weights": args.export / "weights.bin",
             "octree_source": args.weights / "decoder.safetensors", "gaussian_source": args.weights / "decoder_gs.safetensors",
             "tool": Path(__file__)}
    before = {name: digest(path) for name, path in files.items()}
    result = audit(args.export, args.weights)
    if before != {name: digest(path) for name, path in files.items()}:
        raise RuntimeError("inputs changed during source audit")
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump({"input_sha256": before, "hashes_unchanged_after_run": True, **result}, stream, indent=2)
    print("matched source tensors:", result["matched_source_tensors"])


if __name__ == "__main__":
    main()

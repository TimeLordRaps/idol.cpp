"""Score a decoder export against saved cases, retaining input identity and aggregate output errors.

Saved reference and calibration cache provenance must be established separately. These fixed-anchor output scores
do not measure construction or visual quality.
"""
import argparse
import json
from pathlib import Path

import quant_idol as qi
from eval_octree import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--case", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.out.exists() or args.threads < 1:
        parser.error("output exists or thread count is invalid")
    files = {"runner": args.runner, "candidate_manifest": args.candidate / "manifest.txt",
             "candidate_weights": args.candidate / "weights.bin", "tool": Path(__file__), "scorer": Path(qi.__file__)}
    files.update({f"case_{i}_{name}": case / name for i, case in enumerate(args.case)
                  for name in ("anchors.bin", "cond.bin", "ref_oct.bin", "ref_gs.bin")})
    before = {name: digest(path) for name, path in files.items()}
    scores = qi.score(args.runner, args.candidate, args.case, args.threads, "float")
    if before != {name: digest(path) for name, path in files.items()}:
        raise RuntimeError("inputs changed during scoring")
    with args.out.open("x", encoding="utf-8") as out:
        json.dump({"input_sha256": before, "hashes_unchanged_after_run": True,
                   "reference_case_provenance": "not fully established", "scores": scores}, out, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()

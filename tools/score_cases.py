"""Record per-case fixed-anchor decoder scores without disclosing input filenames in the report."""
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
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists() or args.threads < 1:
        parser.error("output exists or thread count invalid")
    files = {"runner": args.runner, "manifest": args.candidate / "manifest.txt", "weights": args.candidate / "weights.bin",
             "tool": Path(__file__), "scorer": Path(qi.__file__)}
    files.update({f"case_{i}_{name}": case / name for i, case in enumerate(args.case)
                  for name in ("anchors.bin", "cond.bin", "ref_oct.bin", "ref_gs.bin")})
    before = {k: digest(v) for k, v in files.items()}
    rows = []
    for i, case in enumerate(args.case):
        rows.append({"index": i, "scores": qi.score(args.runner, args.candidate, [case], args.threads, "float")})
        print("scored case", i + 1, flush=True)
    if before != {k: digest(v) for k, v in files.items()}:
        raise RuntimeError("inputs changed during scoring")
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump({"input_sha256": before, "hashes_unchanged_after_run": True, "cases": rows}, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()

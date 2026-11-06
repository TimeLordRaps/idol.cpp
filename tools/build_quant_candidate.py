"""Build one preserved-octree quantization candidate with recorded input digests.

The calibration file's bytes are recorded, but its earlier source provenance is not established by this tool.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import idolq
import quant_idol as qi
from eval_octree import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--hessians", type=Path, help="required for feedback-assisted rounding; unused for nearest-code rounding")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=(2, 3, 4, 5, 6, 8), required=True)
    parser.add_argument("--group", type=int, choices=(32, 64), required=True)
    parser.add_argument("--method", choices=("rtn", "gptq"), required=True)
    parser.add_argument("--keep-float-matrix", action="append", default=[], metavar="NAME",
                        help="leave this eligible Gaussian matrix in float32; repeat for more matrices")
    parser.add_argument("--sensitivity-report", type=Path, help="input-bound Gaussian sensitivity report")
    parser.add_argument("--keep-top-sensitive", type=int, default=0,
                        help="keep this many matrices with the largest worst-case calibration sensitivity in float32")
    args = parser.parse_args()
    if args.out.exists() or args.report.exists():
        parser.error("candidate or report already exists")
    if args.method == "gptq" and args.hessians is None:
        parser.error("--hessians is required for feedback-assisted rounding")
    if (args.keep_top_sensitive < 0 or bool(args.sensitivity_report) != bool(args.keep_top_sensitive) or
            (args.keep_top_sensitive and (args.keep_float_matrix or args.method != "rtn" or args.bits != 8))):
        parser.error("sensitivity selection requires nearest-code eight-bit rounding, a report and a positive count")
    files = {"manifest": args.export / "manifest.txt", "weights": args.export / "weights.bin",
             "builder": Path(__file__), "quantizer": Path(qi.__file__), "packing": Path(idolq.__file__)}
    if args.method == "gptq":
        files["hessians"] = args.hessians
    if args.sensitivity_report:
        files["sensitivity"] = args.sensitivity_report
    before = {name: digest(path) for name, path in files.items()}
    keep, tensors = qi.parse_manifest(args.export)
    plan = qi.uniform_plan(tensors, args.bits, args.group, preserve_octree=True)
    if args.sensitivity_report:
        sensitivity = json.loads(args.sensitivity_report.read_text(encoding="utf-8"))
        values = sensitivity.get("matrices", {})
        if (not sensitivity.get("hashes_unchanged_after_run") or sensitivity.get("bits") != args.bits or
                sensitivity.get("group") != args.group or sensitivity.get("input_sha256", {}).get("manifest") != before["manifest"] or
                sensitivity.get("input_sha256", {}).get("export_weights") != before["weights"] or
                set(values) != set(plan) or args.keep_top_sensitive >= len(plan) or
                any(not math.isfinite(v.get("max_relative_squared_error", float("nan"))) or
                    v["max_relative_squared_error"] < 0 for v in values.values())):
            parser.error("sensitivity report does not match this export and quantization plan")
        args.keep_float_matrix = sorted(values, key=lambda name: (-values[name]["max_relative_squared_error"], name))[:args.keep_top_sensitive]
    if len(set(args.keep_float_matrix)) != len(args.keep_float_matrix) or set(args.keep_float_matrix) - set(plan):
        parser.error("kept matrix names must be distinct, eligible Gaussian weights")
    for name in args.keep_float_matrix:
        del plan[name]
    if not plan:
        raise RuntimeError("empty quantization plan")
    hessians = torch.load(args.hessians, map_location="cpu") if args.method == "gptq" else {}
    missing = sorted(set(plan) - set(hessians)) if args.method == "gptq" else []
    qi.write_tier(args.export, args.out, keep, tensors, plan, hessians, args.method)
    after = {name: digest(path) for name, path in files.items()}
    if before != after:
        raise RuntimeError("inputs changed during candidate construction")
    report = {"input_sha256": before, "inputs_unchanged_after_run": True, "bits": args.bits, "group": args.group,
              "method": args.method, "octree_preserved_float32": True, "quantized_matrices": len(plan),
              "additional_float32_matrices": args.keep_float_matrix,
              "gptq_fell_back_to_nearest_for": missing,
              "calibration_source_provenance": ("input-bound sensitivity report" if args.sensitivity_report else
                                                "unknown" if args.method == "gptq" else "not applicable"),
              "output_sha256": {name: digest(args.out / name) for name in ("manifest.txt", "weights.bin")}}
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print("candidate complete:", args.out, flush=True)


if __name__ == "__main__":
    main()

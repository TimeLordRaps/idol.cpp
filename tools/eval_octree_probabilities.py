"""Measure octree child-probability differences at saved fixed reference locations.

The score is the per-location total variation between eight-child probability vectors. Saved locations need not be
parent-cell centers actually visited by sampling. This does not establish a
distance between complete sampled trees or visual quality. Input hashes are compared before and after the run; prior
cache provenance remains a separate question.
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

from eval_octree import digest


def snapshot(runner, candidates, cases):
    return {"runner": digest(runner),
            "candidates": {label: {name: digest(path / name) for name in ("manifest.txt", "weights.bin")}
                           for label, path in candidates},
            "cases": [{name: digest(path / name) for name in ("anchors.bin", "cond.bin", "ref_oct.bin", "ref_gs.bin")}
                      for path in cases]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--case", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--level", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument('--reference-label', help='candidate expected to reproduce each saved reference')
    parser.add_argument('--reference-tolerance', type=float, default=1e-4,
                        help='maximum mean probability and relative logit difference for that reference')
    args = parser.parse_args()
    try:
        candidates = []
        for value in args.candidate:
            label, folder = value.split("=", 1)
            if not label or not folder:
                raise ValueError("empty candidate label or path")
            candidates.append((label, Path(folder)))
    except ValueError as exc:
        parser.error(str(exc))
    labels = {label for label, _ in candidates}
    if (not 1 <= args.level <= 20 or args.threads < 1 or len(labels) != len(candidates)
            or (args.reference_label is not None and args.reference_label not in labels)
            or not math.isfinite(args.reference_tolerance) or args.reference_tolerance < 0):
        parser.error("invalid level, threads or candidate labels")
    temp = args.out.with_name(args.out.name + ".tmp")
    if args.out.exists() or temp.exists():
        parser.error("output or temporary output already exists")
    before = snapshot(args.runner, candidates, args.case)
    report = {"status": "running", "scope": "Eight-child probabilities at saved fixed locations; these need not be parent-cell "
              "centers visited during sampling. Not a full-tree or visual-quality comparison. Calibration and case provenance are not fully bound.",
              "level": args.level, "threads": args.threads, "activation_kernel": "float",
              "input_sha256": before, "cases": [],
              "reference_validation": {"label": args.reference_label, "tolerance": args.reference_tolerance,
                                       "status": "pending" if args.reference_label else "not_requested"}}
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write("{}")

    def save():
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        os.replace(temp, args.out)

    save()
    env = dict(os.environ, IDOL_QUANT_KERNEL="float")
    try:
        for index, case in enumerate(args.case):
            values = {}
            for label, folder in candidates:
                command = [str(args.runner), str(folder), "--check", "--octree-only", "--case", str(case),
                           "--level", str(args.level), "--threads", str(args.threads), "--tolerance", "1e9"]
                done = subprocess.run(command, capture_output=True, text=True, env=env)
                if done.returncode:
                    raise RuntimeError(f"case {index + 1}, {label}: exit {done.returncode}: {done.stderr}")
                lines = [line for line in done.stdout.splitlines() if line.startswith("OCT_RESULT ")]
                if len(lines) != 1:
                    raise RuntimeError(f"case {index + 1}, {label}: missing result")
                fields = dict(item.split("=", 1) for item in lines[0].split()[1:])
                mean, worst, relative = (float(fields[key]) for key in ("oct_prob_mean", "oct_prob_max", "oct_rel"))
                if (not all(math.isfinite(v) for v in (mean, worst, relative)) or not 0 <= mean <= worst <= 1 or
                        relative < 0 or int(fields["rows"]) < 1 or int(fields["tokens"]) < 1):
                    raise RuntimeError(f"case {index + 1}, {label}: invalid metrics")
                values[label] = {"mean_child_probability_total_variation": mean,
                                 "worst_location_total_variation": worst, "relative_logit_difference": relative,
                                 "location_count": int(fields["rows"]), "latent_tokens": int(fields["tokens"])}
            if len({(v["location_count"], v["latent_tokens"]) for v in values.values()}) != 1:
                raise RuntimeError(f"case {index + 1}: inconsistent dimensions")
            if args.reference_label:
                reference = values[args.reference_label]
                if (reference['mean_child_probability_total_variation'] > args.reference_tolerance or
                        reference['relative_logit_difference'] > args.reference_tolerance):
                    report['reference_validation']['status'] = 'failed'
                    report['reference_validation']['failed_case'] = index + 1
                    report['reference_validation']['observed'] = reference
                    raise RuntimeError(f'case {index + 1}: reference does not match saved outputs; check model and level')
            report["cases"].append({"case_sha256": before["cases"][index], "candidates": values})
            save()
            print(f"completed case {index + 1}/{len(args.case)}", flush=True)
        if snapshot(args.runner, candidates, args.case) != before:
            raise RuntimeError("input hashes changed during evaluation")
        if args.reference_label:
            report['reference_validation']['status'] = 'passed'
        report["summary"] = {label: {"median_case_mean_total_variation": statistics.median(means),
                                     "max_location_total_variation": max(worst),
                                     "median_relative_logit_difference": statistics.median(logits)}
                             for label, _ in candidates
                             for means, worst, logits in [([case["candidates"][label]["mean_child_probability_total_variation"]
                                                           for case in report["cases"]],
                                                          [case["candidates"][label]["worst_location_total_variation"]
                                                           for case in report["cases"]],
                                                          [case["candidates"][label]["relative_logit_difference"]
                                                           for case in report["cases"]])]}
        report["status"] = "complete"
        report["hashes_unchanged_after_run"] = True
        save()
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        try:
            save()
        except OSError as save_error:
            print(f"could not save failure status: {save_error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()

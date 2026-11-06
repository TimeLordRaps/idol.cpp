"""Compare sampled octree cell counts across decoder exports and saved latents.

The empirical count difference is not the distance between exact model distributions. Inputs are hashed before and
after the run, but prior calibration and latent provenance must be established separately.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot(runner, reference, candidates, latents):
    return {"runner": digest(runner),
            "reference": {n: digest(reference / n) for n in ("manifest.txt", "weights.bin")},
            "candidates": {label: {n: digest(path / n) for n in ("manifest.txt", "weights.bin")}
                           for label, path in candidates},
            "latents": [digest(p) for p in latents]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--latent", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--anchors", type=int, default=8192)
    parser.add_argument("--level", type=int, default=8)
    parser.add_argument("--compare-levels", default="3,4,5,6,7,8")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    try:
        levels = [int(x) for x in args.compare_levels.split(",")]
        candidates = []
        for value in args.candidate:
            label, folder = value.split("=", 1)
            if not label or not folder:
                raise ValueError("empty candidate label or path")
            candidates.append((label, Path(folder)))
    except ValueError as exc:
        parser.error(str(exc))
    if (args.anchors < 1 or not 1 <= args.level <= 20 or args.threads < 1 or args.seed < 0 or args.noise_seed < 0 or
            args.noise_seed == args.seed or not levels or len(set(levels)) != len(levels) or
            any(not 1 <= level <= args.level for level in levels) or
            len({label for label, _ in candidates}) != len(candidates) or
            any(label == "reference_other_seed" for label, _ in candidates)):
        parser.error("invalid anchors, levels, seeds, threads or candidate labels")
    temp = args.out.with_name(args.out.name + ".tmp")
    if args.out.exists() or temp.exists():
        parser.error("output or temporary output already exists")
    before = snapshot(args.runner, args.reference, candidates, args.latent)
    report = {"status": "running", "scope": "Finite sampled cell counts; not exact distribution distance or visual quality. "
              "Same-seed paths may consume random numbers differently. The alternate reference seed illustrates sampling variation.",
              "anchors": args.anchors, "sample_level": args.level, "comparison_levels": levels,
              "seed": args.seed, "noise_seed": args.noise_seed, "threads": args.threads,
              "activation_kernel": "float", "candidate_order": [label for label, _ in candidates],
              "input_sha256": before, "cases": []}
    # Claim the output name atomically; a pre-existing result is never overwritten.
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write("{}")

    def save():
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        os.replace(temp, args.out)

    save()
    env = dict(os.environ, IDOL_QUANT_KERNEL="float")
    try:
        for index, latent in enumerate(args.latent):
            cmd = [str(args.runner), str(args.reference), str(candidates[0][1]), str(latent)]
            for _, path in candidates[1:]:
                cmd += ["--candidate", str(path)]
            cmd += ["--anchors", str(args.anchors), "--level", str(args.level), "--seed", str(args.seed),
                    "--noise-seed", str(args.noise_seed), "--compare-levels", ",".join(map(str, levels)),
                    "--threads", str(args.threads)]
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"case {index + 1} failed (exit {result.returncode}): {result.stderr}")
            found = {}
            for line in result.stdout.splitlines():
                if not line.startswith("RESULT "):
                    continue
                fields = dict(part.split("=", 1) for part in line.split()[1:])
                candidate, level = int(fields["candidate"]), int(fields["cell_level"])
                value = float(fields["cell_total_variation"])
                reference_occupied = int(fields["reference_occupied"])
                candidate_occupied = int(fields["candidate_occupied"])
                shared_occupied = int(fields["shared_occupied"])
                if (candidate not in range(-1, len(candidates)) or level not in levels or
                        not math.isfinite(value) or not 0 <= value <= 1 or (candidate, level) in found or
                        not 1 <= reference_occupied <= args.anchors or not 1 <= candidate_occupied <= args.anchors or
                        not 0 <= shared_occupied <= min(reference_occupied, candidate_occupied) or int(fields["tokens"]) < 1 or
                        int(fields["anchors"]) != args.anchors or int(fields["level"]) != args.level or
                        int(fields["seed"]) != args.seed or
                        int(fields["candidate_seed"]) != (args.noise_seed if candidate == -1 else args.seed)):
                    raise RuntimeError(f"invalid result in case {index + 1}: {line}")
                found[(candidate, level)] = {"cell_total_variation": value, "reference_occupied": reference_occupied,
                                              "candidate_occupied": candidate_occupied, "shared_occupied": shared_occupied}
            if len(found) != (len(candidates) + 1) * len(levels):
                raise RuntimeError(f"incomplete result in case {index + 1}")
            for level in levels:
                if len({found[(i, level)]["reference_occupied"] for i in range(-1, len(candidates))}) != 1:
                    raise RuntimeError(f"inconsistent reference counts in case {index + 1}, level {level}")
            report["cases"].append({"latent_sha256": before["latents"][index],
                                    "levels": {str(level): {**{label: found[(i, level)] for i, (label, _) in enumerate(candidates)},
                                                            "reference_other_seed": found[(-1, level)]} for level in levels}})
            save()
            print(f"completed case {index + 1}/{len(args.latent)}", flush=True)
        after = snapshot(args.runner, args.reference, candidates, args.latent)
        if before != after:
            raise RuntimeError("input hashes changed during evaluation")
        labels = [label for label, _ in candidates] + ["reference_other_seed"]
        report["summary"] = {str(level): {label: {"median_cell_total_variation": statistics.median(values),
                                                   "min": min(values), "max": max(values)}
                                          for label in labels
                                          for values in [[case["levels"][str(level)][label]["cell_total_variation"]
                                                          for case in report["cases"]]]} for level in levels}
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

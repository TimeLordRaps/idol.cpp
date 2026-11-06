"""Collect fresh Gaussian-decoder calibration matrices from explicitly recorded latent files.

The first holdout latents are excluded. CPU execution uses a pre-case available-memory guard on Windows and Linux.
The recorded hashes bind the inputs used in this run, not the origin or representativeness of the latent files.
"""
import argparse
import json
import math
import os
from pathlib import Path

import torch
import quant_idol as qi
from eval_construction import available_bytes
from eval_octree import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--holdout", type=int, default=6)
    parser.add_argument("--calib", type=int, default=16)
    parser.add_argument("--anchors", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--min-available-gb", type=float, default=6.0)
    args = parser.parse_args()
    paths = qi.latent_files(args.latents)
    selected = paths[args.holdout:args.holdout + args.calib]
    if (args.holdout < 1 or args.calib < 1 or len(selected) != args.calib or args.anchors < 1 or
            args.threads < 1 or not math.isfinite(args.min_available_gb) or args.min_available_gb < 0):
        parser.error("invalid parameters or insufficient calibration latents")
    inputs = {"config": args.weights / "config.json", "octree_weights": args.weights / "decoder.safetensors",
              "gaussian_weights": args.weights / "decoder_gs.safetensors", "tool": Path(__file__),
              "quantization_tool": Path(qi.__file__)}
    inputs.update({f"holdout_{i}": p for i, p in enumerate(paths[:args.holdout])})
    inputs.update({f"calibration_{i}": p for i, p in enumerate(selected)})
    inputs.update({"sdk/" + p.relative_to(args.sdk).as_posix(): p for p in sorted((args.sdk / "idol").rglob("*.py"))})
    if not any(name.startswith("sdk/") for name in inputs):
        parser.error("missing software development kit sources")
    before = {key: digest(path) for key, path in inputs.items()}
    heldout = [before[f"holdout_{i}"] for i in range(args.holdout)]
    calibration = [before[f"calibration_{i}"] for i in range(args.calib)]
    if len(set(heldout)) != len(heldout) or len(set(calibration)) != len(calibration) or set(heldout) & set(calibration):
        parser.error("holdout and calibration latents must be unique and byte-disjoint")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.out_dir / "report.json"
    report = {"status": "running", "inputs": before, "holdout_count": args.holdout,
              "calibration_count": args.calib, "anchors": args.anchors, "level": qi.LEVEL,
              "threads": args.threads, "torch_version": str(torch.__version__), "completed_cases": 0,
              "latent_origin_and_representativeness": "not established by this run"}
    def save():
        temp = args.out_dir / "report.tmp"
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        os.replace(temp, report_path)
    save()
    try:
        torch.set_num_threads(args.threads)
        if available_bytes() < args.min_available_gb * 1_000_000_000:
            raise RuntimeError("insufficient available memory before loading models")
        octree, gaussian = qi.load_models(args.weights, args.sdk)
        sums, counts, handles = {}, {}, []
        for name, module in gaussian.named_modules():
            if isinstance(module, torch.nn.Linear):
                key = "gs." + name + ".weight"
                def hook(_module, inputs, _output, key=key):
                    x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
                    sums[key] = sums.get(key, 0) + x.T @ x
                    counts[key] = counts.get(key, 0) + x.shape[0]
                handles.append(module.register_forward_hook(hook))
        with torch.no_grad():
            for i, path in enumerate(selected):
                if available_bytes() < args.min_available_gb * 1_000_000_000:
                    raise RuntimeError("available memory fell below guard before calibration case " + str(i + 1))
                cond = qi.read_latent(path)
                anchors = qi.sample_anchors(octree, cond, args.anchors, 5000 + i)
                gaussian(anchors, cond)
                report["completed_cases"] = i + 1
                save()
                print(f"calibration case {i+1}/{len(selected)}", flush=True)
        for handle in handles:
            handle.remove()
        matrices = {name: (value / counts[name]).cpu() for name, value in sums.items() if counts[name] >= 256}
        if not matrices or any(not torch.isfinite(matrix).all() for matrix in matrices.values()):
            raise RuntimeError("calibration matrices are absent or non-finite")
        if before != {key: digest(path) for key, path in inputs.items()}:
            raise RuntimeError("calibration inputs changed during run")
        target = args.out_dir / "hessians.pt"
        with target.open("xb") as stream:
            torch.save(matrices, stream)
        report["matrix_count"] = len(matrices)
        report["hessians_sha256"] = digest(target)
        report["hashes_unchanged_after_run"] = True
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        save()
        raise
    save()


if __name__ == "__main__":
    main()

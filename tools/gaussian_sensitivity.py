"""Measure one-matrix-at-a-time eight-bit error on calibration latents, not held-out latents.

This is an allocation heuristic: effects can interact, and low proxy error does not prove construction fidelity.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import idolq
import quant_idol as qi
from eval_octree import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--anchors", type=int, default=1024)
    parser.add_argument("--group", type=int, choices=(32, 64), default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.out.exists() or args.count < 1 or args.anchors < 1:
        parser.error("output exists or count is invalid")
    calibration = json.loads(args.calibration_report.read_text())
    if calibration.get("status") != "complete" or args.count > calibration["calibration_count"]:
        parser.error("complete calibration with enough cases is required")
    selected = qi.latent_files(args.latents)[calibration["holdout_count"]:][:args.count]
    if len(selected) != args.count or any(digest(p) != calibration["inputs"][f"calibration_{i}"]
                                          for i, p in enumerate(selected)):
        parser.error("calibration latents do not match the recorded split")
    inputs = {"config": args.weights / "config.json", "octree_weights": args.weights / "decoder.safetensors",
              "gaussian_weights": args.weights / "decoder_gs.safetensors", "manifest": args.export / "manifest.txt",
              "export_weights": args.export / "weights.bin", "calibration_report": args.calibration_report,
              "tool": Path(__file__), "quantization": Path(qi.__file__), "packing": Path(idolq.__file__)}
    inputs.update({f"latent_{i}": p for i, p in enumerate(selected)})
    inputs.update({"sdk/" + p.relative_to(args.sdk).as_posix(): p for p in sorted((args.sdk / "idol").rglob("*.py"))})
    before = {name: digest(path) for name, path in inputs.items()}
    if any(before[name] != calibration["inputs"].get(name) for name in ("config", "octree_weights", "gaussian_weights")):
        parser.error("model weights differ from calibration inputs")
    sdk_names = {name for name in before if name.startswith("sdk/")}
    if (not sdk_names or sdk_names != {name for name in calibration["inputs"] if name.startswith("sdk/")} or
            any(before[name] != calibration["inputs"][name] for name in sdk_names)):
        parser.error("development-kit sources differ from calibration inputs")
    _, tensors = qi.parse_manifest(args.export)
    plan = qi.uniform_plan(tensors, 8, args.group, preserve_octree=True)
    source = np.memmap(args.export / "weights.bin", dtype=np.uint8, mode="r")
    meta = {name: (shape, offset, nbytes) for name, shape, offset, nbytes in tensors if name in plan}
    qi.checkpoint()
    octree, gaussian = qi.load_models(args.weights, args.sdk)
    octree, gaussian = octree.to(args.device), gaussian.to(args.device)
    parameters = {"gs." + name: param for name, param in gaussian.named_parameters()}
    if set(plan) - set(parameters):
        raise RuntimeError("exported matrices do not match model parameters")
    cases = []
    with torch.no_grad():
        for i, path in enumerate(selected):
            qi.checkpoint()
            cond = qi.read_latent(path).to(args.device)
            anchors = qi.sample_anchors(octree, cond, args.anchors, 5000 + i)
            reference = gaussian(anchors, cond)
            denominator = float(reference.square().sum())
            if denominator <= 0 or not np.isfinite(denominator):
                raise RuntimeError("invalid reference output")
            cases.append((cond, anchors, reference, denominator))
        result = {}
        for index, name in enumerate(sorted(plan)):
            qi.checkpoint()
            shape, offset, size = meta[name]
            array = np.frombuffer(source[offset:offset + size], dtype="<f4").reshape(shape)
            param = parameters[name]
            original = param.detach().clone()
            if not np.array_equal(array, original.cpu().numpy().reshape(shape)):
                raise RuntimeError("exported matrix differs from development-kit weights: " + name)
            codes, scale, minimum = idolq.quantize_tensor(array, 8, args.group, "rtn")
            reconstructed = idolq.dequantize(codes, scale, minimum, args.group).to(args.device).reshape(parameters[name].shape)
            try:
                param.copy_(reconstructed)
                errors = [float((gaussian(anchors, cond) - reference).square().sum()) / denominator
                          for cond, anchors, reference, denominator in cases]
                if not all(np.isfinite(errors)):
                    raise RuntimeError("non-finite sensitivity")
                result[name] = {"mean_relative_squared_error": float(np.mean(errors)),
                                "max_relative_squared_error": float(max(errors))}
            finally:
                param.copy_(original)
            print(f"sensitivity {index + 1}/{len(plan)}", flush=True)
    if before != {name: digest(path) for name, path in inputs.items()}:
        raise RuntimeError("sensitivity inputs changed during run")
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump({"scope": __doc__, "input_sha256": before, "hashes_unchanged_after_run": True,
                   "group": args.group, "bits": 8, "anchors": args.anchors, "calibration_cases": args.count,
                   "device": args.device, "torch_version": str(torch.__version__), "matrices": result}, stream, indent=2)


if __name__ == "__main__":
    main()

"""Mixed-precision plans for the real Idol decoders: sensitivity, allocation, and verification in the C++ runtime.

Steps (each cached under --work):
  1. Sensitivity. Quantize one weight matrix at a time to each candidate (bits, group) with importance-weighted rounding,
     keep everything else in float32, and measure the error of both decoders on a few latents set aside for this. The
     error is the sum, over the octree decoder and the Gaussian decoder, of the squared output error divided by the
     squared size of the float32 output. Errors from different matrices add roughly in quadrature, so it is treated as
     additive.
  2. Allocation (alloc.py). Start every matrix at its cheapest candidate and keep upgrading the matrix that removes the
     most error per byte. Each step is a plan. --octree-weight changes the relative weight of octree matrix errors in
     this allocation proxy; it does not change the measured errors or prove an improvement in sampled geometry.
  3. For plans at chosen average bits per weight, write an export with the chosen rounding (rtn or gptq), run it through
     idol_decode on the held-out latents of tools/quant_idol.py, and record the error, the Gaussian-level metrics, the
     resident weight bytes and the time.

Usage:
    python tools/quant_idol_search.py --export DIR --weights DIR --sdk DIR --latents DIR --run idol_decode --work DIR
           --out search.json [--targets 2.5,3,3.5,4,4.5,5,6] [--method gptq] [--device cuda]
Run it after tools/quant_idol.py has filled --work with the held-out cases and the calibration Hessians.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alloc  # noqa: E402
import idolq  # noqa: E402
import octree_probability_sensitivity as ops  # noqa: E402
import quant_idol as qi  # noqa: E402

CANDIDATES = [(2, 32), (3, 32), (4, 32), (4, 64), (5, 64), (6, 64), (8, 64)]


def usable(array):
    inp = int(np.prod(array.shape[1:]))
    return [c for c in CANDIDATES if inp % c[1] == 0]


class Evaluator:
    """Both decoders on a few latents, with chosen matrices replaced."""

    def __init__(self, args, device):
        self.device = device
        models = qi.load_models(args.weights, args.sdk)
        self.models = {"oct.": models[0].to(device), "gs.": models[1].to(device)}
        files = qi.latent_files(args.latents)
        files = files[max(args.holdout, args.holdout + args.calib - args.sens_latents):][: args.sens_latents]
        self.cases = []
        with torch.no_grad():
            for i, path in enumerate(files):
                cond = qi.read_latent(path).to(device)
                anchors = qi.sample_anchors(self.models["oct."], cond, args.anchors, 9000 + i)
                level = torch.tensor([float(1 << qi.LEVEL)], device=device)
                ref_o = self.models["oct."](anchors, level, cond)
                ref_g = self.models["gs."](anchors, cond)
                self.cases.append((cond, anchors, level, ref_o, ref_g, float(ref_o.pow(2).sum()), float(ref_g.pow(2).sum())))
        self.params = {p + n: prm for p, m in self.models.items() for n, prm in m.named_parameters()}
        self.original = {}

    def squared_error(self, replacements):
        for name, value in replacements.items():
            prm = self.params[name]
            self.original[name] = prm.data.clone()
            prm.data.copy_(torch.as_tensor(np.asarray(value), device=self.device).reshape(prm.shape))
        total = 0.0
        try:
            with torch.no_grad():
                for cond, anchors, level, ref_o, ref_g, so, sg in self.cases:
                    total += float((self.models["oct."](anchors, level, cond) - ref_o).pow(2).sum()) / so
                    total += float((self.models["gs."](anchors, cond) - ref_g).pow(2).sum()) / sg
        finally:
            for name in replacements:
                self.params[name].data.copy_(self.original.pop(name))
        return total / len(self.cases)

    def octree_probability_error(self, name, value):
        """Mean squared per-location probability variation, using this evaluator's calibration cases."""
        if not self.cases:
            raise ValueError("no sensitivity cases")
        parameter = self.params[name]
        with torch.no_grad():
            original = parameter.detach().clone()
            try:
                parameter.copy_(torch.as_tensor(np.asarray(value), device=self.device).reshape(parameter.shape))
                errors = [ops.squared_child_probability_variation(ref_o, self.models["oct."](anchors, level, cond))
                          for cond, anchors, level, ref_o, _ref_g, _so, _sg in self.cases]
                return float(torch.stack(errors).mean())
            finally:
                parameter.copy_(original)


def sensitivities(tensors, evaluator, hessians, work):
    path = work / "sensitivity.json"
    if path.exists():
        return json.loads(path.read_text())
    arrays = dict(tensors)
    result = {}
    names = idolq.matrix_names(tensors)
    for index, name in enumerate(names):
        if name not in evaluator.params:
            continue
        result[name] = {}
        h = hessians.get(name)
        for cand in usable(arrays[name]):
            # Give a shared-device guard a chance to yield between sensitivity evaluations.
            qi.checkpoint()
            codes, scale, minimum = idolq.quantize_tensor(
                arrays[name], cand[0], cand[1], "rtn", {name: [(0, None, h)]} if h is not None else None, name)
            restored = idolq.dequantize(codes, scale, minimum, cand[1]).numpy()
            result[name][f"{cand[0]},{cand[1]}"] = evaluator.squared_error({name: restored})
        print(f"sensitivity {index + 1}/{len(names)} {name}", flush=True)
    path.write_text(json.dumps(result))
    return result


def probability_sensitivities(tensors, evaluator, hessians, args):
    """Octree-only probability proxy with a resumable, input-fingerprinted cache."""
    all_latents = qi.latent_files(args.latents)
    selected = all_latents[max(args.holdout, args.holdout + args.calib - args.sens_latents):][:args.sens_latents]
    if args.sens_latents < 1 or len(selected) != args.sens_latents:
        raise ValueError("not enough selected sensitivity latents")
    code = [Path(__file__), Path(qi.__file__), Path(idolq.__file__), Path(alloc.__file__), Path(ops.__file__)]
    fingerprint, inputs = ops.fingerprint(args, selected, code, qi.LEVEL)
    path = args.work / "octree-probability-sensitivity.json"
    cache = ops.load_cache(path, fingerprint)
    arrays = {name: array for name, array in tensors if name.startswith("oct.")}
    names = idolq.matrix_names(list(arrays.items()))
    if not names or any(name not in evaluator.params for name in names) or set(cache["values"]) - set(names):
        raise ValueError("octree sensitivity names do not match the model")
    for name in names:
        if name in cache["values"]:
            expected = {f"{bits},{group}" for bits, group in usable(arrays[name])}
            values = cache["values"][name]
            if (set(values) != expected or any(not math.isfinite(v) or v < 0 for v in values.values())):
                raise ValueError(f"invalid cached octree sensitivity: {name}")
    if cache.get("status") == "complete" and set(cache["values"]) != set(names):
        raise ValueError("completed octree sensitivity cache is incomplete")
    if cache.get("status") == "complete":
        return cache["values"], fingerprint
    for index, name in enumerate(names):
        if name in cache["values"]:
            continue
        entries = {}
        h = hessians.get(name)
        for bits, group in usable(arrays[name]):
            qi.checkpoint()
            codes, scale, minimum = idolq.quantize_tensor(
                arrays[name], bits, group, "rtn", {name: [(0, None, h)]} if h is not None else None, name)
            restored = idolq.dequantize(codes, scale, minimum, group).numpy()
            entries[f"{bits},{group}"] = evaluator.octree_probability_error(name, restored)
        cache["values"][name] = entries
        cache["inputs"] = inputs
        ops.save_cache(path, cache)
        print(f"octree probability sensitivity {index + 1}/{len(names)} {name}", flush=True)
    after, _ = ops.fingerprint(args, selected, code, qi.LEVEL)
    if after != fingerprint:
        raise RuntimeError("octree sensitivity inputs changed during evaluation")
    cache["status"] = "complete"
    cache["inputs"] = inputs
    ops.save_cache(path, cache)
    return cache["values"], fingerprint


def describe(plan):
    counts = {}
    for bits, group in plan.values():
        key = f"{bits}b/g{group}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def select_within_budget(front, shapes, budget):
    """Return the richest available plan within a packed-byte budget, or (None, None)."""
    selected, selected_bytes = None, None
    for _, plan in front:
        size = sum(idolq.blob_bytes(shapes[n][0], shapes[n][1], b, g) for n, (b, g) in plan.items())
        if size <= budget:
            selected, selected_bytes = plan, size
        else:
            break
    return selected, selected_bytes


def weight_octree_sensitivity(sens, multiplier):
    """Weight the octree proxy when allocating bytes; cached measurements are not modified."""
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("octree weight must be finite and positive")
    return {name: {candidate: error * (multiplier if name.startswith("oct.") else 1.0)
                   for candidate, error in choices.items()} for name, choices in sens.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("--export", "--weights", "--sdk", "--latents", "--run", "--work", "--out"):
        parser.add_argument(flag, type=Path, required=True)
    parser.add_argument("--calib", type=int, default=16)
    parser.add_argument("--holdout", type=int, default=6)
    parser.add_argument("--anchors", type=int, default=1024)
    parser.add_argument("--sens-latents", type=int, default=3)
    parser.add_argument("--targets", default="2.5,3,3.5,4,4.5,5,6", help="average bits per quantized weight")
    parser.add_argument("--method", default="gptq", choices=("rtn", "gptq"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--octree-weight", type=float, default=1.0,
                        help="relative weight of octree sensitivity in allocation (default: 1)")
    parser.add_argument("--octree-objective", choices=("logit", "probability"), default="logit",
                        help="octree allocation proxy; probability uses mean squared child-probability variation")
    parser.add_argument("--sensitivity-only", action="store_true", help="compute the selected sensitivity cache and stop")
    parser.add_argument("--keep", action="store_true", help="retain each scored mixed-plan export")
    parser.add_argument("--exports", type=Path, help="directory for exports (default: --work)")
    args = parser.parse_args()
    if not math.isfinite(args.octree_weight) or args.octree_weight <= 0:
        parser.error("--octree-weight must be finite and positive")
    if args.out.exists():
        parser.error("--out already exists")

    work = args.work
    cases = sorted(p for p in (work / "cases").iterdir() if p.is_dir())
    keep, tensors_meta = qi.parse_manifest(args.export)
    source = np.memmap(args.export / "weights.bin", dtype=np.uint8, mode="r")
    tensors = [(n, np.frombuffer(source[o:o + b], dtype="<f4").reshape(s)) for n, s, o, b in tensors_meta]
    # Only matrices the format can hold are candidates; the rest stay float32 in every plan.
    tensors_q = [(n, a) for n, a in tensors if qi.eligible(n, a.shape, 32)]
    hessians = torch.load(work / "hessians.pt")

    evaluator = Evaluator(args, args.device)
    sens = sensitivities(tensors_q, evaluator, hessians, work)
    probability_fingerprint = None
    if args.octree_objective == "probability":
        octree_sens, probability_fingerprint = probability_sensitivities(tensors_q, evaluator, hessians, args)
        sens = {**sens, **octree_sens}
    del evaluator
    if args.sensitivity_only:
        args.out.write_text(json.dumps({"status": "complete", "octree_objective": args.octree_objective,
                                        "probability_sensitivity_fingerprint": probability_fingerprint}, indent=1))
        return
    sizes = {n: a.size for n, a in tensors_q}
    quantized_params = sum(sizes[n] for n in sens)
    front = alloc.allocation_front(tensors_q, weight_octree_sensitivity(sens, args.octree_weight), usable)
    shapes = {n: a.shape for n, a in tensors_q}

    results = {"octree_weight": args.octree_weight, "octree_objective": args.octree_objective,
               "probability_sensitivity_fingerprint": probability_fingerprint,
               "points": [], "unattainable_targets": []}
    for target in (float(t) for t in args.targets.split(",")):
        want = target * quantized_params / 8  # bytes of quantized blobs: effective bits per weight, scales included
        pick, blob = select_within_budget(front, shapes, want)
        if pick is None:
            results["unattainable_targets"].append(target)
            print(f"target {target} bits per quantized weight is below the smallest available plan", flush=True)
            args.out.write_text(json.dumps(results, indent=1))
            continue
        actual_bpw = 8 * blob / quantized_params
        folder = (args.exports or work) / f"mixed_{str(target).replace('.', '_')}_{args.method}"
        qi.write_tier(args.export, folder, keep, tensors_meta, {n: c for n, c in pick.items()}, hessians, args.method)
        row = {"label": f"mixed {actual_bpw:.3f} bpw {args.method}", "target_bpw": target,
               "actual_bpw": actual_bpw, "packed_bytes": blob, "quantized_weights": quantized_params,
               "method": args.method, "plan": describe(pick),
               "assignments": {name: list(choice) for name, choice in pick.items()},
               **qi.score(args.run, folder, cases, args.threads, "float")}
        results["points"].append(row)
        print({key: value for key, value in row.items() if key != "assignments"}, flush=True)
        args.out.write_text(json.dumps(results, indent=1))
        if not args.keep:
            for f in folder.iterdir():
                f.unlink()
            folder.rmdir()


if __name__ == "__main__":
    sys.exit(main())

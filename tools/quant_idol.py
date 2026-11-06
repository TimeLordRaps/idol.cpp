"""Quantize the real Idol decoders and score every tier against float32 on held-out latents.

Needs a float32 export from tools/export_idol.py, the downloaded weights (for the PyTorch reference modules), the SDK
source, a folder of real latents (raw float32, 8192 x 16, one file per image: what the denoiser produces) and the
idol_decode executable. Steps, each cached under --work:

  cases     The held-out latents: anchors sampled by the float32 octree decoder, and the float32 outputs of both decoders
            on them. These are what a tier is scored against (relative L2 error of the octree logits and of the
            Gaussian decoder's raw outputs, as idol_decode --check prints it).
  hessians  Mean of x x^T at the input of every Linear, over the calibration latents (anchors sampled the same way).
  tiers     For each tier (bits, group, method) write an export with every eligible weight matrix quantized, run
            idol_decode on each held-out case, and record error, resident weight bytes, time and peak memory.
  baseline  The same score for the weights the deployed model actually runs: every block Linear rounded to float16.

Usage:
    python tools/quant_idol.py --export DIR --weights DIR --sdk DIR --latents DIR --run idol_decode --work DIR
           [--calib 16 --holdout 6 --anchors 2048] [--tiers 8:64:rtn,6:64:rtn,4:64:gptq,...] --out tiers.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import idolq  # noqa: E402

LEVEL = 8  # octree depth the model samples to


def checkpoint():
    """Called between weight matrices. A caller that shares the GPU replaces it (see the GPU guard in The Day)."""


def parse_manifest(export):
    """Returns (header lines kept verbatim, [(name, shape, offset, nbytes)] for the float tensors)."""
    keep, tensors = [], []
    for line in (Path(export) / "manifest.txt").read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] in ("decoders", "rep", "case"):
            keep.append(line)
        elif parts[0] == "qtensor":
            raise SystemExit("--export must be the float32 export, not a quantized one")
        else:
            ndim = int(parts[1])
            tensors.append((parts[0], tuple(int(d) for d in parts[2:2 + ndim]), int(parts[2 + ndim]), int(parts[3 + ndim])))
    return keep, tensors


def load_models(weights, sdk, half_blocks=False):
    sys.path.insert(0, str(sdk))
    from safetensors.torch import load_file
    from idol.models.decoder import GaussianFixedlenDecoder, OctreeProbabilityFixedlenDecoder

    config = json.load(open(Path(weights) / "config.json"))
    models = []
    for key, cls, name in (("decoder", OctreeProbabilityFixedlenDecoder, "decoder"),
                           ("decoder_gs", GaussianFixedlenDecoder, "decoder_gs")):
        model = cls(**config[key]["args"])  # the constructor leaves block Linears in float16
        if not half_blocks:
            model = model.float()           # so convert first, to load the file's float32 values as they are
        model.load_state_dict(load_file(str(Path(weights) / f"{name}.safetensors")), strict=True)
        model = model.float().eval().requires_grad_(False)
        model.dtype = torch.float32
        models.append(model)
    return models


def latent_files(directory):
    return sorted(Path(directory).glob("*.f32"))


def read_latent(path):
    return torch.from_numpy(np.fromfile(path, dtype=np.float32).reshape(1, -1, 16))


@torch.no_grad()
def sample_anchors(oct_model, cond, count, seed):
    torch.manual_seed(seed)
    return oct_model.sample(oct_model, cond, count, LEVEL)


def make_cases(args, work):
    """Held-out cases: the first --holdout latents."""
    oct_model, gs_model = load_models(args.weights, args.sdk)
    cases = []
    for i, path in enumerate(latent_files(args.latents)[: args.holdout]):
        folder = work / "cases" / path.stem
        cases.append(folder)
        if (folder / "ref_gs.bin").exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        cond = read_latent(path)
        with torch.no_grad():
            anchors = sample_anchors(oct_model, cond, args.anchors, 1000 + i)
            ref_oct = oct_model(anchors, torch.tensor([float(1 << LEVEL)]), cond)[0]
            ref_gs = gs_model(anchors, cond)[0]
        for name, array in (("anchors", anchors[0]), ("cond", cond[0]), ("ref_oct", ref_oct), ("ref_gs", ref_gs)):
            np.ascontiguousarray(array.numpy(), dtype=np.float32).tofile(folder / (name + ".bin"))
        print(f"case {path.stem}", flush=True)
    return cases


def collect_hessians(args, work):
    target = work / "hessians.pt"
    if target.exists():
        return torch.load(target)
    oct_model, gs_model = load_models(args.weights, args.sdk)
    sums, counts, handles = {}, {}, []

    def watch(prefix, model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                key = f"{prefix}{name}.weight"

                def hook(_m, inp, _o, key=key):
                    x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                    sums[key] = sums.get(key, 0) + x.T @ x
                    counts[key] = counts.get(key, 0) + x.shape[0]
                handles.append(module.register_forward_hook(hook))

    watch("oct.", oct_model)
    watch("gs.", gs_model)
    files = latent_files(args.latents)[args.holdout: args.holdout + args.calib]
    if not files:
        raise SystemExit("no calibration latents left after the held-out ones")
    with torch.no_grad():
        for i, path in enumerate(files):
            cond = read_latent(path)
            anchors = sample_anchors(oct_model, cond, args.anchors, 5000 + i)  # runs the octree decoder at every level
            gs_model(anchors, cond)
            print(f"hessian pass {i + 1}/{len(files)}", flush=True)
    for handle in handles:
        handle.remove()
    result = {k: (v / counts[k]).cpu() for k, v in sums.items() if counts[k] >= 256}
    torch.save(result, target)
    return result


def eligible(name, shape, group):
    return len(shape) == 2 and name.endswith(".weight") and shape[1] % group == 0 and shape[1] >= group


def uniform_plan(tensors, bits, group, preserve_octree=False):
    """Select matrices for one tier, optionally leaving the octree decoder in float32."""
    def selected(name, shape):
        if preserve_octree and name.startswith("oct."):
            return False
        if bits == 16:
            return ".blocks." in name and name.endswith(".weight") and len(shape) == 2
        return eligible(name, shape, group)
    return {name: (bits, 0 if bits == 16 else group) for name, shape, _, _ in tensors if selected(name, shape)}


def write_tier(export, out, keep, tensors, plan, hessians, method):
    """plan: name -> (bits, group). Writes the export with those matrices quantized."""
    # Never overwrite an existing export, including a partial or user-retained one.
    out.mkdir(parents=True, exist_ok=False)
    source = np.memmap(Path(export) / "weights.bin", dtype=np.uint8, mode="r")
    manifest = list(keep)
    offset = 0
    with open(out / "weights.bin", "wb") as stream:
        def place(data):
            nonlocal offset
            pad = (-offset) % idolq.ALIGN
            stream.write(b"\0" * pad)
            offset += pad
            start = offset
            stream.write(data)
            offset += len(data)
            return start

        for name, shape, off, nbytes in tensors:
            if name in plan and plan[name][0] == 16:
                # "16:0:fp16": the deployed model's weights, float16-rounded, stored back as float32
                array = np.frombuffer(source[off:off + nbytes], dtype="<f4").astype(np.float16).astype("<f4")
                start = place(array.tobytes())
                manifest.append(f"{name} {len(shape)} {' '.join(str(d) for d in shape)} {start} {nbytes}")
            elif name in plan:
                bits, group = plan[name]
                array = np.frombuffer(source[off:off + nbytes], dtype="<f4").reshape(shape)
                h = hessians.get(name)
                m = method if h is not None else "rtn"
                checkpoint()
                codes, scale, minimum = idolq.quantize_tensor(
                    array, bits, group, m, {name: [(0, None, h)]} if h is not None else None, name)
                blob = idolq.pack_blob(codes, scale, minimum, bits, group)
                start = place(blob)
                manifest.append(f"qtensor {name} {shape[0]} {shape[1]} {bits} {group} {start} {len(blob)}")
            else:
                start = place(source[off:off + nbytes].tobytes())
                manifest.append(f"{name} {len(shape)} {' '.join(str(d) for d in shape)} {start} {nbytes}")
    (out / "manifest.txt").write_text("\n".join(manifest) + "\n", encoding="ascii")


def score(run, export, cases, threads, kernel="float"):
    rows = []
    probability_rows = []
    env = dict(os.environ, IDOL_QUANT_KERNEL=kernel)  # "float": weights only; "int8": the VNNI kernel, activations too
    for case in cases:
        cmd = [str(run), str(export), "--check", "--case", str(case), "--level", str(LEVEL), "--tolerance", "1e9"]
        if threads:
            cmd += ["--threads", str(threads)]
        done = subprocess.run(cmd, capture_output=True, text=True, env=env)
        text = done.stdout
        if done.returncode:
            raise RuntimeError(f"decoder failed for case {case} (exit {done.returncode})\n"
                               f"stdout:\n{done.stdout}\nstderr:\n{done.stderr}")
        found = re.search(r"RESULT oct_rel=(\S+) gs_rel=(\S+) oct_s=(\S+) gs_s=(\S+) resident_mb=(\S+) peak_mb=(\S+) "
                          r"pos=(\S+) col=(\S+) opa=(\S+) lsc=(\S+)", text)
        if not found:
            raise RuntimeError(f"decoder did not report a result for case {case}\nstdout:\n{text}\nstderr:\n{done.stderr}")
        values = [float(v) for v in found.groups()]
        if not np.isfinite(values).all():
            raise RuntimeError(f"decoder reported non-finite metrics for case {case}")
        rows.append(values)
        probability = re.search(r"oct_prob_mean=(\S+) oct_prob_max=(\S+)", text)
        if probability:
            pair = [float(v) for v in probability.groups()]
            if not np.isfinite(pair).all() or not all(0 <= v <= 1 for v in pair) or pair[0] > pair[1]:
                raise RuntimeError(f"invalid octree probability metrics for case {case}")
            probability_rows.append(pair)
    if not rows:
        raise RuntimeError("no cases to score")
    a = np.array(rows)
    result = {"cases_count": len(rows), "oct_rel": float(np.median(a[:, 0])), "gs_rel": float(np.median(a[:, 1])),
              "oct_rel_max": float(a[:, 0].max()), "gs_rel_max": float(a[:, 1].max()),
              "oct_s": float(np.median(a[:, 2])), "gs_s": float(np.median(a[:, 3])),
              "resident_mb": float(a[0, 4]), "peak_mb": float(a[:, 5].max()), "pos": float(np.median(a[:, 6])),
              "col": float(np.median(a[:, 7])), "opa": float(np.median(a[:, 8])), "lsc": float(np.median(a[:, 9]))}
    if probability_rows:
        if len(probability_rows) != len(rows):
            raise RuntimeError("octree probability metrics were present for only some cases")
        result["oct_prob_tv_mean_median"] = float(np.median([r[0] for r in probability_rows]))
        result["oct_prob_tv_max"] = float(max(r[1] for r in probability_rows))
    return result


def baseline_fp16(args, cases):
    """The deployed model's weights (block Linears rounded to float16), run in float32, against the float32 references."""
    oct_model, gs_model = load_models(args.weights, args.sdk, half_blocks=True)
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    oct_err, gs_err = [], []
    for case in cases:
        cond = torch.from_numpy(np.fromfile(case / "cond.bin", dtype=np.float32).reshape(1, -1, 16))
        anchors = torch.from_numpy(np.fromfile(case / "anchors.bin", dtype=np.float32).reshape(1, -1, 3))
        with torch.no_grad():
            o = oct_model(anchors, torch.tensor([float(1 << LEVEL)]), cond)[0].numpy()
            g = gs_model(anchors, cond)[0].numpy()
        oct_err.append(rel(o, np.fromfile(case / "ref_oct.bin", dtype=np.float32).reshape(o.shape)))
        gs_err.append(rel(g, np.fromfile(case / "ref_gs.bin", dtype=np.float32).reshape(g.shape)))
    return {"oct_rel": float(np.median(oct_err)), "gs_rel": float(np.median(gs_err))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calib", type=int, default=16)
    parser.add_argument("--holdout", type=int, default=6)
    parser.add_argument("--anchors", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--tiers", default="8:64:rtn,6:64:rtn,5:64:rtn,4:64:rtn,3:64:rtn,2:32:rtn")
    parser.add_argument("--int8", action="store_true", help="also score the integer (AVX-VNNI) kernel for each tier")
    parser.add_argument("--keep", action="store_true", help="keep the tier exports (they are large)")
    parser.add_argument("--preserve-octree", action="store_true", help="leave octree matrices in float32")
    parser.add_argument("--exports", type=Path, help="directory for tier exports (default: --work)")
    args = parser.parse_args()

    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    cases = make_cases(args, work)
    keep, tensors = parse_manifest(args.export)
    results = {"baseline_fp16_weights": baseline_fp16(args, cases),
               "float32": score(args.run, args.export, cases, args.threads), "tiers": []}
    print("float32", results["float32"], "fp16 weights", results["baseline_fp16_weights"], flush=True)
    hessians = collect_hessians(args, work)
    for spec in args.tiers.split(","):
        bits, group, method = spec.split(":")
        bits, group = int(bits), int(group)
        plan = uniform_plan(tensors, bits, group, args.preserve_octree)
        suffix = "_octree_float" if args.preserve_octree else ""
        folder = (args.exports or work) / f"tier_{bits}_{group}_{method}{suffix}"
        write_tier(args.export, folder, keep, tensors, plan, hessians, method)
        row = {"label": f"{bits}-bit g{group} {method}{' octree-float' if args.preserve_octree else ''}",
               "bits": bits, "group": group, "method": method, "octree_preserved_float32": args.preserve_octree,
               "matrices": len(plan), **score(args.run, folder, cases, args.threads, "float")}
        if args.int8:
            row["int8"] = score(args.run, folder, cases, args.threads, "int8")
        results["tiers"].append(row)
        print(row, flush=True)
        args.out.write_text(json.dumps(results, indent=1))
        if not args.keep:
            for f in folder.iterdir():
                f.unlink()
            folder.rmdir()
    args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    sys.exit(main())

"""Cached sensitivity of octree child probabilities to quantizing one matrix at a time.

The objective is mean squared per-location total variation, evaluated at sampled anchors from selected calibration
latents. Those locations need not be parent-cell centers visited during sampling. It is an allocation proxy, not a
full-tree or visual-quality metric. The cache binds the listed input bytes and selected
settings; it does not prove the provenance of the supplied calibration matrix or bind every runtime dependency.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from eval_octree import digest


def squared_child_probability_variation(reference, candidate):
    if reference.shape != candidate.shape or reference.ndim < 1 or reference.shape[-1] != 8 or reference.numel() == 0:
        raise ValueError("expected matching nonempty eight-child logit rows")
    p, q = torch.softmax(reference, dim=-1), torch.softmax(candidate, dim=-1)
    variation = 0.5 * (p - q).abs().sum(dim=-1)
    result = variation.square().mean()
    if not torch.isfinite(result):
        raise ValueError("non-finite octree probability sensitivity")
    return result


def fingerprint(args, latent_files, code_files, level):
    inputs = [("export_manifest", args.export / "manifest.txt"), ("export_weights", args.export / "weights.bin"),
              ("config", args.weights / "config.json"), ("octree_weights", args.weights / "decoder.safetensors"),
              ("gaussian_weights", args.weights / "decoder_gs.safetensors"), ("calibration", args.work / "hessians.pt")]
    inputs += [(f"latent_{i}", path) for i, path in enumerate(latent_files)]
    inputs += [(f"sdk/{path.relative_to(args.sdk).as_posix()}", path)
               for path in sorted((args.sdk / "idol").rglob("*.py"))]
    inputs += [(f"code/{path.name}", path) for path in code_files]
    if not latent_files or not any(name.startswith("sdk/") for name, _ in inputs):
        raise ValueError("missing calibration latents or development-kit source")
    fields = {"files": {name: digest(path) for name, path in inputs},
              "settings": {"anchors": args.anchors, "level": level, "holdout": args.holdout,
                           "calibration": args.calib, "sensitivity_latents": args.sens_latents,
                           "device": args.device, "quant_device": os.environ.get("IDOL_QUANT_DEVICE", "cpu"),
                           "omp_threads": os.environ.get("OMP_NUM_THREADS"), "python": sys.version,
                           "torch": str(torch.__version__), "numpy": np.__version__}}
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), fields


def load_cache(path, expected_fingerprint):
    if not path.exists():
        return {"version": 1, "fingerprint": expected_fingerprint, "status": "partial", "values": {}}
    cache = json.loads(path.read_text(encoding="utf-8"))
    if (cache.get("version") != 1 or cache.get("fingerprint") != expected_fingerprint or
            cache.get("status") not in ("partial", "complete") or not isinstance(cache.get("values"), dict)):
        raise ValueError("octree probability sensitivity cache does not match current inputs; use a different work directory")
    return cache


def save_cache(path, cache):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(cache, stream, indent=1)
    os.replace(temp, path)

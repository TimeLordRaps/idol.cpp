"""Approximate orthographic Gaussian-splat previews and compare their pixels.

This diagnostic renderer is not the upstream Idol renderer. It projects each 3D covariance into a fixed camera,
truncates at three standard deviations, applies a half-pixel variance floor, and composites front to back against
black. Camera framing and rasterization choices affect the results; this is not ground-truth reconstruction quality.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial.transform import Rotation
from compare_splats import digest


def read_splats(path):
    v = PlyData.read(str(path))["vertex"].data
    required = ["x", "y", "z", "opacity"] + [f"f_dc_{i}" for i in range(3)] + \
               [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    if not len(v) or any(name not in v.dtype.names for name in required):
        raise ValueError("missing Gaussian properties")
    a = np.column_stack([v[name] for name in required]).astype(np.float64)
    if not np.isfinite(a).all():
        raise ValueError("non-finite Gaussian properties")
    positions = a[:, :3]
    opacity = 1 / (1 + np.exp(-np.clip(a[:, 3], -80, 80)))
    colors = np.clip(0.5 + 0.28209479177387814 * a[:, 4:7], 0, 1)
    scales = np.exp(a[:, 7:10])
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("invalid Gaussian scales")
    # PLY quaternions are stored w,x,y,z; SciPy accepts x,y,z,w.
    rotation = Rotation.from_quat(a[:, [11, 12, 13, 10]]).as_matrix()
    covariance = (rotation * scales[:, None, :] ** 2) @ np.swapaxes(rotation, 1, 2)
    return positions, opacity, colors, covariance


def render(data, axes, size=128, extent=0.75, depth_sign=1):
    positions, opacity, colors, covariance = data
    horizontal, vertical, depth = axes
    scale = size / (2 * extent)
    centers = positions[:, [horizontal, vertical]] * scale + size / 2
    projected = covariance[:, [horizontal, vertical]][:, :, [horizontal, vertical]] * scale**2
    projected[:, 0, 0] += 0.25
    projected[:, 1, 1] += 0.25
    inverse = np.linalg.inv(projected)
    radii = 3 * np.sqrt(np.diagonal(projected, axis1=1, axis2=2))
    rgb = np.zeros((size, size, 3), dtype=np.float64)
    transmittance = np.ones((size, size), dtype=np.float64)
    # Positive depth_sign places the camera at positive depth; negative reverses the viewing direction.
    for i in np.argsort(-depth_sign * positions[:, depth]):
        x, y = centers[i]
        rx, ry = radii[i]
        x0, x1 = max(0, math.floor(x-rx)), min(size, math.ceil(x+rx)+1)
        y0, y1 = max(0, math.floor(y-ry)), min(size, math.ceil(y+ry)+1)
        if x0 >= x1 or y0 >= y1:
            continue
        dx = np.arange(x0, x1)[None, :] + 0.5 - x
        dy = np.arange(y0, y1)[:, None] + 0.5 - y
        q = inverse[i, 0, 0]*dx*dx + 2*inverse[i, 0, 1]*dx*dy + inverse[i, 1, 1]*dy*dy
        alpha = np.where(q <= 9, np.minimum(0.999, opacity[i] * np.exp(-0.5*q)), 0)
        trans = transmittance[y0:y1, x0:x1]
        rgb[y0:y1, x0:x1] += (trans * alpha)[:, :, None] * colors[i]
        trans *= 1 - alpha
    return rgb, 1 - transmittance


def metrics(reference, candidate):
    a, aa = reference
    b, ba = candidate
    mse = float(np.mean((a-b)**2))
    left, right = aa >= 0.5, ba >= 0.5
    union = np.count_nonzero(left | right)
    foreground = (aa >= 1/255) | (ba >= 1/255)
    foreground_mae = float(np.mean(np.abs(a[foreground] - b[foreground]))) if foreground.any() else 0.0
    white_a, white_b = a + (1-aa)[:, :, None], b + (1-ba)[:, :, None]
    return {"rgb_mean_absolute_difference": float(np.mean(np.abs(a-b))), "rgb_mean_squared_difference": mse,
            "foreground_rgb_mean_absolute_difference": foreground_mae,
            "white_background_rgb_mean_absolute_difference": float(np.mean(np.abs(white_a-white_b))),
            "rgb_peak_signal_to_noise_ratio_db": None if mse == 0 else float(-10 * np.log10(mse)),
            "alpha_mean_absolute_difference": float(np.mean(np.abs(aa-ba))),
            "alpha_threshold_0_5_iou": float(np.count_nonzero(left & right) / union) if union else 1.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--extent", type=float, default=0.75)
    args = parser.parse_args()
    if not 16 <= args.size <= 1024 or not math.isfinite(args.extent) or args.extent <= 0:
        parser.error("invalid preview dimensions")
    source = {"reference": digest(args.reference), "candidate": digest(args.candidate), "tool": digest(Path(__file__))}
    a, b = read_splats(args.reference), read_splats(args.candidate)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    result = {"scope": __doc__, "input_sha256": source, "size": args.size, "extent": args.extent, "views": {}}
    for name, axes, sign in (("xy-front", (0, 1, 2), 1), ("xy-back", (0, 1, 2), -1),
                             ("xz-front", (0, 2, 1), 1), ("xz-back", (0, 2, 1), -1),
                             ("yz-front", (1, 2, 0), 1), ("yz-back", (1, 2, 0), -1)):
        left = render(a, axes, args.size, args.extent, sign)
        right = render(b, axes, args.size, args.extent, sign)
        result["views"][name] = metrics(left, right)
        for label, (rgb, _) in (("reference", left), ("candidate", right)):
            Image.fromarray(np.uint8(np.clip(rgb[::-1] * 255 + 0.5, 0, 255))).save(args.out_dir / f"{name}-{label}.png")
    after = {"reference": digest(args.reference), "candidate": digest(args.candidate), "tool": digest(Path(__file__))}
    if source != after:
        raise RuntimeError("inputs changed during rendering")
    result["hashes_unchanged_after_run"] = True
    with (args.out_dir / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()

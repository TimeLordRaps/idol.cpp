"""Compare two generated Gaussian point clouds, without registering or rendering them.

Positions are in the exported object's coordinate system. Distances use nearest Gaussian centers; occupancy and
orthographic projections use centers only, not Gaussian footprints, opacity compositing, or a ground-truth surface.
Random sampling can affect every metric. Compare independent full-precision seeds as a noise baseline.
Optional index-paired measurements require the caller to establish that both outputs use the same point ordering;
matching point counts do not establish correspondence.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    vertex = PlyData.read(str(path))["vertex"].data
    names = ("x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2")
    if not len(vertex) or any(name not in vertex.dtype.names for name in names):
        raise ValueError("expected nonempty Gaussian vertices with position, opacity and color fields")
    xyz = np.column_stack([vertex[n] for n in names[:3]]).astype(np.float64)
    raw_opacity = np.asarray(vertex["opacity"], dtype=np.float64)
    color = np.column_stack([vertex[n] for n in names[4:]]).astype(np.float64) * 0.28209479177387814 + 0.5
    if not np.isfinite(xyz).all() or not np.isfinite(raw_opacity).all() or not np.isfinite(color).all():
        raise ValueError("non-finite Gaussian data")
    opacity = 1 / (1 + np.exp(-np.clip(raw_opacity, -80, 80)))
    return xyz, opacity, color


def occupied(points, resolution, dimensions):
    # Fixed object coordinate frame shared by both inputs, with no clipping and no alignment.
    cells = np.floor(points[:, dimensions] * resolution).astype(np.int64)
    return set(map(tuple, cells))


def iou(left, right):
    union = len(left | right)
    return len(left & right) / union if union else 1.0


def compare(reference, candidate, resolutions=(16, 32, 64), paired=False):
    a, aw, ac = reference
    b, bw, bc = candidate
    if paired and len(a) != len(b):
        raise ValueError("paired comparison requires equal point counts")
    da, ia = cKDTree(b).query(a, workers=1)
    db, ib = cKDTree(a).query(b, workers=1)
    # Directed measurements expose missing and extra structure separately.
    def direction(distance, weight, color, matched):
        return {"mean_center_distance": float(distance.mean()),
                "p95_center_distance": float(np.percentile(distance, 95)),
                "opacity_weighted_mean_center_distance": float(np.average(distance, weights=weight)),
                "nearest_color_mean_absolute_difference": float(np.abs(color - matched).mean())}
    result = {"reference_count": len(a), "candidate_count": len(b),
            "reference_to_candidate": direction(da, aw, ac, bc[ia]),
            "candidate_to_reference": direction(db, bw, bc, ac[ib]),
            "center_voxel_iou": {str(r): iou(occupied(a, r, [0, 1, 2]), occupied(b, r, [0, 1, 2]))
                                 for r in resolutions},
            "orthographic_center_occupancy_iou": {
                str(r): {name: iou(occupied(a, r, axes), occupied(b, r, axes))
                         for name, axes in (("xy", [0, 1]), ("xz", [0, 2]), ("yz", [1, 2]))}
                for r in resolutions}}
    if paired:
        distance = np.linalg.norm(a - b, axis=1)
        result["index_paired"] = {
            "assumption": "point indices correspond; equal counts alone do not establish this",
            "mean_center_distance": float(distance.mean()),
            "p95_center_distance": float(np.percentile(distance, 95)),
            "max_center_distance": float(distance.max()),
            "opacity_mean_absolute_difference": float(np.abs(aw - bw).mean()),
            "color_mean_absolute_difference": float(np.abs(ac - bc).mean()),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--paired", action="store_true", help="also compare matching indices; only use when point correspondence is established")
    args = parser.parse_args()
    before = {"reference": digest(args.reference), "candidate": digest(args.candidate), "tool": digest(Path(__file__))}
    measurements = compare(load(args.reference), load(args.candidate), paired=args.paired)
    after = {"reference": digest(args.reference), "candidate": digest(args.candidate), "tool": digest(Path(__file__))}
    if before != after:
        raise RuntimeError("inputs changed during comparison")
    result = {"scope": __doc__, "input_sha256": before, "hashes_unchanged_after_run": True, "measurements": measurements}
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()

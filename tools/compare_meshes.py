"""Compare sampled surfaces of two meshes in their existing coordinates, without alignment.

The meshes may themselves be approximations extracted from Gaussian centers. Surface distances are finite Monte Carlo
estimates, not ground-truth reconstruction accuracy. Compare full-precision alternate seeds as a sampling baseline.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from compare_splats import digest


def sample_surface(vertices, faces, count, seed):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all() or
            faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or
            faces.min() < 0 or faces.max() >= len(vertices) or count < 1):
        raise ValueError("invalid triangle mesh")
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1) / 2
    total = areas.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("mesh has no finite surface area")
    rng = np.random.default_rng(seed)
    selected = triangles[rng.choice(len(faces), size=count, p=areas / total)]
    uv = rng.random((count, 2))
    reflected = uv.sum(axis=1) > 1
    uv[reflected] = 1 - uv[reflected]
    points = selected[:, 0] + uv[:, :1] * (selected[:, 1]-selected[:, 0]) + uv[:, 1:] * (selected[:, 2]-selected[:, 0])
    return points, float(total)


def compare(a, b, count=20000, seed=23):
    pa, area_a = sample_surface(a.vertices, a.faces, count, seed)
    pb, area_b = sample_surface(b.vertices, b.faces, count, seed)
    noise, _ = sample_surface(a.vertices, a.faces, count, seed + 1)
    da = cKDTree(pb).query(pa, workers=1)[0]
    db = cKDTree(pa).query(pb, workers=1)[0]
    noise_distance = cKDTree(noise).query(pa, workers=1)[0]
    diagonal = float(np.linalg.norm(np.ptp(np.asarray(a.vertices, dtype=np.float64), axis=0)))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("reference mesh has no finite bounding-box diagonal")
    symmetric_mean = float((da.mean() + db.mean()) / 2)
    # Distance to triangles avoids treating finite target-surface sampling density as geometric error.
    def triangle_distances(mesh, points):
        return np.concatenate([trimesh.proximity.closest_point(mesh, batch)[1]
                               for batch in np.array_split(points, max(1, (len(points) + 511) // 512))])
    triangle_a, triangle_b = triangle_distances(b, pa), triangle_distances(a, pb)
    triangle_mean = float((triangle_a.mean() + triangle_b.mean()) / 2)
    triangle_scores = {}
    for fraction in (0.001, 0.005, 0.01):
        recall = float(np.mean(triangle_a <= fraction * diagonal))
        precision = float(np.mean(triangle_b <= fraction * diagonal))
        triangle_scores[str(fraction)] = {'distance_threshold': fraction * diagonal,
            'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    scores = {}
    for fraction in (0.005, 0.01):
        threshold = fraction * diagonal
        recall, precision = float(np.mean(da <= threshold)), float(np.mean(db <= threshold))
        baseline = float(np.mean(noise_distance <= threshold))
        scores[str(fraction)] = {"distance_threshold": threshold, "precision": precision, "recall": recall,
                                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                                "reference_resampling_recall": baseline}
    def direction(d):
        return {"mean_surface_distance": float(d.mean()), "p95_surface_distance": float(np.percentile(d, 95))}
    return {"surface_samples_per_mesh": count, "sample_seed": seed, "reference_area": area_a,
            "sampled_point_to_triangle": {
                "method": "Area-weighted sampled points to the closest target-mesh triangles; no alignment. "
                          "F1 is the harmonic mean of precision and recall within the recorded distance threshold. "
                          "p95 is the 95th percentile; the maximum is over sampled source points, not the continuous surface.",
                "symmetric_mean_distance": triangle_mean,
                "symmetric_mean_over_reference_diagonal": triangle_mean / diagonal,
                "reference_to_candidate_mean": float(triangle_a.mean()),
                "candidate_to_reference_mean": float(triangle_b.mean()),
                "symmetric_p95_distance": float(max(np.percentile(triangle_a, 95), np.percentile(triangle_b, 95))),
                "maximum_sampled_distance": float(max(triangle_a.max(), triangle_b.max())),
                "f_scores_by_reference_diagonal_fraction": triangle_scores},
            "distance_units": "input model-coordinate units; no alignment or rescaling",
            "reference_bounding_box_diagonal": diagonal,
            "symmetric_mean_sample_distance": symmetric_mean,
            "symmetric_mean_distance_over_reference_diagonal": symmetric_mean / diagonal,
            "symmetric_p95_sample_distance": float(max(np.percentile(da, 95), np.percentile(db, 95))),
            "sampled_hausdorff_distance": float(max(da.max(), db.max())),
            "surface_f_scores_by_reference_diagonal_fraction": scores,
            "candidate_area": area_b, "reference_to_candidate": direction(da),
            "candidate_to_reference": direction(db), "reference_vertices": len(a.vertices),
            "same_reference_mesh_different_sample_seed": direction(noise_distance),
            "candidate_vertices": len(b.vertices), "reference_faces": len(a.faces), "candidate_faces": len(b.faces),
            "reference_watertight": bool(a.is_watertight), "candidate_watertight": bool(b.is_watertight)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    inputs = {"reference": digest(args.reference), "candidate": digest(args.candidate), "tool": digest(Path(__file__))}
    a = trimesh.load_mesh(str(args.reference), process=False)
    b = trimesh.load_mesh(str(args.candidate), process=False)
    result = {"scope": __doc__, "input_sha256": inputs, "measurements": compare(a, b, args.samples, args.seed)}
    after = {"reference": digest(args.reference), "candidate": digest(args.candidate), "tool": digest(Path(__file__))}
    if inputs != after:
        raise RuntimeError("mesh inputs changed during comparison")
    result["hashes_unchanged_after_run"] = True
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()

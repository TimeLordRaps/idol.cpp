"""Checks deterministic surface sampling and the response to displaced meshes."""
import importlib.util
import sys
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in ("numpy", "scipy", "plyfile", "trimesh", "rtree")):
    print("skipped: needs numpy, scipy, plyfile, trimesh and rtree")
    sys.exit(77)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import numpy as np
import trimesh
from compare_meshes import compare, sample_surface

vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
faces = np.array([[0, 1, 2]])
points, area = sample_surface(vertices, faces, 100, 3)
assert area == 0.5 and np.array_equal(points, sample_surface(vertices, faces, 100, 3)[0])
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
same = compare(mesh, mesh, 100, 3)
assert same["reference_to_candidate"]["mean_surface_distance"] == 0
assert same['symmetric_mean_sample_distance'] == 0
assert same['surface_f_scores_by_reference_diagonal_fraction']['0.01']['f1'] == 1
assert same['sampled_point_to_triangle']['symmetric_mean_distance'] < 1e-14
assert same['same_reference_mesh_different_sample_seed']['mean_surface_distance'] > 0
shifted = trimesh.Trimesh(vertices=vertices + [0, 0, 1], faces=faces, process=False)
changed = compare(mesh, shifted, 100, 3)
assert abs(changed["reference_to_candidate"]["mean_surface_distance"] - 1) < 1e-12
assert abs(changed['symmetric_mean_sample_distance'] - 1) < 1e-12
assert abs(changed['sampled_point_to_triangle']['symmetric_mean_distance'] - 1) < 1e-12
assert abs(changed['symmetric_mean_distance_over_reference_diagonal'] - 1 / np.sqrt(2)) < 1e-12
assert changed['surface_f_scores_by_reference_diagonal_fraction']['0.01']['f1'] == 0
scaled = compare(trimesh.Trimesh(vertices=vertices * 10, faces=faces, process=False),
                 trimesh.Trimesh(vertices=(vertices + [0, 0, 1]) * 10, faces=faces, process=False), 100, 3)
assert abs(scaled['symmetric_mean_distance_over_reference_diagonal'] - changed['symmetric_mean_distance_over_reference_diagonal']) < 1e-12
assert abs(scaled['sampled_point_to_triangle']['symmetric_mean_over_reference_diagonal'] -
           changed['sampled_point_to_triangle']['symmetric_mean_over_reference_diagonal']) < 1e-12
assert not changed["reference_watertight"]
square = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
first = trimesh.Trimesh(vertices=square, faces=[[0, 1, 2], [0, 2, 3]], process=False)
second = trimesh.Trimesh(vertices=square, faces=[[0, 1, 3], [1, 2, 3]], process=False)
retessellated = compare(first, second, 100, 3)
assert retessellated['sampled_point_to_triangle']['symmetric_mean_distance'] < 1e-14
assert retessellated['sampled_point_to_triangle']['f_scores_by_reference_diagonal_fraction']['0.001']['f1'] == 1
print("mesh comparison synthetic checks passed")

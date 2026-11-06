"""Checks that the construction metrics react to known geometric and color changes."""
import importlib.util
import sys
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in ("numpy", "scipy", "plyfile")):
    print("skipped: needs numpy, scipy and plyfile")
    sys.exit(77)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import numpy as np
from compare_splats import compare

points = np.array([[0.01, 0.01, 0.01], [0.51, 0.51, 0.51]])
weights = np.array([0.2, 0.8])
colors = np.full((2, 3), 0.5)
base = (points, weights, colors)
same = compare(base, base, (16,))
assert same["reference_to_candidate"]["mean_center_distance"] == 0
assert same["center_voxel_iou"]["16"] == 1
assert all(value == 1 for value in same["orthographic_center_occupancy_iou"]["16"].values())
shifted = compare(base, (points + [0.125, 0, 0], weights, colors + 0.1), (16,))
assert abs(shifted["reference_to_candidate"]["mean_center_distance"] - 0.125) < 1e-12
assert abs(shifted["candidate_to_reference"]["p95_center_distance"] - 0.125) < 1e-12
assert abs(shifted["reference_to_candidate"]["nearest_color_mean_absolute_difference"] - 0.1) < 1e-12
assert shifted["center_voxel_iou"]["16"] == 0
assert shifted["orthographic_center_occupancy_iou"]["16"]["yz"] == 1
paired = compare(base, (points + [0.125, 0, 0], weights + 0.1, colors + 0.2), (16,), paired=True)
assert abs(paired["index_paired"]["mean_center_distance"] - 0.125) < 1e-12
assert abs(paired["index_paired"]["opacity_mean_absolute_difference"] - 0.1) < 1e-12
assert abs(paired["index_paired"]["color_mean_absolute_difference"] - 0.2) < 1e-12
reordered = compare(base, (points[::-1], weights[::-1], colors[::-1]), (16,), paired=True)
assert reordered["center_voxel_iou"]["16"] == 1
assert reordered["index_paired"]["mean_center_distance"] > 0
try:
    compare(base, (points[:1], weights[:1], colors[:1]), (16,), paired=True)
    raise AssertionError("paired comparison accepted unequal point counts")
except ValueError:
    pass
print("construction metric synthetic checks passed")

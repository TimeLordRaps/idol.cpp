"""Checks preview projection, alpha, and comparison on a known Gaussian."""
import importlib.util
import sys
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in ("numpy", "scipy", "plyfile", "PIL")):
    print("skipped: needs numpy, scipy, plyfile and Pillow")
    sys.exit(77)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import numpy as np
from render_splats_preview import metrics, render

positions = np.array([[0.0, 0.0, 0.0]])
opacity = np.array([0.8])
colors = np.array([[1.0, 0.0, 0.0]])
covariance = np.array([np.eye(3) * 0.01])
preview = render((positions, opacity, colors, covariance), (0, 1, 2), 32, 1)
assert preview[0][16, 16, 0] > 0 and preview[0][16, 16, 1] == 0
assert 0 < preview[1].max() <= 0.8
same = metrics(preview, preview)
assert same["rgb_mean_squared_difference"] == 0 and same["alpha_threshold_0_5_iou"] == 1
shifted = render((positions + [0.5, 0, 0], opacity, colors, covariance), (0, 1, 2), 32, 1)
assert metrics(preview, shifted)["rgb_mean_absolute_difference"] > 0
assert metrics(preview, shifted)["alpha_threshold_0_5_iou"] < 1
two = (np.array([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]]), np.array([0.8, 0.8]),
       np.array([[1., 0., 0.], [0., 0., 1.]]), np.array([np.eye(3)*0.01, np.eye(3)*0.01]))
front = render(two, (0, 1, 2), 32, 1, 1)[0][16, 16]
back = render(two, (0, 1, 2), 32, 1, -1)[0][16, 16]
assert front[0] > front[2] and back[2] > back[0]
print("Gaussian preview synthetic checks passed")

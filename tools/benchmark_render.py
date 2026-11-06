"""Accelerated version of the existing diagnostic orthographic renderer.

The arithmetic and camera convention match render_splats_preview; this remains
an approximate diagnostic, not the upstream Idol renderer or ground truth.
"""
import math
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def composite(centers, inverse, radii, opacity, colors, order, size):
    rgb = np.zeros((size, size, 3), dtype=np.float64)
    transmittance = np.ones((size, size), dtype=np.float64)
    for index in order:
        x, y = centers[index, 0], centers[index, 1]
        rx, ry = radii[index, 0], radii[index, 1]
        x0, x1 = max(0, math.floor(x-rx)), min(size, math.ceil(x+rx)+1)
        y0, y1 = max(0, math.floor(y-ry)), min(size, math.ceil(y+ry)+1)
        for row in range(y0, y1):
            dy = row + 0.5 - y
            for column in range(x0, x1):
                dx = column + 0.5 - x
                q = inverse[index, 0, 0]*dx*dx + 2*inverse[index, 0, 1]*dx*dy + inverse[index, 1, 1]*dy*dy
                alpha = min(0.999, opacity[index] * math.exp(-0.5*q)) if q <= 9 else 0.0
                weight = transmittance[row, column] * alpha
                for channel in range(3):
                    rgb[row, column, channel] += weight * colors[index, channel]
                transmittance[row, column] *= 1 - alpha
    return rgb, 1 - transmittance


def render(data, axes, size=256, extent=0.75, depth_sign=1):
    positions, opacity, colors, covariance = data
    horizontal, vertical, depth = axes
    scale = size / (2 * extent)
    centers = positions[:, [horizontal, vertical]] * scale + size / 2
    projected = covariance[:, [horizontal, vertical]][:, :, [horizontal, vertical]] * scale**2
    projected[:, 0, 0] += 0.25
    projected[:, 1, 1] += 0.25
    return composite(centers, np.linalg.inv(projected), 3*np.sqrt(np.diagonal(projected, axis1=1, axis2=2)),
                     opacity, colors, np.argsort(-depth_sign * positions[:, depth]), size)


VIEWS = [('xy-front',(0,1,2),1),('xy-back',(0,1,2),-1),('xz-front',(0,2,1),1),
         ('xz-back',(0,2,1),-1),('yz-front',(1,2,0),1),('yz-back',(1,2,0),-1)]

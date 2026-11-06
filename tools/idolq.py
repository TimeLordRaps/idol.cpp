"""Group-wise weight quantization for Idol decoders in the layout read by the native runtime.

A weight matrix W (out x in) is cut into groups of `group` inputs per output row. Each group gets a scale and a minimum
(both rounded to half precision, as stored) and each weight an unsigned code of `bits` bits: w ~ min + code * scale.
Two ways to choose the codes:

  * rtn   Round to nearest, with the scale and minimum fitted per group to minimise the importance-weighted error
          (a clipping search followed by a least-squares refit) instead of taking the plain min and max.
  * gptq  The same, but weights are rounded one input at a time and each rounding error is pushed onto the inputs not
          yet rounded, using the second moment of the layer's inputs on calibration images (Frantar et al., GPTQ).

The blob layout is documented in src/kernels.hpp and docs/format.html. The functions here need numpy and torch.
"""

import os

import numpy as np
import torch


ALIGN = 64
HEADER = 64
BITS = (2, 3, 4, 5, 6, 8)
PLANES = {2: (2,), 3: (2, 1), 4: (4,), 5: (4, 1), 6: (4, 2), 8: (8,)}
CLIP_GRID = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55)


def levels(bits):
    return (1 << bits) - 1


def to_half(x):
    return x.half().float()


# ------------------------------------------------------------------------------------------------ group parameters
def _quantize_with(w, scale, minimum, bits):
    """Codes and reconstruction for w (..., G) given per-group scale and minimum (..., 1)."""
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.clamp(torch.round((w - minimum) / safe), 0, levels(bits))
    codes = torch.where(scale > 0, codes, torch.zeros_like(codes))
    return codes, minimum + codes * scale


def fit_groups(w, importance, bits, refit=True):
    """Scale and minimum per group. w is (R, ng, G); importance is (ng, G) or None. Returns half-rounded (R, ng, 1)."""
    lo, hi = w.amin(-1, keepdim=True), w.amax(-1, keepdim=True)
    centre = (lo + hi) / 2
    weight = importance[None] if importance is not None else torch.ones(1, *w.shape[1:])
    best_err = None
    best = None
    for p in CLIP_GRID:
        a, b = centre - p * (centre - lo), centre + p * (hi - centre)
        scale, minimum = to_half((b - a) / levels(bits)), to_half(a)
        _, rec = _quantize_with(w, scale, minimum, bits)
        err = (weight * (w - rec) ** 2).sum(-1, keepdim=True)
        if best is None:
            best_err, best = err, (scale, minimum)
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best = (torch.where(better, scale, best[0]), torch.where(better, minimum, best[1]))
    scale, minimum = best
    if refit:
        # With the codes fixed, the best scale and minimum are the weighted least-squares line through (code, w).
        codes, _ = _quantize_with(w, scale, minimum, bits)
        sw = weight.expand_as(w)
        s0, s1, s2 = sw.sum(-1, keepdim=True), (sw * codes).sum(-1, keepdim=True), (sw * codes * codes).sum(-1, keepdim=True)
        t0, t1 = (sw * w).sum(-1, keepdim=True), (sw * codes * w).sum(-1, keepdim=True)
        det = s0 * s2 - s1 * s1
        ok = det.abs() > 1e-12 * s0 * s2.clamp_min(1e-30)
        new_scale = to_half(torch.where(ok, (s0 * t1 - s1 * t0) / det.where(ok, torch.ones_like(det)), scale))
        new_min = to_half(torch.where(ok, (s2 * t0 - s1 * t1) / det.where(ok, torch.ones_like(det)), minimum))
        _, rec = _quantize_with(w, new_scale, new_min, bits)
        err = (weight * (w - rec) ** 2).sum(-1, keepdim=True)
        better = err < best_err
        scale, minimum = torch.where(better, new_scale, scale), torch.where(better, new_min, minimum)
    return scale, minimum


def quantize_rtn(weight, bits, group, importance=None):
    """weight (out, in) -> codes (out, in) uint8, scale and minimum (out, in / group) float32 (half-rounded)."""
    out, inp = weight.shape
    w = weight.reshape(out, inp // group, group).float()
    imp = importance.reshape(inp // group, group).float() if importance is not None else None
    scale, minimum = fit_groups(w, imp, bits)
    codes, _ = _quantize_with(w, scale, minimum, bits)
    return codes.reshape(out, inp).to(torch.uint8), scale[..., 0], minimum[..., 0]


def quantize_gptq(weight, hessian, bits, group, damp=0.01, block=128):
    """Same outputs as quantize_rtn, with error feedback through the inverse of `hessian` (in x in, mean of x x^T)."""
    out, inp = weight.shape
    # IDOL_QUANT_DEVICE=cuda runs the column loop on the GPU, about 10x faster; the results come back on the CPU.
    device = torch.device(os.environ.get("IDOL_QUANT_DEVICE", "cpu"))
    w = weight.clone().float().to(device)
    h = hessian.clone().double().to(device)
    dead = torch.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0
    h += damp * torch.mean(torch.diag(h)) * torch.eye(inp, dtype=torch.float64, device=device)
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True).float()
    diag = torch.diag(hessian).float().to(device)
    codes = torch.zeros(out, inp, dtype=torch.uint8, device=device)
    scales = torch.zeros(out, inp // group, device=device)
    mins = torch.zeros(out, inp // group, device=device)
    block = max(block, group) - max(block, group) % group
    for i1 in range(0, inp, block):
        i2 = min(i1 + block, inp)
        w1 = w[:, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        scale = minimum = None
        for i in range(i2 - i1):
            col = i1 + i
            if col % group == 0:
                g = w1[:, i:i + group].reshape(out, 1, group)
                s, m = fit_groups(g, diag[col:col + group].reshape(1, group), bits, refit=False)
                scale, minimum = s[:, 0, 0], m[:, 0, 0]
                scales[:, col // group], mins[:, col // group] = scale, minimum
            c, q = _quantize_with(w1[:, i], scale, minimum, bits)
            codes[:, col] = c.to(torch.uint8)
            err = (w1[:, i] - q) / hinv1[i, i]
            w1[:, i:] -= err[:, None] * hinv1[i, i:][None]
            err1[:, i] = err
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    return codes.cpu(), scales.cpu(), mins.cpu()


def dequantize(codes, scale, minimum, group):
    out, inp = codes.shape
    c = codes.float().reshape(out, inp // group, group)
    return (minimum[..., None] + c * scale[..., None]).reshape(out, inp)


# ------------------------------------------------------------------------------------------------ the blob
def pack_blob(codes, scale, minimum, bits, group):
    """Panel layout for the runtime: 16 output columns per panel, groups of rows, scales then bit planes."""
    out, inp = codes.shape
    panels = (out + 15) // 16
    pad = panels * 16 - out
    c = np.pad(codes.numpy().astype(np.uint8), ((0, pad), (0, 0)))
    s = np.pad(scale.numpy().astype(np.float32), ((0, pad), (0, 0)))
    m = np.pad(minimum.numpy().astype(np.float32), ((0, pad), (0, 0)))
    ng = inp // group
    # codes[panel, k, column]
    ck = c.reshape(panels, 16, inp).transpose(0, 2, 1)
    chunks = ck.reshape(panels, ng, group // 8, 8, 16)
    planes = []
    shift = 0
    for width in PLANES[bits]:
        per = 8 // width
        pc = (chunks >> shift) & ((1 << width) - 1)
        pc = pc.reshape(panels, ng, group // 8, width, per, 16).astype(np.uint16)
        packed = np.zeros((panels, ng, group // 8, width, 16), dtype=np.uint16)
        for sub in range(per):
            packed |= pc[:, :, :, :, sub, :] << (sub * width)
        planes.append(packed.astype(np.uint8))
        shift += width
    body = np.concatenate(planes, axis=3).reshape(panels, ng, group // 8 * bits * 16)
    sh = np.ascontiguousarray(s.reshape(panels, 16, ng).transpose(0, 2, 1).astype(np.float16)).view(np.uint8)
    mh = np.ascontiguousarray(m.reshape(panels, 16, ng).transpose(0, 2, 1).astype(np.float16)).view(np.uint8)
    sh, mh = sh.reshape(panels, ng, 32), mh.reshape(panels, ng, 32)
    blob = np.concatenate([sh, mh, body], axis=2)
    assert blob.shape[2] == HEADER + 2 * group * bits
    return blob.tobytes()


def blob_bytes(out, inp, bits, group):
    return ((out + 15) // 16) * (inp // group) * (HEADER + 2 * group * bits)


def matrix_names(tensors):
    return [name for name, array in tensors if array.ndim >= 2 and name.endswith("weight")]


# ------------------------------------------------------------------------------------------------ whole tensors
def quantize_tensor(array, bits, group, method, hessians=None, name=None):
    """Returns (codes, scale, minimum) for the whole matrix, row blocks using their own Hessian."""
    w = torch.from_numpy(np.array(array)).reshape(array.shape[0], -1).float()
    blocks = hessians[name] if hessians else [(0, None, None)]
    codes, scales, mins = [], [], []
    for r0, r1, h in sorted(blocks, key=lambda b: b[0]):
        part = w[r0:r1]
        if method == "gptq":
            c, s, m = quantize_gptq(part, h, bits, group)
        else:
            c, s, m = quantize_rtn(part, bits, group, torch.diag(h) if h is not None else None)
        codes.append(c)
        scales.append(s)
        mins.append(m)
    return torch.cat(codes), torch.cat(scales), torch.cat(mins)


def resident_bytes(tensors, plan, config):
    """Weight bytes the runtime holds for this plan: quantized blobs, padded biases, and the float32 rest."""
    total = 0
    for name, array in tensors:
        if name in plan and plan[name] is not None:
            bits, group = plan[name]
            total += blob_bytes(array.shape[0], int(np.prod(array.shape[1:])), bits, group)
        elif name.endswith("bias") and array.ndim == 1:
            total += ((array.shape[0] + 15) // 16) * 16 * 4
        else:
            total += array.size * 4
    return total

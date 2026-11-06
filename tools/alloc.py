"""Bit allocation by marginal return, shared by the search tools.

Given each weight matrix's error when quantized alone to each candidate (bits, group), give every matrix its cheapest
candidate, then repeatedly upgrade the matrix whose upgrade removes the most squared error per byte. Errors from
different matrices add roughly in quadrature, so the squared error is treated as additive. Every step is a plan; the
sequence is the allocation front.
"""

import heapq

import numpy as np

import idolq


def tensor_bytes(array, cand):
    return idolq.blob_bytes(array.shape[0], int(np.prod(array.shape[1:])), cand[0], cand[1])


def allocation_front(tensors, sens, usable):
    """tensors: [(name, ndarray)]; sens: name -> {"bits,group": squared error}; usable(array) -> candidates that fit.
    Returns [(resident bytes, plan)] from the cheapest plan to the richest, one upgrade per step."""
    arrays = dict(tensors)
    names = idolq.matrix_names(tensors)
    hulls = {}
    for name in names:
        points = sorted(((tensor_bytes(arrays[name], c), sens[name][f"{c[0]},{c[1]}"], c) for c in usable(arrays[name])),
                        key=lambda p: (p[0], p[1]))
        best = []
        for size, err, cand in points:  # keep points that lower the error
            if not best or err < best[-1][1]:
                best.append((size, err, cand))
        hull = []
        for p in best:  # lower convex hull in (size, error)
            while len(hull) >= 2 and (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) >= \
                    (p[1] - hull[-2][1]) * (hull[-1][0] - hull[-2][0]):
                hull.pop()
            hull.append(p)
        hulls[name] = hull
    level = {n: 0 for n in names}
    fixed = idolq.resident_bytes(tensors, {}, None) - sum(arrays[n].size * 4 for n in names)
    total = fixed + sum(hulls[n][0][0] for n in names)
    heap = []

    def push(name):
        i = level[name]
        if i + 1 < len(hulls[name]):
            (s0, e0, _), (s1, e1, _) = hulls[name][i], hulls[name][i + 1]
            heapq.heappush(heap, ((e1 - e0) / (s1 - s0), name, i))

    for n in names:
        push(n)
    front = [(total, {n: hulls[n][level[n]][2] for n in names})]
    while heap:
        _, name, i = heapq.heappop(heap)
        if level[name] != i:
            continue
        level[name] = i + 1
        total += hulls[name][i + 1][0] - hulls[name][i][0]
        push(name)
        front.append((total, {n: hulls[n][level[n]][2] for n in names}))
    return front

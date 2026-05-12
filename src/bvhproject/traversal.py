"""CPU traversal of a GPU-built LBVH for correctness validation."""

from __future__ import annotations
import numpy as np
from .oracle import _ray_aabb, _moller_trumbore


def traverse_lbvh(
    ray_origin:  np.ndarray,
    ray_dir:     np.ndarray,
    nodes:       np.ndarray,   # dtype NODE_DTYPE, shape (2N-1,)
    tris:        np.ndarray,   # dtype TRI_DTYPE,  shape (N,)  — original order
    root_idx:    int,
    t_max:       float = float('inf'),
) -> tuple[int, float]:
    """Return (original_prim_idx, t) for the nearest hit, or (-1, inf).

    node['prim_idx'] stores the original triangle index for leaves, -1 for
    internal nodes (the CUDA binary resolves the Morton-sort permutation).
    """
    origin    = ray_origin.astype(np.float64)
    direction = ray_dir.astype(np.float64)
    stack    = [root_idx]
    best_t   = t_max
    best_idx = -1

    while stack:
        idx  = stack.pop()
        node = nodes[idx]

        if not _ray_aabb(origin, direction,
                          node['aabb_min'].astype(np.float64),
                          node['aabb_max'].astype(np.float64), best_t):
            continue

        prim = int(node['prim_idx'])
        if prim >= 0:  # leaf
            tri = tris[prim]
            t   = _moller_trumbore(
                origin, direction,
                tri['v0'].astype(np.float64),
                tri['v1'].astype(np.float64),
                tri['v2'].astype(np.float64),
                t_max=best_t)
            if t is not None and t < best_t:
                best_t   = t
                best_idx = prim
        else:
            stack.append(int(node['left']))
            stack.append(int(node['right']))

    return best_idx, best_t

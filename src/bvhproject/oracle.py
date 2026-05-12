"""CPU SAH BVH oracle — standalone implementation for correctness comparison.

Implements the same Surface Area Heuristic split strategy as
asst3/src/cs248a_renderer/model/bvh.py but with no dependency on open3d,
slangpy, or cs248a-renderer.  Takes raw numpy triangle data (from bridge.py)
and produces a traversable BVH tree.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque


@dataclass
class _BVHNode:
    aabb_min: np.ndarray  # float32 [3]
    aabb_max: np.ndarray  # float32 [3]
    left:  int = -1   # child index, -1 = leaf
    right: int = -1
    prim_start: int = 0  # [prim_start, prim_end) in _SahOracle.prims_reordered
    prim_end:   int = 0

    @property
    def is_leaf(self):
        return self.left == -1


def _tri_aabb(v0, v1, v2):
    mn = np.minimum(np.minimum(v0, v1), v2) - 1e-6
    mx = np.maximum(np.maximum(v0, v1), v2) + 1e-6
    return mn.astype(np.float32), mx.astype(np.float32)


def _aabb_area(mn, mx):
    d = mx - mn
    return max(float(2.0 * (d[0]*d[1] + d[1]*d[2] + d[2]*d[0])), 0.0)


def _sah_cost(parent_area, l_mn, l_mx, l_count, r_mn, r_mx, r_count):
    if parent_area <= 0:
        return float('inf')
    return (_aabb_area(l_mn, l_mx) / parent_area * l_count +
            _aabb_area(r_mn, r_mx) / parent_area * r_count)


class SahOracle:
    """Standalone CPU SAH BVH for correctness comparison.

    Args:
        vertices: float32 (V, 3) — all vertices
        faces:    int     (F, 3) — triangle vertex indices
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self._build(vertices.astype(np.float32), np.asarray(faces, dtype=np.int64))

    def _build(self, verts, faces):
        n = len(faces)
        # Store per-triangle data
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        centroids  = (v0 + v1 + v2) / 3.0
        aabb_mins  = np.minimum(np.minimum(v0, v1), v2) - 1e-6
        aabb_maxs  = np.maximum(np.maximum(v0, v1), v2) + 1e-6
        orig_idxs  = np.arange(n, dtype=np.int64)

        # Work list: (prim indices for this node, depth)
        self._nodes: List[_BVHNode] = []
        self._v0 = v0
        self._v1 = v1
        self._v2 = v2
        self._prim_order: List[int] = []  # reordered prim indices

        # BFS build
        q = deque()
        # Each entry: (list of prim indices, depth)
        root_idxs = list(range(n))
        root_aabb_min = aabb_mins.min(axis=0)
        root_aabb_max = aabb_maxs.max(axis=0)
        root_node = _BVHNode(root_aabb_min, root_aabb_max)
        self._nodes.append(root_node)
        q.append((0, root_idxs))

        MAX_NODES = 4 * n + 16

        while q and len(self._nodes) < MAX_NODES:
            node_idx, prim_idxs = q.popleft()
            node = self._nodes[node_idx]

            if len(prim_idxs) <= 1:
                node.prim_start = len(self._prim_order)
                self._prim_order.extend(prim_idxs)
                node.prim_end = len(self._prim_order)
                continue

            # Find best SAH split
            parent_area = _aabb_area(node.aabb_min, node.aabb_max)
            best_cost = float('inf')
            best_split = None

            for axis in range(3):
                mn_all = aabb_mins[prim_idxs, axis].min()
                mx_all = aabb_maxs[prim_idxs, axis].max()
                if mx_all - mn_all < 1e-7:
                    continue
                for t in np.linspace(mn_all, mx_all, 16, endpoint=False)[1:]:
                    left_mask  = centroids[prim_idxs, axis] < t
                    right_mask = ~left_mask
                    lc, rc = left_mask.sum(), right_mask.sum()
                    if lc == 0 or rc == 0:
                        continue
                    l_mn = aabb_mins[prim_idxs][left_mask].min(axis=0)
                    l_mx = aabb_maxs[prim_idxs][left_mask].max(axis=0)
                    r_mn = aabb_mins[prim_idxs][right_mask].min(axis=0)
                    r_mx = aabb_maxs[prim_idxs][right_mask].max(axis=0)
                    cost = _sah_cost(parent_area, l_mn, l_mx, lc, r_mn, r_mx, rc)
                    if cost < best_cost:
                        best_cost = cost
                        best_split = (axis, t, l_mn, l_mx, r_mn, r_mx)

            if best_split is None or best_cost >= len(prim_idxs):
                # Make leaf
                node.prim_start = len(self._prim_order)
                self._prim_order.extend(prim_idxs)
                node.prim_end = len(self._prim_order)
                continue

            axis, t, l_mn, l_mx, r_mn, r_mx = best_split
            pis = np.array(prim_idxs, dtype=np.int64)
            left_idxs  = pis[centroids[pis, axis] < t].tolist()
            right_idxs = pis[centroids[pis, axis] >= t].tolist()

            li = len(self._nodes)
            self._nodes.append(_BVHNode(l_mn, l_mx))
            ri = li + 1
            self._nodes.append(_BVHNode(r_mn, r_mx))
            node.left  = li
            node.right = ri

            q.append((li, left_idxs))
            q.append((ri, right_idxs))

    def traverse(self, ray_origin: np.ndarray, ray_dir: np.ndarray,
                 t_max: float = float('inf')) -> tuple[int, float]:
        """Return (original_prim_idx, t), or (-1, inf) on miss."""
        origin = ray_origin.astype(np.float64)
        direction = ray_dir.astype(np.float64)
        stack = [0]
        best_t   = t_max
        best_idx = -1

        while stack:
            idx  = stack.pop()
            node = self._nodes[idx]

            if not _ray_aabb(origin, direction,
                              node.aabb_min.astype(np.float64),
                              node.aabb_max.astype(np.float64), best_t):
                continue

            if node.is_leaf:
                for j in range(node.prim_start, node.prim_end):
                    pi = self._prim_order[j]
                    t = _moller_trumbore(
                        origin, direction,
                        self._v0[pi].astype(np.float64),
                        self._v1[pi].astype(np.float64),
                        self._v2[pi].astype(np.float64),
                        t_max=best_t)
                    if t is not None and t < best_t:
                        best_t   = t
                        best_idx = pi
            else:
                stack.append(node.left)
                stack.append(node.right)

        return best_idx, best_t


# ---------------------------------------------------------------------------
# Shared ray-intersection helpers (also imported by traversal.py)
# ---------------------------------------------------------------------------

def _ray_aabb(origin, direction, aabb_min, aabb_max, t_max):
    inv = np.where(np.abs(direction) > 1e-12, 1.0 / direction,
                   np.sign(direction) * 1e12)
    t0 = (aabb_min - origin) * inv
    t1 = (aabb_max - origin) * inv
    t_enter = np.maximum(np.minimum(t0, t1), 0.0).max()
    t_exit  = np.minimum(np.maximum(t0, t1), t_max).min()
    return t_enter <= t_exit


def _moller_trumbore(origin, direction, v0, v1, v2, t_min=1e-6, t_max=float('inf')):
    e1 = v1 - v0
    e2 = v2 - v0
    h  = np.cross(direction, e2)
    a  = float(np.dot(e1, h))
    if abs(a) < 1e-10:
        return None
    f = 1.0 / a
    s = origin - v0
    u = f * float(np.dot(s, h))
    if u < 0.0 or u > 1.0:
        return None
    q = np.cross(s, e1)
    v = f * float(np.dot(direction, q))
    if v < 0.0 or u + v > 1.0:
        return None
    t = f * float(np.dot(e2, q))
    return t if t_min <= t <= t_max else None

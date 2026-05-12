"""Binary I/O bridge between the Python mesh loader and the CUDA binary.

Triangle file format (triangles.bin):
    [uint32  magic  = 0x54524900 "TRI\\0"]
    [uint32  n_tris]
    [float32 scene_min[3]]
    [float32 scene_max[3]]
    [Triangle × n_tris]  where Triangle = float32[9] (v0, v1, v2)

LBVH file format (output.lbvh.bin):
    [uint32  magic  = 0x4C424856 "LBVH"]
    [uint32  n_prims]
    [uint32  n_nodes = 2*n_prims - 1]
    [int32   root_idx]
    [LBVHNode × n_nodes]  (40 bytes each)
    [uint32  sorted_prim_idx × n_prims]
"""

from __future__ import annotations
import struct
import numpy as np
from pathlib import Path

TRI_MAGIC  = 0x54524900
LBVH_MAGIC = 0x4C424856

# Numpy dtypes that match the C++ structs exactly.
TRI_HDR_DTYPE = np.dtype([
    ('magic',     np.uint32),
    ('n_tris',    np.uint32),
    ('scene_min', np.float32, (3,)),
    ('scene_max', np.float32, (3,)),
])  # 32 bytes

TRI_DTYPE = np.dtype([
    ('v0', np.float32, (3,)),
    ('v1', np.float32, (3,)),
    ('v2', np.float32, (3,)),
])  # 36 bytes

LBVH_HDR_DTYPE = np.dtype([
    ('magic',    np.uint32),
    ('n_prims',  np.uint32),
    ('n_nodes',  np.uint32),
    ('root_idx', np.int32),
])  # 16 bytes

NODE_DTYPE = np.dtype([
    ('aabb_min', np.float32, (3,)),
    ('aabb_max', np.float32, (3,)),
    ('left',     np.int32),
    ('right',    np.int32),
    ('parent',   np.int32),
    ('prim_idx', np.int32),
])  # 40 bytes


def write_triangles(path: str | Path,
                    vertices: np.ndarray,
                    faces: np.ndarray) -> None:
    """Write triangles to a binary file consumable by lbvh_build.

    Args:
        path:     Output path.
        vertices: float32 array of shape (V, 3).
        faces:    int array of shape (F, 3) — vertex indices per triangle.
    """
    path = Path(path)
    v0 = vertices[faces[:, 0]].astype(np.float32)
    v1 = vertices[faces[:, 1]].astype(np.float32)
    v2 = vertices[faces[:, 2]].astype(np.float32)

    scene_min = np.minimum(np.minimum(v0, v1), v2).min(axis=0)
    scene_max = np.maximum(np.maximum(v0, v1), v2).max(axis=0)
    # Pad degenerate scenes
    span = scene_max - scene_min
    scene_max = np.where(span < 1e-6, scene_min + 1.0, scene_max)

    hdr = np.zeros(1, dtype=TRI_HDR_DTYPE)
    hdr['magic']     = TRI_MAGIC
    hdr['n_tris']    = len(faces)
    hdr['scene_min'] = scene_min
    hdr['scene_max'] = scene_max

    tris = np.zeros(len(faces), dtype=TRI_DTYPE)
    tris['v0'] = v0
    tris['v1'] = v1
    tris['v2'] = v2

    with open(path, 'wb') as f:
        hdr.tofile(f)
        tris.tofile(f)


def read_triangles(path: str | Path):
    """Read a triangles.bin file.

    Returns:
        tris      : structured numpy array with dtype TRI_DTYPE
        scene_min : float32 array [3]
        scene_max : float32 array [3]
    """
    path = Path(path)
    with open(path, 'rb') as f:
        hdr = np.frombuffer(f.read(TRI_HDR_DTYPE.itemsize), dtype=TRI_HDR_DTYPE)[0]
        if hdr['magic'] != TRI_MAGIC:
            raise ValueError(f"Bad triangle file magic: 0x{hdr['magic']:08X}")
        n = int(hdr['n_tris'])
        tris = np.frombuffer(f.read(n * TRI_DTYPE.itemsize), dtype=TRI_DTYPE)
    return tris, hdr['scene_min'].copy(), hdr['scene_max'].copy()


def read_lbvh(path: str | Path):
    """Read an LBVH binary file produced by lbvh_build.

    Returns:
        nodes           : structured numpy array with dtype NODE_DTYPE, shape (2N-1,)
        sorted_prim_idx : uint32 array of length N
        root_idx        : int
    """
    path = Path(path)
    with open(path, 'rb') as f:
        hdr = np.frombuffer(f.read(LBVH_HDR_DTYPE.itemsize), dtype=LBVH_HDR_DTYPE)[0]
        if hdr['magic'] != LBVH_MAGIC:
            raise ValueError(f"Bad LBVH magic: 0x{hdr['magic']:08X}")
        n_prims = int(hdr['n_prims'])
        n_nodes = int(hdr['n_nodes'])
        root    = int(hdr['root_idx'])
        nodes   = np.frombuffer(f.read(n_nodes * NODE_DTYPE.itemsize), dtype=NODE_DTYPE)
        sorted_prim_idx = np.frombuffer(f.read(n_prims * 4), dtype=np.uint32)
    return nodes, sorted_prim_idx, root

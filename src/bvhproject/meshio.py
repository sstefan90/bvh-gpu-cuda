"""Minimal mesh loaders that avoid heavy dependencies.

OBJ loader: parses v/f lines only (no mtl, no normals, no UVs).
Fan-triangulates quads and n-gons.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path


def load_obj(path: str | Path):
    """Return (vertices float32[V,3], faces int64[F,3])."""
    vertices = []
    faces    = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                parts = line.split()[1:]
                idxs  = [int(p.split('/')[0]) - 1 for p in parts]  # 0-based
                for i in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[i], idxs[i + 1]])
    if not vertices:
        raise ValueError(f"No vertices found in {path}")
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int64)

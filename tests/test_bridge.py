"""Tests for the binary bridge (bridge.py).

Verifies that write_triangles → read_triangles round-trips correctly.
Does not require lbvh_build or a GPU.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from bvhproject.bridge import write_triangles, read_triangles, TRI_MAGIC


def make_cube_mesh():
    """Return vertices and faces for a unit cube."""
    v = np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1],
    ], dtype=np.float32)
    f = np.array([
        [0,1,2],[0,2,3],[4,5,6],[4,6,7],
        [0,1,5],[0,5,4],[2,3,7],[2,7,6],
        [0,3,7],[0,7,4],[1,2,6],[1,6,5],
    ], dtype=np.int64)
    return v, f


def test_round_trip_triangles():
    v, f = make_cube_mesh()
    with tempfile.NamedTemporaryFile(suffix='.tri.bin', delete=False) as tmp:
        path = Path(tmp.name)

    write_triangles(path, v, f)
    tris, scene_min, scene_max = read_triangles(path)
    path.unlink()

    assert len(tris) == len(f), "triangle count mismatch"
    # Check first triangle vertices match
    np.testing.assert_allclose(tris['v0'][0], v[f[0, 0]], atol=1e-6)
    np.testing.assert_allclose(tris['v1'][0], v[f[0, 1]], atol=1e-6)
    np.testing.assert_allclose(tris['v2'][0], v[f[0, 2]], atol=1e-6)


def test_scene_aabb_correct():
    v, f = make_cube_mesh()
    with tempfile.NamedTemporaryFile(suffix='.tri.bin', delete=False) as tmp:
        path = Path(tmp.name)
    write_triangles(path, v, f)
    _, scene_min, scene_max = read_triangles(path)
    path.unlink()

    np.testing.assert_allclose(scene_min, [0, 0, 0], atol=1e-5)
    np.testing.assert_allclose(scene_max, [1, 1, 1], atol=1e-5)


def test_bad_magic_raises():
    with tempfile.NamedTemporaryFile(suffix='.tri.bin', delete=False) as tmp:
        tmp.write(b'\x00' * 64)
        path = Path(tmp.name)
    with pytest.raises(ValueError, match="magic"):
        read_triangles(path)
    path.unlink()


def test_large_scene():
    rng = np.random.default_rng(0)
    v = rng.random((1000, 3), dtype=np.float32)
    f = rng.integers(0, 1000, size=(500, 3)).astype(np.int64)
    # Make sure faces are valid (no degenerate)
    f = np.clip(f, 0, 999)
    with tempfile.NamedTemporaryFile(suffix='.tri.bin', delete=False) as tmp:
        path = Path(tmp.name)
    write_triangles(path, v, f)
    tris, _, _ = read_triangles(path)
    path.unlink()
    assert len(tris) == 500

"""End-to-end correctness test: GPU LBVH vs CPU SAH BVH.

Requires:
  • lbvh_build compiled and on PATH (or at ./lbvh_build)
  • A GPU device present

Run with:
    pytest tests/test_correctness.py -v
Or for a quick sanity check with a small synthetic scene:
    pytest tests/test_correctness.py::test_cube_scene -v
"""

import subprocess
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pytest

from bvhproject.bridge   import write_triangles, read_triangles, read_lbvh
from bvhproject.traversal import traverse_lbvh
from bvhproject.oracle   import SahOracle
from bvhproject.validate import _generate_rays

LBVH_BUILD = Path(__file__).parent.parent / 'lbvh_build'
T_TOL = 1e-3


def _requires_gpu():
    """Skip if no CUDA device is visible."""
    try:
        import subprocess as sp
        sp.run(['nvidia-smi', '-L'], check=True, capture_output=True)
    except Exception:
        pytest.skip("No GPU available")


def _run_lbvh_build(tri_bin: Path, lbvh_bin: Path, leveled: bool = False):
    cmd = [str(LBVH_BUILD), str(tri_bin), str(lbvh_bin)]
    if leveled:
        cmd.append('--leveled')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"lbvh_build failed:\n{result.stderr}")
    return result.stdout


def _validate_in_memory(tri_bin, lbvh_bin, n_rays=256):
    tris, scene_min, scene_max = read_triangles(tri_bin)
    nodes, _, root_idx = read_lbvh(lbvh_bin)

    n_tris = len(tris)
    v3 = np.empty((n_tris * 3, 3), dtype=np.float32)
    v3[0::3] = tris['v0']
    v3[1::3] = tris['v1']
    v3[2::3] = tris['v2']
    f3 = np.arange(n_tris * 3, dtype=np.int64).reshape(n_tris, 3)
    oracle = SahOracle(v3, f3)

    origins, directions = _generate_rays(
        scene_min.astype(np.float64), scene_max.astype(np.float64), n_rays)

    n_disagree = 0
    for orig, direc in zip(origins, directions):
        lp, lt = traverse_lbvh(orig, direc, nodes, tris, root_idx)
        sp, st = oracle.traverse(orig, direc)
        lhit, shit = lp >= 0, sp >= 0
        if lhit and shit:
            if abs(lt - st) >= T_TOL:
                n_disagree += 1
        elif lhit != shit:
            n_disagree += 1

    return n_disagree == 0, n_disagree


@pytest.fixture(scope='session')
def lbvh_build_path():
    if not LBVH_BUILD.exists():
        pytest.skip(f"lbvh_build not found at {LBVH_BUILD}. Run `make` first.")
    return LBVH_BUILD


def test_cube_scene(lbvh_build_path):
    """Cornell-box-scale scene: 12-triangle unit cube, 256 rays."""
    _requires_gpu()
    v = np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1],
    ], dtype=np.float32)
    f = np.array([
        [0,1,2],[0,2,3],[4,5,6],[4,6,7],
        [0,1,5],[0,5,4],[2,3,7],[2,7,6],
        [0,3,7],[0,7,4],[1,2,6],[1,6,5],
    ], dtype=np.int64)

    with tempfile.TemporaryDirectory() as tmp:
        tri_bin  = Path(tmp) / 'cube.tri.bin'
        lbvh_bin = Path(tmp) / 'cube.lbvh.bin'
        write_triangles(tri_bin, v, f)
        out = _run_lbvh_build(tri_bin, lbvh_bin)
        print(out)
        ok, n_fail = _validate_in_memory(tri_bin, lbvh_bin, n_rays=256)
        assert ok, f"Cube scene: {n_fail} ray(s) disagree"


def test_cube_scene_leveled(lbvh_build_path):
    """Same cube scene with level-scheduled refit."""
    _requires_gpu()
    v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                  [0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=np.float32)
    f = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],
                  [0,1,5],[0,5,4],[2,3,7],[2,7,6],
                  [0,3,7],[0,7,4],[1,2,6],[1,6,5]], dtype=np.int64)

    with tempfile.TemporaryDirectory() as tmp:
        tri_bin  = Path(tmp) / 'cube.tri.bin'
        lbvh_bin = Path(tmp) / 'cube.lbvh.leveled.bin'
        write_triangles(tri_bin, v, f)
        _run_lbvh_build(tri_bin, lbvh_bin, leveled=True)
        ok, n_fail = _validate_in_memory(tri_bin, lbvh_bin, n_rays=256)
        assert ok, f"Cube scene (leveled): {n_fail} ray(s) disagree"


@pytest.mark.skipif(
    not (Path(__file__).parent.parent.parent / 'asst3/resources/bunny.obj').exists(),
    reason="bunny.obj not found"
)
def test_bunny_scene(lbvh_build_path):
    """Stanford bunny (~70k tris), 512 rays, both refit variants."""
    _requires_gpu()
    bunny = Path(__file__).parent.parent.parent / 'asst3/resources/bunny.obj'
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(bunny))
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.triangles, dtype=np.int64)

    with tempfile.TemporaryDirectory() as tmp:
        tri_bin  = Path(tmp) / 'bunny.tri.bin'
        for lbvh_name, leveled in [('bunny.lbvh.bin', False),
                                    ('bunny.lbvh.leveled.bin', True)]:
            lbvh_bin = Path(tmp) / lbvh_name
            write_triangles(tri_bin, v, f)
            _run_lbvh_build(tri_bin, lbvh_bin, leveled=leveled)
            ok, n_fail = _validate_in_memory(tri_bin, lbvh_bin, n_rays=512)
            tag = 'leveled' if leveled else 'atomic'
            assert ok, f"Bunny ({tag}): {n_fail} ray(s) disagree"

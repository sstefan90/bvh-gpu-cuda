"""Full-pipeline correctness validator.

Compares the GPU-built LBVH against the CPU SAH BVH on a fixed ray batch.
Primary metric: t-value agreement within a floating-point tolerance.
  |t_lbvh - t_sah| < TOL  for every ray that both BVHs report as a hit.

Usage (after building lbvh_build and exporting triangles):
    python -m bvhproject.validate \\
        --tri  data/bunny.tri.bin \\
        --lbvh data/bunny.lbvh.bin

Outputs a summary table and exits with code 0 on full agreement, 1 otherwise.
"""

from __future__ import annotations
import argparse
import sys
import time
import numpy as np

from .bridge   import read_triangles, read_lbvh
from .traversal import traverse_lbvh
from .oracle   import SahOracle

T_TOL = 1e-3   # t-value agreement tolerance (world-space units)


def _generate_rays(scene_min, scene_max, n_rays=512, seed=42):
    """Generate rays that originate outside the scene AABB and point inward."""
    rng    = np.random.default_rng(seed)
    center = 0.5 * (scene_min + scene_max)
    radius = 1.5 * np.linalg.norm(scene_max - scene_min) * 0.5

    origins    = np.zeros((n_rays, 3), dtype=np.float64)
    directions = np.zeros((n_rays, 3), dtype=np.float64)
    for i in range(n_rays):
        # Sample point on a sphere around the scene
        theta    = rng.uniform(0, np.pi)
        phi      = rng.uniform(0, 2 * np.pi)
        origin   = center + radius * np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)])
        # Perturb target slightly to avoid degenerate rays
        target   = center + rng.uniform(-0.1, 0.1, 3) * (scene_max - scene_min)
        direction = target - origin
        direction = direction / np.linalg.norm(direction)
        origins[i]    = origin
        directions[i] = direction
    return origins, directions


def run(tri_path: str, lbvh_path: str, n_rays: int = 512, verbose: bool = True):
    # Load data
    tris, scene_min, scene_max = read_triangles(tri_path)
    nodes, _sorted_prim_idx, root_idx = read_lbvh(lbvh_path)

    vertices = np.vstack([
        tris['v0'], tris['v1'], tris['v2']]).reshape(-1, 3)
    n_tris = len(tris)
    faces  = np.arange(n_tris * 3).reshape(n_tris, 3)

    # Build SAH oracle.
    # Vertex layout: v3[3*i] = tris[i].v0, v3[3*i+1] = tris[i].v1, v3[3*i+2] = tris[i].v2
    if verbose:
        print(f"Building SAH oracle on {n_tris} triangles …", flush=True)
    t0 = time.time()
    v3 = np.empty((n_tris * 3, 3), dtype=np.float32)
    v3[0::3] = tris['v0']
    v3[1::3] = tris['v1']
    v3[2::3] = tris['v2']
    f3 = np.arange(n_tris * 3, dtype=np.int64).reshape(n_tris, 3)
    oracle = SahOracle(v3, f3)
    if verbose:
        print(f"  SAH build: {time.time()-t0:.2f}s")

    origins, directions = _generate_rays(scene_min.astype(np.float64),
                                          scene_max.astype(np.float64), n_rays)

    n_hits_lbvh = n_hits_sah = 0
    n_agree = n_disagree = n_both_miss = 0
    max_delta_t = 0.0
    mismatches = []

    for i, (orig, direc) in enumerate(zip(origins, directions)):
        lbvh_prim, lbvh_t = traverse_lbvh(orig, direc, nodes, tris, root_idx)
        sah_prim,  sah_t  = oracle.traverse(orig, direc)

        lbvh_hit = lbvh_prim >= 0
        sah_hit  = sah_prim  >= 0

        if lbvh_hit: n_hits_lbvh += 1
        if sah_hit:  n_hits_sah  += 1

        if not lbvh_hit and not sah_hit:
            n_both_miss += 1
            continue

        if lbvh_hit and sah_hit:
            delta_t = abs(lbvh_t - sah_t)
            max_delta_t = max(max_delta_t, delta_t)
            if delta_t < T_TOL:
                n_agree += 1
            else:
                n_disagree += 1
                mismatches.append((i, lbvh_prim, lbvh_t, sah_prim, sah_t))
        else:
            # One hit, one miss
            n_disagree += 1
            mismatches.append((i, lbvh_prim, lbvh_t, sah_prim, sah_t))

    n_tested = n_rays - n_both_miss
    pct = 100.0 * n_agree / n_tested if n_tested else 100.0

    if verbose:
        print(f"\n=== Validation results ({n_rays} rays) ===")
        print(f"  Rays with both-miss:          {n_both_miss:6d}")
        print(f"  Rays where both report a hit: {n_tested:6d}")
        print(f"  Agreement (|Δt| < {T_TOL}):     {n_agree:6d} / {n_tested}  ({pct:.1f}%)")
        print(f"  Disagreements:                {n_disagree:6d}")
        print(f"  Max |Δt| on agreeing rays:    {max_delta_t:.2e}")
        if mismatches:
            print(f"\n  First 5 mismatches:")
            for ri, lp, lt, sp, st in mismatches[:5]:
                print(f"    ray {ri:4d}: LBVH prim={lp} t={lt:.6f}  |  SAH prim={sp} t={st:.6f}")

    return n_disagree == 0


def main():
    p = argparse.ArgumentParser(description="Validate GPU LBVH vs CPU SAH BVH")
    p.add_argument('--tri',    required=True, help='triangles.bin path')
    p.add_argument('--lbvh',   required=True, help='output.lbvh.bin path')
    p.add_argument('--rays',   type=int, default=512, help='number of test rays')
    p.add_argument('--quiet',  action='store_true')
    args = p.parse_args()

    ok = run(args.tri, args.lbvh, n_rays=args.rays, verbose=not args.quiet)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

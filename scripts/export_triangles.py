#!/usr/bin/env python3
"""Export a mesh's triangles to the bvhproject binary format.

Uses the built-in OBJ parser (bvhproject.meshio) — no open3d needed.

Usage:
    python scripts/export_triangles.py --mesh ../asst3/resources/bunny.obj \\
                                        --output data/bunny.tri.bin
    python scripts/export_triangles.py --mesh ../asst3/resources/bunny.obj \\
                                        --info   # print stats only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from bvhproject.bridge  import write_triangles
from bvhproject.meshio  import load_obj


def main():
    ap = argparse.ArgumentParser(description="Export mesh triangles to .tri.bin")
    ap.add_argument('--mesh',   required=True, help='Input OBJ file')
    ap.add_argument('--output', help='Output .tri.bin path')
    ap.add_argument('--info',   action='store_true', help='Print stats and exit')
    args = ap.parse_args()

    mesh_path = Path(args.mesh)
    if not mesh_path.exists():
        print(f"Error: not found: {mesh_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {mesh_path} …")
    vertices, faces = load_obj(mesh_path)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    scene_min = np.minimum(np.minimum(v0, v1), v2).min(axis=0)
    scene_max = np.maximum(np.maximum(v0, v1), v2).max(axis=0)

    print(f"  Vertices:  {len(vertices)}")
    print(f"  Triangles: {len(faces)}")
    print(f"  AABB min:  {scene_min}")
    print(f"  AABB max:  {scene_max}")

    if args.info:
        return

    if not args.output:
        print("Error: --output required (unless --info)", file=sys.stderr)
        sys.exit(1)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_triangles(out, vertices, faces)
    print(f"Wrote → {out}  ({out.stat().st_size/1e6:.2f} MB)")


if __name__ == '__main__':
    main()

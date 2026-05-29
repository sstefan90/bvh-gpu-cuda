#!/bin/bash
# Correctness check: build with P=1,2,4 and compare root AABBs.
#
#SBATCH --job-name=lbvh_correct
#SBATCH --partition=gpu-turing
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=sbatch/logs/correctness_%j.out
#SBATCH --error=sbatch/logs/correctness_%j.err

ml course/cme213/nvhpc/24.1
mkdir -p sbatch/logs

SCENE=${1:-data/bunny.tri.bin}

echo "=== MPI CORRECTNESS: $SCENE ==="
for P in 1 2 4; do
    echo ""
    echo "--- P=$P ---"
    mpirun -np $P ./lbvh_mpi "$SCENE" "data/correct_P${P}" 2>&1
done

echo ""
echo "=== Python AABB comparison ==="
python - <<'PY'
import struct, sys

def read_root_aabb(path):
    with open(path, "rb") as f:
        magic, n_prims, n_nodes, root_idx = struct.unpack("<IIIi", f.read(16))
        assert magic == 0x4C424856, f"Bad magic in {path}"
        nodes = []
        for _ in range(n_nodes):
            data = struct.unpack("<6f4i", f.read(40))
            nodes.append(data)
    root = nodes[root_idx]
    return root[:3], root[3:6]   # (aabb_min, aabb_max)

try:
    mn1, mx1 = read_root_aabb("data/correct_P1_rank0.lbvh.bin")
    print(f"P=1 root AABB: min={mn1}  max={mx1}")
    for P in [2, 4]:
        # For P>1 the root AABB is in the top-level tree.
        try:
            tl = f"data/correct_P{P}_toplevel.bin"
            mn, mx = read_root_aabb(tl)
            print(f"P={P} root AABB: min={mn}  max={mx}")
            # Compare
            ok = all(abs(mn[k]-mn1[k]) < 1e-3 for k in range(3)) and \
                 all(abs(mx[k]-mx1[k]) < 1e-3 for k in range(3))
            print(f"  P={P} vs P=1 AABB match: {'PASS' if ok else 'FAIL'}")
        except FileNotFoundError as e:
            print(f"  {e}")
except FileNotFoundError as e:
    print(f"Output file not found: {e}")
PY

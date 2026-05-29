#!/bin/bash
# Weak-scaling sweep: 10M triangles per rank, P in {1, 2, 4}.
# Total problem size scales with P.
#
# Submit with:  sbatch sbatch/run_scaling_weak.sh
#
#SBATCH --job-name=lbvh_weak
#SBATCH --partition=gpu-turing
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=sbatch/logs/weak_%j.out
#SBATCH --error=sbatch/logs/weak_%j.err

ml course/cme213/nvhpc/24.1
mkdir -p sbatch/logs data

N_PER_RANK=10000000

echo "=== WEAK SCALING: ${N_PER_RANK} triangles/rank ==="

for P in 1 2 4; do
    N=$((P * N_PER_RANK))
    SCENE="data/weak_P${P}_${N}.tri.bin"

    if [ ! -f "$SCENE" ]; then
        echo "Generating $N synthetic triangles for P=$P..."
        python - "$N" "$SCENE" <<'PY'
import numpy as np, struct, sys
N = int(sys.argv[1])
out = sys.argv[2]
rng = np.random.default_rng(42)
verts = rng.random((N, 9), dtype=np.float32)
all_v = verts.reshape(-1, 3)
scene_min = all_v.min(axis=0).astype(np.float32)
scene_max = all_v.max(axis=0).astype(np.float32)
magic = 0x54524900
with open(out, "wb") as f:
    f.write(struct.pack("<II", magic, N))
    f.write(scene_min.tobytes())
    f.write(scene_max.tobytes())
    f.write(verts.tobytes())
print(f"Wrote {N} triangles to {out}")
PY
    fi

    echo ""
    echo "--- P=$P ranks, N=$N ---"
    mpirun -np $P ./lbvh_mpi "$SCENE" "data/weak_P${P}" 2>&1
done

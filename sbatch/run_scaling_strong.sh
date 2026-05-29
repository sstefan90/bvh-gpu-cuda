#!/bin/bash
# Strong-scaling sweep: fixed N=50M triangles, P in {1, 2, 4}.
# Generates synthetic 50M-triangle scene if not already present.
#
# Submit with:  sbatch sbatch/run_scaling_strong.sh
#
#SBATCH --job-name=lbvh_strong
#SBATCH --partition=gpu-turing
#SBATCH --nodes=1
#SBATCH --ntasks=4          # request max GPUs; mpirun -np controls actual count
#SBATCH --gpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=sbatch/logs/strong_%j.out
#SBATCH --error=sbatch/logs/strong_%j.err

ml course/cme213/nvhpc/24.1
mkdir -p sbatch/logs data

SCENE=data/strong_50M.tri.bin

# Generate synthetic 50M-triangle scene if not present.
if [ ! -f "$SCENE" ]; then
    echo "Generating 50M synthetic triangles..."
    python - <<'PY'
import numpy as np, struct, sys
N = 50_000_000
rng = np.random.default_rng(42)
verts = rng.random((N, 9), dtype=np.float32)
scene_min = verts.min(axis=0)[:3]
scene_max = verts.max(axis=0)[:3]
# Recompute proper scene AABB across all 9 components
all_v = verts.reshape(-1, 3)
scene_min = all_v.min(axis=0).astype(np.float32)
scene_max = all_v.max(axis=0).astype(np.float32)
magic = 0x54524900
with open("data/strong_50M.tri.bin", "wb") as f:
    f.write(struct.pack("<II", magic, N))
    f.write(scene_min.tobytes())
    f.write(scene_max.tobytes())
    f.write(verts.tobytes())
print(f"Wrote {N} triangles to data/strong_50M.tri.bin")
PY
fi

echo "=== STRONG SCALING: N=50M ==="
for P in 1 2 4; do
    echo ""
    echo "--- P=$P ranks ---"
    mpirun -np $P ./lbvh_mpi "$SCENE" "data/strong_P${P}" 2>&1
done

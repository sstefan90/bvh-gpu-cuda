#!/usr/bin/env bash
# End-to-end correctness check for a single mesh.
#
# Usage:  ./scripts/run_correctness.sh [mesh.obj] [n_rays]
# Example: ./scripts/run_correctness.sh ../asst3/resources/bunny.obj 512
#
# Requires:
#   • lbvh_build compiled (make)
#   • Python env active (.venv)
#   • NVHPC module loaded (ml course/cme213/nvhpc/24.1)

set -euo pipefail

MESH="${1:-../asst3/resources/bunny.obj}"
N_RAYS="${2:-512}"
NAME=$(basename "${MESH%.*}")
TRI_BIN="data/${NAME}.tri.bin"
LBVH_BIN="data/${NAME}.lbvh.bin"

cd "$(dirname "$0")/.."

echo "=== Step 1: Export triangles from ${MESH} ==="
.venv/bin/python scripts/export_triangles.py --mesh "${MESH}" --output "${TRI_BIN}"

echo ""
echo "=== Step 2: Build LBVH on GPU (atomic refit) ==="
./lbvh_build "${TRI_BIN}" "${LBVH_BIN}"

echo ""
echo "=== Step 3: Validate LBVH vs SAH oracle (${N_RAYS} rays) ==="
.venv/bin/python -m bvhproject.validate \
    --tri  "${TRI_BIN}" \
    --lbvh "${LBVH_BIN}" \
    --rays "${N_RAYS}"

echo ""
echo "=== Step 4: Re-run with level-scheduled refit ==="
LBVH_BIN_LVL="data/${NAME}.lbvh.leveled.bin"
./lbvh_build "${TRI_BIN}" "${LBVH_BIN_LVL}" --leveled

.venv/bin/python -m bvhproject.validate \
    --tri  "${TRI_BIN}" \
    --lbvh "${LBVH_BIN_LVL}" \
    --rays "${N_RAYS}" \
    --quiet && echo "Level-scheduled refit: PASS"

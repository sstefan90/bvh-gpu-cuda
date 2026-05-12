#pragma once
#include "lbvh_types.h"
#include <cuda_runtime.h>

// Set leaf-node AABBs from the sorted triangle array.
// Leaf i (flat index N-1+i) gets the AABB of triangles[sorted_prim_idx[i]].
void compute_leaf_aabbs(
    float*          d_aabb_min,         // [2N-1][3] flattened row-major
    float*          d_aabb_max,
    const Triangle* d_triangles,        // original triangle array
    const uint32_t* d_sorted_prim_idx, // permutation after Morton sort
    int             n
);

// Bottom-up AABB refit — atomic variant (★ custom kernel, Stage 4a).
// Each leaf thread walks parent pointers upward; the second child to arrive at
// an internal node performs the AABB union and continues.
// d_counter must be 0-initialised before the call.
void refit_aabbs_atomic(
    float*       d_aabb_min,
    float*       d_aabb_max,
    const int*   d_left,
    const int*   d_right,
    const int*   d_parent,
    int*         d_counter,   // [N-1], must be cudaMemset to 0
    int          n
);

// Bottom-up AABB refit — level-scheduled variant (★ custom kernel, Stage 4b).
// One kernel launch per depth level; no atomics.  Slower to set up but avoids
// atomic contention; used for the M3 optimization comparison.
void refit_aabbs_leveled(
    float*       d_aabb_min,
    float*       d_aabb_max,
    const int*   d_left,
    const int*   d_right,
    const int*   d_parent,
    int          n
);

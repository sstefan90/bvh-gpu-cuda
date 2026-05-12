#pragma once
#include "lbvh_types.h"
#include <cuda_runtime.h>

// Encode N triangles into 30-bit Morton codes.
// d_triangles  : GPU buffer of Triangle[n]
// d_codes      : GPU output: Morton code per triangle [n]
// d_prim_idx   : GPU output: identity permutation 0..N-1 [n]
// scene_min/max: pre-computed scene AABB (from TriFileHeader)
void morton_encode(
    const Triangle* d_triangles,
    uint32_t*       d_codes,
    uint32_t*       d_prim_idx,
    float3          scene_min,
    float3          scene_max,
    int             n
);

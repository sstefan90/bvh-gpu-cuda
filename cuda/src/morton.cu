// Morton encoding kernel Stage 1 of LBVH pipeline

// Here's the task fro each thread:
//   Read three vertices (AoS — 9 consecutive floats).
//   Compute centroid.
//   Normalize centroid to [0, 1]^3 using the scene AABB.
//   Bit-interleave (x, y, z) into a 30-bit Morton code.
//

// FUTURE work to optimize
//  switch to SoA vertex layout to coalesce reads.
//  The current AoS layout issues one 36-byte load per thread, so adjcent threads
//  are 36 bytes apart, so reads are NOT coalesced.  With SoA the x-coordinates
//  of all triangles are contiguous, which means coalesced 4byte reads.

#include "morton.cuh"
#include <cuda_runtime.h>

// Morton code interleaving
__device__ __forceinline__ uint32_t expand_bits(uint32_t v)
{
    v &= 0x000003FFu;
    v = (v | (v << 16u)) & 0xFF0000FFu;
    v = (v | (v << 8u)) & 0x0300F00Fu;
    v = (v | (v << 4u)) & 0x030C30C3u;
    v = (v | (v << 2u)) & 0x09249249u;
    return v;
}

// Map a normalized position in [0, 1]^3 to a 30-bit Morton code.
__device__ __forceinline__ uint32_t morton3D(float nx, float ny, float nz)
{
    uint32_t ix = (uint32_t)fminf(fmaxf(nx * 1024.0f, 0.0f), 1023.0f);
    uint32_t iy = (uint32_t)fminf(fmaxf(ny * 1024.0f, 0.0f), 1023.0f);
    uint32_t iz = (uint32_t)fminf(fmaxf(nz * 1024.0f, 0.0f), 1023.0f);
    return expand_bits(ix) | (expand_bits(iy) << 1u) | (expand_bits(iz) << 2u);
}

__global__ void morton_encode_kernel(
    const Triangle *__restrict__ triangles,
    uint32_t *__restrict__ codes,
    uint32_t *__restrict__ prim_idx,
    float3 scene_min,
    float3 scene_max,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const Triangle &t = triangles[i];

    // Centroid of bounding box of this triangle
    float cx = 0.5f * (fminf(fminf(t.v0[0], t.v1[0]), t.v2[0]) +
                       fmaxf(fmaxf(t.v0[0], t.v1[0]), t.v2[0]));

    float cy = 0.5f * (fminf(fminf(t.v0[1], t.v1[1]), t.v2[1]) +
                       fmaxf(fmaxf(t.v0[1], t.v1[1]), t.v2[1]));
    float cz = 0.5f * (fminf(fminf(t.v0[2], t.v1[2]), t.v2[2]) +
                       fmaxf(fmaxf(t.v0[2], t.v1[2]), t.v2[2]));

    // Normalize the data
    float span_x = scene_max.x - scene_min.x;
    float span_y = scene_max.y - scene_min.y;
    float span_z = scene_max.z - scene_min.z;
    float nx = (span_x > 0.0f) ? (cx - scene_min.x) / span_x : 0.5f;
    float ny = (span_y > 0.0f) ? (cy - scene_min.y) / span_y : 0.5f;
    float nz = (span_z > 0.0f) ? (cz - scene_min.z) / span_z : 0.5f;

    codes[i] = morton3D(nx, ny, nz);
    prim_idx[i] = (uint32_t)i;
}

// method to call the kernel and configure resources
void morton_encode(
    const Triangle *d_triangles,
    uint32_t *d_codes,
    uint32_t *d_prim_idx,
    float3 scene_min,
    float3 scene_max,
    int n)
{
    constexpr int BLOCK = 256;
    int grid = (n + BLOCK - 1) / BLOCK;
    morton_encode_kernel<<<grid, BLOCK>>>(
        d_triangles, d_codes, d_prim_idx, scene_min, scene_max, n);
}

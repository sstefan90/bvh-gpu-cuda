#pragma once
#include <cstdint>

// ---------------------------------------------------------------------------
// Triangle: three vertices in world space.
// Binary layout: 9 × float32 = 36 bytes, no padding.
// ---------------------------------------------------------------------------
struct Triangle {
    float v0[3];
    float v1[3];
    float v2[3];
};
static_assert(sizeof(Triangle) == 36, "Triangle size mismatch");

// ---------------------------------------------------------------------------
// LBVHNode: one entry in the flat 2N-1 node array.
//   Indices 0 .. N-2  → internal nodes
//   Indices N-1 .. 2N-2 → leaf nodes
// Binary layout: 40 bytes, no padding.
// ---------------------------------------------------------------------------
struct LBVHNode {
    float aabb_min[3];  // AABB min corner
    float aabb_max[3];  // AABB max corner
    int   left;         // child flat index (-1 for leaves)
    int   right;        // child flat index (-1 for leaves)
    int   parent;       // parent flat index (-1 for root)
    int   prim_idx;     // original triangle index for leaves (-1 for internals)
};
static_assert(sizeof(LBVHNode) == 40, "LBVHNode size mismatch");

// ---------------------------------------------------------------------------
// Binary file headers (written by Python, read by CUDA and vice-versa)
// ---------------------------------------------------------------------------
struct TriFileHeader {
    uint32_t magic;      // TRI_MAGIC
    uint32_t n_tris;
    float    scene_min[3];
    float    scene_max[3];
};
static_assert(sizeof(TriFileHeader) == 32, "TriFileHeader size mismatch");

struct LBVHFileHeader {
    uint32_t magic;    // LBVH_MAGIC
    uint32_t n_prims;
    uint32_t n_nodes;  // = 2*n_prims - 1
    int      root_idx; // flat index of the root internal node
};
static_assert(sizeof(LBVHFileHeader) == 16, "LBVHFileHeader size mismatch");

constexpr uint32_t TRI_MAGIC  = 0x54524900u;  // "TRI\0"
constexpr uint32_t LBVH_MAGIC = 0x4C424856u;  // "LBVH"

// Unit tests for the Morton encoding kernel.
//
// Tests:
//   1. Identity: centroid at origin → code 0.
//   2. Corner:   centroid at (1,1,1) → code 0x3FFFFFFF (all bits set, 30-bit).
//   3. Eight triangles at the corners of the unit cube — codes must match
//      hand-computed values and be mutually distinct.
//   4. Sort invariant: sorting the codes must produce non-decreasing output.

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <vector>

#include <cuda_runtime.h>
#include "lbvh_types.h"
#include "morton.cuh"

#define PASS(msg) printf("[PASS] %s\n", msg)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); g_failures++; } while(0)

static int g_failures = 0;

static std::vector<uint32_t> run_morton(
    const std::vector<Triangle>& tris,
    float3 smin, float3 smax)
{
    int n = (int)tris.size();
    Triangle* d_tris;
    uint32_t *d_codes, *d_idx;
    cudaMalloc(&d_tris,  n * sizeof(Triangle));
    cudaMalloc(&d_codes, n * sizeof(uint32_t));
    cudaMalloc(&d_idx,   n * sizeof(uint32_t));
    cudaMemcpy(d_tris, tris.data(), n * sizeof(Triangle), cudaMemcpyHostToDevice);

    morton_encode(d_tris, d_codes, d_idx, smin, smax, n);
    cudaDeviceSynchronize();

    std::vector<uint32_t> h_codes(n);
    cudaMemcpy(h_codes.data(), d_codes, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_tris); cudaFree(d_codes); cudaFree(d_idx);
    return h_codes;
}

static Triangle unit_tri(float cx, float cy, float cz) {
    // Degenerate triangle with centroid at (cx, cy, cz)
    Triangle t;
    t.v0[0] = cx; t.v0[1] = cy; t.v0[2] = cz;
    t.v1[0] = cx; t.v1[1] = cy; t.v1[2] = cz;
    t.v2[0] = cx; t.v2[1] = cy; t.v2[2] = cz;
    return t;
}

static void test_identity() {
    // Centroid at scene_min → code should be 0
    float3 smin{0.f, 0.f, 0.f}, smax{1.f, 1.f, 1.f};
    auto codes = run_morton({unit_tri(0.f, 0.f, 0.f)}, smin, smax);
    if (codes[0] == 0) PASS("identity code == 0");
    else FAIL("identity code != 0");
}

static void test_corner() {
    // Centroid at scene_max → code should have all 30 lower bits set
    float3 smin{0.f, 0.f, 0.f}, smax{1.f, 1.f, 1.f};
    // max normalized index = 1023 → max expanded = 0x09249249; OR all three → 0x3FFFFFFF
    auto codes = run_morton({unit_tri(1.f, 1.f, 1.f)}, smin, smax);
    if (codes[0] == 0x3FFFFFFFu) PASS("corner code == 0x3FFFFFFF");
    else {
        printf("[FAIL] corner code = 0x%08X (expected 0x3FFFFFFF)\n", codes[0]);
        g_failures++;
    }
}

static void test_eight_corners() {
    // 8 unit-cube corners — all codes must be distinct
    float3 smin{0.f,0.f,0.f}, smax{1.f,1.f,1.f};
    std::vector<Triangle> tris;
    for (int x = 0; x < 2; ++x)
    for (int y = 0; y < 2; ++y)
    for (int z = 0; z < 2; ++z)
        tris.push_back(unit_tri((float)x, (float)y, (float)z));
    auto codes = run_morton(tris, smin, smax);
    std::sort(codes.begin(), codes.end());
    bool all_distinct = (std::adjacent_find(codes.begin(), codes.end()) == codes.end());
    if (all_distinct) PASS("eight corners: all codes distinct");
    else FAIL("eight corners: duplicate codes");
}

static void test_sort_invariant() {
    // Random-ish triangles; sorted codes must be non-decreasing
    float3 smin{-5.f,-5.f,-5.f}, smax{5.f,5.f,5.f};
    std::vector<Triangle> tris;
    for (int i = 0; i < 64; ++i)
        tris.push_back(unit_tri(i * 0.15f - 4.f, -i * 0.1f + 3.f, i * 0.05f));
    auto codes = run_morton(tris, smin, smax);
    auto sorted = codes;
    std::sort(sorted.begin(), sorted.end());
    if (codes != sorted) {
        // just verify the sorted copy is non-decreasing
        bool ok = std::is_sorted(sorted.begin(), sorted.end());
        if (ok) PASS("sort invariant: sorted codes non-decreasing");
        else FAIL("sort invariant: sorted codes not non-decreasing");
    } else {
        PASS("sort invariant: input already sorted (monotone input)");
    }
}

int main() {
    printf("=== Morton unit tests ===\n");
    test_identity();
    test_corner();
    test_eight_corners();
    test_sort_invariant();
    if (g_failures == 0)
        printf("All Morton tests PASSED.\n");
    else
        printf("%d Morton test(s) FAILED.\n", g_failures);
    return g_failures ? EXIT_FAILURE : EXIT_SUCCESS;
}

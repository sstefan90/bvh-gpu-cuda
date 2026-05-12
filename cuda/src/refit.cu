// AABB refit kernels  Stage 4 of LBVH pipeline
//
// Two implementations
//   Atomic  — each leaf walks up; second-arriving child does the union.
//   Leveled — one kernel launch per depth level; no atomics.
//
// AABB arrays use a flat AoS layout: aabb_min[node*3 + axis].

// FUTURE WORK
//  switch to SOA layout (three separate float arrays
//  per min/max component) for coalesced reads when both children are processed
//  by the same warp.

#include "refit.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <queue>

// initialize the AABB leafs
__global__ void leaf_aabb_kernel(
    float *__restrict__ aabb_min,
    float *__restrict__ aabb_max,
    const Triangle *__restrict__ triangles,
    const uint32_t *__restrict__ sorted_prim_idx,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const Triangle &t = triangles[sorted_prim_idx[i]];
    int leaf = (n - 1) + i; // flat leaf index

    for (int k = 0; k < 3; ++k)
    {
        float mn = fminf(fminf(t.v0[k], t.v1[k]), t.v2[k]) - 1e-6f;
        float mx = fmaxf(fmaxf(t.v0[k], t.v1[k]), t.v2[k]) + 1e-6f;
        aabb_min[leaf * 3 + k] = mn;
        aabb_max[leaf * 3 + k] = mx;
    }
}

void compute_leaf_aabbs(
    float *d_aabb_min,
    float *d_aabb_max,
    const Triangle *d_triangles,
    const uint32_t *d_sorted_prim_idx,
    int n)
{
    constexpr int BLOCK = 256;
    leaf_aabb_kernel<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        d_aabb_min, d_aabb_max, d_triangles, d_sorted_prim_idx, n);
}

// atomic refit implementation

__global__ void refit_atomic_kernel(
    float *__restrict__ aabb_min,
    float *__restrict__ aabb_max,
    const int *__restrict__ d_left,
    const int *__restrict__ d_right,
    const int *__restrict__ d_parent,
    int *__restrict__ counter,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    int node = d_parent[(n - 1) + i];
    while (node != -1)
    {
        // Release fence: flush any AABB writes from this thread (prior iteration)
        // so that the other child's thread can read them after seeing counter == 1.
        __threadfence();

        int old = atomicAdd(&counter[node], 1);
        if (old == 0)
            return; // first to arrive: stop

        // Acquire fence: ensure we see the other child's AABB writes
        // (which that thread released via its own __threadfence() before atomicAdd).
        __threadfence();

        int lc = d_left[node], rc = d_right[node];
        for (int k = 0; k < 3; ++k)
        {
            aabb_min[node * 3 + k] =
                fminf(aabb_min[lc * 3 + k], aabb_min[rc * 3 + k]);
            aabb_max[node * 3 + k] =
                fmaxf(aabb_max[lc * 3 + k], aabb_max[rc * 3 + k]);
        }
        node = d_parent[node];
    }
}

void refit_aabbs_atomic(
    float *d_aabb_min,
    float *d_aabb_max,
    const int *d_left,
    const int *d_right,
    const int *d_parent,
    int *d_counter,
    int n)
{
    constexpr int BLOCK = 256;
    refit_atomic_kernel<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        d_aabb_min, d_aabb_max, d_left, d_right, d_parent, d_counter, n);
}

// Level-scheduled refit (more intuitive ina way)
// Computes a BFS ordering on the host, then launches one kernel per level
// from deepest to shallowest. each level is independent.

__global__ void refit_level_kernel(
    float *__restrict__ aabb_min,
    float *__restrict__ aabb_max,
    const int *__restrict__ d_left,
    const int *__restrict__ d_right,
    const int *__restrict__ level_nodes,
    int level_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= level_size)
        return;

    int node = level_nodes[idx];
    int lc = d_left[node], rc = d_right[node];
    for (int k = 0; k < 3; ++k)
    {
        aabb_min[node * 3 + k] =
            fminf(aabb_min[lc * 3 + k], aabb_min[rc * 3 + k]);
        aabb_max[node * 3 + k] =
            fmaxf(aabb_max[lc * 3 + k], aabb_max[rc * 3 + k]);
    }
}

void refit_aabbs_leveled(
    float *d_aabb_min,
    float *d_aabb_max,
    const int *d_left,
    const int *d_right,
    const int *d_parent,
    int n)
{
    // BFS on the host to group internal nodes by depth level (deepest first).
    int total = 2 * n - 1;
    std::vector<int> h_left(n - 1), h_right(n - 1), h_parent(total);
    cudaMemcpy(h_left.data(), d_left, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_right.data(), d_right, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_parent.data(), d_parent, total * sizeof(int), cudaMemcpyDeviceToHost);

    // Find the root of the tree
    int root = 0;
    for (int i = 0; i < n - 1; ++i)
        if (h_parent[i] == -1)
        {
            root = i;
            break;
        }

    std::vector<std::vector<int>> levels;
    std::queue<std::pair<int, int>> q; // (node, depth)
    q.push({root, 0});
    while (!q.empty())
    {
        auto [node, depth] = q.front();
        q.pop();
        if ((int)levels.size() <= depth)
            levels.resize(depth + 1);
        if (node < n - 1)
        { // internal
            levels[depth].push_back(node);
            q.push({h_left[node], depth + 1});
            q.push({h_right[node], depth + 1});
        }
    }

    // Process from deepest to shallowest
    constexpr int BLOCK = 256;
    for (int lvl = (int)levels.size() - 1; lvl >= 0; --lvl)
    {
        if (levels[lvl].empty())
            continue;
        int sz = (int)levels[lvl].size();

        int *d_level_nodes;
        cudaMalloc(&d_level_nodes, sz * sizeof(int));
        cudaMemcpy(d_level_nodes, levels[lvl].data(), sz * sizeof(int),
                   cudaMemcpyHostToDevice);

        refit_level_kernel<<<(sz + BLOCK - 1) / BLOCK, BLOCK>>>(
            d_aabb_min, d_aabb_max, d_left, d_right, d_level_nodes, sz);
        cudaDeviceSynchronize();
        cudaFree(d_level_nodes);
    }
}

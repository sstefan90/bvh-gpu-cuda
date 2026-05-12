// Karras binary tree! This is our stage 3 pipeline
//
// One thread per internal node i (0 .. N-2).  
// here's the plam for each thread:
//   find out the direction d of its range using delta (LCP length).
//   Finds the range endpoint j via exponential search + binary search.
//   Finds the split position gamma using binary search [rnge is i-j]
//   the we writes left/right child indices and parent pointers.

#include "karras.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

// Longest common prefix of codes[i] and codes[j].
__device__ __forceinline__ int delta(
    const uint32_t* __restrict__ codes,
    int i, int j, int n
) {
    if (j < 0 || j >= n) return -1;
    if (codes[i] == codes[j])
        return 32 + __clz((uint32_t)(i ^ j));
    return __clz(codes[i] ^ codes[j]);
}

__global__ void karras_kernel(
    const uint32_t* __restrict__ codes,
    int* __restrict__ node_left,
    int* __restrict__ node_right,
    int* __restrict__ node_parent,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;

    // Direction of the range for internal node i
    int d_fwd = delta(codes, i, i + 1, n);
    int d_bwd = delta(codes, i, i - 1, n);
    int d     = (d_fwd > d_bwd) ? 1 : -1;
    int delta_min = min(d_fwd, d_bwd);

    // Exponential search for upper bound on range length


    int lmax = 2;
    while (delta(codes, i, i + lmax * d, n) > delta_min)
        lmax <<= 1;

    // Binary search for the exact range endpoint

    int l = 0;
    for (int s = lmax >> 1; s >= 1; s >>= 1) {
        if (delta(codes, i, i + (l + s) * d, n) > delta_min)
            l += s;
    }
    int j = i + l * d;

    // Binary sarch for splt position gamma inside [min(i, j), m ax(i,j)]


    int delta_node = delta(codes, i, j, n);
    int s = 0, t = l;
    do {
        t = (t + 1) >> 1;
        if (delta(codes, i, i + (s + t) * d, n) > delta_node)
            s += t;
    } while (t > 1);


    // gamma is the last index in the left subtree
    int gamma = i + s * d + min(d, 0);

   
    // flat structure for node assignment: 2N-1, internal nodes: 0..N-2,  leaf nodes: N-1..2N-2
    int lc = (min(i, j) == gamma)     ? (n - 1 + gamma)     : gamma;
    int rc = (max(i, j) == gamma + 1) ? (n - 1 + gamma + 1) : (gamma + 1);

    node_left[i]  = lc;
    node_right[i] = rc;


    node_parent[lc] = i;
    node_parent[rc] = i;
}

// Scan host-side parent array to find the one internal node with parent == -1.
static int find_root(const int* h_parent, int n) {
    for (int i = 0; i < n - 1; ++i)
        if (h_parent[i] == -1) return i;
    return 0;  //edge case here, hoopefully it's ok
}

int karras_build(
    const uint32_t* d_codes,
    int*            d_left,
    int*            d_right,
    int*            d_parent,
    int             n
) {
    if (n <= 1) return 0;

    constexpr int BLOCK = 256;
    int grid = (n - 1 + BLOCK - 1) / BLOCK;
    karras_kernel<<<grid, BLOCK>>>(d_codes, d_left, d_right, d_parent, n);
    cudaDeviceSynchronize();

    // Copy parent array to host so we can find the parent
    int total_nodes = 2 * n - 1;
    std::vector<int> h_parent(total_nodes);
    cudaMemcpy(h_parent.data(), d_parent, total_nodes * sizeof(int),
               cudaMemcpyDeviceToHost);
    return find_root(h_parent.data(), n);
}

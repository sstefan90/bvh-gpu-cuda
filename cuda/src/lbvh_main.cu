// This is the  host driver for the single-GPU LBVH build pipeline

// Her's how we use it:  ./lbvh_build <triangles.bin> <output.lbvh.bin> [--leveled]

// Here are the stages in the pipeline
//   Stage 1  Morton encode       (morton_encode)
//   Stage 2  CUB DeviceRadixSort       (SortPairs)
//   Stage 3  Karras construction     (karras_build)
//   Stage 4  AABB refit       (refit_aabbs_atomic | refit_aabbs_leveled) (we can configure which one!)

// Timings for each stage are printed to stdout.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "lbvh_types.h"
#include "cuda_helpers.h"
#include "morton.cuh"
#include "karras.cuh"
#include "refit.cuh"

//IO helpers

static TriFileHeader read_tri_header(FILE* f) {
    TriFileHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fprintf(stderr, "Failed to read triangle file header\n");
        exit(EXIT_FAILURE);
    }
    if (hdr.magic != TRI_MAGIC) {
        fprintf(stderr, "Bad triangle file magic: 0x%08X\n", hdr.magic);
        exit(EXIT_FAILURE);
    }
    return hdr;
}

static void write_lbvh(const char* path,
                        const std::vector<LBVHNode>& nodes,
                        const std::vector<uint32_t>& sorted_prim_idx,
                        int root_idx) {
    FILE* f = fopen(path, "wb");
    if (!f) { perror(path); exit(EXIT_FAILURE); }

    uint32_t n_prims = (uint32_t)sorted_prim_idx.size();
    uint32_t n_nodes = (uint32_t)nodes.size();

    LBVHFileHeader hdr{LBVH_MAGIC, n_prims, n_nodes, root_idx};
    fwrite(&hdr,   sizeof(hdr), 1, f);
    fwrite(nodes.data(),           sizeof(LBVHNode), n_nodes, f);
    fwrite(sorted_prim_idx.data(), sizeof(uint32_t), n_prims, f);
    fclose(f);
}

//THIS is the main pipeline!

int main(int argc, char* argv[]) {

    // we gotta reaad in the input first ofc
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <triangles.bin> <output.lbvh.bin> [--leveled]\n",
                argv[0]);
        return EXIT_FAILURE;
    }
    const char* tri_path  = argv[1];
    const char* lbvh_path = argv[2];
    bool use_leveled = (argc >= 4 && std::string(argv[3]) == "--leveled");

    // let's ingest the triangle binary so we can build the bvh
    FILE* f = fopen(tri_path, "rb");
    if (!f) { perror(tri_path); return EXIT_FAILURE; }
    TriFileHeader tri_hdr = read_tri_header(f);
    int n = (int)tri_hdr.n_tris;
    printf("Loaded %d triangles from %s\n", n, tri_path);

    std::vector<Triangle> h_tris(n);
    if ((int)fread(h_tris.data(), sizeof(Triangle), n, f) != n) {
        fprintf(stderr, "Truncated triangle file\n"); return EXIT_FAILURE;
    }
    fclose(f);

    float3 scene_min = {tri_hdr.scene_min[0], tri_hdr.scene_min[1], tri_hdr.scene_min[2]};
    float3 scene_max = {tri_hdr.scene_max[0], tri_hdr.scene_max[1], tri_hdr.scene_max[2]};

    // let's allocate the buffers here, careful with the pointers and types
    int n_nodes = 2 * n - 1;

    Triangle* d_tris;
    uint32_t *d_codes, *d_codes_sorted;
    uint32_t *d_prim_idx, *d_prim_idx_sorted;
    int *d_left, *d_right, *d_parent, *d_counter;
    float *d_aabb_min, *d_aabb_max;

    CUDA_CHECK(cudaMalloc(&d_tris,            n * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&d_codes,           n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_codes_sorted,    n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_prim_idx,        n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_prim_idx_sorted, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_left,    (n-1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_right,   (n-1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_parent,  n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counter, (n-1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_aabb_min, n_nodes * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_aabb_max, n_nodes * 3 * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_parent,  -1, n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter,  0, (n-1)   * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_tris, h_tris.data(), n * sizeof(Triangle),
                          cudaMemcpyHostToDevice));

    CudaTimer timer;

    // finally, the fun part! Morton encode
    timer.begin();
    morton_encode(d_tris, d_codes, d_prim_idx, scene_min, scene_max, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    float t_morton = timer.end();
    printf("Stage 1 Morton encode:        %8.3f ms\n", t_morton);

    // stage 2 (if all goes well, CUB radix sort)
    void*  d_tmp    = nullptr;
    size_t tmp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_tmp, tmp_bytes,
        d_codes, d_codes_sorted, d_prim_idx, d_prim_idx_sorted, n);
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));

    timer.begin();
    cub::DeviceRadixSort::SortPairs(d_tmp, tmp_bytes,
        d_codes, d_codes_sorted, d_prim_idx, d_prim_idx_sorted, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    float t_sort = timer.end();
    printf("Stage 2 CUB radix sort:       %8.3f ms\n", t_sort);
    cudaFree(d_tmp);

    // build the Karras tree nicely
    timer.begin();
    int root_idx = karras_build(d_codes_sorted, d_left, d_right, d_parent, n);
    float t_karras = timer.end();
    printf("Stage 3 Karras construction:  %8.3f ms  (root=%d)\n", t_karras, root_idx);

    // we need to calculate the leaves for the AABB
    compute_leaf_aabbs(d_aabb_min, d_aabb_max, d_tris, d_prim_idx_sorted, n);

    // let's now refit the AABB
    timer.begin();
    if (use_leveled) {
        refit_aabbs_leveled(d_aabb_min, d_aabb_max, d_left, d_right, d_parent, n);
    } else {
        refit_aabbs_atomic(d_aabb_min, d_aabb_max, d_left, d_right, d_parent,
                           d_counter, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    float t_refit = timer.end();
    printf("Stage 4 AABB refit (%s): %8.3f ms\n",
           use_leveled ? "leveled" : "atomic ", t_refit);
    printf("Total GPU build time:         %8.3f ms\n",
           t_morton + t_sort + t_karras + t_refit);

    // ---- Copy results to host ----
    std::vector<int> h_left(n-1), h_right(n-1), h_parent(n_nodes);
    std::vector<float> h_aabb_min(n_nodes*3), h_aabb_max(n_nodes*3);
    std::vector<uint32_t> h_sorted_prim_idx(n);

    CUDA_CHECK(cudaMemcpy(h_left.data(),   d_left,   (n-1)*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_right.data(),  d_right,  (n-1)*sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(h_parent.data(), d_parent, n_nodes*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_aabb_min.data(), d_aabb_min, n_nodes*3*sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaMemcpy(h_aabb_max.data(), d_aabb_max, n_nodes*3*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sorted_prim_idx.data(), d_prim_idx_sorted, n*sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // ---- Assemble flat LBVHNode array ----
    std::vector<LBVHNode> nodes(n_nodes);
    for (int i = 0; i < n-1; ++i) {
        auto& nd = nodes[i];
        for (int k = 0; k < 3; ++k) {
            nd.aabb_min[k] = h_aabb_min[i*3+k];
            nd.aabb_max[k] = h_aabb_max[i*3+k];
        }
        nd.left     = h_left[i];
        nd.right    = h_right[i];
        nd.parent   = h_parent[i];
        nd.prim_idx = -1;
    }
    for (int i = 0; i < n; ++i) {
        int ni = n-1+i;
        auto& nd = nodes[ni];
        for (int k = 0; k < 3; ++k) {
            nd.aabb_min[k] = h_aabb_min[ni*3+k];
            nd.aabb_max[k] = h_aabb_max[ni*3+k];
        }
        nd.left     = -1;
        nd.right    = -1;
        nd.parent   = h_parent[ni];
        nd.prim_idx = (int)h_sorted_prim_idx[i];
    }

    // ---- Write LBVH binary ----
    write_lbvh(lbvh_path, nodes, h_sorted_prim_idx, root_idx);
    printf("Wrote LBVH (%d nodes) to %s\n", n_nodes, lbvh_path);

    // Cleanup
    cudaFree(d_tris); cudaFree(d_codes); cudaFree(d_codes_sorted);
    cudaFree(d_prim_idx); cudaFree(d_prim_idx_sorted);
    cudaFree(d_left); cudaFree(d_right); cudaFree(d_parent);
    cudaFree(d_counter); cudaFree(d_aabb_min); cudaFree(d_aabb_max);
    return EXIT_SUCCESS;
}

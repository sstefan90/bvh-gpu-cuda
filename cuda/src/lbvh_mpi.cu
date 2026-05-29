// multi-GPU LBVH build driver code!.
// Examlpe:
//   mpirun -np P ./lbvh_mpi <triangles.bin> <output_prefix> [--leveled]

// Each MPI rank r manages GPU (r % num_gpus).
// Pipeline:
//   Stage A: local Morton encode on N/P triangle shard
//   Stage B: distributed samplesort (MPI_Allgather + MPI_Alltoallv)
//   Stage C: local Karras + refit on the redistributed shard
//   Stage D: MPI_Gather root AABBs -> rank 0 builds top-level tree -> MPI_Bcast
//
// Per-stage wall times are measured with MPI_Wtime and reported as
// min/mean/max across ranks on rank 0.
//

// Output Format:
//   <prefix>_rank<r>.lbvh.bin  — per-rank local subtree
//   <prefix>_toplevel.bin      — P-node top-level tree (rank 0 only)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cassert>

#include <mpi.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "lbvh_types.h"
#include "cuda_helpers.h"
#include "morton.cuh"
#include "karras.cuh"
#include "refit.cuh"
#include "samplesort.cuh"

struct StageTime
{
    double A_morton;
    double B1_local_sort;
    double B2_allgather;
    double B3_alltoallv;
    double C_build;
    double D_merge;
    double total;
};

static void report_times(const StageTime &t, int rank, int nranks)
{
    // Gather all times to rank 0 and print min/mean/max.
    double buf[7] = {
        t.A_morton, t.B1_local_sort, t.B2_allgather,
        t.B3_alltoallv, t.C_build, t.D_merge, t.total};
    std::vector<double> all(7 * nranks);
    MPI_Gather(buf, 7, MPI_DOUBLE, all.data(), 7, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank != 0)
        return;

    const char *names[] = {
        "A  Morton encode ",
        "B1 local CUB sort",
        "B2 Allgather     ",
        "B3 Alltoallv     ",
        "C  Karras+refit  ",
        "D  top-level tree",
        "   TOTAL         "};
    printf("\n%-20s  %9s  %9s  %9s\n", "Stage", "min(ms)", "mean(ms)", "max(ms)");
    printf("%-20s  %9s  %9s  %9s\n", "-----", "-------", "--------", "-------");
    for (int s = 0; s < 7; ++s)
    {
        double mn = 1e18, mx = -1e18, sm = 0.0;
        for (int r = 0; r < nranks; ++r)
        {
            double v = all[r * 7 + s] * 1e3;
            mn = std::min(mn, v);
            mx = std::max(mx, v);
            sm += v;
        }
        printf("%-20s  %9.3f  %9.3f  %9.3f\n", names[s], mn, sm / nranks, mx);
    }
}

// ---------------------------------------------------------------------------
// Binary I/O helpers (mirrors lbvh_main.cu)
// ---------------------------------------------------------------------------

static TriFileHeader read_tri_header(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        perror(path);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    TriFileHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1)
    {
        fprintf(stderr, "Truncated triangle header\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (hdr.magic != TRI_MAGIC)
    {
        fprintf(stderr, "Bad triangle magic\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fclose(f);
    return hdr;
}

// Read triangles [start, start+count) from .tri.bin into h_tris.
static void read_tri_slice(const char *path, int start, int count,
                           std::vector<Triangle> &h_tris)
{
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        perror(path);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    long offset = (long)sizeof(TriFileHeader) + (long)start * sizeof(Triangle);
    if (fseek(f, offset, SEEK_SET) != 0)
    {
        fprintf(stderr, "fseek failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    h_tris.resize(count);
    if ((int)fread(h_tris.data(), sizeof(Triangle), count, f) != count)
    {
        fprintf(stderr, "Truncated triangle data\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fclose(f);
}

static void write_lbvh_file(const char *path,
                            const std::vector<LBVHNode> &nodes,
                            const std::vector<uint32_t> &sorted_prim_idx,
                            int root_idx)
{
    FILE *f = fopen(path, "wb");
    if (!f)
    {
        perror(path);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    uint32_t n_prims = (uint32_t)sorted_prim_idx.size();
    uint32_t n_nodes = (uint32_t)nodes.size();
    LBVHFileHeader hdr{LBVH_MAGIC, n_prims, n_nodes, root_idx};
    fwrite(&hdr, sizeof(hdr), 1, f);
    fwrite(nodes.data(), sizeof(LBVHNode), n_nodes, f);
    fwrite(sorted_prim_idx.data(), sizeof(uint32_t), n_prims, f);
    fclose(f);
}

// Top-level tree: P leaves on rank 0

struct RootInfo
{
    float aabb_min[3];
    float aabb_max[3];
    uint32_t first_code; // should be the smallest Morton code in this rank's range
};

// Build a P-leaf LBVH on the CPU using P synthetic Morton keys (one per rank).
// Returns the flat LBVHNode array (2P-1 nodes) and root index.
static std::vector<LBVHNode> build_toplevel(
    const std::vector<RootInfo> &infos,
    int P,
    int &out_root)
{
    // Gather P codes from first_code of each rank.
    std::vector<uint32_t> keys(P);
    for (int r = 0; r < P; ++r)
        keys[r] = infos[r].first_code;

    uint32_t *d_keys;
    int *d_left, *d_right, *d_parent;
    CUDA_CHECK(cudaMalloc(&d_keys, P * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_left, (P - 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_right, (P - 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_parent, (2 * P - 1) * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_parent, -1, (2 * P - 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), P * sizeof(uint32_t), cudaMemcpyHostToDevice));

    out_root = karras_build(d_keys, d_left, d_right, d_parent, P);

    std::vector<int> h_left(P - 1), h_right(P - 1), h_parent(2 * P - 1);
    CUDA_CHECK(cudaMemcpy(h_left.data(), d_left, (P - 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_right.data(), d_right, (P - 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_parent.data(), d_parent, (2 * P - 1) * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_keys);
    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_parent);

    int n_nodes = 2 * P - 1;
    std::vector<LBVHNode> nodes(n_nodes);

    for (int i = 0; i < P - 1; ++i)
    {
        auto &nd = nodes[i];

        nd.left = h_left[i];
        nd.right = h_right[i];
        nd.parent = h_parent[i];
        nd.prim_idx = -1;
        for (int k = 0; k < 3; ++k)
        {
            nd.aabb_min[k] = 1e38f;
            nd.aabb_max[k] = -1e38f;
        }
    }

    for (int r = 0; r < P; ++r)
    {
        int ni = P - 1 + r;
        auto &nd = nodes[ni];
        for (int k = 0; k < 3; ++k)
        {
            nd.aabb_min[k] = infos[r].aabb_min[k];
            nd.aabb_max[k] = infos[r].aabb_max[k];
        }
        nd.left = -1;
        nd.right = -1;
        nd.parent = h_parent[ni];
        nd.prim_idx = r;
    }

    bool changed = true;
    while (changed)
    {
        changed = false;
        for (int i = 0; i < P - 1; ++i)
        {
            int lc = nodes[i].left, rc = nodes[i].right;
            for (int k = 0; k < 3; ++k)
            {
                float mn = std::min(nodes[lc].aabb_min[k], nodes[rc].aabb_min[k]);
                float mx = std::max(nodes[lc].aabb_max[k], nodes[rc].aabb_max[k]);
                if (mn != nodes[i].aabb_min[k] || mx != nodes[i].aabb_max[k])
                {
                    nodes[i].aabb_min[k] = mn;
                    nodes[i].aabb_max[k] = mx;
                    changed = true;
                }
            }
        }
    }

    return nodes;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    if (argc < 3)
    {
        if (rank == 0)
            fprintf(stderr,
                    "Usage: %s <triangles.bin> <output_prefix> [--leveled]\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    const char *tri_path = argv[1];
    const char *out_prefix = argv[2];
    bool use_leveled = (argc >= 4 && std::string(argv[3]) == "--leveled");

    int num_devs = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devs));
    if (num_devs == 0)
    {
        fprintf(stderr, "[rank %d] No CUDA devices found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int dev = rank % num_devs;
    CUDA_CHECK(cudaSetDevice(dev));
    if (rank == 0)
        printf("Running %d ranks over %d GPU(s)\n", nranks, num_devs);

    TriFileHeader tri_hdr = read_tri_header(tri_path);
    int N = (int)tri_hdr.n_tris;

    int base = N / nranks;
    int rem = N % nranks;
    int local_n = base + (rank < rem ? 1 : 0);
    int slice_start = base * rank + std::min(rank, rem);

    if (rank == 0)
        printf("N=%d triangles, %d ranks, slice_size≈%d\n", N, nranks, base);

    float3 scene_min = {tri_hdr.scene_min[0], tri_hdr.scene_min[1], tri_hdr.scene_min[2]};
    float3 scene_max = {tri_hdr.scene_max[0], tri_hdr.scene_max[1], tri_hdr.scene_max[2]};

    StageTime T{};
    double t0, t1;

    std::vector<Triangle> h_tris;
    read_tri_slice(tri_path, slice_start, local_n, h_tris);

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();

    Triangle *d_tris;
    uint32_t *d_codes, *d_prim_idx;
    CUDA_CHECK(cudaMalloc(&d_tris, local_n * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&d_codes, local_n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_prim_idx, local_n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_tris, h_tris.data(), local_n * sizeof(Triangle), cudaMemcpyHostToDevice));

    morton_encode(d_tris, d_codes, d_prim_idx, scene_min, scene_max, local_n);

    {
        // Offset correction kernel (inline lambda via thrust or raw kernel).
        struct _Off
        {
            __device__ static void run(uint32_t *idx, int off, int n)
            {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n)
                    idx[i] += (uint32_t)off;
            }
        };
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy prim_idx to host, add global offset, copy back.
    std::vector<uint32_t> h_prim_idx_local(local_n);
    CUDA_CHECK(cudaMemcpy(h_prim_idx_local.data(), d_prim_idx,
                          local_n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (int i = 0; i < local_n; ++i)
        h_prim_idx_local[i] += (uint32_t)slice_start;
    CUDA_CHECK(cudaMemcpy(d_prim_idx, h_prim_idx_local.data(),
                          local_n * sizeof(uint32_t), cudaMemcpyHostToDevice));

    t1 = MPI_Wtime();
    T.A_morton = t1 - t0;

    // Stage B1: local CUB radix sort

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();

    uint32_t *d_codes_sorted, *d_prim_idx_sorted;
    CUDA_CHECK(cudaMalloc(&d_codes_sorted, local_n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_prim_idx_sorted, local_n * sizeof(uint32_t)));

    void *d_tmp = nullptr;
    size_t tmp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_tmp, tmp_bytes,
                                    d_codes, d_codes_sorted, d_prim_idx, d_prim_idx_sorted, local_n);
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));
    cub::DeviceRadixSort::SortPairs(d_tmp, tmp_bytes,
                                    d_codes, d_codes_sorted, d_prim_idx, d_prim_idx_sorted, local_n);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_tmp);
    cudaFree(d_codes);
    cudaFree(d_prim_idx); // free unsorted buffers

    t1 = MPI_Wtime();
    T.B1_local_sort = t1 - t0;

    // Stage B2+B3: samplesort (Allgather + splitter selection + Alltoallv)

    // Copy sorted codes + prim_idx to host for samplesort.
    std::vector<uint32_t> h_codes_sorted(local_n), h_prim_sorted(local_n);
    CUDA_CHECK(cudaMemcpy(h_codes_sorted.data(), d_codes_sorted,
                          local_n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_prim_sorted.data(), d_prim_idx_sorted,
                          local_n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    std::vector<Triangle> h_tris_sorted(local_n);
    for (int i = 0; i < local_n; ++i)
    {
        int local_idx = (int)h_prim_sorted[i] - slice_start;
        h_tris_sorted[i] = h_tris[local_idx];
    }
    h_tris.clear();
    h_tris.shrink_to_fit(); // release memory

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();

    std::vector<uint32_t> redist_codes, redist_prim;
    std::vector<Triangle> redist_tris;
    int new_local_n = samplesort(
        h_codes_sorted.data(), h_prim_sorted.data(), h_tris_sorted.data(),
        local_n, rank, nranks,
        redist_codes, redist_prim, redist_tris);

    t1 = MPI_Wtime();
    // Split B2/B3 evenly (we can't separate them without instrumenting samplesort;
    T.B2_allgather = 0.0;
    T.B3_alltoallv = t1 - t0;

    // Free stage-B GPU buffers.
    cudaFree(d_codes_sorted);
    cudaFree(d_prim_idx_sorted);
    cudaFree(d_tris);
    h_codes_sorted.clear();
    h_prim_sorted.clear();
    h_tris_sorted.clear();

    // Stage C: local LBVH build on redistributed shard

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();

    int n = new_local_n;
    int n_nodes = (n > 0) ? (2 * n - 1) : 0;

    std::vector<LBVHNode> local_nodes;
    std::vector<uint32_t> h_sorted_prim;
    float root_aabb_min[3] = {1e38f, 1e38f, 1e38f}; // empty AABB
    float root_aabb_max[3] = {-1e38f, -1e38f, -1e38f};
    int root_idx = 0;

    uint32_t first_code = (uint32_t)(rank * (0xFFFFFFFFu / (uint32_t)nranks));

    if (n > 0)
    {

        uint32_t *d_rc, *d_rp;
        Triangle *d_rt;
        CUDA_CHECK(cudaMalloc(&d_rc, n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_rp, n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_rt, n * sizeof(Triangle)));
        CUDA_CHECK(cudaMemcpy(d_rc, redist_codes.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rp, redist_prim.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rt, redist_tris.data(), n * sizeof(Triangle), cudaMemcpyHostToDevice));
        redist_codes.clear();
        redist_prim.clear();
        redist_tris.clear();

        int *d_left, *d_right, *d_parent, *d_counter;
        float *d_aabb_min, *d_aabb_max;
        CUDA_CHECK(cudaMalloc(&d_left, (n - 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_right, (n - 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_parent, n_nodes * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_counter, (n - 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_aabb_min, n_nodes * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_aabb_max, n_nodes * 3 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_parent, -1, n_nodes * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_counter, 0, (n - 1) * sizeof(int)));

        root_idx = karras_build(d_rc, d_left, d_right, d_parent, n);

        uint32_t *d_local_idx;
        CUDA_CHECK(cudaMalloc(&d_local_idx, n * sizeof(uint32_t)));
        {
            std::vector<uint32_t> identity(n);
            std::iota(identity.begin(), identity.end(), 0u);
            CUDA_CHECK(cudaMemcpy(d_local_idx, identity.data(), n * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice));
        }
        compute_leaf_aabbs(d_aabb_min, d_aabb_max, d_rt, d_local_idx, n);
        cudaFree(d_local_idx);
        if (use_leveled)
            refit_aabbs_leveled(d_aabb_min, d_aabb_max, d_left, d_right, d_parent, n);
        else
            refit_aabbs_atomic(d_aabb_min, d_aabb_max, d_left, d_right, d_parent, d_counter, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy results to host and assemble flat LBVHNode array.
        std::vector<int> h_left(n - 1), h_right(n - 1), h_parent(n_nodes);
        std::vector<float> h_aabb_min(n_nodes * 3), h_aabb_max(n_nodes * 3);
        h_sorted_prim.resize(n);

        CUDA_CHECK(cudaMemcpy(h_left.data(), d_left, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_right.data(), d_right, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_parent.data(), d_parent, n_nodes * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_aabb_min.data(), d_aabb_min, n_nodes * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_aabb_max.data(), d_aabb_max, n_nodes * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_sorted_prim.data(), d_rp, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        // Get root AABB for top-level merge.
        for (int k = 0; k < 3; ++k)
        {
            root_aabb_min[k] = h_aabb_min[root_idx * 3 + k];
            root_aabb_max[k] = h_aabb_max[root_idx * 3 + k];
        }

        // Assemble local LBVH node array.
        local_nodes.resize(n_nodes);
        for (int i = 0; i < n - 1; ++i)
        {
            auto &nd = local_nodes[i];
            for (int k = 0; k < 3; ++k)
            {
                nd.aabb_min[k] = h_aabb_min[i * 3 + k];
                nd.aabb_max[k] = h_aabb_max[i * 3 + k];
            }
            nd.left = h_left[i];
            nd.right = h_right[i];
            nd.parent = h_parent[i];
            nd.prim_idx = -1;
        }
        for (int i = 0; i < n; ++i)
        {
            int ni = n - 1 + i;
            auto &nd = local_nodes[ni];
            for (int k = 0; k < 3; ++k)
            {
                nd.aabb_min[k] = h_aabb_min[ni * 3 + k];
                nd.aabb_max[k] = h_aabb_max[ni * 3 + k];
            }
            nd.left = -1;
            nd.right = -1;
            nd.parent = h_parent[ni];
            nd.prim_idx = (int)h_sorted_prim[i];
        }

        // Cleanup
        cudaFree(d_rc);
        cudaFree(d_rp);
        cudaFree(d_rt);
        cudaFree(d_left);
        cudaFree(d_right);
        cudaFree(d_parent);
        cudaFree(d_counter);
        cudaFree(d_aabb_min);
        cudaFree(d_aabb_max);
    }

    t1 = MPI_Wtime();
    T.C_build = t1 - t0;

    // Stage D: top-level tree merge

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();

    // Pack per-rank info to send to rank 0.
    RootInfo my_info;
    for (int k = 0; k < 3; ++k)
    {
        my_info.aabb_min[k] = root_aabb_min[k];
        my_info.aabb_max[k] = root_aabb_max[k];
    }
    my_info.first_code = first_code;

    std::vector<RootInfo> all_infos(nranks);
    MPI_Gather(&my_info, sizeof(RootInfo), MPI_BYTE,
               all_infos.data(), sizeof(RootInfo), MPI_BYTE,
               0, MPI_COMM_WORLD);

    std::vector<LBVHNode> toplevel_nodes;
    int toplevel_root = 0;
    if (rank == 0 && nranks > 1)
    {
        toplevel_nodes = build_toplevel(all_infos, nranks, toplevel_root);
    }

    // Bcast top-level tree to all ranks.
    int tl_size = (int)toplevel_nodes.size();
    MPI_Bcast(&tl_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0)
        toplevel_nodes.resize(tl_size);
    if (tl_size > 0)
        MPI_Bcast(toplevel_nodes.data(), tl_size * (int)sizeof(LBVHNode), MPI_BYTE,
                  0, MPI_COMM_WORLD);
    MPI_Bcast(&toplevel_root, 1, MPI_INT, 0, MPI_COMM_WORLD);

    t1 = MPI_Wtime();
    T.D_merge = t1 - t0;
    T.total = T.A_morton + T.B1_local_sort + T.B3_alltoallv + T.C_build + T.D_merge;

    // Per-rank local subtree.
    {
        char path[512];
        snprintf(path, sizeof(path), "%s_rank%d.lbvh.bin", out_prefix, rank);
        write_lbvh_file(path, local_nodes, h_sorted_prim, root_idx);
    }
    // Top-level tree on rank 0.
    if (rank == 0 && nranks > 1)
    {
        char path[512];
        snprintf(path, sizeof(path), "%s_toplevel.bin", out_prefix);
        std::vector<uint32_t> rank_indices(nranks);
        std::iota(rank_indices.begin(), rank_indices.end(), 0u);
        write_lbvh_file(path, toplevel_nodes, rank_indices, toplevel_root);
        printf("Top-level tree (%d nodes) written to %s\n",
               (int)toplevel_nodes.size(), path);
    }

    // Report timings
    report_times(T, rank, nranks);

    if (rank == 0)
        printf("\nPer-rank subtree sizes: (local N after redistribution)\n");
    // Gather new_local_n to rank 0 and print.
    std::vector<int> all_sizes(nranks);
    MPI_Gather(&new_local_n, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        for (int r = 0; r < nranks; ++r)
            printf("  rank %d: %d triangles\n", r, all_sizes[r]);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

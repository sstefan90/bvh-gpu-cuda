// MPI unit test for samplesort.
//
// Run with:
//   mpirun -np 2 ./build/test_samplesort
//   mpirun -np 4 ./build/test_samplesort
//
// For each rank count the test:
//   1. Assigns each rank a contiguous block of synthetic Morton codes
//      (e.g. rank r gets codes [r*100, r*100+100) in random order within block).
//   2. Locally sorts the block (std::sort on host, no GPU needed for this test).
//   3. Runs samplesort to redistribute.
//   4. Asserts:
//      a. Global sort order (no rank holds a code that belongs to another bucket).
//      b. No records dropped or duplicated (gather total counts to rank 0).
//      c. Received records are sorted.

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cuda_runtime.h>    // needed by samplesort.cuh -> lbvh_types.h

#include "samplesort.cuh"

static int fail_count = 0;

#define EXPECT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "[rank %d] FAIL: %s\n", rank, msg); \
        ++fail_count; \
    } \
} while(0)

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Initialize CUDA device (required since samplesort.cu is compiled with -dc).
    int num_devs = 0;
    cudaGetDeviceCount(&num_devs);
    if (num_devs > 0) cudaSetDevice(rank % num_devs);

    const int local_n = 200;  // codes per rank before redistribution

    // Generate synthetic Morton codes: rank r owns the range
    // [r * local_n * 10, r * local_n * 10 + local_n * 10) with step 10,
    // shuffled so they are NOT pre-sorted.
    std::vector<uint32_t> h_codes(local_n);
    for (int i = 0; i < local_n; ++i)
        h_codes[i] = (uint32_t)(rank * local_n * 10 + i * 10);
    // Shuffle deterministically.
    for (int i = local_n - 1; i > 0; --i) {
        int j = (i * 1103515245 + 12345) % (i + 1);
        if (j < 0) j += (i + 1);
        std::swap(h_codes[i], h_codes[j]);
    }
    // Local sort (simulating post-CUB state).
    std::sort(h_codes.begin(), h_codes.end());

    // prim_idx: identity (just track code origin for verification).
    std::vector<uint32_t> h_prim(local_n);
    std::iota(h_prim.begin(), h_prim.end(), (uint32_t)(rank * local_n));

    // Triangles: dummy (zeros).
    std::vector<Triangle> h_tris(local_n);
    for (auto& t : h_tris)
        for (int k = 0; k < 9; ++k) reinterpret_cast<float*>(&t)[k] = 0.0f;

    // Run samplesort.
    std::vector<uint32_t> out_codes, out_prim;
    std::vector<Triangle> out_tris;
    int new_n = samplesort(
        h_codes.data(), h_prim.data(), h_tris.data(),
        local_n, rank, nranks,
        out_codes, out_prim, out_tris
    );

    // Test a: received records are sorted.
    for (int i = 1; i < new_n; ++i)
        EXPECT(out_codes[i-1] <= out_codes[i], "output not sorted");

    // Test b: global sort order — max of rank r must be <= min of rank r+1.
    uint32_t local_min = (new_n > 0) ? out_codes.front() : 0xFFFFFFFFu;
    uint32_t local_max = (new_n > 0) ? out_codes.back()  : 0u;

    std::vector<uint32_t> all_min(nranks), all_max(nranks);
    MPI_Allgather(&local_min, 1, MPI_UINT32_T, all_min.data(), 1, MPI_UINT32_T, MPI_COMM_WORLD);
    MPI_Allgather(&local_max, 1, MPI_UINT32_T, all_max.data(), 1, MPI_UINT32_T, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int r = 0; r < nranks - 1; ++r) {
            bool ordered = (all_max[r] <= all_min[r+1]);
            if (!ordered) {
                fprintf(stderr, "[rank 0] FAIL: global order broken at r=%d: max[%d]=%u > min[%d]=%u\n",
                        r, r, all_max[r], r+1, all_min[r+1]);
                ++fail_count;
            }
        }
    }

    // Test c: no records dropped or duplicated — total count must equal nranks*local_n.
    int total_received = 0;
    MPI_Reduce(&new_n, &total_received, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        int expected = nranks * local_n;
        if (total_received != expected) {
            fprintf(stderr, "[rank 0] FAIL: total count %d != expected %d\n",
                    total_received, expected);
            ++fail_count;
        }
    }

    // Aggregate failure count across all ranks.
    int global_fails = 0;
    MPI_Reduce(&fail_count, &global_fails, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (global_fails == 0)
            printf("[test_samplesort] P=%d: ALL TESTS PASSED\n", nranks);
        else
            printf("[test_samplesort] P=%d: %d FAILURE(S)\n", nranks, global_fails);
    }

    MPI_Finalize();
    return (global_fails > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}

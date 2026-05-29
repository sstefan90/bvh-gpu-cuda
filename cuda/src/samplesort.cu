// Distributed samplesort (Stage B) for multi-GPU LBVH build.

#include "samplesort.cuh"
#include <mpi.h>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cassert>

static void select_splitters(
    const std::vector<uint32_t> &gathered_samples,
    int nranks,
    std::vector<uint32_t> &splitters)
{
    std::vector<uint32_t> sorted = gathered_samples;
    std::sort(sorted.begin(), sorted.end());

    splitters.resize(nranks - 1);
    int total = (int)sorted.size();
    for (int k = 0; k < nranks - 1; ++k)
    {

        int idx = ((2 * k + 1) * nranks) / 2;
        if (idx >= total)
            idx = total - 1;
        splitters[k] = sorted[idx];
    }
}

// Returns the bucket index [0, nranks) for a given code given sorted splitters.
static inline int bucket_for(uint32_t code, const std::vector<uint32_t> &splitters)
{
    int lo = 0, hi = (int)splitters.size();
    while (lo < hi)
    {
        int mid = (lo + hi) / 2;
        if (code < splitters[mid])
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo;
}

int samplesort(
    const uint32_t *h_codes,
    const uint32_t *h_prim_idx,
    const Triangle *h_tris,
    int local_n,
    int rank,
    int nranks,
    std::vector<uint32_t> &out_codes,
    std::vector<uint32_t> &out_prim_idx,
    std::vector<Triangle> &out_tris)
{

    if (nranks == 1)
    {
        out_codes.assign(h_codes, h_codes + local_n);
        out_prim_idx.assign(h_prim_idx, h_prim_idx + local_n);
        out_tris.assign(h_tris, h_tris + local_n);
        return local_n;
    }

    int n_samples_per_rank = nranks - 1;
    std::vector<uint32_t> local_samples(n_samples_per_rank);
    for (int k = 0; k < n_samples_per_rank; ++k)
    {
        // Evenly spaced positions in [0, local_n): index = (k+1)*local_n/nranks
        int idx = (int)(((long long)(k + 1) * local_n) / nranks);
        if (idx >= local_n)
            idx = local_n - 1;
        local_samples[k] = (local_n > 0) ? h_codes[idx] : 0u;
    }

    // MPI_Allgather samples
    int total_samples = nranks * n_samples_per_rank;
    std::vector<uint32_t> gathered(total_samples);
    MPI_Allgather(
        local_samples.data(), n_samples_per_rank, MPI_UINT32_T,
        gathered.data(), n_samples_per_rank, MPI_UINT32_T,
        MPI_COMM_WORLD);

    // select splitters (same result on every rank)
    std::vector<uint32_t> splitters;
    select_splitters(gathered, nranks, splitters);

    // partition local records into P buckets
    std::vector<int> send_counts(nranks, 0);
    for (int i = 0; i < local_n; ++i)
        send_counts[bucket_for(h_codes[i], splitters)]++;

    // Build send displacements (in element units).
    std::vector<int> send_displs(nranks, 0);
    for (int r = 1; r < nranks; ++r)
        send_displs[r] = send_displs[r - 1] + send_counts[r - 1];

    // Pack into SortRecord buffer ordered by bucket.
    std::vector<SortRecord> send_buf(local_n);
    std::vector<int> fill(nranks, 0);
    for (int i = 0; i < local_n; ++i)
    {
        int b = bucket_for(h_codes[i], splitters);
        int pos = send_displs[b] + fill[b]++;
        send_buf[pos] = {h_codes[i], h_prim_idx[i], h_tris[i]};
    }

    // exchange bucket sizes
    std::vector<int> recv_counts(nranks);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    std::vector<int> recv_displs(nranks, 0);
    for (int r = 1; r < nranks; ++r)
        recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];
    int new_local_n = recv_displs[nranks - 1] + recv_counts[nranks - 1];

    std::vector<int> send_bytes(nranks), send_byte_displs(nranks);
    std::vector<int> recv_bytes(nranks), recv_byte_displs(nranks);
    constexpr int REC = (int)sizeof(SortRecord);
    for (int r = 0; r < nranks; ++r)
    {
        send_bytes[r] = send_counts[r] * REC;

        send_byte_displs[r] = (send_counts[r] > 0) ? send_displs[r] * REC : 0;
        recv_bytes[r] = recv_counts[r] * REC;
        recv_byte_displs[r] = (recv_counts[r] > 0) ? recv_displs[r] * REC : 0;
    }

    // MPI_Alltoallv of packed records
    std::vector<SortRecord> recv_buf(new_local_n > 0 ? new_local_n : 1);

    MPI_Alltoallv(
        send_buf.data(), send_bytes.data(), send_byte_displs.data(), MPI_BYTE,
        recv_buf.data(), recv_bytes.data(), recv_byte_displs.data(), MPI_BYTE,
        MPI_COMM_WORLD);
    recv_buf.resize(new_local_n);

    // Unpack and sort received records by Morton code
    std::sort(recv_buf.begin(), recv_buf.end(),
              [](const SortRecord &a, const SortRecord &b)
              { return a.code < b.code; });

    out_codes.resize(new_local_n);
    out_prim_idx.resize(new_local_n);
    out_tris.resize(new_local_n);
    for (int i = 0; i < new_local_n; ++i)
    {
        out_codes[i] = recv_buf[i].code;
        out_prim_idx[i] = recv_buf[i].prim_idx;
        out_tris[i] = recv_buf[i].tri;
    }
    return new_local_n;
}

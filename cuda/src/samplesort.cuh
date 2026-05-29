#pragma once
#include <cstdint>
#include <vector>
#include "lbvh_types.h"

struct SortRecord
{
    uint32_t code;
    uint32_t prim_idx;
    Triangle tri;
};
static_assert(sizeof(SortRecord) == 44, "SortRecord size mismatch");

int samplesort(
    const uint32_t *h_codes,
    const uint32_t *h_prim_idx,
    const Triangle *h_tris,
    int local_n,
    int rank,
    int nranks,
    std::vector<uint32_t> &out_codes,
    std::vector<uint32_t> &out_prim_idx,
    std::vector<Triangle> &out_tris);

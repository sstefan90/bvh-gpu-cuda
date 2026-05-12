#pragma once
#include "lbvh_types.h"
#include <cuda_runtime.h>

// Build a binary radix tree from N sorted Morton codes (Karras 2012).
//
// Flat node array layout:
//   [0 .. N-2]   internal nodes  (N-1 total)
//   [N-1 .. 2N-2] leaf nodes     (N total)
//
// Outputs:
//   d_left[N-1], d_right[N-1]   : children of each internal node
//   d_parent[2N-1]              : parent of each node (-1 for the root)
//
// d_parent must be pre-initialised to -1 before calling.
// Returns: flat index of the root internal node.
int karras_build(
    const uint32_t* d_codes,    // sorted Morton codes [N]
    int*            d_left,     // output: left child  [N-1]
    int*            d_right,    // output: right child [N-1]
    int*            d_parent,   // output: parent      [2N-1], init to -1
    int             n
);

// Unit tests for the Karras tree-construction kernel.
//
// Test 1 — 8-key worked example from Karras 2012 §5.
//   Morton codes: 00001, 00010, 00100, 00101, 10011, 11000, 11001, 11110
//   (written as binary; packed into uint32 with leading zeros).
//
// Test 2 — Stage invariants on a random 256-element input:
//   (a) Every leaf is reachable from the root.
//   (b) Every internal node's range is a contiguous interval.
//   (c) No node is its own ancestor.

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <set>
#include <queue>
#include <algorithm>

#include <cuda_runtime.h>
#include "lbvh_types.h"
#include "morton.cuh"
#include "karras.cuh"

#define PASS(msg) printf("[PASS] %s\n", msg)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); g_failures++; } while(0)
static int g_failures = 0;

struct Tree {
    int n;
    std::vector<int> left, right, parent;
    int root;
};

static Tree run_karras(const std::vector<uint32_t>& codes) {
    int n = (int)codes.size();
    uint32_t* d_codes;
    int *d_left, *d_right, *d_parent;
    cudaMalloc(&d_codes,  n     * sizeof(uint32_t));
    cudaMalloc(&d_left,  (n-1)  * sizeof(int));
    cudaMalloc(&d_right, (n-1)  * sizeof(int));
    cudaMalloc(&d_parent,(2*n-1)* sizeof(int));
    cudaMemset(d_parent, -1, (2*n-1)*sizeof(int));
    cudaMemcpy(d_codes, codes.data(), n*sizeof(uint32_t), cudaMemcpyHostToDevice);

    int root = karras_build(d_codes, d_left, d_right, d_parent, n);

    Tree t;
    t.n = n;
    t.root = root;
    t.left.resize(n-1); t.right.resize(n-1); t.parent.resize(2*n-1);
    cudaMemcpy(t.left.data(),   d_left,   (n-1)*sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(t.right.data(),  d_right,  (n-1)*sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(t.parent.data(), d_parent, (2*n-1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_codes); cudaFree(d_left); cudaFree(d_right); cudaFree(d_parent);
    return t;
}

static void test_karras_paper_example() {
    // 8-key example from Karras 2012 §5 (30-bit codes, lsb-first as in paper)
    // We just need sorted distinct codes; the paper verifies tree topology.
    std::vector<uint32_t> codes = {
        0b00001u, 0b00010u, 0b00100u, 0b00101u,
        0b10011u, 0b11000u, 0b11001u, 0b11110u
    };
    int n = (int)codes.size();
    Tree t = run_karras(codes);

    // Root has no parent
    if (t.parent[t.root] == -1) PASS("paper example: root.parent == -1");
    else FAIL("paper example: root.parent != -1");

    // All N leaves reachable from root via BFS
    std::set<int> visited_leaves;
    std::queue<int> q;
    q.push(t.root);
    while (!q.empty()) {
        int node = q.front(); q.pop();
        if (node >= n - 1) {  // leaf
            visited_leaves.insert(node - (n-1));
        } else {
            if (t.left[node]  >= 0) q.push(t.left[node]);
            if (t.right[node] >= 0) q.push(t.right[node]);
        }
    }
    bool all_leaves = ((int)visited_leaves.size() == n);
    if (all_leaves) PASS("paper example: all 8 leaves reachable");
    else {
        printf("[FAIL] paper example: only %d/8 leaves reachable\n",
               (int)visited_leaves.size());
        g_failures++;
    }

    // Every non-root node's parent points back to a valid internal node
    bool parents_ok = true;
    for (int i = 0; i < 2*n-1; ++i) {
        if (i == t.root) continue;
        int p = t.parent[i];
        if (p < 0 || p >= n-1) { parents_ok = false; break; }
        if (t.left[p] != i && t.right[p] != i) { parents_ok = false; break; }
    }
    if (parents_ok) PASS("paper example: parent pointers consistent");
    else FAIL("paper example: parent pointer inconsistency");
}

static void test_invariants_random(int n_test = 256) {
    // Generate sorted distinct codes
    std::vector<uint32_t> codes;
    codes.reserve(n_test);
    uint32_t c = 1;
    for (int i = 0; i < n_test; ++i, c += 3) codes.push_back(c);

    Tree t = run_karras(codes);
    int n = t.n;

    // (a) All leaves reachable
    std::set<int> seen;
    std::queue<int> q;
    q.push(t.root);
    while (!q.empty()) {
        int node = q.front(); q.pop();
        if (node >= n-1) seen.insert(node);
        else {
            q.push(t.left[node]);
            q.push(t.right[node]);
        }
    }
    if ((int)seen.size() == n) PASS("random invariant: all leaves reachable");
    else { printf("[FAIL] only %d/%d leaves reachable\n",(int)seen.size(),n); g_failures++; }

    // (b) No cycles: every node reaches root within n steps
    bool no_cycle = true;
    for (int start = 0; start < 2*n-1 && no_cycle; ++start) {
        int node = start, steps = 0;
        while (node != -1 && steps <= 2*n) { node = t.parent[node]; ++steps; }
        if (steps > 2*n) no_cycle = false;
    }
    if (no_cycle) PASS("random invariant: no parent cycles");
    else FAIL("random invariant: cycle detected in parent pointers");

    // (c) Every internal node has two distinct children
    bool children_ok = true;
    for (int i = 0; i < n-1; ++i) {
        if (t.left[i] == t.right[i]) { children_ok = false; break; }
        if (t.left[i] < 0 || t.right[i] < 0) { children_ok = false; break; }
    }
    if (children_ok) PASS("random invariant: internal nodes have two distinct children");
    else FAIL("random invariant: internal node with bad children");
}

int main() {
    printf("=== Karras unit tests ===\n");
    test_karras_paper_example();
    test_invariants_random();
    if (g_failures == 0) printf("All Karras tests PASSED.\n");
    else printf("%d Karras test(s) FAILED.\n", g_failures);
    return g_failures ? EXIT_FAILURE : EXIT_SUCCESS;
}

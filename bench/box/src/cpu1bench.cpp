// cpu1bench: how well does ggml's CPU backend thread the expert FFN of Qwen3.6 (the ~31 ms CPU phase of a 58 ms decode round),
// and what would a task-parallel layout buy BEFORE anyone writes it? Real shapes and types: gate / up 2048 -> 512 Q2_K,
// down 512 -> 2048 Q3_K, 256 experts; a "pair" = one (token, expert) FFN = 3 matvecs = ~3.1 M multiply-adds.
//   ggml   one graph (MUL_MAT_ID gate, MUL_MAT_ID up, SWIGLU, MUL_MAT_ID down) over all pairs, n threads: every op splits rows
//          over the threads and ends in a barrier (what the server does today).
//   task   n persistent workers, each owns pairs/n whole pairs and runs the same graph single-threaded; one spin barrier per
//          layer (the CPU1 proposal, emulated with stock ggml ops, no new kernel).
// Prints microseconds per layer call and the speedup over 1 thread; pairs 4 = one decode token at ~50% cache hit, 12 = an MTP
// verify batch of 3, 8 / 24 = no cache. Expert ids are re-drawn every call (the miss set moves, so do the weights touched).
// CONTROL: `cpu1bench ITERS fixed` draws the ids ONCE, so the same expert bytes are re-read from the CPU caches (4 pairs = 4.6 MB
// fit the 12 MB L3). If the 6-thread scaling jumps there, the stall in the normal mode is DRAM bandwidth, not the cores.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

static const int N_EMBD = 2048, N_FF = 512, N_EXPERT = 256;
static bool g_fixed_ids = false;

struct weights {
    ggml_context * ctx; ggml_backend_buffer_t buf;
    ggml_tensor * gate, * up, * down;
};

static void fill_experts(ggml_tensor * t, int64_t n_per_row, int64_t nrows, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::vector<float> src(n_per_row * nrows);
    for (float & v : src) v = nd(rng);
    const size_t nb = ggml_row_size(t->type, n_per_row) * nrows;
    std::vector<char> q(nb);
    ggml_quantize_chunk(t->type, src.data(), q.data(), 0, nrows, n_per_row, nullptr);
    for (int e = 0; e < N_EXPERT; ++e) ggml_backend_tensor_set(t, q.data(), e * nb, nb);   // same values, distinct memory
}

struct layer {             // one FFN graph over n_pairs (token, expert) pairs
    ggml_context * ctx = nullptr; ggml_backend_buffer_t buf = nullptr; ggml_backend_t be = nullptr;
    ggml_cgraph * gf = nullptr; ggml_tensor * x = nullptr, * ids = nullptr; int n_pairs = 0;
    std::mt19937 rng; std::vector<int32_t> idbuf;

    void init(const weights & w, int pairs, int n_threads, unsigned seed) {
        n_pairs = pairs; rng.seed(seed); idbuf.resize(pairs);
        be = ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(be, n_threads);
        ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
        ctx = ggml_init(ip);
        x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_EMBD, 1, 1);          // one token's hidden state, broadcast over its experts
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, pairs, 1);
        ggml_tensor * g = ggml_mul_mat_id(ctx, w.gate, x, ids);
        ggml_tensor * u = ggml_mul_mat_id(ctx, w.up,   x, ids);
        ggml_tensor * a = ggml_swiglu_split(ctx, g, u);
        ggml_tensor * d = ggml_mul_mat_id(ctx, w.down, a, ids);
        gf = ggml_new_graph(ctx); ggml_build_forward_expand(gf, d);
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        std::vector<float> xv(N_EMBD); std::normal_distribution<float> nd(0.0f, 1.0f); for (float & v : xv) v = nd(rng);
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
    }
    bool drawn = false;
    void run() {
        for (int i = 0; i < n_pairs && !(g_fixed_ids && drawn); ++i) {       // distinct experts, as top-k routing gives
            int e; bool dup; do { e = rng() % N_EXPERT; dup = false; for (int j = 0; j < i; ++j) dup |= idbuf[j] == e; } while (dup);
            idbuf[i] = e;
        }
        drawn = true;
        ggml_backend_tensor_set(ids, idbuf.data(), 0, ggml_nbytes(ids));
        ggml_backend_graph_compute(be, gf);
    }
};

static double bench_ggml(const weights & w, int pairs, int nt, int iters) {
    layer L; L.init(w, pairs, nt, 1);
    for (int i = 0; i < iters / 10; ++i) L.run();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) L.run();
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
}

static double bench_task(const weights & w, int pairs, int nt, int iters) {
    const int nw = std::min(nt, pairs);
    std::vector<layer> Ls(nw);
    for (int i = 0; i < nw; ++i) Ls[i].init(w, pairs / nw + (i < pairs % nw ? 1 : 0), 1, 100 + i);
    std::atomic<int> gen{0}, done{0}; std::atomic<bool> stop{false};
    std::vector<std::thread> th;
    for (int i = 1; i < nw; ++i) th.emplace_back([&, i] {
        int seen = 0;
        while (true) {
            while (gen.load(std::memory_order_acquire) == seen) { if (stop.load(std::memory_order_relaxed)) return; }
            seen++; Ls[i].run(); done.fetch_add(1, std::memory_order_release);
        }
    });
    auto once = [&] { done.store(0); gen.fetch_add(1); Ls[0].run(); while (done.load(std::memory_order_acquire) < nw - 1) {} };
    for (int i = 0; i < iters / 10; ++i) once();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) once();
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
    stop.store(true); for (auto & t : th) t.join();
    return us;
}

int main(int argc, char ** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 3000;
    g_fixed_ids = argc > 2 && strcmp(argv[2], "fixed") == 0;
    std::mt19937 rng(7);
    weights w;
    ggml_init_params ip = { ggml_tensor_overhead() * 8, nullptr, true };
    w.ctx  = ggml_init(ip);
    w.gate = ggml_new_tensor_3d(w.ctx, GGML_TYPE_Q2_K, N_EMBD, N_FF, N_EXPERT);
    w.up   = ggml_new_tensor_3d(w.ctx, GGML_TYPE_Q2_K, N_EMBD, N_FF, N_EXPERT);
    w.down = ggml_new_tensor_3d(w.ctx, GGML_TYPE_Q3_K, N_FF, N_EMBD, N_EXPERT);
    ggml_backend_t be0 = ggml_backend_cpu_init();
    w.buf = ggml_backend_alloc_ctx_tensors(w.ctx, be0);
    fill_experts(w.gate, N_EMBD, N_FF, rng); fill_experts(w.up, N_EMBD, N_FF, rng); fill_experts(w.down, N_FF, N_EMBD, rng);
    printf("cpu1bench [%s expert ids]: %d iters per cell, weights %.0f MiB, us per layer call (speedup vs ggml 1 thread)\n", g_fixed_ids ? "FIXED" : "re-drawn", iters, ggml_backend_buffer_get_size(w.buf) / 1048576.0);
    for (int pairs : {4, 8, 12, 24}) {
        const double base = bench_ggml(w, pairs, 1, iters);
        printf("pairs %2d | ggml t1 %7.1f us (%.1f G MAC/s)", pairs, base, pairs * 3.1457 / base * 1e3);
        for (int nt : {2, 3, 4, 6}) { const double us = bench_ggml(w, pairs, nt, iters); printf(" | t%d %7.1f (%.2fx)", nt, us, base / us); }
        printf("\n         | task                        ");
        for (int nt : {2, 3, 4, 6}) { const double us = bench_task(w, pairs, nt, iters); printf(" | t%d %7.1f (%.2fx)", nt, us, base / us); }
        printf("\n"); fflush(stdout);
    }
    return 0;
}

// Phase-2 diagnostic microbench: does mul_mat_id over an MXFP4 slot buffer scale with the pool
// size S at DECODE shape (n_tokens=1, n_expert_used=4)? The full-model nsys is confounded by
// prefill/scheduler/two-path; this isolates the slot matmul the cache (and phase-3's custom op)
// depends on. If time grows with S -> the kernel path is pathological in S and phase 3 inherits it;
// if flat -> the S-dependent cost is elsewhere. gpt-oss dims. Timing only (values irrelevant).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

static const int64_t n_embd = 2880;
static const int64_t n_ff   = 2880;
static const int     U      = 4;   // n_expert_used
static const int     ITERS  = 60;

static double bench_S(ggml_backend_t backend, int S) {
    ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, /*.no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * slot = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, n_embd, n_ff, S+1); // MXFP4 like real experts
    ggml_tensor * x    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,  n_embd, U, 1);
    ggml_tensor * ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32,  U, 1);
    ggml_set_input(slot); ggml_set_input(x); ggml_set_input(ids);
    ggml_tensor * out  = ggml_mul_mat_id(ctx, slot, x, ids);
    ggml_set_output(out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend); (void) buf;

    std::vector<char>  slot_h(ggml_nbytes(slot), 0x11);
    std::vector<float> x_h(n_embd*U, 0.03f);
    std::vector<int32_t> ids_h(U); for (int i = 0; i < U; ++i) ids_h[i] = i; // route to slots 0..3
    ggml_backend_tensor_set(slot, slot_h.data(), 0, ggml_nbytes(slot));
    ggml_backend_tensor_set(x,    x_h.data(),    0, ggml_nbytes(x));
    ggml_backend_tensor_set(ids,  ids_h.data(),  0, ggml_nbytes(ids));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    for (int i = 0; i < 5; ++i) { ggml_backend_graph_compute(backend, gf); } // warmup
    ggml_backend_synchronize(backend);

    const int64_t t0 = ggml_time_us();
    for (int i = 0; i < ITERS; ++i) { ggml_backend_graph_compute(backend, gf); }
    ggml_backend_synchronize(backend);
    const int64_t t1 = ggml_time_us();

    ggml_free(ctx);
    return double(t1 - t0) / ITERS; // us/op
}

int main() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) { printf("test-slot-matmul-bench: no GPU; SKIP\n"); return 0; }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    printf("test-slot-matmul-bench: %s, mul_mat_id MXFP4 [%lld,%lld,S+1] x [.,%d,1] decode\n",
           ggml_backend_dev_name(dev), (long long) n_embd, (long long) n_ff, U);

    double base = 0.0;
    for (int S : {8, 16, 32, 48, 64, 96, 128}) {
        const double us = bench_S(backend, S);
        if (S == 8) base = us;
        printf("  S=%3d (%3d experts): %8.2f us/op   %.2fx vs S=8\n", S, S+1, us, us / base);
    }
    ggml_backend_free(backend);
    return 0;
}

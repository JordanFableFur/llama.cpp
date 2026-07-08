// M3 gate for the persistent GPU expert slot cache (SLOT-CACHE-DESIGN.md).
// Validates ggml_mul_mat_id_skip on the CPU backend:
//   (a) all-zero mask  -> byte-identical to plain ggml_mul_mat_id
//   (b) all-one  mask  -> all outputs zero
//   (c) random   mask  -> plain result with skipped (token,k) columns zeroed
//   (d) mutate the mask between evals of the SAME graph -> output tracks the new mask
//       (catches any accidental caching of mask-derived state)
// The skip contract is self-contained: mask[e] != 0 means "skip expert e" (write zeros).

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

static const int K   = 32;  // n_embd
static const int N   = 8;   // n_ff
static const int E   = 6;   // n_expert
static const int U   = 2;   // n_expert_used
static const int T   = 5;   // n_tokens

static int g_fail = 0;

static void check(const char * name, bool ok) {
    printf("  gate %s: %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok) { g_fail++; }
}

int main() {
    ggml_backend_t backend = ggml_backend_cpu_init();

    ggml_init_params ip = { /*.mem_size=*/ ggml_tensor_overhead()*32 + ggml_graph_overhead(), nullptr, /*.no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * as   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, N, E);
    ggml_tensor * b    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, U, T);
    ggml_tensor * ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, U, T);
    ggml_tensor * mask = ggml_new_tensor_1d(ctx, GGML_TYPE_I8,  E);
    ggml_set_input(as); ggml_set_input(b); ggml_set_input(ids); ggml_set_input(mask);

    ggml_tensor * plain = ggml_mul_mat_id(ctx, as, b, ids);
    ggml_tensor * skip  = ggml_mul_mat_id_skip(ctx, as, b, ids, mask);
    ggml_set_output(plain); ggml_set_output(skip);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    (void) buf;

    // deterministic inputs
    std::vector<float> h_as(K*N*E), h_b(K*U*T);
    for (int i = 0; i < K*N*E; ++i) h_as[i] = 0.01f*((i*7) % 19) - 0.1f;
    for (int i = 0; i < K*U*T; ++i) h_b[i]  = 0.02f*((i*5) % 13) - 0.1f;
    std::vector<int32_t> h_ids(U*T);
    for (int i = 0; i < U*T; ++i) h_ids[i] = (i*3 + 1) % E;   // spread over experts
    ggml_backend_tensor_set(as,  h_as.data(),  0, ggml_nbytes(as));
    ggml_backend_tensor_set(b,   h_b.data(),   0, ggml_nbytes(b));
    ggml_backend_tensor_set(ids, h_ids.data(), 0, ggml_nbytes(ids));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, plain);
    ggml_build_forward_expand(gf, skip);

    const size_t out_n = (size_t) N*U*T;
    std::vector<float> vp(out_n), vs(out_n);

    auto run = [&](std::vector<int8_t> & m) {
        ggml_backend_tensor_set(mask, m.data(), 0, ggml_nbytes(mask));
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_tensor_get(plain, vp.data(), 0, out_n*sizeof(float));
        ggml_backend_tensor_get(skip,  vs.data(), 0, out_n*sizeof(float));
    };

    // reference: plain, but zero every (token,k) column whose routed expert is masked
    auto expect_masked = [&](const std::vector<int8_t> & m) {
        std::vector<float> r = vp;
        for (int t = 0; t < T; ++t)
            for (int u = 0; u < U; ++u)
                if (m[h_ids[u + t*U]]) {
                    float * col = &r[(size_t)(u + t*U)*N];  // dst layout [N, U, T]
                    for (int n = 0; n < N; ++n) col[n] = 0.0f;
                }
        return r;
    };

    // (a) all zero -> identical to plain
    { std::vector<int8_t> m(E, 0); run(m); check("(a) zero-mask==plain", memcmp(vp.data(), vs.data(), out_n*sizeof(float)) == 0); }

    // (b) all one -> all zero
    { std::vector<int8_t> m(E, 1); run(m);
      bool allz = true; for (float x : vs) if (x != 0.0f) { allz = false; break; }
      check("(b) one-mask==zero", allz); }

    // (c) random mask -> plain with skipped columns zeroed
    { std::vector<int8_t> m(E); for (int e = 0; e < E; ++e) m[e] = (int8_t)((e*5+2) % 3 == 0);
      run(m); auto ref = expect_masked(m);
      check("(c) random-mask==ref", memcmp(ref.data(), vs.data(), out_n*sizeof(float)) == 0); }

    // (d) mutate mask across evals on the SAME graph -> output tracks the new mask
    { std::vector<int8_t> m1(E); for (int e = 0; e < E; ++e) m1[e] = (int8_t)(e % 2);
      run(m1); auto r1 = expect_masked(m1);
      bool t1 = memcmp(r1.data(), vs.data(), out_n*sizeof(float)) == 0;
      std::vector<int8_t> m2(E); for (int e = 0; e < E; ++e) m2[e] = (int8_t)((e+1) % 2); // inverted
      run(m2); auto r2 = expect_masked(m2);
      bool t2 = memcmp(r2.data(), vs.data(), out_n*sizeof(float)) == 0;
      check("(d) mask-mutation tracks", t1 && t2); }

    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("test-mul-mat-id-skip: %s\n", g_fail == 0 ? "ALL GATES PASS" : "FAILURES PRESENT");
    return g_fail == 0 ? 0 : 1;
}

// M5b unit oracle for the persistent GPU expert slot cache (SLOT-CACHE-DESIGN.md).
// Verifies the HIT path on the SAME device (CUDA): promote a known expert set into a slot
// buffer, build the expert->slot map, and compare
//     mul_mat_id(slots, x, get_rows(e2s, ids))   vs   mul_mat_id(full, x, ids)
// on CUDA. Hit rows (routed expert resident) must be byte-identical (same kernel, same dtype,
// per-expert matmul independent); miss rows (routed to the zeroed dummy slot) must be exactly 0.
// This catches slot-buffer stride errors, remap off-by-ones, and wrong-offset promotions that
// the forced-empty whole-graph gate (M5) cannot see because it never reads promoted slots.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

static const int K = 32; // n_embd
static const int N = 8;   // n_ff
static const int E = 6;   // n_expert
static const int S = 3;   // slots (dummy slot index = S)
static const int U = 2;   // n_expert_used
static const int T = 5;   // n_tokens

int main() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        printf("test-slot-matmul: no GPU device found; SKIP\n");
        return 0;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    printf("test-slot-matmul: device = %s\n", ggml_backend_dev_name(dev));

    ggml_init_params ip = { ggml_tensor_overhead()*32 + ggml_graph_overhead(), nullptr, /*.no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * full = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, N, E);
    ggml_tensor * slots= ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, N, S+1);
    ggml_tensor * x    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, U, T);
    ggml_tensor * ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, U, T);
    ggml_tensor * e2s  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, E);
    ggml_set_input(full); ggml_set_input(slots); ggml_set_input(x); ggml_set_input(ids); ggml_set_input(e2s);

    // get_rows treats a 2D index tensor's ne[1] as a batch dim (asserts a->ne[2]==b->ne[1]),
    // so flatten ids to 1D, gather slot ids from e2s, then reshape back to [U,T].
    ggml_tensor * ids_flat = ggml_reshape_1d(ctx, ids, U*T);
    ggml_tensor * remap = ggml_reshape_2d(ctx, ggml_get_rows(ctx, e2s, ids_flat), U, T); // slot per (u,t)
    ggml_tensor * ref   = ggml_mul_mat_id(ctx, full,  x, ids);
    ggml_tensor * test  = ggml_mul_mat_id(ctx, slots, x, remap);
    ggml_set_output(ref); ggml_set_output(test);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend); (void) buf;

    // deterministic inputs
    std::vector<float> h_full(K*N*E), h_x(K*U*T);
    for (int i = 0; i < K*N*E; ++i) h_full[i] = 0.013f*((i*7) % 23) - 0.15f;
    for (int i = 0; i < K*U*T; ++i) h_x[i]    = 0.017f*((i*5) % 17) - 0.12f;

    // resident set: experts 1,3,4 -> slots 0,1,2 ; everything else -> dummy slot S
    std::vector<int32_t> h_e2s(E, S);
    const int resident_expert[3] = {1, 3, 4};
    h_e2s[1] = 0; h_e2s[3] = 1; h_e2s[4] = 2;

    // build the slot buffer by promoting each resident expert's contiguous slice (mirrors promote())
    std::vector<float> h_slots((size_t)K*N*(S+1), 0.0f); // slot S stays zero (dummy)
    for (int s = 0; s < 3; ++s) {
        const int e = resident_expert[s];
        memcpy(&h_slots[(size_t)s*K*N], &h_full[(size_t)e*K*N], (size_t)K*N*sizeof(float));
    }

    // ids: mix of resident and miss experts
    std::vector<int32_t> h_ids(U*T);
    for (int i = 0; i < U*T; ++i) h_ids[i] = i % E; // 0,1,2,3,4,5,0,1,2,3 -> hits {1,3,4}, misses {0,2,5}

    ggml_backend_tensor_set(full,  h_full.data(),  0, ggml_nbytes(full));
    ggml_backend_tensor_set(slots, h_slots.data(), 0, ggml_nbytes(slots));
    ggml_backend_tensor_set(x,     h_x.data(),     0, ggml_nbytes(x));
    ggml_backend_tensor_set(ids,   h_ids.data(),   0, ggml_nbytes(ids));
    ggml_backend_tensor_set(e2s,   h_e2s.data(),   0, ggml_nbytes(e2s));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ref);
    ggml_build_forward_expand(gf, test);
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> vr((size_t)N*U*T), vt((size_t)N*U*T);
    ggml_backend_tensor_get(ref,  vr.data(), 0, vr.size()*sizeof(float));
    ggml_backend_tensor_get(test, vt.data(), 0, vt.size()*sizeof(float));

    int hit_rows = 0, miss_rows = 0, hit_bad = 0, miss_bad = 0;
    float hit_maxdiff = 0.0f;
    for (int t = 0; t < T; ++t) {
        for (int u = 0; u < U; ++u) {
            const int e = h_ids[u + t*U];
            const bool resident = h_e2s[e] != S;
            const size_t off = (size_t)(u + t*U)*N; // dst layout [N,U,T]: row (u,t) at (u + t*U)*N
            if (resident) {
                hit_rows++;
                for (int n = 0; n < N; ++n) {
                    float d = vt[off+n] - vr[off+n];
                    if (d < 0) d = -d;
                    if (d > hit_maxdiff) hit_maxdiff = d;
                }
                if (memcmp(&vt[off], &vr[off], N*sizeof(float)) != 0) hit_bad++;
            } else {
                miss_rows++;
                for (int n = 0; n < N; ++n) if (vt[off+n] != 0.0f) { miss_bad++; break; }
            }
        }
    }

    printf("  hit rows  %d: byte-mismatches %d (max abs diff %.3e)\n", hit_rows, hit_bad, hit_maxdiff);
    printf("  miss rows %d: nonzero %d\n", miss_rows, miss_bad);
    const bool ok = (hit_bad == 0) && (miss_bad == 0) && (hit_rows > 0) && (miss_rows > 0);
    printf("test-slot-matmul: %s\n", ok ? "PASS (hits byte-identical, misses zero)" : "FAIL");

    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}

// Phase-2 staging-ring unit oracle for the persistent GPU expert slot cache (SLOT-CACHE-DESIGN.md,
// phase-2 addendum point 3). Verifies the promotion path that phase 2 will use:
//   host memcpy expert rows -> pinned ring slot (cudaHostAlloc via host buffer type, 2x double-buffered)
//   -> tensor_set_async on a DEDICATED copy backend/stream -> GPU slot, with per-ring-slot events so a
//   ring buffer is not overwritten until its prior async copy finished.
// Gate: every promoted GPU slot is byte-identical to its source expert, with both ring slots cycling.
// This isolates the ring mechanism before it is wired into live promotion (pause point i).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

static const int64_t K = 1024;         // rows
static const int64_t N = 1024;         // cols  -> 4 MiB/expert (F32), fits the 16 MiB ring slot
static const int     E = 6;            // source experts
static const int     S = 6;            // GPU slots
static const size_t  RING_BYTES = 16 * 1024 * 1024;

int main() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) { printf("test-slot-ring: no GPU device; SKIP\n"); return 0; }
    printf("test-slot-ring: device = %s\n", ggml_backend_dev_name(dev));

    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
    if (!host_buft) { printf("test-slot-ring: no host (pinned) buffer type; SKIP\n"); return 0; }

    ggml_backend_t copy_backend = ggml_backend_dev_init(dev, nullptr); // its own stream = dedicated copy stream

    // pinned double-buffered ring
    ggml_backend_buffer_t ring_buf[2];
    void *                ring_ptr[2];
    ggml_backend_event_t  ring_evt[2];
    bool                  ring_used[2] = { false, false };
    for (int r = 0; r < 2; ++r) {
        ring_buf[r] = ggml_backend_buft_alloc_buffer(host_buft, RING_BYTES);
        ring_ptr[r] = ggml_backend_buffer_get_base(ring_buf[r]);
        ring_evt[r] = ggml_backend_event_new(dev);
    }

    // GPU slot tensor
    ggml_init_params ip = { ggml_tensor_overhead()*4, nullptr, /*.no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * slots = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, N, S);
    ggml_backend_buffer_t slot_buf = ggml_backend_alloc_ctx_tensors(ctx, copy_backend);
    (void) slot_buf;

    const size_t expert_bytes = (size_t) K*N*sizeof(float);

    // deterministic source experts (host)
    std::vector<std::vector<float>> src(E, std::vector<float>(K*N));
    for (int e = 0; e < E; ++e)
        for (int i = 0; i < K*N; ++i)
            src[e][i] = 0.011f*((i*3 + e*7) % 29) - 0.13f;

    // promotion order: expert e -> slot e, cycling the two ring buffers
    for (int e = 0; e < E; ++e) {
        const int r = e % 2;
        if (ring_used[r]) {
            ggml_backend_event_synchronize(ring_evt[r]); // don't overwrite a ring slot mid-flight
        }
        memcpy(ring_ptr[r], src[e].data(), expert_bytes);                       // weight -> pinned ring
        ggml_backend_tensor_set_async(copy_backend, slots, ring_ptr[r],
                                      (size_t) e * expert_bytes, expert_bytes); // pinned ring -> GPU slot (async)
        ggml_backend_event_record(ring_evt[r], copy_backend);
        ring_used[r] = true;
    }
    ggml_backend_synchronize(copy_backend);

    // verify every slot byte-identical to its source
    int bad = 0;
    std::vector<float> back(K*N);
    for (int e = 0; e < E; ++e) {
        ggml_backend_tensor_get(slots, back.data(), (size_t) e * expert_bytes, expert_bytes);
        if (memcmp(back.data(), src[e].data(), expert_bytes) != 0) bad++;
    }
    printf("  promoted %d experts through 2x%zu MiB ring: %d byte-mismatches\n", E, RING_BYTES/(1024*1024), bad);
    const bool ok = (bad == 0);
    printf("test-slot-ring: %s\n", ok ? "PASS (all slots byte-identical, both ring buffers cycled)" : "FAIL");

    for (int r = 0; r < 2; ++r) { ggml_backend_event_free(ring_evt[r]); ggml_backend_buffer_free(ring_buf[r]); }
    ggml_free(ctx);
    ggml_backend_free(copy_backend);
    return ok ? 0 : 1;
}

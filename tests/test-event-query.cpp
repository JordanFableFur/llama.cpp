// Standalone test for ggml_backend_event_query (added for the slot-cache phase-2 async promotion).
// Contract: non-blocking; true = completed; an in-flight event returns false; a backend without a
// native query blocks (synchronize) then returns true - it never reports completion early.
// Verifies: (1) CUDA native path - in-flight event -> false, completed event -> true.
//           (2) reports the fallback situation on this backend set (CPU has no events, so no
//               event-capable backend here lacks a native query -> fallback not reachable; it is
//               correct by construction: synchronize then true).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    int fail = 0;

    ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu) { printf("test-event-query: no GPU device; SKIP\n"); return 0; }
    printf("test-event-query: device = %s\n", ggml_backend_dev_name(gpu));

    ggml_backend_t backend = ggml_backend_dev_init(gpu, nullptr);
    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(gpu);

    // pinned source + a GPU sink; enqueue enough async work that the event is still in-flight
    // when we query it right after recording.
    const size_t CHUNK = 16 * 1024 * 1024;
    const int    NCOPY = 64; // ~1 GiB of H2D -> tens of ms, dwarfs the host code before the query
    ggml_backend_buffer_t hbuf = ggml_backend_buft_alloc_buffer(host_buft, CHUNK);
    void * hptr = ggml_backend_buffer_get_base(hbuf);
    memset(hptr, 0x5a, CHUNK);

    ggml_init_params ip = { ggml_tensor_overhead()*2, nullptr, /*.no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * sink = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, CHUNK);
    ggml_backend_buffer_t sbuf = ggml_backend_alloc_ctx_tensors(ctx, backend); (void) sbuf;

    ggml_backend_event_t ev = ggml_backend_event_new(gpu);
    if (!ev) { printf("test-event-query: GPU has no events?! FAIL\n"); return 1; }

    for (int i = 0; i < NCOPY; ++i) {
        ggml_backend_tensor_set_async(backend, sink, hptr, 0, CHUNK);
    }
    ggml_backend_event_record(ev, backend);

    const bool q_inflight = ggml_backend_event_query(ev); // expect false (work still queued)
    printf("  in-flight query: %s (expect false)\n", q_inflight ? "true" : "false");
    if (q_inflight) { printf("  NOTE: completed before query - could not observe in-flight state\n"); fail++; }

    ggml_backend_event_synchronize(ev);
    const bool q_done = ggml_backend_event_query(ev); // expect true
    printf("  post-sync query: %s (expect true)\n", q_done ? "true" : "false");
    if (!q_done) fail++;

    // fallback situation on this build
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_event_t cpu_ev = cpu ? ggml_backend_event_new(cpu) : nullptr;
    if (!cpu_ev) {
        printf("  fallback: CPU backend has no events on this build -> synchronize-fallback not "
               "reachable here (correct by construction: synchronize then true)\n");
    } else {
        const bool q = ggml_backend_event_query(cpu_ev); // would exercise the fallback
        printf("  fallback query on CPU event: %s (expect true)\n", q ? "true" : "false");
        if (!q) fail++;
        ggml_backend_event_free(cpu_ev);
    }

    ggml_backend_event_free(ev);
    ggml_backend_buffer_free(hbuf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("test-event-query: %s\n", fail == 0 ? "PASS" : "FAIL");
    return fail == 0 ? 0 : 1;
}

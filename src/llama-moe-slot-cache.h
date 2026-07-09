#pragma once

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <vector>

struct llama_model;

// Persistent per-context GPU expert slot cache (EXPERIMENTS.md item 15, SLOT-CACHE-DESIGN.md).
// Owned by llama_context like the KV cache: slot residency is per-session routing state, so two
// contexts over the same model must not share it. Env-gated by GGML_MOE_SLOT_CACHE=<slots/layer>;
// when unset, llama_context holds no cache and the MoE graph is built unchanged (clean off-switch).
//
// Milestones (SLOT-CACHE-DESIGN.md phase-1 log): M1 plumbing (done); M2 slot tensors + promotion
// (this); M3 CPU mask-skip op; M4 option-(b) oracle; M5 option-(c) get_rows wiring; M6 gates+bench.
struct llama_moe_slot_cache {
    int  n_slots       = 0;    // slots per CPU-resident layer (dummy miss-sink slot is n_slots)
    int  n_expert      = 0;
    int  n_expert_used = 0;
    bool initialized   = false;

    // online hit accounting (compute-relevant: residency at access time); reported at teardown
    uint64_t hits = 0;
    uint64_t reqs = 0;
    uint64_t promotions = 0;

    // token-boundary host-cost telemetry (poll + ring memcpy + publish), reported at teardown
    uint64_t update_ns    = 0;
    uint64_t update_calls = 0;

    // phase-2 async promotion: dedicated copy stream + pinned staging ring (SLOT-CACHE-DESIGN.md
    // phase-2 addendum). Promotions run off the decode stream; the compute map (e2s/skip) is only
    // updated once event_query confirms the copy landed. Off => promote() synchronous (phase 1).
    bool         async_promote  = false;
    bool         compute_disabled = false; // GGML_MOE_SLOT_NOSLOT: keep bookkeeping, use plain CPU
                                           // compute path (no GPU slot / combine) - isolates the
                                           // two-path compute cost from cache bookkeeping cost
    bool         book_disabled = false;    // GGML_MOE_SLOT_NOBOOK: also skip cpy-sink + update
                                           // (pure allocation) - isolates allocation vs bookkeeping
    bool         promo_disabled = false;   // GGML_MOE_SLOT_NOPROMO: cpy-sink ON, update OFF -
                                           // isolates the graph cpy-sink cost from promotion
    bool         cap_disabled = false;     // GGML_MOE_SLOT_NOCAP: two-path built, but fused capture
                                           // AND update OFF - isolates two-path structural split cost
                                           // from the capture (empty cache, no promotion) [P3.0 decomp]
    int          capture_mode = 3;         // routing capture: 3 = fused into mul_mat_id_skip CPU forward
                                           // (P3.0 default: op writes routed ids to ls.routed as a side
                                           // effect - no graph node/callback/sync);
                                           // 0 = GPU-sink (GGML_MOE_SLOT_GPUSINK, phase-2, diagnostic);
                                           // 1 = eval callback (GGML_MOE_SLOT_CALLBACK, diagnostic);
                                           // 2 = CPU cpy-sink (GGML_MOE_SLOT_CPYSINK, diagnostic)
    int          n_ubatch = 0;
    // GPU-sink: one persistent [n_expert_used, n_ubatch, n_layers] i32 GPU tensor; each cached layer
    // cpys its routed ids to its z-slice (GPU->GPU, no graph split); one D2H per token into routed_host.
    ggml_context_ptr        routed_ctx;
    ggml_backend_buffer_ptr routed_buf;
    ggml_tensor *           routed_gpu = nullptr;
    std::vector<int32_t>    routed_host;
    int          ring_size      = 32;    // pinned staging buffers = max promotions in flight (knob)
    int          promote_budget = 32;    // max promotions ENQUEUED per token, GLOBAL across layers (knob)
    ggml_backend_t copy_backend = nullptr;

    struct ring_slot {
        ggml_backend_buffer_t buf = nullptr;
        void *                ptr = nullptr;
        ggml_backend_event_t  evt = nullptr;
        bool busy = false;
        // deferred map update applied when evt completes:
        int layer_idx = -1, expert = -1, slot = -1;
    };
    std::vector<ring_slot> ring;

    // one entry per CPU-resident MoE layer; GPU tensors live in the owned bufs below
    struct layer_slots {
        int il = -1;

        ggml_tensor * gate = nullptr;   // [n_embd, n_ff,  n_slots+1] on GPU (slot n_slots = zeroed dummy)
        ggml_tensor * up   = nullptr;   // [n_embd, n_ff,  n_slots+1]
        ggml_tensor * down = nullptr;   // [n_ff,   n_embd, n_slots+1]
        ggml_tensor * e2s  = nullptr;   // [1, n_expert] i32 (GPU): expert -> slot, value n_slots = miss
        ggml_tensor * skip = nullptr;   // [n_expert] I8 (CPU): 1 = resident (mul_mat_id_skip skips it)
        ggml_tensor * routed = nullptr; // [n_expert_used, n_ubatch] i32 (CPU): this eval's routed ids (cpy sink)

        // source expert weights (CPU-resident, host-readable), promotion copies from these
        ggml_tensor * src_gate = nullptr;
        ggml_tensor * src_up   = nullptr;
        ggml_tensor * src_down = nullptr;

        std::vector<int32_t> e2s_host;   // GPU-visible compute map: expert -> slot; only updated when
                                         // a promotion COMPLETES (n_slots = not resident / in-flight)
        std::vector<int8_t>  skip_host;  // host mirror of skip: 1 = resident (completed)
        std::vector<int32_t> mru;        // experts OCCUPYING a slot (in-flight or completed), front = MRU
        std::vector<int32_t> slot_of;    // expert -> assigned slot (bookkeeping), -1 if none. Differs from
                                         // e2s_host only for in-flight promotions (assigned but not landed)
    };

    std::vector<layer_slots>             layers;
    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    explicit llama_moe_slot_cache(int n_slots) : n_slots(n_slots) {}
    ~llama_moe_slot_cache();

    static std::unique_ptr<llama_moe_slot_cache> maybe_create_from_env();

    // lazily allocate slot buffers for every CPU-resident MoE layer of the model (idempotent)
    void init(const llama_model & model, int n_expert_used, int n_ubatch);

    // eval-callback routing capture (replaces the graph cpy-sink): CAPTURE ONLY - copies the routed
    // ids of each cached layer's gate mul_mat_id into ls.routed; mutates no LRU/map/mask state (all
    // deferred to update_after_decode at the token boundary). Returns, on ask, whether to observe t.
    bool eval_capture(struct ggml_tensor * t, bool ask);

    // after a decode, promote each layer's routed misses into slots (LRU) and upload the maps
    void update_after_decode(int n_tokens);

    // cache entry for layer il, or nullptr if that layer is not cached
    const layer_slots * find(int il) const {
        for (const auto & ls : layers) {
            if (ls.il == il) {
                return &ls;
            }
        }
        return nullptr;
    }

    // index of layer il within `layers` (its z-slice in routed_gpu), or -1 if not cached
    int find_pos(int il) const {
        for (int i = 0; i < (int) layers.size(); ++i) {
            if (layers[i].il == il) { return i; }
        }
        return -1;
    }

    // GPU-sink: one batched D2H of routed_gpu into routed_host (called at the token boundary)
    void download_routing();

    // routing ids for layer at position `pos` this eval (source depends on capture_mode)
    const int32_t * layer_routing(int pos) const {
        if (capture_mode == 0) {
            return routed_host.data() + (size_t) pos * n_expert_used * n_ubatch;
        }
        return (const int32_t *) layers[pos].routed->data;
    }

    // synchronous promotion: copy expert_id's weights into slot on the GPU (phase 1)
    void promote(layer_slots & ls, int expert_id, int slot);

    // phase-2 async promotion helpers
    void promote_async(int layer_idx, int expert_id, int slot); // stage via ring, async copy, record event
    void poll_completions();                                    // apply map updates for finished promotions
    void upload_maps(layer_slots & ls);                         // sync-upload e2s + skip for one layer
};

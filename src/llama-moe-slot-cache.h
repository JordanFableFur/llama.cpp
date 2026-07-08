#pragma once

#include "ggml.h"
#include "ggml-cpp.h"

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
    int  n_slots     = 0;      // slots per CPU-resident layer (dummy miss-sink slot is n_slots)
    bool initialized = false;

    // one entry per CPU-resident MoE layer; GPU tensors live in the owned bufs below
    struct layer_slots {
        int il = -1;

        ggml_tensor * gate = nullptr;   // [n_embd, n_ff,  n_slots+1] on GPU (slot n_slots = zeroed dummy)
        ggml_tensor * up   = nullptr;   // [n_embd, n_ff,  n_slots+1]
        ggml_tensor * down = nullptr;   // [n_ff,   n_embd, n_slots+1]
        ggml_tensor * e2s  = nullptr;   // [1, n_expert] i32 (GPU): expert -> slot, value n_slots = miss
        ggml_tensor * skip = nullptr;   // [n_expert] I8 (CPU): 1 = resident (mul_mat_id_skip skips it)

        // source expert weights (CPU-resident, host-readable), promotion copies from these
        ggml_tensor * src_gate = nullptr;
        ggml_tensor * src_up   = nullptr;
        ggml_tensor * src_down = nullptr;

        std::vector<int32_t> e2s_host;  // host mirror of e2s (LRU bookkeeping lands here in M6)
    };

    std::vector<layer_slots>             layers;
    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    explicit llama_moe_slot_cache(int n_slots) : n_slots(n_slots) {}

    static std::unique_ptr<llama_moe_slot_cache> maybe_create_from_env();

    // lazily allocate slot buffers for every CPU-resident MoE layer of the model (idempotent)
    void init(const llama_model & model);

    // cache entry for layer il, or nullptr if that layer is not cached
    const layer_slots * find(int il) const {
        for (const auto & ls : layers) {
            if (ls.il == il) {
                return &ls;
            }
        }
        return nullptr;
    }

    // synchronous promotion: copy expert_id's weights into slot on the GPU (phase 1)
    void promote(layer_slots & ls, int expert_id, int slot);
};

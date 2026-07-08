#pragma once

#include <memory>

// Persistent per-context GPU expert slot cache (EXPERIMENTS.md item 15, SLOT-CACHE-DESIGN.md).
// Owned by llama_context like the KV cache: slot residency is per-session routing state, so two
// contexts over the same model must not share it. Env-gated by GGML_MOE_SLOT_CACHE=<slots/layer>;
// when unset, llama_context holds no cache and the MoE graph is built unchanged (clean off-switch).
//
// Milestones (see SLOT-CACHE-DESIGN.md phase-1 log): M1 = this skeleton + plumbing (no-op);
// M2 = per-layer slot tensors + promotion; M3 = CPU mask-skip; M4 = option-(b) oracle;
// M5 = option-(c) get_rows indirection wiring; M6 = gates + bench.
struct llama_moe_slot_cache {
    int n_slots = 0;   // slots per CPU-resident layer

    explicit llama_moe_slot_cache(int n_slots) : n_slots(n_slots) {}

    // returns a configured cache if GGML_MOE_SLOT_CACHE is a positive int, else nullptr
    static std::unique_ptr<llama_moe_slot_cache> maybe_create_from_env();
};

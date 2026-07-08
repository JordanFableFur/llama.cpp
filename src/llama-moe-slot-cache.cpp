#include "llama-moe-slot-cache.h"

#include "llama-impl.h"

#include <cstdlib>

std::unique_ptr<llama_moe_slot_cache> llama_moe_slot_cache::maybe_create_from_env() {
    const char * env = getenv("GGML_MOE_SLOT_CACHE");
    if (!env || !*env) {
        return nullptr;
    }
    const int s = atoi(env);
    if (s <= 0) {
        return nullptr;
    }
    LLAMA_LOG_INFO("%s: MoE slot cache enabled, %d slots/layer\n", __func__, s);
    return std::make_unique<llama_moe_slot_cache>(s);
}

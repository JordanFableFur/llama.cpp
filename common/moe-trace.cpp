#include "moe-trace.h"

#include "log.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// Two env-gated diagnostics sharing one eval-callback and one ids read:
//   GGML_MOE_TRACE=<file>       - dump per-token per-layer routed expert ids (binary; see below)
//   GGML_MOE_CACHE_SIM=S1,S2... - in-engine per-layer LRU cache hit/miss counter (item 15 phase 0),
//                                 reported at process exit. Validates simulate-cache.py online.
//
// Trace binary format (little-endian, host byte order):
//   header once: int32 magic 'MOET' (0x54454f4d), int32 n_expert, int32 n_expert_used
//   frames:      int32 layer, int32 n_tokens, int32[n_tokens * n_expert_used] expert ids
// Frames arrive one per MoE layer per graph eval; tokens within a frame are in sequence order.
// The gate mul_mat_id carries the same routed ids as up/down, so only the gate op is recorded.

namespace {

struct lru_cache {
    int                  slots = 0;
    std::vector<int32_t> mru;      // most-recently-used at front; size <= slots
    uint64_t             hits = 0;
    uint64_t             miss = 0;

    void access(int32_t e) {
        for (size_t i = 0; i < mru.size(); ++i) {
            if (mru[i] == e) {
                hits++;
                if (i != 0) { mru.erase(mru.begin() + i); mru.insert(mru.begin(), e); }
                return;
            }
        }
        miss++;
        if ((int) mru.size() >= slots) { mru.pop_back(); }
        mru.insert(mru.begin(), e);
    }
};

struct moe_trace_state {
    std::mutex           mutex;

    // trace dump
    FILE *               fp          = nullptr;
    bool                 header_done = false;
    std::vector<int32_t> frame;

    // in-engine cache sim
    bool                 sim_on = false;
    std::vector<int>     sim_slots;                  // S values
    std::map<int, std::vector<lru_cache>> sim;       // layer -> one lru_cache per S

    std::vector<uint8_t> ids_host;
};

moe_trace_state g;

void moe_trace_report() {
    if (g.fp) {
        fflush(g.fp);
        fclose(g.fp);
        g.fp = nullptr;
    }
    if (!g.sim_on || g.sim.empty()) {
        return;
    }
    // atexit runs after the llama/ggml log backend is torn down, so LOG_* is dropped here;
    // write straight to stderr (captured by the run's 2>&1 redirect).
    fprintf(stderr, "\n=== MoE cache sim (in-engine per-layer LRU, decode) ===\n");
    fprintf(stderr, "%-8s %10s %12s   (%zu layers)\n", "slots", "hit%", "requests", g.sim.size());
    for (size_t si = 0; si < g.sim_slots.size(); ++si) {
        uint64_t hits = 0, tot = 0;
        for (const auto & kv : g.sim) {
            const lru_cache & c = kv.second[si];
            hits += c.hits;
            tot  += c.hits + c.miss;
        }
        const double hr = tot ? 100.0 * (double) hits / (double) tot : 0.0;
        fprintf(stderr, "%-8d %9.1f%% %12llu\n", g.sim_slots[si], hr, (unsigned long long) tot);
    }
    fflush(stderr);
}

bool moe_trace_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);

    const struct ggml_tensor * src0 = t->src[0];

    if (ask) {
        return t->op == GGML_OP_MUL_MAT_ID && src0 && strstr(src0->name, "ffn_gate_exps") != nullptr;
    }

    const struct ggml_tensor * ids = t->src[2];
    if (!ids) {
        return true;
    }

    int layer = -1;
    if (sscanf(src0->name, "blk.%d.", &layer) != 1) {
        return true;
    }

    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = ids->ne[1];
    const int64_t n_expert      = src0->ne[2];

    std::lock_guard<std::mutex> lock(g.mutex);

    g.ids_host.resize(ggml_nbytes(ids));
    ggml_backend_tensor_get(ids, g.ids_host.data(), 0, ggml_nbytes(ids));
    const uint8_t * base = g.ids_host.data();

    if (g.fp) {
        if (!g.header_done) {
            const int32_t header[3] = { 0x54454f4d, (int32_t) n_expert, (int32_t) n_expert_used };
            fwrite(header, sizeof(int32_t), 3, g.fp);
            g.header_done = true;
        }
        g.frame.clear();
        g.frame.reserve(2 + n_tokens * n_expert_used);
        g.frame.push_back(layer);
        g.frame.push_back((int32_t) n_tokens);
        for (int64_t r = 0; r < n_tokens; ++r) {
            for (int64_t k = 0; k < n_expert_used; ++k) {
                const int32_t e = *(const int32_t *) (base + r * ids->nb[1] + k * ids->nb[0]);
                g.frame.push_back(e);
            }
        }
        fwrite(g.frame.data(), sizeof(int32_t), g.frame.size(), g.fp);
    }

    if (g.sim_on) {
        auto it = g.sim.find(layer);
        if (it == g.sim.end()) {
            std::vector<lru_cache> caches(g.sim_slots.size());
            for (size_t si = 0; si < g.sim_slots.size(); ++si) { caches[si].slots = g.sim_slots[si]; }
            it = g.sim.emplace(layer, std::move(caches)).first;
        }
        for (int64_t r = 0; r < n_tokens; ++r) {
            for (int64_t k = 0; k < n_expert_used; ++k) {
                const int32_t e = *(const int32_t *) (base + r * ids->nb[1] + k * ids->nb[0]);
                for (auto & c : it->second) { c.access(e); }
            }
        }
    }

    return true;
}

} // namespace

void common_moe_trace_maybe_install(common_params & params) {
    const char * trace_path = getenv("GGML_MOE_TRACE");
    const char * sim_env    = getenv("GGML_MOE_CACHE_SIM");
    if ((!trace_path || !*trace_path) && (!sim_env || !*sim_env)) {
        return;
    }
    if (params.cb_eval != nullptr) {
        LOG_WRN("%s: GGML_MOE_TRACE/CACHE_SIM set but an eval callback is already installed; skipping\n", __func__);
        return;
    }

    if (trace_path && *trace_path) {
        g.fp = fopen(trace_path, "wb");
        if (!g.fp) {
            LOG_ERR("%s: failed to open MoE trace file '%s'\n", __func__, trace_path);
        } else {
            LOG_INF("%s: MoE routing trace enabled -> %s\n", __func__, trace_path);
        }
    }

    if (sim_env && *sim_env) {
        std::string s(sim_env);
        size_t pos = 0;
        while (pos < s.size()) {
            size_t comma = s.find(',', pos);
            std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            int v = atoi(tok.c_str());
            if (v > 0) { g.sim_slots.push_back(v); }
            if (comma == std::string::npos) { break; }
            pos = comma + 1;
        }
        if (!g.sim_slots.empty()) {
            g.sim_on = true;
            std::string joined;
            for (size_t i = 0; i < g.sim_slots.size(); ++i) {
                joined += (i ? "," : "") + std::to_string(g.sim_slots[i]);
            }
            LOG_INF("%s: MoE cache sim enabled, slots/layer = %s\n", __func__, joined.c_str());
        }
    }

    if (g.fp || g.sim_on) {
        atexit(moe_trace_report);
        params.cb_eval           = moe_trace_cb;
        params.cb_eval_user_data = nullptr;
    }
}

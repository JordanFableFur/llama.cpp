#include "moe-trace.h"

#include "log.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

// Binary trace format (little-endian, host byte order):
//   header once: int32 magic 'MOET' (0x54454f4d), int32 n_expert, int32 n_expert_used
//   then frames: int32 layer, int32 n_tokens, int32[n_tokens * n_expert_used] expert ids
// Frames arrive one per MoE layer per graph eval; within a frame tokens are in
// positional (sequence) order. The gate mul_mat_id carries the same routed ids
// as up/down, so only the gate op is recorded (one frame per layer).

namespace {

struct moe_trace_state {
    std::mutex           mutex;
    FILE *               fp            = nullptr;
    bool                 header_done   = false;
    std::vector<uint8_t> ids_host;
    std::vector<int32_t> frame;
};

moe_trace_state g_moe_trace;

void moe_trace_close() {
    if (g_moe_trace.fp) {
        fflush(g_moe_trace.fp);
        fclose(g_moe_trace.fp);
        g_moe_trace.fp = nullptr;
    }
}

// off-hot-path: small ids copy + buffered fwrite in the eval callback.
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

    std::lock_guard<std::mutex> lock(g_moe_trace.mutex);
    if (!g_moe_trace.fp) {
        return true;
    }

    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = ids->ne[1];
    const int64_t n_expert      = src0->ne[2];

    g_moe_trace.ids_host.resize(ggml_nbytes(ids));
    ggml_backend_tensor_get(ids, g_moe_trace.ids_host.data(), 0, ggml_nbytes(ids));
    const uint8_t * base = g_moe_trace.ids_host.data();

    if (!g_moe_trace.header_done) {
        const int32_t header[3] = { 0x54454f4d, (int32_t) n_expert, (int32_t) n_expert_used };
        fwrite(header, sizeof(int32_t), 3, g_moe_trace.fp);
        g_moe_trace.header_done = true;
    }

    g_moe_trace.frame.clear();
    g_moe_trace.frame.reserve(2 + n_tokens * n_expert_used);
    g_moe_trace.frame.push_back(layer);
    g_moe_trace.frame.push_back((int32_t) n_tokens);
    for (int64_t r = 0; r < n_tokens; ++r) {
        for (int64_t k = 0; k < n_expert_used; ++k) {
            const int32_t e = *(const int32_t *) (base + r * ids->nb[1] + k * ids->nb[0]);
            g_moe_trace.frame.push_back(e);
        }
    }
    fwrite(g_moe_trace.frame.data(), sizeof(int32_t), g_moe_trace.frame.size(), g_moe_trace.fp);

    return true;
}

} // namespace

void common_moe_trace_maybe_install(common_params & params) {
    const char * path = getenv("GGML_MOE_TRACE");
    if (!path || !*path) {
        return;
    }
    if (params.cb_eval != nullptr) {
        LOG_WRN("%s: GGML_MOE_TRACE set but an eval callback is already installed; skipping trace\n", __func__);
        return;
    }
    g_moe_trace.fp = fopen(path, "wb");
    if (!g_moe_trace.fp) {
        LOG_ERR("%s: failed to open MoE trace file '%s'\n", __func__, path);
        return;
    }
    atexit(moe_trace_close);
    params.cb_eval           = moe_trace_cb;
    params.cb_eval_user_data = nullptr;
    LOG_INF("%s: MoE routing trace enabled -> %s\n", __func__, path);
}

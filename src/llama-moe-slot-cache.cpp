#include "llama-moe-slot-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cstdlib>
#include <cstring>
#include <vector>

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

void llama_moe_slot_cache::promote(layer_slots & ls, int expert_id, int slot) {
    // one expert occupies a contiguous [.., .., expert] slice; copy it into the slot slice.
    const size_t st_g = ls.src_gate->nb[2];
    const size_t st_u = ls.src_up->nb[2];
    const size_t st_d = ls.src_down->nb[2];
    ggml_backend_tensor_set(ls.gate, (const char *) ls.src_gate->data + (size_t) expert_id * st_g, (size_t) slot * st_g, st_g);
    ggml_backend_tensor_set(ls.up,   (const char *) ls.src_up->data   + (size_t) expert_id * st_u, (size_t) slot * st_u, st_u);
    ggml_backend_tensor_set(ls.down, (const char *) ls.src_down->data + (size_t) expert_id * st_d, (size_t) slot * st_d, st_d);
}

void llama_moe_slot_cache::init(const llama_model & model) {
    if (initialized) {
        return;
    }
    initialized = true;

    const int n_expert = (int) model.hparams.n_expert;
    if (n_expert <= 0) {
        LLAMA_LOG_INFO("%s: model is not MoE; slot cache inactive\n", __func__);
        return;
    }

    size_t total_bytes = 0;
    const int S1 = n_slots + 1;

    for (int il = 0; il < (int) model.layers.size(); ++il) {
        const auto & L = model.layers[il];
        if (!L.ffn_gate_exps || !L.ffn_up_exps || !L.ffn_down_exps) {
            continue;
        }
        // only cache experts that are CPU-resident (host buffer); GPU-resident layers stay as-is
        if (!ggml_backend_buffer_is_host(L.ffn_gate_exps->buffer)) {
            continue;
        }

        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(model.dev_layer(il));

        ggml_init_params ip = {
            /*.mem_size   =*/ 8 * ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) {
            LLAMA_LOG_ERROR("%s: ggml_init failed at layer %d\n", __func__, il);
            return;
        }

        layer_slots ls;
        ls.il = il;
        ls.src_gate = L.ffn_gate_exps;
        ls.src_up   = L.ffn_up_exps;
        ls.src_down = L.ffn_down_exps;
        ls.gate = ggml_new_tensor_3d(ctx, L.ffn_gate_exps->type, L.ffn_gate_exps->ne[0], L.ffn_gate_exps->ne[1], S1);
        ls.up   = ggml_new_tensor_3d(ctx, L.ffn_up_exps->type,   L.ffn_up_exps->ne[0],   L.ffn_up_exps->ne[1],   S1);
        ls.down = ggml_new_tensor_3d(ctx, L.ffn_down_exps->type, L.ffn_down_exps->ne[0], L.ffn_down_exps->ne[1], S1);
        ls.e2s  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n_expert);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            LLAMA_LOG_ERROR("%s: slot buffer alloc FAILED at layer %d (out of VRAM?)\n", __func__, il);
            ggml_free(ctx);
            return;
        }
        total_bytes += ggml_backend_buffer_get_size(buf);

        ls.e2s_host.assign(n_expert, n_slots);   // all experts -> dummy miss slot initially
        ggml_backend_tensor_set(ls.e2s, ls.e2s_host.data(), 0, ggml_nbytes(ls.e2s));

        // the dummy slot (index n_slots) is the miss sink: zero it so missed experts contribute
        // nothing to the GPU path. (Unpromoted real slots are never indexed until promoted.)
        {
            std::vector<char> z(ls.gate->nb[2], 0);
            ggml_backend_tensor_set(ls.gate, z.data(), (size_t) n_slots * ls.gate->nb[2], ls.gate->nb[2]);
            z.assign(ls.up->nb[2], 0);
            ggml_backend_tensor_set(ls.up,   z.data(), (size_t) n_slots * ls.up->nb[2],   ls.up->nb[2]);
            z.assign(ls.down->nb[2], 0);
            ggml_backend_tensor_set(ls.down, z.data(), (size_t) n_slots * ls.down->nb[2], ls.down->nb[2]);
        }

        ctxs.emplace_back(ctx);
        bufs.emplace_back(buf);
        layers.push_back(std::move(ls));
    }

    // skip masks live on the CPU (read by mul_mat_id_skip, which runs on CPU with the CPU-resident
    // weights); one small I8 [n_expert] tensor per cached layer, all in one CPU buffer.
    if (!layers.empty()) {
        ggml_init_params ipc = {
            /*.mem_size   =*/ (layers.size() + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx_cpu = ggml_init(ipc);
        for (auto & ls : layers) {
            ls.skip = ggml_new_tensor_1d(ctx_cpu, GGML_TYPE_I8, n_expert);
        }
        ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(ctx_cpu, ggml_backend_cpu_buffer_type());
        if (!buf_cpu) {
            LLAMA_LOG_ERROR("%s: skip-mask CPU buffer alloc FAILED\n", __func__);
            ggml_free(ctx_cpu);
            return;
        }
        std::vector<int8_t> zeros(n_expert, 0);
        for (auto & ls : layers) {
            ggml_backend_tensor_set(ls.skip, zeros.data(), 0, ggml_nbytes(ls.skip)); // empty: skip nothing
        }
        ctxs.emplace_back(ctx_cpu);
        bufs.emplace_back(buf_cpu);
    }

    LLAMA_LOG_INFO("%s: cached %zu CPU-resident MoE layers, %d(+1) slots each, %.1f MiB VRAM\n",
                   __func__, layers.size(), n_slots, total_bytes / (1024.0 * 1024.0));

    // M2 gate: promote expert 0 -> slot 0 on the first cached layer and verify the bytes round-trip.
    if (!layers.empty()) {
        auto & ls = layers[0];
        promote(ls, /*expert_id =*/ 0, /*slot =*/ 0);
        const size_t st = ls.src_gate->nb[2];
        std::vector<char> back(st);
        ggml_backend_tensor_get(ls.gate, back.data(), 0, st);
        const bool ok = memcmp(back.data(), (const char *) ls.src_gate->data, st) == 0;
        LLAMA_LOG_INFO("%s: M2 promote self-test (layer %d expert 0 -> slot 0, gate %zu B): %s\n",
                       __func__, ls.il, st, ok ? "MATCH" : "MISMATCH");
    }
}

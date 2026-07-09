#include "llama-moe-slot-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
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
    auto cache = std::make_unique<llama_moe_slot_cache>(s);
    cache->async_promote = getenv("GGML_MOE_SLOT_SYNC") == nullptr; // async (phase 2) default; SYNC=1 => phase 1
    cache->compute_disabled = getenv("GGML_MOE_SLOT_NOSLOT") != nullptr; // isolate bookkeeping vs two-path
    cache->book_disabled    = getenv("GGML_MOE_SLOT_NOBOOK") != nullptr; // isolate allocation vs bookkeeping
    if (cache->book_disabled) { cache->compute_disabled = true; }        // NOBOOK implies no slot compute
    cache->promo_disabled   = getenv("GGML_MOE_SLOT_NOPROMO") != nullptr; // cpy-sink on, update off
    cache->cap_disabled     = getenv("GGML_MOE_SLOT_NOCAP")   != nullptr; // two-path on, capture+update off
    if      (getenv("GGML_MOE_SLOT_CALLBACK")) { cache->capture_mode = 1; } // eval callback (diagnostic)
    else if (getenv("GGML_MOE_SLOT_CPYSINK"))  { cache->capture_mode = 2; } // CPU cpy-sink (diagnostic)
    else if (getenv("GGML_MOE_SLOT_GPUSINK"))  { cache->capture_mode = 0; } // GPU-sink (phase-2 default, diagnostic)
    else                                       { cache->capture_mode = 3; } // fused into mul_mat_id_skip (P3.0, default)
    if (const char * r = getenv("GGML_MOE_SLOT_RING")) {
        const int rr = atoi(r);
        if (rr > 0) { cache->ring_size = rr; }
    }
    cache->promote_budget = cache->ring_size; // default: promote up to a full ring per token
    if (const char * b = getenv("GGML_MOE_SLOT_BUDGET")) {
        const int bb = atoi(b);
        if (bb > 0) { cache->promote_budget = bb; }
    }
    LLAMA_LOG_INFO("%s: MoE slot cache enabled, %d slots/layer, promotion=%s, ring=%d, budget/token=%d\n",
                   __func__, s, cache->async_promote ? "async" : "sync", cache->ring_size, cache->promote_budget);
    return cache;
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

llama_moe_slot_cache::~llama_moe_slot_cache() {
    if (copy_backend) {
        ggml_backend_synchronize(copy_backend);
    }
    for (auto & rs : ring) {
        if (rs.evt) { ggml_backend_event_free(rs.evt); }
        if (rs.buf) { ggml_backend_buffer_free(rs.buf); }
    }
    if (copy_backend) {
        ggml_backend_free(copy_backend);
    }
    if (reqs > 0) {
        // teardown runs after the log backend is gone; write straight to stderr
        fprintf(stderr, "\n=== MoE slot cache: online hit rate %.1f%% (%llu/%llu requests, %d slots/layer, "
                "%s, %llu promotions) ===\n",
                100.0 * (double) hits / (double) reqs, (unsigned long long) hits, (unsigned long long) reqs,
                n_slots, async_promote ? "async" : "sync", (unsigned long long) promotions);
        if (update_calls > 0) {
            fprintf(stderr, "=== MoE slot cache: token-boundary host cost avg %.3f ms/call (%llu calls) ===\n",
                    (double) update_ns / (double) update_calls / 1e6, (unsigned long long) update_calls);
        }
        fflush(stderr);
    }
}

void llama_moe_slot_cache::init(const llama_model & model, int n_expert_used_, int n_ubatch_) {
    if (initialized) {
        return;
    }
    initialized = true;

    n_expert       = (int) model.hparams.n_expert;
    n_expert_used  = n_expert_used_;
    n_ubatch       = n_ubatch_;
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
            /*.mem_size   =*/ (2 * layers.size() + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx_cpu = ggml_init(ipc);
        for (auto & ls : layers) {
            ls.skip   = ggml_new_tensor_1d(ctx_cpu, GGML_TYPE_I8,  n_expert);
            ls.routed = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_I32, n_expert_used, n_ubatch);
        }
        ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(ctx_cpu, ggml_backend_cpu_buffer_type());
        if (!buf_cpu) {
            LLAMA_LOG_ERROR("%s: skip/routed CPU buffer alloc FAILED\n", __func__);
            ggml_free(ctx_cpu);
            return;
        }
        std::vector<int8_t> zeros(n_expert, 0);
        for (auto & ls : layers) {
            ls.skip_host.assign(n_expert, 0);
            ls.slot_of.assign(n_expert, -1);
            ls.mru.clear();
            ggml_backend_tensor_set(ls.skip, zeros.data(), 0, ggml_nbytes(ls.skip)); // empty: skip nothing
        }
        ctxs.emplace_back(ctx_cpu);
        bufs.emplace_back(buf_cpu);
    }

    // GPU-sink capture (default): one [n_expert_used, n_ubatch, n_layers] i32 GPU tensor. Each cached
    // layer cpys its routed ids to its z-slice (GPU->GPU, no graph split); one batched D2H per token.
    if (capture_mode == 0 && !layers.empty()) {
        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(model.dev_layer(layers[0].il));
        ggml_init_params ipg = { 2 * ggml_tensor_overhead(), nullptr, /*.no_alloc=*/ true };
        ggml_context * ctx_g = ggml_init(ipg);
        routed_gpu = ggml_new_tensor_3d(ctx_g, GGML_TYPE_I32, n_expert_used, n_ubatch, (int64_t) layers.size());
        ggml_backend_buffer_t buf_g = ggml_backend_alloc_ctx_tensors_from_buft(ctx_g, buft);
        if (!buf_g) {
            LLAMA_LOG_ERROR("%s: routed_gpu buffer alloc FAILED\n", __func__);
            ggml_free(ctx_g);
            return;
        }
        routed_host.assign((size_t) n_expert_used * n_ubatch * layers.size(), 0);
        routed_ctx.reset(ctx_g);
        routed_buf.reset(buf_g);
    }

    // phase-2: dedicated copy stream (separate backend instance) + pinned staging ring
    if (async_promote && !layers.empty()) {
        ggml_backend_dev_t dev = model.dev_layer(layers[0].il);
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
        if (!host_buft) {
            LLAMA_LOG_WARN("%s: no pinned host buffer type; using synchronous promotion\n", __func__);
            async_promote = false;
        } else {
            copy_backend = ggml_backend_dev_init(dev, nullptr);
            size_t ring_bytes = 0;
            for (auto & ls : layers) {
                ring_bytes = std::max(ring_bytes, ls.src_gate->nb[2] + ls.src_up->nb[2] + ls.src_down->nb[2]);
            }
            ring.resize(ring_size);
            bool ok = (copy_backend != nullptr);
            for (auto & rs : ring) {
                rs.buf  = ggml_backend_buft_alloc_buffer(host_buft, ring_bytes);
                rs.ptr  = rs.buf ? ggml_backend_buffer_get_base(rs.buf) : nullptr;
                rs.evt  = ggml_backend_event_new(dev);
                rs.busy = false;
                if (!rs.ptr || !rs.evt) { ok = false; }
            }
            if (!ok) {
                LLAMA_LOG_WARN("%s: async promotion setup failed; using synchronous promotion\n", __func__);
                async_promote = false;
            } else {
                LLAMA_LOG_INFO("%s: async promotion: dedicated copy stream + %d x %.1f MiB pinned ring (%.0f MiB)\n",
                               __func__, ring_size, ring_bytes / (1024.0 * 1024.0),
                               ring_size * ring_bytes / (1024.0 * 1024.0));
            }
        }
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

void llama_moe_slot_cache::upload_maps(layer_slots & ls) {
    ggml_backend_tensor_set(ls.e2s,  ls.e2s_host.data(),  0, ggml_nbytes(ls.e2s));
    ggml_backend_tensor_set(ls.skip, ls.skip_host.data(), 0, ggml_nbytes(ls.skip));
}

void llama_moe_slot_cache::promote_async(int layer_idx, int expert, int slot) {
    int r = -1;
    for (int i = 0; i < (int) ring.size(); ++i) { if (!ring[i].busy) { r = i; break; } }
    if (r < 0) { return; }

    layer_slots & ls = layers[layer_idx];
    char * base = (char *) ring[r].ptr;
    const size_t g = ls.src_gate->nb[2], u = ls.src_up->nb[2], d = ls.src_down->nb[2];
    // stage the expert's gate/up/down slices contiguously in the pinned ring buffer
    memcpy(base,         (const char *) ls.src_gate->data + (size_t) expert * g, g);
    memcpy(base + g,     (const char *) ls.src_up->data   + (size_t) expert * u, u);
    memcpy(base + g + u, (const char *) ls.src_down->data + (size_t) expert * d, d);
    // async copy pinned ring -> GPU slots on the dedicated copy stream
    ggml_backend_tensor_set_async(copy_backend, ls.gate, base,         (size_t) slot * g, g);
    ggml_backend_tensor_set_async(copy_backend, ls.up,   base + g,     (size_t) slot * u, u);
    ggml_backend_tensor_set_async(copy_backend, ls.down, base + g + u, (size_t) slot * d, d);
    ggml_backend_event_record(ring[r].evt, copy_backend);
    ring[r].busy = true; ring[r].layer_idx = layer_idx; ring[r].expert = expert; ring[r].slot = slot;
    promotions++;
}

void llama_moe_slot_cache::poll_completions() {
    for (auto & rs : ring) {
        if (!rs.busy) { continue; }
        if (!ggml_backend_event_query(rs.evt)) { continue; } // copy still in flight
        layer_slots & ls = layers[rs.layer_idx];
        ls.e2s_host[rs.expert]  = rs.slot; // now compute-resident (bytes have landed)
        ls.skip_host[rs.expert] = 1;
        upload_maps(ls);
        rs.busy = false;
    }
}

bool llama_moe_slot_cache::eval_capture(struct ggml_tensor * t, bool ask) {
    const ggml_tensor * src0 = t->src[0];
    if (ask) {
        // observe the gate mul_mat_id of cached CPU layers only (up/down share the same ids)
        if (t->op != GGML_OP_MUL_MAT_ID || !src0 || !strstr(src0->name, "ffn_gate_exps")) {
            return false;
        }
        int layer = -1;
        return sscanf(src0->name, "blk.%d.", &layer) == 1 && find(layer) != nullptr;
    }
    // data phase: copy the routed ids to host (no graph node, no cross-backend cpy in the graph)
    int layer = -1;
    if (sscanf(src0->name, "blk.%d.", &layer) != 1) { return true; }
    const layer_slots * cls = find(layer);
    if (!cls || !cls->routed) { return true; }
    const ggml_tensor * ids = t->src[2];
    if (!ids || !ggml_is_contiguous(ids)) { return true; }
    const size_t nb = ggml_nbytes(ids);
    if (nb > ggml_nbytes(cls->routed)) { return true; } // n_tokens exceeded n_ubatch (guard)
    ggml_backend_tensor_get(ids, cls->routed->data, 0, nb); // [n_expert_used, n_tokens] prefix of routed
    return true;
}

void llama_moe_slot_cache::download_routing() {
    if (capture_mode == 0 && routed_gpu) {
        // one batched D2H at the token boundary (pipeline already idle): all layers' ids at once
        ggml_backend_tensor_get(routed_gpu, routed_host.data(), 0, ggml_nbytes(routed_gpu));
    }
}

void llama_moe_slot_cache::update_after_decode(int n_tokens) {
    if (!initialized || layers.empty()) {
        return;
    }
    const auto t_begin = std::chrono::steady_clock::now();
    download_routing(); // GPU-sink: bring this eval's routed ids to host in one copy
    if (async_promote) {
        poll_completions(); // land finished promotions first (frees rings, publishes maps)
    }

    auto touch = [](layer_slots & ls, int e) {
        for (size_t i = 0; i < ls.mru.size(); ++i) {
            if (ls.mru[i] == e) {
                if (i != 0) { ls.mru.erase(ls.mru.begin() + i); ls.mru.insert(ls.mru.begin(), e); }
                return;
            }
        }
    };

    int budget = promote_budget; // GLOBAL per-token promotion budget (FIFO across layers)
    for (int li = 0; li < (int) layers.size(); ++li) {
        layer_slots & ls = layers[li];
        const int32_t * routed = layer_routing(li); // GPU-sink: routed_host slice; else ls.routed
        bool sync_dirty = false;
        for (int t = 0; t < n_tokens; ++t) {
            for (int k = 0; k < n_expert_used; ++k) {
                const int e = routed[k + t*n_expert_used];
                if (e < 0 || e >= n_expert) {
                    continue;
                }
                reqs++;
                if (ls.e2s_host[e] != n_slots) { hits++; touch(ls, e); continue; } // resident (landed)
                if (ls.slot_of[e]  != -1)      {         touch(ls, e); continue; } // in-flight: slot reserved

                // genuine miss with no slot assigned -> maybe promote
                if (!async_promote) {
                    int slot;
                    if ((int) ls.mru.size() < n_slots) { slot = (int) ls.mru.size(); }
                    else {
                        const int v = ls.mru.back(); ls.mru.pop_back();
                        slot = ls.slot_of[v]; ls.slot_of[v] = -1; ls.e2s_host[v] = n_slots; ls.skip_host[v] = 0;
                    }
                    promote(ls, e, slot);
                    ls.slot_of[e] = slot; ls.e2s_host[e] = slot; ls.skip_host[e] = 1;
                    ls.mru.insert(ls.mru.begin(), e);
                    promotions++; sync_dirty = true;
                } else {
                    if (budget <= 0) { continue; }
                    bool ring_free = false;
                    for (auto & rs : ring) { if (!rs.busy) { ring_free = true; break; } }
                    if (!ring_free) { continue; } // no staging buffer free; promote a later token
                    int slot; bool demoted = false;
                    if ((int) ls.mru.size() < n_slots) { slot = (int) ls.mru.size(); }
                    else {
                        const int v = ls.mru.back(); ls.mru.pop_back();
                        slot = ls.slot_of[v]; ls.slot_of[v] = -1; ls.e2s_host[v] = n_slots; ls.skip_host[v] = 0;
                        demoted = true;
                    }
                    ls.slot_of[e] = slot; ls.mru.insert(ls.mru.begin(), e); // reserve; e2s stays dummy until landed
                    if (demoted) { upload_maps(ls); } // demote-first: victim -> dummy visible BEFORE the copy
                    promote_async(li, e, slot);
                    budget--;
                }
            }
        }
        if (!async_promote && sync_dirty) { upload_maps(ls); }
    }
    update_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - t_begin).count();
    update_calls++;
}

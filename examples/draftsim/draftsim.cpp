// llama-draftsim — teacher-forced argmax capture for EXPERIMENTS.md item 17 gate (b).
//
// Feeds a fixed token sequence through the model one token at a time (positions 0..T-1, so the
// GGML_MOE_DRAFT_SIM per-position resident masks line up) and records the model's greedy next-token
// prediction (argmax logits) at every position. Run it twice on the same sequence:
//   TRUE  : GGML_MOE_TRACE=<ids>  GGML_DRAFTSIM_OUT=<true.argmax>          (masking off; also dumps routing)
//   DRAFT : GGML_MOE_DRAFT_SIM=<mask> GGML_DRAFTSIM_OUT=<draft.argmax>     (routing restricted to resident)
// then compare the two argmax streams offline (top-1 agreement, agreement@K). An all-resident mask
// must reproduce the TRUE stream byte-for-byte (validation gate). This is a measurement harness only.
//
// Output format: int32 magic 'ARGX' (0x58475241), int32 n_pos, then int32[n_pos] argmax token ids.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    common_params params;
    params.n_predict = 1;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }
    // teacher-forced single-token stepping: no warmup decode (keeps positions aligned with the masks),
    // and the whole sequence must fit the context window.
    params.warmup = false;

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    common_init_result_ptr init = common_init_from_params(params);
    llama_model   * model = init->model();
    llama_context * ctx   = init->context();
    if (!model || !ctx) {
        fprintf(stderr, "draftsim: failed to load model/context\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true, true);
    if (tokens.empty()) {
        fprintf(stderr, "draftsim: empty prompt\n");
        return 1;
    }
    const int n_ctx = llama_n_ctx(ctx);
    int T = (int) tokens.size();
    if (T > n_ctx) {
        fprintf(stderr, "draftsim: truncating %d tokens to n_ctx=%d\n", T, n_ctx);
        T = n_ctx;
    }
    fprintf(stderr, "draftsim: %d tokens, n_ctx=%d, n_vocab=%d\n", T, n_ctx, n_vocab);

    std::vector<int32_t> argmax(T, -1);

    // Single-token teacher-forced decode: token t at position t, read logits predicting t+1.
    for (int t = 0; t < T; ++t) {
        llama_batch batch = llama_batch_get_one(&tokens[t], 1);
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "draftsim: decode failed at %d\n", t);
            return 1;
        }
        const float * logits = llama_get_logits_ith(ctx, -1);
        int best = 0;
        float bestv = logits[0];
        for (int v = 1; v < n_vocab; ++v) {
            if (logits[v] > bestv) { bestv = logits[v]; best = v; }
        }
        argmax[t] = best;
        if ((t % 512) == 0) { fprintf(stderr, "\rdraftsim: %d/%d", t, T); fflush(stderr); }
    }
    fprintf(stderr, "\rdraftsim: %d/%d done\n", T, T);

    const char * out = getenv("GGML_DRAFTSIM_OUT");
    if (out && *out) {
        FILE * f = fopen(out, "wb");
        if (!f) { fprintf(stderr, "draftsim: cannot open %s\n", out); return 1; }
        int32_t hdr[2] = { 0x58475241, T };
        fwrite(hdr, sizeof(int32_t), 2, f);
        fwrite(argmax.data(), sizeof(int32_t), T, f);
        fclose(f);
        fprintf(stderr, "draftsim: wrote %d argmax ids to %s\n", T, out);
    } else {
        fprintf(stderr, "draftsim: GGML_DRAFTSIM_OUT unset — no argmax written\n");
    }

    llama_backend_free();
    return 0;
}

# Persistent GPU expert slot cache - design proposal (item 15)

Status: **proposal, pre-implementation.** Needs human + Fable review before any CUDA is
written. No code here. Grounded in measured data (EXPERIMENTS.md item 1, Tier 0) and the
moe-cache post-mortem; aligned with upstream issue #20757.

## Problem

gpt-oss-120b MXFP4 (59 GB) does not fit in 32 GB VRAM. Best clean config offloads MoE
experts of `ncmoe` layers to CPU (`-ncmoe 22`: ~41 tg). Generation is DDR5-bandwidth-bound:
each token streams ~1.2 GB of expert weights from host RAM. The routed experts are highly
skewed per layer (Gini 0.72) and reused across consecutive tokens - so most of that streaming
is redundant. A GPU-resident cache of the currently-hot experts turns most expert reads into
on-GPU accesses.

## What the measurement already decided (item 1)

Offline simulation over real per-token routing traces (wikitext 47.7K, code 40.9K, chat 52K
tokens; hook validated vs imatrix at gini-diff 0.02-0.03):

- **Topology: per-layer independent pools.** Per-layer == a shared cross-layer pool within
  1-2 points at every budget. Per-layer is far simpler (no global slot map, no cross-layer
  eviction) - build per-layer.
- **Eviction: plain LRU.** Layer-cycle-aware eviction (the issue-#20757 / SpecMD suggestion)
  ties or loses to LRU on all three workloads - the "LRU provably wrong" case needs the pool
  smaller than one layer-cycle's working set, which does not hold here. LFU-decay adds only
  +4-6 pts and only at <=16 slots. Use LRU.
- **Sizing: hit rate vs slots/layer (per-layer LRU), min across the three workloads:**

  | slots/layer | 8 | 16 | 24 | 32 | 48 | 64 |
  |---|---|---|---|---|---|---|
  | worst-workload hit % | 40 | 59 | 74 | 83 | 92 | 97 |

  48 slots/layer (37.5% of 128 experts) -> 92-96% across workloads; 64 -> 96-98%. Below 32
  slots workload sensitivity is large (chat 63% vs code 40% at 8), so do not go below ~32.
- **Predictor headroom (oracle - LRU): 4-9 pts at 32 slots, 2-5 at 48.** The optional
  pre-attention MLP prefetcher is a phase-2 refinement, not the core win.

## VRAM budget (the binding constraint)

Per expert (gate+up+down, MXFP4) ~= 12 MB (55 GB experts / 4608 expert-instances). The cache
covers only the `ncmoe` CPU-resident layers:

| slots/layer | x22 CPU layers | GPU cost |
|---|---|---|
| 24 | 528 experts | ~6.3 GB |
| 32 | 704 | ~8.4 GB |
| 48 | 1056 | ~12.7 GB |

The slot cache competes with statically-GPU-resident experts for the same VRAM. On a 32 GB
card with `-ncmoe 22` the static split already uses most of it, so the practical first target
is **~24-32 slots/layer (74-88% hit, ~6-8 GB)**, raising `ncmoe` if needed to free VRAM. Tune
slots to fit; the sim curve says the marginal experts are cheap in hit rate above 48 anyway.

## Failure mode to avoid (moe-cache post-mortem)

The moe-cache branch routed **all 36 layers** through the cache and paid **108 host syncs/token**
(3 tensors x 36 layers) plus de-residenting churn -> tg regressed 12 -> 10.8. Lessons baked into
this design:

1. **Cache only the CPU-resident (`ncmoe`) layers.** GPU-resident layers keep their dense
   buffers and fusion - never route them through the cache.
2. **One routing snapshot per layer.** gate/up/down share the same routed ids; decode ids once
   per layer, not three times.
3. **Hot path (cache hits) must be sync-free.** No per-token host<->device sync for hits. Only
   misses touch the copy engine, asynchronously.

## Proposed mechanism

**Placement: graph-level MoE hook for CPU layers, not a custom buffer type.** A buffer type
cannot split one `mul_mat_id` across resident/non-resident experts. Instead, at each CPU MoE
layer during decode:

1. **Slot pool (per CPU layer):** **three** separate GPU slot buffers (gate/up/down, one per
   `_exps` tensor - review finding 4b; a triple-packed buffer would need a custom stride), each
   holding `S` experts, plus a host-side `expert_to_slot[128]` map (-1 = not resident) and an
   LRU recency list.
2. **Split the routed top-4 into hits and misses** using `expert_to_slot`.
3. **Hits -> GPU:** remap routed expert ids to slot ids and run `mul_mat_id` against the slot
   buffer. No sync.
4. **Misses -> CPU now:** compute the missed experts on CPU against the existing host weights
   (no stall waiting for an upload), then **async-promote** each missed expert into the LRU
   victim slot via `cudaMemcpyAsync` on a dedicated copy stream; update `expert_to_slot` when
   the copy completes (arXiv 2512.16473 compute-now/promote-later). Next token that routes it
   hits.
5. Combine hit-output and miss-output by routed position, weight, and sum as today.

**Slot indirection:** the only per-token host work is reading the small routed-ids tensor
(already needed) and updating two small host arrays - O(4) per layer, no large sync. Steady
state (>90% hits) runs almost entirely on GPU.

**Eviction:** per-layer LRU over slot occupants. Warmup fills empty slots first.

## Staged implementation plan (each gate cites a measurement)

- **Phase 0 - telemetry.** Add per-layer hit/miss counters (extend moe-cache stats dcc2cb143).
  *Gate:* online hit rate matches the offline sim within +/-5% on the same workload. If not,
  the model of reuse is wrong - stop and reconcile before building the cache.
- **Phase 1 - synchronous slot cache, CPU layers only.** Per-layer LRU pool, hits on GPU /
  misses on CPU, promote synchronously (simplest, correct). *Gates (revised per review
  finding 2 - do NOT hold phase 1 to 41 tg):* (a) temp-0 seed-fixed output byte-identical to
  baseline; (b) online hit rate matches the offline sim within +/-5% on the same workload;
  (c) tg above a sanity floor (~35) - synchronous ~4-8 misses/token x 12 MB blocking H2D can
  legitimately sit below 41 while the architecture is sound. The >=41 bar is phase 2's.
- **Phase 2 - async promote (compute-now/promote-later). SUPERVISED (review finding 3).** Move
  promotion to a copy stream through a small **pinned staging ring** (2x16 MB; shard-3 experts
  are unpinned -> pageable copies otherwise, and the copy must respect the one-registered-region
  rule - this is item 12's machinery at small scale). Keep the hit path sync-free. *Gates:*
  **equal-VRAM win (finding 1)** - N GB of dynamic slots must beat the best static config using
  the same N GB, in tg (reframes "beat 41 tg"; slots do NOT fit at ncmoe 22, so run ncmoe ~26-28);
  tg improves over phase 1; nsys confirms promotions are on the copy stream and the decode stream
  has no per-token host sync; p99 inter-token latency not worse.
- **Phase 3 (optional) - predictor prefetch.** 2-layer MLP (2511.10676) predicting next-token
  top-4 from the current hidden state, prefetch during attention. *Gate:* only if phase 2
  leaves a measured hit-rate gap and the tg gain exceeds the added complexity (headroom is only
  4-9 pts per item 1) - otherwise ship phase 2.

## Open questions - RESOLVED in review (finding 4)

- Two `mul_mat_id` calls per layer (GPU-hits + CPU-misses, combined on GPU) is the phase-1
  shape - ids are already host-read, so the split/remap are host array ops and the CPU-miss
  activation round-trip is status quo, not a new sync. **[resolved: build this]**
- Slot buffer layout: **three** separate buffers (gate/up/down), not a packed triple.
  **[resolved]**
- Copy-stream contention at decode is a non-issue (no prefill uploads in flight); measure
  anyway at phase 2 via nsys. **[resolved: defer to phase 2]**
- Slot/`ncmoe` autotune: skip until phase 2 ships; hand-tuned slots-per-free-VRAM is fine for
  the experiment. **[resolved: defer]**

## Baseline to beat

**Equal-VRAM (review finding 1), not raw 41 tg.** Slots do not fit at ncmoe 22 (~3 GB free),
so the cache runs at ncmoe ~26-28; the fair test is N GB of dynamic slots vs the same N GB of
static expert layers, both in tg. (41 tg at ncmoe 22 remains the reference point.) **Hit rate
is necessary but not sufficient** - the moe-cache regression proves a high-hit-rate cache can
still lose to sync overhead. Measure tg at each gate; never project it from hit rate.

## Review (Fable, 2026-07-08) — approved with four findings; address before phase 1

1. **The VRAM table understates the constraint: slots do not fit at ncmoe 22.** At ncmoe 22
   the static split leaves ~3 GB free — not even the 24-slot config (6.3 GB) fits. The cache
   necessarily runs at higher ncmoe (~26-28), which means the honest evaluation is
   **equal-VRAM**: N GB spent on dynamic slots vs the same N GB spent on static expert
   layers, both measured in tg. Make this the explicit phase-2 success criterion (sim
   arithmetic favors slots — 6 GB = 4 static layers (18% of CPU work removed,
   deterministic) vs 24 slots/layer across all CPU layers at 74-88% hit — but it must be
   measured, and it reframes "beat 41 tg" as "beat the best static config at the same
   memory").
2. **Phase-1 gate is too strict and could kill a healthy design.** Synchronous promotion
   costs ~4-8 misses/token x 12 MB of blocking H2D early on; phase 1 can easily sit below
   41 tg while the architecture is sound. Split the gate: phase 1 = correctness
   (temp-0 byte-identical) + online hit rate matching the sim ±5% + tg above a sanity floor
   (~35); the ≥41/equal-VRAM bar belongs to phase 2.
3. **Promotion path has a hidden dependency on pinned staging (item 12's machinery).**
   Misses promote via cudaMemcpyAsync from mmap'd host weights — but shard 3's experts are
   unpinned (36 GB WDDM cap), so those promotions are pageable copies (~2-3 GB/s, serializing
   the copy stream) and the copy must respect the one-registered-region rule (the
   copy-straddle landmine). Phase 2 should bounce promotions through a small pinned staging
   ring (2x16 MB suffices at ~12 MB/expert). That is item 12's mechanism at smaller scale —
   build it inside the cache first; it de-risks the standalone item 12 later. Supervised
   session for this piece.
4. **Open questions, answered where the code already decides them:** (a) two mul_mat_id
   calls per layer is the right phase-1 shape — ids are already read to host on this path, so
   the hit/miss split and slot-id remap are host-side array ops; the combine happens on GPU
   where the weighted sum already lives, and the CPU-miss branch's activation round-trip is
   the status quo for ncmoe layers, not a new sync. (b) three separate slot buffers
   (gate/up/down), one per _exps tensor — mul_mat_id consumes per-matrix expert tensors, so
   a triple-packed buffer would need a custom stride. (c) copy-stream contention at decode is
   a non-issue (no prefill uploads in flight); measure anyway at phase 2 via nsys.
   (d) skip the autotune until phase 2 ships; hand-tuned slots-per-free-VRAM is fine for the
   experiment.

Green light for phase 0 and phase 1 unattended; phase 2 supervised (finding 3).

## Phase status (2026-07-08, continued Opus run)

### Phase 0 - DONE, gate PASSED.
In-engine per-layer LRU counter `GGML_MOE_CACHE_SIM=S1,S2,...` added to `common/moe-trace.cpp`
(branch `experiments/slot-cache`, reuses the routing-trace hook's ids read; reports hit% per
slot budget at process exit). Validated on a 6.1K-token wikitext decode: online == offline
(`simulate-cache.py`) **exactly** at every budget - 8/16/24/32/48/64 slots -> 43.6 / 63.3 /
74.5 / 82.3 / 91.9 / 96.5 % on both, 36 layers, 883,152 requests. The offline reuse model is
faithful in-engine; cleared to build the cache. Logs: `bench-results/p0-run2.log`.

### Phase 1 - NOT STARTED; re-scoped after reading the code (bigger than the plan assumed).
Two findings from `src/llama-graph.cpp` `build_moe_ffn` and the moe-cache branch change the
phase-1 estimate. Surface to the human before committing a run to it:

1. **Static-graph obstacle (the crux).** Expert compute is a single `ggml_mul_mat_id` per
   tensor over the *full* expert weight, indexed by `selected_experts`
   (`build_lora_mm_id(gate_exps, cur, selected_experts, ...)`, ~line 1988; same for up/down).
   The hit/miss split depends on `selected_experts` *values*, which do not exist at graph-build
   time (ggml builds the graph before execution). So the design's "two mul_mat_ids (hits over
   slots + misses over CPU)" cannot be expressed as a static graph. Two ways out:
   - **(a) Custom ggml/CUDA op** that, at compute time (ids already on-device), gathers each
     routed expert's rows from its GPU slot if resident else from the CPU-mapped weight, then
     matmuls. Keeps the hot path on-device (no per-token host sync) - the *right* design, but a
     new CUDA op (substantial).
   - **(b) Host-sync dispatch:** read `selected_experts` to host per layer, partition, dispatch
     per-layer ops. Simple but ~36 host syncs/token - the exact moe-cache failure mode. Only
     useful as a throwaway correctness oracle, not the shippable path.
2. **The moe-cache branch is NOT reusable.** It sits on an ancient base (`git diff ai-main
   experiments/moe-cache`: 627 files, +35K/-107K lines). Porting it forward is larger than
   reimplementing against current `build_moe_ffn`. **Treat phase 1 as greenfield on current
   ai-main**, not "adapt moe-cache scaffolding."

**Revised phase-1 scope:** a focused custom-op implementation (option a) + per-layer LRU slot
buffers (3 per CPU layer) + synchronous promotion, behind the phase-1 gates (byte-identical
output, online hit rate matches sim +/-5%, tg >= ~35 floor). This is a dedicated,
verification-heavy CUDA effort - appropriate for a run scoped *only* to phase 1, and arguably
worth a human design check on the custom-op shape first (it is a new pattern per AGENTS.md).
Not attempted unattended-and-unverified in this session.

### Review addendum (Fable, 2026-07-08) — option (c), and a broken gate fixed

**Stopping was correct; phase 0 passing validates the design's core assumption in-engine.**
Two additions before the supervised phase-1 session:

**Option (c) — static graph via slot-map indirection; evaluate BEFORE committing to (a).**
The hit/miss split does not need to be decided at graph-build time if the indirection is a
*data* dependency instead of a *structure* one:

1. Per CPU layer, keep `expert_to_slot` as a small GPU tensor (128 x i32): slot index for
   resident experts, a dummy-slot sentinel for misses. Slot buffers get one extra zeroed
   dummy slot.
2. In the graph: `slot_ids = ggml_get_rows(expert_to_slot, selected_experts)` — on-device,
   no sync. GPU path: the EXISTING `mul_mat_id` over the slot buffer (S+1 experts) indexed
   by `slot_ids`. Misses land in the zero dummy slot and contribute nothing. No new CUDA op.
3. Miss path: the CPU `mul_mat_id` already runs per layer today (ids reach the CPU backend
   via the scheduler's existing cross-backend copy — this sync is the status quo that yields
   41 tg, not a new cost). Add a *mask-skip variant* of the CPU kernel (plain C, small):
   takes a miss-mask input and skips resident experts' rows — that skip IS the win (the
   DDR5 reads it avoids). Outputs combine by summation exactly as today's weighted sum.
4. Host side per layer: update LRU from ids (already host-visible on this path), async-upload
   the 512-byte map, event-ordered after the promotion copy so a stale map costs an extra
   miss, never a wrong result.

Cost comparison: (c) reuses two battle-tested kernels + one small C variant + graph wiring;
(a) is a new gather+matmul CUDA op. AGENTS.md's "prefer reusing existing infrastructure"
points at (c). Check first: CUDA `ggml_get_rows` must support I32 sources (if not, that
one small kernel is still far less than (a)). If (c) benches poorly (e.g. the per-layer CPU
op dependency dominates at high hit rates), (a) remains the fallback — decide on phase-1
measurements, not taste.

**Gate fix — "byte-identical vs baseline" is unsatisfiable and would fail a correct
implementation.** Moving hit experts from CPU to GPU legitimately changes numerics (different
kernels, FMA order, dequant path) even when perfectly correct. Replace with:
- byte-identical vs **the option-(b) host-sync oracle** (same placement, same kernels — build
  (b) first as a throwaway; this is what (b) is for);
- token-agreement and perplexity-delta-within-noise vs ai-main baseline (catches real bugs,
  tolerates legitimate numeric drift).

Phase-1 supervised session: implement (b) oracle → implement (c) → gate (c) against (b) →
bench; escalate to (a) only if (c)'s measured overhead demands it.

## Phase 1 execution log (2026-07-08, supervised run)

**Feasibility gate PASSED.** CUDA `get_rows` supports I32 src0 (`ggml/src/ggml-cuda/getrows.cu:195`)
and asserts I32 indices - so `slot_ids = get_rows(expert_to_slot, selected_experts)` works
on-device, no new kernel. Option (c) is viable.

**Integration surface (from src/llama-graph.{h,cpp}).** `build_moe_ffn` receives `up_exps /
gate_exps / down_exps` (residency = `ggml_backend_buffer_is_host`) and computes
`selected_experts` internally at ~line 1894; expert matmuls are `build_lora_mm_id(..., selected_experts, ...)`
at ~1975/1988/down. Blocker: `llm_graph_context` is per-graph/transient and carries no mutable
persistent handle (`hparams`/`cparams` are const refs; no `llama_context*`/`llama_model*`). The
slot cache persists across tokens, so it needs a persistent home threaded into the builder.

**Component milestones:**
- M1. Persistent cache object (owns per-layer slot buffers, expert_to_slot, LRU), reachable from build_moe_ffn.
- M2. GPU slot buffers: 3 per CPU layer, S+1 experts (slot S = zeroed dummy for misses); alloc + populate + synchronous promotion (raw MXFP4 row copy from the mmap'd weight).
- M3. CPU mask-skip mul_mat_id variant (ggml-cpu) - the skip of resident-expert rows IS the bandwidth win.
- M4. Option (b) host-sync oracle driver (correctness reference).
- M5. Option (c) get_rows indirection wiring (gate/up/down) + combine.
- M6. Gates: (c) byte-identical vs (b); token-agreement + ppl-within-noise vs baseline; tg bench (>= ~35 floor, then equal-VRAM).

**Two decisions flagged for supervision (AGENTS.md: pause on invasive / new-pattern changes):**
- D1 (plumbing): thread a persistent cache through core classes (clean but invasive) vs a
  file-scoped cache keyed by tensor pointer (matches the GGML_MOE_TRACE hook precedent;
  contained; slightly hacky). Blocks all code.
- D2 (miss compute): a new CPU mask-skip op is REQUIRED for a real tg win (reusing the full CPU
  matmul and subtracting hits saves no DDR5 bandwidth). Confirmed necessary, not optional.

### M1 DONE (2026-07-08). Plumbing landed + verified on `experiments/slot-cache` (d929a7607).
Design locked (owner's call): `llama_moe_slot_cache` owned by `llama_context` (KV-cache pattern,
per-session state), reached from `build_moe_ffn` via `llm_graph_params.moe_cache` (null ->
today's graph = off-switch). Env: `GGML_MOE_SLOT_CACHE=<slots/layer>`. Verified: compiles;
no env -> inert (byte-normal output); =32 -> cache created (enable line fires under `-v`),
output still correct. M1 is a no-op cache (plumbing only).

**M2 plan (next):** on the first `build_moe_ffn` call for each CPU-resident layer (residency =
`ggml_backend_buffer_is_host(gate_exps->buffer)`), register the layer's expert-tensor metadata
(shapes, MXFP4 type, per-expert byte stride = `nb[2]`). Lazily (KV-cache style) allocate the
cache's own `ggml_context` + a CUDA backend buffer sized `3 tensors x n_cpu_layers x (S+1) x
per-expert-bytes`; slot `S` per tensor is a zeroed dummy (miss sink). Promotion = copy expert
`e`'s contiguous slice (`src->data + e*nb[2]`, length `nb[2]`) into slot `s` via
`ggml_backend_tensor_set(slot_tensor, src, s*nb[2], nb[2])` (synchronous for phase 1). Host-side
`expert_to_slot[128]` mirror + LRU per layer, updated after decode. Gate on: buffer alloc
succeeds within VRAM budget (log MiB), and a promoted slot's bytes match the source (memcmp a
sample). Then M3 (CPU mask-skip op), M4 (oracle), M5 (get_rows wiring), M6 (gates+bench).

### M2 DONE (2026-07-08). Slot buffers + promotion, verified on `experiments/slot-cache` (70f50adbf).
Gate artifacts: `init: cached 36 CPU-resident MoE layers, 8(+1) slots each, 4084.6 MiB VRAM`;
`promote self-test (layer 0 expert 0 -> slot 0, gate 4406400 B): MATCH`; off-switch byte-identical.

### M3 SPEC - PAUSE FOR REVIEW (condition 2). CPU mask-skip indexed matmul. No code written yet.
Does NOT match the addendum verbatim (addendum = 2-sentence prose; this is the concrete op), so
per the ground rules this is written and pushed for review before implementation.

**Chosen form (least invasive):** reuse `GGML_OP_MUL_MAT_ID` with an OPTIONAL 4th source
`src[3] = skip_mask`; add one builder; branch only in the CPU forward kernel. No new op enum,
no other backend touched. Rejected alternatives: a new `GGML_OP_*` (invasive across every
backend's dispatch); appending a zero dummy expert to the CPU weight (needs a full weight copy,
defeats the memory point).

**Signature (ggml.c):**
```
ggml_tensor * ggml_mul_mat_id_skip(
    ggml_context * ctx,
    ggml_tensor  * as,    // [n_embd, n_ff, n_expert]  full expert weights, CPU-resident
    ggml_tensor  * b,     // [n_embd, n_expert_used, n_tokens]  activations
    ggml_tensor  * ids,   // [n_expert_used, n_tokens] i32  routed expert ids
    ggml_tensor  * skip); // [n_expert] i8  1 = expert resident on GPU (skip), 0 = miss (compute)
```
Builds a normal MUL_MAT_ID node (same shape inference / dst as `ggml_mul_mat_id(as,b,ids)`) but
sets `result->src[3] = skip`. `ggml_mul_mat_id` (3-src) is unchanged.

**Semantics (CPU forward):** dst = [n_ff, n_expert_used, n_tokens]. For each token t, used-slot k:
`e = ids[k,t]`; if `src[3]` present and `skip[e] != 0` -> write dst[:,k,t] = 0 and CONTINUE
(no read of `as[:,:,e]` - this avoided DDR5 read is the entire win); else compute dst[:,k,t] via
the existing per-expert matmul inner loop (identical numerics to today's mul_mat_id). When
`src[3]` is null, behavior is bit-identical to the current kernel. Runs on CPU because `as` is
CPU-resident; CUDA never sees a 4-src MUL_MAT_ID.

**Combine (in build_moe_ffn, M5):** GPU slot path (mul_mat_id over the S+1 slot buffer, misses ->
zeroed dummy slot) yields hits-correct/misses-zero; this CPU path yields misses-correct/hits-zero;
elementwise sum = full result. gate and up each get this treatment; down likewise on the combined
activations.

**M3 test plan (standalone, before any graph wiring; CPU backend):** construct random
`as[K,N,E]`, `b`, `ids`, and masks. Gate = all three exact:
- (a) skip = all-zero  -> output byte-identical to plain `ggml_mul_mat_id(as,b,ids)`.
- (b) skip = all-one   -> output all zeros.
- (c) skip = random    -> output equals `plain mul_mat_id` with the skipped (token,k) columns
  overwritten by zero (reference computed in the test).
Add as a `test-backend-ops`-style case or a tiny standalone in bench-results/. Only after (a)(b)(c)
pass does M4/M5 proceed. If any fails: STOP with artifacts (condition 3).

**Open question for reviewer:** skip-mask dtype/semantics - I8 boolean `[n_expert]` (host-derived
from e2s each token) as above, vs passing the raw `e2s` `[1,n_expert]` i32 and testing
`e2s[e] != n_slots` in the kernel (one fewer host array, but couples the kernel to the sentinel).
I lean I8 boolean (kernel stays dumb). Confirm before I build.

### M3 DONE (2026-07-08). ggml_mul_mat_id_skip, verified on `experiments/slot-cache` (80dc387b2).
Gates (tests/test-mul-mat-id-skip.cpp): (a) zero-mask==plain, (b) one-mask==zero, (c) random==ref,
(d) mask-mutation tracks - ALL PASS. CUDA supports_op rejects 4-src MUL_MAT_ID. Off-switch identical.

### M4 RULING (2026-07-08, owner-approved): host-sync (b) oracle dropped; replaced by gate structure below.
A faithful host-sync (b) would need to read selected_experts mid-eval and rebuild per-layer dispatch
- the exact static-graph machinery (c) exists to avoid - so it cannot serve as an independent
reference. Replaced by two byte-exact gates that together cover what (b) would have, without a
throwaway path:
- **M5 (whole-graph byte gate):** option-(c) wiring with a FORCED-EMPTY cache -> e2s all-dummy
  (get_rows sends every expert to the zeroed dummy slot, GPU path contributes 0) and skip_mask
  all-zero (CPU mul_mat_id_skip computes every expert = plain mul_mat_id). Sum == baseline,
  byte-identical. Exercises the indirection, dummy-slot zeroing, mask plumbing, and the combine.
- **M5b (unit byte gate, same-device):** standalone CUDA test (M3 style). Promote a known expert
  set into a slot buffer, build e2s, compare `mul_mat_id(slots, x, remap(ids))` vs
  `mul_mat_id(full, x, ids)` on CUDA - hit rows byte-identical, miss rows zero. Same device/kernel/
  dtype sidesteps CPU-vs-GPU numerics and gives the HIT path (promoted-slot compute + get_rows
  remap + slot indexing) the same exactness standard. Stronger than M2's byte-copy check, which
  only verified the copied bytes, not the compute that indexes them.
- **M6 (statistical gates):** real promotion -> token-agreement + ppl-within-noise vs baseline,
  online hit rate matches the Phase-0 sim (GGML_MOE_CACHE_SIM), tg >= ~35 floor.
Build order: M5b first (de-risks the slot primitive), then M5 wiring, then M6.

### M5b + M5 DONE (2026-07-08), verified on `experiments/slot-cache` (85058a0df, 1a3a3c2a0).
- M5b unit oracle (tests/test-slot-matmul.cpp, CUDA): hit rows byte-identical (max abs diff
  0.000e+00), miss rows zero. Slot compute/remap/offset correct.
- M5 whole-graph: forced-empty cache -> BYTE-IDENTICAL to baseline (48 tokens). Full (c) wiring
  correct (get_rows remap, dummy-slot zero, skip plumbing, combine). Off-switch == on-empty.

### M6 PLAN + design decision (routing capture). Real promotion + statistical gates.
The dynamic layer: per token, update per-layer LRU from the routed experts, promote misses
(sync copy expert->slot, evict LRU), upload e2s + skip for the next token. Token t's misses are
computed on CPU at t (skip=0 for them) and promoted for t+1 (compute-now/promote-later, sync).

**Design decision - how to get the routed expert ids to host for the LRU/promotion update:**
- **(A) graph cpy-sink [recommended]:** build_moe_ffn adds `cpy(selected_experts -> per-layer
  persistent CPU tensor)` as a graph output; llama_context reads it post-decode and updates the
  cache. No cb_eval conflict (the trace/counter hook owns cb_eval); tiny (neu*nt ints/layer).
  Cost: ~36 cross-backend cpys + a post-decode read/layer - i.e. per-token sync, the moe-cache
  failure mode for tg. Phase 1 accepts this (async promotion is phase 2); phase-1 tg gate is a
  >=35 FLOOR, not "tg improves".
- **(B) eval-callback:** capture ffn_moe_topk during eval like GGML_MOE_TRACE. Conflicts with the
  trace/counter (one cb_eval) and fires mid-eval (awkward to drive promotion). Rejected.

**M6 gates:** (1) token-agreement + ppl-within-noise vs baseline (GPU-computed hits vs CPU
baseline differ in the last bits, so ppl-within-noise is the robust gate, token-agreement the
sanity check); (2) actual cache hit rate matches the Phase-0 GGML_MOE_CACHE_SIM LRU curve for the
same S/workload (same LRU policy -> must agree); (3) tg128 >= ~35 floor.

### M6 DONE - PHASE 1 COMPLETE (2026-07-08), verified on `experiments/slot-cache` (d5649f3d3).
Dynamic per-token LRU promotion (option-A cpy-sink capture -> update_after_decode -> sync promote
+ map upload).

**Correctness gates - ALL PASS (exact):**
- ppl-within-noise: PPL **458.9669 OFF == 458.9669 ON** (byte-identical over 11 chunks). The
  MXFP4 GPU-slot + CPU-skip combine is numerically exact vs the all-CPU baseline.
- hit-rate matches Phase-0 sim: actual **43.6% == sim 43.6%** (S=8, identical 883152 requests).
- token-agreement: 100% identical (temp0, 64 tokens, real promotion).

**Performance floor - BELOW 35, as anticipated. Not a correctness failure; the phase-2 mandate.**

| tg128 (ncmoe36 fa1) | OFF | S=8 (43.6% hit) | S=32 (90.4% hit) |
|---|---|---|---|
| t/s | 19.40 | 3.49 | 1.47 |

perplexity/pass: OFF 68.6s, S=8 ON 279s (~4x). **More slots is worse** - the cost is the
synchronous machinery (blocking cudaMemcpy promotion + per-layer e2s/skip uploads + 36 cpy-sink
cross-backend syncs/token), and larger S churns more promotion during warmup. This quantifies,
on this exact design, the moe-cache post-mortem's failure mode.

**Verdict:** the slot cache is CORRECT (ppl byte-identical, hit rate exact - the whole mechanism
works end to end) but synchronous promotion makes it 5-13x slower than baseline. This is the
measured justification for **phase 2 (SUPERVISED): async promote on a copy stream + pinned staging
ring (review finding 3), and the equal-VRAM comparison (finding 1)**. Phase 1 delivered a correct,
gated foundation; phase 2 is where it earns its tg. Do NOT ship phase 1 as a tg win.

**Reusable, verified assets from phase 1:** ggml_mul_mat_id_skip (+CUDA guard), the slot-buffer /
promotion / get_rows-indirection primitives (tests/test-slot-matmul.cpp, test-mul-mat-id-skip.cpp),
GGML_MOE_SLOT_CACHE end-to-end, GGML_MOE_CACHE_SIM. Phase 2 swaps the promotion path from sync to
async; the compute/wiring/gates all carry forward.

### Phase 2 spec addendum (Fable, 2026-07-08) — async promotion: design constraints and gates

The new hazard class in phase 2 is **races between promotion copies and slot reads**. Pin these
before writing code:

1. **Token-boundary event polling, not stream-wait gymnastics.** Promotions run on a dedicated
   copy stream, each with a CUDA event recorded at completion. At each token boundary the host
   polls events (cudaEventQuery, non-blocking): only for COMPLETED promotions does it update the
   host LRU/e2s/skip-mask and enqueue the 512-byte e2s upload — on the DECODE stream, so stream
   order guarantees every subsequent get_rows sees the new map only after the slot bytes landed.
   An in-flight promotion is invisible: costs one extra miss, never a wrong read. No
   cudaStreamWaitEvent needed on the decode stream at all.
2. **Victim safety:** the LRU victim being overwritten must not be readable by in-flight compute.
   With map updates deferred to token boundaries this holds by construction (the old occupant's
   e2s entry flips to dummy in the same boundary update BEFORE the promotion into that slot is
   enqueued — enforce this ordering: demote first, promote after).
3. **Staging ring (finding 3, mandatory):** promotions must NOT cudaMemcpyAsync directly from
   mmap'd weights — shard-3 experts are unpinned (pageable copies serialize the copy stream) and
   direct copies risk the one-registered-region rule. Route every promotion: host memcpy expert
   rows -> pinned ring slot (cudaHostAlloc, 2 x 16 MB, double-buffered) -> cudaMemcpyAsync ring ->
   slot on the copy stream. The ring is allocated by the cache object (NOT registered mmap pages),
   so the straddle landmine does not apply. Unit-test the ring standalone first (byte-compare a
   promoted slot, both ring slots cycling).
4. **Promotion budget:** cap promotions at N/token (start N=2-4, make it a knob). Unbounded
   promotion during warmup floods the ring and the copy stream; the LRU converges anyway via
   repeated routing. Record warmup length (tokens to steady-state hit rate) as a reported metric.
5. **Carry-forward gates (re-run, not assumed):** forced-empty byte gate and M5b unit oracle must
   still pass after the async swap; ppl-within-noise and hit-rate-matches-sim re-verified.
6. **New performance gates:** (a) nsys trace shows promotions on the copy stream and NO per-token
   host sync on the decode stream (the 36 cpy-sink ids syncs from phase 1 remain — they are the
   status-quo cost, but no NEW syncs); (b) **equal-VRAM verdict (finding 1)**: best cache config
   (ncmoe ~26-28 + slots) vs best static config (ncmoe 22) at equal total VRAM, tg, clean box,
   r>=5 — this is THE ship/no-ship number; (c) p99 inter-token latency not worse than static
   baseline; (d) warmup: time-to-steady-state reported.
7. **Pause points for the supervised run:** (i) after the staging ring + its unit test, before
   wiring into promotion; (ii) after first nsys trace, before the full bench matrix; (iii) final
   numbers before any BENCHMARKS.md/README claim.

### Phase 2 progress. Pause (i): staging ring DONE + unit-tested (experiments/slot-cache d0ba557e1).
Ring mechanism proven standalone (tests/test-slot-ring.cpp): host memcpy -> pinned 2x16 MiB
double-buffer (ggml_backend_dev_host_buffer_type) -> tensor_set_async on a dedicated copy backend
(ggml_backend_dev_init(dev) = separate stream) -> GPU slot, event-gated double-buffering. 6 experts,
0 byte-mismatches, both buffers cycled. Wiring architecture = mirror the model loader's async-upload
path (llama-model-loader.cpp:1483-1640), all portable via the registry.

**DECISION NEEDED before wiring (pause i): the non-blocking completion check.** ggml has
tensor_set_async, pinned buffers, a separate-stream backend, and blocking event_synchronize -
but NO non-blocking event query. The spec's token-boundary poll (point 1) wants cudaEventQuery.
- (A) Add `ggml_backend_event_query(event) -> bool` (CUDA: cudaEventQuery; other backends: a
  synchronize-based or always-true fallback). Small, general, matches the spec exactly. But a new
  ggml core API (new-pattern; AGENTS.md pause).
- (B) Lag-by-one-token + blocking synchronize: enqueue promotion at token t; at t+1's boundary,
  event_synchronize (by then the ~12 MiB copy is almost always already done, so it returns
  immediately - effectively non-blocking) then update the map. No new API, but a subtle "rarely
  blocks" caveat if a copy overruns a token.
Lean (A): cleaner, no caveat, and event_query is a generally useful primitive. Awaiting owner call.

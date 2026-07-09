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

### Phase 2 progress. Pause (ii): async promotion CORRECT, nsys taken. Two findings before the bench matrix.
Async promotion landed on experiments/slot-cache (1849fcd27). Correctness: ppl 458.9669 ==
baseline (byte-identical); all carry-forward unit oracles re-run PASS; token-agreement identical.

**nsys trace (bench-results/p2-trace.nsys-rep, generation window, ncmoe36 S=8):**
- Compute kernels on stream 15 (2125), cleanly separate from transfers -> promotions do NOT block
  the compute stream (gate 6a's core intent holds; no new decode-compute-stream sync observed).
- Promotions (pinned HtoD from the ring): 2049 ops / 1116 MiB on streams 14 (1821) + 16 (228),
  async, ring cycling under load.
- **DtoH DOMINATES: 162 GB Device-to-Host on stream 14 (90.7% of memcpy time)** - the ncmoe-36
  CPU-expert activation round-trips, inherent to offload, NOT the cache. Most promotions share
  stream 14 with this flood (imperfect copy-stream isolation).

**Findings / recommendations for the bench matrix (owner decision, pause ii):**
1. **Ring starvation is the blocker.** 2-buffer ring polled once/token -> ~2 promotions/token vs
   ~144 misses/token across 36 layers -> hit rate collapses to 2.7% (S=8) / 14.1% (S=32) vs sync's
   43.6% / 90.4%. The cache never warms, so it adds overhead (doubled matmul + cpy-sink + poll)
   without payoff -> tg 4.20 (S=8) barely above sync 3.49, both << baseline 19.40. FIX: many more
   ring buffers (e.g. 16-32) + higher budget so promotion keeps pace with misses. The spec's
   2x16 MiB was too conservative for a 36-layer model.
2. **ncmoe 36 is the wrong evaluation point** - it is DtoH-activation-bound (162 GB), so the cache
   can only help by cutting CPU compute via HIGH hit rate. The equal-VRAM verdict (finding 1)
   belongs at ncmoe ~26-28 where fewer layers round-trip. Also investigate the stream-14 promotion
   sharing (copy_backend should be fully dedicated).
Recommend: un-starve the ring (fix 1), then run the equal-VRAM matrix at ncmoe 26-28. Paused for
owner direction before the bench matrix.

### Phase 2 pause (iii): the ceiling is real - phase 2 does not ship as a tg win. (2026-07-08)
Ring resize un-starved promotion; correctness stayed byte-exact throughout. But tg did not move,
and the equal-VRAM matrix (partial - run stopped after the static half) makes the verdict clear.

**Ring fix worked (decode, ncmoe36):** hit rate S=32 14.1% -> **89.1%** (sync 90.4%), S=48 92.4%;
ppl 458.9669 == baseline (byte-exact at 89% real GPU hits); all 4 unit oracles re-pass.

**But tg is flat at the ceiling (ncmoe36):** S=32 1.51, S=48 1.05 - vs baseline-off 19.40, DESPITE
89-92% hit. Higher hit rate cannot help because every CPU layer still pays its full activation
DtoH round-trip + the two-path (GPU slot + CPU skip) combine - the copy is a graph edge, not a
data decision (nsys: 162 GB DtoH dominates, unchanged by hit rate).

**Equal-VRAM matrix, static (off) champion + references, tg128 r5 (clean box):**

| config | tg128 t/s |
|---|---|
| off ncmoe22 (static champion) | 30.50 +/- 7.08 |
| off ncmoe24 | 29.24 +/- 7.01 |
| off ncmoe26 | 26.98 +/- 5.93 |
| off ncmoe28 | 24.80 +/- 5.79 |
| **cache ncmoe36 S32 (async, 89% hit)** | **1.51** |
| **cache ncmoe36 S48 (async, 92% hit)** | **1.05** |

(Cache ncmoe 24/26/28 points not captured - matrix stopped after the static half. The two-path
overhead scales with CPU-layer count, so lower ncmoe is somewhat less bad, but starts ~13-20x below
static and cannot close the gap: the activation-edge cost per CPU layer is independent of hit rate.)

**SHIP VERDICT: NO.** Phase 2 is CORRECT (ppl byte-exact at 89% real GPU hits - the whole
mechanism, incl. async promotion, is verified end to end) but the option-(c) two-path design cannot
beat a static split on tg, because a CPU-resident layer pays the activation round-trip whether it
computes 4 experts or 0. Warmup ~36 tokens (budget 32/token, ~1152 slots) - as expected, not the issue.

**ESCALATION TRIGGER (was always the finding-1 ceiling): the custom op / conditional graph.** To win,
a fully-resident layer must skip the CPU op AND its activation DtoH entirely - i.e. option (a)'s
custom on-device gather+matmul, OR a conditional graph edge that elides the CPU path when a layer's
routed experts are all resident. That removes the graph-edge round-trip the two-path design cannot.
This is a phase-3 design question for human review, not a phase-2 tuning knob.

**Verified reusable assets (all byte-exact, carry forward): ggml_mul_mat_id_skip (+CUDA guard),
ggml_backend_event_query, slot buffers + async ring + get_rows indirection, GGML_MOE_SLOT_CACHE /
_SYNC / _RING / _BUDGET, GGML_MOE_CACHE_SIM. The correctness scaffolding for phase 3 is done.**

### Phase 2 diagnostics (2026-07-08): anomaly reconciled - NOT VRAM, NOT the async mechanism.
Owner caught that async-S32 (1.51 tg, 89% hit) < sync-S8 (3.49 tg, 44% hit) - inconsistent with a
purely hit-rate-independent activation wall. Three diagnostics (async, ncmoe36, 32-ring):

| S | tg128 | hit% | token-boundary host ms | peak VRAM (used) |
|---|---|---|---|---|
| 8  | 3.85 | 67.4 | 25.1 |  9.7 GB |
| 32 | 1.50 | 90.2 | 10.2 | 20.6 GB |
| 48 | 1.05 | 93.8 |  6.4 | 27.8 GB |

1. **VRAM oversubscription REFUTED.** Peak used 27.8 GB at S=48 (< 32 GB, no WDDM paging cliff). The
   slot pools fit; the big-S slowdown is not VRAM thrash.
2. **Async mechanism SOUND.** Apples-to-apples: async S=8 (3.85 tg, 67% hit) > sync S=8 (3.49, 44%).
   Same S, async wins - no cost bug in the ring/event machinery. (Owner's 1.51-vs-3.49 was S32-vs-S8.)
3. **Token-boundary host cost real but not the S-trend cause:** 25 ms (S8) -> 6 ms (S48), i.e. it
   scales with promotion volume (miss rate), INVERSELY with S. ~10% of the token at S8.
4. **The real cost: the two-path GPU orchestration, and it grows with hit rate.** Subtracting boundary,
   non-boundary token time balloons 235 -> 657 -> 946 ms as S grows. Higher hit -> more expert compute
   shifted onto the GPU slot path, whose per-layer cross-backend combine (ggml_add of the GPU-slot
   result and the CPU-skip result) + scheduling overhead costs MORE than the CPU compute it displaces
   at ncmoe 36. This is structural to option (c) (both paths run every layer, then combine), independent
   of VRAM. It is the ~4x, and higher hit rate makes it worse, not better.

**Reconciled ceiling:** option (c)'s per-layer two-path combine is the binding cost at high ncmoe, on
TOP of the hit-rate-independent activation DtoH. Phase 3's single-path custom op / conditional graph
(compute each expert once on the right backend, no per-layer combine, elide the CPU op + its DtoH when
a layer is fully resident) removes BOTH. Design phase 3 against this reconciled stack.

**Consistency flag:** static ncmoe22 champion read 30.5 tg here (r5, +/-7 variance) vs Tier-0's 41
(r8, pinned GGML_CUDA_REGISTER_HOST=1). This run set had NO pinning and high variance - re-run the
champion with the full flag set before any ship/no-ship table is published.

### Phase 2, deeper diagnostics (2026-07-08): slot kernel cleared; S-swing mechanism bounded.
Owner rejected the two-path-combine explanation (it is constant in S; the data scales with S) and
suspected the slot mul_mat_id falling off the fast kernel path as S grows. Tested directly.

**Isolated slot-matmul microbench (tests/test-slot-matmul-bench.cpp), MXFP4 [2880,2880,S+1], decode
(1 tok, 4 routed), CUDA:** FLAT ~18 us/op from S=8 (50 MB) to S=128 (560 MB) - 0.95-1.00x. The slot
kernel is S-INDEPENDENT. The MMQ-fallback/dequant hypothesis is REFUTED. The nsys "MMQ at S=48 not
S=8" was prefill caught in the slower run's profiling window (S=48 at 1.05 tg spends more wall-time
in prefill during the fixed --delay), not a decode dispatch change; the shared decode MMVQ kernel
only grew 1.3x per instance, nowhere near 4x.

**Reconciled cost stack (what the S-swing is NOT):** not VRAM (27.8 GB < 32 at S=48), not the async
machinery (async S8 3.85 > sync S8 3.49), not the slot kernel (flat 18 us), not the token-boundary
host cost (that is INVERSE in S: 25 -> 6 ms). The residual 235 -> 946 ms non-boundary swing is a
property of the two-path scheduler orchestration (GPU slot path + CPU skip path + per-layer
cross-backend combine) interacting with hit rate - i.e. MORE GPU-resident experts is somehow slower
at steady state, with the matmul itself proven cheap. Exact mechanism not positively isolated
(candidate: cross-backend combine/critical-path serialization that shifts as the GPU path carries
more of the 4 routed experts) - but it is INHERENT TO THE TWO-PATH DESIGN, which phase 3 removes.

**Bearing on phase 3 (the reason this mattered):** phase 3's custom op computes resident experts
against these same slot buffers. That substrate is now VERIFIED clean and S-independent, so phase 3
does NOT inherit a kernel pathology, and its projected win (single path, no per-layer combine, elide
the CPU op + its activation DtoH for fully-resident layers) rests on eliminating exactly the two-path
orchestration that carries the residual S-swing. The phase-3 doc can be written against this stack.

### Phase 2 - FULL COST DECOMPOSITION (2026-07-08). The residual is the cpy-sink, and it is fixable.
Toggle isolation (ncmoe36, tg128 r3, baseline 18.5) + full-residency sim + champion re-run.

| build mode | S=8 | S=48 | isolates |
|---|---|---|---|
| off (baseline) | 18.5 | 18.5 | - |
| NOBOOK (slot buffers allocated; no cpy-sink, no promotion) | 18.4 | 18.5 | allocation + VRAM = **FREE**, flat in S |
| NOPROMO (cpy-sink ON, promotion OFF) | 4.12 | 1.02 | promotion = nearly free |
| NOSLOT (cpy-sink + promotion, plain-CPU compute) | 3.83 | 1.03 | - |
| twopath (cpy-sink + promotion + GPU-slot + combine) | 3.77 | 1.03 | two-path compute + combine = **FREE** |

**The entire 18 -> 1 collapse is the CPY-SINK** - the 36 GPU->CPU routing-id copies (`ggml_cpy(selected_experts -> routed_cpu)`) inserted into the decode graph, one per CPU layer, which split the graph and serialize the pipeline. NOBOOK (no cpy-sink) == baseline; adding only the cpy-sink drops it to 4/1. Combine, two-path compute, promotion, and allocation are all FREE. This is an IMPLEMENTATION artifact, not a ceiling.

**Fix:** capture routed ids via the eval callback (the GGML_MOE_TRACE mechanism - reads the ids tensor post-op, NO graph node, NO cross-backend copy, NO serialization) instead of a graph cpy-sink. Both phase 2 and phase 3 need this; it is the single highest-value change.

**Full-residency (phase-3 conditional-elision candidates), P(all 4 routed resident):**
| S | wiki | code | chat |
|---|---|---|---|
| 32 | 54.7 | 55.4 | 78.7 |
| 48 | 75.7 | 78.8 | 87.6 |
| 64 | 88.7 | 90.0 | 93.3 |
(> naive hit^4 - within-layer routing is correlated, which helps phase 3.)

**Static champion (pinned, -b 4096 -ub 2048, r5):** ncmoe22 29.7, 24 29.1, 26 27.1, 28 25.6 tg
(+/-5-7). NOTE the 29.7-vs-Tier0-41 gap is GPU clock-warming (-p 0 here vs -p 2048 prefill in
Tier 0), not a regression - control -p in the go/no-go comparison.

**Reframed go/no-go (positive):** the combine phase-3 was feared to inherit is FREE. The killer
is the cpy-sink, which is fixable via eval-callback capture. With that fixed the cache runs at
~baseline; phase 3's conditional elision of fully-resident layers (76-88% at S=48) then removes the
per-CPU-layer activation DtoH for those layers - the remaining real ceiling - to beat static.
DECISIVE NEXT MEASUREMENT: implement eval-callback routing capture (replacing the cpy-sink) and
re-measure; if tg recovers toward baseline, phase 3 is viable; then scope the conditional graph.

### Phase 2 - the capture floor (2026-07-08). Go/no-go: NO for this architecture; phase-3 prerequisite found.
The cpy-sink was not a bad implementation of a good idea - per-token routing EXTRACTION is itself the
cost, and no placement escapes it.

**Capture-mechanism A/B (ncmoe36, tg128 S=8, baseline 18.4):**
| mechanism | tg | note |
|---|---|---|
| NOBOOK (no capture) | 18.4 | = baseline (capture is the only cost) |
| cpy-sink (cross-backend cpy -> CPU) | 3.7 | 36 graph splits |
| callback (eval-callback observe) | 5.0 | disables the sched fast path |
| GPU-sink (GPU->GPU cpy + 1 D2H) | 5.2 | dead-end output nodes; best, still ~4x down |

**Isolation (GPU-sink build):** NOBOOK 18.4 / NOPROMO (capture, no update) 5.47 / full 5.04. The floor
is the CAPTURE, not the update (update adds ~0.4 tg). Capture correctness confirmed: hit rate 66.5%
matches across all mechanisms (the capture-bug canary).

**Why:** extracting selected_experts per token forces it to be materialized as an extra graph output
(or observed via a callback that disables the fast path) 36x/token, which breaks the pipelining/graph
reuse the baseline relies on. The cpy-sink (approved (A) at M6) was blamed on callback-conflict grounds;
nobody priced that 36 extraction points = 36 materialization/sync points. Review caught the cost only
after paying it - a real lesson about what design review misses and measurement does not.

**GO/NO-GO: NO for the current architecture.** No capture mechanism recovers to baseline, so the cache
cannot beat the static champion (29.7 tg, warm-clock corrected) at ncmoe36. Two-path compute, combine,
promotion, allocation, and the slot kernel are all FREE (measured); routing capture is the whole cost.

**Phase-3 prerequisite (the real design input):** the CPU op `mul_mat_id_skip` ALREADY has
selected_experts materialized on host (the scheduler copies it so the CPU op can run - that read is
free). Capture must REUSE that, e.g. the op writes the routed ids to a cache buffer as a side effect,
instead of a separate extraction node/callback. Fuse capture into existing work; do not add work to
extract what is already there. Until capture is free, phase 3 is not viable - and this is now measured,
not assumed.

**Also credited (traces keep paying):** full-residency 76-88% at S=48 beats the 72% independence
estimate (within-layer routing correlation); champion corrected 41 -> 29.7 tg (warm-clock -p 2048 vs
-p 0), another measurement-conditions artifact caught by reconciling drift.

## Phase 3 scope (Fable, 2026-07-08) — fused capture, then conditional elision

The phase-2 verdict killed extraction-based capture, not the cache. Phase 3 is two measured
steps, each gated, built on the verified-free assets (skip op, slot buffers, get_rows
indirection, event_query, ring, all unit oracles).

**P3.0 — fused capture (prerequisite, contained).** mul_mat_id_skip's CPU forward already holds
selected_experts in host memory as a paid-for src (proven baseline-cost by NOBOOK). Have the op
write the ids to a cache-owned host side-buffer during forward — no graph node, no callback, no
extra materialization, no new sync. Threading note: op instances write disjoint per-layer slots;
token boundary reads after graph completion — race-free by construction, but assert layer-slot
disjointness in debug. *Gates:* (a) full correctness chain re-run (ppl-exact, forced-empty,
unit oracles, hit-rate-matches-sim — the ppl chain was NOT run on the dead capture mechanisms,
so it is owed here); (b) tg ≈ NOBOOK within 5% at ncmoe36 S=8 AND S=48 (capture is free or this
scope stops).

**P3.1 — conditional elision (the payoff).** Per-token graph construction chooses each CPU
layer's path from the residency map published at the last token boundary: fully-resident layer
→ GPU-only (slot mul_mat_id via get_rows; NO CPU op, NO activation DtoH/HtoD round-trip);
any-miss layer → the existing two-path. llama.cpp already rebuilds the graph per ubatch, so
per-token topology is legal; the known cost is loss of graph *reuse*. *Measure first:* the
reuse-loss overhead alone (toggle topology every token with no cache, diff vs stable topology)
— if reuse loss exceeds the round-trip savings this design is dead before implementation.
*Then gates:* (a) nsys shows zero CPU op and zero activation round-trip on fully-resident
layers; (b) correctness chain again (elision is a correctness-critical graph change — a layer
elided while a routed expert is non-resident must be impossible by construction: elide only
from the boundary-published map, never mid-token state); (c) tg > NOBOOK baseline.

**P3.2 — the verdict.** Equal-VRAM vs the corrected static champion (29.7 tg, ncmoe 22, pinned,
cold-clock protocol). Cache config: ncmoe 36 + S=48 (~28 GB) is already VRAM-matched to the
champion (~29 GB). Ship bar: **beat 29.7 by ≥10% (≥32.7 tg)** with p99 inter-token latency not
worse and warmup reported. Rough ceiling from measured terms (stated, not promised): baseline
18.4 at ncmoe36 pays 36 round-trips; 76-88% of layers fully resident at S=48 → elision removes
most of that cost → mid-30s plausible IF round-trips dominate the ncmoe36/ncmoe22 gap; P3.1's
reuse-loss measurement is the term that decides it.

**Stop conditions:** P3.0 gate (b) fails → capture is fundamentally expensive even fused; park
the cache, publish the observer-effect finding. P3.1 reuse-loss measurement exceeds projected
elision savings → park before implementation. Any correctness gate fails → stop per standing
condition 3.

### P3.0 DONE (2026-07-09), `experiments/slot-cache` (d24a8320c). Capture is FREE; the wall is the two-path split — NOT capture.
Fused capture: `ggml_mul_mat_id_skip` gained an optional src[4]; its CPU forward copies the routed
ids (already in host memory as src[2]) into the cache's host side-buffer (`ls->routed`) during the
matmul it already runs — no extra graph node, callback, or sync. Only the gate op captures (up/down
route identically). `capture_mode 3` is now the default. New `GGML_MOE_SLOT_NOCAP` builds the
two-path but disables capture+update, to isolate the two-path structural cost from the capture —
**the cell the phase-2 A/B never measured.**

**Gate (a) — full correctness chain, ALL PASS (exact):**
- ppl-exact: **458.9669** OFF == ON (full active, 97.7% GPU residency) == forced-empty (NOPROMO).
  Byte-identical over 11 chunks, n_ctx=512, wikichunk. Off-switch re-verified byte-identical.
- unit oracles 4/4: test-mul-mat-id-skip (incl. new gate (e): fused capture copies ids for every
  token INDEPENDENT of the skip mask), test-slot-matmul (M5b byte-identical), test-slot-ring,
  test-event-query.
- hit-rate canary: **66.5% (S=8) / 92.4% (S=48)** — EXACT match to cpy-sink/callback/GPU-sink and
  the phase-1/2 sim. Fused capture extracts identical routing → capture is correct.

**Gate (b) — tg ≈ NOBOOK within 5%: FAIL. But the decomposition flips the verdict.**

| ncmoe36 tg128 (r3, -p 0) | S=8 | S=48 | isolates |
|---|---|---|---|
| NOBOOK (plain path, no two-path, no capture) | 19.15 ± 3.33 | 18.40 ± 3.87 | baseline (flat in S) |
| NOCAP (two-path built, **capture OFF**, update OFF) | 5.59 ± 0.41 | 1.48 ± 0.03 | two-path split ALONE |
| fused (two-path + fused capture + update) | 5.16 ± 0.31 | 1.52 ± 0.04 | + capture + promotion |

**Capture was never the cost.** NOCAP (capture fully disabled) already collapses tg to 5.59/1.48;
adding fused capture + update on top is free (5.16/1.52, within noise). P3.0's objective — cost-free
capture — is ACHIEVED. The phase-2 verdict ("per-token routing EXTRACTION is itself the cost") was a
**misattribution**: every capture mechanism in that A/B was measured on top of the two-path, which had
already collapsed tg to ~5 before any capture was added. The one point that seemed to isolate capture
(NOSLOT 3.83 vs NOBOOK 18.4) used the PLAIN compute path + cpy-sink — a *different* graph split. There
are two independent split costs; the two-path structure is the one the cache actually incurs.

**The wall is the two-path graph structure:** each cached CPU layer becomes GPU-slot `mul_mat_id` +
CPU-skip `mul_mat_id_skip` + GPU `add`. The CPU op sandwiched between GPU ops is a split point, 36×/
token, serializing the pipeline. It WORSENS with S (bigger slot `mul_mat_id`): S=48 is 1.5, S=8 is 5.6.

**Verdict: do NOT park.** The doc's stop condition ("gate (b) fails → capture fundamentally expensive
→ park") rests on a premise the NOCAP measurement disproves. Capture is free; the isolated, quantified
wall (19→5.6 at S=8, 18.4→1.5 at S=48) is **exactly what P3.1 conditional elision removes** — a
fully-resident layer goes GPU-only (no CPU op, no split). P3.1 is now the decisive test, and its target
cost is cleanly measured rather than assumed. **STOP for review before P3.1 per standing conditions.**

## P3.1 amendment (Fable, 2026-07-09) — binary dispatch; two-path retired everywhere

P3.0's NOCAP cell changes the partial-layer economics: a two-path layer costs ~3.5 ms (S=8) to
~17 ms (S=48); with 5-8 any-miss layers/token the original P3.1 (elide full, two-path partial)
loses to the champion even working perfectly. Amendments:

1. **Binary per-layer dispatch.** Fully-resident (per boundary-published map) -> GPU-only slot
   path, CPU op and activation round-trip elided. Any-miss -> the PLAIN CPU path: all 4 experts
   on CPU (skip-op with all-compute mask, so fused capture rides along), no slot matmul, no
   combine, no split. A plain CPU layer is ~1.4 ms vs the split's 3.5-17 ms; sacrificing GPU
   compute for partial layers' hits is a clear win. The two-path structure retires from the
   design entirely.
2. **Elided-layer capture gap + fix candidate.** Elided layers run no CPU op -> ids never reach
   host -> LRU blind exactly where the cache succeeds. Candidate: batched same-backend GPU-sink
   for elided layers only, one D2H post-compute. NOTE: the GPU-sink's 5.2 tg verdict was
   measured ON TOP of the two-path (contaminated, like every phase-2 capture number); its clean
   cost is unknown.
3. **Pre-measurements (before any build):**
   (a) NOBOOK + batched GPU-sink, capture-only — the uncontaminated cell. tg ~= NOBOOK ->
       elided capture solved; collapse -> phase 3 needs capture-free residency tracking, stop
       and re-scope.
   (b) Graph-reuse loss (unchanged): per-token topology toggle vs stable, no cache.
   The split-cost-vs-K curve is moot under binary dispatch (no splits remain).
4. **Cost model (measured terms + the two unknowns):** ~4 ms GPU base + (5-8 partial layers x
   ~1.4 ms) + reuse tax (3b) + capture (3a) ~= 13-16 ms/token -> 60-75 tg naive ceiling; sober
   band 35-50. Ship bar unchanged: >= 32.7 tg equal-VRAM, p99 not worse, warmup reported.
5. **Process note (third misattribution, for the report):** phase 2's capture A/B lacked the
   all-off control (two-path with NO capture); every mechanism verdict inherited the two-path
   collapse. Standing rule: an A/B over mechanisms is uninterpretable without the
   everything-off cell measured on the same substrate.

### P3.1 pre-measurements DONE (2026-07-09), `experiments/slot-cache` (d24a8320c). Both GO. No implementation.
Measured with existing toggles only — no new code. Logs in scratchpad (p3a-*, p3b-*).

**3(a) — uncontaminated elided-layer capture cost.** Batched GPU-sink capture on the PLAIN
substrate (no two-path): `GGML_MOE_SLOT_CACHE=8 NOSLOT=1 GPUSINK=1 NOPROMO=1` (plain compute path,
GPU-sink cpy nodes built, capture-only). tg128 ncmoe36 = **18.02 ± 4.62 ≈ NOBOOK 19.15** — capture
is FREE once off the two-path. Capture-fired confirmation (same config, update ON): online hit rate
**64.7%** — sensible/non-degenerate, proving the GPU-sink nodes wrote correct routing ids on the
clean substrate (a zeroed routed_gpu would give a degenerate rate). The phase-2 "GPU-sink 5.2 tg"
was entirely two-path contamination. **Verdict: elided-layer capture SOLVED** (amendment rule:
tg ≈ NOBOOK → solved). Caveat: the one boundary D2H is excluded by NOPROMO, but it is a single
post-compute copy/token, not a pipeline-breaking in-graph node — the in-graph cpy nodes are what
3(a) measures and they are free.

**3(b) — graph-reuse loss.** `LLAMA_GRAPH_REUSE_DISABLE=1` (per-token rebuild) vs default (reuse),
no cache. Reuse mechanism confirmed active and the knob fires: llama-completion `graphs reused`
**126/127 → 0**. Magnitude is BELOW the box noise floor in every regime measured:

| tg128 (no cache) | reuse ON | reuse OFF | Δ (ON−OFF) |
|---|---|---|---|
| ncmoe36 bench -r3 | 18.63 ± 5.35 | 18.06 ± 4.41 | +0.57 |
| ncmoe36 bench -r10 | 23.21 ± 3.63 | 21.88 ± 3.48 | +1.33 (ON faster) |
| ncmoe36 completion ms/tok | 72.97 | 62.73 | OFF faster |
| ncmoe24 bench -r10 | 31.55 ± 5.21 | 31.00 ± 5.29 | +0.55 (ON faster) |

The ON/OFF delta is smaller than the error bars everywhere, its **sign is inconsistent** across
tools/configs, and same-config mean drift (ncmoe36 ON 18.6 → 23.2 across runs) exceeds any delta.
The box's per-token variance (±16%, intrinsic to the 162 GB/token expert DtoH) swamps the fixed
graph-rebuild host cost. **Bound: reuse tax ≲ 0.5–1.3 tg (~0.5–2.7 ms/token), sign indeterminate —
a small fixed host cost, far below P3.1's projected elision savings (tens of ms/token).** A precise
figure would need in-process instrumentation (out of scope for a pre-measurement). **Verdict: reuse
loss does NOT exceed projected elision savings → GO** (amendment stop condition not triggered).

**Both pre-measurements GO.** The two unknowns in the amendment's cost model are now bounded:
elided capture is free (3a), reuse tax is negligible (3b). Neither stop condition fires. Awaiting
review before any P3.1 implementation.

### P3.1 mispredict analysis — GATE FAILED (2026-07-09, overnight). STOP: binary dispatch is not viable. Not built.
Jordan's overnight decision made speculative elision conditional on an offline mispredict analysis
BEFORE any implementation, with a hard gate: *stop if mean dropped weight-mass > ~1% OR the
top-expert-escape rate isn't rare.* Tooling: `GGML_MOE_TRACEW` weighted routing trace (ids + final
normalized router weights) + `analyze-mispredict.py` (per-token LRU, `experiments/slot-cache`
ae7b42c0d). Artifact: `bench-results/p31-mispredict-analysis.txt`.

An elided (GPU-only) layer takes an **escape** when this token routes to an expert the boundary map
didn't have resident; that expert's contribution is dropped (zero-fill) or renormalized away. The
weighted trace prices the damage.

| S=48 (ship target) | WIKI (ppl workload, 883,152 reqs) | CHAT (best case, lowest churn) | gate |
|---|---|---|---|
| per-layer escape \| elided | 24.74% | 5.71% | — |
| per-token escape (any layer) | **99.49%** | 29.20% | — |
| **mean dropped weight-mass** | **7.20%** | **1.67%** | **>1% → FAIL both** |
| **top-1-expert escape rate** | **21.0%** | **19.4%** | **not rare → FAIL both** |

Cross-validated: wiki per-layer escape 24.74% == the doc's own full-residency 75.7% complement; the
trace is the exact 883,152-request wiki workload; captured weights sum to 1.000.

**Why it fails, structurally:** 76% full-residency is a *per-layer* figure; across 36 layers it
compounds to ~100% per-token escape (wiki). The top-expert-escape rate (~19–21%) is
**domain-invariant** — an escape lands on the highest-weight expert ~1 in 5 times regardless of
workload, so renormalization cannot rescue it (renorm only helps when dropped experts are low-mass;
here they routinely carry 0.3–0.5 weight, mean escaped rank ~1.8 of 4). Even S=64 (2.97% dropped,
18% top-1, and it exceeds the VRAM budget) fails the gate. **Both gate criteria fail in the
best-case domain**, so the failure is not wiki-specific.

**Both of the amendment's escape routes are closed by the same numbers:**
- *Speculative + renorm (the chosen design):* mean dropped mass 7.2% (wiki) / 1.67% (chat), top-1
  escape ~20% → quality perturbation on ~every token, far outside ppl-within-noise.
- *Capture-then-replay (the priced fallback):* cost = elided + P(escape)·plain. P(escape) = 99.5%
  (wiki) → replay ~always → strictly **worse than plain**. Viable only on low-churn chat-average
  (29%), but its high-churn decile spikes to 94% and it still fails the dropped-mass gate. Jordan's
  "if per-token escape is low" condition is not met on the workloads that matter.

**VERDICT: STOP — do not implement P3.1.** The cache's achievable residency at the VRAM budget
(76% per-layer at S=48) is fundamentally too low for token-level elision to be either exact-cheap
(replay ~always) or speculatively-acceptable (drops a top expert on ~every token). Exact per-layer
miss handling = the two-path = the split P3.0 proved slow. So the slot cache cannot beat the static
champion by any elision route, confirming and completing the phase-2 NO at the design level.

**Publishable results stand (the campaign's actual yield):** (1) the observer-effect / two-path
misattribution finding (P3.0 NOCAP); (2) free fused + free GPU-sink capture once off the two-path
(P3.0, 3a); (3) this per-token-escape compounding law that kills elision at realistic MoE residency.
Nothing written to BENCHMARKS.md / TECH-REPORT.md — awaiting Jordan's review.

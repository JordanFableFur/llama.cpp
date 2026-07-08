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

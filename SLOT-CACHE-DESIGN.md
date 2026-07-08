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

1. **Slot pool (per CPU layer):** a GPU buffer holding `S` expert triples (gate/up/down) +
   a host-side `expert_to_slot[128]` map (-1 = not resident) and an LRU recency list.
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
  misses on CPU, promote synchronously (simplest, correct). *Gates:* temp-0 seed-fixed output
  byte-identical to baseline; tg >= 41 (ncmoe 22 clean baseline) - i.e. at least not a
  regression even before async.
- **Phase 2 - async promote (compute-now/promote-later).** Move promotion to a copy stream;
  keep the hit path sync-free. *Gates:* tg improves over phase 1; nsys confirms promotions are
  on the copy stream and the decode stream has no per-token host sync; p99 inter-token latency
  not worse.
- **Phase 3 (optional) - predictor prefetch.** 2-layer MLP (2511.10676) predicting next-token
  top-4 from the current hidden state, prefetch during attention. *Gate:* only if phase 2
  leaves a measured hit-rate gap and the tg gain exceeds the added complexity (headroom is only
  4-9 pts per item 1) - otherwise ship phase 2.

## Open questions for review

- Can `mul_mat_id` cleanly run twice per layer (GPU-hits subset + CPU-misses subset) and
  combine, or is a fused custom op needed? This determines phase-1 feasibility.
- Slot buffer layout: one buffer of `S` triples per layer, or three (gate/up/down) - which
  matches `mul_mat_id`'s expected `_exps` tensor stride?
- Does promoting on a copy stream contend with the existing prefetch/upload path enough to
  matter at these transfer sizes (~12 MB/expert, a few misses/token)?
- VRAM: fixed slots vs `ncmoe` is a joint optimization - is a small autotune (pick slots to
  fill free VRAM after static placement) worth it?

## Baseline to beat

41 tg (ncmoe 22, clean box). **Hit rate is necessary but not sufficient** - the moe-cache
regression proves a high-hit-rate cache can still lose to sync overhead. Measure tg at each
gate; never project it from hit rate.

# Handoff - next unattended run (slot cache, item 15)

Green-lit: phase 0 and phase 1 unattended. Phase 2 is supervised (needs a human in the loop).

## Prompt

> Read SLOT-CACHE-DESIGN.md including the review section; implement phase 0, then phase 1,
> honoring every gate; stop at phase 2.
>
> - Branch off ai-main as `experiments/slot-cache`. Reuse: the `experiments/routing-trace`
>   branch has the `GGML_MOE_TRACE` hook + `simulate-cache.py` (base for the phase-0 hit/miss
>   counter); the `experiments/moe-cache` branch (9cf4f1a9f) has cache telemetry (dcc2cb143)
>   and prior scaffolding.
> - Phase 0's gate is a HARD STOP: if the online (in-engine) hit rate does not match the
>   offline sim within +/-5% on the same workload, halt and write up the discrepancy - do NOT
>   build the cache on a broken reuse model.
> - Honor CLAUDE.md benchmarking rules (one llama process at a time, clean box, redirect long
>   runs to files, no benching during builds) and AGENTS.md commit standards
>   (Generated-by: trailer, experiments/* branches, never touch master/upstream).

## Status (updated 2026-07-08)

- **Phase 0: DONE, gate PASSED.** In-engine `GGML_MOE_CACHE_SIM` counter on branch
  `experiments/slot-cache` reproduces `simulate-cache.py` exactly (online == offline at all
  slot budgets). See SLOT-CACHE-DESIGN.md "Phase status".
- **Phase 1: re-scoped, NOT started.** Reading the code showed phase 1 is bigger than assumed:
  the hit/miss split hits a static-graph wall (needs a custom on-device op, not two static
  mul_mat_ids), and the moe-cache branch is on an incompatible ancient base (not reusable).
  Phase 1 is greenfield CUDA. See SLOT-CACHE-DESIGN.md "Phase 1 - re-scoped" for the exact
  interception point and the custom-op vs host-sync options.

**Revised prompt for the phase-1 run** (branch `experiments/slot-cache` already exists with
phase 0):

> Read SLOT-CACHE-DESIGN.md (esp. "Phase status" and review finding 4). Phase 0 is done on
> experiments/slot-cache. Implement phase 1 there: the custom on-device gather+matmul op
> (option a), per-layer LRU slot buffers for CPU layers only, synchronous promotion. Honor the
> phase-1 gates (temp-0 byte-identical output; online hit rate matches simulate-cache.py +/-5%;
> tg >= ~35). Stop at phase 2. The custom-op shape is a new pattern - if a human is reachable,
> get a design check before writing the CUDA; otherwise prototype the op behind an env flag and
> keep the default path untouched.

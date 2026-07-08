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

## Status
- See the "Phase 0" section appended below once it is run; if present, start the next run at
  phase 1.

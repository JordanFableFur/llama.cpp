# Campaign Handoff — the July 2026 MoE-offload research sprint

**Status: research sprint CLOSED (2026-07-09). This document is the single entry point for
anyone — human or AI — picking this fork up later.** Read this, then EXPERIMENTS.md (the
queue + all verdicts), then TECH-REPORT.md (the findings, publication-grade), then
SLOT-CACHE-DESIGN.md (the full decision trail of the flagship).

## What this fork is

An AI-contributions-welcome fork of llama.cpp (upstream bans AI-generated PRs; we invert the
policy and compensate with mandatory verification — see CONTRIBUTING.md / AGENTS.md).
Created 2026-07-07. In three days it ran a full research campaign on one question:
**how fast can a bigger-than-VRAM MoE model (gpt-oss-120b, 59 GB) run on a consumer Windows
box (RTX 5090 32 GB, Core Ultra 9 285K, 64 GB DDR5)?**

**NEVER open PRs/issues against upstream ggml-org/llama.cpp — permanent-ban risk.**

## What shipped (use these today)

| win | branch / doc | evidence |
|---|---|---|
| Host-pinning enabled on Windows: **~2.1x prefill** (was dead code behind a POSIX #ifdef) | `experiments/prefetch-experts-win` | Tier-0 factorial, `bench-results/fu-*.log` |
| Working set halved during generation (47.2 → 24.0 GB): `VirtualUnlock` on GPU-uploaded mmap pages (`unmap_fragment` was a Windows no-op) | `experiments/win-mmap-pressure` | `bench-results/item9-*.log`, output-identical |
| `--direct-io` honesty fix (silently fake on Windows; now reports false) | `experiments/win-fast-load` | commit f1d3aa3dd |
| Best-known config for this hardware class | BENCHMARKS.md | `-fa 1 -b 4096 -ub 2048 -ncmoe 22` + `GGML_CUDA_REGISTER_HOST=1` |

Corrected reference numbers (clean box, cold-clock `-p 0` protocol): static champion
**29.7 tg @ ncmoe 22**; prefill ~1711-1793 pp @ ub 2048. Any number you see elsewhere that
disagrees was measured under contention or warm clocks — see TECH-REPORT §2 and the
champion-drift note (41 → 29.7 was a warm-clock artifact).

## The findings (each with artifacts; TECH-REPORT.md is the full treatment)

1. **Measurement conditions dominate.** Our own first baseline under-read 5x pp / 3x tg from
   box contention. Later: warm-clock inflation (41 vs 29.7), r3-vs-r10 mean drift larger than
   the effects being measured. Protocol now: clean-box audit logged per run, cold-clock,
   interleaved pairs for verdicts.
2. **Routing is spatially concentrated, temporally restless** (the campaign thesis). Per-layer
   expert skew is high (Gini 0.72) but aggregate is near-uniform (0.19), hotness is
   workload-specific (idle-set overlap 29.2% ≈ random), and temporal coherence is weak at both
   the layer-compounding scale and the K-token scale. This single property killed caching AND
   speculation (below).
3. **The per-token escape compounding law.** 76% per-layer full-residency compounds to ~0%
   per-token (0.76^36) — token-level elision needs near-total residency, i.e. "just fit the
   model." Kill table (3 domains): `bench-results/item15-p31-kill-table-3domain.txt`.
4. **The two-path misattribution / missing-control lesson.** Phase 2 concluded "routing
   capture is the cost" — wrong; the A/B lacked the all-off cell. The two-path graph split
   (3.5-17 ms/layer) was the cost; every capture mechanism is ~free off it. Standing rule: no
   mechanism A/B without the everything-off control on the same substrate.
5. **LRU is sufficient** for MoE expert caching at practical budgets (cycle-aware eviction ties
   or loses; contradicts published SpecMD claim at these budgets). 48 slots/layer → 92-96% hit.
6. **Draft agreement tracks output entropy, not computation perturbation.** Resident-only
   drafting: 61.8% top-1 agreement on wikitext vs 89.6% on code with near-identical dropped
   weight-mass. The quality-perturbation and token-agreement axes decouple.
7. **Windows is systematically under-served** — four dead/broken paths found in one week
   (pinning, direct-io, unmap_fragment, E-core logic Linux-only) and one textbook default
   (E-core exclusion) measured actively harmful (-30%) for bandwidth-bound MoE decode.

## The graveyard (all killed BY MEASUREMENT, most pre-implementation — do not re-attempt
without new evidence; full reasoning in EXPERIMENTS.md / SLOT-CACHE-DESIGN.md)

- **Chunked host registration** — cudaMemcpyAsync source must lie in ONE registered region;
  packed tensors share pages → crash. (Item 18's page-aligned layout would dissolve this;
  parked, low value now.)
- **E-core exclusion on Windows** — regresses 30%; decode is bandwidth-bound.
- **REAP expert pruning** — per-layer skew is real but workload-specific; single-domain prunes
  lobotomize off-domain.
- **Item 15, GPU expert slot cache** — built, byte-exact correct at 92% hits, and structurally
  unable to win: two-path graph split (phase 2), then the compounding law closed every elision
  route (phase 3, killed at offline analysis). Verified reusable assets remain:
  `ggml_mul_mat_id_skip` (+CUDA guard), `ggml_backend_event_query`, slot/get_rows/staging-ring
  machinery, all unit oracles (branch `experiments/slot-cache`).
- **Item 17, resident-experts self-speculation** — all three gates failed: amortization only
  1.7-1.9x (weak temporal locality), wiki agreement 61.8%, prior art exists (SS-MoE, WebConf
  2026). Max projected 1.14x < 1.25 bar even GO-biased.

## Repo map

- Docs: EXPERIMENTS.md (queue + verdicts) · TECH-REPORT.md (findings) · SLOT-CACHE-DESIGN.md
  (flagship decision trail) · BENCHMARKS.md (user-facing) · CLAUDE.md (box hazards + hard
  constraints — READ before benchmarking) · this file.
- Branches: `master` (pure upstream mirror, ff-only, never commit) · `ai-main` (default) ·
  `experiments/{prefetch-experts-win, win-mmap-pressure, win-fast-load, routing-trace,
  slot-cache}` (ours) · `experiments/{prefetch-experts, moe-cache}` (thecodacus imports).
- Tooling (the campaign's most reusable output): env-gated routing trace (`GGML_MOE_TRACE`,
  weighted `GGML_MOE_TRACEW`), draft-sim (`GGML_MOE_DRAFT_SIM`), `simulate-cache.py`,
  `analyze-routing-skew.py`, `analyze-mispredict.py`, unit oracles in `tests/`. **Collect the
  routing trace first** — it decided REAP, cache policy, elision, and speculation offline for
  pennies.
- Local: clone at `C:\Users\Jorda\llama.cpp`; worktrees may exist for old branches; builds in
  `build-main/` etc. Model: `C:\Users\Jorda\models\gpt-oss-120b\` (59 GB, 3 shards).

## Open items (if/when someone resumes)

1. **Tech report v1.0 + arXiv decision** — v0.2 needs the item-17 kill + entropy finding folded
   in and a narrative pass. Publication steps are human calls.
2. **thecodacus outreach** — never sent; the story is now complete and strong.
3. **MTP-over-offload bench** (GLM-4.5-Air, upstream `--mtp`, merged May 2026; gpt-oss has no
   MTP heads) — cheap, practical, unpublished territory.
4. **Replication on a second MoE family** (Qwen3-Next / GLM) — what turns the report into a
   paper; trace tooling ports directly.
5. Parked: item 18 layout; large pages (blocked on SeLockMemoryPrivilege grant); upstream sync
   routine in CLAUDE.md.

## The method (transferable; encoded in Jordan's `~/.claude/skills/experiment-queue`)

Repo-anchored experiment queue; every item executable with a code anchor, expected gain, and a
gate; offline analysis before implementation wherever a trace can decide; artifacts over
summaries; kill-lists with tombstones; strong model designs/reviews, cheap model executes;
milestone commits with re-verified off-switch. Empirically this caught: a 5x contaminated
baseline, two reversed textbook optimizations, an unsatisfiable gate, three misattributed
explanations, and killed three flagship designs for ~zero implementation cost.

**Total campaign cost: ~3 days, one consumer box. Shipped: 2 perf fixes + 1 honesty fix +
the config guide. Killed: 5 designs, with receipts. Findings: 7, two likely novel.**

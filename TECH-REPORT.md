# Running a 120B Mixture-of-Experts Model on a Consumer Windows PC: Measurements, Dead Code, and the Anatomy of an Expert Cache

**Tech report v1.0 — 2026-07-09.** Status: single model, single machine; consolidated. Every
number cites a raw artifact committed to this repository (`bench-results/`, branch names inline).
Authored by AI agents (Claude Fable 5, Claude Opus 4.8) under the verification-over-authorship
policy of this fork ([CONTRIBUTING.md](CONTRIBUTING.md)); all claims are gated on committed logs.

## Abstract

We study single-user inference of gpt-oss-120b (MXFP4, 59 GB, 128 experts/layer, top-4 routing)
on a consumer workstation (RTX 5090 32 GB, Core Ultra 9 285K, 64 GB DDR5, Windows 11), where the
model exceeds VRAM and expert weights are partially CPU-resident. Findings: (1) benchmark
contamination dominates published-style numbers — our own initial baseline under-read by ~5x
prefill / ~3x generation purely from box contention; (2) on a clean box, three configuration
levers (host memory pinning, micro-batch size, offload split) multiply prefill ~7.6x
(224 → 1711 t/s) and generation ~1.1x (37.9 → 41 t/s) with zero new algorithmic work — but one
of the three (pinning) was dead code on Windows, gated behind a POSIX-only guard; (3) per-token
expert-routing traces show high per-layer skew (Gini 0.72) with near-uniform aggregate (0.19)
and workload-specific hotness (cross-domain idle-expert overlap 29.2% vs 25% random floor) —
killing static expert pruning for general-purpose models while quantifying exactly the regime
where a dynamic cache wins; (4) cache-policy simulation over these traces finds plain LRU within
noise of, or better than, layer-cycle-aware eviction at practical budgets, contradicting a
published claim, and 48 slots/layer (37.5% of experts) suffices for 92–96% hit rates across
workloads; (5) a slot cache implemented on these findings is **byte-exact correct at 92% real
GPU hit rates but slower than no cache at all** — profiling attributes the loss not to the
matmul (S-independent, ~18 µs), promotion machinery (async, verified overlap), or VRAM, but to
per-layer two-path orchestration in the graph scheduler, a cost that *rises* with hit rate — later
positively isolated (§5.1: two-path alone drops tg 18.4→1.5 at S=48, capture is free) and, since
routing escapes a fully-resident-layer prediction on ~100% of tokens at feasible VRAM budgets,
token-level elision is killed at the design gate and the cache parked; (6) the same routing
geometry defeats self-speculation (draft = the model with routing restricted to VRAM-resident
experts): a batched verify amortizes expert reads only 1.7–1.9x (a K=8 window still touches ~18 of
32 possible experts/layer), and acceptance is too low to cover it — projected speedup ≤1.14x in all
three domains against a 1.25x bar, and the mechanism is anyway prior art (SS-MoE). We release the
trace tooling, cache simulator, the draft-agreement diagnostic, all raw logs, and the verified
primitives.

**Thesis.** gpt-oss-120b's expert routing is *spatially concentrated but temporally restless* —
few experts dominate each layer (per-layer Gini 0.72), yet the hot set churns token-to-token so
completely that a 76%-per-layer-resident cache still misses on ~100% of tokens somewhere in 36
layers. Concentration is what makes caching and speculation look promising (high achievable hit
rate, high per-layer skew); restlessness is what defeats both at a consumer VRAM budget. Every
technique that tries to convert residency into a decode speedup — exact elision, speculative
elision, capture-then-replay, resident-only self-drafting — founders on the same fact: the residual
misses land on high-weight experts on essentially every token, and there is no temporal stability in
the working set to amortize against. Hit rate is necessary and, on this class of hardware, decisively
insufficient.

## 1. Setup

| | |
|---|---|
| Model | gpt-oss-120b, MXFP4 GGUF (59.02 GiB), 36 MoE layers, 128 experts, top-4 |
| GPU | RTX 5090, 32 GB, PCIe 5.0 x16, driver 596.21, CUDA 13.2 |
| CPU/RAM | Core Ultra 9 285K (8P+16E), 64 GB DDR5 |
| OS | Windows 11 Pro 26200, WDDM |
| Software | this fork @ ai-main (upstream-synced), VS2022 + CUDA 13.2 builds |

Offload: `--n-cpu-moe N` keeps N layers' expert tensors CPU-resident (computed on CPU at
decode; streamed to GPU at prefill). Measurement protocol: llama-bench, r≥3 (r=8 for definitive
tables), clean idle box verified by process/VRAM audit before each run, one inference process at
a time, outputs to files. Raw logs: `bench-results/tier0-*.log`, `def-*-r8.log`.

## 2. Finding: measurement conditions dominate (a cautionary artifact)

Our first recorded baseline — 46.5 t/s prefill / 12.0 t/s generation (pp2048/tg128, ncmoe 24) —
was reproduced byte-for-byte-identical in build and command on a clean box at **224.3 / 37.9**.
The original was measured while a second llama process and a GPU-resident app contended the box.
Every relative claim survived (direction), no absolute claim did (magnitude, up to 5x). Community
benchmarks of MoE offload rarely state box conditions; we consider this the single largest source
of unreliable numbers in circulation, including — until the redo — ours.

## 3. Finding: three flags and a resurrected feature (Tier-0 factorial, r=8)

Isolation factorial at ncmoe 24, ub 512 (`def-1-ubknee-r8.log`, `fu-*.log`):

| config | pp2048 | tg128 |
|---|---|---|
| baseline | 224.3 ± 30.7 | 37.9 ± 1.0 |
| + flash-attn | 253.8 ± 15.1 | 37.1 ± 1.5 |
| + host pinning (`GGML_CUDA_REGISTER_HOST=1`) | 494.0 ± 23.6 (fa0) / 566.5 ± 4.0 (fa1) | ~unchanged |
| + ub 2048 (`-b 4096`) | **1793 ± 83** | ~unchanged |
| ncmoe 22 (fits just under VRAM cliff) | 1711 ± 192 | **41.1 ± 0.7** |

- **Host pinning is ~2.1x prefill** — and was a no-op on Windows: the mmap registration path
  was guarded `#ifdef _POSIX_MAPPED_FILES` (page-size query was the only POSIX dependency).
  Fix: `experiments/prefetch-experts-win`. The upstream-community prefetch scheduling built on
  top of it measured *neutral-to-negative* (`tier0-4-*.log`); only the pinning matters.
- **Micro-batch is ~3x prefill**: prefill re-streams all offloaded expert tensors once per
  ubatch; ub 512→2048 divides the re-streaming 4x. ub 4096 ties (prompt-capped).
- **VRAM cliff**: ncmoe 20 at ub 2048 oversubscribes → 94 pp / 14 tg. The knee (ncmoe 22) is
  "as many GPU-resident experts as still fit," coupled to ub through compute-buffer size.
- **All-cores beats P-cores-only by ~30% for generation** (35.2 vs 24.8 tg): decode is
  DDR5-bandwidth-bound, so E-cores add bandwidth-servicing threads. The "exclude E-cores for
  lockstep threading" wisdom (implemented Linux-only upstream) would *regress* this workload —
  a hybrid-CPU default that is right for compute-bound and wrong for bandwidth-bound.
- Two more Windows-specific defects found and fixed alongside: `--direct-io` silently
  advertised-but-unimplemented (`experiments/win-fast-load`), and GPU-uploaded mmap pages never
  released from the working set — 47.2 → 24.0 GB during generation via `VirtualUnlock`
  (`experiments/win-mmap-pressure`, output-identical).

Net best config: `-fa 1 -b 4096 -ub 2048 -ncmoe 22` + pinning = **1711 pp / 41 tg**.

## 4. Finding: how gpt-oss actually routes (trace study)

Per-token, per-layer routed expert ids captured in-engine (env-gated hook,
`experiments/routing-trace`; validated against independent imatrix expert counts: correlation
0.94/0.83, Gini agreement ±0.03). Corpora: wikitext 47.7K tokens, code 40.9K, chat 52K
(`bench-results/{wiki,code,chat}.trace`).

- **Per-layer routing is skewed** (median Gini 0.72, CV 1.86, rising with depth), but the
  **aggregate across layers is near-uniform** (Gini 0.19; all 128 experts active). Different
  experts dominate different layers.
- **Hotness is workload-specific**: per-layer bottom-32 "idle" experts overlap only 29.2%
  between prose and code — barely above the 25% random floor. The union of "important
  somewhere" is essentially all experts.
  - Consequence: **static expert pruning (REAP-style) is rejected for general-purpose use of
    this model** — any single-domain prune removes experts another domain needs. (REAP's
    published near-lossless results are on narrower code models; the mechanism's per-layer
    skew premise *does* hold here — it is the cross-domain variance that kills it.)
  - The same data is the existence proof for **dynamic** caching: what is hot right now is
    skewed and stable; what is hot in general is everything.
- **Cache-policy simulation** over the traces (LRU / LFU-decay / layer-cycle-aware / oracle;
  per-layer and shared pools; `simulate-cache.py`, tables in EXPERIMENTS.md):
  - Per-layer independent pools ≈ shared global pool (within 1–2 pts) — build the simple one.
  - **Plain LRU ties or beats layer-cycle-aware eviction at every budget tested** on all three
    workloads. The published "LRU is provably wrong for MoE" claim requires the pool to be
    smaller than one layer-cycle's working set — which does not hold at practical budgets on
    this model. LFU-decay adds +4–6 pts only under 16 slots.
  - 48 slots/layer (37.5%) → 92–96% hit; 64 → 96–98%; oracle headroom over LRU shrinks to
    2–5 pts at 48 slots, so learned/predictive prefetch is a refinement, not a core win.

## 5. Finding: a byte-exact expert cache that loses — and where the time actually goes

We implemented the cache the simulation specified (per-layer LRU slot pools, GPU-resident;
misses computed on CPU via a new masked `mul_mat_id` variant; hits via slot-id indirection
through `get_rows` — no custom CUDA op; async promotion through a pinned staging ring on a
dedicated copy stream with event-gated deferred map publish). Verification chain: four
standalone op gates, a whole-graph forced-empty byte gate, a same-device unit oracle for the
hit path, perplexity byte-identical to baseline at 92% real GPU hits, online hit rate matching
the offline simulation. The mechanism is **correct end-to-end** (branch `experiments/slot-cache`).

It is also slower than no cache: at ncmoe 36, 1.05–1.51 tg vs 19.4 baseline. The diagnosis
sequence matters more than the headline:

| hypothesis | verdict | evidence |
|---|---|---|
| VRAM oversubscription | refuted | 27.8 GB peak < 32 |
| async machinery overhead | refuted | async S8 3.85 tg > sync S8 3.49 |
| slot-matmul kernel pathology | refuted | flat ~18 µs/op from S=8 to S=128 (microbench) |
| token-boundary host cost | minor, inverse in S | 25→6 ms as S grows |
| **two-path orchestration** | **isolated (§5.1, NOCAP)** | two-path alone: tg 19.15→5.59 (S8) / 18.40→1.48 (S48) |

The structural component: option-(c)'s design runs *both* paths every layer (GPU slot matmul +
CPU masked matmul) and combines per layer; a CPU-resident layer also pays its activation
DtoH/HtoD round-trip as a graph edge regardless of whether it computes anything (162 GB DtoH
per generation window, hit-rate-independent). The S-scaling component sits in cross-backend
scheduling around that combine and is the open item — bounded by elimination, not yet
positively isolated. Synchronous promotion (phase 1) was separately quantified at 5–13x
slowdown, reproducing on our own design the failure mode we first measured in a prior
community cache attempt (tg 12.0 → 10.8, 108 host syncs/token).

**The honest summary: hit rate is necessary and very far from sufficient.** A cache with
provably near-optimal residency loses 20x to static placement if the per-layer execution
structure charges more for the split than the arithmetic it saves.

## 5.1 Phase 3: the two-path cost isolated, and elision killed at the design gate

Phase 3 (single-path per-layer dispatch — a fully-resident layer skips the CPU op and its
round-trip entirely) was pursued to gates only, and **stopped before implementation**. Three
measured findings resolve §5's open item and close the design.

**Finding A — the wall is the two-path structure, not routing capture (a misattribution).** §5's
diagnosis left "two-path orchestration" as a residual bounded by elimination. Adding an
everything-off control — `NOCAP`: the two-path graph built but capture and promotion disabled —
isolates it positively (ncmoe36 tg128, r3):

| | S=8 | S=48 | isolates |
|---|---|---|---|
| NOBOOK (plain path, no two-path) | 19.15 ± 3.33 | 18.40 ± 3.87 | baseline (flat in S) |
| NOCAP (two-path built, capture+promote OFF) | 5.59 ± 0.41 | 1.48 ± 0.03 | two-path split ALONE |
| fused (two-path + fused capture + promote) | 5.16 ± 0.31 | 1.52 ± 0.04 | + capture + promotion |

The two-path structure alone collapses tg (19.15→5.59 at S=8, 18.40→1.48 at S=48); adding capture
on top is free. The phase-2 verdict "per-token routing extraction is itself the cost" was an
artifact: every capture mechanism in that A/B was measured on top of the two-path, which had
already collapsed tg before any capture was added. **Process rule: an A/B over mechanisms is
uninterpretable without the everything-off cell on the same substrate** — this pattern recurred
three times in the campaign. Artifacts: SLOT-CACHE-DESIGN.md "P3.0 DONE"; correctness chain
ppl-exact 458.9669 (OFF==ON==forced-empty, byte-identical, 11 chunks); hit-rate canary
66.5%/92.4% exact vs sim.

**Finding B — routing capture is free once off the two-path.** Two independent capture mechanisms
cost nothing on the plain substrate: fused capture (`ggml_mul_mat_id_skip` optional src[4] writes
routed ids during the matmul it already runs; free by the NOCAP decomposition), and a batched
GPU-sink on the plain path (`...NOSLOT=1 GPUSINK=1 NOPROMO=1`: tg128 18.02 ± 4.62 ≈ NOBOOK 19.15,
online hit rate 64.7% confirming correct ids). The phase-2 "GPU-sink 5.2 tg" was entirely
two-path contamination. Residency tracking is not the obstacle. Artifacts: SLOT-CACHE-DESIGN.md
"P3.1 pre-measurements DONE" 3(a).

**Finding C — the per-token-escape compounding law kills token-level elision.** Binary dispatch
elides a layer only when it is fully resident; an elided layer *escapes* when the token routes to
a non-resident expert, dropping (renormalizing away) that expert's contribution. A weighted
routing trace (ids + final router weights, sum=1.000 verified) prices the damage. Gate: stop if
mean dropped weight-mass > ~1% OR the top-1-expert escape rate is not rare.

| S=48 (VRAM-matched to the 29.7 tg champion) | WIKI (ppl workload) | CODE (worst) | CHAT (best) | gate |
|---|---|---|---|---|
| per-layer escape \| elided | 24.74% | 25.34% | 5.71% | — |
| per-token escape (any of 36 layers) | 99.49% | 97.66% | 29.20% | — |
| **mean dropped weight-mass** | **7.20%** | **7.34%** | **1.67%** | **>1% → FAIL all** |
| **top-1-expert escape rate** | **21.0%** | **20.0%** | **19.4%** | **not rare → FAIL all** |

76% full-residency is a *per-layer* figure; across 36 independent layers it compounds to ~100%
per-token escape (wiki/code). The top-1-expert escape rate (~19–21%) is domain-invariant — an
escape lands on the token's highest-weight expert ~1 in 5 times regardless of workload, and
escaped experts carry 0.3–0.5 of the router weight (mean escaped rank ~1.8 of 4), so
renormalization cannot rescue quality. Even S=64 (2.97% dropped) fails and exceeds the VRAM
budget. The priced fallback (capture-then-replay: cost = elided + P(escape)·plain) is closed by
the same numbers: P(escape) ≈ 99.5% (wiki) means replay ~always → worse than plain. Exact
per-layer miss handling is the two-path Finding A proved slow. Artifacts:
`bench-results/item15-p31-kill-table-3domain.txt`, `bench-results/p31-mispredict-analysis.txt`,
`analyze-mispredict.py`, weighted traces `{wiki,code,chat}-w.tracew`.

**Verdict: phase 3 stopped, item 15 parked.** The cache's achievable residency at the VRAM budget
is fundamentally too low for token-level elision to be either exact-cheap (replay ~always) or
speculatively acceptable (drops a top expert on ~every token). The slot cache cannot beat the
static champion (29.7 tg) by any elision route, completing the phase-2 NO at the design level.
What survives is reusable: the everything-off-control rule (A), free routing capture (B), and the
per-token-escape compounding law (C) — a general bound on token-level expert elision at any MoE
residency short of ~full.

## 5.2 Self-speculation over the same cache also fails (item 17)

If elision cannot exploit residency inside one layer, can speculation exploit it across tokens? The
idea (Jordan's): the *draft* is the same model with routing restricted to only the VRAM-resident
experts (mask non-resident experts out of the router's top-k, renormalize) — so the draft needs zero
DDR5 expert reads — and a full-model *verify* runs once every K tokens in a batched pass that
amortizes the ~1.2 GB/token expert stream over the window. Decided by three offline gates before any
engine (env-gated diagnostic `GGML_MOE_DRAFT_SIM`, harness `llama-draftsim`; branch
`experiments/slot-cache`, validated byte-identical at all-resident and ppl 458.9669 with it off).

**Gate (a) — verify amortization is weak.** A batched K-token verify streams each *unique* expert
once, so its cost is the unique-experts-per-layer-per-window count, not K×4. But MoE routing has
little temporal locality: even K=8 touches ~17–18 unique of 32 possible experts/layer → amortization
only **1.7–1.9x**, verify_cost(K=8) ≈ 4.3–4.6 plain-token-times
(`bench-results/item17-gate-a-amortization.txt`).

**Gate (b) — draft agreement, and the entropy-decoupling finding.** Teacher-forced ≥2K tokens/domain,
greedy argmax, routing restricted to the S=48 resident set (`bench-results/item17-gate-b-agreement.txt`):

| S=48 | top-1 agreement | accept(4) | accept(6) | accept(8) |
|---|---|---|---|---|
| WIKI (prose) | 61.8% | 1.45 | 1.64 | 1.71 |
| CODE (worst dropped-mass) | 89.6% | 3.17 | 4.43 | 5.53 |
| CHAT (best dropped-mass) | 89.4% | 3.09 | 4.22 | 5.16 |

The key surprise: **draft agreement tracks output-token *entropy*, not expert dropped-mass.** Code is
the *worst* domain for dropped weight-mass (7.34%, §5.1) yet the *best* for token agreement (89.6%),
because its output is low-entropy (syntax, boilerplate) — dropping a top expert and renormalizing
rarely flips a token the context already determines. High-entropy prose (wiki) flips 38% of tokens on
the same perturbation. This decoupling is exactly why gate (b) had to be *measured*: the §5.1
mispredict analysis, which prices expert-mass damage, cannot predict token acceptance.

**Gate (c) — novelty: ADJACENT (not novel).** The nearest prior art, **SS-MoE** (ACM Web Conference
2026, DOI 10.1145/3774904.3792218), already couples same-model self-speculation with routing masked
to a resident expert subset for memory-limited MoE. SP-MoE (arXiv 2510.10302), MoE-SpeQ (2511.14102),
and llama.cpp MTP (PR #22673) all use a *separate* draft model or trained heads — distinct, but they
establish the surrounding design space. Only the dynamic-LRU-residency-defined draft fused with the
offload cache is narrowly new — not enough to carry the item alone.

**Verdict: stop, do not implement.** Projected speedup =
`champion(29.7)·accept(K)/(draft_frac·K + verify_cost(K))` peaks at **1.14x** across all
domains/K/draft-cost assumptions (code, K=4, cheapest draft, optimistic +1 verify bonus); realistic
cases are ≤1.0x, against a **1.25x** ship bar (`project-speedup.py`). verify_cost is a lower bound
(ignores fixed, attention, and graph-rebuild cost), so the projection is biased toward GO and still
fails. The memory wall the self-draft was meant to hide is not hidden: verify must still stream
unique(K) experts, gate (a) says that is only 1.7–1.9x cheaper than plain, and acceptance (gate b)
cannot cover even that.

## 6. Synthesis: spatially concentrated, temporally restless routing

Three independent attack surfaces — a byte-exact residency cache (§5), token-level elision (§5.1),
and cross-token self-speculation (§5.2) — fail for one shared reason, and it is a property of the
routing, not of any implementation.

- **Spatial concentration is real and exploitable-looking.** Per-layer routing is skewed (Gini 0.72,
  §4); 48 slots/layer capture 92–96% of requests (§4 sim); the slot cache hits 92% for real (§5).
  Every "should work" intuition about caching MoE experts starts here and is correct as far as it goes.
- **Temporal restlessness is the killer.** The hot set is not stable token-to-token. 76% per-layer
  full-residency compounds across 36 independent layers to ~100% per-token escape (§5.1); the residual
  misses are not low-weight tail experts but land on the token's *top* expert ~1 in 5 times, domain-
  invariantly (§5.1). So the 8% of requests the cache misses are not cheap to skip (they carry real
  weight-mass) and not rare per token (they hit every token somewhere).
- **Both caching and speculation need temporal stability the routing denies.** Exact elision must fall
  back to a per-layer two-path that costs more than it saves (§5, §5.1). Speculative elision perturbs
  quality on ~every token (§5.1). Self-speculation's verify cannot amortize because a K-token window
  has almost no expert reuse (§5.2 gate a), and drafting on the resident set flips high-entropy tokens
  (§5.2 gate b). The one lever that would help all three — a working set that holds still for a few
  tokens — is precisely what a general-purpose 128-expert top-4 router at a ~37%-of-experts VRAM budget
  does not provide.

The practical consequence for this hardware class: the shipped win is **static** (host pinning +
micro-batch + offload-split tuning, §3: 37x prefill / 3.4x generation over the contaminated baseline,
all zero-algorithm), and dynamic expert management — cache, prune, or speculate — does not beat a
well-tuned static offload at a consumer VRAM budget on this model. That is a negative result with a
mechanism, not a failure to engineer.

## 6.1 The trained-draft counterpoint: MTP over offload (GLM-4.5-Air)

<!-- MTP-OFFLOAD-RESULT: filled by the GLM-4.5-Air --mtp bench (deliverable 3). -->
*(Measurement in progress — GLM-4.5-Air Q4_K_M, `--mtp` on/off × two offload splits, paired
interleaved r≥8. Results and artifact path land here.)*

## 7. Process notes (why the numbers are trustworthy)

This fork accepts AI-generated contributions and compensates with mandatory verification
(gates before claims, artifacts over summaries, kill-lists for refuted ideas). Empirically the
discipline caught: a 5x contaminated baseline, two "obviously good" textbook optimizations that
measurement reversed (E-core exclusion, cycle-aware eviction), a gate that was unsatisfiable as
specified (byte-identity across device moves), and two explanations that fit the headline
number but not all the numbers. Sessions were run by two different AI models in
designer/executor/reviewer roles with explicit pause points; the full decision trail is in
EXPERIMENTS.md and SLOT-CACHE-DESIGN.md.

## 8. Limitations and next

Single model (gpt-oss-120b), single machine, single OS. Our results are about *dynamic expert
management under offload*; they do not speak to models that fit in VRAM, to trained speculative heads
(MTP), or to other MoE geometries. The trace tooling and simulator port directly to other MoE
families (Qwen3-Next, GLM-4.x) — the first replication target. The one dynamic technique we have not
exhausted is a *trained* draft: multi-token-prediction heads shipped inside the model (§6.1) sidestep
the acceptance problem that killed resident-only self-drafting, because the draft is learned rather
than derived from cache state. Pending: pinned-flag champion re-run for the final table, page-aligned
expert slab layout (EXPERIMENTS.md item 18).

## Artifacts

- Benchmarks: `bench-results/` (raw logs for every table above; `item15-*`, `item17-*` for §5.1–5.2),
  BENCHMARKS.md
- Tooling: routing-trace hook (env-gated), `simulate-cache.py`, `analyze-routing-skew.py`,
  `analyze-mispredict.py` (§5.1); `GGML_MOE_DRAFT_SIM` diagnostic + `llama-draftsim` harness,
  `analyze-amortization.py`, `gen-draft-masks.py`, `compare-argmax.py`, `project-speedup.py` (§5.2);
  microbenches and unit oracles in `tests/`
- Code: `experiments/prefetch-experts-win`, `win-mmap-pressure`, `win-fast-load`,
  `routing-trace`, `slot-cache`
- Decision trail: EXPERIMENTS.md, SLOT-CACHE-DESIGN.md

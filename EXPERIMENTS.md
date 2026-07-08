# Experiment queue — MoE bigger-than-VRAM on consumer Windows

Target workload: gpt-oss-120b MXFP4 (59 GB) on RTX 5090 (32 GB) + Core Ultra 9 285K (8P+16E) + 64 GB DDR5, Windows 11.

> **2026-07-08 — baselines below were re-measured and are WRONG (measured under load).** See "## Tier 0 results" for the clean-box numbers. Headline: the recorded ~46 pp / ~12 tg figures were taken while a second llama process + ComfyUI contended the box. On a clean idle box the *same* build+command gives **224 pp / 37.9 tg** — ~5x pp, ~3x tg — purely from removing contention. The real per-lever wins (pinning, ub, fa) are all measured against the clean baseline now.

Original (contended, do not trust the absolute numbers — llama-bench `-ngl 99 -ncmoe 24 -p 2048 -n 128 -r 3`):

| config | pp2048 t/s | tg128 t/s |
|---|---|---|
| baseline (ai-main) | 46.5 ± 1.0 | 12.0 ± 0.0 |
| prefetch branch, pinning off | 46.8 ± 1.0 | 12.3 ± 0.1 |
| prefetch + Windows host pinning (`experiments/prefetch-experts-win`) | 54.5 ± 0.2 | 12.4 ± 0.1 |
| moe-cache branch, `--moe-expert-cache-size 48` (llama-cli, single run) | ~51 | **10.8 — regression** |

moe-cache verdict: do NOT adopt as-is (tg regresses; routes all 36 layers through the cache, 108 host syncs/token). NOTE: its ~51/10.8 was also a contended-box run — the *relative* regression vs its own contended baseline may hold, but the absolute number is not comparable to the clean table below. Its 48-slot pools also fail to allocate unless ~23 GB VRAM is actually free — check `nvidia-smi` for squatters (ComfyUI/python) before any run.

Key diagnostics behind the original ranking (partially invalidated by the contention finding):
- ~~tg reads ~1.2 GB/token from DDR5 (~15 GB/s effective) — tg is scheduling/TLB-bound.~~ CORRECTED: the "15 GB/s effective" was a contended-box artifact. Clean box does 37 tg = ~44 GB/s effective DDR5. tg IS bandwidth-bound (more cores help — see item 3), not TLB-bound.
- PCIe bus at ~7% utilization during pp (measured under load — re-check on clean box) — prefill re-streams the same 40.6 GB once per ubatch (4x at default ub=512). ub sweep confirms this is real (item 1).
- CUDA constraint (measured the hard way): a cudaMemcpyAsync source must lie within ONE cudaHostRegister'd region; partial/chunked pinning of packed weights crashes ("invalid argument"). Pinning is all-or-nothing per mapping. WDDM caps total pinned ~36 GB on this box.

## Tier 0 — zero-code flag/env sweeps (do first, minutes each)

1. **ub sweep**: `llama-bench -ngl 99 -ncmoe 24 -fa 1 -p 2048 -n 0 -b 4096 -ub 512,1024,2048,4096 -r 3` on prefetch-win build with `GGML_CUDA_REGISTER_HOST=1`. Expected: up to 3-4x pp (stop re-streaming per-ubatch). Watch VRAM (bigger compute bufs vs 3x564 MB prefetch slots).
2. **ncmoe sweep 22/23/24**: 24 layers of experts ≈ 37.9 GiB > 36 GB pin cap; 22-23 may pin fully (no unpinned fragment log line) AND stream fewer bytes. Watch `pinned X MiB` lines + VRAM headroom.
3. **P-core threading for tg**: `-t 24` vs `-t 8` vs `-t 8 -C 0xFF --cpu-strict 1` vs `--prio 2 --poll 100` combos, `-p 0 -n 128 -r 5`. E-core exclusion is Linux-only (common/common.cpp:149,203) — Windows defaults to 24 lockstep threads incl. E-cores. Expected +10-30% tg.
4. **Prefetch slot depth**: `GGML_SCHED_PREFETCH_EXPERTS=2,3,6` — diagnostic: if no change, per-copy throughput is the wall, not pipeline depth.
5. **HAGS on/off** (needs reboot) and `--prio 2` under background load: tail-latency effects.

## Tier 0 results (measured 2026-07-08, clean idle box)

Build: `experiments/prefetch-experts-win` @ 5f83fbbe7 (9862) unless noted; plain baseline on `ai-main` d3a07f50e (9914). Model gpt-oss-120b MXFP4, ngl 99, ncmoe 24, p2048, n128, r3. ComfyUI/python VRAM squatter killed before running. All raw logs in `bench-results/tier0-*.log` and `bench-results/fu-*.log`.

**Headline: the recorded baselines were measured under load and undercount ~3–5x.** Isolation factorial (ncmoe 24, ub 512):

| build | pinning (`GGML_CUDA_REGISTER_HOST`) | fa | pp2048 | tg128 |
|---|---|---|---|---|
| ai-main d3a07f50e | off | 0 | 224.3 ± 30.7 | 37.9 ± 1.0 |
| ai-main d3a07f50e | off | 1 | 253.8 ± 15.1 | 37.1 ± 1.5 |
| prefetch 5f83fbbe7 | off | 0 | 228.9 ± 28.9 | 37.7 ± 0.9 |
| prefetch 5f83fbbe7 | off | 1 | 264.9 ± 9.4 | 35.8 ± 0.9 |
| prefetch 5f83fbbe7 | **on** | 0 | 494.0 ± 23.6 | 37.8 ± 1.3 |
| prefetch 5f83fbbe7 | **on** | 1 | 566.5 ± 4.0 | 34.0 ± 0.2 |

Row 1 is byte-identical build+command to the recorded 46.5/12.0 "baseline" — the 5x pp / 3x tg gap is pure box contention. tg math: 12 t/s ≈ 15 GB/s effective DDR5 (contended), 38 t/s ≈ 44 GB/s (idle) — tg is DDR5-bandwidth-bound, not TLB-bound as originally diagnosed.

Per-lever verdict (clean box):
- **Host pinning is the real win: ~2.1x pp** (229→494 at fa0, 265→567 at fa1), tg unaffected. This is the *only* valuable part of the prefetch-win branch.
- **Micro-batch (item 1) stacks ~2–3x pp on top.** ub sweep (prefetch build, pin on, fa1, `-b 4096 -p 2048 -n 0`):

  Definitive r=8 (`def-1-ubknee-r8.log`), ncmoe 24:

  | ub | 512 | 1024 | 2048 | 4096 |
  |---|---|---|---|---|
  | pp2048 t/s | 536 ± 36 | 1037 ± 21 | **1793 ± 83** | 1717 ± 163 |
  | tg128 t/s | 36.3 | 37.4 | 37.4 | 37.3 |

  ub=2048 is the knee (4096 ties/slightly regresses — p=2048 caps the effective ubatch). tg is flat across ub (as expected — ub is a prefill lever). Confirmed at r=8; the earlier ±327 noise was a single bad load.
- **ncmoe at the ub=2048 knee (item 2 redux, r=8, `def-2-ncmoe-at-ub2048-r8.log`) — VRAM cliff found:**

  | ncmoe | 20 | 22 | 24 |
  |---|---|---|---|
  | pp2048 t/s | 93.9 ± 1.2 | **1710.9 ± 192** | 1480.2 ± 239 |
  | tg128 t/s | 14.1 ± 0.2 | **41.1 ± 0.7** | 37.0 ± 0.9 |

  **ncmoe=22 is the sweet spot** — best pp AND tg. ncmoe=20 falls off a cliff (94 pp / 14 tg): at ub=2048 the compute buffers are ~8x larger, so putting 16 expert layers on GPU oversubscribes the 32 GB card (minus ~5 GB desktop) and thrashes. The cliff is between ncmoe 20 and 22. So ncmoe and ub are coupled through VRAM: you can't freely lower ncmoe at high ub. ncmoe=22 = "as many GPU-resident experts as still fit."
- **Flash-attn is minor: +13–16% pp, neutral-to-slightly-negative tg.** Keep it, not a headline.
- **Prefetch scheduling (item 4) is dead weight.** Depth 2/3/6 flat (pp 328/329/325 — per-copy throughput is the wall). Worse, *enabling* it regresses pp: ub512 518→328 (feature off vs on), ub2048 tie (1133 vs 1101). The branch's prefetch code adds nothing; only its pinning code matters.
- **ncmoe 22/23/24 (item 2), pin on, fa1:** pp 620 / 572 / 518, tg 39.4 / 38.5 / 35.4. Fewer CPU layers wins (more on GPU), NOT better pinning: every split pins the same 29,681 MiB (shard 2 alone) and leaves shard 3's CPU experts unpinned (`failed to register 7.1 / 8.8 / 10.4 GiB: out of memory` at 22/23/24). Shard 2 + any of shard 3 > 36 GB cap; all-or-nothing per mapping. The "22–23 pin fully" hypothesis is FALSE.
- **Threading (item 3), tg, r5:** `-t 24` = 35.2, `-t 8` = 24.8, `-t 8 -C 0xFF --cpu-strict 1` = 24.6, `-t 24 --prio 2 --poll 100` = 33.4. **All-cores wins; E-core exclusion hurts ~30%** (tg is bandwidth-bound). This KILLS Tier 1 item 8 — the Windows E-core patch would regress this workload. Marked below.
- **Item 5 (HAGS)** not run — needs a reboot.

**Net:** best clean-box config = `-fa 1 -b 4096 -ub 2048 -ncmoe 22` + `GGML_CUDA_REGISTER_HOST=1` → **1711 pp / 41 tg** vs the recorded 46.5 / 12.0 (≈37x pp, ≈3.4x tg). The gains are almost entirely (a) not benchmarking under load, (b) host pinning (~2x pp), (c) ub=2048 (~3x pp), (d) ncmoe=22 not 24 (+16% tg, sits just above the VRAM cliff). Flash-attn +15% pp. The "MoE-bigger-than-VRAM is slow" premise was substantially a benchmarking-under-load artifact; the branch's value is its host-pinning code, not its prefetch scheduling (which is neutral-to-negative).

## Re-ranked queue (post Tier 0, 2026-07-08)

Clean-box measurement changed the priorities. pp is already ~1790 (fa+ub2048+pin); tg ~37 and bandwidth-bound. Ranking by value/risk against the clean baseline:

Code anchors from tonight's source read are inline so these are executable, not aspirational.

1. **Item 14 interim — `has_direct_io()→false` on Windows** (`src/llama-mmap.cpp:173`). The Windows `llama_file::impl` ctor (`:86`) opens a buffered `FILE*` and `read_raw` (`:129`) always uses cached `ReadFile`; `has_direct_io()` returned `true`, a pure lie only consulted under `--direct-io` (`llama-model-loader.cpp:565`, default off). Safe honesty fix. **[DONE this session, branch `experiments/win-fast-load`, building `build-fastload`.]**
2. **Item 14 real — overlapped unbuffered loader. CORRECTED VALUE: only helps `--no-mmap --direct-io`.** Our benches use mmap (default), which faults pages in / PrefetchVirtualMemory — it never calls `read_raw` for bulk weights. So the "4× cold load" does NOT speed our mmap iteration. Still a legit contribution for no-mmap users, but deprioritized for this workload. Verify via deterministic `--no-mmap --direct-io` output-diff.
3. **Item 9 — mmap PrefetchVirtualMemory clamp (PROMOTED: this is the real mmap-path load-time win).** Loader prefetches the whole 59 GB even though ~21 GB is uploaded to GPU and freed; clamp the prefetch to the CPU-resident range + `VirtualUnlock` GPU-uploaded ranges (`unmap_fragment` is a Windows no-op today). Touches the path we actually use; lower risk than #4 (Win32 mmap hints, no cudaMemcpy landmine). Verify: cold-load time + output-diff. **Recommended next code target.**
4. **Item 12 — pinned-staging bounce for the unpinned fragment.** Tier 0 proved it: every ncmoe pins the same 29,681 MiB (shard 2) and leaves shard 3's 7–10 GB CPU experts pageable (`register_host` is per-mapping, `llama-mmap.cpp:634`; all-or-nothing). Route unregistered sources through a 2×64 MB `cudaHostAlloc` ring. Top *pp* win, but sits ON the documented "cudaMemcpyAsync source must lie in one registered region → invalid argument" crash landmine. **NOT to be landed unattended — needs a supervised session.** Env-gated (default off).
5. **Item 13 — large pages for CPU experts, env-gated.** Downgraded: tg is bandwidth-bound now, not TLB-bound, so +5–15% is optimistic. Clean default-off hypothesis test.
6. **Items 6/7/10/11 (moe-cache) — PARKED** per owner. Premise ("tg 12→18-25") is void — clean tg is already 37/41. Revisit only under a different design (item 15).
7. **Item 8 — KILLED** (E-core exclusion regresses this workload, Tier 0 item 3).
8. **Item 15 (persistent GPU slot cache) / Item 16 (REAP pruning)** — flagship/offline; re-baseline against ~41 tg (ncmoe 22) before projecting gains. Beyond one night.

## Tier 1 — small patches, high confidence

6. **moe-cache hybrid placement fix**: per-layer buft patterns so GPU-resident layers (blk 0..11) keep dense CUDA buffers + fusion; only CPU layers route through the cache; give all slack VRAM to those pools. Removes 36 syncs/token + 634 MB/token of pointless cache traffic. Est. tg 12 → 18-25. (moecache agent idea 2)
7. **Cache telemetry split pp/tg + routing trace dump**: extend dcc2cb143 stats; decides ideas 9-11 offline. (moecache agent idea 1)
8. ~~**Windows E-core default fix**: EfficiencyClass filter in common_cpu_get_num_physical_cores~~ **KILLED by Tier 0 item 3 (2026-07-08).** tg is DDR5-bandwidth-bound here; all-24-cores (35.2 tg) beats P-cores-only (24.8). Excluding E-cores would regress this MoE-offload workload ~30%. The upstream "E-cores harm lockstep threading" wisdom is wrong for CPU-resident MoE experts. Do not implement.
9. **mmap pressure pair**: clamp PrefetchVirtualMemory to CPU-resident range (loader passes -1 = whole 59 GB today); VirtualUnlock GPU-uploaded ranges (unmap_fragment is a Windows no-op). p99 token latency + load time. (windows agent idea 3)

## Tier 2 — medium patches, gated on Tier 0/1 measurements

10. **One routing snapshot per layer** in moe-cache: up/gate/down share ids — cache decoded ids per graph generation; persistent events; pinned ids staging. 108 → 36 syncs/token. (moecache agent idea 3)
11. **Cross-layer speculative prefetch** ("Prefetch C"): reuse layer L's unique_eids to prefetch L+1's three tensors during attention; API already exists (moe-cache.cu:571). Gate on measured cross-layer persistence ρ > 0.25 from #7's trace. (moecache agent idea 4)
12. **Pinned-staging bounce** for the unpinned mmap fragment: route unregistered sources through a 2x64 MB cudaHostAlloc ring at ~12-20 GB/s instead of WDDM pageable ~2-3 GB/s. Gate on nsys confirming pageable-copy stalls. (pcie agent idea 2)
13. **Large pages (MEM_LARGE_PAGES)** for CPU expert weights (--no-mmap path + SeLockMemoryPrivilege), env-gated. ~300K 4K-page touches/token today; 2 MB pages fix STLB. Est. +5-15% tg. (windows agent idea 2)
14. **Windows unbuffered overlapped load** (--direct-io is silently fake on Windows today — has_direct_io() returns true, ctor ignores it): FILE_FLAG_NO_BUFFERING + OVERLAPPED QD8+. Cold load ~60s → ~10-15s. Interim honest fix: return false. (windows agent idea 4)

## Item 1 results — routing trace + offline cache simulation (2026-07-08)

Design data for item 15. Env-gated hook `GGML_MOE_TRACE=<file>` (`common/moe-trace.cpp`,
installed via the existing eval-callback path when no other callback is present) records the
routed expert ids of each MoE layer's gate `mul_mat_id`, per token, in sequence order.
`simulate-cache.py` replays a trace through cache policies at several slot budgets.

**Capture is at DECODE.** During batched prefill the eval callback only fires for
GPU-resident-expert layers (a backend-split artifact); single-token evals fire all 36 layers.
So wiki/code traces feed the corpus as single-token decode steps (`-b 1 -ub 1`, teacher-forced)
and chat is free generation. Routing is placement-independent, so runs use `-ncmoe 36` (experts
on CPU) to free VRAM for a 98K context. Raw traces + logs in `bench-results/{wiki,code,chat*}.trace`,
`trace-*.log`, sim in `bench-results/cache-sim.log`.

**Hook validated** against imatrix per-expert counts (`simulate-cache.py verify`): wikitext
corr **0.94** / gini-diff **0.03**, code corr **0.83** / gini-diff **0.024**. Gini (skew
magnitude) matches almost exactly; sub-1.0 correlation is the trace being a ~45K-token slice at
full context vs the imatrix's full corpus in 512-tok chunks. Rules out transposition / layer-shift.

Hit-rate (%) = resident-expert requests / (tokens x 36 layers x 4). Slots are **per layer**;
"shared" = one pool of `slots x 36` across all layers. Per-layer least-stale == per-layer LRU
(one access/layer/token), so it is not shown separately.

wikitext (47.7K tok):

| slots/layer | 8 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|
| perlayer LRU      | 45.6 | 64.5 | 75.4 | 82.9 | 92.0 | 96.6 |
| perlayer LFU-decay| 49.3 | 67.9 | 77.5 | 84.0 | 92.3 | 96.7 |
| perlayer ORACLE   | 65.1 | 80.4 | 87.8 | 92.1 | 96.7 | 98.6 |
| shared LRU        | 44.4 | 64.6 | 76.2 | 84.4 | 93.8 | 97.9 |
| shared CYCLE      | 40.4 | 61.5 | 74.7 | 83.6 | 93.6 | 97.8 |
| shared ORACLE     | 69.1 | 82.7 | 89.6 | 93.7 | 97.7 | 99.2 |

code (40.9K tok):

| slots/layer | 8 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|
| perlayer LRU      | 39.6 | 59.0 | 74.3 | 83.9 | 93.4 | 97.1 |
| perlayer LFU-decay| 46.0 | 66.6 | 77.6 | 85.2 | 93.6 | 97.1 |
| perlayer ORACLE   | 61.2 | 79.1 | 87.9 | 92.7 | 97.1 | 98.7 |
| shared LRU        | 39.5 | 58.1 | 75.1 | 84.7 | 94.1 | 97.5 |
| shared CYCLE      | 37.8 | 58.5 | 74.0 | 84.0 | 93.7 | 97.3 |
| shared ORACLE     | 65.5 | 81.4 | 89.4 | 93.8 | 97.6 | 99.0 |

chat (52.0K tok, 4 merged generations):

| slots/layer | 8 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|
| perlayer LRU      | 62.7 | 80.8 | 87.1 | 91.6 | 95.8 | 98.0 |
| perlayer LFU-decay| 68.5 | 83.1 | 88.5 | 92.3 | 96.0 | 98.0 |
| perlayer ORACLE   | 78.1 | 89.6 | 93.8 | 96.1 | 98.2 | 99.1 |
| shared LRU        | 60.1 | 81.1 | 87.1 | 92.0 | 96.3 | 98.2 |
| shared CYCLE      | 62.8 | 80.4 | 87.3 | 91.6 | 96.1 | 98.2 |
| shared ORACLE     | 81.7 | 90.8 | 94.7 | 96.6 | 98.5 | 99.3 |

**Verdict (decides the item-15 design):**
- **Per-layer independent pools ~= shared pool** (within ~1-2 pts everywhere). The per-layer
  design wins on simplicity — no cross-layer slot map, no global eviction bookkeeping. Build it
  per-layer.
- **Plain LRU is sufficient; layer-cycle-aware eviction is REFUTED.** shared CYCLE ties or loses
  to shared LRU on all three workloads (e.g. wiki 8-slot 40.4 vs 44.4). The "LRU provably wrong"
  premise needs the pool to be smaller than one layer-cycle's working set; at these budgets it is
  not, so LRU never evicts a soon-needed expert. LFU-decay adds only +4-6 pts and only at tight
  budgets (<=16 slots); at practical budgets it ties LRU. Use LRU.
- **Sizing: 48 slots/layer (37.5% of 128 experts) -> 92-96% hit across all workloads; 64 -> 96-98%.**
  Below 32 slots workload sensitivity is large (chat 62.7% vs code 39.6% at 8 slots), but all
  workloads converge by 48. This is the Gini-0.72 per-layer skew paying off, and it confirms the
  cache is robust across workloads (unlike static REAP pruning, item 16).
- **Predictor headroom (ORACLE - LRU) ~= 4-9 pts at 32 slots, shrinking to 2-5 pts at 48.** The
  optional pre-attention MLP prefetcher (item 15) is a secondary optimization; a plain LRU cache
  already captures most of the achievable hit rate.
- **Caveat:** hit rate is necessary but not sufficient for a tg win. The moe-cache post-mortem
  (108 host syncs/token) shows a naive implementation erases the benefit. Item 15 must keep the
  hot path sync-free (persistent slot residency, async promote on miss). Do NOT project a tg
  number from hit rate alone — measure it.

## Tier 3 — the flagship project

15. **Persistent GPU expert slot cache** (upstream issue #20757 design + literature). **Design now fixed by item 1 measurement (2026-07-08):** build **per-layer independent LRU slot pools** (per-layer ~= shared within 1-2 pts, far simpler; cycle-aware eviction refuted — ties/loses to LRU), size **~48 slots/layer** (37.5% of 128 experts -> 92-96% hit across wiki/code/chat; 64 -> 96-98%). Miss = compute on CPU now + async promote for later (arXiv 2512.16473). Optional pre-attention expert predictor (2511.10676: 93-97% top-4 accuracy from a 2-layer MLP) buys only ~4-9 pts (oracle-LRU gap) at 32 slots, shrinking with budget — secondary, not core. Prototype on an 8 GB card hit 98-100% steady-state hits and 12-14 t/s on this model. Build on moe-cache branch after #6/#10 or fresh; keep the hot path sync-free (the moe-cache 108-syncs/token post-mortem is the failure mode). **Top Tier 3 target** (item 16/REAP rejected). Baseline to beat is 41 tg (ncmoe 22, clean). Hit rate is necessary but not sufficient — measure tg, do not project it.
16. **REAP expert pruning** (2510.13999, ICLR 2026). **Corrected + partially measured 2026-07-08.**
    - **NOT "GGUF surgery."** Cerebras tooling (github.com/CerebrasResearch/reap) operates on **HF PyTorch/safetensors**, needs the original gpt-oss HF checkpoint (not our GGUF), runs calibration forward passes on the full model (brutal on one 32 GB card), wants 24,576 samples × 16k tok, and **gpt-oss arch is not in their MODEL_ATTRS** (would need porting). gpt-oss was **not** among their evaluated models (Qwen3-Coder, GLM, Mixtral, Llama-4, Kimi); no released gpt-oss checkpoint. So item 16 = a multi-day HF port + re-quant to MXFP4 GGUF, not an afternoon.
    - Saliency: `S_j = mean_x[ g_j(x) · ||f_j(x)||_2 ]` (router gate × expert output L2 norm). Prune min-S experts per layer. Paper notes frequency-based pruning fails on uniform routing; REAP's saliency is meant to handle non-uniform.
    - **Routing skew measured on our GGUF via imatrix (`analyze-routing-skew.py`, wikitext 153K tok):** per-layer Gini **0.72** (CV 1.86, rising to 0.86 by layer 35) — genuinely skewed, which is REAP's per-layer operating regime. BUT aggregate-across-layers Gini **0.19** (all 128 experts active, top-32 = 37% vs 25% uniform). Different experts dominate different layers, so the *aggregate* is near-uniform — which is what the parked note below actually observed. **Refutes "uniform → REAP hopeless" for per-layer pruning; the per-layer skew REAP uses is real.** Importance proxy (count×RMS act): bottom-32/layer carry ~15% of mass.
    - **RESOLVED — REJECT for gpt-oss.** Cross-domain overlap (wikitext vs code, 300/250 chunks): per-layer bottom-32 idle-expert overlap = **29.2%**, barely above the 25% random floor. The skewed experts are workload-specific — prose-idle ≠ code-idle. So the union of "important somewhere" is ~all 128 experts; any single-domain prune lobotomizes off-domain, and the "idle everywhere" intersection is tiny. Confirms the parked note quantitatively. REAP's near-lossless wins were on narrower/code models (Qwen3-Coder, Kimi); gpt-oss is general and spreads load across workloads. A ~24% size cut isn't worth a multi-day HF port + off-domain quality loss. **Do not pursue.**
    - This same data *strengthens* item 15 (dynamic slot cache): high per-layer skew (Gini 0.72) + per-workload variation is exactly what a cache exploits — it holds whichever experts are hot *now* instead of betting on a static set. What REAP can't do statically, a slot cache does dynamically. **Pivot Tier 3 effort to #15.**
    - (Reusable tooling: `analyze-routing-skew.py`, imatrix built in build-main. Note: clean calibration text needed for any real PPL baseline — our wikitext dump has `<unk>`/`@-@` artifacts → PPL 321, unusable.)

Parked / rejected:
- Chunked host registration — crashes (copy-straddle), see above.
- Upload-only-routed-experts at prefill — worthless at ub≥512 (P(expert unused) ≈ e^-16).
- Compressed transfer (FloE/Huffman) — MXFP4 already near floor.
- Draft-model expert speculation — high complexity, predictor (#15) cheaper.
- Static hot-expert pinning — gpt-oss aggregate routing is near-uniform (skew is per-workload only). CONFIRMED by measurement (item 16): aggregate-across-layers Gini 0.19. But note per-*layer* routing is skewed (Gini 0.72) — static global pinning still loses, yet per-layer REAP pruning may not.

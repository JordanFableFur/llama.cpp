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

> **Superseded 2026-07-09 — see "## Consolidation + final re-rank" at the end of this file.** The
> dynamic-expert-management arc (items 15/16/17/18) is closed; only static items remain live.

Clean-box measurement changed the priorities. pp is already ~1790 (fa+ub2048+pin); tg ~37 and bandwidth-bound. Ranking by value/risk against the clean baseline:

Code anchors from tonight's source read are inline so these are executable, not aspirational.

1. **Item 14 interim — `has_direct_io()→false` on Windows** (`src/llama-mmap.cpp:173`). The Windows `llama_file::impl` ctor (`:86`) opens a buffered `FILE*` and `read_raw` (`:129`) always uses cached `ReadFile`; `has_direct_io()` returned `true`, a pure lie only consulted under `--direct-io` (`llama-model-loader.cpp:565`, default off). Safe honesty fix. **[DONE this session, branch `experiments/win-fast-load`, building `build-fastload`.]**
2. **Item 14 real — overlapped unbuffered loader. CORRECTED VALUE: only helps `--no-mmap --direct-io`.** Our benches use mmap (default), which faults pages in / PrefetchVirtualMemory — it never calls `read_raw` for bulk weights. So the "4× cold load" does NOT speed our mmap iteration. Still a legit contribution for no-mmap users, but deprioritized for this workload. Verify via deterministic `--no-mmap --direct-io` output-diff.
3. **Item 9 — mmap working-set release via Windows `unmap_fragment`. [DONE 2026-07-08, branch `experiments/win-mmap-pressure` @ 51a5a12e6.]** The loader already calls `unmap_fragment` on the fragments outside the CPU-resident tensor span (the GPU-uploaded regions, `llama-model-loader.cpp:1681`), but it was a Windows no-op, so those ~20 GB stayed resident. Implemented it via `VirtualUnlock` (trims the pages from the working set; mirrors the POSIX munmap-fragment path; inward page alignment keeps CPU-resident weights; `ERROR_NOT_LOCKED` expected+ignored). This realizes the intended "clamp to CPU-resident range" as a steady-state result. The eager whole-file prefetch is retained (it is readahead - every byte is read once during load regardless; the waste was residency, not I/O). **Measured (gpt-oss-120b MXFP4, ngl 99 ncmoe 22, mmap on, pinning off, RTX 5090; logs `bench-results/item9-*.log`, `ws-*.log`, `load-*.log`, `tg-*.log`):**

   | metric | before (no-op) | after (VirtualUnlock) |
   |---|---|---|
   | peak process working set (generation) | 47.2 GB | **24.0 GB** |
   | tg128 t/s (r5) | 31.79 +/- 6.65 | 32.58 +/- 6.61 (unchanged) |
   | load time (warm) | 3.74 s | 3.51 s (unchanged) |
   | temp-0 seed-42 output | - | byte-identical |

   ~23 GB (~49%) working-set reduction, no tg/output regression. On the 64 GB box this drops process RAM from ~74% to ~38%, removing the standby-list pressure that risks evicting hot CPU-resident expert pages. **Caveats:** load measured warm (standby-list clear needs admin/RAMMap or a reboot - not available headlessly, so true cold-load delta unmeasured); the WS reduction is the direct mechanism. A full 5-min soak (p99 inter-token, transition-faults/sec) was not run; tg stddev is unchanged (+/-6.6 both) as a jitter proxy. Verdict: **clean win, safe to land.**
4. **Item 12 — pinned-staging bounce for the unpinned fragment.** Tier 0 proved it: every ncmoe pins the same 29,681 MiB (shard 2) and leaves shard 3's 7–10 GB CPU experts pageable (`register_host` is per-mapping, `llama-mmap.cpp:634`; all-or-nothing). Route unregistered sources through a 2×64 MB `cudaHostAlloc` ring. Top *pp* win, but sits ON the documented "cudaMemcpyAsync source must lie in one registered region → invalid argument" crash landmine. **NOT to be landed unattended — needs a supervised session.** Env-gated (default off).
5. **Item 13 — large pages for CPU experts, env-gated. [BLOCKED 2026-07-08 - cannot enable headlessly; see Tier 2 item 13 for details + secpol steps.]** Downgraded: tg is bandwidth-bound now, not TLB-bound, so +5-15% is optimistic. Requires SeLockMemoryPrivilege which is not granted on this box (absent from token; `secedit` needs admin to change). Per session ground rules, documented and deferred rather than landing untestable ggml-core code unsupervised.
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
13. **Large pages (MEM_LARGE_PAGES)** for CPU expert weights (--no-mmap path + SeLockMemoryPrivilege), env-gated. ~300K 4K-page touches/token today; 2 MB pages fix STLB. Est. +5-15% tg (optimistic - tg is bandwidth-bound, not TLB-bound; see Tier 0). (windows agent idea 2)
    - **BLOCKED 2026-07-08 - privilege not available, cannot verify headlessly.** `whoami /priv` shows no SeLockMemory/"Lock pages in memory" in the process token; `secedit /export /areas USER_RIGHTS` needs admin (unavailable). Without the privilege `VirtualAlloc(..., MEM_LARGE_PAGES, ...)` fails, so the feature cannot be exercised or verified. Per the session rule ("verify the allocation actually used large pages before believing any number"; "if the privilege can't be enabled headlessly, document the exact secpol steps and move on"), NOT implemented this session - landing invasive, unverifiable ggml-core code unsupervised violates the verify-before-land discipline.
    - **To enable (needs a human with admin):** (1) `secpol.msc` -> Local Policies -> User Rights Assignment -> "Lock pages in memory" -> add the user account (or the account running llama). (2) Log off and back on (or reboot) so the token picks up the privilege. (3) Confirm with `whoami /priv` (should list `SeLockMemoryPrivilege`). Then a supervised session can implement + verify.
    - **Implementation plan for that session:** hook `ggml_backend_cpu_buffer_type_alloc_buffer` (`ggml/src/ggml-backend.cpp:2305`). When `getenv("GGML_CPU_LARGE_PAGES")`: once, `AdjustTokenPrivileges` to enable `SeLockMemoryPrivilege` (clean fallback + one clear log line if it fails); then `VirtualAlloc(NULL, round_up(size, GetLargePageMinimum()), MEM_RESERVE|MEM_COMMIT|MEM_LARGE_PAGES, PAGE_READWRITE)`, else fall back to `ggml_aligned_malloc`. Free path needs care: the shared `ggml_backend_cpu_buffer_i` frees via `ggml_aligned_free`, so large-page buffers need a parallel iface whose `free_buffer` calls `VirtualFree(ptr, 0, MEM_RELEASE)` (track allocation method). Verify large pages were actually used (e.g. working-set/large-page counters or a `VirtualQuery`) before trusting any tg delta. Expectation LOW (bandwidth-bound); a null result is a valid outcome.
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

17. **Resident-experts self-speculation (Jordan's idea, 2026-07-08).** Draft = the same model
    with routing restricted to VRAM-resident experts (slot cache + static layers); missing
    experts substituted by best-resident (or dropped, weights renormalized). Verify = full model
    (CPU experts) every K tokens in ONE batched pass — amortizes the per-token ~1.2 GB DDR5
    expert reads across K tokens. Synergy with item 15: ~92% cache hit rate means the draft
    deviates on only ~8% of expert selections → acceptance plausibly high; measured temporal
    locality means a K-token verify batch touches far fewer than K×4 unique experts/layer.
    **Offline gates BEFORE any implementation (computable from existing bench-results/*.trace +
    model, no GPU):**
    (a) unique-experts-per-K-token-window per layer from the traces → the amortization factor;
    (b) draft-vs-true token agreement under resident-only routing (teacher-forced,
        substitute-and-compare) → acceptance rate; proceed only if projected speedup
        K·accept/(K·draft_cost + verify_cost) beats phase-2 slot cache alone by >25%;
    (c) novelty check: SP-MoE (2510.10302) / MoE-SpeQ (2511.14102) couple a SEPARATE draft
        model with expert prefetch — verify the self-draft-via-masked-routing formulation
        is actually new before claiming it.
    Sequenced AFTER item 15 phases 0-2 (the cache is what makes the draft accurate). Heaviest
    engineering item on the board (masked routing + rollback in the decode graph).

    ### Item 17 — OFFLINE GATES RUN 2026-07-09 → **STOP, do not implement.** `experiments/slot-cache` (444125a82).
    All three gates run before any engine; no masked-routing/rollback graph built. Tooling committed:
    `GGML_MOE_DRAFT_SIM` diagnostic (restricts routing to a simulated S=48 LRU resident set, renorm
    over the present set; validated byte-identical at all-resident, ppl 458.9669 with env unset),
    harness `llama-draftsim`, `analyze-amortization.py`, `gen-draft-masks.py`, `compare-argmax.py`,
    `project-speedup.py`. Artifacts `bench-results/item17-gate-{a,b}-*.txt`.

    - **Gate (a) — amortization factor (verify cost).** Unique experts touched per layer per K-token
      window (batched verify streams these once instead of K×4). Weak temporal locality: even K=8
      touches ~17–18 unique of 32 possible → amortization only **1.7–1.9×**; verify_cost(K=8) ≈
      4.3–4.6 plain-token-times (wiki/code/chat). A K-token verify is not cheap.
    - **Gate (b) — draft-vs-true token agreement (the decisive number).** Teacher-forced ≥2K
      tokens/domain, greedy argmax, routing restricted to S=48 resident + renorm:

      | S=48 | top-1 agreement | accept(4) | accept(6) | accept(8) |
      |---|---|---|---|---|
      | WIKI (ppl) | 61.8% | 1.45 | 1.64 | 1.71 |
      | CODE (worst mass) | 89.6% | 3.17 | 4.43 | 5.53 |
      | CHAT (best mass) | 89.4% | 3.09 | 4.22 | 5.16 |

      Note agreement tracks output-token *entropy* (code/chat predictable → high) not dropped-mass —
      why gate (b) had to be measured, not inferred from the item-15 mispredict analysis.
    - **Projected speedup** = champion(29.7)·accept(K)/(draft_frac·K + verify_cost(K)), verify_cost
      from gate (a), draft_frac swept 0.1–0.3 (resident-only draft), optimistic +1 verify-bonus
      variant shown. **Max over all domains/K/assumptions = 1.14× (code K=4, +1 bonus, cheapest
      draft); realistic cases ≤1.0×.** Ship bar **1.25×** → **FAIL in every domain.** verify_cost is
      a lower bound (ignores fixed/attention/graph-rebuild cost), so the projection is biased toward GO
      and still fails.
    - **Gate (c) — novelty: ADJACENT (not novel).** Closest prior art **SS-MoE** (ACM WebConf 2026,
      DOI 10.1145/3774904.3792218): same-model self-speculation with routing masked to a resident
      expert subset for memory-limited MoE — pre-empts the broad mechanism. SP-MoE (2510.10302),
      MoE-SpeQ (2511.14102), MTP PR #22673 all use a *separate* draft model / trained heads (distinct).
      Only narrowly novel: the draft's resident set being a dynamic per-layer LRU cache fused with the
      offload cache (vs SS-MoE's static hot-expert partition). Not enough to carry the item alone.

    **Why STOP:** the memory wall the self-draft was meant to hide is not hidden — verify must still
    stream ~unique(K) experts (gate a, only 1.7–1.9× amortized), and acceptance is too low/short
    (gate b) to cover that cost; the one domain with high agreement (code, 89.6%) still projects
    ≤1.14×. Independently, the mechanism is already published (gate c). All three gates say no.

18. **MoE-aware GGUF layout: page-aligned expert slabs (Jordan's idea, 2026-07-08).** Offline
    repack (GGUF stays the container; this is a layout convention + loader awareness): pad every
    expert slab to a page boundary (~18 MB overhead on 59 GB) and store each expert's
    gate/up/down adjacent as one contiguous "promotion unit." What it buys, tied to measured
    constraints: (a) DISSOLVES the copy-straddle landmine — experts stop sharing pages, so
    per-expert cudaHostRegister/copy becomes legal (the constraint that killed chunked pinning
    and forced the staging ring's host-memcpy hop); (b) slot-cache promotion = one contiguous
    copy instead of three scattered ones, and pinned-slab experts can skip the ring's memcpy;
    (c) selective pinning under the 36 GB WDDM cap becomes possible (choose WHICH experts are
    pinned, e.g. by promotion frequency). Honest sizing: incremental (simplifies + trims
    promotion cost; does not change bandwidth math). **Gates:** (a) repack script + loader
    accepts padded layout, temp-0 output identical; (b) per-expert registration proven on the
    repacked file (the old crash signature must NOT reproduce); (c) promotion-cost delta
    measured in the phase-2 cache A/B. Sequenced after item 15 phase 2 (that's what it
    optimizes). Do not reorder experts by hotness in the file — workload-dependence killed
    static ordering (item 16).

Parked / rejected:
- Chunked host registration — crashes (copy-straddle), see above.
- Upload-only-routed-experts at prefill — worthless at ub≥512 (P(expert unused) ≈ e^-16).
- Compressed transfer (FloE/Huffman) — MXFP4 already near floor.
- Draft-model expert speculation — high complexity, predictor (#15) cheaper.
- Static hot-expert pinning — gpt-oss aggregate routing is near-uniform (skew is per-workload only). CONFIRMED by measurement (item 16): aggregate-across-layers Gini 0.19. But note per-*layer* routing is skewed (Gini 0.72) — static global pinning still loses, yet per-layer REAP pruning may not.

## Session report — 2026-07-08 (unattended Opus run)

Worked items 1, 9, 13, housekeeping, and the item-15 design doc. One benchmark process at a
time; box was clean (2.9 GB VRAM idle) throughout; no OOM/wedge.

**Landed:**
- **Item 1 (routing trace + cache sim) — DONE.** `GGML_MOE_TRACE` hook + `simulate-cache.py`;
  3 workloads traced (wiki 47.7K / code 40.9K / chat 52K tok), hook validated vs imatrix. Verdict:
  per-layer LRU pools, ~48 slots/layer -> 92-96% hit; cycle-aware eviction refuted; predictor
  optional. Hook code on `experiments/routing-trace` (bcdbf9402, pushed). Verdict + simulator +
  full tables on `ai-main`. Raw artifacts in `bench-results/`.
- **Item 9 (Windows unmap_fragment via VirtualUnlock) — DONE.** Working set 47.2 -> 24.0 GB during
  generation; tg and temp-0 output unchanged. Code on `experiments/win-mmap-pressure` (51a5a12e6,
  pushed). Verify logs `bench-results/item9-*.log`, `ws-*.log`.
- **BENCHMARKS.md — added** to `ai-main` (public clean-box results + methodology + contention warning).
- **SLOT-CACHE-DESIGN.md — added** to `ai-main` (item-15 proposal synthesizing item 1 + moe-cache
  post-mortem + issue #20757; staged plan with measurement gates; pre-implementation, needs review).

**Blocked / not done:**
- **Item 13 (large pages) — BLOCKED.** SeLockMemoryPrivilege not granted; cannot enable or verify
  headlessly. Documented grant steps + implementation plan (Tier 2 item 13). Not implemented
  (would be untestable ggml-core code).
- **Housekeeping 4(b) (sync master from upstream + merge into ai-main) — SKIPPED, needs supervision.**
  Directly conflicts with this session's ground rules ("do NOT touch master, do NOT interact with
  upstream ggml-org"). An `upstream` remote (ggml-org) exists, but fetching = interacting with
  upstream, and advancing master + merging into ai-main is a hard-to-validate operation to run
  unattended. Left for a supervised session. All local branches ARE pushed and in sync with origin.

**Branch state (all pushed to origin except noted):**
- `ai-main` — head of the queue; item 1/9/13 verdicts, BENCHMARKS.md, SLOT-CACHE-DESIGN.md, simulator.
- `experiments/routing-trace` (bcdbf9402) — MoE trace hook (item 1 code).
- `experiments/win-mmap-pressure` (51a5a12e6) — VirtualUnlock unmap_fragment (item 9 code).
- `experiments/win-fast-load` (f1d3aa3dd) — has_direct_io()->false (prior session).
- `experiments/prefetch-experts-win` (394cec7cc) — Windows host pinning (the ~2x pp win; prior session).
- `experiments/prefetch-experts` (5f83fbbe7), `experiments/moe-cache` (9cf4f1a9f) — prior sessions.
- `master` — untouched, mirrors origin/master (bec4772f6).

**Next session (recommended order):** (1) supervised: master ff-sync + merge to ai-main. (2) begin
item 15 phase 0/1 per SLOT-CACHE-DESIGN.md (telemetry gate first). (3) if a human grants
SeLockMemoryPrivilege, revisit item 13. (4) consider PRs for routing-trace and win-mmap-pressure
(both verified, self-contained).

## Consolidation + final re-rank — 2026-07-09

The dynamic-expert-management arc is closed. Every technique that tries to convert expert *residency*
into a decode speedup failed against a well-tuned static offload, for one measured reason: gpt-oss-120b
routing is **spatially concentrated but temporally restless** (per-layer Gini 0.72, but 76%-per-layer
residency compounds to ~100% per-token miss across 36 layers). See TECH-REPORT.md v1.0 §5–6.

**Final status of every dynamic idea:**
- **Item 15 (persistent GPU slot cache) — PARKED.** Byte-exact correct at 92% real GPU hits, but
  slower than no cache (two-path split, isolated via NOCAP: tg 18.4→1.5 at S=48). Phase-3 elision
  killed at the design gate (per-token-escape compounding). Reusable primitives kept on
  `experiments/slot-cache`. (SLOT-CACHE-DESIGN.md, TECH-REPORT §5–5.1.)
- **Item 16 (REAP static pruning) — REJECTED.** Cross-domain idle-expert overlap 29.2% ≈ random;
  workload-specific hotness lobotomizes off-domain.
- **Item 17 (resident-only self-speculation) — STOP.** Three offline gates: amortization 1.7–1.9×
  (a), projected speedup ≤1.14× vs 1.25× bar (b), novelty ADJACENT to SS-MoE (c). Not implemented.
  Entropy-decoupling finding: draft agreement tracks output-token entropy, not dropped-mass.
  (TECH-REPORT §5.2, `bench-results/item17-*`.)
- **Item 18 (page-aligned expert slabs) — PARKED with item 15.** It optimized slot-cache promotion;
  with the cache parked it loses its primary consumer. Its independent value (dissolving the
  copy-straddle landmine for item 12 pinning) stands but is low-priority.
- **MTP-over-offload (GLM-4.5-Air, trained-draft counterpoint) — <!-- MTP-VERDICT -->measurement pending.**
  The one dynamic technique not yet exhausted: a *trained* multi-token-prediction head sidesteps the
  acceptance problem that killed resident-only self-drafting. Bench: `--spec-type draft-mtp` on/off ×
  two offload splits, paired r≥8. (TECH-REPORT §6.1.)

**What shipped (all static, zero-algorithm):** host pinning (`prefetch-experts-win`, ~2.1× pp),
ub=2048 micro-batch (~3× pp), ncmoe=22 offload split (+16% tg), Windows working-set release
(`win-mmap-pressure`), honest `--direct-io` (`win-fast-load`). Net vs contaminated baseline: ~37× pp,
~3.4× tg — none of it from dynamic expert management.

**Remaining live queue (static only):** item 12 (pinned-staging bounce, supervised — crash landmine),
item 13 (large pages, blocked on SeLockMemoryPrivilege), item 2/14 (no-mmap direct-io loader, niche).

**Branch list (current):**
- `ai-main` — queue + BENCHMARKS.md + TECH-REPORT.md v1.0 + SLOT-CACHE-DESIGN.md + simulators.
- `experiments/slot-cache` — slot cache + item-17 draft-sim diagnostic + gate tooling (parked, kept).
- `experiments/routing-trace` — MoE trace hook (item 1).
- `experiments/win-mmap-pressure` — VirtualUnlock working-set release (item 9, shippable).
- `experiments/win-fast-load` — honest has_direct_io() (shippable).
- `experiments/prefetch-experts-win` — Windows host pinning (the ~2× pp win, shippable).
- `master` — untouched, mirrors origin/master.

**Session report (2026-07-09, consolidation):** folded item-15 phase-3 + item-17 gates into
TECH-REPORT.md v1.0 (thesis + entropy-decoupling); BENCHMARKS.md user-facing pass (cold-clock note,
branch links); ran the MTP-over-offload bench (GLM-4.5-Air, downloaded); final re-rank above. One
llama process at a time; box clean. Stopped before any external-publication step (human call).

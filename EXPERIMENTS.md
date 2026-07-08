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

1. **Item 14 — Windows overlapped unbuffered loader.** Low risk (I/O only, no extra memory → no OOM/wedge path), 4× cold load. Compounds every future bench. Ship the honest `has_direct_io()→false` interim first. **[executing overnight, branch `experiments/win-fast-load`]**
2. **Item 12 — pinned-staging bounce for the unpinned fragment.** Tier 0 proved it: shard 3's 7–10 GB CPU experts fail to pin (36 GB cap) and stream pageable during pp. Route them through a cudaHostAlloc ring. Top *pp* code win at ncmoe 24. Env-gated (default off). **[executing overnight if #14 clean, branch `experiments/pinned-bounce`]**
3. **Item 13 — large pages (MEM_LARGE_PAGES) for CPU experts, env-gated.** Downgraded: tg is bandwidth-bound now, not TLB-bound, so the +5–15% estimate is optimistic. Still a clean default-off hypothesis test.
4. **Items 6/7/10/11 (moe-cache) — PARKED** per owner (skip moe-cache as-is). Also: their premise ("tg 12→18-25") is void — clean tg is already 37. Revisit only if a fundamentally different design (item 15) is attempted.
5. **Item 8 — KILLED** (E-core exclusion regresses this workload, see Tier 0 item 3).
6. **Item 15 (persistent GPU slot cache) / Item 16 (REAP pruning)** — flagship/offline, unchanged in principle but re-baseline against ~37 tg before projecting gains. Beyond one night.

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

## Tier 3 — the flagship project

15. **Persistent GPU expert slot cache** (upstream issue #20757 design + literature): fixed slot pool across all layers, persistent expert→slot map across tokens, miss = compute on CPU now + async promote for later (arXiv 2512.16473), eviction = least-stale/layer-cycle-aware (SpecMD 2602.03921, LRU provably wrong), optional pre-attention expert predictor for prefetch lead time (2511.10676: 93-97% top-4 accuracy from a 2-layer MLP). Prototype on an 8 GB card hit 98-100% steady-state hits and 12-14 t/s on this model; projected 25-40 t/s tg here. Build on moe-cache branch after #6/#10 or fresh.
16. **REAP expert pruning** (offline GGUF surgery, 2510.13999): 128 → 96 experts (~45 GB) near-lossless on other models; validate perplexity on gpt-oss first. Multiplies every cache win.

Parked / rejected:
- Chunked host registration — crashes (copy-straddle), see above.
- Upload-only-routed-experts at prefill — worthless at ub≥512 (P(expert unused) ≈ e^-16).
- Compressed transfer (FloE/Huffman) — MXFP4 already near floor.
- Draft-model expert speculation — high complexity, predictor (#15) cheaper.
- Static hot-expert pinning — gpt-oss aggregate routing is near-uniform (skew is per-workload only).

# Experiment queue — MoE bigger-than-VRAM on consumer Windows

Target workload: gpt-oss-120b MXFP4 (59 GB) on RTX 5090 (32 GB) + Core Ultra 9 285K (8P+16E) + 64 GB DDR5, Windows 11.
Measured so far (llama-bench, `-ngl 99 -ncmoe 24 -p 2048 -n 128 -r 3`):

| config | pp2048 t/s | tg128 t/s |
|---|---|---|
| baseline (ai-main) | 46.5 ± 1.0 | 12.0 ± 0.0 |
| prefetch branch, pinning off | 46.8 ± 1.0 | 12.3 ± 0.1 |
| prefetch + Windows host pinning (`experiments/prefetch-experts-win`) | 54.5 ± 0.2 | 12.4 ± 0.1 |

Key diagnostics behind the ranking:
- PCIe bus at ~7% utilization during pp (4.3 GB/s of ~55 GB/s practical) — prefill re-streams the same 40.6 GB once per ubatch (4x at default ub=512).
- tg reads ~1.2 GB of expert weights/token from DDR5 (~15 GB/s effective of ~80) — tg is scheduling/TLB-bound, not bandwidth-bound.
- CUDA constraint (measured the hard way): a cudaMemcpyAsync source must lie within ONE cudaHostRegister'd region; partial/chunked pinning of packed weights crashes ("invalid argument"). Pinning is all-or-nothing per mapping. WDDM caps total pinned ~36 GB on this box.
- moe-cache branch routes ALL 36 layers through the cache (overrides --n-cpu-moe, src/llama.cpp:288-309) and pays 108 host syncs/token — expected to underperform until fixed.

## Tier 0 — zero-code flag/env sweeps (do first, minutes each)

1. **ub sweep**: `llama-bench -ngl 99 -ncmoe 24 -fa 1 -p 2048 -n 0 -b 4096 -ub 512,1024,2048,4096 -r 3` on prefetch-win build with `GGML_CUDA_REGISTER_HOST=1`. Expected: up to 3-4x pp (stop re-streaming per-ubatch). Watch VRAM (bigger compute bufs vs 3x564 MB prefetch slots).
2. **ncmoe sweep 22/23/24**: 24 layers of experts ≈ 37.9 GiB > 36 GB pin cap; 22-23 may pin fully (no unpinned fragment log line) AND stream fewer bytes. Watch `pinned X MiB` lines + VRAM headroom.
3. **P-core threading for tg**: `-t 24` vs `-t 8` vs `-t 8 -C 0xFF --cpu-strict 1` vs `--prio 2 --poll 100` combos, `-p 0 -n 128 -r 5`. E-core exclusion is Linux-only (common/common.cpp:149,203) — Windows defaults to 24 lockstep threads incl. E-cores. Expected +10-30% tg.
4. **Prefetch slot depth**: `GGML_SCHED_PREFETCH_EXPERTS=2,3,6` — diagnostic: if no change, per-copy throughput is the wall, not pipeline depth.
5. **HAGS on/off** (needs reboot) and `--prio 2` under background load: tail-latency effects.

## Tier 1 — small patches, high confidence

6. **moe-cache hybrid placement fix**: per-layer buft patterns so GPU-resident layers (blk 0..11) keep dense CUDA buffers + fusion; only CPU layers route through the cache; give all slack VRAM to those pools. Removes 36 syncs/token + 634 MB/token of pointless cache traffic. Est. tg 12 → 18-25. (moecache agent idea 2)
7. **Cache telemetry split pp/tg + routing trace dump**: extend dcc2cb143 stats; decides ideas 9-11 offline. (moecache agent idea 1)
8. **Windows E-core default fix**: EfficiencyClass filter in common_cpu_get_num_physical_cores (mirrors Linux CPUID path). Upstream-quality patch. (windows agent idea 1)
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

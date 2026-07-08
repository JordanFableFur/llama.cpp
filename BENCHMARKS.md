# Benchmarks

Measured results for this fork. Every number carries its measurement conditions;
a number without conditions is noise. See [EXPERIMENTS.md](EXPERIMENTS.md) for the
full campaign log, raw tables, and per-lever analysis.

## Methodology

- **One llama process at a time, on an otherwise idle box.** The single largest
  error we found was benchmarking under load: a second llama instance plus a
  ~7 GB VRAM squatter (ComfyUI) made the *same* build+command report **5x lower
  prefill and 3x lower generation** than a clean box. Always check `nvidia-smi`
  and the process list before, and kill GPU/RAM squatters.
- Redirect long runs to a file (`> log 2>&1`); never pipe (Windows pipe buffering
  deadlocks long runs).
- No benchmarking while a build compiles.
- llama-bench suppresses info logs: confirm env-gated code paths fired with `-v`
  and the expected log line, and confirm the run actually completed (exit code /
  result rows), not just that a line printed.
- Claims carry artifacts: the llama-bench table and the proving log line, not a
  summary.

## Contention warning

The recorded "baseline" of 46.5 pp / 12.0 tg for gpt-oss-120b on this class of
hardware was measured under load and is wrong. The clean-box baseline for the
identical build+command is **224 pp / 37.9 tg**. If your numbers are far below
the tables here, suspect contention first.

## Target workload

gpt-oss-120b MXFP4 (59 GB, 128 experts, 4 active, 36 layers) on:
- RTX 5090 (32 GB), CUDA arch 1200 (Blackwell, native FP4)
- Core Ultra 9 285K (8P + 16E, 24 logical, no HT)
- 64 GB DDR5, Windows 11

Build: VS 2022, `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF`.

## Tier 0 - clean-box flag/env sweeps (2026-07-08)

All rows: llama-bench, `-ngl 99`, gpt-oss-120b MXFP4, clean idle box, r>=3.

### Host pinning + flash-attn (ncmoe 24, ub 512)

Pinning = env `GGML_CUDA_REGISTER_HOST=1` (cudaHostRegister on the mmap).

| build | pinning | fa | pp2048 t/s | tg128 t/s |
|---|---|---|---|---|
| ai-main | off | 0 | 224.3 +/- 30.7 | 37.9 +/- 1.0 |
| ai-main | off | 1 | 253.8 +/- 15.1 | 37.1 +/- 1.5 |
| prefetch-win | off | 0 | 228.9 +/- 28.9 | 37.7 +/- 0.9 |
| prefetch-win | off | 1 | 264.9 +/- 9.4  | 35.8 +/- 0.9 |
| prefetch-win | **on** | 0 | 494.0 +/- 23.6 | 37.8 +/- 1.3 |
| prefetch-win | **on** | 1 | 566.5 +/- 4.0  | 34.0 +/- 0.2 |

Host pinning is ~2.1x prefill; tg unaffected.

### Micro-batch sweep (pinning on, fa1, ncmoe 24, r=8)

| ub | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| pp2048 t/s | 536 +/- 36 | 1037 +/- 21 | **1793 +/- 83** | 1717 +/- 163 |
| tg128 t/s  | 36.3 | 37.4 | 37.4 | 37.3 |

ub=2048 is the knee (p=2048 caps the effective ubatch, so 4096 ties/regresses).
tg is flat across ub (prefill lever only).

### ncmoe at the ub=2048 knee (pinning on, fa1, r=8)

| ncmoe | 20 | 22 | 24 |
|---|---|---|---|
| pp2048 t/s | 93.9 +/- 1.2 | **1710.9 +/- 192** | 1480.2 +/- 239 |
| tg128 t/s  | 14.1 +/- 0.2 | **41.1 +/- 0.7** | 37.0 +/- 0.9 |

ncmoe=22 is the sweet spot (best pp AND tg). At ub=2048 the compute buffers are
~8x larger; ncmoe=20 puts too many expert layers on the 32 GB card and thrashes
off a VRAM cliff (94 pp / 14 tg). ncmoe and ub are coupled through VRAM.

### Threading for generation (tg, r=5)

| config | tg128 t/s |
|---|---|
| -t 24 (all cores) | 35.2 |
| -t 8 (P-cores only) | 24.8 |
| -t 8 -C 0xFF --cpu-strict 1 | 24.6 |
| -t 24 --prio 2 --poll 100 | 33.4 |

All-cores wins. Excluding E-cores regresses ~30% - tg is DDR5-bandwidth-bound for
CPU-resident MoE experts, so more cores = more bandwidth. The usual "E-cores harm
lockstep threading" advice is wrong for this workload.

## Best config (RTX 5090-class)

```
GGML_CUDA_REGISTER_HOST=1 llama-bench -m gpt-oss-120b-mxfp4.gguf \
  -ngl 99 -ncmoe 22 -fa 1 -b 4096 -ub 2048
```

-> **~1711 pp / ~41 tg** on a clean idle box, vs the contended 46.5 / 12.0
(~37x pp, ~3.4x tg). The gains: (1) not benchmarking under load, (2) host
pinning (~2x pp), (3) ub=2048 (~3x pp), (4) ncmoe=22 sitting just above the VRAM
cliff (+16% tg), (5) flash-attn (+15% pp).

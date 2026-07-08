#!/usr/bin/env bash
# Tier 0 zero-code flag/env sweeps for EXPERIMENTS.md.
# Build: experiments/prefetch-experts-win (llama.cpp-prefetch/build).
# Rules baked in from CLAUDE.md:
#   - one llama process at a time (this script is strictly sequential)
#   - never pipe a long run's stdout: every invocation redirects to a file
#   - verify env-gated paths fired with -v + the log line ("pinned ... MiB")
# Each sweep writes its own log so a mid-run crash never loses completed work.

set -u
BENCH=/c/Users/Jorda/llama.cpp-prefetch/build/bin/Release/llama-bench.exe
MODEL=/c/Users/Jorda/models/gpt-oss-120b/gpt-oss-120b-mxfp4-00001-of-00003.gguf
OUT=/c/Users/Jorda/llama.cpp/bench-results
mkdir -p "$OUT"

run() {  # run <logfile> <env-assignments-or-"-"> <args...>
  local log="$OUT/$1"; shift
  local envs="$1"; shift
  echo "=================================================================="
  echo ">>> $log"
  echo ">>> env: $envs"
  echo ">>> args: $*"
  echo "=================================================================="
  if [ "$envs" = "-" ]; then
    "$BENCH" "$@" > "$log" 2>&1
  else
    env $envs "$BENCH" "$@" > "$log" 2>&1
  fi
  local rc=$?
  echo "exit=$rc  ($log)"
  # surface the result table + pinning proof immediately
  grep -E '^\|' "$log" | tail -n 8
  grep -iE 'pinned .* MiB|failed to register' "$log" | sort -u
  echo
  return $rc
}

# --- Item 1: ubatch sweep (pp only), pinning ON --------------------------
# Expected up to 3-4x pp once we stop re-streaming experts per ubatch.
run "tier0-1-ubsweep.log" "GGML_CUDA_REGISTER_HOST=1" \
    -m "$MODEL" -ngl 99 -ncmoe 24 -fa 1 -p 2048 -n 0 -b 4096 -ub 512,1024,2048,4096 -r 3 -v

# --- Item 2: ncmoe sweep 22/23/24, pinning ON, -v for pinned lines -------
# 24 layers experts ~37.9 GiB > 36 GB pin cap; 22-23 may pin fully.
run "tier0-2-ncmoe-sweep.log" "GGML_CUDA_REGISTER_HOST=1" \
    -m "$MODEL" -ngl 99 -ncmoe 22,23,24 -fa 1 -p 2048 -n 128 -r 3 -v

# --- Item 3: P-core threading for tg (tg only, r=5) -----------------------
# A+B: -t 8 vs 24 in one load (threads are runtime).
run "tier0-3a-threads-8-24.log" "GGML_CUDA_REGISTER_HOST=1" \
    -m "$MODEL" -ngl 99 -ncmoe 24 -fa 1 -p 0 -n 128 -t 8,24 -r 5
# C: 8 threads pinned to the 8 P-cores (logical 0-7, no HT on 285K) + strict.
run "tier0-3c-pcore-strict.log" "GGML_CUDA_REGISTER_HOST=1" \
    -m "$MODEL" -ngl 99 -ncmoe 24 -fa 1 -p 0 -n 128 -t 8 -C 0xFF --cpu-strict 1 -r 5
# D: high priority + busy-poll.
run "tier0-3d-prio-poll.log" "GGML_CUDA_REGISTER_HOST=1" \
    -m "$MODEL" -ngl 99 -ncmoe 24 -fa 1 -p 0 -n 128 -t 24 --prio 2 --poll 100 -r 5

# --- Item 4: prefetch slot depth (env), pinning ON -----------------------
# Diagnostic: if depth doesn't move pp, per-copy throughput is the wall.
for d in 2 3 6; do
  run "tier0-4-prefetch-$d.log" "GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_PREFETCH_EXPERTS=$d" \
      -m "$MODEL" -ngl 99 -ncmoe 24 -fa 1 -p 2048 -n 128 -r 3
done

echo "############### Tier 0 sweep complete ###############"

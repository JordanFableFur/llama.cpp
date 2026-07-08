#!/usr/bin/env bash
# Isolation factorial after the Tier 0 sweep surfaced two surprises:
#   (1) flash-attn (-fa 1) looks like a ~10x pp / ~3x tg win vs the recorded
#       baselines, which were all measured with no -fa;
#   (2) enabling the prefetch-experts feature (GGML_SCHED_PREFETCH_EXPERTS)
#       appears to REGRESS pp once -fa is on (518 -> 328 at ub=512).
# This script separates flash-attn x pinning x prefetch so each effect is
# attributable, and rules out box-contention as the cause of (1).
# Rules: one process at a time (sequential), redirect every run to a file.
set -u
BENCH=/c/Users/Jorda/llama.cpp-prefetch/build/bin/Release/llama-bench.exe
MAIN=/c/Users/Jorda/llama.cpp/build-main/bin/Release/llama-bench.exe
MODEL=/c/Users/Jorda/models/gpt-oss-120b/gpt-oss-120b-mxfp4-00001-of-00003.gguf
OUT=/c/Users/Jorda/llama.cpp/bench-results
mkdir -p "$OUT"

run() {  # run <logfile> <bin> <env|-> <args...>
  local log="$OUT/$1"; shift; local bin="$1"; shift; local envs="$1"; shift
  echo "=================================================================="
  echo ">>> $log"
  echo ">>> env: $envs   args: $*"
  echo "=================================================================="
  if [ "$envs" = "-" ]; then "$bin" "$@" > "$log" 2>&1; else env $envs "$bin" "$@" > "$log" 2>&1; fi
  echo "exit=$?  ($log)"
  grep -E '^\|' "$log" | tail -n 6
  grep -iE 'pinned .*MiB|failed to register' "$log" | sort -u
  echo
}

# ---- flash-attn x pinning, ncmoe 24, ub 512 (prefetch feature OFF) --------
# A: pinning OFF, fa 0 vs 1  -> isolates fa alone; fa0 should reproduce ~46 pp.
run "fu-A-pinoff-fa01.log"  "$BENCH" "-" \
    -m "$MODEL" -ngl 99 -ncmoe 24 -fa 0,1 -p 2048 -n 128 -r 3
# B: pinning ON,  fa 0 vs 1  -> pinning delta; fa0 should reproduce ~54 pp.
run "fu-B-pinon-fa01.log"   "$BENCH" "GGML_CUDA_REGISTER_HOST=1" \
    -m "$MODEL" -ngl 99 -ncmoe 24 -fa 0,1 -p 2048 -n 128 -r 3

# ---- plain ai-main build (no prefetch branch at all), fa 0 vs 1 ------------
# C: rules out that the fa win is specific to the prefetch build.
run "fu-C-main-fa01.log"    "$MAIN"  "-" \
    -m "$MODEL" -ngl 99 -ncmoe 24 -fa 0,1 -p 2048 -n 128 -r 3

# ---- prefetch feature ON vs OFF at the good operating point ---------------
# fa on, ub 2048 (the pp knee), pinning on. Does the branch's headline
# feature help or hurt at the point that actually matters?
run "fu-D-ub2048-prefOFF.log" "$BENCH" "GGML_CUDA_REGISTER_HOST=1" \
    -m "$MODEL" -ngl 99 -ncmoe 24 -fa 1 -p 2048 -n 128 -b 4096 -ub 2048 -r 3
run "fu-E-ub2048-prefON.log"  "$BENCH" "GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_PREFETCH_EXPERTS=3" \
    -m "$MODEL" -ngl 99 -ncmoe 24 -fa 1 -p 2048 -n 128 -b 4096 -ub 2048 -r 3

echo "############### isolation factorial complete ###############"

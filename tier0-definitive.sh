#!/usr/bin/env bash
# Definitive clean-box baseline: nail the ub knee (was noisy 1133-1707 across
# loads) and map the ncmoe tradeoff at the good config, with tight error bars.
# All: prefetch-win build, pinning ON, fa ON. One process at a time, redirected.
set -u
BENCH=/c/Users/Jorda/llama.cpp-prefetch/build/bin/Release/llama-bench.exe
MODEL=/c/Users/Jorda/models/gpt-oss-120b/gpt-oss-120b-mxfp4-00001-of-00003.gguf
OUT=/c/Users/Jorda/llama.cpp/bench-results
mkdir -p "$OUT"

run() {  # run <logfile> <env|-> <args...>
  local log="$OUT/$1"; shift; local envs="$1"; shift
  echo "=================================================================="
  echo ">>> $log   env: $envs   args: $*"
  echo "=================================================================="
  env $envs "$BENCH" "$@" > "$log" 2>&1
  echo "exit=$?  ($log)"
  grep -E '^\|' "$log" | grep -vE 'model *\||-----'
  echo
}

# 1) High-rep ub knee, ncmoe 24 (r=8 to kill the variance).
run "def-1-ubknee-r8.log" "GGML_CUDA_REGISTER_HOST=1" \
    -m "$MODEL" -ngl 99 -ncmoe 24 -fa 1 -p 2048 -n 128 -b 4096 -ub 512,1024,2048,4096 -r 8

# 2) ncmoe tradeoff at the ub=2048 knee (r=8). 20/22/24 = 3 reloads.
run "def-2-ncmoe-at-ub2048-r8.log" "GGML_CUDA_REGISTER_HOST=1" \
    -m "$MODEL" -ngl 99 -ncmoe 20,22,24 -fa 1 -p 2048 -n 128 -b 4096 -ub 2048 -r 8

echo "############### definitive clean baseline complete ###############"

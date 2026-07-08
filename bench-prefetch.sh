#!/usr/bin/env bash
# Benchmark protocol: baseline vs prefetch-experts vs prefetch+Windows-pinning
# Workload: gpt-oss-120b MXFP4 (~59GB) with MoE experts partially on CPU (--n-cpu-moe)
# Usage: ./bench-prefetch.sh <n-cpu-moe> [repetitions]

set -e
MODEL=/c/Users/Jorda/models/gpt-oss-120b/gpt-oss-120b-mxfp4-00001-of-00003.gguf
NCMOE=${1:-24}
REPS=${2:-3}
ARGS="-m $MODEL -ngl 99 -ncmoe $NCMOE -p 2048 -n 128 -r $REPS -o md"
OUT=/c/Users/Jorda/llama.cpp/bench-results
mkdir -p $OUT

echo "=== 1/3 baseline (ai-main) ==="
/c/Users/Jorda/llama.cpp/build-main/bin/Release/llama-bench.exe $ARGS | tee $OUT/1-baseline-ncmoe$NCMOE.md

echo "=== 2/3 prefetch branch, pinning off ==="
/c/Users/Jorda/llama.cpp-prefetch/build/bin/Release/llama-bench.exe $ARGS | tee $OUT/2-prefetch-ncmoe$NCMOE.md

echo "=== 3/3 prefetch branch + GGML_CUDA_REGISTER_HOST=1 (our Windows fix) ==="
GGML_CUDA_REGISTER_HOST=1 /c/Users/Jorda/llama.cpp-prefetch/build/bin/Release/llama-bench.exe $ARGS | tee $OUT/3-prefetch-pinned-ncmoe$NCMOE.md

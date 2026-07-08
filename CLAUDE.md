IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

# This fork

AI-contributions-welcome fork. `master` mirrors upstream (never commit to it); `ai-main` is the default branch; experiments live on `experiments/*` branches. The ranked experiment queue with measured baselines is [EXPERIMENTS.md](EXPERIMENTS.md) — read it before proposing or benchmarking anything.

# Benchmarking rules (learned the hard way on this machine)

- ONE llama process at a time. A 59 GB model load while another instance holds RAM commits-OOMs the whole box (0xC000012D) and wedges every shell.
- Never pipe a long run's stdout — Windows pipe buffering deadlocks it. Redirect to a file (`> log 2>&1`) and read the file.
- No benchmarks while builds are compiling; numbers on a loaded machine are garbage.
- llama-bench suppresses info logs: verify env-gated code paths fired with `-v` and the log line (e.g. `pinned X MiB`), AND confirm the run completed (exit code / result rows), not just that the line printed.
- `-no-cnv` was removed from llama-cli on current master — raw completion is `llama-completion`. Old branches (May base) still have llama-cli `-no-cnv`.
- Builds: VS 2022 generator, `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF`. llama-bench does NOT expose `--moe-expert-cache-size` (common/arg.cpp only).

# Known hard constraints (do not re-attempt)

- cudaMemcpyAsync source must lie within ONE cudaHostRegister'd region. Partial/chunked pinning of packed weights = "invalid argument" crash. Pinning is all-or-nothing per mmap.
- WDDM caps total pinned host memory ~36 GB on this 64 GB box; not raisable.
- Claims require artifacts: paste the llama-bench table and relevant log lines into results, never a summary of what should have happened.

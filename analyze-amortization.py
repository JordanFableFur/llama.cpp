#!/usr/bin/env python3
# GATE (a) for EXPERIMENTS.md item 17 (resident-experts self-speculation).
# The verify pass of the self-speculation scheme runs the full model over a K-token window in ONE
# batched pass. Because a batched matmul loads each *unique* expert's weights once and applies it to
# all tokens in the window that route to it, the DDR5 expert-read cost of verifying K tokens is
# proportional to the UNIQUE experts touched per layer in the window -- not K x n_expert_used.
# This script measures that unique-expert count per layer per K-token window from a routing trace,
# for K = 2,4,6,8, across all three domains. tg on this box is DDR5-bandwidth-bound, so unique-expert
# reads are the verify cost.
#
# Amortization factor = (K * n_expert_used) / unique(K):  plain decode reads K*neu experts/layer over
# K tokens; batched verify reads unique(K)/layer. Higher factor = cheaper verify per drafted token.
# verify_cost_per_token(K) [in plain-token expert-read units] = unique(K) / (K * neu).
#
# Trace format (ID-only, GGML_MOE_TRACE): int32 magic 0x54454f4d, n_expert, n_expert_used;
#   frames: int32 layer, int32 n_tokens, int32[nt*neu] ids.  (same parser as simulate-cache.py)
#
# Usage: python analyze-amortization.py <name1>=<trace1> [<name2>=<trace2> ...] [--K 2,4,6,8]

import sys, collections
import numpy as np

MAGIC = 0x54454f4d

def parse_trace(path):
    raw = np.fromfile(path, dtype=np.int32)
    assert raw[0] == MAGIC, f"bad magic in {path}: {raw[0]:#x}"
    n_expert, neu = int(raw[1]), int(raw[2])
    i, N = 3, len(raw)
    per_layer = collections.defaultdict(list)
    while i + 2 <= N:
        layer = int(raw[i]); nt = int(raw[i+1])
        if nt <= 0 or i + 2 + nt*neu > N:
            break
        per_layer[layer].append(raw[i+2:i+2+nt*neu].reshape(nt, neu))
        i += 2 + nt*neu
    layers = sorted(per_layer)
    reqs = {L: np.concatenate(per_layer[L], 0).astype(np.int32) for L in layers}
    T = min(r.shape[0] for r in reqs.values())
    return n_expert, neu, layers, T, {L: r[:T] for L, r in reqs.items()}

def unique_per_window(reqs, layers, T, K):
    # non-overlapping (tiled) windows -- the verify batch consumes K tokens then advances.
    nwin = T // K
    per_layer_mean = []
    all_counts = []
    for L in layers:
        r = reqs[L][:nwin*K].reshape(nwin, K, -1)   # [nwin, K, neu]
        counts = np.array([len(np.unique(w)) for w in r])  # unique experts in each window
        per_layer_mean.append(counts.mean())
        all_counts.append(counts)
    all_counts = np.concatenate(all_counts)
    return np.mean(per_layer_mean), all_counts, nwin

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    Ks = [2, 4, 6, 8]
    for a in sys.argv[1:]:
        if a.startswith("--K"):
            Ks = [int(x) for x in a.split("=")[1].split(",")]
    traces = {}
    for a in args:
        name, path = a.split("=", 1)
        traces[name] = parse_trace(path)

    neu = None
    for name, (n_expert, nu, layers, T, reqs) in traces.items():
        neu = nu
        print(f"{name}: {len(layers)} layers, {T} tokens, n_expert={n_expert}, n_expert_used={nu}")
    print()

    # Table: rows = K, columns per domain = unique/layer/window, unique-per-token (=unique/K),
    # amortization factor (=K*neu/unique).
    names = list(traces)
    hdr = f"{'K':>3} | " + " | ".join(f"{n:^34}" for n in names)
    print(hdr); print("-"*len(hdr))
    print(f"{'':>3} | " + " | ".join(f"{'uniq/lyr':>10} {'uniq/tok':>9} {'amort x':>11}" for _ in names))
    print("-"*len(hdr))
    results = {n: {} for n in names}
    for K in Ks:
        cells = []
        for name in names:
            n_expert, nu, layers, T, reqs = traces[name]
            mean_uniq, counts, nwin = unique_per_window(reqs, layers, T, K)
            uniq_per_tok = mean_uniq / K
            amort = (K * nu) / mean_uniq
            results[name][K] = dict(uniq=mean_uniq, uniq_per_tok=uniq_per_tok, amort=amort,
                                    nwin=nwin, sat=mean_uniq/min(K*nu, n_expert))
            cells.append(f"{mean_uniq:>10.2f} {uniq_per_tok:>9.3f} {amort:>10.2f}x")
        print(f"{K:>3} | " + " | ".join(cells))
    print()
    print("uniq/lyr = mean unique experts touched per layer per K-token window (max = min(K*neu, n_expert))")
    print("uniq/tok = uniq/lyr / K = experts loaded per layer per token in the batched verify (plain = neu)")
    print(f"amort x  = (K*neu)/uniq = expert-read reduction of batched verify vs plain (neu={neu})")
    print()
    # verify_cost per drafted token in plain-token expert-read units, for the speedup formula.
    print("verify_cost_per_token(K) = uniq/tok / neu  [1.0 = one plain token's expert-read cost per layer]:")
    hdr2 = f"{'K':>3} | " + " | ".join(f"{n:>12}" for n in names)
    print(hdr2)
    for K in Ks:
        print(f"{K:>3} | " + " | ".join(f"{results[n][K]['uniq_per_tok']/neu:>12.4f}" for n in names))

if __name__ == "__main__":
    main()

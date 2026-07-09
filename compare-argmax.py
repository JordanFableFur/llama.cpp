#!/usr/bin/env python3
# GATE (b) scoring (EXPERIMENTS.md item 17): compare draft-vs-true greedy predictions from
# llama-draftsim ('ARGX' files) and report draft-vs-true token agreement.
#   top-1 agreement      = per-position P(draft argmax == true argmax)  [= per-token acceptance alpha]
#   agreement@K          = P(K consecutive positions all agree)         [sliding windows]
#   accept(K)            = mean accepted run length per K-token draft round (min(run_from_t, K)) — the
#                          expected accepted tokens the speculative-speedup formula wants.
#   run-length dist      = maximal consecutive-agreement runs (mean, histogram).
#
# Usage: python compare-argmax.py <true.argmax> <draft.argmax> [--warm N] [--Ks 4,6,8]

import sys
import numpy as np

MAGIC = 0x58475241  # 'ARGX'

def load(path):
    raw = np.fromfile(path, dtype=np.int32)
    assert raw[0] == MAGIC, f"bad argmax magic {raw[0]:#x} in {path}"
    n = int(raw[1])
    return raw[2:2+n].copy()

def main():
    true_p, draft_p = sys.argv[1], sys.argv[2]
    warm = 0
    Ks = [4, 6, 8]
    for i, a in enumerate(sys.argv):
        if a == "--warm": warm = int(sys.argv[i+1])
        if a == "--Ks":   Ks = [int(x) for x in sys.argv[i+1].split(",")]
    t = load(true_p); d = load(draft_p)
    T = min(len(t), len(d))
    t, d = t[:T], d[:T]
    agree = (t == d)
    if warm > 0:
        agree = agree[warm:]
    n = len(agree)
    print(f"true={true_p}  draft={draft_p}")
    print(f"positions compared: {n} (warm-skip {warm}), byte-identical: {bool(agree.all())}")
    alpha = float(agree.mean())
    print(f"top-1 agreement (alpha):            {alpha*100:6.2f}%")

    # agreement@K over sliding windows
    for K in Ks:
        if n >= K:
            # all-agree in window [i, i+K)
            w = np.lib.stride_tricks.sliding_window_view(agree, K).all(axis=1)
            print(f"agreement@{K} (K consecutive all agree): {w.mean()*100:6.2f}%")

    # maximal consecutive-agreement run-length distribution
    runs = []
    cur = 0
    for a in agree:
        if a: cur += 1
        else:
            if cur: runs.append(cur)
            cur = 0
    if cur: runs.append(cur)
    runs = np.array(runs) if runs else np.array([0])
    print(f"agreement run-length: mean {runs.mean():.2f}, p50 {np.percentile(runs,50):.0f}, "
          f"p90 {np.percentile(runs,90):.0f}, max {runs.max()}")

    # accept(K): expected accepted tokens per K-draft round = mean over start positions of the number
    # of consecutive agreements from that start, capped at K (standard speculative acceptance).
    print("accept(K) = mean accepted run per K-draft round (cap K):")
    # run_from[t] = consecutive agreements starting at t
    run_from = np.zeros(n, dtype=np.int32)
    c = 0
    for i in range(n-1, -1, -1):
        c = c+1 if agree[i] else 0
        run_from[i] = c
    for K in Ks:
        acc = np.minimum(run_from, K).mean()
        print(f"  accept({K}) = {acc:6.3f}   (of {K}; efficiency {acc/K*100:.1f}%)")

if __name__ == "__main__":
    main()

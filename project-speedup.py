#!/usr/bin/env python3
# Item 17 decision input: projected self-speculation speedup vs the static champion (29.7 tg).
#   effective_tg(K) = 29.7 * accepted(K) / (draft_frac*K + verify_cost(K))
# with, in units of one champion plain-token time (1/29.7 s):
#   verify_cost(K)  = mean_unique_experts_per_layer(K) / n_expert_used     [GATE (a), per domain]
#                     (batched verify of K tokens is DDR5-bandwidth-bound; it streams unique(K)
#                      experts/layer once instead of the K*neu a plain decode of K tokens would)
#   draft_frac      = per-token draft cost as a fraction of a plain token (resident-only, all-GPU,
#                     no DDR5 expert reads); swept 0.1/0.2/0.3 since it is not separately measured
#   accepted(K)     = mean accepted tokens per K-draft round                [GATE (b), per domain]
# Ship bar: speedup = effective_tg/29.7 > 1.25 (beat the champion by >25%). The +1 verify bonus token
# (standard speculative decoding always emits one correct token per round) is shown as the optimistic
# variant. verify_cost ignores fixed per-pass and attention compute, so it is a LOWER bound -> the
# projection is biased toward GO.
import sys

CHAMPION = 29.7
NEU = 4
# GATE (a) mean unique experts / layer / K-window (from analyze-amortization.py, bench-results/item17-gate-a):
UNIQ = {
    'wiki': {2: 6.67, 4: 10.87, 6: 14.23, 8: 17.11},
    'code': {2: 6.86, 4: 11.47, 6: 15.24, 8: 18.41},
    'chat': {2: 6.92, 4: 11.57, 6: 15.34, 8: 18.55},
}

def verify_cost(domain, K):
    return UNIQ[domain][K] / NEU

def main():
    # accepted(K) per domain from GATE (b); pass as domain=K:acc,K:acc,...
    # e.g. project-speedup.py wiki=4:0.9,6:1.1,8:1.3 code=... chat=...
    accepted = {}
    for a in sys.argv[1:]:
        dom, rest = a.split("=", 1)
        accepted[dom] = {int(k): float(v) for k, v in (kv.split(":") for kv in rest.split(","))}
    if not accepted:
        print("usage: project-speedup.py wiki=4:A,6:A,8:A code=... chat=...  (A = accepted(K) from gate b)")
        return
    for dom, acc in accepted.items():
        print(f"\n===== {dom}  (champion {CHAMPION} tg; ship bar {CHAMPION*1.25:.1f} tg = speedup 1.25) =====")
        print(f"{'K':>3} {'verify_cost':>11} {'accepted':>9}   " +
              "  ".join(f"f={f:.1f}:tg/spd" for f in (0.1, 0.2, 0.3)) +
              "   |  +1bonus f=0.2")
        for K in sorted(acc):
            vc = verify_cost(dom, K)
            cells = []
            for f in (0.1, 0.2, 0.3):
                tg = CHAMPION * acc[K] / (f*K + vc)
                cells.append(f"{tg:5.1f}/{tg/CHAMPION:4.2f}")
            tgb = CHAMPION * (acc[K] + 1) / (0.2*K + vc)
            print(f"{K:>3} {vc:>11.3f} {acc[K]:>9.3f}   " + "   ".join(cells) +
                  f"   |  {tgb:5.1f}/{tgb/CHAMPION:4.2f}")

if __name__ == "__main__":
    main()

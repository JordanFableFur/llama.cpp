#!/usr/bin/env python3
# Measure MoE routing skew from an imatrix GGUF (per-expert selection counts).
# Decides REAP-pruning viability: REAP prunes per-layer, so per-layer skew is
# what matters, not aggregate. Usage: python analyze-routing-skew.py <imatrix.gguf>
import sys, numpy as np
from gguf import GGUFReader

def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return 0.0 if x.sum() == 0 else (2*np.sum(np.arange(1, n+1)*x)/(n*x.sum())) - (n+1)/n

def load(path):
    tb = {t.name: t for t in GGUFReader(path).tensors}
    gate = sorted([n for n in tb if n.endswith("ffn_gate_exps.weight.counts")],
                  key=lambda s: int(s.split('.')[1]))
    counts = np.vstack([np.asarray(tb[n].data, float).reshape(-1) for n in gate])  # (layers, experts)
    return tb, gate, counts

def report(path):
    tb, gate, A = load(path)
    print(f"\n### {path}  ({A.shape[0]} layers x {A.shape[1]} experts)")
    cvs = [c.std()/c.mean() for c in A]; gs = [gini(c) for c in A]
    print(f"per-layer CV   median {np.median(cvs):.2f} (max {max(cvs):.2f})")
    print(f"per-layer gini median {np.median(gs):.2f} (max {max(gs):.2f})")
    agg = A.sum(0)
    print(f"aggregate gini {gini(agg):.3f}  active {int((agg>0).sum())}/{A.shape[1]}")
    return A

if __name__ == "__main__":
    paths = sys.argv[1:] or ["bench-results/gptoss.imatrix.gguf"]
    mats = [report(p) for p in paths]
    if len(mats) == 2:
        # cross-domain overlap: are the per-layer bottom-k experts the SAME?
        A, B = mats
        k = 32  # bottom-25%
        overlaps = []
        for L in range(A.shape[0]):
            botA = set(np.argsort(A[L])[:k]); botB = set(np.argsort(B[L])[:k])
            overlaps.append(len(botA & botB)/k)
        ov = np.array(overlaps)
        print(f"\n=== CROSS-DOMAIN bottom-{k} expert overlap (per layer) ===")
        print(f"mean overlap {ov.mean()*100:.1f}%  (100% = same experts idle in both; "
              f"{k/A.shape[1]*100:.0f}% = random)")
        print(f"VERDICT: {'consistent -> prunable (REAP viable)' if ov.mean() > 0.6 else 'domain-specific -> union flat, prunability shrinks'}")

#!/usr/bin/env python3
# P3.1 mispredict analysis (SLOT-CACHE-DESIGN.md, overnight decision by Jordan).
# Speculative binary-dispatch elision predicts a layer fully-resident from the boundary-published
# LRU map, then this token may route to a non-resident expert (an "escape"). This script prices the
# damage from existing routing, using a WEIGHTED trace (ids + final router weights) so it can report
# the DROPPED WEIGHT-MASS, not just the escape rate -- dropping a 0.08 4th expert and a 0.45 top
# expert are different universes.
#
# GATE (Jordan): stop before building if mean dropped weight-mass > ~1% OR top-expert-escape not rare.
#
# Weighted trace format (common/moe-trace.cpp, GGML_MOE_TRACEW):
#   header: int32 magic 'MOEW' (0x57454f4d), int32 n_expert, int32 n_expert_used
#   frames: int32 layer, int32 n_tokens, int32[nt*neu] ids, float32[nt*neu] weights  (row-major r*neu+k)
#
# Usage: python analyze-mispredict.py <trace.tracew> [S1 S2 ...]   (default S = 32 48 64)

import sys, collections
import numpy as np

MAGIC = 0x57454f4d

def parse(path):
    raw = np.fromfile(path, dtype=np.uint8)
    w = raw.view(np.int32)
    assert w[0] == MAGIC, f"bad magic {w[0]:#x} in {path} (expected weighted 'MOEW')"
    n_expert, neu = int(w[1]), int(w[2])
    ids_by_layer = collections.defaultdict(list)
    wt_by_layer  = collections.defaultdict(list)
    off = 3  # int32 words
    N = len(w)
    while off + 2 <= N:
        layer = int(w[off]); nt = int(w[off+1])
        if nt <= 0 or off + 2 + 2*nt*neu > N:
            break
        ids = w[off+2 : off+2+nt*neu].reshape(nt, neu).copy()
        wf  = raw.view(np.float32)[off+2+nt*neu : off+2+2*nt*neu].reshape(nt, neu).copy()
        ids_by_layer[layer].append(ids)
        wt_by_layer[layer].append(wf)
        off += 2 + 2*nt*neu
    layers = sorted(ids_by_layer)
    ids = {L: np.concatenate(ids_by_layer[L], 0) for L in layers}
    wts = {L: np.concatenate(wt_by_layer[L], 0)  for L in layers}
    T = min(v.shape[0] for v in ids.values())
    ids = {L: v[:T] for L, v in ids.items()}
    wts = {L: v[:T] for L, v in wts.items()}
    return n_expert, neu, layers, T, ids, wts

class LRU:
    __slots__ = ("cap", "order", "present")
    def __init__(self, cap):
        self.cap = cap
        self.order = collections.deque()   # most-recent right
        self.present = set()
    def __contains__(self, e): return e in self.present
    def touch(self, e):
        if e in self.present:
            self.order.remove(e); self.order.append(e)
        else:
            if len(self.order) >= self.cap:
                old = self.order.popleft(); self.present.discard(old)
            self.order.append(e); self.present.add(e)

def analyze(path, slot_list):
    n_expert, neu, layers, T, ids, wts = parse(path)
    print(f"trace {path}: {len(layers)} layers, {T} tokens, n_expert={n_expert}, n_expert_used={neu}")

    # sanity: final weights should sum ~1 per (token, layer). If not, they are pre-norm; renormalize.
    sums = np.concatenate([wts[L].sum(1) for L in layers])
    frac_norm = float(np.mean(np.abs(sums - 1.0) < 0.05))
    print(f"weight-sum≈1 fraction: {frac_norm:.3f} (mean sum {sums.mean():.4f}); "
          f"{'using as-is' if frac_norm > 0.9 else 'RENORMALIZING (captured pre-norm)'}")
    if frac_norm <= 0.9:
        for L in layers:
            s = wts[L].sum(1, keepdims=True); s[s == 0] = 1.0
            wts[L] = wts[L] / s

    for S in slot_list:
        # per-(layer,token): boundary map = residency after t-1's promotions (LRU, cap S).
        # policy "always": elide every warm layer -> escape = route_t has an expert not resident.
        # dropped mass = sum of final weights of escaped experts (renorm error is bounded by this).
        esc_layer_tok = 0      # elided layer-tokens with >=1 escape
        elided_layer_tok = 0   # elided layer-tokens (warm)
        total_layer_tok = 0
        dropped = []           # dropped mass per elided layer-token (0 if no escape)
        dropped_when_esc = []  # dropped mass per escaping layer-token
        top1_escapes = 0       # escapes where the escaped set includes the token's top-1 (max-weight) expert
        esc_ranks = []         # rank (0=top) of each escaped expert within its token's 4
        per_token_any = np.zeros(T, dtype=bool)   # any elided layer escaped this token
        per_token_churn = np.zeros(T)             # layer-averaged miss fraction (churn signal)
        # self-heal: for each escaped (t, e), is e resident at t+1? and consecutive-escape run per layer
        healed_next = 0; escapes_total = 0
        run_len = collections.Counter()

        for L in layers:
            lru = LRU(S)
            idL = ids[L]; wL = wts[L]
            prev_route = None
            cur_run = 0
            for t in range(T):
                route = idL[t]; w = wL[t]
                warm = len(lru.order) >= min(S, n_expert)   # slots full => elidable candidate
                resident = [e in lru for e in route]
                n_miss = sum(0 if r else 1 for r in resident)
                per_token_churn[t] += n_miss
                elide = warm  # policy "always": elide all warm layers (worst case / simplest impl)
                if elide:
                    elided_layer_tok += 1
                    esc_mask = [not r for r in resident]
                    dm = float(sum(w[k] for k in range(neu) if esc_mask[k]))
                    dropped.append(dm)
                    if any(esc_mask):
                        esc_layer_tok += 1
                        dropped_when_esc.append(dm)
                        per_token_any[t] = True
                        order = np.argsort(-w)            # rank 0 = highest weight
                        rank_of = {int(order[i]): i for i in range(neu)}
                        top1 = int(order[0])
                        if esc_mask[top1]: top1_escapes += 1
                        for k in range(neu):
                            if esc_mask[k]: esc_ranks.append(rank_of[k])
                        cur_run += 1
                    else:
                        if cur_run: run_len[cur_run] += 1
                        cur_run = 0
                else:
                    if cur_run: run_len[cur_run] += 1
                    cur_run = 0
                total_layer_tok += 1
                escapes_total += n_miss
                # promote this token's routed experts (boundary) -> resident for t+1
                # check self-heal: were the escaped experts resident next token? (they will be, unless evicted)
                for k in range(neu):
                    if not resident[k]:
                        healed_next += 1   # promoted now -> resident at t+1 by construction of LRU insert
                for e in route: lru.touch(e)
                prev_route = route
            if cur_run: run_len[cur_run] += 1

        per_token_churn /= max(len(layers), 1)
        LT = max(total_layer_tok, 1)
        EL = max(elided_layer_tok, 1)
        print(f"\n===== S = {S} =====")
        print(f"elision rate (warm layer-tokens):          {elided_layer_tok/LT*100:6.2f}%")
        print(f"(a) per-layer escape rate | elided:        {esc_layer_tok/EL*100:6.2f}%   "
              f"(unconditional {esc_layer_tok/LT*100:.2f}%)")
        print(f"(b) per-token escape rate (any layer):     {per_token_any.mean()*100:6.2f}%")
        drp = np.array(dropped); dwe = np.array(dropped_when_esc)
        print(f"(c) MEAN DROPPED WEIGHT-MASS / elided-LT:  {drp.mean()*100:6.3f}%   <-- GATE metric (~1%)")
        print(f"    mean dropped mass | escaping LT:        {dwe.mean()*100:6.3f}%")
        print(f"    dropped mass p50/p90/p99 (escaping):    "
              f"{np.percentile(dwe,50)*100:.2f}% / {np.percentile(dwe,90)*100:.2f}% / {np.percentile(dwe,99)*100:.2f}%")
        er = np.array(esc_ranks)
        print(f"    TOP-1-expert escape rate | escaping LT: {top1_escapes/max(esc_layer_tok,1)*100:6.2f}%   <-- GATE (must be rare)")
        print(f"    escaped-expert rank hist (0=top..{neu-1}): "
              f"{[int((er==i).sum()) for i in range(neu)]}  (mean rank {er.mean():.2f})")
        # (d) stratify by churn decile
        hi = per_token_churn >= np.percentile(per_token_churn, 90)
        print(f"(d) high-churn decile (top 10% churn tokens, n={hi.sum()}):")
        print(f"    per-token escape rate:                  {per_token_any[hi].mean()*100:6.2f}%")
        # dropped mass on high-churn tokens: recompute mean over elided layer-tokens in those tokens
        # (approx via per_token_any already; report escape concentration)
        print(f"    low-churn decile per-token escape:      {per_token_any[per_token_churn <= np.percentile(per_token_churn,10)].mean()*100:6.2f}%")
        # self-heal
        print(f"self-heal: escaped experts promoted-for-next-token: {healed_next}/{escapes_total} "
              f"({healed_next/max(escapes_total,1)*100:.1f}% resident by t+1 by construction)")
        runs = sorted(run_len.items())
        long_runs = sum(c for r, c in run_len.items() if r >= 3)
        print(f"consecutive-escape runs >=3 (per layer):   {long_runs}  "
              f"(run-length hist {dict(list(runs)[:6])}{' ...' if len(runs)>6 else ''})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: analyze-mispredict.py <trace.tracew> [S ...]"); sys.exit(1)
    slots = [int(x) for x in sys.argv[2:]] or [32, 48, 64]
    analyze(sys.argv[1], slots)

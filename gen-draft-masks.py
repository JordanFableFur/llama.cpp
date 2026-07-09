#!/usr/bin/env python3
# GATE (b) helper (EXPERIMENTS.md item 17). Build the per-(position,layer) resident-expert masks that
# GGML_MOE_DRAFT_SIM consumes, from a TRUE routing trace (GGML_MOE_TRACE, ids-only, magic 'MOET').
# For each layer an LRU(S) is driven by the true routed experts; the mask for position t is the LRU
# residency BEFORE token t is added (the boundary map the elision would see). Layers not yet warm
# (<S distinct experts seen) emit all-resident, matching analyze-mispredict.py's warm gating and
# avoiding a <n_expert_used resident set.
#
# Mask file (consumed by llama-graph.cpp moe_draft_sim_state):
#   int32 magic 'MDSK'(0x4b53444d), int32 n_layer, int32 n_token, int32 n_expert
#   payload: for pos: for L: uint8[ceil(n_expert/8)] resident bitmask (bit e set => resident)
#
# Usage:
#   python build-draft-masks.py <true.trace> <S> <out.mask>        # LRU masks
#   python build-draft-masks.py --allres <true.trace> <out.mask>   # all-resident (validation gate)

import sys, collections
import numpy as np

TRACE_MAGIC = 0x54454f4d   # 'MOET'
MASK_MAGIC  = 0x4b53444d   # 'MDSK'

def parse_trace(path):
    raw = np.fromfile(path, dtype=np.int32)
    assert raw[0] == TRACE_MAGIC, f"bad trace magic {raw[0]:#x} in {path}"
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
    reqs = {L: r[:T] for L, r in reqs.items()}
    return n_expert, neu, layers, T, reqs

def write_mask(path, n_layer, n_token, n_expert, bitmasks):
    # bitmasks: uint8 array [n_token, n_layer, bytes_per]
    with open(path, "wb") as f:
        np.array([MASK_MAGIC, n_layer, n_token, n_expert], dtype=np.int32).tofile(f)
        bitmasks.tofile(f)

def main():
    allres = (sys.argv[1] == "--allres")
    if allres:
        trace, out = sys.argv[2], sys.argv[3]
        n_expert, neu, layers, T, reqs = parse_trace(trace)
        n_layer = max(layers) + 1
        bytes_per = (n_expert + 7) // 8
        bm = np.full((T, n_layer, bytes_per), 0xFF, dtype=np.uint8)  # every expert resident
        write_mask(out, n_layer, T, n_expert, bm)
        print(f"[allres] {out}: n_layer={n_layer} n_token={T} n_expert={n_expert} (all resident)")
        return

    trace, S, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    n_expert, neu, layers, T, reqs = parse_trace(trace)
    n_layer = max(layers) + 1
    bytes_per = (n_expert + 7) // 8
    bm = np.full((T, n_layer, bytes_per), 0xFF, dtype=np.uint8)  # default all-resident (unwarm / absent layers)

    n_masked_lt = 0
    for L in layers:
        order = collections.deque()   # MRU right
        present = set()
        r = reqs[L]
        for t in range(T):
            warm = len(order) >= min(S, n_expert)
            if warm:
                # emit residency BEFORE adding token t: clear all bits, set resident experts
                row = np.zeros(bytes_per, dtype=np.uint8)
                for e in present:
                    row[e >> 3] |= (1 << (e & 7))
                bm[t, L] = row
                n_masked_lt += 1
            # update LRU with this token's true routing
            for e in r[t]:
                e = int(e)
                if e in present:
                    order.remove(e); order.append(e)
                else:
                    if len(order) >= S:
                        old = order.popleft(); present.discard(old)
                    order.append(e); present.add(e)
    write_mask(out, n_layer, T, n_expert, bm)
    total_lt = T * len(layers)
    print(f"[LRU S={S}] {out}: n_layer={n_layer} n_token={T} n_expert={n_expert}; "
          f"masked {n_masked_lt}/{total_lt} layer-tokens ({100.0*n_masked_lt/total_lt:.1f}% warm/elided)")

if __name__ == "__main__":
    main()

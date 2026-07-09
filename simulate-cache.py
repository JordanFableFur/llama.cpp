#!/usr/bin/env python3
# Offline expert-cache simulation for the persistent GPU slot cache (EXPERIMENTS.md item 15).
# Replays a routing trace (produced by GGML_MOE_TRACE=<file>) through several cache
# policies at several slot budgets and reports per-workload hit-rate curves. Also verifies
# a trace against an imatrix GGUF (aggregate routing must match).
#
# Trace binary format (see common/moe-trace.cpp):
#   header: int32 magic 0x54454f4d, int32 n_expert, int32 n_expert_used
#   frames: int32 layer, int32 n_tokens, int32[n_tokens*n_expert_used] expert ids
#
# Usage:
#   python simulate-cache.py verify <trace> <imatrix.gguf>
#   python simulate-cache.py sim   <name1>=<trace1> [<name2>=<trace2> ...]
import sys, collections, heapq
import numpy as np

MAGIC = 0x54454f4d

def parse_trace(path):
    raw = np.fromfile(path, dtype=np.int32)
    assert raw[0] == MAGIC, f"bad magic in {path}: {raw[0]:#x}"
    n_expert, neu = int(raw[1]), int(raw[2])
    i = 3
    per_layer = collections.defaultdict(list)   # layer -> list of [nt, neu]
    N = len(raw)
    while i + 2 <= N:
        layer = int(raw[i]); nt = int(raw[i+1])
        if nt <= 0 or i + 2 + nt*neu > N:   # torn/partial final frame (e.g. killed run)
            break
        block = raw[i+2:i + 2 + nt*neu].reshape(nt, neu)
        per_layer[layer].append(block)
        i += 2 + nt*neu
    layers = sorted(per_layer)
    reqs = {L: np.concatenate(per_layer[L], axis=0).astype(np.int32) for L in layers}  # [T, neu]
    T = min(r.shape[0] for r in reqs.values())
    # align to common T (all layers see the same tokens; guard against a trailing partial frame)
    reqs = {L: r[:T] for L, r in reqs.items()}
    return n_expert, neu, layers, T, reqs

# ---------------- verification against imatrix ----------------
def verify(trace, imat):
    from gguf import GGUFReader
    n_expert, neu, layers, T, reqs = parse_trace(trace)
    print(f"trace: {len(layers)} layers, {T} tokens, n_expert={n_expert}, n_expert_used={neu}")
    tb = {t.name: t for t in GGUFReader(imat).tensors}
    gate = {int(n.split('.')[1]): np.asarray(tb[n].data, float).reshape(-1)
            for n in tb if n.endswith("ffn_gate_exps.weight.counts")}
    cors, gdiffs = [], []
    for L in layers:
        if L not in gate: continue
        tc = np.bincount(reqs[L].reshape(-1), minlength=n_expert).astype(float)
        ic = gate[L]
        c = np.corrcoef(tc, ic)[0, 1]
        cors.append(c)
        gdiffs.append(abs(_gini(tc) - _gini(ic)))
    print(f"per-layer trace-vs-imatrix count correlation: median {np.median(cors):.4f} "
          f"min {min(cors):.4f}  (1.0 = identical distribution)")
    print(f"per-layer gini abs-diff: median {np.median(gdiffs):.4f} max {max(gdiffs):.4f}")
    print("VERDICT:", "MATCH - hook is correct" if np.median(cors) > 0.98
          else "MISMATCH - investigate hook")

def _gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return 0.0 if x.sum() == 0 else (2*np.sum(np.arange(1, n+1)*x)/(n*x.sum())) - (n+1)/n

# ---------------- next-use (Belady) precompute ----------------
def next_use(seq):
    n = len(seq)
    nxt = np.full(n, n, dtype=np.int64)
    order = np.argsort(seq, kind='stable')
    s = seq[order]
    same = s[1:] == s[:-1]
    nxt[order[:-1][same]] = order[1:][same]
    return nxt

# ---------------- full-residency rate (phase-3 viability) ----------------
# P(all n_expert_used routed experts of a layer are resident at the token) under per-layer LRU(S).
# A layer is fully resident on a token => phase 3 can run pure GPU slot path, no CPU op / DtoH /
# combine. frac_partial = 1 - P(full) is the fraction of layer-tokens that still pay the split cost.
def residency_full(reqs, layers, neu, S):
    full = tot = 0
    per_layer = {}
    for L in layers:
        seq = reqs[L]                    # [T, neu]
        T = seq.shape[0]
        cache = collections.OrderedDict()
        lfull = 0
        for t in range(T):
            row = seq[t]
            resident = all((int(e) in cache) for e in row)  # BEFORE this token's promotions
            if resident:
                lfull += 1
            for e in row:
                e = int(e)
                if e in cache:
                    cache.move_to_end(e)
                else:
                    if len(cache) >= S:
                        cache.popitem(last=False)
                    cache[e] = 1
        full += lfull; tot += T
        per_layer[L] = lfull / T if T else 0.0
    return full / tot if tot else 0.0, per_layer

def residency(specs):
    workloads = {}
    for spec in specs:
        name, paths = spec.split('=', 1)
        plist = paths.split(',')
        workloads[name] = parse_merge(plist) if len(plist) > 1 else parse_trace(plist[0])
    for name, (n_expert, neu, layers, T, reqs) in workloads.items():
        print(f"\n=== full-residency P(all {neu} routed experts resident) : {name} ({T} tok) ===")
        hdr = "slots/layer | " + " | ".join(f"{s:>5}" for s in SLOTS)
        print(hdr); print("-"*len(hdr))
        row_hit = []; row_full = []
        for S in SLOTS:
            hit = sim_perlayer(reqs, layers, neu, S, 'lru')
            pf, per = residency_full(reqs, layers, neu, S)
            row_hit.append(hit); row_full.append(pf)
        print("per-expert hit  | " + " | ".join(f"{h*100:5.1f}" for h in row_hit))
        print("P(layer full)   | " + " | ".join(f"{f*100:5.1f}" for f in row_full))
        # naive independence prediction hit^neu for reference
        print("hit^neu (indep) | " + " | ".join(f"{(h**neu)*100:5.1f}" for h in row_hit))

# ---------------- per-layer independent pools (S slots each) ----------------
def sim_perlayer(reqs, layers, neu, S, policy, decay=0.98):
    hits = reqs_total = 0
    for L in layers:
        seq = reqs[L].reshape(-1)          # T*neu, expert ids in token order
        reqs_total += len(seq)
        if policy == 'oracle':
            nxt = next_use(seq)
            cache = {}                      # expert -> next-use index
            for i, e in enumerate(seq):
                e = int(e)
                if e in cache:
                    hits += 1; cache[e] = nxt[i]
                else:
                    if len(cache) >= S:
                        victim = max(cache, key=cache.get)
                        del cache[victim]
                    cache[e] = nxt[i]
        elif policy == 'lru':
            cache = collections.OrderedDict()
            for e in seq:
                e = int(e)
                if e in cache:
                    hits += 1; cache.move_to_end(e)
                else:
                    if len(cache) >= S: cache.popitem(last=False)
                    cache[e] = 1
        elif policy == 'lfu':               # decayed frequency
            score = {}
            for e in seq:
                e = int(e)
                for k in score: score[k] *= decay
                if e in score:
                    hits += 1; score[e] += 1.0
                else:
                    if len(score) >= S:
                        del score[min(score, key=score.get)]
                    score[e] = 1.0
    return hits / reqs_total

# ---------------- shared pool across all layers (S*n_layer slots) ----------------
def build_global(reqs, layers, neu, n_expert):
    T = reqs[layers[0]].shape[0]
    R = np.stack([reqs[L] for L in layers], axis=0)          # [nl, T, neu]
    seq_e = R.transpose(1, 0, 2).reshape(-1).astype(np.int64)  # t-major, layer, k
    nl = len(layers)
    seq_l = np.tile(np.repeat(np.arange(nl), neu), T)          # matching layer index (0..nl-1)
    return seq_l, seq_e, nl

def sim_shared(reqs, layers, neu, n_expert, S, policy):
    seq_l, seq_e, nl = build_global(reqs, layers, neu, n_expert)
    cap = S * nl
    item = seq_l * n_expert + seq_e
    n = len(item)
    hits = 0
    if policy == 'lru':
        cache = collections.OrderedDict()
        for it in item:
            it = int(it)
            if it in cache:
                hits += 1; cache.move_to_end(it)
            else:
                if len(cache) >= cap: cache.popitem(last=False)
                cache[it] = 1
    elif policy == 'cycle':                 # layer-cycle-aware: evict layer needed furthest ahead
        buckets = [collections.OrderedDict() for _ in range(nl)]
        size = 0
        for idx in range(n):
            it = int(item[idx]); L = int(seq_l[idx])
            b = buckets[L]
            if it in b:
                hits += 1; b.move_to_end(it)
            else:
                if size >= cap:
                    for d in range(nl-1, 0, -1):
                        vb = buckets[(L + d) % nl]
                        if vb:
                            vb.popitem(last=False); size -= 1; break
                    else:
                        b_cur = buckets[L]
                        if b_cur: b_cur.popitem(last=False); size -= 1
                b[it] = 1; size += 1
    elif policy == 'oracle':
        nxt = next_use(item)
        cache = {}                          # item -> next-use index
        heap = []                           # (-next_use, item) lazy max-heap
        for i in range(n):
            it = int(item[i])
            if it in cache:
                hits += 1; cache[it] = nxt[i]; heapq.heappush(heap, (-nxt[i], it))
            else:
                if len(cache) >= cap:
                    while heap:
                        negnu, cand = heapq.heappop(heap)
                        if cand in cache and cache[cand] == -negnu:
                            del cache[cand]; break
                cache[it] = nxt[i]; heapq.heappush(heap, (-nxt[i], it))
    return hits / n

# ---------------- driver ----------------
SLOTS = [8, 16, 24, 32, 48, 64]

def parse_merge(paths):
    # concatenate several traces (same model) into one token stream per layer
    parsed = [parse_trace(p) for p in paths]
    n_expert, neu, layers, _, _ = parsed[0]
    merged = {}
    for L in layers:
        merged[L] = np.concatenate([pr[4][L] for pr in parsed if L in pr[4]], axis=0)
    T = min(r.shape[0] for r in merged.values())
    merged = {L: r[:T] for L, r in merged.items()}
    return n_expert, neu, layers, T, merged

def sim(specs):
    workloads = {}
    for spec in specs:
        name, paths = spec.split('=', 1)
        plist = paths.split(',')
        workloads[name] = parse_merge(plist) if len(plist) > 1 else parse_trace(plist[0])
    # report identical n_expert/neu
    for name, (n_expert, neu, layers, T, reqs) in workloads.items():
        print(f"\n# workload '{name}': {len(layers)} layers, {T} tokens, "
              f"n_expert={n_expert}, n_expert_used={neu}, requests/layer={T*neu}")

    for name, (n_expert, neu, layers, T, reqs) in workloads.items():
        print(f"\n=== workload: {name} ===")
        hdr = "slots/layer | " + " | ".join(f"{s:>3}" for s in SLOTS)
        print(hdr); print("-"*len(hdr))
        rows = [
            ("perlayer LRU",    lambda S: sim_perlayer(reqs, layers, neu, S, 'lru')),
            ("perlayer LFU-dec", lambda S: sim_perlayer(reqs, layers, neu, S, 'lfu')),
            ("perlayer ORACLE", lambda S: sim_perlayer(reqs, layers, neu, S, 'oracle')),
            ("shared   LRU",    lambda S: sim_shared(reqs, layers, neu, n_expert, S, 'lru')),
            ("shared   CYCLE",  lambda S: sim_shared(reqs, layers, neu, n_expert, S, 'cycle')),
            ("shared   ORACLE", lambda S: sim_shared(reqs, layers, neu, n_expert, S, 'oracle')),
        ]
        for label, fn in rows:
            cells = []
            for S in SLOTS:
                cells.append(f"{fn(S)*100:5.1f}")
            print(f"{label:>16} | " + " | ".join(f"{c:>3}" for c in cells))
            sys.stdout.flush()

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("verify", "sim", "residency"):
        print(__doc__); sys.exit(1)
    if sys.argv[1] == "verify":
        verify(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "residency":
        residency(sys.argv[2:])
    else:
        sim(sys.argv[2:])

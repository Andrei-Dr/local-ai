"""Ceiling check: Belady-optimal hit rate (perfect foresight, instant free swaps) vs the runtime policy and cheaper-admission variants."""
import sys, collections, heapq, numpy as np
sys.path.insert(0, ".")
from sim import load
from warmup import Cache
trace, slots = sys.argv[1], int(sys.argv[2])
layers = load(trace); ids = sorted(layers)[::5]; n = min(len(layers[l]) for l in ids); cut = n // 2
res = collections.Counter()
for l in ids:
    seq = layers[l][:n]
    # next-use index per (t, j)
    nxt = np.full(seq.shape, 10**9, dtype=np.int64); last = {}
    for t in range(n - 1, -1, -1):
        for j, e in enumerate(seq[t].tolist()):
            nxt[t, j] = last.get(e, 10**9); last[e] = t
    cache = {}  # expert -> next use
    for t in range(n):
        for j, e in enumerate(seq[t].tolist()):
            hit = e in cache
            if t >= cut: res["bel_hit"] += hit; res["lk"] += 1
            nu = int(nxt[t, j])
            if hit or len(cache) < slots: cache[e] = nu
            else:
                v = max(cache, key=cache.get)
                if cache[v] > nu: del cache[v]; cache[e] = nu; res["bel_up"] += t >= cut
    for name, (adm, ins) in {"runtime a3/i2": (3, 2), "a2/i2": (2, 2), "a2/i4": (2, 4), "a1/i2 (ungated)": (1, 2), "a1/i8": (1, 8)}.items():
        c = Cache(slots, 16, adm, ins); up0 = 0
        for t in range(cut - 2000, n):
            if t == cut: up0 = c.uploads
            h = c.step(seq[t])
            if t >= cut: res[name + "_hit"] += h
        res[name + "_up"] += c.uploads - up0
lk = res["lk"]; tok = lk / 8 / len(ids)
print(f"{trace.split('/')[-1]} | {slots} slots | hit % | uploads per token per layer")
print(f"  Belady optimal (foresight, free swaps): {100*res['bel_hit']/lk:5.1f} | {res['bel_up']/tok/len(ids):.2f}")
for name in ("runtime a3/i2", "a2/i2", "a2/i4", "a1/i2 (ungated)", "a1/i8"):
    print(f"  {name:38s}: {100*res[name+'_hit']/lk:5.1f} | {res[name+'_up']/tok/len(ids):.2f}")

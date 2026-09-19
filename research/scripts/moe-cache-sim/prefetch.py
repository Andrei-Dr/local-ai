#!/usr/bin/env python3
"""P2 go/no-go beyond recall: does token-keyed expert prefetch raise the hit rate of the REAL cache policy?

usage: prefetch.py TRACE.bin --slots 30 [--shift 2] [--top 4,8,16] [--uploads 2,4,8] [--layer-step 5]

predict.py shows recall of a token-conditioned expert set, but the runtime cannot swap 30 slots per token: uploads
are asynchronous, budgeted, and a slot only becomes visible two steps after it was scheduled (scheduled at the end of
step t, copied during t+1, published at the end of t+1). So the usable key for the experts of token t is the token at
t-SHIFT with SHIFT >= 2 when stepping one token at a time (a draft round covers ~2.7 tokens and knows its draft
tokens one round early, which is about the same horizon).

Baseline = the runtime policy (free slots fill on first miss, gated LRU admission A misses / W tokens, I inserts per
step). Prefetch = baseline + after every step, look up the just-seen token in a table
(layer, token) -> experts used SHIFT tokens later (learned on the first half of the trace), and schedule up to U
uploads of the top-P predicted experts that are not cached, evicting the least recently used slot that is not itself
predicted. Reports hit rate on the second half and uploads per token per layer (PCIe cost).
"""
import argparse
import collections

import numpy as np

from sim import load
from warmup import Cache


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--slots", type=int, required=True)
    ap.add_argument("--shift", type=int, default=2)
    ap.add_argument("--top", default="4,8,16")
    ap.add_argument("--uploads", default="2,4,8")
    ap.add_argument("--train", type=float, default=0.5)
    ap.add_argument("--layer-step", type=int, default=5)
    ap.add_argument("--protect", type=int, default=0, help="never evict an expert used within the last N tokens (the runtime's admission window is 16)")
    a = ap.parse_args()

    layers = load(a.trace)
    toks = np.fromfile(a.trace + ".tok", dtype="<i4")
    ids = sorted(layers)[:: a.layer_step]
    n = min(min(len(layers[l]) for l in ids), len(toks))
    cut, S = int(n * a.train), a.shift
    k = layers[ids[0]].shape[1]
    print(f"{len(ids)} layers (every {a.layer_step}), {n} tokens (train {cut}), top-{k}, {a.slots} slots, key = token at t-{S}")

    configs = [(0, 0)] + [(p, u) for p in map(int, a.top.split(",")) for u in map(int, a.uploads.split(","))]
    res = {c: [0, 0, 0] for c in configs}  # hits, lookups, prefetch uploads
    for l in ids:
        seq = layers[l]
        table = collections.defaultdict(collections.Counter)
        for t in range(S, cut):
            table[int(toks[t - S])].update(seq[t].tolist())
        ranked = {tok: [e for e, _ in c.most_common(max(map(int, a.top.split(","))))] for tok, c in table.items()}
        for (P, U) in configs:
            c = Cache(a.slots, 16, 3, 2)
            last = {}
            for t in range(cut - 2000, n):  # 2000 tokens of warm-up before scoring
                h = c.step(seq[t])
                for e in seq[t].tolist():
                    last[e] = t
                if t >= cut:
                    res[(P, U)][0] += h
                    res[(P, U)][1] += k
                if not P:
                    continue
                pred = ranked.get(int(toks[t]), [])[:P]
                keep, flying, sched = set(pred), {e for _, e in c.inflight}, 0
                for e in pred:
                    if sched >= U:
                        break
                    if e in c.live or e in flying:
                        continue
                    if len(c.live) + len(c.inflight) >= c.slots:
                        victim = next((v for v in c.live if v not in keep and last.get(v, -1) < t - a.protect), None)  # LRU order
                        if victim is None:
                            break
                        del c.live[victim]
                    c.inflight.append((c.t + 2, e))
                    sched += 1
                if t >= cut:
                    res[(P, U)][2] += sched
    base = res[(0, 0)]
    print(f"baseline (runtime policy): hit {100 * base[0] / base[1]:.1f}%")
    print(f"{'top-P':>6} {'U/step':>7} | {'hit %':>6} {'delta':>6} | prefetch uploads per token per layer")
    for (P, U) in configs[1:]:
        h, lk, up = res[(P, U)]
        print(f"{P:>6} {U:>7} | {100 * h / lk:6.1f} {100 * (h / lk - base[0] / base[1]):+6.1f} | {up / (lk / k):.2f}")


if __name__ == "__main__":
    main()

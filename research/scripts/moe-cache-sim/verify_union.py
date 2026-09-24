#!/usr/bin/env python3
"""verify_union.py — cost of a speculative verify batch in CPU experts, from a routing trace
usage: verify_union.py TRACE.bin [--slots 21]
For K consecutive tokens verified together: distinct experts per layer (no cache) and misses per check against a per-layer
LRU of --slots experts, stepping through the trace in K-token checks.
"""
import argparse, sys
from collections import OrderedDict
import numpy as np
import sim


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('trace'); ap.add_argument('--slots', type=int, default=21)
    a = ap.parse_args()
    layers = list(sim.load(a.trace).values()); n = min(len(v) for v in layers); k = layers[0].shape[1]
    print(f'{a.trace}: {len(layers)} layers, {n} tokens, top-{k}, LRU {a.slots} slots/layer')
    print(' K  distinct/layer  x1tok  misses/check  misses/token')
    for K in (1, 2, 3, 4, 6, 8, 12, 16):
        dist, miss = [], []
        for seq in layers:
            s = seq[:n - n % K].reshape(-1, K * k)
            dist.append(np.mean([len(set(r)) for r in s[::7]]))
            lru, m, steps = OrderedDict(), 0, 0
            for row in s:
                steps += 1
                for e in set(row.tolist()):
                    if e in lru: lru.move_to_end(e)
                    else:
                        m += 1; lru[e] = 1
                        if len(lru) > a.slots: lru.popitem(last=False)
            miss.append(m / steps)
        d, mm = np.mean(dist), np.mean(miss)
        print(f'{K:2d}  {d:14.1f}  {d / k:5.2f}  {mm:12.1f}  {mm / K:12.2f}')


if __name__ == '__main__':
    sys.exit(main())

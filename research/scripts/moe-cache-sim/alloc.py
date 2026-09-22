#!/usr/bin/env python3
"""Non-uniform slots per layer: does giving each MoE layer its own slot count (same TOTAL = same VRAM) beat uniform?

usage: alloc.py FIT.bin [TEST.bin ...] [--uniform 26] [--tokens 12000] [--grid 12,16,20,...] [--step 2]

Per layer, the gate-lru hit curve h_l(s) is measured on FIT's first half (sim.py's policy: admit after 3 misses in 16
tokens, evict LRU; same caveats as sim.py: instant admission, one step per token). A greedy marginal allocator then
hands out the uniform total (uniform x layers) in --step increments to the layer whose curve gains the most hits,
reading between grid points linearly (the curves are close to concave). The allocation is scored on FIT's second half
and on every TEST trace (a different workload) against uniform, so a gain that only fits its own data shows up.
Lossless by construction: same experts, same kernels, only which layer holds how many cache slots."""
import argparse, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim import load, simulate  # noqa: E402

W, A = 16, 3


def curve(args):
    seq, grid = args
    return [simulate(seq, s, "gate-lru", 0, W, A)[:2] for s in grid]


def hits_at(seq_by_layer, alloc):
    jobs = [(seq_by_layer[l], [alloc[l]]) for l in sorted(seq_by_layer)]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(curve, jobs))
    h = sum(r[0][0] for r in res)
    n = sum(r[0][1] for r in res)
    return h / n


def interp(grid, vals, s):
    return float(np.interp(s, grid, vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fit"); ap.add_argument("test", nargs="*")
    ap.add_argument("--uniform", type=int, default=26); ap.add_argument("--tokens", type=int, default=12000)
    ap.add_argument("--grid", default="8,12,16,20,24,26,28,32,36,40,48,56"); ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--min", type=int, default=8)
    a = ap.parse_args()
    grid = [int(x) for x in a.grid.split(",")]
    full = {l: s[: a.tokens] for l, s in load(a.fit).items()}
    half = a.tokens // 2
    fit = {l: s[:half] for l, s in full.items()}
    hold = {l: s[half:] for l, s in full.items()}
    layers = sorted(fit)
    with ProcessPoolExecutor() as ex:
        curves = dict(zip(layers, ex.map(curve, [(fit[l], grid) for l in layers])))
    rate = {l: [h / n for h, n in curves[l]] for l in layers}
    total = a.uniform * len(layers)
    alloc = {l: a.min for l in layers}
    left = total - a.min * len(layers)
    while left >= a.step:
        best, gain = None, -1.0
        for l in layers:
            s = alloc[l]
            if s + a.step > grid[-1]:
                continue
            g = interp(grid, rate[l], s + a.step) - interp(grid, rate[l], s)
            if g > gain:
                best, gain = l, g
        alloc[best] += a.step
        left -= a.step
    uni = {l: a.uniform for l in layers}
    print("layer slots (fit on %s first %d tokens, total %d = %d x %d):" % (Path(a.fit).name, half, total, a.uniform, len(layers)))
    print("  " + " ".join("%d:%d" % (l, alloc[l]) for l in layers))
    print("  min %d max %d | hit at uniform %d per layer, by layer: %s" % (
        min(alloc.values()), max(alloc.values()), a.uniform,
        " ".join("%.0f" % (100 * interp(grid, rate[l], a.uniform)) for l in layers)))
    rows = [("fit half", fit), ("held-out half", hold)]
    for t in a.test:
        rows.append((Path(t).name, {l: s[: a.tokens] for l, s in load(t).items()}))
    for name, data in rows:
        hu, ha = hits_at(data, uni), hits_at(data, alloc)
        print("%-24s uniform %5.1f%% | per-layer %5.1f%% | %+5.2f pts (misses %+.1f%%)" % (
            name, 100 * hu, 100 * ha, 100 * (ha - hu), 100 * ((1 - ha) / (1 - hu) - 1)))


if __name__ == "__main__":
    main()

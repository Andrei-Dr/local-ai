#!/usr/bin/env python3
"""policies.py TRACE.bin [--slots 26] [--tokens 12000] — hit rate and uploads/token of the cache policies in sim.py at one slot
count, all layers pooled: the runtime policy (gate-lru, admit 3 in 16) vs plain LRU, decayed LFU at several half-lives and other
gate settings. Same caveats as sim.py (instant admission, one step per token); read the DIFFERENCES between rows."""
import argparse, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim import load, simulate  # noqa: E402

ARMS = [("gate-lru", 0, 16, 3), ("lru", 0, 16, 1), ("lfu", 32, 16, 1), ("lfu", 64, 16, 1), ("lfu", 128, 16, 1),
        ("lfu", 256, 16, 1), ("gate-lru", 0, 32, 3), ("gate-lru", 0, 16, 2), ("gate-lru", 0, 8, 2), ("gate-lru", 0, 32, 4)]
D, SLOTS, TOK = None, 26, 12000


def init(path, slots, tok):
    global D, SLOTS, TOK
    D, SLOTS, TOK = {l: s[:tok] for l, s in load(path).items()}, slots, tok


def run(a):
    pol, hl, w, ad = a
    h = n = u = 0
    for s in D.values():
        hh, nn, uu = simulate(s, SLOTS, pol, hl, w, ad); h += hh; n += nn; u += uu
    return a, h / n, u / TOK


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("trace"); ap.add_argument("--slots", type=int, default=26)
    ap.add_argument("--tokens", type=int, default=12000); a = ap.parse_args()
    with ProcessPoolExecutor(initializer=init, initargs=(a.trace, a.slots, a.tokens)) as ex:
        for arm, hit, up in ex.map(run, ARMS):
            print(f"  {arm[0]:9s} half-life {arm[1]:4d} window {arm[2]:3d} admit {arm[3]} | hit {100 * hit:5.1f}% | uploads/token {up:6.2f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""lfu_variants.py — the LFU forms policies.py did not cover (Andrei 09-23: "why not pure undecayed LFU?"), at 26 slots, all
layers pooled, 12k tokens of the q36 code / prose traces, next to gate-lru and LRU:
  lfu-pure    in-cache frequency, no decay (sim.py lfu with half-life 0): counts start at 1 on admission, so a newcomer is the
              likeliest victim
  lfu-hl256   the decayed LFU policies.py measured (half-life 256 tokens), for reference
  glfu        global LFU: counts over the whole history, surviving eviction (the textbook "perfect LFU"); admit every miss
  glfu-gated  global LFU that admits a miss only when its count beats the coldest cached count (TinyLFU-style admission)"""
import sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim import load, simulate
from concurrent.futures import ProcessPoolExecutor

def glfu(seq, slots, gate):
    """global LFU: counts over the whole history (survive eviction); admit every miss (gate=False) or only when the
    newcomer's count beats the coldest cached count (gate=True, TinyLFU-style)."""
    cnt = collections.Counter(); cache = set(); h = n = u = 0
    for row in seq:
        for e in map(int, row):
            n += 1; cnt[e] += 1
            if e in cache: h += 1; continue
            if len(cache) >= slots:
                v = min(cache, key=cnt.__getitem__)
                if gate and cnt[e] <= cnt[v]: continue
                cache.discard(v)
            cache.add(e); u += 1
    return h, n, u

def run(args):
    path, pol, tok = args
    D = {l: s[:tok] for l, s in load(path).items()}
    h = n = u = 0
    for s in D.values():
        if pol == "gate-lru": r = simulate(s, 26, "gate-lru", 0, 16, 3)
        elif pol == "lru": r = simulate(s, 26, "lru", 0, 16, 1)
        elif pol == "lfu-pure": r = simulate(s, 26, "lfu", 0, 16, 1)
        elif pol == "lfu-hl256": r = simulate(s, 26, "lfu", 256, 16, 1)
        elif pol == "glfu": r = glfu(s, 26, False)
        else: r = glfu(s, 26, True)
        h += r[0]; n += r[1]; u += r[2]
    return path.split("/")[-1], pol, h / n, u / tok

ROOT = Path(__file__).resolve().parents[3]


def main():
    jobs = [(str(ROOT / f"bench/box/traces/q36_{c}_tok.bin"), p, 12000) for c in ("code", "prose")
            for p in ("gate-lru", "lru", "lfu-pure", "lfu-hl256", "glfu", "glfu-gated")]
    with ProcessPoolExecutor(8) as ex:
        for f, p, hr, up in ex.map(run, jobs):
            print(f"{f:18s} {p:11s} hit {100*hr:5.1f}%  uploads/token {up:7.1f}", flush=True)


if __name__ == "__main__":
    main()

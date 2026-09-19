#!/usr/bin/env python3
"""Replay a llama-moe-trace .bin through device-side expert-cache policies.

usage: sim.py TRACE.bin --experts 256 --expert-mb 1.02 [--slots 8,16,32,64] [--vram-mb 600]

Trace format (llama-moe-trace): repeated records of
    int32 layer, int32 n_tokens, int32 k, then n_tokens*k int32 expert ids
one record per MoE layer per decoded chunk. Tokens are replayed in order per layer.

Policies (per-layer slot pools, as in ggml-org/llama.cpp#27861):
    lru        admit every miss, evict least recently used
    lfu        admit every miss, evict lowest decayed frequency (half-life = --half-life tokens)
    gate-lru   admit an expert only after A misses within the last W tokens ("window 16, admit 3"), evict LRU
Misses are assumed computed on the CPU (never stall on a fetch); an admission = one async upload.

Reports, per slots-per-layer: hit rate, uploads per token, upload MB per token, and the share of
expert bytes per token that still has to be read from DDR4 (the quantity that bounds decode speed).
"""
import argparse
import collections
import struct
import sys

import numpy as np


def load(path):
    per_layer = collections.defaultdict(list)
    with open(path, "rb") as f:
        data = f.read()
    off = 0
    while off + 12 <= len(data):
        layer, n_tok, k = struct.unpack_from("<iii", data, off)
        off += 12
        n = n_tok * k
        ids = np.frombuffer(data, dtype="<i4", count=n, offset=off).reshape(n_tok, k)
        off += 4 * n
        per_layer[layer].append(ids)
    return {l: np.concatenate(v) for l, v in sorted(per_layer.items())}


def simulate(seq, slots, policy, half_life, window, admit):
    """seq: (n_tokens, k) expert ids for one layer. Returns (hits, lookups, uploads)."""
    cache = collections.OrderedDict()  # expert -> decayed frequency; order = recency
    recent_miss = collections.defaultdict(collections.deque)  # expert -> token indexes of recent misses
    decay = 0.5 ** (1.0 / half_life) if half_life > 0 else 1.0
    hits = lookups = uploads = 0
    for t, row in enumerate(seq):
        if policy == "lfu" and decay < 1.0:
            for e in cache:
                cache[e] *= decay
        for e in map(int, row):
            lookups += 1
            if e in cache:
                hits += 1
                cache[e] += 1.0
                cache.move_to_end(e)
                continue
            if policy == "gate-lru":
                q = recent_miss[e]
                q.append(t)
                while q and q[0] <= t - window:
                    q.popleft()
                if len(q) < admit:
                    continue
                q.clear()
            if len(cache) >= slots:
                if policy == "lfu":
                    victim = min(cache, key=cache.get)
                    del cache[victim]
                else:
                    cache.popitem(last=False)
            cache[e] = 1.0
            uploads += 1
    return hits, lookups, uploads


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--experts", type=int, required=True, help="experts per layer")
    ap.add_argument("--expert-mb", type=float, required=True, help="MiB per expert (gate+up+down)")
    ap.add_argument("--slots", default="4,8,16,32,64,128", help="slots per layer to sweep")
    ap.add_argument("--vram-mb", type=float, default=0, help="spare VRAM; marks which slot counts fit")
    ap.add_argument("--half-life", type=float, default=64, help="LFU decay half-life in tokens")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--admit", type=int, default=3)
    ap.add_argument("--skip", type=int, default=0, help="ignore the first N tokens per layer (warm-up)")
    a = ap.parse_args()

    layers = load(a.trace)
    if not layers:
        sys.exit("empty trace")
    n_layers = len(layers)
    n_tok = min(len(v) for v in layers.values())
    k = next(iter(layers.values())).shape[1]
    print(f"trace: {n_layers} MoE layers, {n_tok} tokens, top-{k} of {a.experts} experts, {a.expert_mb:.2f} MiB/expert")
    print(f"expert bytes read per token with no cache: {n_layers * k * a.expert_mb:.0f} MiB")

    # sanity: the README's failure signature is every id appearing exactly k times with sub-chance reuse
    first = next(iter(layers.values()))
    reuse = np.mean([len(set(first[i]) & set(first[i - 1])) for i in range(1, len(first))]) / k
    print(f"adjacent-token reuse, first MoE layer: {reuse:.3f} (chance = {k / a.experts:.3f})")
    counts = np.bincount(first.ravel(), minlength=a.experts)
    top = np.sort(counts)[::-1]
    print(f"traffic concentration, first MoE layer: top 10% of experts carry {top[: max(1, a.experts // 10)].sum() / counts.sum():.1%}\n")

    print(f"{'slots/layer':>11} {'cache MiB':>9} {'fits':>4} | " + " | ".join(f"{p:^27}" for p in ("lru", "lfu", f"gate-lru w{a.window} a{a.admit}")))
    print(f"{'':>11} {'':>9} {'':>4} | " + " | ".join(f"{'hit%':>6} {'upl/tok':>8} {'MiB/tok':>9}" for _ in range(3)))
    for s in map(int, a.slots.split(",")):
        if s >= a.experts:
            continue
        mb = s * n_layers * a.expert_mb
        fits = "" if not a.vram_mb else ("yes" if mb <= a.vram_mb else "no")
        cells = []
        for pol in ("lru", "lfu", "gate-lru"):
            h = lk = up = 0
            for seq in layers.values():
                seq = seq[a.skip : n_tok]
                r = simulate(seq, s, pol, a.half_life, a.window, a.admit)
                h, lk, up = h + r[0], lk + r[1], up + r[2]
            toks = n_tok - a.skip
            cells.append(f"{100 * h / lk:6.1f} {up / toks:8.2f} {up / toks * a.expert_mb:9.1f}")
        print(f"{s:>11} {mb:>9.0f} {fits:>4} | " + " | ".join(cells))
    print("\nread as: DDR4 expert bytes per token scale with (1 - hit%); uploads compete with that same bus at PCIe rate.")


if __name__ == "__main__":
    main()

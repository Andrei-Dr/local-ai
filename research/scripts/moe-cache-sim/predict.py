#!/usr/bin/env python3
"""How predictable is MoE routing from token identity? (the premise of draft-driven expert prefetch)

usage: predict.py TRACE.bin [--train 0.5] [--budget 8,16,32]

Needs TRACE.bin.tok (int32 token ids, written by the patched llama-moe-trace). For every layer, learns on the first
part of the trace a table  key -> expert counts  for two keys: the token itself ("unigram": what a drafted token gives
us for free) and (previous token, token) ("bigram"), then on the rest measures RECALL@B: the share of the experts a
token really used that are inside the B experts the table would have prefetched. Baselines at the same budget:
"recent" = the B most recently used experts of that layer (what an LRU already holds), "global" = the B most popular
experts. Reports per layer band, because routing in early layers tracks the token and later layers track context.
"""
import argparse
import collections
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
import sim  # noqa: E402  (trace loader)


def topb(counter, b):
    return [e for e, _ in counter.most_common(b)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--train", type=float, default=0.5)
    ap.add_argument("--budget", default="8,16,32")
    ap.add_argument("--shift", type=int, default=0, help="predict token t from the tokens up to t-SHIFT: 0 = the token itself (a drafted token, known only when it is verified), 1 = lookahead (what a prefetch that must land a round early can use)")
    a = ap.parse_args()

    layers = sim.load(a.trace)
    toks = np.fromfile(a.trace + ".tok", dtype="<i4")
    n = min(min(len(v) for v in layers.values()), len(toks))
    cut = int(n * a.train)
    budgets = [int(x) for x in a.budget.split(",")]
    k = next(iter(layers.values())).shape[1]
    seen = len(set(toks[:cut].tolist()))
    cover = np.mean([t in set(toks[:cut].tolist()) for t in toks[cut:n]])
    print(f"{len(layers)} layers, {n} tokens (train {cut}), top-{k}; {seen} distinct train tokens, {100 * cover:.1f}% of test tokens seen in training")

    rows = []
    for li, (layer, seq) in enumerate(layers.items()):
        uni, bi, glob = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter), collections.Counter()
        S = a.shift
        for t in range(S + 1, cut):
            ex = seq[t].tolist()
            uni[int(toks[t - S])].update(ex)
            bi[(int(toks[t - S - 1]), int(toks[t - S]))].update(ex)
            glob.update(ex)
        rec = {(m, b): 0 for m in ("unigram", "bigram", "recent", "global") for b in budgets}
        recent = collections.OrderedDict()
        for t in range(1, cut):  # warm the recency list
            for e in seq[t].tolist():
                recent.pop(e, None); recent[e] = 1
        for t in range(cut, n):
            actual = set(seq[t].tolist())
            u, g2 = uni.get(int(toks[t - S])), bi.get((int(toks[t - S - 1]), int(toks[t - S])))
            rlist = list(recent.keys())[::-1]
            for b in budgets:
                gb = topb(glob, b)
                pu = (topb(u, b) if u else []); pu += [e for e in gb if e not in pu][: b - len(pu)]
                pb = (topb(g2, b) if g2 else []); pb += [e for e in pu if e not in pb][: b - len(pb)]
                rec[("unigram", b)] += len(actual & set(pu)); rec[("bigram", b)] += len(actual & set(pb))
                rec[("recent", b)] += len(actual & set(rlist[:b])); rec[("global", b)] += len(actual & set(gb))
            for e in seq[t].tolist():
                recent.pop(e, None); recent[e] = 1
        rows.append((layer, {key: v / ((n - cut) * k) for key, v in rec.items()}))

    bands = [("first quarter", rows[: len(rows) // 4]), ("middle half", rows[len(rows) // 4: 3 * len(rows) // 4]), ("last quarter", rows[3 * len(rows) // 4:]), ("all layers", rows)]
    print(f"\nshift {a.shift}: recall@B of a token's real experts (chance = B/n_expert)\n{'layers':>14} {'B':>3} | {'unigram':>8} {'bigram':>8} {'recent(LRU)':>12} {'global':>8}")
    for name, rs in bands:
        if not rs:
            continue
        for b in budgets:
            m = lambda key: 100 * np.mean([r[1][(key, b)] for r in rs])
            print(f"{name:>14} {b:>3} | {m('unigram'):7.1f}% {m('bigram'):7.1f}% {m('recent'):11.1f}% {m('global'):7.1f}%")
    print("\nread as: prefetch only pays where unigram/bigram recall clearly beats 'recent' at the same budget - that is the\nexperts an LRU would NOT already hold. Expect that in the early layers, if anywhere.")


if __name__ == "__main__":
    main()

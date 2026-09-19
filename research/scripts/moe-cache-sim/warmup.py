#!/usr/bin/env python3
"""P4c go/no-go: what does warming the expert cache from the PROMPT's routing buy at the start of decode?

usage: warmup.py CODE.bin PROSE.bin --slots 30 [--prompt 64,256,1024] [--gen 200] [--layer-step 4]

The runtime only observes and serves batches of 1-4 tokens (build_moe_ffn, mc_max_tokens), so prefill neither
fills nor refreshes the cache: a request starts decoding on whatever the previous request left behind (or on an
empty cache in every benchmark, which launches a fresh server). This replays router traces as a stream of
requests (P prompt tokens, then G generated tokens) and compares, for the G decode tokens only:

    cold      empty cache at decode start                      (every benchmark row today)
    carry     cache as the previous request's decode left it   (a long-lived server today)
    warm      cold + slots preloaded with the prompt's top experts per layer        (P4c in a benchmark)
    refresh   carry + slots re-ranked by prompt frequency, old entries as filler    (P4c in a server)

"same" draws consecutive requests from one corpus; "cross" alternates the two corpora, so the carried cache comes
from the other domain (topic shift). Decode follows the runtime's rules: free slots fill on the first miss,
evictions are gated (A misses within W tokens), at most I uploads per layer per step, an upload becomes visible
two tokens after the miss that admitted it, eviction is LRU.

Caveat: traces are teacher-forced corpus text, so "prompt" and "generation" are adjacent spans of one document.
A real reply is less similar to its prompt than that, so the warm/refresh gains here are an upper bound.
"""
import argparse
import collections

import numpy as np

from sim import load


class Cache:
    def __init__(self, slots, window, admit, inserts):
        self.slots, self.window, self.admit, self.inserts = slots, window, admit, inserts
        self.live = collections.OrderedDict()  # expert -> None, order = recency (last = most recent)
        self.inflight = []                     # (visible_at, expert)
        self.misses = collections.defaultdict(collections.deque)
        self.t = 0
        self.uploads = 0

    def preload(self, ranked):
        """ranked: experts, best first. Replaces the contents; returns the number of uploads it took."""
        keep = list(ranked[: self.slots])
        n_new = sum(1 for e in keep if e not in self.live)
        self.live = collections.OrderedDict((e, None) for e in reversed(keep))  # best = most recent
        self.inflight.clear()
        self.misses.clear()
        return n_new

    def step(self, row):
        """One decoded token. Returns the number of hits among its k experts."""
        self.t += 1
        still = []
        for due, e in self.inflight:
            if due <= self.t:
                self.live[e] = None
                self.live.move_to_end(e)
            else:
                still.append((due, e))
        self.inflight = still
        flying = {e for _, e in self.inflight}

        hits, pending = 0, []
        for e in map(int, row):
            if e in self.live:
                hits += 1
                self.live.move_to_end(e)
                continue
            if e in flying or e in pending:
                continue
            free = self.slots - len(self.live) - len(self.inflight) - len(pending)
            if free > 0:
                pending.append(e)
                continue
            q = self.misses[e]
            q.append(self.t)
            while q and q[0] <= self.t - self.window:
                q.popleft()
            if len(q) >= self.admit:
                pending.append(e)
        for e in pending[::-1][: self.inserts]:  # newest first, as the runtime walks pending in reverse
            if len(self.live) + len(self.inflight) >= self.slots:
                if not self.live:
                    break
                self.live.popitem(last=False)
            self.inflight.append((self.t + 2, e))
            self.uploads += 1
        return hits


def prompt_rank(prompt, n_expert, old=()):
    """Experts by prompt frequency, best first; experts the prompt never routed to are filled from `old` (recent first)."""
    cnt = np.bincount(prompt.ravel(), minlength=n_expert)
    ranked = [int(e) for e in np.argsort(-cnt, kind="stable") if cnt[e] > 0]
    seen = set(ranked)
    ranked += [e for e in reversed(list(old)) if e not in seen]
    return ranked


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("code")
    ap.add_argument("prose")
    ap.add_argument("--slots", type=int, required=True)
    ap.add_argument("--prompt", default="64,256,1024")
    ap.add_argument("--gen", type=int, default=200)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--admit", type=int, default=3)
    ap.add_argument("--inserts", type=int, default=2)
    ap.add_argument("--layer-step", type=int, default=4, help="simulate every Nth layer")
    ap.add_argument("--head", type=int, default=32, help="also report the first N decode tokens")
    args = ap.parse_args()

    traces = {"code": load(args.code), "prose": load(args.prose)}
    layers = sorted(traces["code"])[:: args.layer_step]
    n_expert = 1 + max(int(tr[l].max()) for tr in traces.values() for l in layers)
    k = traces["code"][layers[0]].shape[1]
    n_tok = min(tr[l].shape[0] for tr in traces.values() for l in layers)
    print(f"{len(layers)} layers (every {args.layer_step}), {n_expert} experts top-{k}, {n_tok} tokens/corpus, "
          f"{args.slots} slots, gate {args.admit}/{args.window}, {args.inserts} inserts/step, gen {args.gen}")

    G, H = args.gen, args.head
    for P in map(int, args.prompt.split(",")):
        n_req = n_tok // (P + G)
        for mix in ("same", "cross"):
            tot = collections.defaultdict(lambda: np.zeros(3))  # scenario -> [hits_all, hits_head, preload uploads]
            n_scored = 0
            for l in layers:
                carry = {c: Cache(args.slots, args.window, args.admit, args.inserts) for c in ("carry", "refresh")}
                for r in range(n_req):
                    corpus = ("code", "prose")[r % 2] if mix == "cross" else "code" if r < n_req // 2 else "prose"
                    # cross: request r uses span r of its own corpus, so neighbors in time are from the other corpus
                    seq = traces[corpus][l]
                    prompt, gen = seq[r * (P + G): r * (P + G) + P], seq[r * (P + G) + P: (r + 1) * (P + G)]
                    runs = {"cold": Cache(args.slots, args.window, args.admit, args.inserts),
                            "warm": Cache(args.slots, args.window, args.admit, args.inserts),
                            "carry": carry["carry"], "refresh": carry["refresh"]}
                    pre = {"warm": runs["warm"].preload(prompt_rank(prompt, n_expert)),
                           "refresh": runs["refresh"].preload(prompt_rank(prompt, n_expert, runs["refresh"].live))}
                    for name, c in runs.items():
                        h = np.fromiter((c.step(row) for row in gen), dtype=np.int64, count=G)
                        if r > 0:  # request 0 has nothing to carry
                            tot[name] += (h.sum(), h[:H].sum(), pre.get(name, 0))
                    n_scored += r > 0
            line = [f"P={P:5d} {mix:5s}"]
            for name in ("cold", "carry", "warm", "refresh"):
                a, hd, up = tot[name]
                line.append(f"{name} {100 * a / (n_scored * G * k):5.1f}% (first {H}: {100 * hd / (n_scored * H * k):5.1f}%"
                            + (f", {up / n_scored:4.1f} preload uploads/layer" if name in ("warm", "refresh") else "") + ")")
            print(" | ".join(line))


if __name__ == "__main__":
    main()

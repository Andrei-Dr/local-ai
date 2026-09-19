"""Best of both worlds? One scored cache: score(e) = decayed use frequency + gamma * context affinity, where affinity =
sum over the last C seen tokens of P(e | token) from the token->expert table (shift 0: tokens already seen, no lookahead
problem). Evict the lowest score; admit a candidate only if it outscores the victim by a margin; I uploads per step,
visible 2 steps later. Candidates = this step's misses (+ the top uncached affinity experts when prefetch=1)."""
import sys, collections, itertools, numpy as np
sys.path.insert(0, ".")
from sim import load
trace, slots = sys.argv[1], int(sys.argv[2])
layers = load(trace); toks = np.fromfile(trace + ".tok", dtype="<i4")
ids = sorted(layers)[::8]; n = min(min(len(layers[l]) for l in ids), len(toks)); cut = n // 2
C, I, MARGIN = 8, 2, 0.2
def run(seq, table, gamma, half, prefetch):
    d = 0.5 ** (1.0 / half); f = collections.defaultdict(float); live = set(); infl = []; ctx = collections.deque(maxlen=C)
    hits = ups = 0
    for t in range(cut - 3000, n):
        live |= {e for due, e in infl if due <= t}; infl = [(due, e) for due, e in infl if due > t]
        for e in list(f): f[e] *= d
        ctx.append(int(toks[t])); aff = collections.Counter()
        if gamma:
            for tk in ctx:
                for e, p in table.get(tk, ()): aff[e] += p
        row = seq[t].tolist(); miss = []
        for e in row:
            if e in live: hits += t >= cut
            else: miss.append(e)
            f[e] += 1.0
        score = lambda e: f[e] + gamma * aff[e]
        flying = {e for _, e in infl}
        cands = [e for e in miss if e not in flying]
        if prefetch and gamma: cands += [e for e, _ in aff.most_common(8) if e not in live and e not in flying and e not in cands]
        cands.sort(key=score, reverse=True)
        for e in cands[:I]:
            if len(live) + len(infl) < slots: infl.append((t + 2, e)); ups += t >= cut; continue
            v = min(live, key=score)
            if score(e) > score(v) * (1 + MARGIN): live.discard(v); infl.append((t + 2, e)); ups += t >= cut
    return hits, ups
grid = [(0, 16, 0), (0, 64, 0), (0, 256, 0), (1, 64, 0), (3, 64, 0), (3, 64, 1), (10, 64, 1), (3, 256, 1)]
tot = {g: [0, 0] for g in grid}
for l in ids:
    seq = layers[l]; cnt = collections.defaultdict(collections.Counter)
    for t in range(cut): cnt[int(toks[t])].update(seq[t].tolist())
    table = {tk: [(e, c / sum(cn.values()) * 8) for e, c in cn.most_common(16)] for tk, cn in cnt.items()}
    for g in grid:
        h, u = run(seq, table, *g); tot[g][0] += h; tot[g][1] += u
lk = (n - cut) * 8 * len(ids)
print(f"{trace.split('/')[-1]} | {slots} slots | {len(ids)} layers | runtime policy = 60%, Belady = 75-79%")
for g in grid: print(f"  gamma {g[0]:>2} half-life {g[1]:>3} prefetch {g[2]} : hit {100*tot[g][0]/lk:5.1f}% | uploads/token/layer {tot[g][1]/((n-cut)*len(ids)):.2f}")

"""For every step: is the table's best UNCACHED prediction worth more than the cache's COLDEST entry?
Value = number of times the expert is used over the next W tokens (from t+2, when an upload would be visible)."""
import sys, collections, numpy as np
sys.path.insert(0, ".")
from sim import load
from warmup import Cache
trace, slots, W, S = sys.argv[1], int(sys.argv[2]), 16, 2
layers = load(trace); toks = np.fromfile(trace + ".tok", dtype="<i4")
ids = sorted(layers)[::5]; n = min(min(len(layers[l]) for l in ids), len(toks)); cut = n // 2
tot = collections.Counter()
for l in ids:
    seq = layers[l]
    table = collections.defaultdict(collections.Counter)
    for t in range(S, cut): table[int(toks[t - S])].update(seq[t].tolist())
    c = Cache(slots, 16, 3, 2)
    for t in range(cut - 2000, n - W - 2):
        c.step(seq[t])
        if t < cut or len(c.live) < slots: continue
        fut = collections.Counter(seq[t + 2: t + 2 + W].ravel().tolist())
        cold = next(iter(c.live))                      # LRU victim
        pred = [e for e, _ in table.get(int(toks[t]), collections.Counter()).most_common(16) if e not in c.live]
        tot["steps"] += 1; tot["cold_uses"] += fut[cold]; tot["cold_any"] += fut[cold] > 0
        if pred:
            tot["has_pred"] += 1; tot["pred_uses"] += fut[pred[0]]; tot["pred_any"] += fut[pred[0]] > 0
            tot["pred_at_t2"] += pred[0] in seq[t + 2]
        best_unc = max((e for e in fut if e not in c.live), key=lambda e: fut[e], default=None)  # oracle
        if best_unc is not None: tot["oracle_uses"] += fut[best_unc]
s = tot["steps"]
print(f"{trace.split('/')[-1]}: per step, uses over the next {W} tokens (from t+2)")
print(f"  coldest cached expert (what a prefetch evicts): {tot['cold_uses']/s:.2f} uses, used at all {100*tot['cold_any']/s:.0f}%")
print(f"  table's best uncached prediction:                {tot['pred_uses']/max(1,tot['has_pred']):.2f} uses, used at all {100*tot['pred_any']/max(1,tot['has_pred']):.0f}%, used at t+2 {100*tot['pred_at_t2']/max(1,tot['has_pred']):.0f}%")
print(f"  ORACLE best uncached expert (perfect foresight):  {tot['oracle_uses']/s:.2f} uses")

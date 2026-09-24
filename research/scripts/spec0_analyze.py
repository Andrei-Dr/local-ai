#!/usr/bin/env python3
"""spec0_analyze.py -- verdicts for spec0 (research/spec-design-2026-09-24.md, Phase 0 P0b: H1, H2, H3)

usage: spec0_analyze.py --t0 DUMP.jsonl [...] [--card DUMP.jsonl ...] [--route DUMP.route ...] [--sib SIB.bin ...]
                        [--slots 21] [--json OUT.json]

Inputs (formats in the llama.cpp fork, branch spec0):
  DUMP.jsonl  LLAMA_SPEC_DUMP, one line per verify step (common/spec-dump.cpp): n_draft, n_accept, replay, draft[d].top /
              target[i].top = 10 x [id, p, logit] (full-vocab softmax at T = 1), draft_tok, tgt_tok (the target's samples)
  DUMP.route  LLAMA_SPEC_ROUTE_DUMP, per step: int32 magic "SPR0", step, n_layers, k, then n_layers x {il, n_tok, ids}
  SIB.bin     llama-spec-sibling: int32 magic "SIB0", n_alt, then per decoded token: kind (0 path, 2 path at a sampled
              position, 1 alternative), pos, token, rank, float prob, n_layers, k, n_layers x {il, ids[k]}
  --route needs exactly one --t0 dump, from the same server process (step indexes restart per process); routing is only
  summarized (alignment + NEW experts of accepted vs rejected draft tokens), no verdict.

Rules (pre-registered in bench/box/spec0.sh, copied from the design doc):
  H1 per-depth collapse: acceptance at depth 2 and 3 conditional on depth-1 acceptance < 0.8 x depth 1 -> TRUE (collapse);
     both >= 0.8 x depth 1 -> FALSE (L1 demoted); one each way -> MIXED. Conditioning: depth d counts the steps that drafted
     at least d tokens and whose depths 1..d-1 were all accepted (a depth-3 token can only be accepted after 1 and 2). The
     looser P(depth d | depth 1 accepted) is printed beside it.
  H2 depth-1 alternatives: P(target token in draft top-2..4 | draft top-1 rejected at depth 1) >= 0.30 -> TRUE; < 0.15 ->
     L3 / L4 dead at T = 0 (open for T > 0 only if the card-sampler number >= 0.30); in between -> NOT DECIDED.
  H3 sibling overlap: mean NEW experts per layer for a depth-1 sibling (not in the accepted path's set, not in the LRU)
     <= 2.0 -> TRUE (width cheap); >= 3.5 -> L4 collapses to L3; in between -> NOT DECIDED.
     Model of a chain n = 3 verify at sampled position p: the batch = path tokens p-1 .. p+2; the LRU = --slots experts per
     layer replayed (admit every expert, evict least recently used; the verify_union.py model) over the path tokens before
     p-1. A sibling's NEW experts = its experts outside (LRU state before the batch) U (batch set). The same quantity for the
     path token at p against the LRU alone is printed as the one-full-token reference (~4 in the design's trace).
Replayed steps (a checkpoint restore re-verifying known tokens) are excluded everywhere.
"""
import argparse
import json
import struct
import sys
from collections import OrderedDict, defaultdict

SPR0 = 0x30525053
SIB0 = 0x30424953


def load_steps(paths, keep_replay=False):
    steps = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    r["_file"] = path
                    steps.append(r)
    return steps if keep_replay else [r for r in steps if not r.get("replay")]


def h1(steps, max_depth=3):
    """per-depth acceptance: chain-conditional (all earlier depths accepted) and conditional on depth 1 only"""
    out = {}
    for d in range(1, max_depth + 1):
        chain = [r for r in steps if r["n_draft"] >= d and r["n_accept"] >= d - 1]
        loose = [r for r in steps if r["n_draft"] >= d and (d == 1 or r["n_accept"] >= 1)]
        out[d] = {"n": len(chain), "acc": sum(r["n_accept"] >= d for r in chain) / len(chain) if chain else None,
                  "n_given_d1": len(loose),
                  "acc_given_d1": sum(r["n_accept"] >= d for r in loose) / len(loose) if loose else None}
    a1, a2, a3 = (out[d]["acc"] for d in (1, 2, 3))
    if None in (a1, a2, a3):
        verdict = "INCONCLUSIVE (a depth has no samples)"
    else:
        low = [a < 0.8 * a1 for a in (a2, a3)]
        verdict = ("TRUE (collapse)" if all(low) else
                   "FALSE (no collapse: L1 demoted)" if not any(low) else "MIXED (one depth collapses)")
    return out, verdict


def h2(steps, depth=1):
    """P(target token at the first rejected depth is in the draft's top-2..4) over steps rejected at that depth"""
    rej = [r for r in steps if r["n_draft"] >= depth and r["n_accept"] == depth - 1 and len(r["draft"]) >= depth
           and r["draft"][depth - 1]["top"]]
    hits = defaultdict(int)
    for r in rej:
        want = r["tgt_tok"][depth - 1]
        ids = [e[0] for e in r["draft"][depth - 1]["top"]]
        rank = ids.index(want) if want in ids else -1
        if 1 <= rank <= 3:
            hits["2..4"] += 1
            hits[rank + 1] += 1
        elif rank > 3:
            hits["5..10"] += 1
    n = len(rej)
    frac = hits["2..4"] / n if n else None
    return {"n_rejected": n, "p_top2_4": frac, "by_rank": {k: hits[k] / n for k in (2, 3, 4)} if n else {},
            "p_top5_10": hits["5..10"] / n if n else None}


def h2_verdict(p0, pc):
    if p0 is None:
        return "INCONCLUSIVE (no depth-1 rejections at T = 0)"
    if p0 >= 0.30:
        return "TRUE at T = 0 (L3 / L4 open)"
    if p0 < 0.15:
        if pc is not None and pc >= 0.30:
            return "DEAD at T = 0; OPEN for T > 0 (card sampler >= 0.30)"
        return "DEAD at T = 0" + ("" if pc is None else " and under the card sampler (< 0.30)")
    return "NOT DECIDED at T = 0 (0.15 <= p < 0.30)"


def load_route(path):
    """{step: {il: ndarray-like list of rows}}"""
    data = open(path, "rb").read()
    off, out = 0, {}
    while off + 16 <= len(data):
        magic, step, n_layers, k = struct.unpack_from("<4i", data, off)
        off += 16
        if magic != SPR0:
            sys.exit(f"{path}: bad record magic at byte {off - 16}")
        layers = {}
        for _ in range(n_layers):
            il, n_tok = struct.unpack_from("<2i", data, off)
            off += 8
            ids = struct.unpack_from(f"<{n_tok * k}i", data, off)
            off += 4 * n_tok * k
            layers[il] = [ids[t * k:(t + 1) * k] for t in range(n_tok)]
        out[step] = layers
    return out


class LRU:
    def __init__(self, slots):
        self.slots, self.d = slots, OrderedDict()

    def touch(self, experts):
        for e in experts:
            if e in self.d:
                self.d.move_to_end(e)
            else:
                self.d[e] = 1
                if len(self.d) > self.slots:
                    self.d.popitem(last=False)


def route_summary(steps, route, slots):
    """alignment check + NEW experts per layer of accepted vs rejected draft tokens against an LRU replayed over the
    earlier verify batches (in step order, whole batches, as the cache would see them). steps: all, replays included"""
    by_step = {r["step"]: r for r in steps}
    lru = defaultdict(lambda: LRU(slots))
    bad, new_acc, new_rej = 0, [], []
    for step in sorted(route):
        layers = route[step]
        r = by_step.get(step)
        n_tok = r["n_draft"] + 1 if r else None
        if r is None or any(len(rows) != n_tok for rows in layers.values()):
            bad += 1
        elif not r.get("replay"):
            for d in range(1, n_tok):
                per_layer = [len(set(rows[d]) - set(lru[il].d)) for il, rows in layers.items()]
                (new_acc if d <= r["n_accept"] else new_rej).append(sum(per_layer) / len(per_layer))
        for il, rows in layers.items():  # every batch loads the cache, replays included
            for row in rows:
                lru[il].touch(row)
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    return {"steps": len(route), "misaligned": bad, "new_per_layer_accepted": mean(new_acc),
            "new_per_layer_rejected": mean(new_rej), "n_accepted": len(new_acc), "n_rejected": len(new_rej)}


def load_sib(path):
    data = open(path, "rb").read()
    magic, n_alt = struct.unpack_from("<2i", data, 0)
    if magic != SIB0:
        sys.exit(f"{path}: not a spec-sibling file")
    off, recs = 8, []
    while off + 28 <= len(data):
        kind, pos, tok, rank = struct.unpack_from("<4i", data, off)
        prob, = struct.unpack_from("<f", data, off + 16)
        n_layers, k = struct.unpack_from("<2i", data, off + 20)
        off += 28
        layers = {}
        for _ in range(n_layers):
            il, = struct.unpack_from("<i", data, off)
            layers[il] = struct.unpack_from(f"<{k}i", data, off + 4)
            off += 4 + 4 * k
        recs.append({"kind": kind, "pos": pos, "tok": tok, "rank": rank, "prob": prob, "layers": layers})
    return n_alt, recs


def h3(sib_paths, slots, n_verify=4):
    sib_new, sib_new_path_only, sib_new_lru_only, path_new, by_rank = [], [], [], [], defaultdict(list)
    for path in sib_paths:
        _, recs = load_sib(path)
        path_tok = {r["pos"]: r for r in recs if r["kind"] in (0, 2)}
        alts = defaultdict(list)
        for r in recs:
            if r["kind"] == 1:
                alts[r["pos"]].append(r)
        n = max(path_tok) + 1 if path_tok else 0
        if any(p not in path_tok for p in range(n)):
            sys.exit(f"{path}: path positions are not contiguous")
        layers = sorted(path_tok[0]["layers"]) if n else []
        lru = {il: LRU(slots) for il in layers}
        replayed = 0  # path tokens [0, replayed) are in the LRU
        for p in sorted(alts):
            batch = range(p - 1, min(n, p - 1 + n_verify))
            if p - 1 < 0 or len(batch) < n_verify:
                continue  # no full chain-n=3 verify window at the end of the text
            while replayed < p - 1:
                for il in layers:
                    lru[il].touch(path_tok[replayed]["layers"][il])
                replayed += 1
            bset = {il: set().union(*(path_tok[q]["layers"][il] for q in batch)) for il in layers}
            for a in alts[p]:
                per = [len(set(a["layers"][il]) - bset[il] - set(lru[il].d)) for il in layers]
                m = sum(per) / len(per)
                sib_new.append(m)
                by_rank[a["rank"] + 1].append(m)
                sib_new_path_only.append(sum(len(set(a["layers"][il]) - bset[il]) for il in layers) / len(layers))
                sib_new_lru_only.append(sum(len(set(a["layers"][il]) - set(lru[il].d)) for il in layers) / len(layers))
            path_new.append(sum(len(set(path_tok[p]["layers"][il]) - set(lru[il].d)) for il in layers) / len(layers))
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    m = mean(sib_new)
    verdict = ("INCONCLUSIVE (no siblings)" if m is None else "TRUE (width cheap)" if m <= 2.0 else
               "FALSE: L4 collapses to L3" if m >= 3.5 else "NOT DECIDED (2.0 < new < 3.5)")
    return {"n_siblings": len(sib_new), "new_per_layer": m, "new_vs_batch_only": mean(sib_new_path_only),
            "new_vs_lru_only": mean(sib_new_lru_only), "path_token_new_vs_lru": mean(path_new),
            "by_target_rank": {k: mean(v) for k, v in sorted(by_rank.items())}}, verdict


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t0", nargs="+", required=True)
    ap.add_argument("--card", nargs="*", default=[])
    ap.add_argument("--route", nargs="*", default=[])
    ap.add_argument("--sib", nargs="*", default=[])
    ap.add_argument("--slots", type=int, default=21)
    ap.add_argument("--json")
    a = ap.parse_args()

    res = {}
    regimes = [("T0", a.t0)] + ([("card", a.card)] if a.card else [])
    for name, paths in regimes:
        steps = load_steps(paths)
        acc, v1 = h1(steps)
        d1 = h2(steps, 1)
        deeper = {d: h2(steps, d) for d in (2, 3)}
        res[name] = {"steps": len(steps), "h1": acc, "h1_verdict": v1, "h2": d1, "h2_deeper": deeper}
        print(f"== {name}: {len(steps)} verify steps (replays excluded)")
        for d, x in acc.items():
            print(f"  H1 depth {d}: acc {fmt(x['acc'])} (n {x['n']}, all earlier accepted) | "
                  f"given depth 1 accepted {fmt(x['acc_given_d1'])} (n {x['n_given_d1']})")
        a1 = acc[1]["acc"]
        if a1:
            print(f"  H1 ratio to depth 1: d2 {fmt(acc[2]['acc'] / a1 if acc[2]['acc'] is not None else None)}"
                  f" d3 {fmt(acc[3]['acc'] / a1 if acc[3]['acc'] is not None else None)} (threshold 0.8) -> {v1}")
        print(f"  H2 depth 1: rejected {d1['n_rejected']}, target token in draft top-2..4 {fmt(d1['p_top2_4'])}"
              f" (by rank {', '.join(f'{k}: {fmt(v)}' for k, v in d1['by_rank'].items())}), top-5..10 {fmt(d1['p_top5_10'])}")
        for d, x in deeper.items():
            print(f"     (same at first rejection depth {d}: n {x['n_rejected']}, top-2..4 {fmt(x['p_top2_4'])})")
    pc = res["card"]["h2"]["p_top2_4"] if "card" in res else None
    res["h2_verdict"] = h2_verdict(res["T0"]["h2"]["p_top2_4"], pc)
    print(f"  H2 VERDICT: {res['h2_verdict']}")

    if a.route:
        if len(a.t0) != 1:
            sys.exit("--route needs exactly one --t0 dump (step indexes restart per server process)")
        steps = load_steps(a.t0, keep_replay=True)
        route = {}
        for path in a.route:
            route.update(load_route(path))
        rs = route_summary(steps, route, a.slots)
        res["route"] = rs
        print(f"== routing (T0 verify batches): {rs['steps']} steps, {rs['misaligned']} misaligned with the dump |"
              f" NEW experts / layer vs an LRU {a.slots} over earlier batches: accepted draft tokens"
              f" {fmt(rs['new_per_layer_accepted'], 2)} (n {rs['n_accepted']}), rejected {fmt(rs['new_per_layer_rejected'], 2)}"
              f" (n {rs['n_rejected']})")

    if a.sib:
        x, v3 = h3(a.sib, a.slots)
        res["h3"], res["h3_verdict"] = x, v3
        print(f"== H3 siblings: {x['n_siblings']} | NEW experts / layer (not in batch, not in LRU {a.slots}):"
              f" {fmt(x['new_per_layer'], 2)} | vs batch only {fmt(x['new_vs_batch_only'], 2)} | vs LRU only"
              f" {fmt(x['new_vs_lru_only'], 2)} | one full path token vs LRU {fmt(x['path_token_new_vs_lru'], 2)}")
        print(f"  by target rank: {', '.join(f'{k}: {fmt(v, 2)}' for k, v in x['by_target_rank'].items())}")
        print(f"  H3 VERDICT: {v3}")

    print(f"SPEC0 H1 T0: {res['T0']['h1_verdict']}" + (f" | card: {res['card']['h1_verdict']}" if "card" in res else ""))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

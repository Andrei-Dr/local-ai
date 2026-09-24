#!/usr/bin/env python3
"""spec2b_analyze.py -- D1 / D2 diagnosis of spec2's R0 failure (research/spec-design-2026-09-24.md section 9)

usage: spec2b_analyze.py DIR [--json OUT]
DIR holds, per arm A in S1 S2 S3 T3 T4: spec2b_A.json (client: per prompt the generated token ids) and spec2b_A.jsonl
(LLAMA_SPEC_DUMP of the same server). All arms: T = 0, LLAMA_MOE_CACHE_SYNC=1, same 3 prompts.

Positions: the dump's target entry i of a step (n_past, n_accept) predicts generated token g = n_past + 1 + i - L, where L =
the prompt length (= the n_past of the prompt's first step). Only i <= n_accept is kept: those predictions saw accepted tokens
only (i > n_accept conditioned on a rejected draft). Steps are assigned to a prompt by matching their emitted tokens (tgt_tok,
concatenated) against the client's generated ids; the longest match wins (the 16-token warm-up also matches the code prompt).
First divergence k of a pair = the first generated index where the ids differ (none = the shorter length); the positions both
runs computed before their first divergence = g <= k (the context of g is generated[0 .. g-1], identical for g <= k).
  delta(pair) = max over those positions of max over the tokens in both runs' top-10 of |logit_x - logit_y|
  margin(run, g) = top-1 logit - top-2 logit of that run's target at g; a flip's margin = the larger of the two runs' margins
D1: pairwise token ids of S1, S2, S3 (verify batches 2 / 3 / 4). D2: (a) S2 vs S3, (b) T4 vs T3.
Verdict (pre-registered, section 9): "batch-shape class" if [D1 shows at least one flip OR every T4-vs-T3 flip sits at a margin
<= the largest margin at which a D1 flip or an (a) difference occurs] AND max|delta|(b) <= 2 x max|delta|(a); else SUSPECT.
  Reading of "an (a) difference": with no D1 flip there is no flip margin, so the (a) side of the threshold is max|delta|(a),
  the largest logit perturbation batch shape was seen to cause (a flip needs a perturbation of about the margin). Stated here
  because the doc sentence does not define it.
Also printed: S3 vs T3 (the same build and flags in this run: a determinism check, must be identical).
"""
import argparse
import json
import os
import sys

ARMS = ("S1", "S2", "S3", "T3", "T4")


def load_arm(d, arm):
    client = {r["prompt"]: r["tokens"] for r in json.load(open(os.path.join(d, f"spec2b_{arm}.json")))}
    steps = [json.loads(l) for l in open(os.path.join(d, f"spec2b_{arm}.jsonl")) if l.strip()]
    by_task = {}
    for r in steps:
        if not r.get("replay"):
            by_task.setdefault(r["task"], []).append(r)
    dists = {}  # prompt -> {g: top}
    best = {}   # prompt -> number of matched tokens
    for task, ss in by_task.items():
        ss.sort(key=lambda r: r["n_past"])
        emitted = [t for r in ss for t in r["tgt_tok"]]
        L = ss[0]["n_past"]
        for prompt, gen in client.items():
            ref = gen[1:]  # generated[0] comes from the prompt pass, not a verify step
            n = min(len(ref), len(emitted))
            if n == 0 or ref[:n] != emitted[:n] or n <= best.get(prompt, 0):
                continue
            best[prompt] = n
            m = {}
            for r in ss:
                for i, t in enumerate(r["target"][:r["n_accept"] + 1]):
                    if t["top"]:
                        m[r["n_past"] + 1 + i - L] = t["top"]
            dists[prompt] = m
    return client, dists


def first_div(a, b):
    return next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), None)


def margin(top):
    return top[0][2] - top[1][2] if top and len(top) > 1 else None


def compare(x, y, runs):
    """per prompt: first divergence, flip margin, max |delta logit| before it"""
    (cx, dx), (cy, dy) = runs[x], runs[y]
    out = {}
    for prompt in cx:
        a, b = cx[prompt], cy.get(prompt, [])
        k = first_div(a, b)
        limit = k if k is not None else min(len(a), len(b)) - 1
        mx, px = dx.get(prompt, {}), dy.get(prompt, {})
        delta, n_pos = 0.0, 0
        for g in sorted(set(mx) & set(px)):
            if g > limit:
                continue
            lx = {e[0]: e[2] for e in mx[g]}
            ly = {e[0]: e[2] for e in px[g]}
            common = set(lx) & set(ly)
            if common:
                delta = max(delta, max(abs(lx[t] - ly[t]) for t in common))
                n_pos += 1
        fm = None
        if k is not None:
            ms = [m for m in (margin(mx.get(k)), margin(px.get(k))) if m is not None]
            fm = max(ms) if ms else None
        out[prompt] = {"first_div": k, "flip_margin": fm, "max_delta": delta, "positions": n_pos}
    return out


def verdict(d1, a, b):
    d1_flips = [p for pair in d1.values() for p in pair.values() if p["first_div"] is not None]
    t4_flips = [p for p in b.values() if p["first_div"] is not None]
    da = max(p["max_delta"] for p in a.values())
    db = max(p["max_delta"] for p in b.values())
    thr = max([p["flip_margin"] for p in d1_flips if p["flip_margin"] is not None] + [da])
    unknown = [p for p in t4_flips if p["flip_margin"] is None]
    cond1 = bool(d1_flips) or (not unknown and all(p["flip_margin"] <= thr for p in t4_flips))
    cond2 = db <= 2 * da
    v = "BATCH-SHAPE CLASS" if cond1 and cond2 else "SUSPECT"
    return v, {"d1_flips": len(d1_flips), "t4_flips": len(t4_flips), "max_delta_a": da, "max_delta_b": db,
               "margin_threshold": thr, "cond1": cond1, "cond2": cond2}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--json")
    a = ap.parse_args()
    runs = {arm: load_arm(a.dir, arm) for arm in ARMS}
    for arm, (c, d) in runs.items():
        missing = [p for p in c if p not in d]
        if missing:
            sys.exit(f"{arm}: no dump steps matched prompt(s) {missing}")

    def show(name, res):
        for p, r in res.items():
            k = "identical" if r["first_div"] is None else f"differ at {r['first_div']} (flip margin {r['flip_margin'] if r['flip_margin'] is None else round(r['flip_margin'], 4)})"
            print(f"  {name} {p}: {k} | max |delta logit| before it {r['max_delta']:.4g} over {r['positions']} positions")
    d1 = {pair: compare(*pair, runs) for pair in (("S1", "S2"), ("S1", "S3"), ("S2", "S3"))}
    print("D1 batch shape alone (verify batches 2 / 3 / 4):")
    for pair, res in d1.items():
        show(f"{pair[0]} vs {pair[1]}", res)
    da, db = d1[("S2", "S3")], compare("T3", "T4", runs)
    print("D2 (b) TEST n4 vs TEST n3:")
    show("T4 vs T3", db)
    det = compare("S3", "T3", runs)
    print("determinism (same build and flags):")
    show("S3 vs T3", det)
    v, info = verdict(d1, da, db)
    print(f"  max|delta|(a) S2 vs S3 {info['max_delta_a']:.4g} | max|delta|(b) T4 vs T3 {info['max_delta_b']:.4g}"
          f" (limit 2 x (a) = {2 * info['max_delta_a']:.4g}) | D1 flips {info['d1_flips']} | T4 flips {info['t4_flips']}"
          f" | margin threshold {info['margin_threshold']:.4g}")
    print(f"SPEC2B_VERDICT {v} (flip condition {info['cond1']}, delta condition {info['cond2']})")
    if a.json:
        json.dump({"verdict": v, **info, "d1": {f"{x}-{y}": r for (x, y), r in d1.items()}, "b": db, "det": det},
                  open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""spec1_sim.py -- speculation policy simulator over the spec0 dumps (research/spec-design-2026-09-24.md section 7, Phase 1)

usage: spec1_sim.py --t0 D.jsonl --t0-route D.route --card C.jsonl --card-route C.route --sib SIB.bin [...]
                    (--cal DIR | --t-fixed MS --t-draft MS --c MS) [--slots 21] [--json OUT]

Time model (pre-registered): t_step = t_fixed + t_draft x depth + c x NEW, in ms.
  depth = draft decodes the policy runs in the step (siblings come from a decode that already ran: no extra decode).
  NEW   = expert-layer units: for every MoE layer, the experts of the verify batch (union over its tokens) that are not in a
          --slots LRU per layer, summed over the layers. One unit = one expert of one layer read by the CPU (~1.2 MiB).
          The LRU is replayed over the dumped verify batches in step order (the served chain n = 3 history, the same for every
          policy; a policy's own rejected tokens would pollute it slightly differently).
  Chain tokens: NEW is exact per step from the routing dump (the last accepted token + the first n drafts).
  Siblings: routing is not in the step dumps, so a sibling of draft rank r costs the mean NEW of a target-rank-r alternative from
          llama-spec-sibling (H3's measure: not in the 4-token verify batch nor the LRU), per layer x layers; the same cost at
          every depth and for both dumps.
Counterfactuals (pre-registered): depth <= 3; the chain's depth d is accepted iff the dumped acceptance reached d (the draft top-1
  does not depend on the chain length); a sibling at depth d is accepted iff the target's token at d (known once depths 1..d-1
  were accepted) equals it, and then only the sibling and the target's own next token are credited (no dumped drafts beyond it).
  Every dumped step is one sample of a step start; tokens/s = sum(tokens) / sum(t_step) over the steps (the position shift a
  different acceptance would cause is ignored). Replayed steps and steps without aligned routing are dropped.
Policies (section 7, nothing else): chain n = 1..3 (n = 3 = STABLE); chain with a confidence stop (depth d kept while the draft's
  p_top1 >= theta); L3 = chain 3 + draft ranks 2..k (k = 2, 3) as siblings at depth 1, always or only when depth 1's p_top1 <
  theta; L4 = at each depth, p_top1 >= theta continues the chain, else that depth's top-1 plus ranks 2..k (k = 2..4) and stop.
  theta (0.05..0.95 by 0.05) and k are tuned on one dump and scored on the other, both directions.
Calibration (--cal DIR = bench/box/spec1cal.sh output): per (arm n, prompt) the measured step time predicted_ms / steps is fitted by
  least squares to t_fixed + t_draft x n + c x E_NEW(n, prompt), E_NEW from the T = 0 dump's steps of the same prompt (the
  calibration prompts are the first two T = 0 tasks, code and reason, same text at T = 0); n = 4 extrapolates E_NEW linearly from
  n = 2, 3 (the dump has 3 drafts). Fit error per arm = predicted vs measured decode t/s. RULE: every arm n = 1..4 within 3%.
Gate: a policy PASSES if its held-out tokens/s beats chain n = 3 by >= 5% on BOTH dumps AND the fit is within 3%.
Kill: no L3 / L4 policy passes -> the tree / sibling line is closed; the confidence stop is kept if >= +2% on both held-out dumps.
Without --cal the run is UNCALIBRATED: numbers print, verdicts are not verdicts.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec0_analyze as s0a  # noqa: E402  (load_steps, load_route, load_sib, LRU, h3)

THETAS = [round(0.05 * i, 2) for i in range(1, 20)]
GATE, KEEP_STOP, FIT_TOL = 0.05, 0.02, 0.03


class Step:
    __slots__ = ("A", "nd", "p1", "dtop", "tgt", "new", "task")


def prep(dump, route_path, slots):
    """the dumped steps with per-step chain NEW units (index n = last token + n drafts)"""
    steps = sorted(s0a.load_steps([dump], keep_replay=True), key=lambda r: r["step"])
    route = s0a.load_route(route_path)
    lru = defaultdict(lambda: s0a.LRU(slots))
    out, n_layers, dropped = [], 0, 0
    for r in steps:
        layers = route.get(r["step"])
        nt = r["n_draft"] + 1
        aligned = bool(layers) and all(len(rows) == nt for rows in layers.values())
        if aligned and not r.get("replay") and r["n_draft"] >= 1 and all(d["top"] for d in r["draft"][:r["n_draft"]]):
            s = Step()
            s.A, s.nd, s.tgt, s.task = r["n_accept"], min(r["n_draft"], 3), r["tgt_tok"], r["task"]
            s.p1 = [d["top"][0][1] for d in r["draft"][:s.nd]]
            s.dtop = [[e[0] for e in d["top"]] for d in r["draft"][:s.nd]]
            s.new = []
            for n in range(s.nd + 1):
                s.new.append(sum(len(set().union(*rows[:n + 1]) - set(lru[il].d)) for il, rows in layers.items()))
            out.append(s)
            n_layers = len(layers)
        elif not r.get("replay"):
            dropped += 1
        for il, rows in (layers or {}).items():
            for row in rows:
                lru[il].touch(row)
    return out, n_layers, dropped


# --- policies: step -> (tokens, draft decodes, NEW units incl. siblings)
def chain(s, n, sib_cost):
    n = min(n, s.nd)
    return min(n, s.A) + 1, n, s.new[n]


def conf_stop(s, theta, sib_cost):
    n = 0
    while n < s.nd and s.p1[n] >= theta:
        n += 1
    decodes = n + 1 if n < s.nd else n  # the decode of the depth that failed theta still ran
    return min(n, s.A) + 1, decodes, s.new[n]


def with_siblings(s, d, sibs, n_chain, sib_cost):
    """chain of n_chain (>= d) with siblings `sibs` at depth d; tokens per the pre-registered counterfactual"""
    if s.A >= d:
        tok = min(n_chain, s.A) + 1
    elif s.A == d - 1:
        tok = d + 1 if s.tgt[d - 1] in sibs else d
    else:
        tok = s.A + 1
    return tok, n_chain, s.new[n_chain] + sum(sib_cost(r) for r in range(2, 2 + len(sibs)))


def l3(s, k, theta, sib_cost):
    n = s.nd
    if theta is not None and s.p1[0] >= theta:
        return chain(s, n, sib_cost)
    return with_siblings(s, 1, s.dtop[0][1:k], n, sib_cost)


def l4(s, theta, k, sib_cost):
    for d in range(1, s.nd + 1):
        if s.p1[d - 1] < theta:
            return with_siblings(s, d, s.dtop[d - 1][1:k], d, sib_cost)
    return chain(s, s.nd, sib_cost)


def score(steps, pol, params, sib_cost):
    tf, td, c = params
    tok = ms = 0.0
    for s in steps:
        t, dec, new = pol(s, sib_cost)
        tok += t
        ms += tf + td * dec + c * new
    return 1000.0 * tok / ms, tok / len(steps)


FAMILIES = {  # name -> (grid, policy factory)
    "chain n=1": ([None], lambda g: lambda s, sc: chain(s, 1, sc)),
    "chain n=2": ([None], lambda g: lambda s, sc: chain(s, 2, sc)),
    "chain n=3 (STABLE)": ([None], lambda g: lambda s, sc: chain(s, 3, sc)),
    "confidence stop": ([(t,) for t in THETAS], lambda g: lambda s, sc: conf_stop(s, g[0], sc)),
    "L3 rank 2 always": ([None], lambda g: lambda s, sc: l3(s, 2, None, sc)),
    "L3 ranks 2-3 always": ([None], lambda g: lambda s, sc: l3(s, 3, None, sc)),
    "L3 rank 2 if p1<theta": ([(t,) for t in THETAS], lambda g: lambda s, sc: l3(s, 2, g[0], sc)),
    "L3 ranks 2-3 if p1<theta": ([(t,) for t in THETAS], lambda g: lambda s, sc: l3(s, 3, g[0], sc)),
    "L4": ([(t, k) for t in THETAS for k in (2, 3, 4)], lambda g: lambda s, sc: l4(s, g[0], g[1], sc)),
}


def gname(g):
    if g is None:
        return "-"
    return f"theta {g[0]:.2f}" + (f" k {g[1]}" if len(g) > 1 else "")


def fit_calibration(cal_dir, t0_steps):
    """least squares on (arm, prompt) step times; returns params, the per-arm table, fit_ok"""
    tasks = sorted({s.task for s in t0_steps})
    task_of = {"code": tasks[0], "reason": tasks[1]} if len(tasks) >= 2 else {}
    e_new = {}
    for p, t in task_of.items():
        ss = [s for s in t0_steps if s.task == t and s.nd >= 3]
        e = [float(np.mean([s.new[n] for s in ss])) for n in range(4)]
        e.append(e[3] + (e[3] - e[2]))  # n = 4: linear extrapolation (the dump drafted 3)
        e_new[p] = e
    rows = []
    for f in sorted(glob.glob(os.path.join(cal_dir, "spec1cal_n*_*.json"))):
        n = int(re.search(r"spec1cal_n(\d)_", os.path.basename(f)).group(1))
        for r in json.load(open(f)):
            if r["prompt"] not in e_new or not r.get("predicted_ms"):
                continue
            steps = r["predicted_n"] if n == 0 else r["steps"]
            if steps <= 0:
                continue
            rows.append((n, r["prompt"], r["predicted_n"], r["predicted_ms"], steps))
    if not rows:
        sys.exit(f"{cal_dir}: no calibration rows")
    X = np.array([[1.0, n, e_new[p][n]] for n, p, *_ in rows])
    y = np.array([ms / st for n, p, tok, ms, st in rows])
    params, *_ = np.linalg.lstsq(X, y, rcond=None)
    per_arm = {}
    for (n, p, tok, ms, st), x in zip(rows, X):
        a = per_arm.setdefault(n, {"tok": 0, "ms": 0.0, "pred_ms": 0.0})
        a["tok"] += tok
        a["ms"] += ms
        a["pred_ms"] += st * float(x @ params)
    table, ok = [], True
    for n in sorted(per_arm):
        a = per_arm[n]
        meas, pred = 1000 * a["tok"] / a["ms"], 1000 * a["tok"] / a["pred_ms"]
        err = pred / meas - 1
        table.append((n, meas, pred, err))
        if 1 <= n <= 4 and abs(err) > FIT_TOL:
            ok = False
    if any(n not in per_arm for n in (1, 2, 3, 4)):
        ok = None  # an arm is missing: INCONCLUSIVE
    return tuple(float(v) for v in params), table, ok, float(np.linalg.cond(X)), e_new


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t0", required=True)
    ap.add_argument("--t0-route", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--card-route", required=True)
    ap.add_argument("--sib", nargs="+", required=True)
    ap.add_argument("--cal")
    ap.add_argument("--t-fixed", type=float)
    ap.add_argument("--t-draft", type=float)
    ap.add_argument("--c", type=float)
    ap.add_argument("--slots", type=int, default=21)
    ap.add_argument("--json")
    a = ap.parse_args()

    dumps = {}
    for name, d, r in (("T0", a.t0, a.t0_route), ("card", a.card, a.card_route)):
        steps, n_layers, dropped = prep(d, r, a.slots)
        dumps[name] = steps
        print(f"{name}: {len(steps)} steps ({dropped} dropped: no aligned routing), {n_layers} MoE layers,"
              f" mean chain NEW units n=0..3: {' / '.join(f'{np.mean([s.new[n] for s in steps if s.nd >= 3]):.0f}' for n in range(4))}")

    h3, _ = s0a.h3(a.sib, a.slots)
    by_rank = {int(k): v * n_layers for k, v in h3["by_target_rank"].items() if v is not None}
    mean_sib = h3["new_per_layer"] * n_layers

    def sib_cost(rank):
        return by_rank.get(rank, mean_sib)
    print(f"sibling cost (NEW units, target rank -> units): {', '.join(f'{k}: {v:.1f}' for k, v in sorted(by_rank.items()))}"
          f" | all {mean_sib:.1f} ({h3['n_siblings']} siblings)")

    res = {"units": "ms; NEW in expert-layer units (one expert of one MoE layer)"}
    if a.cal:
        params, table, fit_ok, cond, e_new = fit_calibration(a.cal, dumps["T0"])
        calibrated = True
        print(f"CALIBRATION FIT ({a.cal}): t_fixed {params[0]:.2f} ms, t_draft {params[1]:.2f} ms / depth,"
              f" c {params[2]:.4f} ms / NEW unit (condition number {cond:.0f})")
        for p, e in e_new.items():
            print(f"  E_NEW {p} n=0..4: {' / '.join(f'{x:.0f}' for x in e)} (n = 4 extrapolated)")
        for n, meas, pred, err in table:
            print(f"  arm n={n}: measured {meas:.2f} t/s, fit {pred:.2f} t/s, error {100 * err:+.1f}%")
        fit_line = {True: "FIT OK (n = 1..4 within 3%)", False: "FIT FAILS (an arm n = 1..4 off by > 3%)",
                    None: "FIT INCONCLUSIVE (an arm n = 1..4 missing)"}[fit_ok]
        print(f"  {fit_line}")
        res.update({"fit": {"t_fixed": params[0], "t_draft": params[1], "c": params[2], "arms": table, "ok": fit_ok}})
    else:
        if None in (a.t_fixed, a.t_draft, a.c):
            sys.exit("give --cal DIR, or all of --t-fixed --t-draft --c")
        params, calibrated, fit_ok = (a.t_fixed, a.t_draft, a.c), False, None
        print("=" * 100)
        print(f"UNCALIBRATED: t_fixed {a.t_fixed} ms, t_draft {a.t_draft} ms / depth, c {a.c} ms / NEW expert-layer unit"
              " (overrides, not fitted): NO VERDICT")
        print("=" * 100)

    base = {d: score(dumps[d], FAMILIES["chain n=3 (STABLE)"][1](None), params, sib_cost)[0] for d in dumps}
    print(f"chain n=3 (STABLE) predicted: T0 {base['T0']:.2f} t/s, card {base['card']:.2f} t/s")
    print(f"{'policy':26s} | {'tuned on card -> scored T0':38s} | {'tuned on T0 -> scored card':38s} | gate")
    table, verdicts = {}, {}
    for name, (grid, make) in FAMILIES.items():
        row = {}
        for tune, held in (("card", "T0"), ("T0", "card")):
            best = max(grid, key=lambda g: score(dumps[tune], make(g), params, sib_cost)[0])
            tps, tpstep = score(dumps[held], make(best), params, sib_cost)
            row[held] = {"tuned_on": tune, "grid": gname(best), "tps": tps, "tok_per_step": tpstep,
                         "gain": tps / base[held] - 1}
        gains = (row["T0"]["gain"], row["card"]["gain"])
        passes = min(gains) >= GATE
        if calibrated:
            gate = "PASS" if passes and fit_ok else ("fails +5%" if not passes else "blocked by the fit")
        else:
            gate = "(uncalibrated) " + ("would pass +5%" if passes else "below +5%")
        table[name], verdicts[name] = row, (passes, gains)
        cell = lambda h: f"{gname(None) if row[h]['grid'] == '-' else row[h]['grid']:>16s} {row[h]['tps']:6.2f} t/s {100 * row[h]['gain']:+5.1f}%"
        print(f"{name:26s} | {cell('T0'):38s} | {cell('card'):38s} | {gate}")

    tag = "" if calibrated else "UNCALIBRATED (not a verdict): "
    tree = [n for n in verdicts if n.startswith(("L3", "L4")) and verdicts[n][0]]
    if calibrated and not fit_ok:
        tree_line = "NO VERDICT: the calibration fit did not pass the 3% rule"
    else:
        tree_line = f"OPEN ({', '.join(tree)} >= +5% on both held-out dumps)" if tree else \
            "CLOSED (no L3 / L4 policy >= +5% on both held-out dumps)"
    g_stop = verdicts["confidence stop"][1]
    stop_line = (f"KEEP ({100 * g_stop[0]:+.1f}% T0 / {100 * g_stop[1]:+.1f}% card, >= +2% on both)" if min(g_stop) >= KEEP_STOP
                 else f"DROP ({100 * g_stop[0]:+.1f}% T0 / {100 * g_stop[1]:+.1f}% card)")
    print(f"SPEC1 {tag}TREE / SIBLING LINE: {tree_line}")
    print(f"SPEC1 {tag}CONFIDENCE STOP: {stop_line}")
    res.update({"calibrated": calibrated, "params": params, "baseline": base, "policies": table,
                "tree_line": tree_line, "confidence_stop": stop_line})
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

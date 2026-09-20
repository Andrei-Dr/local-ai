#!/usr/bin/env python3
"""paired.py — per-question PAIRED non-inferiority between two quality runs over the same question ids
(SPEC 8.x: the faster arm must PROVE it is not worse). Inputs are bench/qual/results/LABEL.jsonl lines:
{"id", "set", "ok", ...} — a later line with the same id replaces the earlier one (resume semantics).

Only DISCORDANT pairs carry information (both-right and both-wrong say nothing about the swap):
  loses = base ok & test not; wins = reverse; n = intersection size.
  diff       = (wins - loses) / n * 100                        (percentage-point accuracy delta)
  paired SE  = sqrt((L + W) - (W - L)^2 / n) / n * 100         (exact McNamar-style variance)
  CI95       = diff +- 1.96 SE
  p (exact)  = two-sided binomial on L + W at p=.5 (conditional McNemar); 0 discordant => p = 1.0
Verdict (ALL only): NON-INFERIOR if CI lower > -margin; WORSE if CI upper < 0; else UNDECIDED — plus the
sample size whose half-width would equal the margin at the observed discordance rate:
  needed_n = ceil(1.96^2 * rate / (margin/100)^2)
Every verdict exits 0 (the board renders it; an empty id intersection exits 2 — comparing nothing is not
a result). Malformed lines are skipped, counted on stderr. stdlib only, no plots, no other stats.

usage: paired.py BASE.jsonl TEST.jsonl [--margin 2.0] [--sets gsm8k,humaneval] [--md]
"""
import argparse, json, math, sys

Z = 1.96


def loads_jsonl(text):
    """id -> {"set","ok"}; later duplicate ids replace earlier ones; returns (rows, malformed_count)."""
    got, bad = {}, 0
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            o = json.loads(ln)
            got[o["id"]] = {"set": o["set"], "ok": bool(o["ok"])}
        except (ValueError, KeyError, TypeError):
            bad += 1
    return got, bad


def mcnemar_p(loses, wins):
    m = loses + wins
    if not m:
        return 1.0
    tail = sum(math.comb(m, i) for i in range(min(loses, wins) + 1)) / (2 ** m)
    return round(min(1.0, 2 * tail), 4)


def stat(bmap, tmap, ids, margin=2.0, with_verdict=False):
    n = len(ids)
    loses = sum(1 for i in ids if bmap[i]["ok"] and not tmap[i]["ok"])
    wins = sum(1 for i in ids if not bmap[i]["ok"] and tmap[i]["ok"])
    disc = loses + wins
    rate = disc / n if n else 0.0
    diff = (wins - loses) / n * 100 if n else 0.0
    se_raw = (math.sqrt(disc - (wins - loses) ** 2 / n) / n * 100) if n and disc else 0.0
    lo, hi = diff - Z * se_raw, diff + Z * se_raw
    row = {"n": n, "loses": loses, "wins": wins, "diff": round(diff, 2), "se": round(se_raw, 2),
           "ci": (round(lo, 2), round(hi, 2)), "p": mcnemar_p(loses, wins),
           "discordance": rate}
    if with_verdict:
        if lo > -margin:
            v = "NON-INFERIOR"
        elif hi < 0:
            v = "WORSE"
        else:
            v = "UNDECIDED"
        row["verdict"] = v
        row["needed_n"] = math.ceil(Z * Z * rate / (margin / 100.0) ** 2) if rate else 0
    return row


def compare(bmap, tmap, sets=None, margin=2.0):
    ob = len(set(bmap) - set(tmap))
    ot = len(set(tmap) - set(bmap))
    if sets:
        bmap = {i: v for i, v in bmap.items() if v["set"] in sets}
        tmap = {i: v for i, v in tmap.items() if v["set"] in sets}
    ids = set(bmap) & set(tmap)
    per = {}
    if ids:
        by_set = {}
        for i in ids:
            by_set.setdefault(bmap[i]["set"], []).append(i)
        for sname in sorted(by_set):
            per[sname] = stat(bmap, tmap, by_set[sname])
    allrow = stat(bmap, tmap, list(ids), margin=margin, with_verdict=bool(ids))
    return {"sets": per, "ALL": allrow, "only_base": ob, "only_test": ot, "ids": len(ids)}


def fmt_plain(name, r):
    lo, hi = r["ci"]
    s = (f"{name:12} n {r['n']:6} wins {r['wins']:4} loses {r['loses']:4} diff {r['diff']:+6.2f}"
         f" se {r['se']:5.2f} ci [{lo:+.2f}, {hi:+.2f}] p {r['p']:.4f} disc {r['discordance']:.3f}")
    if "verdict" in r:
        s += f" verdict {r['verdict']} need_n {r['needed_n']}"
    return s


def fmt_md(name, r):
    lo, hi = r["ci"]
    v = r.get("verdict", "")
    nn = r.get("needed_n", "")
    return (f"| {name} | {r['n']} | {r['wins']} | {r['loses']} | {r['diff']:+.2f} | {r['se']:.2f} "
            f"| [{lo:+.2f}, {hi:+.2f}] | {r['p']:.4f} | {r['discordance']:.3f} | {v} | {nn} |")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="paired.py")
    ap.add_argument("base")
    ap.add_argument("test")
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--sets", default=None)
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args(argv)
    bmap, bad_b = loads_jsonl(open(a.base, encoding="utf-8").read())
    tmap, bad_t = loads_jsonl(open(a.test, encoding="utf-8").read())
    if bad_b or bad_t:
        print(f"malformed lines skipped: base {bad_b}, test {bad_t}", file=sys.stderr)
    res = compare(bmap, tmap, sets=a.sets.split(",") if a.sets else None, margin=a.margin)
    if not res["ids"]:
        print("EMPTY INTERSECTION — no common question ids between the two runs", file=sys.stderr)
        return 2
    line = fmt_md if a.md else fmt_plain
    if a.md:
        print("| set | n | wins | loses | diff pp | se | CI95 | p | disc | verdict | need_n |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
    else:
        print(f"excluded: only in BASE {res['only_base']}, only in TEST {res['only_test']}")
    for sname, r in res["sets"].items():
        print(line(sname, r))
    print(line("ALL", res["ALL"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

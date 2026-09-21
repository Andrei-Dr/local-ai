#!/usr/bin/env python3
"""Offline race/retry analysis over a reseeded qual pass. No server, stdlib only.

usage: race.py BASE.jsonl SALTED.jsonl [SALTED.jsonl ...] [--json OUT.json]

An item's chains = its LAST row in BASE (which must show the chain was cut: finish == length) plus its last row in each
SALTED file, in argument order — the salt-0 draw first, then the reseeds. Only items present in BASE and in EVERY salted
file are analyzed; drops are counted and printed. For N = 1 .. files (chains[:N]): "race" models N parallel draws and pays
N x the tokens of the cheapest finished chain (N x max tokens when none finished), "retry" walks the file order and pays
the tokens up to and including the first finished chain (all N when none). One RACE row per (set, N), then the same over
all sets under the label _all_, then the verdict: seed (some draw can finish the cut chains) / problem (nothing finishes)
/ mixed. Exit 2: unreadable file, fewer than two files, or no analyzable items.
"""
import argparse
import json
import sys


def last_rows(path):
    """id -> last row seen for that id in the jsonl."""
    last = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            last[r["id"]] = r
    return last


def stat(chains, n):
    """Per-item counters over the first n chains: winners of each strategy and the tokens each pays."""
    ch = chains[:n]
    fin = [c for c in ch if c["finish"] == "stop"]
    okf = [c for c in fin if c.get("ok")]
    if fin:
        win = min(fin, key=lambda c: c["tokens"])  # min keeps the lowest index on ties: the earliest file wins the tie
        race = n * win["tokens"]
    else:
        race = n * max(c["tokens"] for c in ch)
    retr = 0
    for c in ch:
        retr += c["tokens"]
        if c["finish"] == "stop":
            break
    return {"finished": 1 if fin else 0, "first_ok": 1 if fin and win.get("ok") else 0, "any_ok": 1 if okf else 0,
            "majority_ok": 1 if okf and 2 * len(okf) > len(fin) else 0, "race_tok": race, "retry_tok": retr}


def verdict(finished, items):
    """seed when at least half of the cut items can be finished by some draw, problem when at most a fifth can, else mixed."""
    if 2 * finished >= items:
        return "seed"
    if 5 * finished <= items:
        return "problem"
    return "mixed"


def run(argv, out=sys.stdout):
    ap = argparse.ArgumentParser(prog="race.py")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", dest="json_out")
    a = ap.parse_args(argv)
    if len(a.files) < 2:
        print("RACE needs BASE plus at least one SALTED file", file=out)
        return 2
    try:
        tabs = [last_rows(f) for f in a.files]
    except (OSError, ValueError) as e:
        print(f"RACE unreadable file: {e}", file=out)
        return 2
    base, saltd = tabs[0], tabs[1:]
    an, miss, notcut = [], 0, 0
    for iid, rb in base.items():
        if rb.get("finish") != "length":
            notcut += 1
            continue
        if any(iid not in t for t in saltd):
            miss += 1
            continue
        an.append((iid, [rb] + [t[iid] for t in saltd]))
    chains = 1 + len(saltd)
    print(f"RACE files={len(a.files)} chains_per_item={chains} base_rows={len(base)} cut={len(base) - notcut} "
          f"analyzable={len(an)} dropped_missing_in_salted={miss} dropped_not_cut={notcut}", file=out)
    if not an:
        print("RACE no analyzable items (no cut item present in every file)", file=out)
        return 2
    groups = {}
    for iid, ch in an:
        groups.setdefault(ch[0].get("set", "?"), []).append((iid, ch))

    rows = []

    def emit(label, members):
        for n in range(1, chains + 1):
            t = {"items": len(members)}
            for k in ("finished", "first_ok", "any_ok", "majority_ok", "race_tok", "retry_tok"):
                t[k] = sum(stat(ch, n)[k] for _, ch in members)
            print(f"RACE set={label} N={n} items={t['items']} finished={t['finished']} first_ok={t['first_ok']} "
                  f"any_ok={t['any_ok']} majority_ok={t['majority_ok']} race_tok={t['race_tok']} retry_tok={t['retry_tok']}", file=out)
            rows.append(dict(t, set=label, N=n))
        return {"finished": t["finished"], "items": t["items"]}

    for label in groups:
        emit(label, groups[label])
    tot = emit("_all_", [(iid, ch) for ms in groups.values() for (iid, ch) in ms])
    v = verdict(tot["finished"], tot["items"])
    print(f"RACE_VERDICT {v}", file=out)
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"files": a.files, "chains_per_item": chains,
                       "dropped": {"missing_in_salted": miss, "not_cut": notcut}, "rows": rows, "verdict": v}, f)
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))

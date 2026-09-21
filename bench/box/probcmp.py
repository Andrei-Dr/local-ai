#!/usr/bin/env python3
"""probcmp.py BASE.slot.json TEST.slot.json: how far do two decode arms differ, token by token?

Both runs come from slotclient.py (MODE=extend, NPROBS=N, temperature 0) on the same restored slot. Text identity alone is a
weak instrument (a changed logit shows only when it flips an argmax); this compares the top-N logprobs of every generated token
up to the first token where the two runs diverge (after that they condition on different text and are not comparable).
Prints one PROBCMP line; exit 0 always (the caller gates on the numbers)."""
import json, sys


def compare(base, test):
    n = same = 0
    dmax = dsum = 0.0
    pairs = 0
    for b, t in zip(base, test):
        n += 1
        tb = dict(map(tuple, b["top"]))
        for tid, lp in t["top"]:
            if tid in tb:
                d = abs(lp - tb[tid])
                dmax = max(dmax, d)
                dsum += d
                pairs += 1
        if b["id"] != t["id"]:
            break
        same += 1
    return {"tokens": n, "same_prefix": same, "identical": same == n == max(len(base), len(test)), "pairs": pairs,
            "max_dlogprob": round(dmax, 6), "mean_dlogprob": round(dsum / pairs, 8) if pairs else None}


def main():
    rows = [json.load(open(p))["rows"][-1].get("probs") for p in sys.argv[1:3]]
    if not all(rows):
        print("PROBCMP no probs in one of the runs (NPROBS unset, or the server returned none)")
        return
    r = compare(*rows)
    print(f"PROBCMP tokens {r['tokens']} | identical {r['identical']} (same prefix {r['same_prefix']}) | "
          f"max |dlogprob| {r['max_dlogprob']} | mean {r['mean_dlogprob']} over {r['pairs']} pairs")


if __name__ == "__main__":
    main()

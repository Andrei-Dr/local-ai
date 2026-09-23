#!/usr/bin/env python3
"""First tensor whose dump differs between two llama-eval-callback logs (common_debug_cb_eval blocks, in graph order).
usage: dumpdiff.py A.log B.log -> prints block count, the first differing block (header + sums), and how many differ."""
import re, sys

HDR = re.compile(r"common_debug_cb_eval:\s+(\S+) = \((\S+)\)\s+(\S+)\((.*)\) = \{(.*)\}")

def blocks(path):
    out, cur = [], None
    for line in open(path, errors="replace"):
        m = HDR.search(line)
        if m:
            cur = {"name": m.group(1), "op": m.group(3), "hdr": line.strip(), "body": []}
            out.append(cur)
        elif cur is not None:
            s = line.strip()
            if s.startswith("sum ="):
                cur["sum"] = s
            elif s and s[0] in "[.-0123456789":
                cur["body"].append(s)
    return out

a, b = blocks(sys.argv[1]), blocks(sys.argv[2])
print(f"blocks: {len(a)} vs {len(b)}")
n_diff, first = 0, None
for i, (x, y) in enumerate(zip(a, b)):
    if x["name"] != y["name"] or x["op"] != y["op"]:
        print(f"STRUCTURE differs at block {i}: {x['hdr']} | {y['hdr']}")
        sys.exit(2)
    if x.get("sum") != y.get("sum") or x["body"] != y["body"]:
        n_diff += 1
        if first is None:
            first = i
if first is None:
    print("IDENTICAL (every tensor dump matches)")
    sys.exit(0)
x, y = a[first], b[first]
print(f"FIRST DIFF at block {first}/{len(a)}: {x['hdr']}")
print(f"   A {x.get('sum')} | B {y.get('sum')}")
for k in range(max(0, first - 3), first):
    print(f"   (prev, same) {a[k]['name']} {a[k]['op']}")
print(f"differing blocks: {n_diff}")
sys.exit(1)

#!/usr/bin/env python3
"""pfprof.py FILE.sqlite — prefill breakdown from an nsys export (graphs off). The prefill window = the longest burst of kernel
activity (gaps < 200 ms merged). Reports wall, GPU-busy union, idle, time per kernel class, H2D / D2H bytes + busy time, and how
much of the upload overlapped compute; then the same for activity AFTER the window (the draft context's prompt pass, if any)."""
import re, sqlite3, sys
from collections import defaultdict

CLASSES = [("mmq", r"mul_mat_q|mmq"), ("mmvq", r"mul_mat_vec_q|mmvq"), ("flash_attn", r"flash_attn|fattn"),
           ("delta_net", r"gated_delta|delta_net|ssm_conv|chunk"), ("mul_mat_f/cublas", r"mul_mat_vec_f|gemm|sgemm|hgemm|cutlass|mul_mat_f"),
           ("get_rows/cpy/concat", r"get_rows|cpy|concat|set_rows|dup"), ("quantize", r"quantize"),
           ("norm/rope/elementwise", r"norm|rope|bin_bcast|unary|scale|soft_max|topk|argsort|glu|silu|add|mul")]


def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def total(iv):
    return sum(e - s for s, e in merge(iv))


def overlap(a, b):
    a, b = merge(a), merge(b); i = j = t = 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s: t += e - s
        if a[i][1] < b[j][1]: i += 1
        else: j += 1
    return t


def report(c, lo, hi, title):
    K = c.execute("select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName "
                  "where k.start>=? and k.end<=?", (lo, hi)).fetchall()
    M = c.execute("select start, end, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and end<=?", (lo, hi)).fetchall()
    kiv = [(s, e) for s, e, _ in K]
    h2d = [(s, e) for s, e, k, _ in M if k == 1]
    busy = total(kiv + h2d)
    span = hi - lo
    print(f"{title}: window {span / 1e6:8.1f} ms | kernels busy {total(kiv) / 1e6:8.1f} | H2D busy {total(h2d) / 1e6:7.1f} "
          f"({sum(b for _, _, k, b in M if k == 1) / 1e9:.2f} GB) | upload hidden under compute {overlap(kiv, h2d) / 1e6:7.1f} | GPU idle {(span - busy) / 1e6:7.1f}")
    per = defaultdict(float); n = defaultdict(int)
    for s, e, name in K:
        cls = next((c_ for c_, rx in CLASSES if re.search(rx, name)), "other:" + name[:28])
        per[cls] += e - s; n[cls] += 1
    kt = sum(per.values()) or 1
    for cls, t in sorted(per.items(), key=lambda kv: -kv[1])[:12]:
        print(f"    {cls:26s} {t / 1e6:8.1f} ms {100 * t / kt:5.1f}%  ({n[cls]} launches)")


def main():
    c = sqlite3.connect(sys.argv[1])
    ks = c.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    if not ks:
        print("no kernels in trace"); return 2
    bursts, cur = [], [ks[0][0], ks[0][1]]
    for s, e in ks[1:]:
        if s - cur[1] < 200e6: cur[1] = max(cur[1], e)
        else: bursts.append(cur); cur = [s, e]
    bursts.append(cur)
    main_b = max(bursts, key=lambda b: b[1] - b[0])
    report(c, main_b[0], main_b[1], "PREFILL (longest burst)")
    after = [b for b in bursts if b[0] > main_b[1]]
    if after:
        report(c, after[0][0], after[-1][1], "AFTER the prefill burst (draft pass + decode)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

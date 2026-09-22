#!/usr/bin/env python3
"""hand1_phases.py — per-layer decode phases from an nsys sqlite export of the MoE-offload server (HAND1).

nsys 2022.4 cannot see kernels inside CUDA graphs, but the host-side CUDA API trace is complete. On the thread that
issues the most cudaGraphLaunch calls, one MoE layer is: D launch (async device graph: expert-cache hits) -> CPU experts
(misses) -> short syncs + H2D of the host expert outputs -> B launch + a LONG cudaStreamSynchronize (the serial device
phase: merge, shared expert, next layer's attention / delta net, router) -> D2H of hidden state + router -> next D launch.
A launch followed by a sync > --long-us is B; any other launch is D. Windows where the server fell back to plain kernel
launches (cudaLaunchKernel present in the 100 ms bin) are skipped.

usage: hand1_phases.py FILE.sqlite [--skip-s 3.7] [--long-us 100]"""
import argparse, sqlite3, statistics as st, sys
from collections import defaultdict


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("db"); ap.add_argument("--skip-s", type=float, default=3.7); ap.add_argument("--long-us", type=float, default=100)
    a = ap.parse_args(argv)
    c = sqlite3.connect(a.db)
    t0 = c.execute("select min(start) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    tid = c.execute("""select globalTid from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId
        where s.value like 'cudaGraphLaunch%' group by globalTid order by count(*) desc limit 1""").fetchone()
    if not tid:
        print("no cudaGraphLaunch in this trace", file=sys.stderr); return 2
    rows = c.execute("""select r.start, r.end, s.value from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId
        where r.globalTid=? and r.start>? order by r.start""", (tid[0], t0 + a.skip_s * 1e9)).fetchall()
    bad = set(int((r[0] - t0) / 1e8) for r in rows if r[2].startswith("cudaLaunchKernel"))
    long_ns = a.long_us * 1e3
    is_B = lambda i: rows[i + 1][2].startswith("cudaStreamSynchronize") and rows[i + 1][1] - rows[i + 1][0] > long_ns
    Ds = [i for i, (s, e, n) in enumerate(rows[:-1])
          if n.startswith("cudaGraphLaunch") and int((s - t0) / 1e8) not in bad and not is_B(i)]
    agg, per, nper = defaultdict(float), [], 0
    for x, y in zip(Ds, Ds[1:]):
        if rows[y][0] - rows[x][0] > 3e6:
            continue
        nper += 1; per.append((rows[y][0] - rows[x][0]) / 1e3)
        seg = rows[x:y]
        for j, (s, e, n) in enumerate(seg):
            nm = n.split("_v")[0]
            if nm == "cudaStreamSynchronize" and e - s > long_ns:
                nm = "[device phase: sync after B launch]"
            agg[nm] += e - s
            if j + 1 < len(seg):
                agg["[CPU experts: gap after D launch]" if j == 0 else "[host gaps between calls]"] += seg[j + 1][0] - e
    if not nper:
        print("no layer periods found", file=sys.stderr); return 2
    per.sort()
    print("layers %d | period us: mean %.1f p10 %.1f p50 %.1f p90 %.1f"
          % (nper, st.mean(per), per[len(per) // 10], per[len(per) // 2], per[9 * len(per) // 10]))
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        print("  %-38s %7.1f us/layer" % (k, v / nper / 1e3))
    return 0


if __name__ == "__main__":
    sys.exit(main())

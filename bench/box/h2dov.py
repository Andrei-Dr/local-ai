import json, sqlite3, sys
sys.path.insert(0, "/ai/bench")
from pfprof import merge, total, overlap
db, label = sys.argv[1], sys.argv[2]
row = json.load(open(f"/ai/bench/runs/{label}.longpf.json"))
n = row["predicted_n"]; win = n / row["decode_tps"] * 1e9
c = sqlite3.connect(db)
last = c.execute("select max(end) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]; lo = last - win
K = [(s, e) for s, e in c.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=?", (lo, last))]
H = [(s, e, b) for s, e, b in c.execute("select start, end, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 and start>=? and end<=?", (lo, last))]
km = merge(K)
idle = [(a[1], b[0]) for a, b in zip(km, km[1:]) if b[0] - a[1] > 20e3]   # gaps > 20 us = waiting on the host
h = [(s, e) for s, e, _ in H]
big = [(s, e) for s, e, b in H if b >= 256 * 1024]                          # cache-fill sized copies
print(f"{label}: decode window {win/1e6:.0f} ms, {n} tok | idle gaps>20us: {len(idle)}, {total(idle)/n/1e6:.2f} ms/tok")
print(f"  H2D all: {len(H)} copies {sum(b for *_, b in H)/n/1e6:.1f} MB/tok busy {total(h)/n/1e6:.2f} ms/tok | inside idle gaps {overlap(h, idle)/max(total(h),1)*100:.0f}% | under kernels {overlap(h, K)/max(total(h),1)*100:.0f}%")
print(f"  H2D >=256KiB (cache fills): {len(big)} copies {sum(b for s, e, b in H if b >= 256*1024)/n/1e6:.1f} MB/tok busy {total(big)/n/1e6:.2f} ms/tok | inside idle gaps {overlap(big, idle)/max(total(big),1)*100:.0f}%")
gl = sorted(e - s for s, e in idle)
if gl: print(f"  idle gap length us: median {gl[len(gl)//2]/1e3:.0f} p90 {gl[int(len(gl)*.9)]/1e3:.0f} max {gl[-1]/1e3:.0f}; gaps per token {len(idle)/n:.1f}")

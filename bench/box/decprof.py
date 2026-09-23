#!/usr/bin/env python3
"""decprof.py FILE.sqlite LABEL -- per-token DECODE breakdown from an nsys export (graphs off). The decode window = the last
predicted_ms of GPU activity, predicted_ms from runs/LABEL.longpf.json (predicted_n / decode_tps). Reports per generated token:
window, kernel-busy union, GPU idle, and time + launches per kernel class (flash_attn split by kernel name)."""
import json, re, sqlite3, sys
from collections import defaultdict
sys.path.insert(0, "/ai/bench")
from pfprof import CLASSES, total


def main():
    db, label = sys.argv[1], sys.argv[2]
    row = json.load(open(f"/ai/bench/runs/{label}.longpf.json"))
    n, tps = row["predicted_n"], row["decode_tps"]
    win = n / tps * 1e9  # ns
    c = sqlite3.connect(db)
    last = c.execute("select max(end) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    lo = last - win
    K = c.execute("select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName "
                  "where k.start>=? and k.end<=?", (lo, last)).fetchall()
    M = c.execute("select start, end, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and end<=?", (lo, last)).fetchall()
    kiv = [(s, e) for s, e, _ in K]
    h2d = [(s, e) for s, e, k, _ in M if k == 1]
    busy = total(kiv)
    print(f"{label}: prompt {row['prompt_n']} tok | decode {n} tok @ {tps:.2f} t/s -> {win / n / 1e6:.2f} ms/token | kernels busy "
          f"{busy / n / 1e6:.2f} ms/token | H2D {total(h2d) / n / 1e6:.2f} ms/token ({sum(b for _, _, k, b in M if k == 1) / n / 1e6:.1f} MB/token) "
          f"| GPU idle (incl. host experts) {(win - total(kiv + h2d)) / n / 1e6:.2f} ms/token")
    per, cnt = defaultdict(float), defaultdict(int)
    for s, e, name in K:
        cls = next((c_ for c_, rx in CLASSES if re.search(rx, name)), "other:" + name[:28])
        if cls == "flash_attn":
            cls = "fa:" + name[:40]
        per[cls] += e - s; cnt[cls] += 1
    for cls, t in sorted(per.items(), key=lambda kv: -kv[1])[:14]:
        print(f"    {cls:46s} {t / n / 1e3:8.1f} us/token  ({cnt[cls] / n:5.1f} launches/token, {t / cnt[cls] / 1e3:7.1f} us each)")
    fa = sum(t for k, t in per.items() if k.startswith("fa:"))
    print(f"    FLASH ATTENTION TOTAL {fa / n / 1e3:.1f} us/token")


if __name__ == "__main__":
    sys.exit(main())

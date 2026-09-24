#!/usr/bin/env python3
"""spec3_analyze.py -- the waterfall of a speculative step (research/spec-design-2026-09-24.md section 10: H4, H5)

usage: spec3_analyze.py DIR [--json OUT]
DIR holds spec3_<ARM>_<i>.json (client rows: prompt, decode_tps, draft_n, draft_n_accepted) for ARM in T1 T2 T3 S3 and
spec3_<ARM>_<i>.jsonl (LLAMA_SPEC_DUMP with the step timing, T arms only). T1 / T2 / T3 = the TEST build at draft n 1 / 2 / 3
with the dump on; S3 = the STABLE build at n 3 with the dump off (prices the dump's own overhead: T3 vs S3 decode t/s).

Per arm, over the steps that drafted exactly n tokens (a shorter draft near the end of a text is a different step) and are
not replays, means of the dump's t fields (ms): draft (whole draft phase), draft_depth[d], process (MTP catch-up decode of the
verify batch), verify (target verify llama_decode + synchronize), verify_gpu (summed GPU time of its CUDA splits), step (host
time since the previous step of the same task; a task's first step has none).
  marginal step cost per position  = least-squares slope of mean step vs n over n = 1, 2, 3
  marginal draft cost per position = the same slope of the mean draft phase
  GPU idle inside the verify       = verify - verify_gpu (host wall of the decode minus the GPU-busy stream time)
Rules (pre-registered, section 10):
  H4: the draft phase is >= 30% of the marginal step cost per extra position (i.e. >= 2.8 ms of the ~9.2 ms per depth).
      If < 15%, M1 is dead (nothing to hide); between: report and decide.
      -> ratio = marginal draft slope / marginal step slope; the absolute draft ms per position is printed beside it.
  H5: GPU idle inside the verify at n3 >= the n3 draft phase time (the window can hold the draft). If not, M1 hides only part.
      -> mean(verify - verify_gpu) at n = 3 vs mean(draft) at n = 3.
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import sys

ARMS = {"T1": 1, "T2": 2, "T3": 3}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def slope(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def load(d):
    arms = {}
    for f in sorted(glob.glob(os.path.join(d, "spec3_*_*.json"))):
        m = re.match(r"spec3_(T1|T2|T3|S3)_(\d+)\.json$", os.path.basename(f))
        if not m:
            continue
        a = arms.setdefault(m.group(1), {"client": [], "steps": []})
        a["client"].append(json.load(open(f)))
        dump = f[:-5] + ".jsonl"
        if os.path.exists(dump):
            a["steps"] += [json.loads(l) for l in open(dump) if l.strip()]
    return arms


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--json")
    a = ap.parse_args()
    arms = load(a.dir)
    if any(k not in arms for k in ARMS):
        sys.exit(f"missing arm(s): {[k for k in ARMS if k not in arms]}")

    res = {"arms": {}}
    for name, n in ARMS.items():
        ss = [s for s in arms[name]["steps"] if s["n_draft"] == n and not s.get("replay") and "t" in s]
        t = lambda k: [s["t"].get(k) for s in ss]
        depth = [mean([s["t"]["draft_depth"][d] for s in ss if len(s["t"]["draft_depth"]) > d]) for d in range(n)]
        idle = [s["t"]["verify"] - s["t"]["verify_gpu"] for s in ss if s["t"].get("verify_gpu") is not None]
        r = {"n": n, "steps": len(ss), "draft": mean(t("draft")), "draft_depth": depth, "process": mean(t("process")),
             "verify": mean(t("verify")), "verify_gpu": mean(t("verify_gpu")), "gpu_idle": mean(idle), "step": mean(t("step")),
             "tokens_per_step": mean([s["n_accept"] + 1 for s in ss]),
             "decode_tps": [round(mean([r["decode_tps"] for r in c]), 2) for c in arms[name]["client"]]}
        res["arms"][name] = r
        print(f"{name} (n {n}, {len(ss)} steps): step {fmt(r['step'])} ms | draft {fmt(r['draft'])} ms"
              f" (per depth {' / '.join(fmt(x) for x in depth)}) | process {fmt(r['process'])} | verify {fmt(r['verify'])}"
              f" (GPU {fmt(r['verify_gpu'])}, idle {fmt(r['gpu_idle'])}) | tokens/step {fmt(r['tokens_per_step'])}"
              f" | decode t/s per pass {r['decode_tps']}")
    if "S3" in arms:
        s3 = [round(mean([r["decode_tps"] for r in c]), 2) for c in arms["S3"]["client"]]
        t3 = res["arms"]["T3"]["decode_tps"]
        res["dump_overhead_pct"] = 100 * (mean(t3) / mean(s3) - 1)
        print(f"dump overhead: T3 (dump on) {t3} vs S3 (STABLE build, dump off) {s3} t/s -> {res['dump_overhead_pct']:+.1f}%")

    ns = [1, 2, 3]
    steps = [res["arms"][f"T{n}"]["step"] for n in ns]
    drafts = [res["arms"][f"T{n}"]["draft"] for n in ns]
    if None in steps or None in drafts:
        h4 = "INCONCLUSIVE (an arm has no timed steps)"
        ms_step = ms_draft = ratio = None
    else:
        ms_step, ms_draft = slope(ns, steps), slope(ns, drafts)
        ratio = ms_draft / ms_step if ms_step > 0 else None
        if ratio is None:
            h4 = "INCONCLUSIVE (step time does not grow with n)"
        elif ratio >= 0.30:
            h4 = "TRUE (the draft is >= 30% of a position's cost: M1 has something to hide)"
        elif ratio < 0.15:
            h4 = "FALSE (< 15%: M1 is dead, nothing to hide)"
        else:
            h4 = "BETWEEN (15-30%: report and decide)"
    print(f"H4 marginal cost per position: step {fmt(ms_step)} ms, draft phase {fmt(ms_draft)} ms"
          f" -> draft share {fmt(ratio if ratio is None else 100 * ratio, 1)}% -> {h4}")
    r3 = res["arms"]["T3"]
    if r3["gpu_idle"] is None or r3["draft"] is None:
        h5 = "INCONCLUSIVE (no GPU timing: needs the CUDA build)"
    else:
        h5 = ("TRUE (the verify's GPU idle holds the n3 draft)" if r3["gpu_idle"] >= r3["draft"]
              else f"FALSE (M1 hides only part: {100 * r3['gpu_idle'] / r3['draft']:.0f}% of the draft)")
    print(f"H5 n3: GPU idle inside the verify {fmt(r3['gpu_idle'])} ms vs draft phase {fmt(r3['draft'])} ms -> {h5}")
    print(f"SPEC3 H4: {h4} | H5: {h5}")
    res.update({"marginal_step_ms": ms_step, "marginal_draft_ms": ms_draft, "draft_share": ratio, "h4": h4, "h5": h5})
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

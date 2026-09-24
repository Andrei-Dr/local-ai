#!/usr/bin/env python3
"""spec4_analyze.py -- per-op GPU time of the verify step from nsys exports (research/spec-design-2026-09-24.md section 11: H6-H8)

usage: spec4_analyze.py N1.sqlite N3.sqlite [--gap-s 1.5] [--json OUT]
Each sqlite = `nsys export --type sqlite` of one STABLE server (no dump; CUDA graphs off, which nsys 2022.4 needs to see the
kernels) that served an unrecorded warm-up request, then after a >= --gap-s pause of GPU activity the measured request (the
code prompt, 200 tokens) at draft n = 1 or 3.

Which kernels are a verify step:
  window   = kernels after the last GPU-idle gap >= --gap-s (the pause the client leaves before the measured request): load,
             server warm-up and the warm-up request are before it.
  target   = the CUDA streams that run gated_delta_net (the 30 Gated DeltaNet layers exist only in the target).
  draft    = streams that run flash attention but no gated_delta_net (the MTP head: one attention + MoE block, its own context
             and stream; its draft decodes and the catch-up decode of each verify batch). Any other stream is reported with its
             kernel time per step and neither counted nor used to split.
  steps    = the target-stream kernels split wherever a draft-stream kernel starts between two of them (every verify is followed
             by the MTP catch-up and the next draft on the draft stream). The first target group of the window is the prompt
             pass (prefill) and is dropped; the rest are verify steps.
Per verify step: GPU busy = union of its kernels' intervals; per op class = the sum of its kernels' durations (class map
CLASSES below, first match on the demangled name wins; unmapped kernels are printed with their time); host launch time = the
summed duration of the cudaLaunchKernel* runtime calls whose correlationId matches the step's kernels.
Per extra draft position = (mean at n3 - mean at n1) / 2 (a verify batch is n + 1 tokens).
Rules (pre-registered, section 11):
  H6: the op classes whose GPU time grows from n1 to n3 account for >= 80% of the busy growth (attribution complete).
      -> sum of the growth of the growing MAPPED classes / growth of GPU busy >= 0.8.
  H7: the GDN layers are >= 50% of the per-position GPU growth -> the chunked / parallel GDN form for small batches becomes the top
      lead (it serves the verify AND prefill). If < 25%, GDN is not the target; the top grower is.
  H8: host launch time per verify grows >= 1.5 ms per position -> launch overhead is a real share (CUDA graphs / fusion lead).
      (Measured with CUDA graphs off, as nsys needs: an upper bound on the served launch cost.)
"""
import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict

# name -> class map, first match on the demangled kernel name wins. mul_mat_vec_q is shared by the dense mat-muls and the expert
# cache chain's mul_mat_id: split by the weight type in the template (the served K2q6 has experts Q2_K (10) gate/up and Q3_K (11)
# down; every dense weight is Q4_K or wider); mul_mat_vec_q_moe is the multi-token mul_mat_id kernel (cache chain only).
CLASSES = [
    ("GDN (gated delta rule + conv)", r"gated_delta_net|ssm_conv|ssm_scan"),
    ("cache-chain mul_mat_id", r"mul_mat_vec_q_moe|mul_mat_(vec_)?q<(\(ggml_type\)1[01]|GGML_TYPE_Q[23]_K)[,>]"),
    ("dense mul_mat", r"mul_mat_vec_q|mul_mat_q|mul_mat_vec_f|mul_mat_f|gemm|gemv|cutlass|cublas"),
    ("quantize (mat-mul inputs)", r"quantize_q8_1|quantize_mmq|quantize_"),
    ("flash attention", r"flash_attn|fattn"),
    ("norms / elementwise", r"norm|rope|bin_bcast|unary|scale|soft_max|topk|argsort|glu|silu|sigmoid|softplus|gelu|clamp|exp|"
                            r"cumsum|sum_rows|sum|mean|add|mul|sub|div|sqr|sqrt|fill|arange|pad|repeat|leaky|tanh|neg|step"),
    ("copies", r"get_rows|set_rows|cpy|copy|concat|dup|convert|transpose|permute|cont"),
]


def classify(name):
    return next((c for c, rx in CLASSES if re.search(rx, name)), None)


def union(iv):
    tot, cur_s, cur_e = 0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return tot + (cur_e - cur_s if cur_e is not None else 0)


def analyze(path, gap_s):
    c = sqlite3.connect(path)
    K = c.execute("select k.start, k.end, k.streamId, k.correlationId, s.value from CUPTI_ACTIVITY_KIND_KERNEL k "
                  "join StringIds s on s.id = k.demangledName order by k.start").fetchall()
    if not K:
        sys.exit(f"{path}: no kernels (CUDA graphs on?)")
    launch = dict(c.execute("select r.correlationId, r.end - r.start from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s "
                            "on s.id = r.nameId where s.value like 'cudaLaunchKernel%'").fetchall())
    # window: after the last gap >= gap_s between consecutive kernels (by start, against the running max end)
    lo, run_end = 0, K[0][1]
    for i in range(1, len(K)):
        if K[i][0] - run_end >= gap_s * 1e9:
            lo = i
        run_end = max(run_end, K[i][1])
    W = K[lo:]
    target = {st for _, _, st, _, n in W if re.search(r"gated_delta_net", n)}
    if not target:
        sys.exit(f"{path}: no gated_delta_net kernel in the window: cannot find the target stream")
    draft = {st for _, _, st, _, n in W if re.search(r"flash_attn|fattn", n)} - target
    if not draft:
        sys.exit(f"{path}: no draft stream (flash attention outside the target stream) in the window")
    groups, cur = [], []
    other = defaultdict(float)
    for s, e, st, corr, n in W:
        if st in target:
            cur.append((s, e, corr, n))
            continue
        other[("draft " if st in draft else "unassigned ") + str(st)] += e - s
        if st in draft and cur:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    steps = groups[1:]  # groups[0] = the prompt pass
    if not steps:
        sys.exit(f"{path}: no verify steps found")
    per_class, unmapped = defaultdict(float), defaultdict(lambda: [0.0, 0])
    busy = launch_t = 0.0
    for g in steps:
        busy += union([(s, e) for s, e, _, _ in g])
        for s, e, corr, n in g:
            cls = classify(n)
            if cls is None:
                u = unmapped[n[:90]]
                u[0] += e - s
                u[1] += 1
                cls = "unmapped"
            per_class[cls] += e - s
            launch_t += launch.get(corr, 0)
    ns = len(steps)
    return {"steps": ns, "busy_ms": busy / ns / 1e6, "launch_ms": launch_t / ns / 1e6,
            "kernels_per_step": sum(len(g) for g in steps) / ns,
            "class_ms": {k: v / ns / 1e6 for k, v in per_class.items()},
            "unmapped": {k: (v[0] / ns / 1e6, v[1] / ns) for k, v in unmapped.items()},
            "other_streams_ms_per_step": {k: v / ns / 1e6 for k, v in other.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("n1")
    ap.add_argument("n3")
    ap.add_argument("--gap-s", type=float, default=1.5)
    ap.add_argument("--json")
    a = ap.parse_args()
    r1, r3 = analyze(a.n1, a.gap_s), analyze(a.n3, a.gap_s)
    for name, r in (("n1", r1), ("n3", r3)):
        print(f"{name}: {r['steps']} verify steps | GPU busy {r['busy_ms']:.2f} ms/step | host launch {r['launch_ms']:.2f} ms/step"
              f" | {r['kernels_per_step']:.0f} kernels/step | other streams (ms/step): {', '.join(f'{k} {v:.2f}' for k, v in r['other_streams_ms_per_step'].items())}")
        for k, (t, cnt) in sorted(r["unmapped"].items(), key=lambda kv: -kv[1][0]):
            print(f"    UNMAPPED {t:.3f} ms/step ({cnt:.1f}/step): {k}")
    pos = 2.0  # n3 verifies 4 tokens, n1 verifies 2
    dbusy = (r3["busy_ms"] - r1["busy_ms"]) / pos
    classes = sorted(set(r1["class_ms"]) | set(r3["class_ms"]))
    growth = {k: (r3["class_ms"].get(k, 0) - r1["class_ms"].get(k, 0)) / pos for k in classes}
    print(f"per extra draft position: GPU busy {dbusy:+.3f} ms | host launch {(r3['launch_ms'] - r1['launch_ms']) / pos:+.3f} ms")
    print(f"  {'op class':32s} {'n1 ms':>8s} {'n3 ms':>8s} {'per position':>13s} {'share of busy growth':>21s}")
    for k in sorted(classes, key=lambda k: -growth[k]):
        share = 100 * growth[k] / dbusy if dbusy else float("nan")
        print(f"  {k:32s} {r1['class_ms'].get(k, 0):8.3f} {r3['class_ms'].get(k, 0):8.3f} {growth[k]:+13.3f} {share:20.1f}%")
    if dbusy <= 0:
        h6 = h7 = "INCONCLUSIVE (GPU busy does not grow from n1 to n3)"
    else:
        att = sum(g for k, g in growth.items() if g > 0 and k != "unmapped") / dbusy
        h6 = ("TRUE" if att >= 0.8 else "FALSE") + f" (growing mapped classes = {100 * att:.0f}% of the busy growth)"
        gdn = growth.get(CLASSES[0][0], 0.0) / dbusy
        top = max((k for k in growth if k != "unmapped"), key=lambda k: growth[k])
        if gdn >= 0.5:
            h7 = f"TRUE (GDN {100 * gdn:.0f}% of the per-position growth: chunked / parallel GDN is the top lead)"
        elif gdn < 0.25:
            h7 = f"FALSE (GDN {100 * gdn:.0f}%: not the target; the top grower is {top})"
        else:
            h7 = f"BETWEEN (GDN {100 * gdn:.0f}%; top grower {top})"
    dl = (r3["launch_ms"] - r1["launch_ms"]) / pos
    h8 = ("TRUE" if dl >= 1.5 else "FALSE") + f" (host launch {dl:+.3f} ms per position, CUDA graphs off)"
    print(f"SPEC4 H6: {h6}")
    print(f"SPEC4 H7: {h7}")
    print(f"SPEC4 H8: {h8}")
    if a.json:
        json.dump({"n1": r1, "n3": r3, "busy_growth_per_position": dbusy, "class_growth_per_position": growth,
                   "h6": h6, "h7": h7, "h8": h8}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

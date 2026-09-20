"""jobreport.py JOB [--ledger bench/box/ledger.jsonl] [--out bench/reports/JOB.md] -- hypothesis-driven A/B packs.
The registry declares each experiment job (title, hypothesis, comparisons with per-metric tables); the report is
generated, never handwritten, so pending arms show up as pending instead of as silence. Exit 0 unless the job name
is unknown."""
import json, re, sys
from pathlib import Path

import abreport

HERE = Path(__file__).parent
JOBS = {
    "p4c1": {
        "title": "Prompt warm-up of the expert cache (P4c), N2-lite, V1 VRAM diet",
        "hyp": "Warming the expert cache during prompt processing removes the cold-miss uploads from decoded "
               "tokens (P4c); capping n-gram drafts at the GDN rollback depth avoids checkpoint+replay (N2-lite); "
               "keeping only head experts resident frees VRAM at a modest slot cost (V1). Each comparison below "
               "should move decode tok/s above the repeat-spread noise floor if the mechanism pays on this box.",
        "cmp": [
            ("P4c Qwen3.6 warm 0 vs 32", r"^p4c_q36_warm0_", r"^p4c_q36_warm32_", ["decode_tps", "wall_tps"]),
            ("P4c Gemma Q2_K_P warm 0 vs 32", r"^p4c_g4q2k_warm0_", r"^p4c_g4q2k_warm32_", ["decode_tps", "wall_tps"]),
            ("N2-lite n-gram cap 2 (c30) vs cap 3 (c29)", r"^n2_q36_c30_ngmod2$", r"^n2_q36_c29_ngmod3$", ["decode_tps", "acceptance"]),
            ("V1 head experts on CPU, 40 slots vs warm-32 baseline (30 slots)", r"^p4c_q36_warm32_", r"^v1_q36_c40_mtp2_otd$", ["decode_tps"]),
            ("V1 38 slots", r"^p4c_q36_warm32_", r"^v1_q36_c38_mtp2_otd$", ["decode_tps"]),
        ],
    },
    "lat1": {
        "title": "Latency / transfer levers",
        "hyp": "Each toggleable lever (graphs off, pinned transfers, upload batching, CPU prefill, partial "
               "offload, OpenMP parking, spin cadence, stream priority) trades in one known direction against a "
               "constant base. The table answers per lever, at repeat-noise resolution, whether the trade pays on "
               "this hardware — decode tok/s and prefill tok/s separately, since levers can move one opposite to "
               "the other.",
        "cmp": [(f"Lever: {lab}", r"^lat_base$", rf"^{lab}$", ["decode_tps", "prefill_tps"]) for lab in
                ("lat_nographs", "lat_pinned", "lat_ub256_c28", "lat_ub512_c24", "lat_cpu_prefill",
                 "lat_offload8", "lat_omp_passive", "lat_spin1k", "lat_prio2")],
    },
    "kq1": {
        "title": "K-quant experts for Qwen3.6",
        "hyp": "K-quant (Q2_K) expert tensors roughly halve transferred expert bytes versus the IQ2_M expert "
               "store; prefill — transfer-bound on/offload — should gain near that ratio, while net decode gain "
               "depends on miss-cost falling faster than the per-expert dequant penalty grows, with MTP n=3 "
               "buying more once per-token work got cheaper.",
        "cmp": [
            ("miss cost, no cache", r"^kq_q36iq2m_off$", r"^kq_q36k2_off$", ["decode_tps", "prefill_tps"]),
            ("best config", r"^kq_q36iq2m_c30_mtp2$", r"^kq_q36k2_c26_mtp2$", ["decode_tps", "acceptance"]),
            ("does n=3 pay now", r"^kq_q36k2_c26_mtp2$", r"^kq_q36k2_c24_mtp3$", ["decode_tps", "acceptance"]),
        ],
        "qual": ["q36_k2_cache26"],
    },
    "mtp1": {
        "title": "In-model MTP head",
        "hyp": "The standalone MTP head squats on ~727 MiB of VRAM (duplicate output.weight + its own 256 experts). "
               "Grafting blk.40 into the target file hands that VRAM back; spent on cache slots it should buy 2-4% "
               "near the hit-rate curve, and speculation stays lossless (textdiff gates every equal-slot pair). "
               "Kill: in-model within noise at equal slots AND no gain when the freed VRAM is spent => stay on -md.",
        "cmp": [("same 30 slots", r"^mtp1_q36_md_c30$", r"^mtp1_q36_inmodel_c30$", ["decode_tps", "acceptance"]),
                ("freed VRAM spent: 42 slots", r"^mtp1_q36_md_c30$", r"^mtp1_q36_inmodel_c42$", ["decode_tps", "acceptance"]),
                ("freed VRAM spent: 46 slots", r"^mtp1_q36_md_c30$", r"^mtp1_q36_inmodel_c46$", ["decode_tps", "acceptance"])],
    },
    "r1": {
        "title": "Cache-aware routing",
        "hyp": "Routing logits bent toward experts already resident in the GPU cache (+bias per mainline patches "
               "0005/0006) converts CPU misses into hits during steady decode; the sweep finds the knee where extra "
               "bias stops buying hit-rate and starts costing quality. Kill: <+1% at every bias, or quality moves "
               "outside 1 sigma versus the base run.",
        "cmp": ([(f"Qwen cache bias {suf}", r"^r1_q36_b0$", rf"^r1_q36_{suf}$", ["decode_tps", "acceptance"]) for suf in ("b025", "b05", "b10", "b20")] +
                [(f"Gemma cache bias {suf}", r"^r1_g4q2k_b0$", rf"^r1_g4q2k_{suf}$", ["decode_tps", "acceptance"]) for suf in ("b05", "b10")]),
        "qual": ["q36_iq2m_cache48_r1b025", "q36_iq2m_cache48_r1b05", "q36_iq2m_cache48_r1b10"],
    },
    "n2b": {
        "title": "N2-lite repeat",
        "hyp": "The n-gram cap 3 vs 2 delta measured +4.5% ALL at n=1 — inside the 3-5% identical-run jitter M1 just "
               "quantified. REPEATS=3/WARMUP=1 with `_runs` samples decides whether N2-lite's win is real. "
               "Kill: <+2% ALL at 3x3 samples => cap stays at 2 and N2 closes.",
        "cmp": [("cap 2 vs cap 3 (3x3 samples)", r"^n2b_q36_c30_ngmod2_", r"^n2b_q36_c29_ngmod3_", ["decode_tps", "acceptance"])],
    },
    "kq1g": {
        "title": "KQ1 + in-model head",
        "hyp": "mtp1's VRAM return applied to the K2 expert-quant file: same 26 slots in-model must sit within noise "
               "of the separate head, and the reclaimed ~700 MiB spent as 38 slots must beat -md on the hit-rate "
               "curve for K2+in-model to be THE Qwen config. Kill: equal at 26 and <+1% at 38 => stay -md.",
        "cmp": [("same 26 slots", r"^kq1g_.*_md_", r"^kq1g_.*_inmodel_c26$", ["decode_tps"]),
                ("freed VRAM spent: 38 slots", r"^kq1g_.*_md_", r"^kq1g_.*_inmodel_c38$", ["decode_tps"])],
    },
}
TELEM = ("power_w_avg", "gpu_util_avg", "pcie_rx_gbs_max", "mem_avail_mib_min", "swap_used_mib_max")
QUAL_SETS = ("gsm8k", "humaneval", "mmlu_pro")
QUAL_REF = "q36_iq2m_cache48"      # quality baseline every job's qual table shows first


def latest_quality_sets(recs):
    """{(label, set): set dict} from the newest quality record CARRYING that set (reruns may cover subsets)."""
    sup = {}
    for r in recs:
        if r.get("kind") != "quality" or not r.get("quality"):
            continue
        for name, sd in (r["quality"].get("sets") or {}).items():
            if sd.get("pct") is None:
                continue
            k = (r["label"], name)
            if k not in sup or (r.get("ts") or "") >= (sup[k].get("ts") or ""):
                sup[k] = sd
    return sup


def dash(x):
    return "-" if x is None else x


def report(job_name, recs):
    """-> (markdown_text | None, exit_code)."""
    job = JOBS.get(job_name)
    if not job:
        return (f"unknown job '{job_name}' (registry: {', '.join(sorted(JOBS))})\n", 2)
    o = [f"# {job_name} — {job['title']}", "", job["hyp"], ""]
    pats = []
    for title, brx, trx, metrics in job["cmp"]:
        o += ["", f"## {title}", ""]
        pats += [brx, trx]
        for mi, m in enumerate(metrics):
            res = abreport.compare(recs, re.compile(brx), re.compile(trx), m)
            if res["empty_arm"]:
                if mi == 0:
                    o.append(f"_pending: no completed rows for {res['pattern']}_")
                break  # arms are metric-independent; no reason to re-detect per metric
            o += [f"**{m}**", "", abreport.render(res, md=True), ""]
    if job.get("qual"):
        sup = latest_quality_sets(recs)
        o += ["", "## Quality (job labels vs the reference run)", "", "| label | gsm8k | humaneval | mmlu_pro |", "|---|---|---|---|"]
        for lab in [QUAL_REF] + job["qual"]:
            cells = []
            for s in QUAL_SETS:
                sd = sup.get((lab, s))
                cells.append("-" if not sd else (f"{sd['pct']:.1f} ±{sd['se_pct']:.1f}" if sd.get("se_pct") is not None else f"{sd['pct']:.1f}"))
            o.append(f"| `{lab}` | " + " | ".join(cells) + " |")
    o += ["", "## Runs seen", ""]
    latest = abreport.latest_completed(recs)
    rx = [re.compile(p) for p in pats]
    seen = 0
    for lab, r in sorted(latest.items()):
        if not any(x.search(lab) for x in rx):
            continue
        seen += 1
        t = r.get("telemetry") or {}
        hit = (r.get("moe_cache") or {}).get("hit_rate_pct")
        o.append(f"- `{lab}` | vram {dash(r.get('vram_mib'))} | hit {dash(hit)} | "
                 + " | ".join(f"{k} {dash(t.get(k))}" for k in TELEM))
    if not seen:
        o.append("_none yet_")
    return "\n".join(o) + "\n", 0


def main():
    argv = sys.argv[1:]
    if not argv:
        print("usage: jobreport.py JOB [--ledger PATH] [--out PATH]", file=sys.stderr)
        sys.exit(2)
    job, ledger, out = argv[0], HERE / "box" / "ledger.jsonl", None
    i = 1
    while i < len(argv):
        if argv[i] == "--ledger":
            i += 1
            ledger = Path(argv[i])
        elif argv[i] == "--out":
            i += 1
            out = Path(argv[i])
        else:
            print("usage: jobreport.py JOB [--ledger PATH] [--out PATH]", file=sys.stderr)
            sys.exit(2)
        i += 1
    if job not in JOBS:                       # unknown job exits before touching the ledger
        text, rc = report(job, [])
        sys.stderr.write(text)
        sys.exit(rc)
    recs = [json.loads(l) for l in open(ledger, encoding="utf-8") if l.strip()] if Path(ledger).exists() else []
    text, rc = report(job, recs)
    if rc:
        sys.stderr.write(text)
        sys.exit(rc)
    out = Path(out) if out else HERE / "reports" / f"{job}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"{out} written")


if __name__ == "__main__":
    main()

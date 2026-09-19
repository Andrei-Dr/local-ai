#!/usr/bin/env python3
"""Rebuild bench/ledger_backfill.jsonl: every measurement taken BEFORE the harness wrote /ai/bench/ledger.jsonl.

Sources, best first:
  1. box/server_<label>.log   per-task timings (prefill/decode tok/s, draft acceptance) -- survives for every server run
  2. box/mon_summary.json     telemetry recomputed on the box from the raw 1 Hz files in /ai/bench/mon (mon_resummarize.py)
  3. box/{moe1,moe2,moe3,round2,validate75}.log   sweep logs: exact args (##### headers), VRAM, CPU busy
  4. NOTES below              numbers that only ever existed in notes.md / llama-bench tables
Records use the same schema as box/ledger.py, with "source" telling where each one came from.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
BOX = HERE / "box"

BONSAI = "Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP.gguf"
QWEN = "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf"
GEMMA = "Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ3_M.gguf"
FFN_CPU = r'-ot "blk\.([0-9]|[1-5][0-9]|6[0-3])\.ffn_(gate|up|down)\.weight=CPU"'
FIXED = "-fa on -c 4096 -t 6 --load-mode none --jinja --parallel 1; 2 prompts, 200 tok, temp 0, thinking off"


def context(label):
    """(model, build, branch, commit, note) for a run label."""
    if label.startswith("p1_"):
        return QWEN, "/ai/src/llama.cpp-moecache/build75", "moe-cache", "8c8650b", None
    if label.startswith("p2_"):
        return (GEMMA if "_g4_" in label else QWEN), "/ai/src/llama.cpp-moecache/build75", "moe-cache", "0e4103f", \
            "ran from the uncommitted tree that became 0e4103f, WITHOUT the free-slot cold-start fix"
    if label.startswith(("q36_", "g4_")):
        return (GEMMA if label.startswith("g4_") else QWEN), "/ai/src/llama.cpp/build75", "i5-tuning", "290857f", None
    if label == "b75_best":
        return BONSAI, "/ai/src/llama.cpp/build75", "i5-tuning", "290857f", None
    note = "61-virtual + FORCE_MMQ + LTO build"
    if "pr27781" in label:
        note += "; mainline #27781 cherry-picked on top (no effect, reverted)"
    return BONSAI, "/ai/src/llama.cpp/build", "i5-tuning", "b06d088", note


def legacy_args(label):
    """Server args for labels that predate the '##### label | args' sweep headers."""
    a, env, offload = [], {}, 2
    m = re.search(r"ngl(\d+)", label)
    if label.startswith("ffncpu") or label == "b75_best":
        a += ["-ngl 99", FFN_CPU]
    elif m:
        a.append(f"-ngl {m.group(1)}")
    else:
        a.append("-ngl ? (first attempt of the ngl fallback loop)")
    if "ub128" in label or label == "b75_best":
        a.append("-ub 128 -b 256")
    n = re.search(r"mtp_n(\d+)", label)
    if n or label == "b75_best":
        a.append(f"--spec-type {'ngram-mod,' if label.startswith('ngram+') else ''}draft-mtp --spec-draft-n-max {n.group(1) if n else 2}")
    elif label.startswith("ngram_mod"):
        a.append("--spec-type ngram-mod")
    if "kq8" in label:
        a.append("-ctk q8_0")
    if "omppassive" in label:
        env["OMP_WAIT_POLICY"] = "passive"
    if label.startswith("nospec") or label == "ffncpu_nospec":
        offload = None  # not recorded for the no-spec baselines
    return " ".join(a), env, offload


def parse_sweeps():
    """label -> {args, offload, vram_mib, cpu_busy_pct, died} from the sweep logs."""
    out = {}
    for name in ("round2", "validate75", "moe1", "moe2", "moe3"):
        p = BOX / f"{name}.log"
        if not p.exists():
            continue
        for line in p.read_text(errors="replace").splitlines():
            h = re.match(r"##### (\S+) \|(?: OFFLOAD=(\d+) \|)? (.*)$", line)
            if h:
                d = out.setdefault(h.group(1), {})
                d["args"] = "-ngl 999 " + h.group(3)
                d["offload"] = int(h.group(2)) if h.group(2) else 32
                d["sweep"] = name
                continue
            r = re.match(r"(\S+)\s+(code|reason)\s+\d+ tok \|.*\| vram (\d+) MiB", line)
            if r:
                out.setdefault(r.group(1), {}).update(vram_mib=int(r.group(3)), sweep=name)
                continue
            t = re.match(r"\s+telemetry\[(\S+)\]:.*CPU busy\s+(\d+)%", line)
            if t:
                out.setdefault(t.group(1), {})["cpu_busy_pct"] = int(t.group(2))
                continue
            x = re.match(r"(\S+): SERVER DIED: (.*)$", line)
            if x:
                out.setdefault(x.group(1), {})["died"] = x.group(2)
    return out


def parse_server_log(path):
    txt = path.read_text(errors="replace")
    tasks = {}
    for m in re.finditer(r"task (\d+) \| prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens \(.*?([\d.]+) tokens per second\)", txt):
        tasks.setdefault(m.group(1), {}).update(prompt_tokens=int(m.group(3)), prefill_tps=float(m.group(4)))
    for m in re.finditer(r"task (\d+) \|\s+eval time =\s*([\d.]+) ms /\s*(\d+) tokens \(.*?([\d.]+) tokens per second\)", txt):
        tasks.setdefault(m.group(1), {}).update(tokens=int(m.group(3)), decode_tps=float(m.group(4)))
    for m in re.finditer(r"task (\d+) \| draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\)", txt):
        tasks.setdefault(m.group(1), {}).update(acceptance=round(float(m.group(2)), 3), draft_accepted=int(m.group(3)), draft_n=int(m.group(4)))
    rows = []
    for kind, (_, t) in zip(("code", "reason"), sorted(tasks.items(), key=lambda kv: int(kv[0]))):
        if "decode_tps" in t:
            rows.append({"prompt": kind, **t})
    err = re.findall(r"(?i)[^\n]*(?:out of memory|failed to allocate|CUDA error)[^\n]*", txt)
    ts = None
    return rows, (err[-1][-200:] if err else None), ts


# measurements that never went through the server harness (llama-bench, perf, one-off numbers in notes.md)
NOTES = [
    {"label": "bonsai_scalar_kernel", "kind": "llama-bench", "model": "Ternary-Bonsai-2-27B-Abliterated-PQ2_0.gguf", "commit": "bac645e", "build": "/ai/src/llama.cpp/build (arch 75, pre-AVX2)",
     "args": "-ngl 24 -fa 1 -t 6 --load-mode none", "results": {"pp1": 0.69, "pp2": 0.70, "pp3": 0.75, "pp4": 0.76, "pp8": 0.78, "tg24": 0.71},
     "telemetry": {"vram_mib": 3454, "gpu_util_avg": 7}, "note": "PQ2_0 x Q8_0 vec_dot ran scalar (no VNNI); CPU-only tg 0.47"},
    {"label": "bonsai_avx2_kernel", "kind": "llama-bench", "model": "Ternary-Bonsai-2-27B-Abliterated-PQ2_0.gguf", "commit": "b06d088", "build": "/ai/src/llama.cpp/build (arch 75)",
     "args": "-ngl 24 -fa 1 -t 6 --load-mode none", "results": {"tg": 2.99, "tg_cpu_only": 1.97, "tg_t12": "no better than -t 6"},
     "note": "AVX2 kernel 38.5 -> 5.6 cycles/32 weights; perf: vec_dot 56%, libgomp spin 20%, mul 6%; DRAM ~13-15 of 38 GB/s"},
    {"label": "bonsai_streamed_verify_arch75", "kind": "llama-bench", "model": "Ternary-Bonsai-2-27B-Abliterated-PQ2_0.gguf", "commit": "b06d088", "build": "arch 75, no FORCE_MMQ",
     "args": "-ngl 24 -fa 1 -t 6 -p 1,2,4,8,16,32 -n 0", "results_ms_per_pass": {"default_offload32": {"1": 334, "2": 676, "4": 1036, "8": 1743, "16": 3313, "32": 1354},
                                                                                   "offload2": {"1": 334, "2": 519, "4": 558, "8": 649, "16": 946, "32": 1364}},
     "note": "GGML_OP_OFFLOAD_MIN_BATCH=2 streams CPU-resident weights over PCIe; verifying 8 tokens costs 1.94x one token"},
    {"label": "bonsai_streamed_verify_builds", "kind": "llama-bench", "model": "Ternary-Bonsai-2-27B-Abliterated-PQ2_0.gguf", "commit": "290857f", "build": "build (61-virtual+MMQ+LTO) vs build75 (75+MMQ)",
     "args": "GGML_OP_OFFLOAD_MIN_BATCH=2 -ngl 24 -fa 1 -t 6 -p 2,4,8,16 -n 0 -r 2", "results_ms_per_pass": {"build_61virtual": {"2": 519, "4": 558, "8": 650, "16": 694}, "build75": {"2": 531, "4": 570, "8": 659, "16": 955}},
     "note": "batch-16 win comes from the Pascal code path; LTO: no effect (CPU-only 2.02 vs 1.97 tok/s)"},
    {"label": "ptq1_vs_pq2_streamed", "kind": "llama-bench", "model": "PQ2_0 vs PTQ1_0 (abliterated)", "commit": "290857f", "build": "/ai/src/llama.cpp/build75",
     "args": "GGML_OP_OFFLOAD_MIN_BATCH=2 -ngl 24", "results_ms_per_pass": {"PQ2_0": {"2": 518, "4": 559, "8": 649, "tg_tps": 2.94}, "PTQ1_0": {"2": 539, "4": 694, "8": 815, "tg_tps": 0.58}},
     "note": "PTQ1_0 slower despite 28% fewer bytes: GPU-side unpack compute-bound, CPU kernel scalar. Graft dropped."},
    {"label": "moe_thread_sweep", "kind": "llama-bench", "model": "Qwen3.6 IQ2_M / Gemma4 IQ3_M", "commit": "290857f", "build": "/ai/src/llama.cpp/build75",
     "args": '-ngl 99 -ot "exps=CPU" -fa 1 -p 0 -n 64 -r 2', "results_tps": {"qwen36": {"t2": 14.15, "t3": 18.93, "t4": 22.40, "t5": 25.20, "t6": 26.05, "t6_no_cuda_graphs": 26.63},
                                                                             "gemma4": {"t3": 11.06, "t4": 13.66, "t5": 15.40, "t6": 16.72, "t6_no_cuda_graphs": 16.95}},
     "note": "fit T=a+b/t: Qwen 18.5 ms fixed + ~20 ms CPU experts; Gemma 21.7 + ~38. perf: iq2_s vec_dot 49% / iq3_s 49% + iq4_nl 12%; gomp spin 27-35%"},
]


def main():
    sweeps = parse_sweeps()
    mon = json.loads((BOX / "mon_summary.json").read_text()) if (BOX / "mon_summary.json").exists() else {}
    recs = []
    for p in sorted(BOX.glob("server_*.log")):
        label = p.stem[len("server_"):]
        if label.startswith(("qual_", "p2b_")):
            continue  # quality runs have their own rows; p2b_* and later are written by the harness itself
        model, build, branch, commit, note = context(label)
        rows, err, _ = parse_server_log(p)
        sw = sweeps.get(label, {})
        if "args" in sw:
            args, env, offload = sw["args"], {}, sw["offload"]
        else:
            args, env, offload = legacy_args(label)
        tel = dict(mon.get(label) or {})
        if "cpu_busy_pct" in sw:
            tel["cpu_busy_pct"] = sw["cpu_busy_pct"]
        rec = {"ts": tel.pop("ts", None), "label": label, "kind": "specbench", "source": "backfill: server log + raw telemetry" + (f" + {sw['sweep']}.log" if "sweep" in sw else ""),
               "model": model, "build": build, "git": {"branch": branch, "commit": commit, "dirty": label.startswith("p2_") or "pr27781" in label},
               "args": args, "env": env or None, "offload_min_batch": offload, "fixed": FIXED,
               "vram_mib": sw.get("vram_mib"), "prompts": rows or None, "telemetry": tel or None, "moe_cache": None,
               "completed": bool(rows), "note": note}
        m = re.search(r"--moe-expert-cache (\d+)", args)
        if m:
            a = re.search(r"--moe-expert-cache-admit (\d+)", args)
            i = re.search(r"--moe-expert-cache-inserts (\d+)", args)
            gated = label.startswith("p2_")
            rec["moe_cache"] = {"slots": int(m.group(1)), "inserts": int(i.group(1)) if i else 2,
                                "admit": int(a.group(1)) if a else (3 if gated else 1), "window": 16 if gated else 0,
                                "hit_rate_pct": None, "note": "stats lines were not visible at the default server verbosity"}
        if not rows:
            rec["error"] = sw.get("died") or err
        recs.append(rec)
    for n in NOTES:
        recs.append({"ts": None, "source": "backfill: notes.md / llama-bench logs", "completed": True,
                     "git": {"branch": "i5-tuning", "commit": n.pop("commit"), "dirty": False}, **n})
    out = HERE / "ledger_backfill.jsonl"
    with out.open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    done = sum(1 for r in recs if r["completed"])
    print(f"{out}: {len(recs)} records ({done} completed, {len(recs) - done} failed/OOM), {sum(1 for r in recs if r.get('telemetry'))} with telemetry")


if __name__ == "__main__":
    main()

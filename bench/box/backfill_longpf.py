#!/usr/bin/env python3
"""backfill_longpf.py -- one-off: append a kind "longpf" ledger record for every runs/*.longpf.json written before longpf.py
recorded itself (2026-09-23 09:30). Build / env / args per label are transcribed from the job scripts that produced them
(bench/box/<job>.sh); git provenance is NOT reconstructable (the ov tree was re-patched between jobs) -> git None,
source "backfill". Idempotent: labels already in the ledger are skipped."""
import json, os, re, time

B = "/ai/bench"
OLD, OV, V2 = "/ai/src/llama.cpp-mainline/build75", "/ai/src/llama.cpp-ov/build75", "/ai/src/llama.cpp-v2/build75"
K2 = "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-K2-expQ2K-downQ3K.gguf"
BASE = "-ngl 999 -c 12288 -ot exps=CPU --moe-expert-cache 22 -md mtp-Qwen3.6-35B-A3B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 2"
# label regex -> (build, env, extra args, source script)
TABLE = [
    (r"pmux3_base(_2)?$",        OLD, "",                                                     "-ub 128 -b 256",             "pmux3"),
    (r"pmux3_ubp(\d+)(_2)?$",    OV,  "",                                                     "-ub 128 -b 4096 -ubp {1}",   "pmux3"),
    (r"mmqdp1_ubp(\d+)$",        OV,  "",                                                     "-ub 128 -b 4096 -ubp {1}",   "mmqdp1"),
    (r"pfprof1_dp4a$",           OV,  "GGML_CUDA_DISABLE_GRAPHS=1 (nsys)",                    "-ub 128 -b 4096 -ubp 4096",  "pfprof1"),
    (r"tu1_ubp4096_old$",        OV,  "GGML_SCHED_MOE_READBACK=1 GGML_SCHED_SYNC_BEFORE_COPY=1", "-ub 128 -b 4096 -ubp 4096", "tu1"),
    (r"tu1_ubp4096_rb$",         OV,  "GGML_SCHED_MOE_READBACK=1",                            "-ub 128 -b 4096 -ubp 4096",  "tu1"),
    (r"tu1_ubp4096_full$",       OV,  "",                                                     "-ub 128 -b 4096 -ubp 4096",  "tu1"),
    (r"tu1_ubp4096_fa$",         OV,  "GGML_CUDA_FA_NO_MMA=1",                                "-ub 128 -b 4096 -ubp 4096",  "tu1"),
    (r"tu1_ub128$",              OV,  "",                                                     "-ub 128 -b 4096",            "tu1"),
    (r"tu1_ub128_expcols$",      OV,  "GGML_CUDA_MMQ_MOE_EXPERT_COLS=1",                      "-ub 128 -b 4096",            "tu1"),
    (r"tu1_ubp4096_all$",        OV,  "GGML_CUDA_FA_NO_MMA=1 GGML_CUDA_MMQ_MOE_EXPERT_COLS=1", "-ub 128 -b 4096 -ubp 4096", "tu1"),
    (r"tu2_pf_T$",               OV,  "GGML_CUDA_FA_TILE_MIN_BATCH=32",                       "-ub 128 -b 4096 -ubp 4096",  "tu2"),
    (r"promo1_pf_old$",          OLD, "",                                                     "-ub 128 -b 256",             "promo1"),
    (r"promo1_pf_new$",          V2,  "GGML_CUDA_FA_TILE_MIN_BATCH=32",                       "-ub 128 -b 4096 -ubp 2048",  "promo1"),
    (r"dec1_pf_[PT]$",           OV,  "GGML_CUDA_FA_TILE_MIN_BATCH=32",                       "-ub 128 -b 4096 -ubp 2048",  "dec1"),
    (r"dec1_pf_O$",              OV,  "GGML_CUDA_FA_TILE_MIN_BATCH=32",                       "-ub 128 -b 256",             "dec1"),
    (r"dec[23]_pf_[PT]_[abc]$",  OV,  "GGML_CUDA_FA_TILE_MIN_BATCH=32",                       "-ub 128 -b 4096 -ubp 2048",  "dec2/dec3"),
]
have = {json.loads(l)["label"] for l in open(f"{B}/ledger.jsonl") if l.strip()}
n = 0
for f in sorted(os.listdir(f"{B}/runs")):
    if not f.endswith(".longpf.json"):
        continue
    label = f[: -len(".longpf.json")]
    if label in have:
        continue
    for rx, build, env, extra, job in TABLE:
        m = re.match(rx, label)
        if m:
            break
    else:
        print(f"SKIP (no mapping): {label}")
        continue
    if label.startswith(("dec1_pf_T", "dec2_pf_T", "dec3_pf_T")):
        env = (env + " LLAMA_MOE_WARM_TAIL=128").strip()
    row = json.load(open(f"{B}/runs/{f}"))
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(os.path.getmtime(f"{B}/runs/{f}"))), "label": label,
           "kind": "longpf", "source": "backfill", "model": K2, "build": build, "git": None,
           "args": f"{BASE} {extra.replace('{1}', m.group(1)) if '{1}' in extra else extra}",
           "offload_min_batch": 32, "env": env or None, "note": f"backfilled from {job}.sh",
           "fixed": "-fa on -t 6 --load-mode none --jinja --parallel 1 --cache-ram 0; longpf.py: 9,279-token prose prompt (40,000 chars), 128 tok decode, temp 0, thinking off",
           "longpf": row, "completed": bool(row.get("prefill_tps"))}
    open(f"{B}/ledger.jsonl", "a").write(json.dumps(rec) + "\n")
    n += 1
print(f"backfilled {n} longpf records")

"""ledger.py LABEL -- append one record for a finished specbench run to /ai/bench/ledger.jsonl.
Reads env MODEL, BUILD, OFFLOAD, LEDGER_ARGS (set by specbench.sh), runs/LABEL.{client,mon}.json and the server log."""
import json, os, re, subprocess, sys, time
label = sys.argv[1]
B = "/ai/bench"
def sh(*a):
    try: return subprocess.run(a, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception: return ""
def load(p):
    try: return json.load(open(p))
    except Exception: return None
build = os.environ.get("BUILD", "/ai/src/llama.cpp/build")
src = os.path.dirname(build.rstrip("/"))
model = os.environ.get("MODEL", "")
log = open(f"{B}/server_{label}.log", errors="replace").read() if os.path.exists(f"{B}/server_{label}.log") else ""
def last(rx):
    m = re.findall(rx, log); return m[-1] if m else None
cache = None
en = last(r"MoE expert cache enabled: (\d+) layers x (\d+) slots, (\d+) inserts/step(?:, admit (\d+) misses / (\d+) tokens)?, ([\d.]+) MiB")
if en:
    st = last(r"moe-cache: steps (\d+) \| hit rate ([\d.]+)% \| uploads (\d+) \(([\d.]+) MiB, ([\d.]+) MiB/step\) \| evictions (\d+)")
    cache = {"layers": int(en[0]), "slots": int(en[1]), "inserts": int(en[2]), "admit": int(en[3] or 1), "window": int(en[4] or 0), "vram_mib": float(en[5])}
    if st: cache.update({"steps": int(st[0]), "hit_rate_pct": float(st[1]), "uploads": int(st[2]), "upload_mib": float(st[3]), "upload_mib_per_step": float(st[4]), "evictions": int(st[5])})
client = load(f"{B}/runs/{label}.client.json") or {}
quality = load(f"{B}/qual/results/{label}.summary.json") if "--quality" in sys.argv else None
if quality is not None or "--quality" in sys.argv:
    log = open(f"{B}/server_qual_{label}.log", errors="replace").read() if os.path.exists(f"{B}/server_qual_{label}.log") else ""
rec = {
    "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "label": label, "kind": "quality" if "--quality" in sys.argv else "specbench", "source": "harness",
    "model": os.path.basename(model), "model_bytes": os.path.getsize(model) if os.path.exists(model) else None,
    "build": build, "git": {"branch": sh("git", "-C", src, "rev-parse", "--abbrev-ref", "HEAD"), "commit": sh("git", "-C", src, "rev-parse", "--short", "HEAD"),
                            "dirty": bool(sh("git", "-C", src, "status", "--porcelain", "--untracked-files=no"))},
    "args": os.environ.get("LEDGER_ARGS", ""), "offload_min_batch": int(os.environ.get("OFFLOAD", "2")),
    "fixed": "-fa on -c 4096 -t 6 --load-mode none --jinja --parallel 1; 2 prompts, 200 tok, temp 0, thinking off",
    "host": {"governor": open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip(),
             "thp": re.search(r"\[(\w+)\]", open("/sys/kernel/mm/transparent_hugepage/enabled").read()).group(1),
             "kernel": os.uname().release, "driver": sh("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")},
    "vram_mib": int(re.sub(r"\D", "", client.get("vram", "") or "0") or 0), "prompts": client.get("rows"),
    "telemetry": load(f"{B}/runs/{label}.mon.json"), "moe_cache": cache, "quality": quality, "completed": bool(client.get("rows")) or bool(quality),
}
if "--quality" in sys.argv:
    rec["fixed"] = "-fa on -c 4096 -t 6 --load-mode none --jinja --parallel 1; GSM8K 50 + HumanEval 41 + MMLU-Pro 70, temp 0, thinking off"
if not rec["completed"]:
    rec["error"] = (re.findall(r"(?i)[^\n]*(?:out of memory|failed to|error)[^\n]*", log) or [""])[-1][-200:]
open(f"{B}/ledger.jsonl", "a").write(json.dumps(rec) + "\n")

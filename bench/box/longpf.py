"""longpf.py LABEL CHARS -- one long-prompt request (first CHARS of corpus/prose_big.txt + a summary instruction) against the
server on 8099; prints prompt tokens, prefill t/s, then the decode t/s of 128 generated tokens (the cache state right after a
long prefill = what the next turn sees). Writes OUT/LABEL.longpf.json."""
import json, os, sys, time, urllib.request
label, chars = sys.argv[1], int(sys.argv[2])
OUT = os.environ.get("OUT", "/ai/bench/runs")
doc = open("/ai/bench/corpus/prose_big.txt", encoding="utf-8").read()[:chars]
body = json.dumps({"messages": [{"role": "user", "content": "Summarize the following text in five bullet points.\n\n" + doc}],
                   "temperature": 0, "max_tokens": 128, "chat_template_kwargs": {"enable_thinking": False}}).encode()
t0 = time.time()
d = json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:8099/v1/chat/completions", body,
                                                            {"Content-Type": "application/json"}), timeout=3600))
t = d.get("timings", {})
row = {"prompt_n": t.get("prompt_n"), "prefill_tps": t.get("prompt_per_second"), "decode_tps": t.get("predicted_per_second"),
       "predicted_n": t.get("predicted_n"), "wall_s": round(time.time() - t0, 1)}
print(f"{label}: prompt {row['prompt_n']} tok @ {row['prefill_tps']:.1f} t/s | decode {row['predicted_n']} tok @ {row['decode_tps']:.2f} t/s | wall {row['wall_s']} s", flush=True)
os.makedirs(OUT, exist_ok=True)
json.dump(row, open(os.path.join(OUT, f"{label}.longpf.json"), "w"))


def server_process(port="8099"):
    """argv + GGML_*/LLAMA_* environment of the llama-server listening on PORT (read from /proc), or None."""
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            argv = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="replace").split("\0")[:-1]
            if not argv or not argv[0].endswith("llama-server") or "--port" not in argv or argv[argv.index("--port") + 1] != port:
                continue
            env = [e for e in open(f"/proc/{pid}/environ", "rb").read().decode(errors="replace").split("\0")
                   if e.startswith(("GGML_", "LLAMA_", "LEDGER_STABLE="))]
            return os.path.realpath(f"/proc/{pid}/exe"), argv, env
        except (OSError, IndexError):
            continue
    return None


# one ledger record per run (kind "longpf"): model, build and args come from the live server, so every caller is covered
if OUT == "/ai/bench/runs":
    sp = server_process()
    if sp is None:
        print(f"{label}: WARNING no llama-server on :8099 found; no ledger record", flush=True)
    else:
        exe, argv, env = sp
        args = argv[1:]
        model = args[args.index("-m") + 1] if "-m" in args else ""
        if "-m" in args:
            del args[args.index("-m"):args.index("-m") + 2]
        offload = next((e.split("=", 1)[1] for e in env if e.startswith("GGML_OP_OFFLOAD_MIN_BATCH=")), "32")
        stable = next((e.split("=", 1)[1] for e in env if e.startswith("LEDGER_STABLE=")), None)  # set by stable.sh's stable_server
        lenv = {k: v for k, v in os.environ.items() if k != "LEDGER_STABLE"}
        lenv.update(MODEL=model, BUILD=os.path.dirname(os.path.dirname(exe)), OFFLOAD=offload, LEDGER_ARGS=" ".join(args),
                    LEDGER_ENV=" ".join(sorted(e for e in env if not e.startswith(("GGML_OP_OFFLOAD_MIN_BATCH=", "LEDGER_STABLE=")))))
        if stable:
            lenv["LEDGER_STABLE"] = stable
        import subprocess
        r = subprocess.run(["python3", "/ai/bench/ledger.py", label, "--longpf"], env=lenv, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"{label}: WARNING ledger.py failed: {r.stderr.strip()[-300:]}", flush=True)

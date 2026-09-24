"""equality_test.py -- does vLLM's native CPU KV offload (patched for the QSA ring) leave q38fn's outputs unchanged?

Runs on Dave's box (stdlib only) against the q38fn server on :8057 (/v1/completions, raw prompts, return_token_ids).
  python3 equality_test.py e0 LABEL            # E0: 3 prompts x 2 runs, greedy, 256 tokens -> results/LABEL.e0.json
  python3 equality_test.py e1 LABEL REF.e0.json [--external]
                                               # E1: preempt-and-restore mid-decode -> results/LABEL.e1.json
  python3 equality_test.py cmp A.e0.json B.e0.json   # first divergence per prompt between two E0 files (e.g. two servers)

PRE-REGISTERED RULES (written before any run; 2026-09-25)
Hypothesis: excluding the CircularBufferSpec ring group from offloading is exact: a request restored from the CPU tier after
preemption produces the same greedy token ids as an undisturbed run.
Prompts: code (argparse.py source), prose (license texts), reasoning (seeded ledger puzzle); ~15k tokens each, raw completion
  prompts, temperature 0, max_tokens 256, ignore_eos. Deviation from the brief's ~2k tokens: with the default hash unit the
  offload chunk is 3216 tokens (the hybrid layout's block size) and a restore needs >= 2 complete chunks (MTP drops the
  trailing chunk; the mamba groups need a 2-chunk window), so a 2k prompt can never be restored from CPU. ~15k tokens gives
  4 complete chunks.
E0 (determinism): each prompt twice, sequentially, alone on the server. With --reset (dev-mode servers) the GPU prefix cache
  AND the CPU tier are wiped before every run, so both runs are cold. Pass = identical ids for every prompt.
E1 (restore): stream the code prompt; after 64 generated tokens POST /reset_prefix_cache?reset_running_requests=true
  (reset_external=false): every running request is preempted (Scheduler._preempt_request) and the GPU prefix cache is wiped
  while the CPU tier is kept, so the request can only resume by recomputing or by loading from CPU. Evidence required:
  vllm:num_preemptions_total delta >= 1 AND vllm:kv_offload_load_bytes delta > 0 (a CPU->GPU load happened).
  Pass = the 256 ids equal the E0 reference ids of the same prompt.
E1 --external (control, diagnostic only): the same with reset_external=true, so the CPU tier is wiped too and the restore is
  a full recompute. It separates preemption-recompute numerics from the offload path.
Verdict: PASS = E0 deterministic AND E1 identical with >= 1 preemption and a CPU load. If E0 itself is not deterministic,
  no PASS: report the first-divergence position (tolerance: E1 must not diverge earlier than E0's own first divergence).
Kill: any mismatch, missing evidence, or server error -> roll back to the plain 1M config (q38fn_1m.sh --rollback).
Safety: E1 refuses to reset while any request other than ours is running (Dave's traffic shares the server).
"""
import hashlib, json, os, random, re, sys, threading, time, urllib.request

HOST = os.environ.get("HOST", "http://127.0.0.1:8057")
MODEL = os.environ.get("MODEL_NAME", "q38fn-mxfp4")
OUT = os.environ.get("OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
GEN, E1_AFTER, CHARS = 256, 64, 60000


def code_prompt():
    src = open("/usr/lib/python3.14/argparse.py", encoding="utf-8").read()[:CHARS]
    return "# argparse.py (excerpt)\n" + src + "\n\n# Question: explain, step by step, how ArgumentParser.parse_known_args " \
        "resolves optional and positional arguments in the code above.\n# Answer:\n"


def prose_prompt():
    d = "/usr/share/common-licenses"
    text = "\n\n".join(open(os.path.join(d, f), encoding="utf-8", errors="replace").read() for f in sorted(os.listdir(d))
                       if os.path.isfile(os.path.join(d, f)))[:CHARS]
    return "The following are software license texts.\n\n" + text + "\n\nIn plain English, summarize how these licenses " \
        "differ in what they require from someone who redistributes modified code.\n\nSummary:"


def reasoning_prompt():
    rng = random.Random(20260925)
    names = [f"account {i}" for i in range(1, 41)]
    lines, bal = [], {n: 100 for n in names}
    while sum(len(x) for x in lines) < CHARS - 400:
        a, b = rng.sample(names, 2); amt = rng.randint(1, 60)
        lines.append(f"Step {len(lines) + 1}: {a} transfers {amt} coins to {b}.")
        bal[a] -= amt; bal[b] += amt
    return "Every account starts with 100 coins. Apply the ledger below in order.\n\n" + "\n".join(lines) + \
        "\n\nQuestion: which account ends with the most coins, and how many does it have? Reason carefully, " \
        "tracking only the accounts that matter, then give the final answer.\n\nReasoning:"


PROMPTS = {"code": code_prompt, "prose": prose_prompt, "reasoning": reasoning_prompt}


def post(path, body=None, timeout=3600):
    req = urllib.request.Request(HOST + path, json.dumps(body).encode() if body is not None else b"",
                                 {"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def metrics():
    txt = urllib.request.urlopen(HOST + "/metrics", timeout=30).read().decode()
    out = {}
    for line in txt.splitlines():
        m = re.match(r"^(vllm:[a-z_]+?)(?:_total)?(?:\{[^}]*\})? ([0-9.eE+-]+)$", line)
        if m and not line.startswith("#"):
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def body(prompt, stream):
    return {"model": MODEL, "prompt": prompt, "max_tokens": GEN, "temperature": 0, "ignore_eos": True,
            "return_token_ids": True, "stream": stream, "stream_options": {"include_usage": True} if stream else None}


def run_once(prompt):
    r = post("/v1/completions", {k: v for k, v in body(prompt, False).items() if v is not None})
    c = r["choices"][0]
    return {"ids": c["token_ids"], "prompt_tokens": r["usage"]["prompt_tokens"], "text_head": c["text"][:120]}


def reset(running, external):
    q = f"/reset_prefix_cache?reset_running_requests={str(running).lower()}&reset_external={str(external).lower()}"
    for _ in range(150):
        try:
            if post(q, timeout=60).get("success"):
                return True
        except Exception as e:  # noqa: BLE001 -- keep retrying, report at the end
            print("reset error:", e, flush=True)
        time.sleep(0.2)
    return False


def first_divergence(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))


def e0(label, do_reset):
    res = {"label": label, "reset_between_runs": do_reset, "prompts": {}}
    for name, fn in PROMPTS.items():
        p = fn(); runs = []
        for _ in range(2):
            if do_reset and not reset(False, True):
                sys.exit("E0: reset_prefix_cache failed")
            t0 = time.time(); r = run_once(p); r["wall_s"] = round(time.time() - t0, 1); runs.append(r)
        fd = first_divergence(runs[0]["ids"], runs[1]["ids"])
        res["prompts"][name] = {"prompt_sha256": hashlib.sha256(p.encode()).hexdigest(), "runs": runs,
                                "identical": fd is None, "first_divergence": fd}
        print(f"E0 {name:9s} prompt {runs[0]['prompt_tokens']} tok | identical={fd is None} first_div={fd} | "
              f"{runs[0]['wall_s']}s/{runs[1]['wall_s']}s | {runs[0]['text_head'][:60]!r}", flush=True)
    res["deterministic"] = all(v["identical"] for v in res["prompts"].values())
    return res


def e1(label, ref_path, external):
    ref = json.load(open(ref_path))["prompts"]["code"]
    p = code_prompt()
    assert hashlib.sha256(p.encode()).hexdigest() == ref["prompt_sha256"], "prompt drifted from the E0 reference"
    m0 = metrics()
    if m0.get("vllm:num_requests_running", 0) > 0:
        sys.exit("E1: other requests are running; refusing to preempt Dave's traffic")
    ids, done, reset_info = [], threading.Event(), {}

    def do_reset():
        m = metrics()
        if m.get("vllm:num_requests_running", 0) > 1:
            reset_info["refused"] = "other requests running"; return
        reset_info["at_token"] = len(ids); reset_info["ok"] = reset(True, external)

    req = urllib.request.Request(HOST + "/v1/completions", json.dumps(body(p, True)).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); resetter = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            for c in d.get("choices", []):
                ids.extend(c.get("token_ids") or [])
            if resetter is None and len(ids) >= E1_AFTER:
                resetter = threading.Thread(target=do_reset); resetter.start()
    if resetter:
        resetter.join()
    time.sleep(12)  # let the stats loggers publish the step that carried the load
    m1 = metrics()
    delta = {k: m1.get(k, 0) - m0.get(k, 0) for k in ("vllm:num_preemptions", "vllm:kv_offload_load_bytes",
                                                     "vllm:kv_offload_store_bytes", "vllm:prompt_tokens")}
    fd = first_divergence(ref["runs"][0]["ids"], ids)
    res = {"label": label, "external_reset": external, "reset": reset_info, "wall_s": round(time.time() - t0, 1),
           "ids": ids, "metric_delta": delta, "identical_to_e0": fd is None, "first_divergence": fd,
           "evidence_ok": delta["vllm:num_preemptions"] >= 1 and (external or delta["vllm:kv_offload_load_bytes"] > 0)}
    print(f"E1 external={external} reset={reset_info} tokens={len(ids)} identical={fd is None} first_div={fd} "
          f"delta={delta}", flush=True)
    return res


if __name__ == "__main__":
    mode, label = sys.argv[1], sys.argv[2]
    if mode == "cmp":
        a, b = (json.load(open(f))["prompts"] for f in sys.argv[2:4])
        for name in a:
            assert a[name]["prompt_sha256"] == b[name]["prompt_sha256"], name
            print(f"cmp {name:9s} first_div={first_divergence(a[name]['runs'][0]['ids'], b[name]['runs'][0]['ids'])}")
        sys.exit(0)
    os.makedirs(OUT, exist_ok=True)
    if mode == "e0":
        out = e0(label, "--reset" in sys.argv)
    elif mode == "e1":
        out = e1(label, sys.argv[3], "--external" in sys.argv)
    else:
        sys.exit(__doc__)
    json.dump(out, open(os.path.join(OUT, f"{label}.{mode}.json"), "w"))

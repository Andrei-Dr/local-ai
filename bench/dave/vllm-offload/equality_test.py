"""equality_test.py -- does vLLM's native CPU KV offload (patched for the QSA ring) leave q38fn's outputs unchanged?

Runs on Dave's box (stdlib only) against the q38fn server on :8057 (/v1/completions, raw prompts, return_token_ids).
  python3 equality_test.py e0 LABEL            # E0 + E0h, any server           -> results/LABEL.e0.json
  python3 equality_test.py e2 LABEL            # E2, needs VLLM_SERVER_DEV_MODE=1 -> results/LABEL.e2.json
  python3 equality_test.py e1 LABEL REF.e0.json [--external]   # E1, dev mode -> results/LABEL.e1.json
  python3 equality_test.py cmp A.e0.json B.e0.json   # first divergence per prompt between two E0 files (e.g. two servers)

PRE-REGISTERED RULES (2026-09-25; revision 2, written before any run on the offload server)
Hypothesis: excluding the CircularBufferSpec ring group from offloading is exact: KV restored from the CPU tier gives the same
greedy token ids as the same restore served from the GPU prefix cache, and a preempted request restored from CPU completes.
Prompts: code (argparse.py source), prose (license texts), reasoning (seeded ledger puzzle); 13-24k tokens, raw completion
  prompts, temperature 0, max_tokens 256, ignore_eos. Deviation from the brief's ~2k tokens: with the default hash unit the
  offload chunk is 3216 tokens (the hybrid layout's block size) and a restore needs >= 2 complete chunks (MTP drops the
  trailing chunk; the mamba groups need a 2-chunk window), so a 2k prompt can never be restored from CPU.
Cache isolation: every run carries a cache_salt; a new salt = a cold run (no GPU or CPU prefix hit), a reused salt = a hit.
Why revision 2: revision 1 (commit e4a4ab8) ran E0 once on the unpatched 1M server as run1 cold / run2 same prompt. Run 2
  diverged from run 1 at tokens 0 / 4 / 43, but run 2 was a GPU prefix-cache hit, so cold-vs-cold determinism and
  hit-vs-cold were confounded. Revision 2 separates them, and adds E2 (same restore boundary, GPU vs CPU source).
E0 (determinism): each prompt twice cold (two fresh salts), sequential, alone. Pass = identical ids for every prompt.
E0h (diagnostic): a third run reusing the first salt = a GPU prefix-cache hit; first divergence vs cold is reported.
E2 (offload exactness, dev mode; revision 3, see below): logprob probes. Per prompt, a cold 256-token reference run gives
  the continuation. Probe = (prompt ids + first k continuation ids, k in 0/64/128/192), max_tokens 1, top-20 logprobs as
  token ids. Per probe four requests: A = fresh salt X (cold), B = fresh salt Y (cold; A vs B = noise floor), G = salt X
  again (GPU prefix hit), then a GPU-only /reset_prefix_cache (reset_external=false) and C = salt X again (restore from CPU).
  G and C resume at the same chunk boundary (checked: C's external hit tokens == G's GPU hit tokens) and recompute the same
  tail, so if the CPU round trip is byte-exact, G vs C differs only by the server's kernel noise.
  Distance d(P, Q) = KL over P's top-20 tokens (both renormalized; a token missing from Q gets Q's lowest listed logprob).
  Pass = every C shows vllm:kv_offload_load_bytes > 0 AND equal boundary AND median d(G, C) <= 2 x median d(A, B) AND
  max d(G, C) <= 2 x max d(A, B) AND argmax(G) != argmax(C) in no more probes than argmax(A) != argmax(B).
E1 (preempt-and-restore, dev mode): stream the code prompt (fresh salt); after 64 generated tokens POST
  /reset_prefix_cache?reset_running_requests=true (reset_external=false): every running request is preempted
  (Scheduler._preempt_request), the GPU prefix cache is wiped, the CPU tier is kept. Run twice (E1a, E1b; fresh salts).
  Evidence: vllm:num_preemptions delta >= 1 AND vllm:kv_offload_load_bytes delta > 0 in each.
  Pass = E1a ids == E0 cold ids if E0h == cold (a restore is then expected to be bit-exact end to end); otherwise (the GPU
  prefix cache itself changes outputs, so no restore path can match a cold run) Pass = E1a ids == E1b ids, with the
  divergence from cold reported next to E1 --external.
E1 --external (control, diagnostic only): the same with reset_external=true, so the CPU tier is wiped too and the restore is
  a full recompute. It separates preemption-recompute numerics from the offload path.
Verdict: PASS = E0 deterministic AND E2 pass AND E1 pass with its evidence. If E0 itself is not deterministic: no PASS on
  token ids; report first-divergence positions, and E2 (non-inferiority against the measured noise floor) carries the
  exactness verdict ("E2 PASS, ids nondeterministic").
Revision 3 (before any run on the offload server): rev-2 E0 on the unpatched server showed cold-vs-cold divergence at
  tokens 1 / 11 / 0 (the server itself is nondeterministic), so id equality cannot prove exactness; E2 moved to the
  logprob probes above. E1 unchanged: evidence + 256 tokens completed, divergence from cold reported (diagnostic).
Kill: any mismatch, missing evidence, or server error -> roll back to the plain 1M config (q38fn_1m.sh --rollback).
Safety: resets refuse to run while any request other than ours is running (Dave's traffic shares the server).
"""
import hashlib, json, math, os, random, re, sys, threading, time, urllib.request

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


def body(prompt, stream, salt):
    return {"model": MODEL, "prompt": prompt, "max_tokens": GEN, "temperature": 0, "ignore_eos": True,
            "return_token_ids": True, "stream": stream, "cache_salt": salt,
            "stream_options": {"include_usage": True} if stream else None}


WATCH = ("prefix_cache", "kv_offload_load_bytes", "kv_offload_store_bytes", "num_preemptions", "prompt_tokens")


def deltas(m0, m1):
    return {k: round(m1.get(k, 0) - m0.get(k, 0), 1) for k in sorted(set(m0) | set(m1))
            if any(w in k for w in WATCH) and not k.endswith(("_created", "_bucket", "_count", "_sum"))}


def run_once(prompt, salt):
    idle()
    m0 = metrics(); t0 = time.time()
    r = post("/v1/completions", {k: v for k, v in body(prompt, False, salt).items() if v is not None})
    wall = round(time.time() - t0, 1)
    time.sleep(12)  # stats loggers publish per step; let the last step land before diffing
    c = r["choices"][0]
    return {"salt": salt, "ids": c["token_ids"], "prompt_tokens": r["usage"]["prompt_tokens"], "wall_s": wall,
            "text_head": c["text"][:120], "metric_delta": deltas(m0, metrics())}


def idle():
    for _ in range(600):
        if metrics().get("vllm:num_requests_running", 0) == 0:
            return
        time.sleep(1)
    sys.exit("server never went idle (other traffic); stopping")


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


def e0(label):
    res = {"label": label, "prompts": {}}
    for name, fn in PROMPTS.items():
        p = fn()
        a = run_once(p, f"{label}-{name}-a"); b = run_once(p, f"{label}-{name}-b"); h = run_once(p, f"{label}-{name}-a")
        fd, fh = first_divergence(a["ids"], b["ids"]), first_divergence(a["ids"], h["ids"])
        res["prompts"][name] = {"prompt_sha256": hashlib.sha256(p.encode()).hexdigest(), "runs": [a, b], "hit": h,
                                "identical": fd is None, "first_divergence": fd, "hit_first_divergence": fh}
        print(f"E0 {name:9s} prompt {a['prompt_tokens']} tok | cold==cold {fd is None} (div {fd}) | hit==cold {fh is None} "
              f"(div {fh}) hit delta {h['metric_delta']} | {a['text_head'][:50]!r}", flush=True)
    res["deterministic"] = all(v["identical"] for v in res["prompts"].values())
    res["hit_equals_cold"] = all(v["hit_first_divergence"] is None for v in res["prompts"].values())
    return res


def probe(ids, salt):
    idle(); m0 = metrics()
    r = post("/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0, "logprobs": 20,
                                 "return_tokens_as_token_ids": True, "cache_salt": salt})
    time.sleep(6)
    top = r["choices"][0]["logprobs"]["top_logprobs"][0]
    return {"salt": salt, "top": top, "argmax": max(top, key=top.get), "metric_delta": deltas(m0, metrics())}


def kl(p, q):
    floor = min(q.values())
    ps = {t: math.exp(v) for t, v in p.items()}; qs = {t: math.exp(q.get(t, floor)) for t in p}
    zp, zq = sum(ps.values()), sum(qs.values())
    return sum(ps[t] / zp * math.log((ps[t] / zp) / (qs[t] / zq)) for t in p)


def e2(label):
    res = {"label": label, "prompts": {}, "probes": []}
    for name, fn in PROMPTS.items():
        p = fn()
        ref = post("/v1/completions", {"model": MODEL, "prompt": p, "max_tokens": GEN, "temperature": 0,
                                       "ignore_eos": True, "return_token_ids": True, "cache_salt": f"{label}-{name}-ref"})
        c0 = ref["choices"][0]; pids, cont = c0["prompt_token_ids"], c0["token_ids"]
        res["prompts"][name] = {"prompt_sha256": hashlib.sha256(p.encode()).hexdigest(), "prompt_tokens": len(pids),
                                "continuation": cont}
        for k in (0, 64, 128, 192):
            ids, x = pids + cont[:k], f"{label}-{name}-{k}-x"
            A = probe(ids, x); B = probe(ids, f"{label}-{name}-{k}-y"); G = probe(ids, x)
            idle()
            if not reset(False, False):
                sys.exit("E2: GPU-only reset failed")
            C = probe(ids, x)
            row = {"prompt": name, "k": k, "A": A, "B": B, "G": G, "C": C, "d_AB": kl(A["top"], B["top"]),
                   "d_GC": kl(G["top"], C["top"]), "d_AC": kl(A["top"], C["top"]),
                   "flip_AB": A["argmax"] != B["argmax"], "flip_GC": G["argmax"] != C["argmax"],
                   "gpu_hit": G["metric_delta"].get("vllm:prefix_cache_hits", 0),
                   "cpu_hit": C["metric_delta"].get("vllm:external_prefix_cache_hits", 0),
                   "load_bytes": C["metric_delta"].get("vllm:kv_offload_load_bytes", 0)}
            res["probes"].append(row)
            print(f"E2 {name:9s} k={k:3d} d_AB={row['d_AB']:.2e} d_GC={row['d_GC']:.2e} d_AC={row['d_AC']:.2e} "
                  f"flipAB={row['flip_AB']} flipGC={row['flip_GC']} gpu_hit={row['gpu_hit']} cpu_hit={row['cpu_hit']} "
                  f"load={row['load_bytes']:.3g}", flush=True)
    P = res["probes"]; med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
    ab, gc = [r["d_AB"] for r in P], [r["d_GC"] for r in P]
    res["summary"] = {"median_d_AB": med(ab), "max_d_AB": max(ab), "median_d_GC": med(gc), "max_d_GC": max(gc),
                      "flips_AB": sum(r["flip_AB"] for r in P), "flips_GC": sum(r["flip_GC"] for r in P),
                      "all_loaded": all(r["load_bytes"] > 0 for r in P),
                      "equal_boundary": all(r["cpu_hit"] == r["gpu_hit"] for r in P)}
    S = res["summary"]
    res["pass"] = (S["all_loaded"] and S["equal_boundary"] and S["median_d_GC"] <= 2 * S["median_d_AB"]
                   and S["max_d_GC"] <= 2 * S["max_d_AB"] and S["flips_GC"] <= S["flips_AB"])
    print("E2 summary", S, "PASS" if res["pass"] else "FAIL", flush=True)
    return res


def e1(label, ref_path, external):
    ref = json.load(open(ref_path))["prompts"]["code"]
    p = code_prompt()
    assert hashlib.sha256(p.encode()).hexdigest() == ref["prompt_sha256"], "prompt drifted from the E0 reference"
    idle(); m0 = metrics()
    ids, reset_info = [], {}

    def do_reset():
        m = metrics()
        if m.get("vllm:num_requests_running", 0) > 1:
            reset_info["refused"] = "other requests running"; return
        reset_info["at_token"] = len(ids); reset_info["ok"] = reset(True, external)

    req = urllib.request.Request(HOST + "/v1/completions", json.dumps(body(p, True, f"{label}-code")).encode(),
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
    delta = deltas(m0, m1)
    fd = first_divergence(ref["runs"][0]["ids"], ids)
    res = {"label": label, "external_reset": external, "reset": reset_info, "wall_s": round(time.time() - t0, 1),
           "ids": ids, "metric_delta": delta, "identical_to_e0": fd is None, "first_divergence": fd,
           "evidence_ok": delta.get("vllm:num_preemptions", 0) >= 1
           and (external or delta.get("vllm:kv_offload_load_bytes", 0) > 0)}
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
        out = e0(label)
    elif mode == "e2":
        out = e2(label)
    elif mode == "e1":
        out = e1(label, sys.argv[3], "--external" in sys.argv)
    else:
        sys.exit(__doc__)
    json.dump(out, open(os.path.join(OUT, f"{label}.{mode}.json"), "w"))

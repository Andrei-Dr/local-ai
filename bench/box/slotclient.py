"""slotclient.py LABEL -- KV-slot save/restore + TTFT probe against a llama-server started with
--slot-save-path (C1). Env: URL base (default http://localhost:8099), OUT (default /ai/bench/runs),
GEN (64), PROMPT_FILE (w4_doc.txt next to this script), REPS (1 = copies of the doc concatenated into
the long prompt), MODE (cold|save|restore|warm|presave|extend, default cold), SLOT (slot filename, default
LABEL.slot — set ONE shared name so a save row and a restore row, which run under different LABELs,
address the same file), TIMEOUT (per-request seconds, default 3600 — deep prefill needs more),
PRE_GEN (presave generation before the save, default 1), DROP (extend: drop the LAST N sidecar token
ids before appending EXT, default 0), EXT (extend suffix text, default "\n\nNow list the three most
important open risks, one line each."). presave writes a token sidecar at OUT/<SLOT>.ids.json that
extend consumes — token-id prompts on /completion are the only restore path a GDN-hybrid model can
pass (recurrent state cannot roll back; text prompts re-render across the BPE junction).
Writes OUT/LABEL.slot.json {"mode","prompt_chars","rows"} and prints one line per request;
slot ops annotate the chat row. Any HTTP error: status on stderr, exit 1."""
import json, os, sys, time, urllib.error, urllib.request

label = sys.argv[1]
URL = os.environ.get("URL", "http://localhost:8099").rstrip("/")
OUT = os.environ.get("OUT", "/ai/bench/runs")
GEN = int(os.environ.get("GEN", "64"))
REPS = max(1, int(os.environ.get("REPS", "1")))
MODE = os.environ.get("MODE", "cold").lower()
TIMEOUT = int(os.environ.get("TIMEOUT", "3600"))
PRE_GEN = int(os.environ.get("PRE_GEN", "1"))
DROP = int(os.environ.get("DROP", "0"))
EXT = os.environ.get("EXT", "\n\nNow list the three most important open risks, one line each.")
PROMPT_FILE = os.environ.get("PROMPT_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "w4_doc.txt"))
QUESTION = "\n\nBased on the text above, summarize its current status and the next steps in one short paragraph."

doc = open(PROMPT_FILE, encoding="utf-8").read()
content = "\n".join([doc] * REPS) + QUESTION
SLOT = os.environ.get("SLOT", f"{label}.slot")
IDS_PATH = os.path.join(OUT, SLOT + ".ids.json")


def api(path, payload):
    req = urllib.request.Request(URL + path, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code} on {path}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"HTTP error on {path}: {e}", file=sys.stderr)
        sys.exit(1)


def chat_once():
    t0 = time.time()
    d = api("/v1/chat/completions", {"messages": [{"role": "user", "content": content}], "temperature": 0,
                                     "max_tokens": GEN, "chat_template_kwargs": {"enable_thinking": False}})
    t = d.get("timings", {})
    return {"prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
            "prefill_tps": t.get("prompt_per_second"), "decode_tps": t.get("predicted_per_second"),
            "wall_s": round(time.time() - t0, 2)}


def slot(action):
    d = api(f"/slots/0?action={action}", {"filename": SLOT})
    t = d.get("timings", {})
    if action == "save":
        return {"slot_bytes": d.get("n_written"), "slot_ms": t.get("save_ms")}
    return {"slot_bytes": d.get("n_read"), "slot_ms": t.get("restore_ms")}


def p_ids(content_text):
    """Token ids of the fully rendered chat prompt: apply-template THEN tokenize the rendered string
    (text prompts cannot guarantee token-junction stability across a save/restore boundary)."""
    t = api("/apply-template", {"messages": [{"role": "user", "content": content_text}],
                                "chat_template_kwargs": {"enable_thinking": False}})
    return api("/tokenize", {"content": t["prompt"], "add_special": False, "parse_special": True})["tokens"]


def trow(d, t0):
    t = d.get("timings", {})
    return {"prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
            "prefill_tps": t.get("prompt_per_second"), "decode_tps": t.get("predicted_per_second"),
            "wall_s": round(time.time() - t0, 2)}


def line(r):
    bs, ms = r.get("slot_bytes"), r.get("slot_ms")
    return (f"{label} {MODE.upper()} | prompt_n {r.get('prompt_n') or 0} | cache_n {r.get('cache_n') or 0}"
            f" | prefill {r.get('prefill_tps') or 0.0:9.1f} t/s | decode {r.get('decode_tps') or 0.0:5.2f} t/s"
            f" | wall {r.get('wall_s', 0):7.2f} s | slot {bs if bs is not None else '-'} bytes in"
            f" {ms if ms is not None else '-'} ms")


rows = []
if MODE == "cold":
    rows.append(chat_once())
elif MODE == "save":
    r = chat_once()
    r.update(slot("save"))
    rows.append(r)
elif MODE == "restore":
    s = slot("restore")  # FIRST: prove the restored prefix is what the following request runs on
    r = chat_once()
    r.update(s)
    rows.append(r)
elif MODE == "warm":
    rows.append(chat_once())
    rows.append(chat_once())
elif MODE == "presave":
    ids = p_ids(content)
    t0 = time.time()
    d = api("/completion", {"prompt": ids, "n_predict": PRE_GEN, "temperature": 0,
                            "cache_prompt": True, "return_tokens": True})
    r = trow(d, t0)
    r.update(slot("save"))
    full = ids + d["tokens"]
    json.dump({"ids": full}, open(IDS_PATH, "w"))
    r["base_n"] = len(full)
    rows.append(r)
elif MODE == "extend":
    try:
        sid = json.load(open(IDS_PATH))["ids"]
    except (OSError, ValueError, KeyError):
        print(f"extend: sidecar {IDS_PATH} missing/unreadable", file=sys.stderr)
        sys.exit(1)
    s = slot("restore")  # FIRST: the completion below must run on the restored state
    e = api("/tokenize", {"content": EXT, "add_special": False, "parse_special": False})["tokens"]
    t0 = time.time()
    d = api("/completion", {"prompt": sid[:len(sid) - DROP] + e, "n_predict": GEN, "temperature": 0,
                            "cache_prompt": True})
    r = trow(d, t0)
    r.update(s)
    r["base_n"], r["ext_n"], r["drop"] = len(sid), len(e), DROP
    r["reply"] = d["content"][:160]
    rows.append(r)
else:
    print(f"unknown MODE '{MODE}' (cold|save|restore|warm|presave|extend)", file=sys.stderr)
    sys.exit(1)
for r in rows:
    print(line(r), flush=True)
os.makedirs(OUT, exist_ok=True)
json.dump({"mode": MODE, "prompt_chars": len(content), "rows": rows}, open(os.path.join(OUT, f"{label}.slot.json"), "w"))

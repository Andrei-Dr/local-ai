"""oaiclient.py LABEL -- engine-agnostic timing over the OpenAI streaming API (vLLM, llama-server, anything).
Same prompts as specclient.py (its prompt block is executed verbatim, so the texts cannot drift), plus LONGPF=CHARS for the
long-prompt row (longpf.py's prompt). Timing is taken on the client for every engine alike:
  prefill t/s = prompt_tokens / time to the first streamed token;  decode t/s = (completion_tokens - 1) / (last - first token).
LONGPF_ONLY=1 runs just that row. Writes OUT/LABEL.oai.json."""
import hashlib, json, os, sys, time, urllib.request
label = sys.argv[1]
URL = os.environ.get("URL", "http://localhost:8099/v1/chat/completions")
OUT = os.environ.get("OUT", ".")
GEN = int(os.environ.get("GEN", "300"))
MODEL = os.environ.get("MODEL_NAME", "m")
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "specclient.py"), encoding="utf-8").read()
exec(src[src.index("PROMPTS = ["):src.index("def post(")])  # noqa: S102 -- PROMPTS / EDIT_SRC, byte-identical to specclient
if os.environ.get("LONGPF"):
    doc = open(os.environ["CORPUS"], encoding="utf-8").read()[:int(os.environ["LONGPF"])]
    PROMPTS.append(("longpf", "Summarize the following text in five bullet points.\n\n" + doc))
    if os.environ.get("LONGPF_ONLY") == "1":  # prefill sweeps: the long row alone
        PROMPTS[:] = PROMPTS[-1:]


def stream(content, max_tokens):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": content}], "temperature": 0,
                       "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; text = []; usage = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                piece = (c.get("delta") or {}).get("content") or ""
                if piece:
                    now = time.time(); first = first or now; last = now; text.append(piece)
    return t0, first, last, "".join(text), usage


rows = []
for kind, q in PROMPTS:
    t0, first, last, text, u = stream(q, 128 if kind == "longpf" else GEN)
    n, p = u.get("completion_tokens", 0), u.get("prompt_tokens", 0)
    row = {"prompt": kind, "prompt_tokens": p, "tokens": n, "ttft_s": round(first - t0, 3),
           "prefill_tps": round(p / (first - t0), 1), "decode_tps": round((n - 1) / (last - first), 2) if last > first else 0,
           "wall_tps": round(n / (last - t0), 2), "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text}
    rows.append(row)
    print(f"{label:16s} {kind:6s} {p:5d}+{n:3d} tok | prefill {row['prefill_tps']:7.1f} t/s | decode {row['decode_tps']:6.2f} t/s | wall {row['wall_tps']:6.2f} | {text[:40]!r}", flush=True)
os.makedirs(OUT, exist_ok=True)
json.dump({"rows": rows}, open(os.path.join(OUT, f"{label}.oai.json"), "w"))

"""turns.py LABEL -- multi-turn prefix reuse on the live server (port 8099). Three chat turns, thinking ON (the real use):
each turn resends the whole conversation the way chat clients do (previous assistant turns WITHOUT their reasoning), and the
server's timings say how many prompt tokens it actually had to process (prompt_n) vs reuse from the slot (cache_n).
On a hybrid (delta-net) model a prefix that diverges from the slot's tokens cannot be partially reused unless a context
checkpoint sits before the divergence: this measures what that costs per turn. Writes OUT/LABEL.turns.json."""
import json, os, sys, time, urllib.request

label = sys.argv[1]
URL = os.environ.get("URL", "http://localhost:8099/v1/chat/completions")
OUT = os.environ.get("OUT", "/ai/bench/runs")
USER = [
    "I have a Python service that parses CSV uploads and writes rows to Postgres. Uploads of ~200k rows take 40 s. "
    "List the three most likely bottlenecks and how you would confirm each one, briefly.",
    "Say it is row-by-row INSERTs. Show the fastest idiomatic fix with psycopg 3, in one short code block.",
    "Now make that fix idempotent on re-upload of the same file, keeping it fast. Short answer.",
]


def post(messages):
    body = json.dumps({"messages": messages, "temperature": 0, "max_tokens": 400,
                       "chat_template_kwargs": {"enable_thinking": True}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time(); d = json.load(urllib.request.urlopen(req, timeout=1800)); return d, time.time() - t0


msgs, rows = [], []
for i, u in enumerate(USER, 1):
    msgs.append({"role": "user", "content": u})
    d, wall = post(msgs)
    t = d.get("timings", {})
    content = d["choices"][0]["message"].get("content") or ""
    msgs.append({"role": "assistant", "content": content})       # clients resend content, never the reasoning
    total = (t.get("prompt_n") or 0) + (t.get("cache_n") or 0)
    row = {"turn": i, "prompt_total": total, "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
           "prompt_ms": t.get("prompt_ms"), "predicted_n": t.get("predicted_n"), "wall_s": round(wall, 2)}
    rows.append(row)
    print(f"{label} turn {i}: prompt {total} tok = reused {row['cache_n']} + processed {row['prompt_n']} "
          f"({(row['prompt_ms'] or 0) / 1000:.1f} s) | generated {row['predicted_n']} | wall {wall:.1f} s", flush=True)
os.makedirs(OUT, exist_ok=True)
json.dump({"rows": rows}, open(os.path.join(OUT, f"{label}.turns.json"), "w"))

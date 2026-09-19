"""specclient.py LABEL VRAM -- two fixed prompts against the local server; prints one row per prompt and
writes /ai/bench/runs/LABEL.client.json for ledger.py."""
import json, os, sys, time, urllib.request
label, vram = sys.argv[1], sys.argv[2]
GEN = int(os.environ.get("GEN", "200"))
PROMPTS = [
    ("code",   "Write a Python class implementing an LRU cache with get and put in O(1), with type hints and a short docstring for each method."),
    ("reason", "A train leaves at 3pm going 60 mph. A second leaves the same station at 4pm going 80 mph on the same track. When does the second catch the first? Show the algebra step by step."),
]
# EDIT=1 adds a copy-heavy prompt (the answer is mostly a verbatim copy of the input): the workload n-gram drafting targets.
EDIT_SRC = '''def parse_rows(path, sep=",", skip_header=True, max_rows=None):
    """Parse a delimited text file into a list of dicts keyed by the header row."""
    rows = []
    header = None
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            line = line.rstrip("\\n")
            if not line:
                continue
            fields = [field.strip() for field in line.split(sep)]
            if header is None:
                header = fields if skip_header else [f"col{i}" for i in range(len(fields))]
                if skip_header:
                    continue
            if len(fields) != len(header):
                raise ValueError(f"line {line_number}: expected {len(header)} fields, got {len(fields)}")
            rows.append(dict(zip(header, fields)))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def summarize_rows(path, column, sep=","):
    """Return count, minimum, maximum and mean of a numeric column parsed by parse_rows."""
    values = [float(row[column]) for row in parse_rows(path, sep=sep) if row.get(column)]
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {"count": len(values), "min": min(values), "max": max(values), "mean": sum(values) / len(values)}
'''
if os.environ.get("EDIT") == "1":
    PROMPTS.append(("edit", "Rename the function parse_rows to parse_records everywhere in the code below and change nothing else. "
                            "Output only the complete updated code in one code block.\n\n```python\n" + EDIT_SRC + "```"))
rows = []
for kind, q in PROMPTS:
    body = json.dumps({"messages": [{"role": "user", "content": q}], "temperature": 0, "max_tokens": GEN,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request("http://localhost:8099/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time(); d = json.load(urllib.request.urlopen(req, timeout=900)); wall = time.time() - t0
    n = d.get("usage", {}).get("completion_tokens", 0); t = d.get("timings", {})
    dn, da = t.get("draft_n"), t.get("draft_n_accepted")
    acc = f"{da}/{dn}={da/dn:.2f}" if dn else "-"
    text = d["choices"][0]["message"].get("content") or ""
    rows.append({"prompt": kind, "tokens": n, "wall_tps": round(n / wall, 2), "decode_tps": round(t.get("predicted_per_second", 0), 2),
                 "prefill_tps": round(t.get("prompt_per_second", 0), 1), "prompt_tokens": t.get("prompt_n"),
                 "draft_n": dn, "draft_accepted": da, "acceptance": round(da / dn, 3) if dn else None, "text_head": text[:80]})
    print(f"{label:14s} {kind:6s} {n:4d} tok | wall {n/wall:5.2f} t/s | decode {t.get('predicted_per_second', 0):5.2f} t/s | prefill {t.get('prompt_per_second', 0):6.1f} t/s | accept {acc:15s} | vram {vram} | {text[:48]!r}", flush=True)
os.makedirs("/ai/bench/runs", exist_ok=True)
json.dump({"vram": vram, "gen_tokens": GEN, "rows": rows}, open(f"/ai/bench/runs/{label}.client.json", "w"))

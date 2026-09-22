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

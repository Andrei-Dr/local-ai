#!/usr/bin/env python3
"""mt1.py LABEL CHARS -- a 3-turn conversation against the server on :8099: turn 1 = the first CHARS of corpus/prose_big.txt
+ a summary request; turns 2 and 3 append the assistant's reply and a short follow-up. Per turn: prompt tokens processed,
tokens reused from the slot cache (timings.cache_n), prefill / decode rate, time to first token (prompt_ms). Writes
runs/LABEL.mt.json. Exact reuse means turns 2-3 process only the new tokens."""
import json, sys, time, urllib.request
label, chars = sys.argv[1], int(sys.argv[2])
doc = open("/ai/bench/corpus/prose_big.txt", encoding="utf-8").read()[:chars]
follow = ["List three risks the text implies, one line each.", "Which of those risks is the most urgent, and why? Two sentences."]
msgs = [{"role": "user", "content": "Summarize the following text in five bullet points.\n\n" + doc}]
rows = []
for turn in range(3):
    body = json.dumps({"messages": msgs, "temperature": 0, "max_tokens": 128, "cache_prompt": True,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t0 = time.time()
    d = json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:8099/v1/chat/completions", body,
                                                                {"Content-Type": "application/json"}), timeout=3600))
    t = d.get("timings", {})
    r = {"turn": turn + 1, "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"), "prompt_ms": t.get("prompt_ms"),
         "prefill_tps": t.get("prompt_per_second"), "decode_tps": t.get("predicted_per_second"), "wall_s": round(time.time() - t0, 1)}
    rows.append(r)
    print(f"{label} turn {r['turn']}: processed {r['prompt_n']} tok, reused {r['cache_n']} | TTFT {(r['prompt_ms'] or 0) / 1000:.2f} s"
          f" | prefill {r['prefill_tps'] or 0:.1f} t/s | decode {r['decode_tps'] or 0:.2f} t/s | wall {r['wall_s']} s", flush=True)
    msgs.append({"role": "assistant", "content": d["choices"][0]["message"].get("content") or ""})
    if turn < 2:
        msgs.append({"role": "user", "content": follow[turn]})
json.dump({"label": label, "chars": chars, "rows": rows}, open(f"/ai/bench/runs/{label}.mt.json", "w"))

#!/usr/bin/env python3
"""dave7_agents.py — parallel agents against Dave's q38fn vLLM at 1M context: does one huge request starve the small ones?
usage: dave7_agents.py --port 8057 --model q38fn-mxfp4 --src DIR --out DIR [--big 500000] [--small 30000] [--n-small 3]

Observed in production (2026-09-24 20:43-20:48, config 1M v1: KV 8.3e9 / GPU, no KV offload): 2 long requests at KV 84-97%,
7 preemptions, decode ~2 t/s for 5 minutes. The fix under test: native CPU KV offload (--kv-offloading-size 64, native).

PRE-REGISTERED (2026-09-25, before the run):
  Load: one BIG request (~--big tokens of source text + a question, 256 output tokens) starts at t = 0; --n-small SMALL requests
  (~--small tokens each, distinct texts, 256 output tokens) start at t = 20 s, while the big one is still prefilling or decoding.
  R1 SMALL LATENCY: every small request's time to first token <= 90 s and total <= 180 s.
  R2 NO STARVATION: in /metrics sampled every 5 s, no window with >= 2 running requests has generation < 10 t/s averaged
     over 10 s, after the first small request has produced its first token.
  R3 (report): preemptions during the run (vllm:num_preemptions_total delta), KV usage max, BIG time to first token and total.
  PASS = R1 and R2. A request error or a server restart = FAIL with the error text.
"""
import argparse, json, os, threading, time, urllib.request


def post_stream(port, body, timeout=3600):
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(dict(body, stream=True, stream_options={'include_usage': True})).encode(),
                                 {'Content-Type': 'application/json'})
    t0 = time.time(); ttft = None; n = 0; pt = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if line.startswith(b'data: ') and line.strip() != b'data: [DONE]':
                d = json.loads(line[6:])
                pt = (d.get('usage') or {}).get('prompt_tokens', pt)
                if d.get('choices') and (d['choices'][0].get('delta') or {}).get('content'):
                    ttft = ttft or time.time() - t0; n += 1
    return ttft, time.time() - t0, n, pt


def metrics(port):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics', timeout=10) as r:
        out = {}
        for line in r.read().decode().splitlines():
            for k in ('num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc', 'num_preemptions_total',
                      'generation_tokens_total'):
                if line.startswith(f'vllm:{k}{{'):
                    out[k] = float(line.rsplit(' ', 1)[1])
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8057); ap.add_argument('--model', default='q38fn-mxfp4')
    ap.add_argument('--src'); ap.add_argument('--out'); ap.add_argument('--big', type=int, default=500_000)
    ap.add_argument('--small', type=int, default=30_000); ap.add_argument('--n-small', type=int, default=3)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    files = sorted(os.path.join(d, f) for d, _, fs in os.walk(a.src) for f in fs if f.endswith(('.c', '.cpp', '.h', '.py')))
    text = ''.join(open(f, errors='replace').read() for f in files)
    cpt = 1.26  # chars/token measured on this tokenizer + haystack (dave6); prompt_tokens are recorded, not assumed
    def body(chars, off):
        return {'model': a.model, 'max_tokens': 256, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False},
                'messages': [{'role': 'user', 'content': text[off:off + chars] + '\n\nSummarize what this code does in 5 bullets.'}]}
    res = {}
    def run(name, b, delay):
        time.sleep(delay)
        try:
            res[name] = dict(zip(('ttft', 'total', 'tokens', 'prompt_tokens'), post_stream(a.port, b)))
        except Exception as e:
            res[name] = {'error': str(e)[:300]}
    m0 = metrics(a.port); samples = []; stop = threading.Event()
    def poll():
        prev = None
        while not stop.is_set():
            m = metrics(a.port); m['t'] = time.time(); samples.append(m); time.sleep(5)
    threading.Thread(target=poll, daemon=True).start()
    ths = [threading.Thread(target=run, args=('big', body(int(a.big * cpt), 0), 0))]
    for i in range(a.n_small):
        ths.append(threading.Thread(target=run, args=(f'small{i}', body(int(a.small * cpt), (i + 1) * 3_000_000), 20)))
    t_start = time.time()
    for t in ths: t.start()
    for t in ths: t.join()
    stop.set(); time.sleep(1); m1 = metrics(a.port)
    first_small = min((t_start + 20 + r['ttft'] for k, r in res.items() if k.startswith('small') and r.get('ttft')), default=None)
    starve = []
    for x, y in zip(samples, samples[2:]):
        if first_small and x['t'] >= first_small and x.get('num_requests_running', 0) >= 2:
            rate = (y['generation_tokens_total'] - x['generation_tokens_total']) / (y['t'] - x['t'])
            if rate < 10: starve.append(round(rate, 1))
    smalls = [r for k, r in res.items() if k.startswith('small')]
    r1 = all(r.get('ttft') is not None and r['ttft'] <= 90 and r['total'] <= 180 for r in smalls)
    r2 = not starve
    out = {'results': res, 'preemptions': m1.get('num_preemptions_total', 0) - m0.get('num_preemptions_total', 0),
           'kv_max': max((s.get('kv_cache_usage_perc', 0) for s in samples), default=None), 'starved_windows': starve,
           'R1': r1, 'R2': r2, 'PASS': r1 and r2}
    json.dump({'summary': out, 'samples': samples}, open(os.path.join(a.out, 'dave7.json'), 'w'), indent=1)
    for k, r in sorted(res.items()):
        print(f'  {k}: {json.dumps(r)}')
    print(f"  preemptions {out['preemptions']:.0f} | KV max {out['kv_max']} | starved 10-s windows {starve}")
    print(f"DAVE7 R1 small latency {'PASS' if r1 else 'FAIL'} | R2 no starvation {'PASS' if r2 else 'FAIL'} | "
          f"{'PASS' if out['PASS'] else 'FAIL'}")


if __name__ == '__main__':
    main()

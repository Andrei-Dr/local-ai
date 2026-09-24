#!/usr/bin/env python3
"""dave6_needle.py — needle-in-a-haystack at ~200k / ~500k / ~950k tokens against a local vLLM (dave6.sh R1, R2).
usage: dave6_needle.py --port 8099 --src DIR --out DIR
The haystack is source text (.c/.cpp/.h/.cu/.py under --src, sorted, concatenated); sizes are measured with the server's
/tokenize. A 6-digit code is planted at depth 10 / 50 / 90% and asked for at the end, greedy, 32 tokens, thinking off.
"""
import argparse, json, os, random, time, urllib.request


def post(port, path, body, timeout=7200):
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


MODEL = 'm'


def ntok(port, text):
    return len(post(port, '/tokenize', {'model': MODEL, 'prompt': text})['tokens'])


def chat(port, content, max_tokens=32):
    t0 = time.time()
    r = post(port, '/v1/chat/completions', {'model': MODEL, 'messages': [{'role': 'user', 'content': content}],
             'max_tokens': max_tokens, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}})
    return r['choices'][0]['message'].get('content') or '', r['usage'], time.time() - t0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--port', type=int, default=8099); ap.add_argument('--src'); ap.add_argument('--out'); ap.add_argument('--model', default='m')
    a = ap.parse_args()
    global MODEL
    MODEL = a.model
    files = sorted(os.path.join(d, f) for d, _, fs in os.walk(a.src) for f in fs if f.endswith(('.c', '.cpp', '.h', '.cu', '.cuh', '.py')))
    text = ''.join(open(f, errors='replace').read() for f in files)
    chars_per_tok = len(text[:400000]) / ntok(a.port, text[:400000])
    print(f'haystack: {len(files)} files, {len(text) / 1e6:.1f} MB, {chars_per_tok:.2f} chars/token', flush=True)
    rows = []
    print('R2 short-context sanity:', flush=True)
    for q in ['What is 17 * 23? Answer with the number only.', 'What is the capital of Australia? One word.',
              'Spell the word "necessary" backwards.']:
        ans, u, dt = chat(a.port, q)
        print(f'  {q!r} -> {ans.strip()!r}', flush=True)
        rows.append({'kind': 'short', 'q': q, 'a': ans})
    rng = random.Random(6)
    for target in (200_000, 500_000, 950_000):
        need = int(target * chars_per_tok)
        if need > len(text):
            text = text * (need // len(text) + 1)
        hay = text[:need]
        for depth in (0.1, 0.5, 0.9):
            code = f'{rng.randrange(10**5, 10**6)}'
            cut = hay.rfind('\n', 0, int(len(hay) * depth)) + 1
            prompt = (hay[:cut] + f'\n/* NOTE: the secret access code is {code}. */\n' + hay[cut:] +
                      '\n\nSomewhere in the text above there is a NOTE with a secret access code. Reply with the code only.')
            try:
                ans, u, dt = chat(a.port, prompt)
                ok = code in ans
                r = {'kind': 'needle', 'target': target, 'depth': depth, 'prompt_tokens': u['prompt_tokens'], 'code': code,
                     'answer': ans.strip()[:80], 'pass': ok, 'wall_s': round(dt, 1), 'prefill_tps': round(u['prompt_tokens'] / dt, 1)}
            except Exception as e:
                r = {'kind': 'needle', 'target': target, 'depth': depth, 'error': str(e)[:300], 'pass': False}
            print('  ' + json.dumps(r), flush=True)
            rows.append(r)
    json.dump(rows, open(os.path.join(a.out, 'dave6.json'), 'w'), indent=1)
    cells = [r for r in rows if r['kind'] == 'needle']
    works = all(r['pass'] for r in cells if r['target'] <= 500_000)
    print(f'R1 VERDICT: {"YARN WORKS (200k + 500k all pass)" if works else "FAIL at <= 500k"} | 950k: '
          f'{sum(r["pass"] for r in cells if r["target"] == 950_000)}/3', flush=True)


if __name__ == '__main__':
    main()

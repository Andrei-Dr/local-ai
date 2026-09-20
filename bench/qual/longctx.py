#!/usr/bin/env python3
"""longctx.py LABEL --depths 4096,16384,... -- retrieval quality AT DEPTH (is q4_0 KV at 131k-262k still
a working model?). RULER-style synthetic NIAH multi-key + variable tracking (Hsieh 2024, arXiv 2404.06654);
generated benign filler + random numbers only, no dataset download.

One prefill at depth costs 25-60 min on the box, so ONE document per depth carries MANY questions: every
request is a single user message `DOCUMENT + "\\n\\n" + QUESTION` whose DOCUMENT bytes are identical across
questions of a depth, sent with cache_prompt=true so only the question is new. Never vary anything before
the question. Target document tokens = depth*0.90 (room for template + question + answer); a first request
that answers 400/500 (context overflow) rebuilds the document 5% smaller, at most 3 retries, then marks the
depth overflow and continues with the next depth.

usage: longctx.py LABEL [--depths 4096,16384,32768,65536,131072] [--url http://localhost:8099] [--seed 1]
       [--needles 10] [--chains 4] [--haystack FILE] [--out results] [--slot-save] [--slot-restore] [--think]
env TIMEOUT (seconds, default 21600). Resumable: OUT/LABEL.longctx.jsonl — finished (depth,kind,index) are
skipped; plan(seed,depth) is a pure function, so a restart rebuilds byte-identical documents. Summary:
OUT/LABEL.longctx.summary.json (per-depth needle pct ± SE, vt mean ± SE, cache_n ratio of questions 2..n
with PREFIX NOT REUSED below 0.9, and needle pct by position bucket pooled over depths).
"""
import argparse, itertools, json, math, os, random, re, sys, time, urllib.error, urllib.request
from pathlib import Path

HERE = Path(__file__).parent
CYCLE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
SEP = " "
STEP = len(CYCLE) + len(SEP)
KEYS = ["apple", "basket", "candle", "dart", "engine", "finger", "garden", "hammer", "iron", "jacket",
        "kettle", "lantern", "marble", "nickel", "orange", "pencil", "quartz", "rocket", "saddle", "temple",
        "umbrella", "violin", "wallet", "yttrium", "zigzag", "anchor", "beacon", "cobalt", "dragon", "ember",
        "falcon", "golden", "harbor", "ignite", "junior", "kernel", "liquid", "meteor", "nectar", "onyx",
        "palace", "quiver", "ribbon", "signal", "tiger", "umber", "velvet", "willow", "xenon", "yellow",
        "zenith", "bronze", "cedar", "delta", "eagle", "fossil", "granite", "helium", "indigo", "jasper",
        "koala", "lotus", "magma", "nova"]                                   # 64 plain nouns, distinct
MAX_SHRINKS = 3


def api(url, path, payload, timeout):
    req = urllib.request.Request(url + path, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def plan(seed, depth, needles, chains):
    """Pure: rng = Random(seed*1000003 + depth) => deterministic per (seed, depth); different depths draw
    different numbers. Draws happen in a FIXED order so any restart replays identical bytes."""
    rng = random.Random(seed * 1000003 + depth)
    keys = KEYS[:needles]
    nums = [rng.randint(1_000_000, 9_999_999) for _ in range(needles)]
    names = ["".join(t) for t in itertools.product("ABCDEFGH", repeat=3)]
    cs = []
    for c in range(chains):
        cs.append({"value": rng.randint(10000, 99999), "vars": names[c * 4:c * 4 + 4]})
    positions = [0.5] if needles == 1 else [i / (needles - 1) for i in range(needles)]
    return {"keys": keys, "nums": nums, "chains": cs, "positions": positions}


def inserts_of(p):
    """(boundary_fraction, tie_order, text) — needles at plan positions, chain hops spread j/3 in order."""
    ins = [(p["positions"][i], i, f"The special magic number for {p['keys'][i]} is {p['nums'][i]}.")
           for i in range(len(p["keys"]))]
    for ci, ch in enumerate(p["chains"]):
        v = ch["vars"]
        hops = [f"VAR {v[0]} = {ch['value']}.", f"VAR {v[1]} = VAR {v[0]}.",
                f"VAR {v[2]} = VAR {v[1]}.", f"VAR {v[3]} = VAR {v[2]}."]
        ins += [(j / 3, 1_000_000 + ci * 10 + j, hop) for j, hop in enumerate(hops)]
    return ins


def filler(tokens_chars, hay):
    if hay:
        r = max(1, math.ceil(tokens_chars / max(1, len(hay))))
        return (hay.rstrip() + SEP) * r
    return (CYCLE + SEP) * max(1, math.ceil(tokens_chars / STEP))


def place(base, inserts):
    """Splice sentences in at evenly spaced cycle boundaries; last boundaries first so offsets survive."""
    bounds = max(1, len(base) // STEP)
    by_b = {}
    for frac, order, text in sorted(inserts, key=lambda x: (x[0], x[1], x[2])):
        b = int(round(frac * (bounds - 1))) if bounds > 1 else 0
        by_b.setdefault(min(bounds - 1, max(0, b)), []).append((order, text))
    chunks, tail = [], len(base)
    for b in sorted(by_b, reverse=True):
        start = b * STEP
        chunks.append(base[start:tail])
        chunks.append(SEP.join(t for _, t in sorted(by_b[b])) + SEP)
        tail = start
    chunks.append(base[:tail])
    return "".join(reversed(chunks))


def document(p, cpt, shrink, hay=None):
    return place(filler(int(p["tokens"] * (0.95 ** shrink) * cpt), hay), inserts_of(p))


def build_questions(p):
    qs = [{"kind": "needle", "index": i, "key": p["keys"][i], "position": p["positions"][i],
           "expect": str(p["nums"][i]),
           "q": f"What is the special magic number for {p['keys'][i]}? Answer with the number only."}
          for i in range(len(p["keys"]))]
    all_vars = [v for c in p["chains"] for v in c["vars"]]
    for ci, ch in enumerate(p["chains"]):
        qs.append({"kind": "vt", "index": ci, "key": ch["vars"][0], "position": None,
                   "expect": list(ch["vars"]), "foreign": [v for v in all_vars if v not in ch["vars"]],
                   "q": f"List every variable that holds the value {ch['value']}."})
    return qs


def score_needle(reply, num):
    return 1.0 if re.search(rf"(?<!\d){re.escape(num)}(?!\d)", reply) else 0.0


def score_vt(reply, expect, foreign):
    if any(re.search(rf"\b{v}\b", reply) for v in foreign):
        return 0.0                                            # naming another chain's variable zeroes it
    return sum(bool(re.search(rf"\b{v}\b", reply)) for v in expect) / len(expect)


def ask(url, doc, question, max_tokens, think, timeout):
    body = {"messages": [{"role": "user", "content": doc + "\n\n" + question}], "temperature": 0,
            "max_tokens": max_tokens, "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": think}}
    t0 = time.time()
    d = api(url, "/v1/chat/completions", body, timeout)
    t = d.get("timings") or {}
    u = d.get("usage") or {}
    txt = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    return txt, {"prompt_n": t.get("prompt_n", u.get("prompt_tokens")), "cache_n": t.get("cache_n"),
                 "prefill_tps": t.get("prompt_per_second"), "decode_tps": t.get("predicted_per_second"),
                 "wall_s": round(time.time() - t0, 3)}


def calibrate(url, hay, timeout):
    probe = hay[:4000]
    d = api(url, "/tokenize", {"content": probe}, timeout)
    toks = d.get("toks") or d.get("tokens") or []
    n = len(toks) if isinstance(toks, list) else int(toks)
    return len(probe) / n if n else 4.0


class _Overflow(Exception):
    """first request of a depth kept failing 400/500 after MAX_SHRINKS shrinks: skip the depth."""


class _LazyJsonl:
    """creates the JSONL on first write (an all-overflow run must not leave an empty file)."""

    def __init__(self, path):
        self.path, self.fh = path, None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        if self.fh:
            self.fh.close()

    def write(self, s):
        if self.fh is None:
            self.fh = self.path.open("a", encoding="utf-8")
        self.fh.write(s)

    def flush(self):
        if self.fh:
            self.fh.flush()


def run(label, depths, url, seed, needles, chains, hay, out_dir, slot_save, slot_restore, think, timeout, reuse_guard=16384):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"{label}.longctx.jsonl"
    done = set()
    if jp.exists():
        for line in jp.open():
            try:
                r = json.loads(line)
                done.add((r["depth"], r["kind"], r["index"]))
            except (json.JSONDecodeError, KeyError):
                pass
    probe = hay if hay else CYCLE * 600
    cpt = calibrate(url, probe, timeout)
    max_tokens = 48 * (8 if think else 1)
    with _LazyJsonl(jp) as out:
        for depth in depths:
            p = plan(seed, depth, needles, chains)
            p["tokens"] = depth * 0.90
            qs = build_questions(p)
            restored = False
            saved = False
            first_try = True
            asked = 0
            try:
                for q in qs:
                    if (depth, q["kind"], q["index"]) in done:
                        continue
                    if slot_restore and not restored:
                        restored = True
                        try:
                            api(url, "/slots/0?action=restore", {"filename": f"{label}_d{depth}.slot"}, timeout)
                        except urllib.error.HTTPError as e:
                            print(f"  {depth}: slot restore ignored (HTTP {e.code}) — cold prefill", flush=True)
                    shrink = 0
                    while True:
                        doc = document(p, cpt, shrink, hay)
                        try:
                            txt, tm = ask(url, doc, q["q"], max_tokens, think, timeout)
                            break
                        except urllib.error.HTTPError as e:
                            if first_try and e.code in (400, 500):
                                if shrink < MAX_SHRINKS:
                                    shrink += 1
                                    continue
                                raise _Overflow(e.code) from e
                            raise
                        except OSError as e:
                            if first_try:
                                if shrink < MAX_SHRINKS:
                                    shrink += 1
                                    continue
                                raise _Overflow(repr(e)) from e
                            raise
                    first_try = False
                    asked += 1
                    score = (score_needle(txt, q["expect"]) if q["kind"] == "needle"
                             else score_vt(txt, q["expect"], q["foreign"]))
                    rec = {"depth": depth, "kind": q["kind"], "index": q["index"], "key": q["key"],
                           "position": q["position"], "score": round(float(score), 4),
                           "prompt_n": tm["prompt_n"], "cache_n": tm["cache_n"], "prefill_tps": tm["prefill_tps"],
                           "decode_tps": tm["decode_tps"], "wall_s": tm["wall_s"], "reply_head": txt[:120]}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    done.add((depth, q["kind"], q["index"]))
                    if slot_save and not saved:
                        saved = True
                        try:
                            api(url, "/slots/0?action=save", {"filename": f"{label}_d{depth}.slot"}, timeout)
                        except (urllib.error.HTTPError, OSError) as e:
                            print(f"  {depth}: slot save ignored ({e})", flush=True)
                    print(f"  {depth} {q['kind']}{q['index']} {'ok' if score >= 0.5 else 'MISS'} "
                          f"pn {tm['prompt_n']} cn {tm['cache_n']}", flush=True)
                    # one prefill at 131k is ~40 min: if the 2nd request of a deep depth did not re-use the document
                    # prefix (hybrid model without a usable context checkpoint), stop instead of re-prefilling per question
                    cn, pn = tm["cache_n"] or 0, tm["prompt_n"] or 0
                    if reuse_guard and depth >= reuse_guard and asked >= 2 and cn < 0.5 * (cn + pn):
                        print(f"  PREFIX NOT REUSED (cached {cn} of {cn + pn}): skipping the rest of depth {depth}", flush=True)
                        break
            except _Overflow as e:
                print(f"  OVERFLOW depth {depth} — still failing ({e}) after {MAX_SHRINKS} shrinks; depth skipped",
                      flush=True)
                continue
    return summarize(label, out_dir)


def summarize(label, out_dir):
    jp = Path(out_dir) / f"{label}.longctx.jsonl"
    rows = [json.loads(l) for l in jp.open()] if jp.exists() else []
    summary = {"label": label, "depths": {}}
    bucket = {}
    for depth in sorted({r["depth"] for r in rows}):
        rs = [r for r in rows if r["depth"] == depth]
        nd = [r for r in rs if r["kind"] == "needle"]
        vt = [r for r in rs if r["kind"] == "vt"]
        dd = {"n": len(rs), "needle_n": len(nd)}
        if nd:
            p = sum(1 if r["score"] >= 0.5 else 0 for r in nd) / len(nd)
            dd["needle_pct"] = round(100 * p, 1)
            dd["needle_se_pct"] = round(100 * math.sqrt(p * (1 - p) / len(nd)), 1)
        if vt:
            mu = sum(r["score"] for r in vt) / len(vt)
            dd["vt_mean"] = round(mu, 4)
            dd["vt_se"] = round(math.sqrt(sum((r["score"] - mu) ** 2 for r in vt) / (len(vt) * (len(vt) - 1))), 4) if len(vt) > 1 else None
        ratios = [r["cache_n"] / r["prompt_n"] for r in rs[1:] if r.get("prompt_n")]
        if ratios:
            rt = sum(ratios) / len(ratios)
            dd["cache_n_mean_ratio"] = round(rt, 3)
            if rt < 0.9:
                dd["prefix_reused"] = False
                print(f"PREFIX NOT REUSED at depth {depth} (questions 2..n cached {rt:.2f})", flush=True)
        dd["first_wall_s"] = rs[0]["wall_s"]
        summary["depths"][str(depth)] = dd
        for r in nd:
            if r.get("position") is not None:
                bi = min(4, int(r["position"] / 0.2))
                g = bucket.setdefault(f"{bi * 0.2:.1f}-{(bi + 1) * 0.2:.1f}", [0.0, 0])
                g[0] += 1 if r["score"] >= 0.5 else 0
                g[1] += 1
    summary["needle_pct_by_position"] = {k: {"pct": round(100 * v[0] / v[1], 1), "n": v[1]}
                                         for k, v in sorted(bucket.items())}
    Path(out_dir, f"{label}.longctx.summary.json").write_text(json.dumps(summary, indent=1))
    for d, dd in summary["depths"].items():
        np_ = dd.get("needle_pct")
        print(f"longctx {label} {d:>8} needle {'%5.1f%% ±%.1f' % (np_, dd['needle_se_pct']) if np_ is not None else '  -'}"
              f" vt {dd.get('vt_mean')} n={dd['n']}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(prog="longctx.py")
    ap.add_argument("label")
    ap.add_argument("--depths", default="4096,16384,32768,65536,131072")
    ap.add_argument("--url", default=os.environ.get("URL", "http://localhost:8099"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--needles", type=int, default=10)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--haystack", default=None)
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--slot-save", action="store_true")
    ap.add_argument("--slot-restore", action="store_true")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--reuse-guard", type=int, default=16384,
                    help="from this depth on, stop a depth when its 2nd request did not re-use the prefix (0 = off)")
    a = ap.parse_args()
    run(a.label, [int(x) for x in a.depths.split(",")], a.url.rstrip("/"), a.seed, a.needles, a.chains,
        open(a.haystack, encoding="utf-8").read() if a.haystack else None, a.out,
        a.slot_save, a.slot_restore, a.think, int(os.environ.get("TIMEOUT", "21600")), a.reuse_guard)


if __name__ == "__main__":
    main()

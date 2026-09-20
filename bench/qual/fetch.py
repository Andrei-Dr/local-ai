#!/usr/bin/env python3
"""Build the fixed quality-eval item sets (deterministic selection, no extra deps).

usage: fetch.py [--gsm8k N] [--mmlu-rows R] [--overrefusal N] [OUT_DIR]   -> OUT_DIR/{gsm8k,humaneval,mmlu_pro[,overrefusal]}.jsonl
       fetch.py --hard [--math N] OUT_DIR                                   -> OUT_DIR/{aime,math_l5,humaneval_plus}.jsonl

Source: Hugging Face datasets-server rows API. Selection is by fixed offsets so every model
is scored on the identical items, and shorter runs (--limit) are nested prefixes of longer ones.
    gsm8k      openai/gsm8k main/test, first N (default 50)
    humaneval  openai/openai_humaneval test, every 4th task (41)
    mmlu_pro   TIGER-Lab/MMLU-Pro test, 14 pages x R rows (R default 5 => 70 items) across all categories
    overrefusal N per source (0 = skip): bench-llm/or-bench @ or-bench-hard-1k train (NEVER the toxic
               config) plus Paul/XSTest train SAFE rows only (contrast/unsafe rows are dropped by label)
HARD sets (--hard; run qual.py with --think): aime = HuggingFaceH4/aime_2024 + yentinglin/aime_2025 (30 + 30, integer golds);
    math_l5 = HuggingFaceH4/MATH-500 level 5 rows whose gold is a plain number (qual.parse_number), dataset order, first N
    (default 40, prefix-safe); humaneval_plus = evalplus/humanevalplus, the SAME every-4th tasks as humaneval with EvalPlus tests.
Nested-prefix rule (qual.py resumes by item id, so enlarged sets must keep the old items FIRST):
gsm8k stays the first N of the test split; per MMLU-Pro page today's base rows (offsets 0,20,40,60,80
of the 100-row page) come first in the file, the extra rows (uniform grid 100//R) of every page follow
after the base block. R must be a multiple of 5 so the base offsets stay on the grid (R=20 => 280).
overrefusal builds ONE fixed global order per source: groups cycle alphabetically in rounds, and inside
each group rows are consumed in van der Corput (bit-reversed index) order, so any N is a prefix of the
same order and stays evenly spread across the split and balanced across categories/types.
"""
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://datasets-server.huggingface.co/rows"
MMLU_PAGES = 14   # page-spaced sampling: 1 request per page, the API rate-limits hard
PAGE_LEN = 100    # rows fetched per page
BASE_STEP = PAGE_LEN // 5  # today's per-page base rows: 0, 20, 40, 60, 80
OR_DS = ("bench-llm/or-bench", "or-bench-hard-1k", "train")     # hard rule: never or-bench-toxic
XS_DS = ("Paul/XSTest", "default", "train")                     # label=="safe" rows only


def rows(dataset, config, split, offset, length):
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{API}?{q}", timeout=60) as r:
                d = json.load(r)
            return [x["row"] for x in d["rows"]], d["num_rows_total"]
        except Exception as e:  # rate limit / transient
            if attempt == 4:
                raise
            print(f"  retry {dataset}@{offset}: {e}", file=sys.stderr)
            time.sleep(15 * (attempt + 1))


def dump(path, items):
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{path}: {len(items)} items")


def _get(rows_fn, ds, cfg, split, off, n, pause=None):
    """Collect up to n rows from a split; the rows API caps a request at PAGE_LEN, so paginate."""
    out, total = [], 0
    while len(out) < n:
        step = min(PAGE_LEN, n - len(out))
        if pause:
            pause()
        chunk, total = rows_fn(ds, cfg, split, off + len(out), step)
        out += chunk
        if len(chunk) < step:
            break  # split exhausted
    return out, total


def select_datasets(rows_fn, gsm_n=50, mmlu_rows=5, pause=None):
    """Pure selection (injectable rows_fn so tests run offline). rows_fn returns ([row], num_rows_total)
    like rows(); pause is called between MMLU-Pro page requests in the real fetch."""
    sel = {}
    g, _ = _get(rows_fn, "openai/gsm8k", "main", "test", 0, gsm_n)
    sel["gsm8k"] = [
        {"id": f"gsm8k/{i}", "question": r["question"], "gold": r["answer"].split("####")[-1].strip().replace(",", "")}
        for i, r in enumerate(g)
    ]

    h = []
    for off in (0, 100):
        h += _get(rows_fn, "openai/openai_humaneval", "openai_humaneval", "test", off, 100)[0]
    sel["humaneval"] = [
        {"id": r["task_id"], "prompt": r["prompt"], "test": r["test"], "entry_point": r["entry_point"]}
        for r in h[::4]
    ]

    _, total = rows_fn("TIGER-Lab/MMLU-Pro", "default", "test", 0, 1)
    stride = total // MMLU_PAGES
    base_pos = list(range(0, PAGE_LEN, BASE_STEP))
    grid = PAGE_LEN // mmlu_rows
    base, extra = [], []
    for k in range(MMLU_PAGES):
        if pause:
            pause()
        page = rows_fn("TIGER-Lab/MMLU-Pro", "default", "test", k * stride + stride // 2, PAGE_LEN)[0]
        for p in base_pos:
            if p < len(page):
                r = page[p]
                base.append({"id": f"mmlu_pro/{r['question_id']}", "category": r["category"], "question": r["question"],
                             "options": r["options"], "gold": r["answer"]})
        for p in (q * grid for q in range(mmlu_rows)):
            if p not in base_pos and p < len(page):
                r = page[p]
                extra.append({"id": f"mmlu_pro/{r['question_id']}", "category": r["category"], "question": r["question"],
                              "options": r["options"], "gold": r["answer"]})
    # interleave categories so a --limit prefix is still spread across subjects (base block only,
    # verbatim today's behavior; the extra block stays page-ordered and always comes after the 70)
    seen, keyed = {}, []
    for it in base:
        seen[it["category"]] = seen.get(it["category"], 0) + 1
        keyed.append((seen[it["category"]], it["category"], it))
    sel["mmlu_pro"] = [it for _, _, it in sorted(keyed, key=lambda x: x[:2])] + extra
    return sel


AIME_DS = (("HuggingFaceH4/aime_2024", "default", "train", "2024"), ("yentinglin/aime_2025", "default", "train", "2025"))


def select_hard(rows_fn, math_n=40, pause=None):
    """Hard reasoning/code sets, published items verbatim. rows_fn like rows()."""
    from qual import parse_number  # same parser scores the replies, so every kept gold is scoreable
    sel = {"aime": []}
    for ds, cfg, split, year in AIME_DS:
        for r in _get(rows_fn, ds, cfg, split, 0, 30, pause=pause)[0]:
            sel["aime"].append({"id": f"aime/{year}-{r['id']}", "question": r["problem"], "gold": str(int(str(r["answer"]).strip()))})
    _, tot = rows_fn("HuggingFaceH4/MATH-500", "default", "test", 0, 1)
    m = [r for r in _get(rows_fn, "HuggingFaceH4/MATH-500", "default", "test", 0, tot, pause=pause)[0]
         if str(r["level"]) == "5" and parse_number(r["answer"]) is not None]
    sel["math_l5"] = [{"id": f"math_l5/{r['unique_id']}", "question": r["problem"], "gold": r["answer"]} for r in m[:math_n]]
    _, tot = rows_fn("evalplus/humanevalplus", "default", "test", 0, 1)
    h = []
    while len(h) < tot:   # EvalPlus test cells are large (up to MBs): small pages
        if pause:
            pause()
        chunk = rows_fn("evalplus/humanevalplus", "default", "test", len(h), 10)[0]
        if not chunk:
            break
        h += chunk
    sel["humaneval_plus"] = [{"id": r["task_id"].replace("HumanEval/", "HumanEvalPlus/"), "prompt": r["prompt"], "test": r["test"],
                              "entry_point": r["entry_point"]} for r in h[::4]]
    return sel


def _vdc(i):
    """van der Corput radical inverse base 2: 0, .5, .25, .75, ... — evenly spread, prefix-stable."""
    rev, x = 0, i
    while x:
        rev = (rev << 1) | (x & 1)
        x >>= 1
    return rev / (1 << max(i.bit_length(), 1))


def stratified_pick(by_group, n):
    """Fixed global order: rounds over groups (alphabetical); inside a group, rows are consumed in
    van der Corput index order. Any prefix of the result is balanced across groups (within 1) and
    evenly spread inside each group; N=50 is a prefix of N=100 automatically."""
    queues = [[g[i] for i in sorted(range(len(g)), key=lambda i: (_vdc(i), i))] for _, g in sorted(by_group.items())]
    out, r = [], 0
    while True:
        took = False
        for q in queues:
            if r < len(q):
                out.append(q[r])
                took = True
        if not took:
            break
        r += 1
    return out[:n]


def select_overrefusal(rows_fn, n=100, pause=None):
    """Published items only, verbatim: OR-Bench-Hard-1K (benign-looking prompts measuring over-refusal;
    the toxic config is never touched) and XSTest rows whose label marks them SAFE (contrast/unsafe
    dropped). rows_fn like rows(); returns {"orbench": [...], "xstest": [...]}, <= n per source."""
    sel = {}
    _, tot = rows_fn(*OR_DS, 0, 1)
    by_cat = {}
    for i, r in enumerate(_get(rows_fn, *OR_DS, 0, tot, pause=pause)[0]):
        c = r.get("category", "?")
        by_cat.setdefault(c, []).append({"id": f"orbench/{i}", "source": "orbench", "category": c, "prompt": r["prompt"]})
    sel["orbench"] = stratified_pick(by_cat, n)
    _, tot = rows_fn(*XS_DS, 0, 1)
    by_type = {}
    for r in _get(rows_fn, *XS_DS, 0, tot, pause=pause)[0]:
        if str(r.get("label", "")).lower() != "safe":
            continue  # hard rule: XSTest unsafe/contrast items never enter the set
        it = {"id": f"xstest/{r['id']}", "source": "xstest", "category": r.get("type", "?"), "prompt": r["prompt"]}
        by_type.setdefault(it["category"], []).append(it)
    sel["xstest"] = stratified_pick(by_type, n)
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", help="target dir (default: ./data next to this script)")
    ap.add_argument("--gsm8k", type=int, default=50, help="gsm8k items, first N of the test split (prefix-safe)")
    ap.add_argument("--mmlu-rows", type=int, default=5, help="rows sampled per MMLU-Pro page; multiple of 5 (20 => 280 items)")
    ap.add_argument("--overrefusal", type=int, default=0, help="over-refusal items PER SOURCE (or-bench-hard-1k + XSTest safe); 0 skips")
    ap.add_argument("--hard", action="store_true", help="build ONLY the hard sets (aime, math_l5, humaneval_plus) into OUT_DIR")
    ap.add_argument("--math", type=int, default=40, help="math_l5 items (prefix-safe)")
    a = ap.parse_args()
    if a.hard:
        out = Path(a.out_dir) if a.out_dir else Path(__file__).parent / "data_hard"
        out.mkdir(parents=True, exist_ok=True)
        sel = select_hard(rows, math_n=a.math, pause=lambda: time.sleep(1))
        for name in ("aime", "math_l5", "humaneval_plus"):
            for it in sel[name]:
                if any(isinstance(v, str) and v.endswith("...") and len(v) > 5000 for v in it.values()):
                    sys.exit(f"{it['id']}: a cell looks truncated by the rows API")
            dump(out / f"{name}.jsonl", sel[name])
        return
    out = Path(a.out_dir) if a.out_dir else Path(__file__).parent / "data"
    out.mkdir(parents=True, exist_ok=True)

    sel = select_datasets(rows, gsm_n=a.gsm8k, mmlu_rows=a.mmlu_rows, pause=lambda: time.sleep(2))
    for name in ("gsm8k", "humaneval", "mmlu_pro"):
        dump(out / f"{name}.jsonl", sel[name])
    if a.overrefusal:
        sel = select_overrefusal(rows, n=a.overrefusal, pause=lambda: time.sleep(2))
        dump(out / "overrefusal.jsonl", sel["orbench"] + sel["xstest"])


if __name__ == "__main__":
    main()

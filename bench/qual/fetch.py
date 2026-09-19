#!/usr/bin/env python3
"""Build the fixed quality-eval item sets (deterministic selection, no extra deps).

usage: fetch.py [OUT_DIR]   -> OUT_DIR/{gsm8k,humaneval,mmlu_pro}.jsonl

Source: Hugging Face datasets-server rows API. Selection is by fixed offsets so every model
is scored on the identical items, and shorter runs (--limit) are nested prefixes of longer ones.
    gsm8k      openai/gsm8k main/test, first 50
    humaneval  openai/openai_humaneval test, every 4th task (41)
    mmlu_pro   TIGER-Lab/MMLU-Pro test, 70 items (14 evenly spaced pages x 5 rows) across all categories
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://datasets-server.huggingface.co/rows"


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


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "data")
    out.mkdir(parents=True, exist_ok=True)

    g, _ = rows("openai/gsm8k", "main", "test", 0, 50)
    dump(out / "gsm8k.jsonl", [
        {"id": f"gsm8k/{i}", "question": r["question"], "gold": r["answer"].split("####")[-1].strip().replace(",", "")}
        for i, r in enumerate(g)
    ])

    h = []
    for off in (0, 100):
        h += rows("openai/openai_humaneval", "openai_humaneval", "test", off, 100)[0]
    dump(out / "humaneval.jsonl", [
        {"id": r["task_id"], "prompt": r["prompt"], "test": r["test"], "entry_point": r["entry_point"]}
        for r in h[::4]
    ])

    _, total = rows("TIGER-Lab/MMLU-Pro", "default", "test", 0, 1)
    stride = total // 14  # 14 pages of 100 rows, 5 rows per page: few requests (the API rate-limits hard)
    m = []
    for k in range(14):
        time.sleep(2)
        page = rows("TIGER-Lab/MMLU-Pro", "default", "test", k * stride + stride // 2, 100)[0]
        for r in page[::20]:
            m.append({"id": f"mmlu_pro/{r['question_id']}", "category": r["category"], "question": r["question"],
                      "options": r["options"], "gold": r["answer"]})
    # interleave categories so a --limit prefix is still spread across subjects
    seen, keyed = {}, []
    for it in m:
        seen[it["category"]] = seen.get(it["category"], 0) + 1
        keyed.append((seen[it["category"]], it["category"], it))
    dump(out / "mmlu_pro.jsonl", [it for _, _, it in sorted(keyed, key=lambda x: x[:2])])


if __name__ == "__main__":
    main()

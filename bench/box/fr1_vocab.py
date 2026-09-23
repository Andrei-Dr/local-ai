#!/usr/bin/env python3
"""fr1_vocab.py TOKENIZE_BIN MODEL OUT_PREFIX -- build FR-Spec draft vocabularies: token frequencies over prose (corpus/prose_big.txt),
wiki (kld/wiki.test.raw) and code (llama.cpp sources + bench scripts), top-K for K in 16k/32k/64k, plus every token of the chat
template strings the server emits (parsed with special tokens). Writes OUT_PREFIX_<K>.bin (int32 LE ids) and prints coverage."""
import collections, glob, os, re, subprocess, sys, tempfile
tok_bin, model, out = sys.argv[1], sys.argv[2], sys.argv[3]

def tokenize(text, special=False):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text); path = f.name
    args = [tok_bin, "-m", model, "-f", path, "--ids", "--log-disable"] + ([] if special else ["--no-parse-special"])
    r = subprocess.run(args, capture_output=True, text=True, timeout=1800)
    os.unlink(path)
    m = re.search(r"\[([\d,\s]*)\]\s*$", r.stdout.strip())
    if not m:
        sys.exit(f"tokenize failed: {r.stderr[-400:]}")
    return [int(x) for x in m.group(1).split(",") if x.strip()]

texts = {"prose": open("/ai/bench/corpus/prose_big.txt", encoding="utf-8", errors="ignore").read()[:6_000_000],
         "wiki": open("/ai/bench/kld/wiki.test.raw", encoding="utf-8", errors="ignore").read()}
code = []
for pat in ("/ai/src/llama.cpp-v2/src/*.cpp", "/ai/src/llama.cpp-v2/common/*.cpp", "/ai/src/llama.cpp-v2/tools/server/*.cpp",
            "/ai/bench/*.py", "/ai/bench/*.sh"):
    for p in sorted(glob.glob(pat)):
        code.append(open(p, encoding="utf-8", errors="ignore").read())
texts["code"] = "\n".join(code)[:6_000_000]
cnt = collections.Counter()
for name, t in texts.items():
    ids = tokenize(t); cnt.update(ids)
    print(f"  {name}: {len(t)/1e6:.1f} MB -> {len(ids)} tokens", flush=True)
special = set(tokenize("<|im_start|>system\nx<|im_end|>\n<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", special=True))
total = sum(cnt.values())
ranked = [t for t, _ in cnt.most_common()]
for K in (16384, 32768, 65536):
    top = ranked[:K]
    cov = sum(cnt[t] for t in top) / total
    ids = sorted(set(top) | special)
    with open(f"{out}_{K}.bin", "wb") as f:
        for t in ids:
            f.write(int(t).to_bytes(4, "little", signed=True))
    print(f"  K={K}: {len(ids)} ids (incl. {len(special - set(top))} extra specials), corpus token coverage {100 * cov:.2f}%")
print(f"  distinct tokens seen: {len(cnt)} of the vocab")

import sys, re, collections
sys.path.insert(0, "/opt/ai/src/llama.cpp/gguf-py")
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
print("=== metadata ===")
for k, f in r.fields.items():
    if any(s in k for s in ("block_count","context_length","embedding_length","feed_forward","head_count","full_attention","nextn","ssm.","key_length","value_length","vocab_size","general.architecture","general.name")):
        try: v = f.contents()
        except Exception: v = "?"
        if isinstance(v, list) and len(v) > 8: v = f"[{len(v)} items] {v[:8]}..."
        print(f"  {k} = {v}")
cat = collections.Counter(); types = collections.defaultdict(collections.Counter); per_layer = collections.Counter(); kinds = collections.defaultdict(set)
for t in r.tensors:
    n = t.name; b = int(t.n_bytes)
    m = re.match(r"blk\.(\d+)\.(.+?)\.(weight|bias)$", n)
    if m:
        per_layer[int(m.group(1))] += b
        part = m.group(2)
        c = "attn" if part.startswith("attn") else "ssm/gdn" if part.startswith("ssm") else "ffn" if part.startswith("ffn") else "nextn/mtp" if "nextn" in part else "other:" + part
    else:
        c = n
    cat[c] += b; types[c][t.tensor_type.name] += 1
tot = sum(cat.values())
print(f"\n=== bytes by category (total {tot/2**30:.2f} GiB, {len(r.tensors)} tensors) ===")
for c, b in cat.most_common(14):
    print(f"  {b/2**20:9.1f} MiB  {100*b/tot:5.1f}%  {c:28s} {dict(types[c])}")
ls = sorted(per_layer.items())
print(f"\n=== per-layer: n={len(ls)} min={min(per_layer.values())/2**20:.1f} max={max(per_layer.values())/2**20:.1f} MiB ===")
print("  first 8:", [f"{b/2**20:.0f}" for _, b in ls[:8]])
full = [i for i in per_layer if any(t.name == f"blk.{i}.attn_q.weight" for t in r.tensors)]
print(f"  full-attention layers ({len(full)}):", full)

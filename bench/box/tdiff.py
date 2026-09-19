"""Tensor-level + metadata diff of two GGUF files. usage: tdiff.py STOCK EDITED"""
import sys, re, collections
sys.path.insert(0, "/opt/ai/src/llama.cpp/gguf-py")
import numpy as np
from gguf import GGUFReader
a, b = GGUFReader(sys.argv[1]), GGUFReader(sys.argv[2])
print("=== metadata ===")
ka, kb = set(a.fields), set(b.fields)
print("  only in stock :", sorted(ka - kb)); print("  only in edited:", sorted(kb - ka))
changed = [k for k in sorted(ka & kb) if not k.startswith("GGUF.") and
           any(not np.array_equal(x, y) for x, y in zip(a.fields[k].parts, b.fields[k].parts)) or
           (k in ka & kb and len(a.fields[k].parts) != len(b.fields[k].parts))]
print("  changed values:", changed)
print("  chat_template identical:", "tokenizer.chat_template" not in changed and "tokenizer.chat_template" in ka & kb)
ta, tb = {t.name: t for t in a.tensors}, {t.name: t for t in b.tensors}
print("\n=== tensors ===")
print(f"  stock={len(ta)} edited={len(tb)}")
extra = sorted(set(tb) - set(ta)); print(f"  only in edited ({len(extra)}):", extra[:12], "..." if len(extra) > 12 else "")
print("  only in stock:", sorted(set(ta) - set(tb)))
same = diff = 0; kinds = collections.Counter(); blocks = set(); meta = []; frac = []
for n in sorted(set(ta) & set(tb)):
    x, y = ta[n], tb[n]
    if x.tensor_type != y.tensor_type or list(x.shape) != list(y.shape): meta.append(n); continue
    if np.array_equal(x.data, y.data): same += 1; continue
    diff += 1
    m = re.match(r"blk\.(\d+)\.(.+)\.weight$", n)
    kinds[m.group(2) if m else n] += 1
    if m: blocks.add(int(m.group(1)))
    xb, yb = np.asarray(x.data).view(np.uint8).ravel(), np.asarray(y.data).view(np.uint8).ravel()
    frac.append(float((xb != yb).mean()))
print(f"  byte-identical: {same}   differ: {diff}   type/shape mismatch: {len(meta)} {meta[:5]}")
print("  differing tensor kinds:", dict(kinds))
if blocks: print(f"  differing blocks: {min(blocks)}..{max(blocks)} ({len(blocks)} blocks)")
if frac: print(f"  fraction of bytes changed within differing tensors: min={min(frac):.4f} max={max(frac):.4f}")

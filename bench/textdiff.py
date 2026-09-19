"""textdiff.py A.client.json B.client.json -- per prompt kind, compare completion texts from two specclient dumps.
Prints one line per kind in both files: IDENTICAL or DIVERGES at char N with 20-char context each side.
Falls back to text_head (marked) when a file predates the text field. Exit 0 if all identical, else 1."""
import json, sys
CTX = 20
def rows_by_kind(path):
    return {r["prompt"]: r for r in json.load(open(path, encoding="utf-8"))["rows"]}
a, b = (rows_by_kind(p) for p in sys.argv[1:3])
ok = True
for kind in sorted(set(a) & set(b)):
    ra, rb = a[kind], b[kind]
    ta, tb = ra.get("text"), rb.get("text")
    if ta is None or tb is None:
        ta, tb = ra.get("text_head") or "", rb.get("text_head") or ""
        head_only = True
    else:
        head_only = False
    if ta == tb:
        print(f"{kind}  IDENTICAL{' (head only)' if head_only else ''}")
        continue
    n = next(i for i in range(min(len(ta), len(tb)) + 1) if i == min(len(ta), len(tb)) or ta[i] != tb[i])
    ca, cb = ta[n:n + CTX], tb[n:n + CTX]
    print(f"{kind}  DIVERGES at char {n}: {ca!r} vs {cb!r}{' (head only)' if head_only else ''}")
    ok = False
sys.exit(0 if ok else 1)

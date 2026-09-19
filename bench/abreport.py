"""abreport.py LEDGER.jsonl BASE_REGEX TEST_REGEX [--md] -- A/B decode table from specbench ledger rows.
Repeats are averaged across records per prompt kind (completed-only, latest-by-ts per label); a verdict
only counts past the within-arm repeat spread (or 1%), so noise cannot be reported as a win."""
import json, re, sys


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def spread(xs):
    """100*(max-min)/mean over one arm's repeats; 0.0 for a single run; None without numbers."""
    xs = [x for x in xs if isinstance(x, (int, float))]
    if not xs:
        return None
    m = mean(xs)
    return 100.0 * (max(xs) - min(xs)) / m if m else 0.0


def arm_stats(records):
    per, acc, hits, vrams, commits = {}, {}, [], [], set()
    for r in records:
        for p in r.get("prompts") or []:
            per.setdefault(p.get("prompt"), []).append(p.get("decode_tps"))
            acc.setdefault(p.get("prompt"), []).append(p.get("acceptance"))
        mc = r.get("moe_cache") or {}
        if isinstance(mc.get("hit_rate_pct"), (int, float)):
            hits.append(mc["hit_rate_pct"])
        if isinstance(r.get("vram_mib"), (int, float)):
            vrams.append(r["vram_mib"])
        c = (r.get("git") or {}).get("commit")
        if c:
            commits.add(c[:9])
    return {"per": per, "acc": acc, "hit": mean(hits), "vram": mean(vrams), "commits": commits, "records": len(records)}


def verdict(delta, nz):
    thr = max(nz if nz is not None else 0.0, 1.0)
    return "WIN" if delta > thr else "LOSS" if delta < -thr else "flat"


def main():
    md = "--md" in sys.argv[1:]
    pos = [a for a in sys.argv[1:] if a != "--md"]
    if len(pos) != 3:
        print("usage: abreport.py LEDGER.jsonl BASE_REGEX TEST_REGEX [--md]", file=sys.stderr)
        sys.exit(2)
    led, rex = pos[0], (re.compile(pos[1]), re.compile(pos[2]))
    latest = {}
    for line in open(led, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("kind") == "specbench" and r.get("completed") and r.get("prompts"):
            if r["label"] not in latest or (r.get("ts") or "") >= (latest[r["label"]].get("ts") or ""):
                latest[r["label"]] = r
    arms = {}
    for name, rx in zip(("base", "test"), rex):
        recs = [r for r in latest.values() if rx.search(r["label"])]
        if not recs:
            print(f"no completed specbench records match arm '{name}' ({rx.pattern})", file=sys.stderr)
            sys.exit(2)
        arms[name] = arm_stats(recs)

    kinds = list(dict.fromkeys(k for st in arms.values() for k in st["per"]))

    def cell(k):  # per kind, per arm: (mean, n) and spread
        out = []
        for name in ("base", "test"):
            v = arms[name]["per"].get(k) or []
            v = [x for x in v if isinstance(x, (int, float))]
            out.append(((mean(v), len(v)), spread(v)))
        return out

    rows = []
    for k in kinds:
        ((bm, bn), sb), ((tm, tn), st_) = cell(k)
        if bm is None or tm is None:
            continue  # prompt kind lacks decode numbers on an arm
        nz = None if bn <= 1 and tn <= 1 else max(v for v in (sb, st_) if v is not None)
        d = 100.0 * (tm - bm) / bm if bm else None
        rows.append([k, f"{bm:.2f} (n={bn})", f"{tm:.2f} (n={tn})",
                     f"{d:+.2f}%" if d is not None else "n/a",
                     "-" if nz is None else f"{nz:.2f}%", verdict(d, nz) if d is not None else "flat"])

    # ALL row: mean over prompts of the per-prompt means; its noise = mean of the numeric per-prompt noises
    av = {}
    for i, name in enumerate(("base", "test")):
        av[name] = mean([v for v in (cell(k)[i][0][0] for k in kinds) if v is not None])
    bz = [max(v for v in (cell(k)[0][1], cell(k)[1][1]) if v is not None)
          for k in kinds if not (cell(k)[0][0][1] <= 1 and cell(k)[1][0][1] <= 1)]
    nz = mean(bz) if bz else None
    if av["base"] and av["test"]:
        d = 100.0 * (av["test"] - av["base"]) / av["base"]
        rows.append(["ALL", f"{av['base']:.2f}", f"{av['test']:.2f}", f"{d:+.2f}%",
                     "-" if nz is None else f"{nz:.2f}%", verdict(d, nz)])

    head = ["prompt", "base decode t/s", "test decode t/s", "delta %", "noise %", "verdict"]
    if md:
        out = ["| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
        out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    else:
        w = [max(len(s) for s in c) for c in zip(head, *rows)] if rows else [max(len(s) for s in c) for c in [head]]
        ln = lambda ss: " | ".join(s.ljust(x) for s, x in zip(ss, w)).rstrip()
        out = [ln(head), "-|-".join("-" * x for x in w)] + [ln(r) for r in rows]
    print("\n".join(out))
    for name in ("base", "test"):
        st = arms[name]
        acc = ", ".join((f"{k} {m:.3f}" if (m := mean(v)) is not None else f"{k} -") for k, v in st["acc"].items() if v) or "-"
        hit = f"{st['hit']:.1f}%" if st["hit"] is not None else "-"
        vram = f"{st['vram']:.0f} MiB" if st["vram"] is not None else "-"
        print(f"{name}: {st['records']} records | acceptance {acc} | cache hit {hit} | vram {vram} | "
              f"commits {' '.join(sorted(st['commits'])) or '-'}")
    sys.exit(0)


if __name__ == "__main__":
    main()

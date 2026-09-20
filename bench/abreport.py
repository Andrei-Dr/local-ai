"""abreport.py LEDGER.jsonl BASE_REGEX TEST_REGEX [--md] [--metric NAME] -- A/B table from specbench rows.
Repeats are averaged across records per prompt kind (completed-only, latest-by-ts per label); a verdict
only counts past the within-arm repeat spread (or 1%), so noise cannot be reported as a win.
Importable core: compare(records, base_re, test_re, metric) -> dict; render(result, md) -> str."""
import json, re, sys

METRICS = ("decode_tps", "wall_tps", "prefill_tps", "acceptance")


def mlabel(m):
    return "decode t/s" if m == "decode_tps" else m


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


def latest_completed(records):
    """One record per label: completed specbench rows with prompts, newest ts (>= keeps file order on ties)."""
    latest = {}
    for r in records:
        if r.get("kind") == "specbench" and r.get("completed") and r.get("prompts"):
            if r["label"] not in latest or (r.get("ts") or "") >= (latest[r["label"]].get("ts") or ""):
                latest[r["label"]] = r
    return latest


def arm_stats(records, metric="decode_tps"):
    per, acc, hits, vrams, commits = {}, {}, [], [], set()
    for r in records:
        for p in r.get("prompts") or []:
            rv = p.get(metric + "_runs")  # M1: multi-run clients embed the raw samples; they are the sample (not the mean)
            if rv is not None:
                per.setdefault(p.get("prompt"), []).extend(rv)
            else:
                per.setdefault(p.get("prompt"), []).append(p.get(metric))
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


def compare(records, base_re, test_re, metric="decode_tps"):
    """-> {"empty_arm": None|"base"|"test", "pattern", "metric", "rows": [[6 cells]], "arms": {...}}"""
    latest = latest_completed(records)
    arms = {}
    for name, rx in (("base", base_re), ("test", test_re)):
        recs = [r for r in latest.values() if rx.search(r["label"])]
        if not recs:
            return {"empty_arm": name, "pattern": rx.pattern, "metric": metric}
        arms[name] = arm_stats(recs, metric)

    kinds = list(dict.fromkeys(k for st in arms.values() for k in st["per"]))

    def cell(k):  # per kind, per arm: ((mean, n), spread)
        out = []
        for name in ("base", "test"):
            v = [x for x in arms[name]["per"].get(k) or [] if isinstance(x, (int, float))]
            out.append(((mean(v), len(v)), spread(v)))
        return out

    rows = []
    for k in kinds:
        ((bm, bn), sb), ((tm, tn), st_) = cell(k)
        if bm is None or tm is None:
            continue  # prompt kind lacks numeric values on an arm
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
    return {"empty_arm": None, "pattern": None, "metric": metric, "rows": rows, "arms": arms}


def render(result, md=False):
    head = ["prompt", f"base {mlabel(result['metric'])}", f"test {mlabel(result['metric'])}", "delta %", "noise %", "verdict"]
    rows = result["rows"]
    if md:
        out = ["| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
        out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    else:
        w = [max(len(s) for s in c) for c in zip(head, *rows)] if rows else [max(len(s) for s in c) for c in [head]]
        ln = lambda ss: " | ".join(s.ljust(x) for s, x in zip(ss, w)).rstrip()
        out = [ln(head), "-|-".join("-" * x for x in w)] + [ln(r) for r in rows]
    for name in ("base", "test"):
        st = result["arms"][name]
        acc = ", ".join((f"{k} {m:.3f}" if (m := mean(v)) is not None else f"{k} -") for k, v in st["acc"].items() if v) or "-"
        hit = f"{st['hit']:.1f}%" if st["hit"] is not None else "-"
        vram = f"{st['vram']:.0f} MiB" if st["vram"] is not None else "-"
        out.append(f"{name}: {st['records']} records | acceptance {acc} | cache hit {hit} | vram {vram} | "
                   f"commits {' '.join(sorted(st['commits'])) or '-'}")
    return "\n".join(out)


def main():
    argv = sys.argv[1:]
    md = "--md" in argv
    metric = "decode_tps"
    pos = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--metric":
            i += 1
            metric = argv[i] if i < len(argv) else ""
        elif a.startswith("--metric="):
            metric = a.split("=", 1)[1]
        elif a != "--md":
            pos.append(a)
        i += 1
    if metric not in METRICS:
        print(f"bad --metric '{metric}' (allowed: {', '.join(METRICS)})", file=sys.stderr)
        sys.exit(2)
    if len(pos) != 3:
        print("usage: abreport.py LEDGER.jsonl BASE_REGEX TEST_REGEX [--md] [--metric NAME]", file=sys.stderr)
        sys.exit(2)
    recs = [json.loads(l) for l in open(pos[0], encoding="utf-8") if l.strip()]
    res = compare(recs, re.compile(pos[1]), re.compile(pos[2]), metric)
    if res["empty_arm"]:
        print(f"no completed specbench records match arm '{res['empty_arm']}' ({res['pattern']})", file=sys.stderr)
        sys.exit(2)
    print(render(res, md))
    sys.exit(0)


if __name__ == "__main__":
    main()

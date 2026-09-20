#!/usr/bin/env python3
"""qreport.py — verdict tables for the three accuracy jobs (kld1 / hq1 / lq1). None of them has run yet;
formats come from bench/box/kld1.sh, hq1.sh (qual.py schemas) and lq1.sh (longctx.py schemas).

  qreport.py kld --log FILE [--md]         blocks '##### kld1_<cand> | hh:mm:ss | <bytes> bytes' + indented
                                           llama-perplexity stat lines; table sorted by mean KLD, ratio vs
                                           the IQ2_M row, BEST-or-disagree verdict; SKIPPED / out-of-memory
                                           candidates listed as notes, never crash.
  qreport.py hard --results DIR --labels A,B[,C] [--base A] [--md]
                                           per set pct ± se, n, truncated, mean tokens, plus a
                                           finished-only pct (chains cut at the cap are the cap's failure);
                                           then the paired comparison vs base — functions IMPORTED from
                                           bench/qual/paired.py, not reimplemented.
  qreport.py lq --results DIR [--prefix lq1_] [--md]
                                           grid rows = depth, cols = arm: needle pct (n) | vt | decode t/s
                                           | NOREUSE; needle pct by position bucket; per-arm needle deltas
                                           vs the f16 arm paired by (depth, kind, index).

Parsers are tolerant: unknown lines ignored, missing field None, empty input prints 'no ...' and exit 0.
Stdlib only."""
import argparse, json, math, os, re, sys
from pathlib import Path

SETS_ORDER = ("aime", "math_l5", "humaneval_plus")
KLD_HDR_RE = re.compile(r"^##### (kld1_\S+)(?: SKIPPED: (.*?))?(?: \| ([^|]*) \| ([^|]*))?\s*$")
STAT_RE = re.compile(r"^\s+([^:\s][^:]*?)\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
                     r"(?:\s*[±]\s*(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?))?")
BYTES_RE = re.compile(r"(\d+)\s+bytes")


def parse_kld(text):
    blocks, cur = [], None
    for ln in (text or "").splitlines():
        m = KLD_HDR_RE.match(ln)
        if m:
            cur = {"label": m.group(1), "metrics": {}, "skipped": (m.group(2) or None), "bytes": None,
                   "candidate": False}
            bm = BYTES_RE.search(m.group(4) or "")
            if bm and not cur["skipped"]:
                cur["bytes"] = int(bm.group(1))
                cur["candidate"] = True
            blocks.append(cur)
            continue
        if cur is None:
            continue
        if "out of memory" in ln.lower():
            cur["oom"] = True
        sm = STAT_RE.match(ln)
        if sm:
            name = " ".join(sm.group(1).split())
            cur["metrics"][name] = float(sm.group(2))
            if sm.group(3) is not None:
                cur["metrics"][name + " err"] = float(sm.group(3))
    return [b for b in blocks]


def report_kld(text, md=False):
    blocks = parse_kld(text)
    cands = [b for b in blocks if b.get("candidate") and "Mean KLD" in b["metrics"]]
    lines = ["# kld1 — KL divergence vs the Q6 reference", ""]
    if not cands:
        return "\n".join(lines + ["no kld1 candidates with KLD metrics in this log.", ""]) + "\n"
    cands.sort(key=lambda b: b["metrics"]["Mean KLD"])
    dead = [b for b in blocks if b.get("candidate") and "Mean KLD" not in b["metrics"]]
    skipped = [b for b in blocks if b.get("skipped")]
    g = lambda b, k: b["metrics"].get(k)
    ref = next((b for b in cands if b["label"].upper().endswith("IQ2_M")), None)
    if md:
        lines += ["| candidate | GiB | mean KLD ± err | median | 99% | 99.9% | max | same top p % | PPL ratio | vs IQ2_M |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
    for b in cands:
        MK = b["metrics"]
        e = MK.get("Mean KLD err")
        ratio = ""
        if ref and b is not ref and ref["metrics"]["Mean KLD"]:
            rt = MK["Mean KLD"] / ref["metrics"]["Mean KLD"]
            ratio = f"x{rt:.2f}" + (" lower divergence from the Q6 reference" if rt < 1 else
                                    " higher divergence from the Q6 reference" if rt > 1 else " parity")
        row = [b["label"].removeprefix("kld1_"),
               "-" if b["bytes"] is None else f"{b['bytes'] / 2**30:.1f}",
               f"{MK['Mean KLD']:.5f}" + (f" ± {e:.5f}" if e is not None else ""),
               _n(g(b, "Median KLD"), "{:.3f}"), _n(g(b, "99.0% KLD"), "{:.3f}"), _n(g(b, "99.9% KLD"), "{:.3f}"),
               _n(g(b, "Maximum KLD"), "{:.3f}"), _n(g(b, "Same top p"), "{:.1f}"),
               _n(g(b, "Mean PPL(Q)/PPL(base)"), "{:.3f}"), ratio]
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    best = cands[0]["label"]
    tails = [b for b in cands if b["metrics"].get("99.0% KLD") is not None]
    best_tail = min(tails, key=lambda b: b["metrics"]["99.0% KLD"]) if tails else None
    lines.append("")
    if best_tail and best_tail["label"] == best:
        lines.append(f"VERDICT: {best.removeprefix('kld1_')} BEST (lowest mean AND lowest 99% KLD)")
    else:
        lines.append(f"VERDICT: metrics disagree — mean KLD picks {best.removeprefix('kld1_')}, "
                     f"99% KLD picks {(best_tail['label'] if best_tail else '?').removeprefix('kld1_')}")
    for b in dead:
        lines.append(f"note: {b['label'].removeprefix('kld1_')} — no metrics"
                     + (" (out of memory)" if b.get("oom") else ""))
    for b in skipped:
        lines.append(f"note: {b['label'].removeprefix('kld1_')} SKIPPED: {b['skipped']}")
    return "\n".join(lines) + "\n"


def _n(v, pat="{:.2f}", dash="-"):
    return pat.format(v) if v is not None else dash


def _paired_mod():
    p = str(Path(__file__).resolve().parent / "qual")
    if p not in sys.path:
        sys.path.insert(0, p)
    import paired
    return paired


def _load_rows(dir_, label):
    try:
        return (json.loads(Path(dir_, f"{label}.summary.json").read_text()),
                Path(dir_, f"{label}.jsonl").read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None, None


def report_hard(dir_, labels, base=None, md=False):
    base = base or labels[0]
    lines = ["# hq1 — hard sets, thinking on", ""]
    data = {}
    for lab in labels:
        s, t = _load_rows(dir_, lab)
        if s is None or t is None:
            lines.append(f"{lab}: MISSING (no summary.json/jsonl in {dir_})")
            continue
        data[lab] = (s, t)
        try:
            rows = [json.loads(l) for l in t.splitlines() if l.strip()]
        except ValueError:
            rows = []
        for sname in list(SETS_ORDER) + sorted(set(s.get("sets", {})) - set(SETS_ORDER)):
            c = s.get("sets", {}).get(sname)
            if not c:
                continue
            fin = [r for r in rows if r.get("set") == sname and r.get("finish") != "length"]
            fo = _n(100 * sum(bool(r.get("ok")) for r in fin) / len(fin) if fin else None, "{:.1f}")
            lines.append(f"{lab} {sname.upper():16} {c.get('pct', '-')!s:>6} ± {c.get('se_pct', '-')}  "
                         f"n {c.get('n', '-')}  trunc {c.get('truncated', '-')}  "
                         f"mean tok {s.get('mean_tokens', '-')}  finished-only {fo}")
    ps = lines[:]
    pa = _paired_mod()
    bs, bt = data.get(base, (None, None))
    bmap = pa.loads_jsonl(bt)[0] if bt is not None else None
    cmp_lines = []
    if bmap is None:
        cmp_lines.append(f"base {base} missing — no paired comparisons possible.")
    else:
        for lab in labels:
            if lab == base:
                continue
            t = data.get(lab, (None, None))[1]
            if t is None:
                cmp_lines.append(f"{lab}: rows missing — paired comparison skipped")
                continue
            tmap = pa.loads_jsonl(t)[0]
            res = pa.compare(bmap, tmap)
            cmp_lines.append(f"{lab} vs {base} ({res['ids']} shared ids; only_base {res['only_base']}, "
                             f"only_test {res['only_test']})")
            for sname in sorted(res["sets"]):
                cmp_lines.append("    " + pa.fmt_plain(sname, res["sets"][sname]))
            cmp_lines.append("    " + pa.fmt_plain("ALL", res["ALL"]))
    return "\n".join(ps + ["", "## Paired vs base", ""] + cmp_lines + [""]) + "\n"


def report_lq(dir_, prefix="lq1_", md=False):
    arms = {}
    for f in sorted(Path(dir_).glob(prefix + "*.longctx.summary.json")):
        lab = f.name.removesuffix(".longctx.summary.json")
        try:
            summary = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        rows = []
        jp = f.with_name(lab + ".longctx.jsonl")
        if jp.exists():
            for ln in jp.read_text(errors="replace").splitlines():
                try:
                    rows.append(json.loads(ln))
                except ValueError:
                    pass
        arms[lab] = (summary, rows)
    lines = ["# lq1 — retrieval quality at depth", ""]
    if not arms:
        return "\n".join(lines + [f"no {prefix}* summaries in {dir_}", ""]) + "\n"
    depths = sorted({int(d) for s, _ in arms.values() for d in s.get("depths", {})})
    lines.append("| depth | " + " | ".join(sorted(arms)) + " |")
    lines.append("|---|" + "---|" * len(arms))
    for d in depths:
        cells = []
        for lab in sorted(arms):
            s, rows = arms[lab]
            dd = s.get("depths", {}).get(str(d))
            if dd is None:
                cells.append("–")
                continue
            tpss = [r.get("decode_tps") for r in rows if r.get("depth") == d and r.get("decode_tps")]
            cell = "{} ({}) vt {}".format(_n(dd.get("needle_pct"), "{:.1f}"), dd.get("needle_n", 0),
                                          _n(dd.get("vt_mean"), "{:.2f}"))
            if tpss:
                cell += f" {sum(tpss) / len(tpss):.1f} t/s"
            if dd.get("prefix_reused") is False:
                cell += " NOREUSE"
            cells.append(cell)
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    lines += ["", "## Needle pct by position (pooled over depths)", ""]
    for lab in sorted(arms):
        buckets = arms[lab][0].get("needle_pct_by_position") or {}
        if buckets:
            lines.append(f"{lab}: " + "  ".join(f"{k}: {v.get('pct', '-')}% (n{v.get('n', 0)})"
                                                for k, v in sorted(buckets.items())))
    lines += ["", "## Delta vs f16 (needle questions paired by (depth, index))", ""]
    f16 = next((l for l in arms if "f16" in l), None)
    if f16 is None:
        lines.append("no f16 arm present — deltas skipped")
    else:
        fok = {(r["depth"], r["kind"], r["index"]): r.get("score", 0) >= 0.5
               for r in arms[f16][1] if r.get("kind") == "needle"}
        for lab in sorted(set(arms) - {f16}):
            lost = gained = shared_n = 0
            for r in arms[lab][1]:
                if r.get("kind") != "needle":
                    continue
                k = (r["depth"], r["kind"], r["index"])
                if k not in fok:
                    continue
                shared_n += 1
                ours = r.get("score", 0) >= 0.5
                if fok[k] and not ours:
                    lost += 1
                elif ours and not fok[k]:
                    gained += 1
            if not shared_n:
                lines.append(f"{lab} vs f16: no overlapped depths")
            else:
                lines.append(f"{lab} vs f16: lost {lost} gained {gained} (paired {shared_n} questions)")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qreport.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("kld")
    k.add_argument("--log", required=True)
    k.add_argument("--md", action="store_true")
    h = sub.add_parser("hard")
    h.add_argument("--results", required=True)
    h.add_argument("--labels", required=True)
    h.add_argument("--base")
    h.add_argument("--md", action="store_true")
    l = sub.add_parser("lq")
    l.add_argument("--results", required=True)
    l.add_argument("--prefix", default="lq1_")
    l.add_argument("--md", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "kld":
        text = ""
        try:
            text = open(a.log, encoding="utf-8", errors="replace").read()
        except OSError:
            pass
        print(report_kld(text, a.md))
    elif a.cmd == "hard":
        print(report_hard(a.results, a.labels.split(","), a.base, a.md))
    else:
        print(report_lq(a.results, a.prefix, a.md))
    return 0


if __name__ == "__main__":
    sys.exit(main())

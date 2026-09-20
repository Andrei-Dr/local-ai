#!/usr/bin/env python3
"""ctxreport.py — turn a finished ctx1 run into the long-context verdict table (C1).

usage: ctxreport.py --runs DIR --log FILE [--out bench/reports/ctx1.md]
DIR/ctx1_*.slot.json are the slotclient row files; FILE is the ctx1.sh job log. Parser is tolerant:
unknown blocks ignored, missing fields None, never crash. Prints the report path; exit 0.
Sections: ladder | 32k variants | ubatch fit | depth decay | two-phase verdict | gates + dead rows.

usage: ctxreport.py --runs DIR [--job ctx1b] — with --job ctx1b the input is the ctx1b.sh log plus
runs/ctx1b_*.slot.json sidecars: presave rows pair with extend rows by ctx tag (c16k_g64, c32k, c131k,
c262k); the table shows both configs, restore/cache/ext counters, speedup, reply head and a per-pair
verdict (RESTORED / RE-PREFILLED / EMPTY REPLY / died / skipped), then per-family two-phase verdicts
(all *_dec_extend RESTORED => WORKS), the verbatim extend gate line and dead rows.
"""
import argparse, json, math, os, re, sys
from pathlib import Path

BUF_RE = re.compile(r"([\w ]*?)(KV|RS) buffer size\s*=\s*([\d.]+) MiB")
HDR_RE = re.compile(r"^##### (\S+) mode=(\w+) ctx=(\d+) reps=(\d+) \| (.*)$")
HDR1B_RE = re.compile(r"^##### (\S+) mode=(presave|extend) ctx=(\d+) reps=(\d+) pre_gen=(-?\d+) "
                      r"drop=(\d+) \| (.*)$")
SKIP_RE = re.compile(r"^##### (\S+) SKIPPED: (.*)$")


def derive_args(args):
    """kv / nkvo / mtp from flags; ub and cache_slots: LAST occurrence wins (llama-server behavior)."""
    t = args.split()
    kv, nkvo, mtp, ub, cache = "f16", False, False, None, None
    for i, x in enumerate(t):
        if x == "-ctk" and i + 1 < len(t):
            kv = "q4_0" if t[i + 1].startswith("q4") else "q8_0" if t[i + 1].startswith("q8") else kv
        elif x == "-nkvo":
            nkvo = True
        elif x == "--spec-type":
            mtp = True
        elif x == "-ub" and i + 1 < len(t) and t[i + 1].isdigit():
            ub = int(t[i + 1])
        elif x == "--moe-expert-cache" and i + 1 < len(t) and t[i + 1].isdigit():
            cache = int(t[i + 1])
    return {"kv": kv, "nkvo": nkvo, "mtp": mtp, "ub": ub if ub is not None else 512,
            "cache_slots": cache if cache is not None else 24}


def parse_log(text):
    """-> (labels, gates). labels[label] = {mode, ctx, reps, args, vram, died, exit, skipped, kv_mib, rs_mib, reason}"""
    labels, gates, cur = {}, [], None
    for ln in text.splitlines():
        m = HDR_RE.match(ln)
        if m:
            cur = labels.setdefault(m.group(1), {})
            cur.update(mode=m.group(2), ctx=int(m.group(3)), reps=int(m.group(4)), args=m.group(5))
            continue
        m = SKIP_RE.match(ln)
        if m:
            cur = None
            labels.setdefault(m.group(1), {})["skipped"] = m.group(2)
            continue
        cur_hdr = ln.startswith("##### ")
        if cur_hdr:
            cur = None
        if "restore gate:" in ln:
            gates.append(ln.strip())
            continue
        if cur is None:
            continue
        m = re.match(r"^\s*vram: (\d+)", ln)
        if m:
            cur["vram"] = int(m.group(1))
        m = re.match(r"^\s*\S+: SERVER DIED: (.*)", ln)
        if m:
            cur["died"], cur["reason"] = True, m.group(1)
        m = re.match(r"^\s*\S+: slotclient exit (\d+)", ln)
        if m:
            cur["exit"] = int(m.group(1))
        if "out of memory" in ln:
            cur["died"] = True
            cur.setdefault("reason", "out of memory")
        for _, kind, mb in BUF_RE.findall(ln):
            cur[kind.lower() + "_mib"] = round(cur.get(kind.lower() + "_mib", 0.0) + float(mb), 2)
    return labels, gates


def status_of(label, labels, runs):
    info = labels.get(label, {})
    if info.get("died"):
        return "died"
    if info.get("exit"):
        return "failed"
    if len(runs.get(label) or []) >= 1:
        return "ok"
    return "skipped"


def proj_seconds(n_tok, ub, F, m):
    """seconds to prefill n_tok at ubatch ub: each ubatch costs a fixed F plus m per token.
    t = (n / ub) * (F + m * ub) — NOT n * m, which ignores the per-ubatch floor entirely."""
    return (n_tok / ub) * (F + m * ub)


def pair_kind(label):
    """ctx1b pairing: strip the trailing _presave / _<mode>_extend / _extend, then reduce to the ctx
    tag (c16k_g64 / c32k / c131k / c262k). -> (tag, 'pre'|'ext'|None)."""
    lab = re.sub(r"^ctx1b_", "", label)
    if lab.endswith("_presave"):
        kind = "pre"
    elif lab.endswith("_extend"):
        kind = "ext"
    else:
        return None, None
    base = re.sub(r"_(presave|extend)$", "", lab)
    t = re.search(r"c\d+k(?:_g\d+)?", base)             # tag BEFORE dropping mode suffixes:
    return (t.group(0) if t else base.split("_")[0]), kind   # c16k_g64_d0 -> c16k_g64, c32k_dec -> c32k


def parse_log1b(text):
    """ctx1b flavor of parse_log: same tolerance, extra header fields."""
    labels, gates, cur = {}, [], None
    for ln in text.splitlines():
        m = HDR1B_RE.match(ln)
        if m:
            cur = labels.setdefault(m.group(1), {})
            cur.update(mode=m.group(2), ctx=int(m.group(3)), reps=int(m.group(4)), pre_gen=int(m.group(5)),
                       drop=int(m.group(6)), args=m.group(7))
            continue
        if ln.startswith("##### "):
            cur = None
        if "extend gate:" in ln:
            gates.append(ln.strip())
            continue
        if cur is None:
            continue
        m = re.match(r"^\s*\S+: SERVER DIED: (.*)", ln)
        if m:
            cur["died"], cur["reason"] = True, m.group(1)
        m = re.match(r"^\s*\S+: slotclient exit (\d+)", ln)
        if m:
            cur["exit"] = int(m.group(1))
        if "out of memory" in ln:
            cur["died"] = True
            cur.setdefault("reason", "out of memory")
    return labels, gates


def pair_verdict(elabel, labels, prow, erow):
    """RESTORED iff the extend processed <10% of the presave prompt tokens AND answered with a
    non-blank reply; EMPTY REPLY beats RE-PREFILLED when both signals are bad."""
    info = labels.get(elabel, {})
    if info.get("died"):
        return "died"
    if info.get("exit"):
        return "died"
    if erow is None:
        return "skipped"
    if not (erow.get("reply") or "").strip():
        return "EMPTY REPLY"
    pp, pe = prow.get("prompt_n"), erow.get("prompt_n")
    if pp and pe is not None and pe < 0.1 * pp:
        return "RESTORED"
    return "RE-PREFILLED"


def _cfg(args):
    d = derive_args(args or "")
    return f"ub{d['ub']} slots{d['cache_slots']}" + (" +MTP" if d["mtp"] else "")


def build_report1b(log_text, runs):
    labels, gates = parse_log1b(log_text)
    md = ["# ctx1b — presave / extend pairs", ""]
    pres, exts = {}, {}
    for lab in sorted(set(labels) | set(runs)):
        if not lab.startswith("ctx1b_"):
            continue
        tag, kind = pair_kind(lab)
        if kind == "pre":
            pres[tag] = lab
        elif kind == "ext":
            exts.setdefault(tag, []).append(lab)
    md += ["| ctx | kv | pre cfg | pre prompt_n | prefill t/s | prefill wall | slot MiB | save ms |"
           " ext cfg | restore ms | ext prompt_n | cache_n | ext_n | decode t/s | wall s | speedup |"
           " reply | verdict |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    verdicts = {}
    any_rows = False
    for tag in sorted(set(pres) | set(exts)):
        plab = pres.get(tag)
        prow = _one(runs.get(plab or "")) if plab else {}
        pinfo = labels.get(plab or "", {})
        for elab in exts.get(tag, []):
            any_rows = True
            einfo = labels.get(elab, {})
            erow = _one(runs.get(elab))
            v = pair_verdict(elab, labels, prow, erow if runs.get(elab) else None)
            verdicts[(tag, elab)] = (v, einfo.get("ctx") or prow.get("ctx") or pinfo.get("ctx"))
            pw, ew = prow.get("wall_s"), erow.get("wall_s")
            rep = (erow.get("reply") or "").replace("\n", " ").replace("|", "/")[:60]
            md.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | '{}' | {} |"
                      .format(pinfo.get("ctx") or einfo.get("ctx") or "-", derive_args(einfo.get("args", ""))["kv"],
                              _cfg(pinfo.get("args")), _fmt(prow.get("prompt_n"), "{:d}"),
                              _fmt(prow.get("prefill_tps")), _wall(pw), _mb(prow.get("slot_bytes")),
                              _fmt(prow.get("slot_ms"), "{:.0f}"), _cfg(einfo.get("args")),
                              _fmt(erow.get("slot_ms"), "{:.0f}"), _fmt(erow.get("prompt_n"), "{:d}"),
                              _fmt(erow.get("cache_n"), "{:d}"), _fmt(erow.get("ext_n"), "{:d}"),
                              _fmt(erow.get("decode_tps"), "{:.2f}"), _fmt(ew, "{:.1f}"),
                              _fmt(None if not (pw and ew) else pw / ew, "{:.1f}"), rep, v))
    if not any_rows:
        md.append("| (no presave/extend rows) |" + "||" * 17)

    md += ["", "## Two-phase verdicts", ""]
    fams = sorted({t for (t, _e), (_v, ctx) in verdicts.items() if ctx and ctx >= 32768})
    for t in fams:
        decs = [v for (tt, e), (v, _c) in verdicts.items() if tt == t and e.endswith("_dec_extend")]
        vv = ("WORKS" if decs and all(x == "RESTORED" for x in decs)
              else "FAILED" if decs else "not run")
        md.append(f"- `{t}`: {vv}")
    if not fams:
        md.append("- (none)")

    md += ["", "## Extend gate", ""]
    md += [f"- `{g}`" for g in gates] or ["- (not reached)"]

    md += ["", "## Dead rows", ""]
    dead = []
    for lab in sorted(set(labels) | set(runs)):
        if not lab.startswith("ctx1b_") or pair_kind(lab)[1] is None:
            continue
        st = status_of(lab, labels, runs)
        if st != "ok":
            info = labels.get(lab, {})
            reason = info.get("reason") or info.get("skipped") \
                or (f"slotclient exit {info['exit']}" if info.get("exit") else "no rows")
            dead.append(f"- `{lab}`: {st}: {reason}")
    md += dead or ["- (none)"]
    md.append("")
    return "\n".join(md) + "\n"


def fit(points):
    """least squares of T = F + m*ub over (ub, T) points; < 2 DISTINCT ub => None."""
    xs = {p[0] for p in points}
    if len(xs) < 2:
        return None
    n = len(points)
    sx = sum(p[0] for p in points); sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points); sxy = sum(p[0] * p[1] for p in points)
    d = n * sxx - sx * sx
    if d == 0:
        return None
    m = (n * sxy - sx * sy) / d
    return (sy - m * sx) / n, m


def _fmt(v, pat="{:.1f}", dash="-"):
    return pat.format(v) if v is not None else dash


def _wall(v):
    if v is None:
        return "-"
    s = f"{v:.0f} s"
    if v >= 600:
        s += f" ({int(v // 3600)}:{int(v % 3600 // 60):02d})"
    return s


def _delta_pct(new, base, nd=1):
    if new is None or base in (None, 0):
        return "-"
    return f"{(new - base) / base * 100:+.{nd}f}%"


def _one(rows):
    return rows[0] if rows else {}


def _mb(b):
    return _fmt(None if b is None else b / 1048576.0, "{:.0f}")


def build_report(log_text, runs):
    labels, gates = parse_log(log_text)
    tags = {}
    for lab in sorted(set(labels) | set(runs)):
        m = re.fullmatch(r"ctx1_(.+?)_(save|restore)", lab)
        if m:
            tags.setdefault(m.group(1), {})[m.group(2)] = lab
    rungs = []
    for tag, sides in tags.items():
        sv = sides.get("save")
        if sv is None or "restore" not in sides:      # rungs are save+restore PAIRS (variants: section 2)
            continue
        info = labels.get(sv, {})
        ctx = info.get("ctx", labels.get(sides["restore"], {}).get("ctx"))
        if ctx is None:
            ctx = int(_one(runs.get(sv) or []).get("prompt_n") or 0) or None
        rungs.append((ctx or 1 << 60, tag, sides))
    rungs.sort(key=lambda x: x[0])

    md = ["# ctx1 — long-context ladder report", ""]
    md += ["## Ladder", "",
           "| ctx | kv | slots | prompt_n | prefill t/s | prefill wall | decode t/s | vram | KV | RS | slot | save ms | restore ms | restore prompt_n | restore wall | speedup | status |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    if not rungs and not any(runs.values()):
        md.append("| (no rows) | | | | | | | | | | | | | | | | |")
        md += ["", "no rows", ""]
    rows_by_tag = {}
    for ctx, tag, sides in rungs:
        s_lab, r_lab = sides["save"], sides["restore"]
        s, r = _one(runs.get(s_lab)), _one(runs.get(r_lab))
        si = labels.get(s_lab, {})
        d = derive_args(si.get("args", ""))
        s_st, r_st = status_of(s_lab, labels, runs), status_of(r_lab, labels, runs)
        rows_by_tag[tag] = dict(ctx=ctx, s=s, r=r, d=d, si=si, s_st=s_st, r_st=r_st, tag=tag)
        spd = (_fmt(s.get("wall_s") / r["wall_s"], "{:.1f}")
               if s.get("wall_s") and r.get("wall_s") else "-")
        flag = ""
        if s.get("prompt_n") and r.get("prompt_n") is not None and r["prompt_n"] >= 0.1 * s["prompt_n"]:
            flag = " **RESTORE DID NOT SKIP PROMPT**"
        md.append(f"| {_fmt(ctx, '{:d}')} | {d['kv']} | {d['cache_slots']} | {_fmt(s.get('prompt_n'), '{:d}')} "
                  f"| {_fmt(s.get('prefill_tps'))} | {_wall(s.get('wall_s'))} | {_fmt(s.get('decode_tps'))} "
                  f"| {_fmt(si.get('vram'), '{:d}')} | {_fmt(si.get('kv_mib'))} | {_fmt(si.get('rs_mib'))} "
                  f"| {_mb(s.get('slot_bytes'))} | {_fmt(s.get('slot_ms'))} | {_fmt(r.get('slot_ms'))} "
                  f"| {_fmt(r.get('prompt_n'), '{:d}')} | {_fmt(r.get('wall_s'), '{:.1f}')} | {spd} "
                  f"| save {s_st} / restore {r_st}{flag} |")
    md.append("")

    md += ["## 32k variants vs baseline `ctx1_c32k_save`", "",
           "| variant | decode Δ% | prefill Δ% | vram ΔMiB |", "|---|---|---|---|"]
    base = _one(runs.get("ctx1_c32k_save"))
    base_vram = labels.get("ctx1_c32k_save", {}).get("vram")
    for lab in [x for x in sorted(set(labels) | set(runs))
                if re.fullmatch(r"ctx1_c32k_.+_save", x) and status_of(x, labels, runs) == "ok"]:
        r0 = _one(runs.get(lab))
        dv = None if not (labels.get(lab, {}).get("vram") is not None and base_vram) else \
            labels[lab]["vram"] - base_vram
        md.append(f"| `{lab}` | {_delta_pct(r0.get('decode_tps'), base.get('decode_tps'))} "
                  f"| {_delta_pct(r0.get('prefill_tps'), base.get('prefill_tps'))} "
                  f"| {'-' if dv is None else f'{dv:+d}'} |")
    md.append("")

    md += ["## Ubatch fit", ""]
    pts = {}
    for ctx, tag, sides in rungs:                      # ladder main rows first (baseline ub)
        lab = sides["save"]
        info = labels.get(lab, {})
        r0 = _one(runs.get(lab))
        if info.get("ctx") == 32768 and status_of(lab, labels, runs) == "ok" and r0.get("prefill_tps"):
            d = derive_args(info.get("args", ""))
            pts.setdefault(d["ub"], (d["ub"], d["ub"] / r0["prefill_tps"]))
    for lab in sorted(labels):                         # then 32k variant rows (pf1024/pf2048/...) — first ub wins
        info = labels.get(lab, {})
        if not lab.startswith("ctx1_c32k") or not lab.endswith("_save"):
            continue
        r0 = _one(runs.get(lab))
        if info.get("ctx") == 32768 and status_of(lab, labels, runs) == "ok" and r0.get("prefill_tps"):
            d = derive_args(info.get("args", ""))
            pts.setdefault(d["ub"], (d["ub"], d["ub"] / r0["prefill_tps"]))
    got = fit(sorted(pts.values()))
    if got is None:
        md += ["not enough rows", ""]
    else:
        F, m = got
        md += [f"F = {F:.3f} s fixed per ubatch, m = {m * 1000:.3f} ms/token, asymptote = "
               f"{(1 / m if m > 0 else float('inf')):.1f} tok/s", "",
               "| ub | predicted t/s | expert-only 100k | 200k | 250k |", "|---|---|---|---|---|"]
        for ub in (512, 1024, 2048, 4096):
            T = F + m * ub
            md.append(f"| {ub} | {ub / T:.1f} | {proj_seconds(100000, ub, F, m):.1f} s | "
                  f"{proj_seconds(200000, ub, F, m):.1f} s | {proj_seconds(250000, ub, F, m):.1f} s |")
        md += ["", "attention cost grows with depth and is NOT in this fit; see the depth-decay table", ""]
    md += ["## Depth decay", "", "| prompt_n | prefill t/s (% of 16k) | decode t/s (% of 16k) |", "|---|---|---|"]
    ref = next((rows_by_tag[t] for t in rows_by_tag if t in ("c16k", "16k") ), None)
    for ctx, tag, sides in rungs:
        rw = rows_by_tag[tag]
        if rw["s_st"] != "ok":
            continue
        s = rw["s"]
        pc = "" if not (ref and ref["s"].get("prefill_tps")) else \
            f" ({s.get('prefill_tps', 0) / ref['s']['prefill_tps'] * 100:.0f}%)"
        dc = "" if not (ref and ref["s"].get("decode_tps")) else \
            f" ({s.get('decode_tps', 0) / ref['s']['decode_tps'] * 100:.0f}%)"
        md.append(f"| {_fmt(s.get('prompt_n'), '{:d}')} | {_fmt(s.get('prefill_tps'))}{pc} | "
                  f"{_fmt(s.get('decode_tps'))}{dc} |")
    md.append("")

    md += ["## Two-phase verdict", ""]
    xv, xs = labels.get("ctx1_c32k_pf2048_xrestore"), labels.get("ctx1_c32k_pf2048_save")
    xr, xsv = _one(runs.get("ctx1_c32k_pf2048_xrestore")), _one(runs.get("ctx1_c32k_pf2048_save"))
    if not xr or not xsv.get("prompt_n"):
        verdict = "not run"
    elif status_of("ctx1_c32k_pf2048_xrestore", labels, runs) == "ok" \
            and xr.get("prompt_n", 0) < 0.1 * xsv["prompt_n"]:
        verdict = "WORKS"
    else:
        verdict = "FAILED"
    md += [f"two-phase (prefill config -> decode config) {verdict}", ""]

    md += ["## Gates and dead rows", ""]
    for g in gates:
        md.append(f"- `{g.removeprefix('    ')}`" if g.startswith("    ") else f"- `{g}`")
    dead = []
    for lab in sorted(set(labels) | set(runs)):
        if not re.fullmatch(r"ctx1_\S+", lab):
            continue
        st = status_of(lab, labels, runs)
        if st != "ok":
            reason = labels.get(lab, {}).get("reason") or labels.get(lab, {}).get("skipped") \
                or ("slotclient exit %d" % labels[lab]["exit"] if labels.get(lab, {}).get("exit") else "no rows")
            dead.append(f"- `{lab}`: {st}: {reason}")
    md += dead or ["- (none)"]
    md.append("")
    return "\n".join(md) + "\n"


def load_runs(dirpath, prefix="ctx1"):
    runs = {}
    for f in sorted(Path(dirpath).glob(prefix + "_*.slot.json")):
        try:
            d = json.load(open(f))
            runs[f.name.removesuffix(".slot.json")] = d.get("rows") or []
        except (json.JSONDecodeError, OSError):
            runs[f.name.removesuffix(".slot.json")] = []
    return runs


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ctxreport.py")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", default="bench/reports/ctx1.md")
    ap.add_argument("--job", choices=("ctx1", "ctx1b"), default="ctx1")
    a = ap.parse_args(argv)
    log = open(a.log, encoding="utf-8", errors="replace").read() if a.log and os.path.exists(a.log) else ""
    md = (build_report(log, load_runs(a.runs)) if a.job == "ctx1"
          else build_report1b(log, load_runs(a.runs, "ctx1b")))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    open(a.out, "w").write(md)
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

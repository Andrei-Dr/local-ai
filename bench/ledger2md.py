#!/usr/bin/env python3
"""Render bench/LEDGER.md (full per-run telemetry) AND bench/SCOREBOARD.md (per-axis model comparison)
from box/ledger.jsonl (written by the harness) + ledger_backfill.jsonl.
usage: ledger2md.py   (sync first: rsync root@i5.local:/ai/bench/ledger.jsonl box/)"""
import json, re
from pathlib import Path

HERE = Path(__file__).parent
SHORT = {"Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP.gguf": "Bonsai-27B PQ2_0-MTP", "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf": "Qwen3.6-35B-A3B IQ2_M",
         "Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ3_M.gguf": "Gemma4-26B-A4B IQ3_M"}


def smodel(m):
    """Short display name for a model file (shared by every table)."""
    return m.replace(".gguf", "").replace("-Uncensored-HauhauCS-Aggressive", "").replace("-Uncensored-HauhauCS-Balanced", "")
AXES = [("math", "gsm8k"), ("code", "humaneval"), ("knowledge", "mmlu_pro")]
# Absolute per-axis floors for the "best config" pick. Eligible = every axis BOTH within 1 sigma of the best
# clean score on that axis AND >= its floor (1 sigma catches noise ties; the floor stops a statistically
# excusable but actually-bad config sneaking in). On knowledge the 1-sigma band anchors on the best CLEAN
# or CLEAN-ENOUGH row and uses that row's SE. Winner = fastest eligible config. Tune here.
FLOOR = {"math": 90.0, "code": 85.0, "knowledge": 65.0}


def mmlu_trunc(r):
    """MMLU-Pro-specific truncation count (a GSM8K/HumanEval cut-off must NOT flag the knowledge column).
    Prefer the per-set field (qual.py writes it now); else recompute from the per-item results jsonl;
    else -1 = unknown, treated as 'possibly contaminated' since the run had some truncation."""
    mp = (r["quality"].get("sets") or {}).get("mmlu_pro") or {}
    if "truncated" in mp:
        return mp["truncated"]
    f = HERE / "qual" / "results" / f"{r['label']}.jsonl"
    if f.exists():
        latest = {}  # the jsonl is append-only across reruns: only the newest answer per item counts
        for l in f.open():
            x = json.loads(l)
            latest[x["id"]] = x
        return sum(1 for x in latest.values() if x.get("set") == "mmlu_pro" and x.get("finish") == "length")
    return -1 if r["quality"].get("truncated", 0) else 0


def _se_pct(sd, pct):
    """Standard error in points: recorded se_pct if present, else binomial from pct/n: 100*sqrt(p(1-p)/n)."""
    se = sd.get("se_pct")
    if se:
        return se
    n = sd.get("n") or 0
    if not n or pct is None:
        return 0.0
    p = max(0.0, min(1.0, pct / 100.0))
    return 100.0 * (p * (1 - p) / n) ** 0.5


def scoreboard(recs):
    """Per-axis model comparison. Answers which model wins on math / code / knowledge / speed, so the
    comparison lives in tooling, not in prose in notes.md. Knowledge rule: a cut-off answer scores wrong,
    so a row with MMLU-Pro cut-offs measures a FLOOR; its true score lies in [pct, pct + 100*cut/n]. Such a
    row renders `(<x)` and competes ONLY while that band is narrower than 1 SE (clean-enough); a wider band,
    or an unknown cut (old records, flagged `!`), never wins an axis nor feeds the picker."""
    qruns = [r for r in recs if r.get("kind") == "quality" and r.get("quality")]
    speed = [r for r in recs if r.get("kind") == "specbench" and r.get("prompts")]
    # tuned-best decode tok/s per model file (from specbench rows; lives under different labels than the quality run)
    best_decode = {}
    for r in speed:
        ds = [p.get("decode_tps") for p in r["prompts"] if p.get("decode_tps") is not None]
        if ds:
            best_decode[r["model"]] = max(best_decode.get(r["model"], 0), max(ds))
    # one quality row per label: latest by ts (reruns share a label with the old-cap run), and per SET the
    # newest record carrying that set — a rerun at a higher cap may cover only some sets
    latest, sup = {}, {}
    for r in qruns:
        lab = r["label"]
        if lab not in latest or (r.get("ts") or "") >= (latest[lab].get("ts") or ""):
            latest[lab] = r
        for name, sd in (r["quality"].get("sets") or {}).items():
            if sd.get("pct") is None:
                continue
            k = (lab, name)
            if k not in sup or (r.get("ts") or "") >= (sup[k].get("ts") or ""):
                sup[k] = r
    rows = []
    for r in latest.values():
        lab, dec = r["label"], best_decode.get(r["model"])
        row = {"label": lab, "model": r["model"], "se": {}, "n": {},
               "speed": dec if dec is not None else r["quality"].get("mean_decode_tps"),
               "speed_tag": "" if dec is not None else "~"}
        for ax, name in AXES:
            pr = sup.get((lab, name))
            sd = (pr["quality"].get("sets") or {}).get(name, {}) if pr else {}
            v = sd.get("pct")
            row[ax] = v
            row["se"][ax] = _se_pct(sd, v) if v is not None else None
            row["n"][ax] = sd.get("n")
            if ax == "knowledge":
                cut = mmlu_trunc(pr) if pr else -1
                n = sd.get("n") or 0
                row["cut"] = cut
                row["kn_ub"] = v + 100.0 * cut / n if (pr and cut >= 0 and n and v is not None) else None
                row["kn_clean"] = bool(pr and cut == 0)
                row["kn_ok"] = bool(row["kn_ub"] is not None and 100.0 * cut / n <= (row["se"][ax] or 0))
        for k in ("cut", "kn_ub", "kn_clean", "kn_ok"):
            row.setdefault(k, -1 if k == "cut" else False)
        row["kn_bad"] = not (row["kn_clean"] or row["kn_ok"])  # contaminated: `!`, never competes
        rows.append(row)
    rows.sort(key=lambda r: r["model"])
    competitive = lambda r: r["kn_clean"] or r["kn_ok"]  # knowledge may compete: zero cut, or band <= 1 SE

    def win(axis):  # knowledge pool = clean + clean-enough, ranked by the floor pct; other axes: any value
        cand = [r for r in rows if r[axis] is not None and (axis != "knowledge" or competitive(r))]
        if not cand:
            return None, any(r[axis] is not None for r in rows)
        return max(cand, key=lambda r: r[axis])["label"], False
    wins = {ax: win(ax) for ax, _ in AXES}
    speed_win = max((r for r in rows if r["speed"] is not None), key=lambda r: r["speed"], default=None)

    o = ["# Model scoreboard", "", f"{len(rows)} model configs. Generated by `bench/ledger2md.py` from the ledger. Do not edit by hand.",
         "Axes: math=GSM8K, code=HumanEval, knowledge=MMLU-Pro (temp 0, thinking off); speed=best decode tok/s (tuned config, may differ from the quality run's).",
         "", "| model | config | math (GSM8K) | code (HumanEval) | knowledge (MMLU-Pro) | speed tok/s |", "|---|---|---|---|---|---|"]
    for r in rows:
        cells = []
        for ax, _ in AXES:
            v = r[ax]
            if v is None:
                cells.append("-"); continue
            mark = " ✅" if wins[ax][0] == r["label"] else ""
            if ax == "knowledge" and not r["kn_clean"]:
                # contaminated or clean-enough: the floor pct plus its [pct, pct+100*cut/n] band; no band = unknown cut
                flag = "" if r["kn_ok"] else "!"
                bound = f" (<={r['kn_ub']:.1f})" if r["kn_ub"] is not None else ""
                v = f"{v:.1f}{flag}{bound}"
            else:
                v = f"{v:.1f}"
            cells.append(v + mark)
        sp = "-" if r["speed"] is None else f"{r['speed_tag']}{r['speed']:.1f}" + (" ✅" if speed_win and r["label"] == speed_win["label"] else "")
        o.append(f"| {smodel(r['model'])} | `{r['label']}` | {cells[0]} | {cells[1]} | {cells[2]} | {sp} |")
    o += ["", "**Axis winners:** " + ", ".join(f"{ax}=" + (wins[ax][0] or "?") for ax, _ in AXES) + (f", speed={speed_win['label']}" if speed_win else "")]
    if any(r["knowledge"] is not None and not r["kn_clean"] for r in rows):
        o += ["", "- `(<x)` = knowledge true score lies in [pct, x] with x = pct + 100*cut/n (cut = cut-off answers, which score wrong) — the pct is a FLOOR, not a point estimate.",
              "- Clean-enough (cut known AND 100*cut/n <= 1 SE, a band inside the noise) competes normally; `!` = contaminated (band wider than 1 SE) or cut unknown (old records) — never wins an axis, never best-config eligible."]
    if any(wins[ax][1] for ax, _ in AXES):
        o += ["", "- knowledge winner unresolved: no clean or clean-enough MMLU-Pro record exists for any label; resolve with a rerun."]
    if any(r["speed_tag"] for r in rows):
        o += ["", "- `~` speed = the quality run's in-run mean decode tok/s (no specbench row for that model); not the tuned best."]

    # --- Top 5 leaderboards (detail views; the matrix above stays canonical) ---
    sruns = {}
    for r in speed:
        if not r.get("completed"):
            continue
        if r["label"] not in sruns or (r.get("ts") or "") >= (sruns[r["label"]].get("ts") or ""):
            sruns[r["label"]] = r

    sents = []
    for lab, r in sruns.items():
        ds = [p.get("decode_tps") for p in r["prompts"] if isinstance(p.get("decode_tps"), (int, float))]
        sents.append({"label": lab, "model": r["model"], "mx": max(ds) if ds else None,
                      "mn": (sum(ds) / len(ds)) if ds else None, "np": len(ds), "r": r})
    sents.sort(key=lambda e: (e["mx"] is None, -(e["mx"] or 0.0), e["label"]))

    def speed_note(r):
        a = r.get("args") or ""
        spec = "no spec"
        if "--spec-type" in a or "-md" in a.split():
            k = re.search(r"--spec-draft-n-max\D*(\d+)", a)
            spec = f"MTP n={k.group(1) if k else '?'}"
        c = r.get("moe_cache")
        cache = "no cache" if not c else f"slots {c.get('slots', '?')}"
        v = r.get("vram_mib")
        return f"{cache}, {spec}, VRAM {v if v is not None else '-'} MiB, commit {(r.get('git') or {}).get('commit', '?')}"

    def srow(i, e):
        cell = "-" if e["mx"] is None else f"{e['mx']:.1f} (mean {e['mn']:.1f} over {e['np']} {'prompt' if e['np'] == 1 else 'prompts'})"
        return f"| {i} | `{e['label']}` | {smodel(e['model'])} | {cell} | {speed_note(e['r'])} |"

    def cell_score(r, ax):  # quality leaderboard score cell: pct ±se (n=N)
        v = r[ax]
        if v is None:
            return "-"
        return f"{v:.1f} ±{(r['se'][ax] or 0):.1f} (n={r['n'].get(ax) if r['n'].get(ax) is not None else '?'})"

    def qsort(ax):
        return lambda r: (r[ax] is None, -(r[ax] or 0.0), (r["speed"] is None, -(r["speed"] or 0.0)), r["label"])

    o += ["", "## Top 5 by dimension"]
    for ax in ("math", "code"):
        ranked = sorted(rows, key=qsort(ax))[:5]
        top = ranked[0] if ranked and ranked[0][ax] is not None else None
        o += ["", f"### {ax} ({'GSM8K' if ax == 'math' else 'HumanEval'})", "", "| rank | config | model | score | note |", "|---|---|---|---|---|"]
        for i, r in enumerate(ranked, 1):
            note = ("tie with #1 within 1 SE" if top is not None and i > 1 and r[ax] is not None
                    and top["se"][ax] is not None and r[ax] >= top[ax] - top["se"][ax] else "")
            o.append(f"| {i} | `{r['label']}` | {smodel(r['model'])} | {cell_score(r, ax)} | {note} |")

    comp = [r for r in rows if r["knowledge"] is not None and (r["kn_clean"] or r["kn_ok"])]
    cont = [r for r in rows if r["knowledge"] is not None and not (r["kn_clean"] or r["kn_ok"])]
    noks = sorted([r for r in rows if r["knowledge"] is None], key=lambda r: r["label"])
    ksrt = lambda rs: sorted(rs, key=lambda r: (-(r["knowledge"] or 0.0), (r["speed"] is None, -(r["speed"] or 0.0)), r["label"]))
    ct = max(comp, key=lambda r: r["knowledge"]) if comp else None

    def kn_cell(r):
        v = r["knowledge"]
        if v is None:
            return "-"
        s = f"{v:.1f}" + ("" if r["kn_clean"] else ("" if r["kn_ok"] else "!"))
        if not r["kn_clean"] and r["kn_ub"] is not None:
            s += f" (<={r['kn_ub']:.1f})"
        if r["se"]["knowledge"] is not None:
            s += f" ±{(r['se']['knowledge'] or 0):.1f}"
        return s + f" (n={r['n'].get('knowledge') if r['n'].get('knowledge') is not None else '?'})"

    o += ["", "### knowledge (MMLU-Pro)", "", "| rank | config | model | score | note |", "|---|---|---|---|---|"]
    for i, r in enumerate(ksrt(comp) + ksrt(cont) + noks, 1):
        if i > 5:
            break
        if r["knowledge"] is None:
            sc, note = "-", ""
        elif r["kn_bad"]:
            sc, note = kn_cell(r), "floor only, not ranked against clean rows"
        else:
            sc = kn_cell(r)
            note = ("tie with #1 within 1 SE" if ct and r is not ct and r["knowledge"] >= ct["knowledge"] - (ct["se"]["knowledge"] or 0) else "")
        o.append(f"| {i} | `{r['label']}` | {smodel(r['model'])} | {sc} | {note} |")
    o += ["", "### speed (decode tok/s)", "", "| rank | config | model | score | note |", "|---|---|---|---|---|"] + [srow(i, e) for i, e in enumerate(sents[:5], 1)]

    per_model = {}
    for e in sents:
        per_model.setdefault(e["model"], e)
    o += ["", "## Fastest config per model file", "", "| rank | config | model | score | note |", "|---|---|---|---|---|"] + [
        srow(i, e) for i, e in enumerate(sorted(per_model.values(), key=lambda e: (e["mx"] is None, -(e["mx"] or 0.0), e["label"])), 1)]

    # --- Best config: fastest that holds quality (within 1 sigma of best AND >= absolute floor, all axes;
    #     knowledge band = best clean-or-clean-enough pct minus THAT row's SE) ---
    best_q = {}
    for ax, _ in AXES:
        cl = [r for r in rows if r[ax] is not None and (ax != "knowledge" or competitive(r))]
        best_q[ax] = max(cl, key=lambda r: r[ax], default=None)

    def eligible(r):
        why = []
        for ax, _ in AXES:
            v = r[ax]
            if v is None:
                return False, [f"no {ax}"]
            if ax == "knowledge" and not competitive(r):
                return False, ["knowledge contaminated (cut-off band wider than 1 SE)"]
            if v < FLOOR[ax]:
                why.append(f"{ax} {v:.1f}<{FLOOR[ax]:g} floor")
            bq = best_q[ax]
            if bq is not None:
                if ax == "knowledge":
                    if v < bq["knowledge"] - (bq["se"]["knowledge"] or 0):  # band anchored on the BEST row's SE
                        why.append(f"knowledge {v:.1f} >1sigma below best {bq['knowledge']:.1f}")
                else:
                    se = (r["se"] or {}).get(ax) or 0
                    if v < bq[ax] - max(se, 0):  # more than 1 sigma below the best score on that axis
                        why.append(f"{ax} {v:.1f} >1sigma below best {bq[ax]:.1f}")
        return (len(why) == 0), why

    elig = [r for r in rows if eligible(r)[0] and r["speed"] is not None]
    o += ["", "## Best config (fastest that holds quality)", "",
          f"Eligible = every axis within 1 sigma of the best score (knowledge: best clean-or-clean-enough pct minus that row's SE) AND >= floor (math {FLOOR['math']:g}, code {FLOOR['code']:g}, knowledge {FLOOR['knowledge']:g}). Winner = fastest eligible.", ""]
    if elig:
        w = max(elig, key=lambda r: r["speed"])
        o.append(f"**WINNER: `{w['label']}` ({smodel(w['model'])}) at {w['speed_tag']}{w['speed']:.1f} tok/s** — GSM8K {w['math']:.1f}, HumanEval {w['code']:.1f}, MMLU-Pro {w['knowledge']:.1f}.")
        o += ["", "| rank | config | speed tok/s | math | code | knowledge |", "|---|---|---|---|---|---|"]
        for i, r in enumerate(sorted(elig, key=lambda r: -r["speed"])[:5], 1):
            o.append(f"| {i} | `{r['label']}` | {r['speed_tag']}{r['speed']:.1f} | {r['math']:.1f} | {r['code']:.1f} | {r['knowledge']:.1f} |")
    else:
        o.append("No config is eligible yet — every candidate is excluded (usually MMLU-Pro not clean; see truncation flags). Resolve with the cap-2048 reruns.")
    misses = sorted([r for r in rows if not eligible(r)[0] and all(r[ax] is not None for ax, _ in AXES)],
                    key=lambda r: (r["speed"] is None, -(r["speed"] or 0.0), r["label"]))[:5]
    if misses:
        o += ["", "Closest misses", "", "| rank | config | speed tok/s | why not |", "|---|---|---|---|"]
        for i, r in enumerate(misses, 1):
            sp = "-" if r["speed"] is None else f"{r['speed_tag']}{r['speed']:.1f}"
            o.append(f"| {i} | `{r['label']}` | {sp} | {', '.join(eligible(r)[1])} |")
    return "\n".join(o) + "\n"


def builds_md(recs):
    """Registry of every distinct build seen in the ledger, so 'the optimal build' is a rebuildable artifact,
    not a path on one box. A build's identity = (source dir, full commit, cmake flags, dirty-diff sha). For each,
    print exactly how to recreate it and which run labels used it. Any dirty build lists its diff blob."""
    runs = [r for r in recs if r.get("kind") in ("specbench", "quality") and r.get("git")]
    builds = {}
    for r in runs:
        g = r["git"]
        src = (r.get("build") or "").rsplit("/", 1)[0]
        key = (src, g.get("commit_full") or g.get("commit"), json.dumps(g.get("cmake"), sort_keys=True), g.get("dirty_diff_sha"))
        b = builds.setdefault(key, {"g": g, "src": src, "build": r.get("build"), "labels": [], "ts": []})
        b["labels"].append(r["label"]); b["ts"].append(r.get("ts") or "")
    o = ["# Build registry", "", f"{len(builds)} distinct builds in the ledger. Generated by `bench/ledger2md.py`. Do not edit by hand.",
         "Each build is identified by source dir + full commit + cmake flags + dirty-diff sha. A row is recreatable by",
         "checking out the commit, applying the diff blob if any (`git apply builds/diffs/<sha>.diff` on the box), and",
         "building with the listed cmake flags. `dirty` builds without a diff sha are from before provenance hardening (2026-09-20) and are NOT fully recreatable.", ""]
    for b in sorted(builds.values(), key=lambda x: max(x["ts"]), reverse=True):
        g, cm = b["g"], (b["g"].get("cmake") or {})
        o.append(f"## `{b['build']}` @ {g.get('describe') or g.get('commit')}")
        o.append("")
        o.append(f"- source: `{b['src']}` branch `{g.get('branch')}` commit `{g.get('commit_full') or g.get('commit')}`")
        o.append(f"- cmake: CUDA arch {cm.get('cuda_arch')}, {cm.get('build_type')}, GGML_CUDA={cm.get('ggml_cuda')}, cxx `{cm.get('cxx')}`")
        tc = g.get("toolchain") or {}
        o.append(f"- toolchain: {tc.get('cxx')}; nvcc {tc.get('nvcc')}")
        if g.get("dirty"):
            ds = g.get("dirty_diff_sha")
            o.append(f"- **DIRTY** — " + (f"diff blob `builds/diffs/{ds}.diff` (recreate: checkout commit, `git apply` that diff)" if ds else "no diff captured (pre-hardening); NOT recreatable"))
            if g.get("untracked"):
                o.append(f"  - untracked files at build time: {', '.join(g['untracked'])}")
        n = len(b["labels"])
        shown = sorted(set(b["labels"]))
        o.append(f"- {n} runs ({len(shown)} labels): {', '.join('`' + l + '`' for l in shown[:20])}{' ...' if len(shown) > 20 else ''}")
        o.append("")
    return "\n".join(o) + "\n"


def load(p):
    return [json.loads(l) for l in p.open()] if p.exists() else []


def pair(rows, key, fmt="{:.2f}"):
    v = [r.get(key) for r in rows or []]
    return " / ".join(fmt.format(x) if x is not None else "-" for x in v) if v else "-"


def am(t, a, b, fmt="{:g}"):
    if not t or t.get(a) is None:
        return "-"
    return f"{fmt.format(t[a])} / {fmt.format(t[b])}" if t.get(b) is not None else fmt.format(t[a])


def main():
    recs = load(HERE / "ledger_backfill.jsonl") + load(HERE / "box" / "ledger.jsonl")
    runs = [r for r in recs if r.get("kind") in ("specbench", "quality")]
    runs.sort(key=lambda r: (SHORT.get(r["model"], r["model"]), r.get("ts") or "", r["label"]))
    out = ["# Run ledger", "", f"{len(recs)} records. Generated by `bench/ledger2md.py` from `bench/box/ledger.jsonl` (harness) + `bench/ledger_backfill.jsonl` (pre-ledger runs). Do not edit by hand.",
           "", "Pairs are code / reasoning prompt. avg / max where two numbers share a cell. Fixed: `-fa on -c 4096 -t 6 --load-mode none --jinja --parallel 1`, 200 tok, temp 0, thinking off.", ""]
    model = None
    for r in runs:
        m = SHORT.get(r["model"], r["model"])
        if m != model:
            model = m
            out += ["", f"## {m}", "",
                    "| time | label | commit | args (OFFLOAD) | decode tok/s | wall tok/s | prefill tok/s | accept | VRAM MiB | GPU % | power W | temp C | PCIe rx GB/s | PCIe tx GB/s | CPU % | DRAM rd GB/s | cache slots / hit % / MiB per step | quality | note |",
                    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        t, p, c, q = r.get("telemetry") or {}, r.get("prompts"), r.get("moe_cache"), r.get("quality")
        g = r.get("git") or {}
        cache = "-" if not c else f"{c.get('slots')} (a{c.get('admit')}) / {c.get('hit_rate_pct') if c.get('hit_rate_pct') is not None else '?'} / {c.get('upload_mib_per_step') if c.get('upload_mib_per_step') is not None else '?'}"
        qual = "-" if not q else "; ".join(f"{k} {v['pct']}±{v['se_pct']} (n={v['n']})" for k, v in q["sets"].items()) + f"; trunc {q['truncated']}, empty {q['empty']}, {q['mean_decode_tps']} tok/s"
        note = " ".join(x for x in (("FAILED: " + (r.get("error") or "no generation")) if not r.get("completed") else None, r.get("note"), f"env {r['env']}" if r.get("env") else None) if x)
        out.append("| " + " | ".join([
            (r.get("ts") or "?")[:16].replace("T", " "), f"`{r['label']}`" + (" (quality)" if r.get("kind") == "quality" else ""),
            f"{g.get('commit', '?')}{'+dirty' if g.get('dirty') else ''}", f"`{r.get('args', '')}` ({r.get('offload_min_batch') if r.get('offload_min_batch') is not None else '?'})",
            pair(p, "decode_tps"), pair(p, "wall_tps"), pair(p, "prefill_tps", "{:.1f}"), pair(p, "acceptance"),
            str(r.get("vram_mib") or "-"), am(t, "gpu_util_avg", "gpu_util_max", "{:.0f}"), am(t, "power_w_avg", "power_w_max", "{:.0f}"), am(t, "gpu_temp_c_max", None, "{:.0f}"),
            am(t, "pcie_rx_gbs_avg", "pcie_rx_gbs_max", "{:.2f}"), am(t, "pcie_tx_gbs_avg", "pcie_tx_gbs_max", "{:.2f}"), am(t, "cpu_busy_pct", None, "{:.0f}"),
            am(t, "dram_read_gbs_avg", "dram_read_gbs_max", "{:.1f}"), cache, qual, note.replace("|", "\\|")]) + " |")
    other = [r for r in recs if r.get("kind") not in ("specbench", "quality")]
    out += ["", "## llama-bench / one-off measurements", ""]
    for r in other:
        body = {k: v for k, v in r.items() if k.startswith("results")}
        out.append(f"- **`{r['label']}`** ({r.get('model')}, commit {r['git']['commit']}, `{r.get('args')}`, {r.get('build')}): `{json.dumps(body)}` — {r.get('note', '')}")
    (HERE / "LEDGER.md").write_text("\n".join(out) + "\n")
    (HERE / "SCOREBOARD.md").write_text(scoreboard(recs))
    (HERE / "BUILDS.md").write_text(builds_md(recs))
    print(f"LEDGER.md: {len(runs)} run rows, {len(other)} other records; SCOREBOARD.md + BUILDS.md written")


if __name__ == "__main__":
    main()

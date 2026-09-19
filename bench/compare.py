#!/usr/bin/env python3
"""Model scoreboard: one row per model quality run, a checkmark on the winner of each axis.
Answers "which model wins on code / math / knowledge / speed" from the ledger, so the per-axis
comparison lives in tooling, not in prose in notes.md.

usage: compare.py            # all quality runs
       compare.py --best     # keep only the best-scoring config per base model
Axes: math=GSM8K, code=HumanEval, knowledge=MMLU-Pro (pct); speed=best decode tok/s.
A score is flagged (!) when its run had truncated answers (cut-offs score wrong => the pct is a FLOOR,
not comparable across models with different cut-off counts). Winners are only marked among clean scores;
if the leader on an axis is contaminated, the axis winner is '?' and the caveat is printed.
Speed prefers a matching specbench decode row (best over its prompts); falls back to the quality run's
in-run mean_decode_tps (tagged ~) when no speed row maps to the label.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
AXES = [("math", "gsm8k"), ("code", "humaneval"), ("knowledge", "mmlu_pro")]


def load(p):
    return [json.loads(l) for l in p.open()] if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", action="store_true", help="one best row per base model (by mean of the three axes)")
    a = ap.parse_args()

    recs = load(HERE / "ledger_backfill.jsonl") + load(HERE / "box" / "ledger.jsonl")
    qruns = [r for r in recs if r.get("kind") == "quality" and r.get("quality")]
    speed = [r for r in recs if r.get("kind") == "specbench" and r.get("prompts")]

    # best decode tok/s per MODEL from all speed rows (max over every specbench prompt for that model file).
    # This is the tuned best (e.g. MTP configs), which lives under different labels than the quality run.
    best_decode_model, best_decode_label = {}, {}
    for r in speed:
        ds = [p.get("decode_tps") for p in r["prompts"] if p.get("decode_tps") is not None]
        if not ds:
            continue
        m, top = r["model"], max(ds)
        if top > best_decode_model.get(m, 0):
            best_decode_model[m] = top
            best_decode_label[m] = r["label"]

    # one quality row per label: keep the latest by timestamp (reruns share a label with the old-cap run)
    latest = {}
    for r in qruns:
        if r["label"] not in latest or (r.get("ts") or "") > (latest[r["label"]].get("ts") or ""):
            latest[r["label"]] = r
    qruns = list(latest.values())

    rows = []
    for r in qruns:
        q = r["quality"]
        s = q["sets"]
        trunc = q.get("truncated", 0)
        label = r["label"]
        # speed: the model's tuned best from specbench rows; else this run's in-run tps (tagged ~)
        dec = best_decode_model.get(r["model"])
        dec_tag = "" if dec is not None else "~"
        if dec is None:
            dec = q.get("mean_decode_tps")
        rows.append({
            "label": label, "model": r["model"],
            "math": s.get("gsm8k", {}).get("pct"), "code": s.get("humaneval", {}).get("pct"),
            "knowledge": s.get("mmlu_pro", {}).get("pct"), "speed": dec, "speed_tag": dec_tag,
            "trunc": trunc, "n": {k: s.get(v, {}).get("n") for k, v in AXES},
            "mmlu_n": s.get("mmlu_pro", {}).get("n"),
        })

    if a.best:
        # base model = everything before the first quant tag; keep the row with the highest 3-axis mean
        def base(m):
            return m.split("-IQ")[0].split("-Q")[0].split("-i1")[0]

        by = {}
        for row in rows:
            b = base(row["model"])
            m = [row[k] for k, _ in AXES if row[k] is not None]
            row["_mean"] = sum(m) / len(m) if m else -1
            if b not in by or row["_mean"] > by[b]["_mean"]:
                by[b] = row
        rows = list(by.values())

    rows.sort(key=lambda r: r["model"])

    # per-axis winner among CLEAN scores only (mmlu_pro clean = no truncation on that run)
    def winner(axis):
        clean = [r for r in rows if r[axis] is not None and (axis != "knowledge" or r["trunc"] == 0)]
        if not clean:
            return None, any(r[axis] is not None for r in rows)  # (no winner, contaminated?)
        best = max(clean, key=lambda r: r[axis])
        # is the overall leader contaminated / excluded?
        allv = [r for r in rows if r[axis] is not None]
        dirty = axis == "knowledge" and max(allv, key=lambda r: r[axis])["trunc"] > 0 and max(allv, key=lambda r: r[axis]) is not best
        return best["label"], dirty
    speed_win = max((r for r in rows if r["speed"] is not None), key=lambda r: r["speed"], default=None)

    wins = {ax: winner(ax) for ax, _ in AXES}
    print(f"# Model scoreboard ({len(rows)} runs){' — best config per base model' if a.best else ''}\n")
    print("| model | config | math (GSM8K) | code (HumanEval) | knowledge (MMLU-Pro) | speed tok/s |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        cells = []
        for ax, _ in AXES:
            v = r[ax]
            if v is None:
                cells.append("-"); continue
            mark = " ✅" if wins[ax][0] == r["label"] else ""
            flag = "!" if (ax == "knowledge" and r["trunc"]) else ""
            cells.append(f"{v:.1f}{flag}{mark}")
        sp = "-" if r["speed"] is None else f"{r['speed_tag']}{r['speed']:.1f}" + (" ✅" if speed_win and r["label"] == speed_win["label"] else "")
        short = r["model"].replace(".gguf", "").replace("-Uncensored-HauhauCS-Aggressive", "").replace("-Uncensored-HauhauCS-Balanced", "")
        print(f"| {short} | `{r['label']}` | {cells[0]} | {cells[1]} | {cells[2]} | {sp} |")

    print("\n**Axis winners:** " + ", ".join(
        f"{ax}=" + (wins[ax][0] if wins[ax][0] else "?") for ax, _ in AXES) +
        (f", speed={speed_win['label']}" if speed_win else ""))
    notes = []
    if any(r["trunc"] for r in rows):
        notes.append("`!` = MMLU-Pro run had truncated answers (cut-offs score wrong => that pct is a FLOOR, not comparable); "
                     "winner is marked only among runs with zero truncation.")
    for ax, _ in AXES:
        if wins[ax][1]:
            notes.append(f"knowledge leader is contaminated by truncation; true {ax} winner unresolved until a clean re-run.")
    if any(r["speed_tag"] for r in rows):
        notes.append("`~` speed = the quality run's in-run mean decode tok/s (no matching specbench row); not the tuned best.")
    for n in notes:
        print("\n- " + n)


if __name__ == "__main__":
    main()

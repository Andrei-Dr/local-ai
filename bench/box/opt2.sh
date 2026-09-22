#!/bin/bash
# OPT2: confidence-gated MTP drafting. --spec-draft-p-min defaults to 0 (always draft n_max tokens); opt1 showed n_max 3 is
# content dependent (+3.7% code, +7.8% edit, -3.2% reason, -8.1% long): long drafts pay on predictable text and waste verify
# work on hard text. p_min stops a draft at the first token whose draft probability is below P, so the draft length adapts per
# position. Speculation is lossless (the target verifies every token), so only speed is measured here.
# Read: 4-prompt mean decode vs 3 interleaved bases (n_max 2, p_min 0); a WIN beats the base mean by more than the base spread
# with no prompt below base by more than that spread.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf; PY=/ai/.venv/bin/python
export BUILD=/ai/src/llama.cpp-mainline/build75 EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32
ARGS="-ot exps=CPU --moe-expert-cache 26 -ub 128 -b 256 -md $HEAD --spec-type draft-mtp"
arm() { local l=$1; shift; echo "##### $l | $*"; ./specbench.sh 999 $l $ARGS "$@" 2>&1; grep -hE "statistics +draft|out of memory" server_$l.log | tail -1 | cut -c1-170 | sed 's/^/    /'; }
arm opt2_base_a   --spec-draft-n-max 2
arm opt2_n3p06    --spec-draft-n-max 3 --spec-draft-p-min 0.6
arm opt2_n3p08    --spec-draft-n-max 3 --spec-draft-p-min 0.8
arm opt2_base_b   --spec-draft-n-max 2
arm opt2_n4p07    --spec-draft-n-max 4 --spec-draft-p-min 0.7
arm opt2_n4p085   --spec-draft-n-max 4 --spec-draft-p-min 0.85
arm opt2_n2p05    --spec-draft-n-max 2 --spec-draft-p-min 0.5
arm opt2_base_c   --spec-draft-n-max 2
echo "--- READ"
$PY - <<'PY'
import json
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
B = [L(b) for b in ("opt2_base_a", "opt2_base_b", "opt2_base_c")]
kinds = list(B[0])
bm = {k: sum(b[k]["decode_tps"] for b in B) / 3 for k in kinds}
bs = {k: max(b[k]["decode_tps"] for b in B) - min(b[k]["decode_tps"] for b in B) for k in kinds}
mean = lambda d: sum(d.values()) / len(d)
print(f"  base       mean {mean(bm):6.2f} (spread {mean(bs):.2f})")
for arm in ("opt2_n3p06", "opt2_n3p08", "opt2_n4p07", "opt2_n4p085", "opt2_n2p05"):
    try: a = L(arm)
    except Exception as e: print(f"  {arm:11s} NO RESULT ({e})"); continue
    m = mean({k: a[k]["decode_tps"] for k in kinds})
    worst = min(a[k]["decode_tps"] - bm[k] + bs[k] for k in kinds)
    win = m - mean(bm) > mean(bs) and worst >= 0
    per = " ".join(f"{k} {100 * (a[k]['decode_tps'] / bm[k] - 1):+.1f}%" for k in kinds)
    acc = " ".join(f"{k} {a[k]['acceptance']}" for k in kinds)
    print(f"  {arm:11s} mean {m:6.2f} ({100 * (m / mean(bm) - 1):+5.1f}%) | {per} | accept {acc} | {'WIN' if win else 'no'}")
PY
echo OPT2_DONE

#!/bin/bash
# OVL14: which part of the overlap costs host time on EVERY graph? ovl13 (11:34): overlap on -> decode after 9.3k: kernels busy
# unchanged (14.94 vs 15.00 ms/token) but host idle 5.98 -> 7.31 ms/token (max gap 16.8 -> 64.6 ms); specbench (short prompts,
# overlap never engages) decode -2.5..-5.8% and short-prompt prefill slower (code 54.9 -> 49.2, edit 143.5 -> 119.0).
# Arms (t2 combo, specbench serving config, 2 interleaved rounds): off | plan (GGML_SCHED_MOE_PREFETCH=2: hoisted layout, no second
# stream) | on (=1). plan ~ on => the plan/layout path (galloc / realloc / graph_copy); plan ~ off => the second backend instance.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
[ "$(git -C $T2 rev-parse --abbrev-ref HEAD)" = "combo" ] || { echo "OVL14_REFUSED: $T2 not on combo"; exit 1; }
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
SERVE="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
for r in a b; do for a in off plan on; do
  l=ovl14_${a}_$r; case $a in off) E="";; plan) E="GGML_SCHED_MOE_PREFETCH=2";; on) E="GGML_SCHED_MOE_PREFETCH=1";; esac
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  echo "##### $l | ${E:-off} | $(date +%T)"
  env -u LD_LIBRARY_PATH $E GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $SERVE 2>&1 | grep -E "^$l|FOREIGN|DIED" | cut -c1-100 | sed 's/^/    /'
  echo "    $(grep -cE 'realloc|reserv' server_$l.log) realloc/reserve lines | $(grep -oE 'graph_reserve[^\n]{0,60}|sched_reserve[^\n]{0,60}' server_$l.log | sort | uniq -c | sort -rn | head -2 | tr '\n' ' ')"
done; done
$PY - <<'PY'
import json
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
A = ("off", "plan", "on")
S = {a: [L(f"ovl14_{a}_{r}") for r in "ab"] for a in A}
kinds = list(S["off"][0]); m = lambda a, k, f="decode_tps": sum(x[k][f] for x in S[a]) / 2
for k in kinds:
    print(f"  {k:6s} decode " + " | ".join(f"{a} {m(a, k):6.2f}" for a in A) + " || prefill " + " | ".join(f"{a} {m(a, k, 'prefill_tps'):6.1f}" for a in A))
M = {a: sum(m(a, k) for k in kinds) / len(kinds) for a in A}
sp = sum(abs(S["off"][0][k]["decode_tps"] - S["off"][1][k]["decode_tps"]) for k in kinds) / len(kinds)
print(f"  MEAN off {M['off']:.2f} (spread {sp:.2f}) | plan {M['plan']:.2f} ({100 * (M['plan'] / M['off'] - 1):+.1f}%) | on {M['on']:.2f} ({100 * (M['on'] / M['off'] - 1):+.1f}%)")
PY
echo OVL14_DONE

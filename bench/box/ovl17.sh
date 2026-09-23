#!/bin/bash
# OVL17: is the overlap's "decode cost" real code cost? ovl16 (12:12): plan gated to >= 256-token batches (nothing planned for short
# prompts, decode or the reserve graph; compute buffer back to 129 MiB) STILL showed decode -6.4% and short-prompt prefill -15%.
# No code path is left that could cost that when nothing is planned -> test the measurement: ABBA-ordered A/A/B, NOTHING planned in
# any arm: U = env unset | Z = GGML_SCHED_MOE_PREFETCH=0 | N = =1 with MIN_IDS=99999999 (plan code runs, can never plan).
# Order U Z N N Z U (two passes of specbench each). U == Z == N (within U's spread) => the earlier losses need re-examination.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
SERVE="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
i=0
for a in U Z N N Z U; do
  i=$((i+1)); l=ovl17_${a}_$i
  case $a in U) E="";; Z) E="GGML_SCHED_MOE_PREFETCH=0";; N) E="GGML_SCHED_MOE_PREFETCH=1 GGML_SCHED_MOE_PREFETCH_MIN_IDS=99999999";; esac
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  echo "##### $l | ${E:-unset} | $(date +%T) | gpu $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader)"
  env -u LD_LIBRARY_PATH $E GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $SERVE 2>&1 | grep -E "^$l" | cut -c1-100 | sed 's/^/    /'
done
$PY - <<'PY'
import json, glob
L = lambda l: {r["prompt"]: r for r in json.load(open(l))["rows"]}
A = {"U": [], "Z": [], "N": []}
for f in sorted(glob.glob("/ai/bench/runs/ovl17_*.client.json")):
    A[f.split("ovl17_")[1][0]].append(L(f))
kinds = list(A["U"][0])
dec = lambda a: [sum(x[k]["decode_tps"] for k in kinds) / len(kinds) for x in A[a]]
pre = lambda a, k: sum(x[k]["prefill_tps"] for x in A[a]) / len(A[a])
for a in "UZN":
    print(f"  {a}: 4-prompt mean decode per run {[round(v, 2) for v in dec(a)]} | prefill code {pre(a, 'code'):.1f} edit {pre(a, 'edit'):.1f} long {pre(a, 'long'):.1f}")
u = dec("U"); sp = abs(u[0] - u[1]); mu = sum(u) / 2
for a in "ZN":
    m = sum(dec(a)) / 2
    print(f"  {a} vs U: {100 * (m / mu - 1):+.1f}% (U spread {sp:.2f}) -> {'SAME' if abs(m - mu) <= sp else 'DIFFERENT'}")
PY
echo OVL17_DONE

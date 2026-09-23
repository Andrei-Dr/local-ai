#!/bin/bash
# OVL19: bisect the overlap's plan-independent cost. ovl17: =1 with MIN_IDS=inf (loop runs, never plans) -7.1% vs unset (ABBA).
# Build: t2 combo 438aecb (+ GGML_SCHED_MOE_PREFETCH_SKIPLOOP=1: mode on, plan loop skipped). ABBA: U S N N S U with
# U = unset | S = =1 + SKIPLOOP | N = =1 + MIN_IDS=inf. S == U => the loop body is the cost; S == N => something outside the loop.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; SHA=438aecb; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
[ -z "$(git -C $T2 status --porcelain --untracked-files=no)" ] || { echo "OVL19_REFUSED: $T2 has local changes"; exit 1; }
[ "$(git -C $T2 rev-parse --abbrev-ref HEAD)" = "combo" ] || { echo "OVL19_REFUSED: $T2 not on combo"; exit 1; }
git -C $T2 merge -q --ff-only $SHA || { echo "OVL19_REFUSED: cannot fast-forward to $SHA"; exit 1; }
cmake --build $NEW -j6 --target llama-server > ovl19_build.log 2>&1 || { grep error ovl19_build.log | head; echo "OVL19_FAILED: build"; exit 1; }
strings $NEW/bin/libggml-base.so* | grep -q SKIPLOOP || { echo "OVL19_FAILED: build lacks SKIPLOOP"; exit 1; }
echo "    built $(git -C $T2 rev-parse --short HEAD) | $(date +%T)"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
SERVE="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
i=0
for a in U S N N S U; do
  i=$((i+1)); l=ovl19_${a}_$i
  case $a in U) E="";; S) E="GGML_SCHED_MOE_PREFETCH=1 GGML_SCHED_MOE_PREFETCH_SKIPLOOP=1";; N) E="GGML_SCHED_MOE_PREFETCH=1 GGML_SCHED_MOE_PREFETCH_MIN_IDS=99999999";; esac
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  echo "##### $l | ${E:-unset} | $(date +%T)"
  env -u LD_LIBRARY_PATH $E GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $SERVE 2>&1 | grep -E "^$l" | cut -c1-100 | sed 's/^/    /'
done
$PY - <<'PY'
import json, glob
L = lambda l: {r["prompt"]: r for r in json.load(open(l))["rows"]}
A = {"U": [], "S": [], "N": []}
for f in sorted(glob.glob("/ai/bench/runs/ovl19_*.client.json")):
    A[f.split("ovl19_")[1][0]].append(L(f))
kinds = list(A["U"][0])
dec = lambda a: [sum(x[k]["decode_tps"] for k in kinds) / len(kinds) for x in A[a]]
pre = lambda a: sum(x["code"]["prefill_tps"] for x in A[a]) / len(A[a])
for a in "USN":
    print(f"  {a}: decode per run {[round(v, 2) for v in dec(a)]} | code prefill {pre(a):.1f}")
u = dec("U"); sp = abs(u[0] - u[1]); mu = sum(u) / 2
for a in "SN":
    m = sum(dec(a)) / 2
    print(f"  {a} vs U: {100 * (m / mu - 1):+.1f}% (U spread {sp:.2f}) -> {'SAME' if abs(m - mu) <= sp else 'DIFFERENT'}")
PY
echo OVL19_DONE

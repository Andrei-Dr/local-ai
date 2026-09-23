#!/bin/bash
# OVL20: does the plan-loop fix (combo a9bd82c: op test first, per-backend CUDA verdict cached) remove the overlap's per-graph cost?
# ovl19: skipping the loop == unset; running it (never planning) -7.3%. ABBA U N O O N U: U unset | N =1 MIN_IDS=inf | O =1 (real).
# Then the 9,279-token prompt (ubp 2048) U vs O, 2 rounds.
# PRE-REGISTERED: FIXED if N is within U's spread; the overlap PASSES if O's specbench mean decode is within U's spread AND O's 9.3k
# prefill beats U by more than U's 9.3k spread AND O's decode after 9.3k is within the larger of the two arms' spreads of U's.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; SHA=a9bd82c; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
[ -z "$(git -C $T2 status --porcelain --untracked-files=no)" ] || { echo "OVL20_REFUSED: $T2 has local changes"; exit 1; }
git -C $T2 merge -q --ff-only $SHA || { echo "OVL20_REFUSED: cannot fast-forward to $SHA"; exit 1; }
cmake --build $NEW -j6 --target llama-server > ovl20_build.log 2>&1 || { grep error ovl20_build.log | head; echo "OVL20_FAILED: build"; exit 1; }
echo "    built $(git -C $T2 rev-parse --short HEAD) | $(date +%T)"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
SERVE="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
i=0
for a in U N O O N U; do
  i=$((i+1)); l=ovl20_${a}_$i
  case $a in U) E="";; N) E="GGML_SCHED_MOE_PREFETCH=1 GGML_SCHED_MOE_PREFETCH_MIN_IDS=99999999";; O) E="GGML_SCHED_MOE_PREFETCH=1";; esac
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  echo "##### $l | ${E:-unset} | $(date +%T)"
  env -u LD_LIBRARY_PATH $E GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $SERVE 2>&1 | grep -E "^$l" | cut -c1-100 | sed 's/^/    /'
done
for r in a b; do for a in U O; do
  l=ovl20_pf_${a}_$r; E=""; [ $a = O ] && E="GGML_SCHED_MOE_PREFETCH=1"
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $E GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 $NEW/bin/llama-server -m $K2 -ngl 999 -fa on -c 12288 -t 6 \
    --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 -ot exps=CPU --moe-expert-cache 22 -md $HEAD \
    --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 4096 -ubp 2048 > server_$l.log 2>&1 &
  NP=$!
  for j in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  echo -n "    "; python3 longpf.py $l 40000 2>&1
  kill $NP; wait $NP 2>/dev/null
done; done
$PY - <<'PY'
import json, glob
L = lambda l: {r["prompt"]: r for r in json.load(open(l))["rows"]}
A = {"U": [], "N": [], "O": []}
for f in sorted(glob.glob("/ai/bench/runs/ovl20_[UNO]_*.client.json")):
    A[f.split("ovl20_")[1][0]].append(L(f))
kinds = list(A["U"][0])
dec = lambda a: [sum(x[k]["decode_tps"] for k in kinds) / len(kinds) for x in A[a]]
for a in "UNO":
    print(f"  {a}: decode per run {[round(v, 2) for v in dec(a)]} | code prefill {sum(x['code']['prefill_tps'] for x in A[a]) / 2:.1f} | long prefill {sum(x['long']['prefill_tps'] for x in A[a]) / 2:.1f}")
u = dec("U"); sp = abs(u[0] - u[1]); mu = sum(u) / 2
mn, mo = sum(dec("N")) / 2, sum(dec("O")) / 2
print(f"  N vs U {100 * (mn / mu - 1):+.1f}% | O vs U {100 * (mo / mu - 1):+.1f}% (U spread {sp:.2f}) -> {'FIXED' if abs(mn - mu) <= sp else 'NOT FIXED'}")
P = lambda a, f: [json.load(open(f"/ai/bench/runs/ovl20_pf_{a}_{r}.longpf.json"))[f] for r in "ab"]
pu, po = P("U", "prefill_tps"), P("O", "prefill_tps"); du, do = P("U", "decode_tps"), P("O", "decode_tps")
psp = abs(pu[0] - pu[1]); dsp = max(abs(du[0] - du[1]), abs(do[0] - do[1]))
ok = abs(mo - mu) <= sp and sum(po) / 2 - sum(pu) / 2 > psp and abs(sum(do) / 2 - sum(du) / 2) <= dsp
print(f"  9.3k prefill U {pu} O {po} ({100 * (sum(po) / sum(pu) - 1):+.1f}%) | decode after U {du} O {do} -> overlap {'PASS' if ok else 'FAIL'}")
PY
echo OVL20_DONE

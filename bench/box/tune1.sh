#!/bin/bash
# TUNE1 (leads 1 + 4, config only, STABLE d0fd493 full serving env incl. GGML_SCHED_MOE_PREFETCH=1):
#   1 draft length at LONG context: n2 vs n3 on the 9,279-token prompt (mtpn1: n3 +4.1% specbench, -2.9% at 9.3k inside noise)
#   4 cache slots now that decode-mode VRAM is known: vram1 +2.2% at cache 30 (inside spread, pre-FR-Spec / pre-overlap)
# Specbench arms (2 rounds): n3 c26 | n3 c28 | n3 c30. 9.3k arms (2 rounds, -c 12288): n2 c22 | n3 c22 | n3 c24.
# PRE-REGISTERED: (1) n3 becomes the long-context default if its 9.3k decode mean is >= n2's minus n2's spread; (4) a cache size
# WINS if its specbench 4-prompt mean beats c26 by more than c26's spread (and no OOM / slot re-allocation failure).
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
V2=/ai/src/llama.cpp-v2/build75; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf; VOCAB=$M/mtp-Qwen3.6-35B-A3B-vocab49k.bin
ENVS="GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_CUDA_FA_MMA_MAX_KV=4096 LLAMA_MTP_VOCAB_FILE=$VOCAB GGML_SCHED_MOE_PREFETCH=1"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$V2
for r in a b; do for c in 26 28 30; do
  l=tune1_c${c}_$r
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  echo "##### $l | $(date +%T)"
  env -u LD_LIBRARY_PATH $ENVS ./specbench.sh 999 $l -ot exps=CPU --moe-expert-cache $c -md $HEAD --spec-type draft-mtp --spec-draft-n-max 3 \
    -ub 128 -b 2048 -ubp 2048 2>&1 | grep -E "^$l|FOREIGN|DIED" | cut -c1-110 | sed 's/^/    /'
  echo "    $(grep -oE 'hit rate [0-9.]+%' server_$l.log | tail -1) $(grep -hoE 'could not re-allocate|out of memory' server_$l.log | head -1)"
done; done
for r in a b; do for a in n2c22 n3c22 n3c24; do
  l=tune1_pf_${a}_$r; n=${a:1:1}; c=${a:3:2}
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $ENVS GGML_OP_OFFLOAD_MIN_BATCH=32 $V2/bin/llama-server -m $K2 -ngl 999 -fa on -c 12288 -t 6 --load-mode none --jinja \
    --parallel 1 --port 8099 --cache-ram 0 -ot exps=CPU --moe-expert-cache $c -md $HEAD --spec-type draft-mtp --spec-draft-n-max $n \
    -ub 128 -b 4096 -ubp 2048 > server_$l.log 2>&1 &
  NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  echo -n "    "; python3 longpf.py $l 40000 2>&1 | tr -d '\n'; echo " | $(grep -hoE 'could not re-allocate|out of memory' server_$l.log | head -1)"
  kill $NP; wait $NP 2>/dev/null
done; done
$PY - <<'PY'
import json, os
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
S = {c: [L(f"tune1_c{c}_{r}") for r in "ab"] for c in (26, 28, 30) if all(os.path.exists(f"/ai/bench/runs/tune1_c{c}_{r}.client.json") for r in "ab")}
kinds = list(S[26][0]); m = lambda c, k: sum(x[k]["decode_tps"] for x in S[c]) / 2
M = {c: sum(m(c, k) for k in kinds) / len(kinds) for c in S}
sp = sum(abs(S[26][0][k]["decode_tps"] - S[26][1][k]["decode_tps"]) for k in kinds) / len(kinds)
print("  specbench n3: " + " | ".join(f"c{c} {M[c]:.2f} ({100 * (M[c] / M[26] - 1):+.1f}%)" for c in S) + f" (c26 spread {sp:.2f})")
P = lambda a: [json.load(open(f"/ai/bench/runs/tune1_pf_{a}_{r}.longpf.json"))["decode_tps"] for r in "ab" if os.path.exists(f"/ai/bench/runs/tune1_pf_{a}_{r}.longpf.json")]
p2, p3, p3c = P("n2c22"), P("n3c22"), P("n3c24")
sp2 = abs(p2[0] - p2[1]) if len(p2) == 2 else 0
print(f"  9.3k decode n2c22 {p2} | n3c22 {p3} | n3c24 {p3c} -> n3 long-context {'OK' if p3 and sum(p3) / len(p3) >= sum(p2) / len(p2) - sp2 else 'NOT OK'}")
PY
echo TUNE1_DONE

#!/bin/bash
# MTPN1 (lead H): re-tune the MTP draft length now that each draft step is cheaper. promo4 (12:55): FR-Spec cut the draft head from
# 248k to 49k rows (the head was ~90% of a draft step's reads) -> the break-even of drafting one more token moved. opt1 (LEGACY, full
# head): n=3 best on edit, n=2 served. STABLE now (c75018b) with the full serving env; arms --spec-draft-n-max 2 | 3 | 4, 2 rounds,
# specbench + the 9,279-token prompt.
# PRE-REGISTERED: n WINS over 2 if its specbench 4-prompt mean decode beats n=2 by more than n=2's spread AND the 9.3k decode is not
# below n=2's by more than n=2's 9.3k spread.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
V2=/ai/src/llama.cpp-v2/build75; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf; VOCAB=$M/mtp-Qwen3.6-35B-A3B-vocab49k.bin
ENVS="GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_CUDA_FA_MMA_MAX_KV=4096 LLAMA_MTP_VOCAB_FILE=$VOCAB"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$V2
for r in a b; do for n in 2 3 4; do
  l=mtpn1_n${n}_$r
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  echo "##### $l | $(date +%T)"
  env -u LD_LIBRARY_PATH $ENVS ./specbench.sh 999 $l -ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max $n \
    -ub 128 -b 2048 -ubp 2048 2>&1 | grep -E "^$l|FOREIGN|DIED" | cut -c1-110 | sed 's/^/    /'
  l=mtpn1_pf_n${n}_$r
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $ENVS GGML_OP_OFFLOAD_MIN_BATCH=32 $V2/bin/llama-server -m $K2 -ngl 999 -fa on -c 12288 -t 6 --load-mode none --jinja \
    --parallel 1 --port 8099 --cache-ram 0 -ot exps=CPU --moe-expert-cache 22 -md $HEAD --spec-type draft-mtp --spec-draft-n-max $n \
    -ub 128 -b 4096 -ubp 2048 > server_$l.log 2>&1 &
  NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  echo -n "    "; python3 longpf.py $l 40000 2>&1
  kill $NP; wait $NP 2>/dev/null
done; done
$PY - <<'PY'
import json
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
S = {n: [L(f"mtpn1_n{n}_{r}") for r in "ab"] for n in (2, 3, 4)}
kinds = list(S[2][0]); m = lambda n, k, f="decode_tps": sum(x[k][f] for x in S[n]) / 2
for k in kinds:
    print(f"  {k:6s} " + " | ".join(f"n{n} {m(n, k):6.2f} (acc {m(n, k, 'acceptance'):.3f})" for n in (2, 3, 4)))
M = {n: sum(m(n, k) for k in kinds) / len(kinds) for n in S}
sp = sum(abs(S[2][0][k]["decode_tps"] - S[2][1][k]["decode_tps"]) for k in kinds) / len(kinds)
P = {n: [json.load(open(f"/ai/bench/runs/mtpn1_pf_n{n}_{r}.longpf.json"))["decode_tps"] for r in "ab"] for n in (2, 3, 4)}
psp = abs(P[2][0] - P[2][1])
for n in (3, 4):
    ok = M[n] - M[2] > sp and sum(P[n]) / 2 >= sum(P[2]) / 2 - psp
    print(f"  n{n}: specbench {M[n]:.2f} vs n2 {M[2]:.2f} ({100 * (M[n] / M[2] - 1):+.1f}%, spread {sp:.2f}) | 9.3k {P[n]} vs {P[2]} -> {'WIN' if ok else 'NOT PROVEN'}")
PY
echo MTPN1_DONE

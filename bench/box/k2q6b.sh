#!/bin/bash
# K2Q6B (follow-up to k2q6 00:17): K2q6 cut KLD 0.2039 -> 0.1146 (-44%), same-top 81.97 -> 86.76%, but its c22 specbench arm hit
# "out of memory" on the long prompt in pass a (prefill mode's compute buffer; k2d at c22 passed both passes = the c22 edge).
# (A) speed at cache sizes with margin: specbench n3 K2 c26 | K2q6 c20 | K2q6 c21; 9.3k n2 K2 c22 | K2q6 c18 | K2q6 c20 (2 rounds).
# (B) token_embd: K2 keeps it IQ3_S; it lives in HOST RAM (get_rows on CPU), so a Q6_K embedding costs ~+190 MB RAM and 0 VRAM.
#     K2q6e = K2q6 + token_embd spliced byte-for-byte from the Q6_K_P source (no requantization).
# PRE-REGISTERED: (A) K2q6 speed PASS at cache c if no OOM / re-allocation failure in any pass AND specbench mean >= K2 c26 - K2
# spread AND 9.3k decode median >= K2 c22 median - K2 c22 spread. (B) K2q6e WINS if KLD < K2q6 by > 3% and same-top not lower
# by > 0.5 pt (speed identical by construction: the embedding lookup is 1 row per token).
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
V2=/ai/src/llama.cpp-v2/build75; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; Q6=$M/$N-Q6_K_P.gguf; K2Q6=$M/$N-K2q6-denseQ4K.gguf
K2Q6E=/mnt/md0/models-cold/$N-K2q6e-denseQ4K-embdQ6K.gguf
HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf; VOCAB=$M/mtp-Qwen3.6-35B-A3B-vocab49k.bin
ENVS="GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_CUDA_FA_MMA_MAX_KV=4096 LLAMA_MTP_VOCAB_FILE=$VOCAB GGML_SCHED_MOE_PREFETCH=1"
echo "--- 0. BUILD K2q6e | $(date +%T)"
if [ ! -s $K2Q6E ]; then
  PYTHONPATH=/ai/src/llama.cpp-v2/gguf-py $PY gguf_splice.py $K2Q6 $Q6 $K2Q6E.tmp '^token_embd\.weight$' 2>&1 | sed 's/^/    /'
  [ ${PIPESTATUS[0]} -eq 0 ] && mv $K2Q6E.tmp $K2Q6E || { rm -f $K2Q6E.tmp; echo "K2Q6B_FAILED: splice"; exit 1; }
fi
echo "--- 1. KLD K2q6e | $(date +%T)"
COMMON="-f /ai/bench/kld/wiki.test.raw -c 2048 --chunks 6 -ngl 999 -ot exps=CPU -t 6 -fa on -b 2048 -ub 512"
env -u LD_LIBRARY_PATH GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 $V2/bin/llama-perplexity -m $K2Q6E $COMMON \
  --kl-divergence-base pfkld_q6.kld --kl-divergence > k2q6b_kld_e.log 2>&1 || { tail -3 k2q6b_kld_e.log; echo "K2Q6B_FAILED: kld"; exit 1; }
echo "  k2q6e: $(grep -E 'Mean +KLD|Same top p' k2q6b_kld_e.log | tr -s ' ' | tr '\n' ' ')"
echo "--- 2. SPEED specbench (n3): K2 c26 | K2q6 c20 | K2q6 c21 (2 rounds) | $(date +%T)"
export EDIT=1 LONG=1 GEN=300 OFFLOAD=32 BUILD=$V2
for r in a b; do for a in k2c26 q6c20 q6c21; do
  l=k2q6b_${a}_$r; F=$K2; c=26; case $a in q6c20) F=$K2Q6; c=20;; q6c21) F=$K2Q6; c=21;; esac
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  echo "##### $l | $(date +%T)"
  MODEL=$F env -u LD_LIBRARY_PATH $ENVS ./specbench.sh 999 $l -ot exps=CPU --moe-expert-cache $c -md $HEAD --spec-type draft-mtp --spec-draft-n-max 3 \
    -ub 128 -b 2048 -ubp 2048 2>&1 | grep -E "^$l|FOREIGN|DIED" | cut -c1-110 | sed 's/^/    /'
  echo "    $(grep -oE 'hit rate [0-9.]+%' server_$l.log | tail -1) $(grep -hoE 'could not re-allocate|out of memory' server_$l.log | head -1)"
done; done
echo "--- 3. 9.3k (n2): K2 c22 | K2q6 c18 | K2q6 c20 (2 rounds) | $(date +%T)"
for r in a b; do for a in k2c22 q6c18 q6c20; do
  l=k2q6b_pf_${a}_$r; F=$K2; c=22; case $a in q6c18) F=$K2Q6; c=18;; q6c20) F=$K2Q6; c=20;; esac
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $ENVS GGML_OP_OFFLOAD_MIN_BATCH=32 $V2/bin/llama-server -m $F -ngl 999 -fa on -c 12288 -t 6 --load-mode none --jinja \
    --parallel 1 --port 8099 --cache-ram 0 -ot exps=CPU --moe-expert-cache $c -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 \
    -ub 128 -b 4096 -ubp 2048 > server_$l.log 2>&1 &
  NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  echo -n "    "; python3 longpf.py $l 40000 2>&1 | tail -1 | tr -d '\n'; echo " | $(grep -hoE 'could not re-allocate|out of memory' server_$l.log | head -1)"
  kill $NP; wait $NP 2>/dev/null
done; done
$PY - <<'PY'
import json, os, re, statistics as st
g = lambda f: (lambda t: (float(re.search(r"Mean +KLD: +([\d.]+)", t).group(1)), float(re.search(r"Same top p: +([\d.]+)", t).group(1))))(open(f).read())
q, e = g("/ai/bench/k2q6_kld.log"), g("/ai/bench/k2q6b_kld_e.log"); rel = 100 * (e[0] / q[0] - 1)
print(f"  (B) KLD K2q6 {q[0]:.6f} top {q[1]:.3f} | K2q6e {e[0]:.6f} top {e[1]:.3f} -> {rel:+.2f}% -> {'WIN' if rel < -3 and e[1] >= q[1] - 0.5 else 'NO'}")
oom = lambda l: bool(re.search("could not re-allocate|out of memory", open(f"/ai/bench/server_{l}.log", errors="ignore").read())) if os.path.exists(f"/ai/bench/server_{l}.log") else True
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
ok = lambda a: all(os.path.exists(f"/ai/bench/runs/k2q6b_{a}_{r}.client.json") and not oom(f"k2q6b_{a}_{r}") for r in "ab")
S = {a: [L(f"k2q6b_{a}_{r}") for r in "ab"] for a in ("k2c26", "q6c20", "q6c21") if ok(a)}
kinds = list(S["k2c26"][0])
M = {a: st.mean(x[k]["decode_tps"] for x in S[a] for k in kinds if k in x) for a in S}
sp = st.mean(abs(S["k2c26"][0][k]["decode_tps"] - S["k2c26"][1][k]["decode_tps"]) for k in kinds)
print("  (A) specbench n3: " + " | ".join(f"{a} {M[a]:.2f} ({100 * (M[a] / M['k2c26'] - 1):+.1f}%) {'PASS' if M[a] >= M['k2c26'] - sp else 'FAIL'}" for a in M)
      + f" (K2 spread {sp:.2f}); OOM arms: {[a for a in ('q6c20', 'q6c21') if a not in S]}")
P = lambda a: [json.load(open(f"/ai/bench/runs/k2q6b_pf_{a}_{r}.longpf.json"))["decode_tps"] for r in "ab" if os.path.exists(f"/ai/bench/runs/k2q6b_pf_{a}_{r}.longpf.json") and not oom(f"k2q6b_pf_{a}_{r}")]
base = P("k2c22"); bsp = abs(base[0] - base[1]) if len(base) == 2 else 0
print("  (A) 9.3k decode n2: " + " | ".join(f"{a} {P(a)} {'PASS' if len(P(a)) == 2 and st.median(P(a)) >= st.median(base) - bsp else 'FAIL'}" for a in ("k2c22", "q6c18", "q6c20")))
PY
echo K2Q6B_DONE

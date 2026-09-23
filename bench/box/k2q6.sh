#!/bin/bash
# K2Q6 (follow-up to k2d, accuracy side): k2d requantized the dense roles FROM K2 (IQ2_S -> Q4_K) and its KLD got worse (+5.8%,
# double rounding). K2q6 = K2 with the same dense roles (attn_gate, attn_q, attn_k, attn_output, ssm_out, ssm_alpha, ssm_beta,
# shared expert) taken from the Q6_K_P source quantized to Q4_K (same llama-quantize type choice as k2d); experts and every other
# tensor byte-identical to K2 (gguf_splice.py). Speed should equal k2d (same dense types): decode parity at c22, prefill +2.8%.
# PRE-REGISTERED: K2q6 WINS (accuracy) if mean KLD vs Q6 (c 2048, 6 chunks) is lower than K2's by > 3% AND same-top not lower
# by > 0.5 pt, AND specbench n3 at c22 is not below K2 @ 26 by more than K2's spread (no OOM). Model default stays Andrei's call.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
V2=/ai/src/llama.cpp-v2/build75; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; Q6=$M/$N-Q6_K_P.gguf; K2Q6=$M/$N-K2q6-denseQ4K.gguf
DONOR=/mnt/md0/models-cold/tmp-k2q6-donor.gguf
HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf; VOCAB=$M/mtp-Qwen3.6-35B-A3B-vocab49k.bin
ENVS="GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_CUDA_FA_MMA_MAX_KV=4096 LLAMA_MTP_VOCAB_FILE=$VOCAB GGML_SCHED_MOE_PREFETCH=1"
RX='attn_gate|attn_q\.|attn_k\.|attn_output|ssm_out|ssm_alpha|ssm_beta|_shexp'
echo "--- 0. BUILD K2q6 | $(date +%T)"
if [ ! -s $K2Q6 ]; then
  if [ ! -s $DONOR ]; then
    # experts -> q4_0 (fastest to write; discarded by the splice)
    env -u LD_LIBRARY_PATH $V2/bin/llama-quantize --allow-requantize --tensor-type '_exps\.weight$=q4_0' $Q6 $DONOR Q4_K 6 > k2q6_quant.log 2>&1 \
      || { tail -5 k2q6_quant.log; rm -f $DONOR; echo "K2Q6_FAILED: quantize"; exit 1; }
  fi
  PYTHONPATH=/ai/src/llama.cpp-v2/gguf-py $PY gguf_splice.py $K2 $DONOR $K2Q6.tmp "$RX" 2>&1 | sed 's/^/    /'
  [ ${PIPESTATUS[0]} -eq 0 ] && mv $K2Q6.tmp $K2Q6 || { rm -f $K2Q6.tmp; echo "K2Q6_FAILED: splice"; exit 1; }
  rm -f $DONOR
fi
python3 gguf_types.py $K2Q6 2>&1 | grep -E "^ROLE|^TOTAL|^EXPERTS|^REST" | sed 's/^/    /'
echo "    dense types vs K2d:"; diff <(python3 gguf_types.py $K2Q6 | grep ^ROLE) <(python3 gguf_types.py /ai/models/$N-K2d-denseQ4K.gguf | grep ^ROLE) > /dev/null && echo "    identical role/type table" || echo "    DIFFERENT role/type table"
echo "--- 1. KLD | $(date +%T)"
COMMON="-f /ai/bench/kld/wiki.test.raw -c 2048 --chunks 6 -ngl 999 -ot exps=CPU -t 6 -fa on -b 2048 -ub 512"
env -u LD_LIBRARY_PATH GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 $V2/bin/llama-perplexity -m $K2Q6 $COMMON \
  --kl-divergence-base pfkld_q6.kld --kl-divergence > k2q6_kld.log 2>&1 || { tail -3 k2q6_kld.log; echo "K2Q6_FAILED: kld"; exit 1; }
echo "  k2q6: $(grep -E 'Mean +KLD|Same top p' k2q6_kld.log | tr -s ' ' | tr '\n' ' ')"
echo "--- 2. SPEED specbench (n3): K2 c26 | K2q6 c22 (2 rounds) | $(date +%T)"
export EDIT=1 LONG=1 GEN=300 OFFLOAD=32 BUILD=$V2
for r in a b; do for a in k2c26 k2q6c22; do
  l=k2q6_${a}_$r; F=$K2; c=26; [ $a = k2q6c22 ] && { F=$K2Q6; c=22; }
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  echo "##### $l | $(date +%T)"
  MODEL=$F env -u LD_LIBRARY_PATH $ENVS ./specbench.sh 999 $l -ot exps=CPU --moe-expert-cache $c -md $HEAD --spec-type draft-mtp --spec-draft-n-max 3 \
    -ub 128 -b 2048 -ubp 2048 2>&1 | grep -E "^$l|FOREIGN|DIED" | cut -c1-110 | sed 's/^/    /'
  echo "    $(grep -oE 'hit rate [0-9.]+%' server_$l.log | tail -1) $(grep -hoE 'could not re-allocate|out of memory' server_$l.log | head -1)"
done; done
$PY - <<'PY'
import json, os, re
g = lambda f: (lambda t: (float(re.search(r"Mean +KLD: +([\d.]+)", t).group(1)), float(re.search(r"Same top p: +([\d.]+)", t).group(1))))(open(f).read())
k, d = g("/ai/bench/k2d_kld_k2.log"), g("/ai/bench/k2q6_kld.log"); rel = 100 * (d[0] / k[0] - 1)
kok = rel < -3.0 and d[1] >= k[1] - 0.5
print(f"  KLD K2 {k[0]:.6f} top {k[1]:.3f} | K2q6 {d[0]:.6f} top {d[1]:.3f} -> {rel:+.2f}% -> {'PASS' if kok else 'FAIL'}")
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
A = [a for a in ("k2c26", "k2q6c22") if all(os.path.exists(f"/ai/bench/runs/k2q6_{a}_{r}.client.json") for r in "ab")]
S = {a: [L(f"k2q6_{a}_{r}") for r in "ab"] for a in A}
kinds = list(S["k2c26"][0]); m = lambda a, kk: sum(x[kk]["decode_tps"] for x in S[a]) / 2
M = {a: sum(m(a, kk) for kk in kinds) / len(kinds) for a in A}
sp = sum(abs(S["k2c26"][0][kk]["decode_tps"] - S["k2c26"][1][kk]["decode_tps"]) for kk in kinds) / len(kinds)
sok = "k2q6c22" in M and M["k2q6c22"] >= M["k2c26"] - sp
print("  specbench n3: " + " | ".join(f"{a} {M[a]:.2f} ({100 * (M[a] / M['k2c26'] - 1):+.1f}%)" for a in A) + f" (K2 spread {sp:.2f}) -> {'PASS' if sok else 'FAIL'}")
print(f"  K2Q6_VERDICT {'WIN' if kok and sok else 'NO'}")
PY
echo K2Q6_DONE

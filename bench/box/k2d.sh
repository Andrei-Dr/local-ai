#!/bin/bash
# K2D (lead 2, model-file side): the dense decode mat-vecs are ALU-bound IQ quants. mv1 (18:30, m=4096 k=14336): IQ2_S 46 GB/s at
# n=3 (398 us) vs Q4_K 140 GB/s (235 us) — IQ2_S reads 44% fewer bytes and takes 1.7x longer; the dense IQ2_S / IQ3_S tensors
# (attn_gate, attn_q, attn_k, attn_output, ssm_out, ssm_alpha, ssm_beta, shared expert) cost ~2.5 ms/token.
# Build K2d = K2 with ONLY those roles requantized to Q4_K; every other tensor keeps its K2 type and is copied byte-for-byte by
# llama-quantize (same type -> no requantization). Source = K2 itself (IQ2_S -> Q4_K: quality ~ K2; a Q6-sourced dense is the
# follow-up if speed wins). VRAM: dense grows ~2.5 -> 4.5 bpw (~+200 MB on CUDA0) -> fewer cache slots fit.
# PRE-REGISTERED: (1) KLD vs Q6 (c 2048, 6 chunks): K2d mean KLD within 1% of K2 (or lower), same-top within 0.5 points;
# (2) K2d WINS if at a cache size that fits (no OOM / re-allocation failure) its specbench 4-prompt mean decode beats K2 @ 26 by
# more than K2's spread, and its 9.3k decode is not below K2 @ 22 by more than that arm's spread. STABLE d0fd493, full serving env.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
V2=/ai/src/llama.cpp-v2/build75; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; K2D=$M/$N-K2d-denseQ4K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf; VOCAB=$M/mtp-Qwen3.6-35B-A3B-vocab49k.bin
ENVS="GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_CUDA_FA_MMA_MAX_KV=4096 LLAMA_MTP_VOCAB_FILE=$VOCAB GGML_SCHED_MOE_PREFETCH=1"
echo "--- 0. BUILD K2d | $(date +%T)"
if [ ! -s $K2D ]; then
  cmake --build $V2 -j6 --target llama-quantize > k2d_build.log 2>&1 || { grep error k2d_build.log | head; echo "K2D_FAILED: build"; exit 1; }
  python3 gguf_types.py $K2 --emit-quantize k2d_all.txt --base q4_K > /dev/null || { echo "K2D_FAILED: emit"; exit 1; }
  grep -vE 'attn_gate|attn_q\\\.|attn_k\\\.|attn_output|ssm_out|ssm_alpha|ssm_beta|_shexp' k2d_all.txt > k2d_keep.txt
  echo "    overrides: all $(wc -l < k2d_all.txt), kept $(wc -l < k2d_keep.txt) (the rest fall to Q4_K)"
  env -u LD_LIBRARY_PATH $V2/bin/llama-quantize --allow-requantize $(cat k2d_keep.txt) $K2 $K2D Q4_K 6 > k2d_quant.log 2>&1 \
    || { tail -5 k2d_quant.log; rm -f $K2D; echo "K2D_FAILED: quantize"; exit 1; }
fi
python3 gguf_types.py $K2D 2>&1 | grep -E "attn_gate|attn_q.weight|ssm_out|shexp|_exps|^TOTAL|^REST" | sed 's/^/    /'
echo "--- 1. KLD | $(date +%T)"
COMMON="-f /ai/bench/kld/wiki.test.raw -c 2048 --chunks 6 -ngl 999 -ot exps=CPU -t 6 -fa on -b 2048 -ub 512"
for a in k2 k2d; do
  F=$K2; [ $a = k2d ] && F=$K2D
  env -u LD_LIBRARY_PATH GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 $V2/bin/llama-perplexity -m $F $COMMON \
    --kl-divergence-base pfkld_q6.kld --kl-divergence > k2d_kld_$a.log 2>&1 || { tail -3 k2d_kld_$a.log; echo "K2D_FAILED: kld $a"; exit 1; }
  echo "  $a: $(grep -E 'Mean +KLD|Same top p' k2d_kld_$a.log | tr -s ' ' | tr '\n' ' ')"
done
echo "--- 2. SPEED specbench (n3): K2 c26 | K2d c22 | K2d c24 (2 rounds) | $(date +%T)"
export EDIT=1 LONG=1 GEN=300 OFFLOAD=32 BUILD=$V2
for r in a b; do for a in k2c26 k2dc22 k2dc24; do
  l=k2d_${a}_$r; F=$K2; c=26; case $a in k2dc22) F=$K2D; c=22;; k2dc24) F=$K2D; c=24;; esac
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  echo "##### $l | $(date +%T)"
  MODEL=$F env -u LD_LIBRARY_PATH $ENVS ./specbench.sh 999 $l -ot exps=CPU --moe-expert-cache $c -md $HEAD --spec-type draft-mtp --spec-draft-n-max 3 \
    -ub 128 -b 2048 -ubp 2048 2>&1 | grep -E "^$l|FOREIGN|DIED" | cut -c1-110 | sed 's/^/    /'
  echo "    $(grep -oE 'hit rate [0-9.]+%' server_$l.log | tail -1) $(grep -oE 'CUDA0 model buffer size = +[0-9.]+ MiB' server_$l.log | head -1) $(grep -hoE 'could not re-allocate|out of memory' server_$l.log | head -1)"
done; done
echo "--- 3. 9.3k (n2): K2 c22 | K2d c18 | K2d c20 (2 rounds) | $(date +%T)"
for r in a b; do for a in k2c22 k2dc18 k2dc20; do
  l=k2d_pf_${a}_$r; F=$K2; c=22; case $a in k2dc18) F=$K2D; c=18;; k2dc20) F=$K2D; c=20;; esac
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $ENVS GGML_OP_OFFLOAD_MIN_BATCH=32 $V2/bin/llama-server -m $F -ngl 999 -fa on -c 12288 -t 6 --load-mode none --jinja \
    --parallel 1 --port 8099 --cache-ram 0 -ot exps=CPU --moe-expert-cache $c -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 \
    -ub 128 -b 4096 -ubp 2048 > server_$l.log 2>&1 &
  NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  echo -n "    "; python3 longpf.py $l 40000 2>&1 | tr -d '\n'; echo " | $(grep -hoE 'could not re-allocate|out of memory' server_$l.log | head -1)"
  kill $NP; wait $NP 2>/dev/null
done; done
$PY - <<'PY'
import json, os, re
g = lambda a: (lambda t: (float(re.search(r"Mean +KLD: +([\d.]+)", t).group(1)), float(re.search(r"Same top p: +([\d.]+)", t).group(1))))(open(f"/ai/bench/k2d_kld_{a}.log").read())
k, d = g("k2"), g("k2d"); rel = 100 * (d[0] / k[0] - 1)
print(f"  KLD K2 {k[0]:.6f} top {k[1]:.3f} | K2d {d[0]:.6f} top {d[1]:.3f} -> {rel:+.2f}% -> {'PASS' if (rel <= 1.0) and d[1] >= k[1] - 0.5 else 'FAIL'}")
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
A = [a for a in ("k2c26", "k2dc22", "k2dc24") if all(os.path.exists(f"/ai/bench/runs/k2d_{a}_{r}.client.json") for r in "ab")]
S = {a: [L(f"k2d_{a}_{r}") for r in "ab"] for a in A}
kinds = list(S["k2c26"][0]); m = lambda a, kk: sum(x[kk]["decode_tps"] for x in S[a]) / 2
M = {a: sum(m(a, kk) for kk in kinds) / len(kinds) for a in A}
sp = sum(abs(S["k2c26"][0][kk]["decode_tps"] - S["k2c26"][1][kk]["decode_tps"]) for kk in kinds) / len(kinds)
print("  specbench n3: " + " | ".join(f"{a} {M[a]:.2f} ({100 * (M[a] / M['k2c26'] - 1):+.1f}%)" for a in A) + f" (K2 spread {sp:.2f})")
P = lambda a: [json.load(open(f"/ai/bench/runs/k2d_pf_{a}_{r}.longpf.json"))["decode_tps"] for r in "ab" if os.path.exists(f"/ai/bench/runs/k2d_pf_{a}_{r}.longpf.json")]
print("  9.3k decode n2: " + " | ".join(f"{a} {P(a)}" for a in ("k2c22", "k2dc18", "k2dc20")))
PY
echo K2D_DONE

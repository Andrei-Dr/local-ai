#!/bin/bash
# PROMO3: lead D into STABLE — GGML_CUDA_FA_MMA_MAX_KV (FlashAttention MMA only below a KV length; tile kernel above).
# fakv1 (11:31): 9.3k decode +6.2 / +6.8 / +7.4% at MAX_KV 2048 / 4096 / 8192 (spread 0.29); specbench within spread; tile kernel
# KLD non-inferior (pmux5). New STABLE commit 1be3f5f1e = 8dca9af + fa-kv (cherry-pick of 3ea44a4), pushed as tu116-served-next.
# Steps: 1 identity baseline with the CURRENT STABLE binary | 2 ff the v2 worktree to 1be3f5f1e, rebuild | 3 identity with the env
# UNSET vs step 1 -> must be IDENTICAL (the gate is inert unless set) | 4 speed with GGML_CUDA_FA_MMA_MAX_KV=4096: specbench +
# 9,279-token prompt, env unset vs set, 2 interleaved rounds.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
V2=/ai/src/llama.cpp-v2; NEW=$V2/build75; SHA=1be3f5f1e; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
SERVE="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
[ -z "$(git -C $V2 status --porcelain --untracked-files=no)" ] || { echo "PROMO3_REFUSED: $V2 has local changes"; exit 1; }
if [ "$(git -C $V2 rev-parse --short=9 HEAD)" != "$SHA" ]; then
  echo "--- 1. IDENTITY BASELINE (current STABLE $(git -C $V2 rev-parse --short HEAD)) | $(date +%T)"
  env -u LD_LIBRARY_PATH LLAMA_MOE_CACHE_SYNC=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 promo3_id_old $SERVE > /dev/null 2>&1
  echo "--- 2. FF + REBUILD | $(date +%T)"
  git -C $V2 merge -q --ff-only $SHA || { echo "PROMO3_REFUSED: cannot fast-forward $V2 to $SHA"; exit 1; }
fi
cmake --build $NEW -j6 --target llama-server llama-perplexity llama-cli llama-bench test-backend-ops > promo3_build.log 2>&1 || { grep error promo3_build.log | head; echo "PROMO3_FAILED: build"; exit 1; }
strings $NEW/bin/libggml-cuda.so* | grep -q GGML_CUDA_FA_MMA_MAX_KV || { echo "PROMO3_FAILED: no GGML_CUDA_FA_MMA_MAX_KV"; exit 1; }
echo "    built $(git -C $V2 rev-parse --short HEAD) | $(date +%T)"
echo "--- 3. IDENTITY (env unset) vs baseline | $(date +%T)"
env -u LD_LIBRARY_PATH LLAMA_MOE_CACHE_SYNC=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 promo3_id_new $SERVE > /dev/null 2>&1
[ -s runs/promo3_id_old.client.json ] || { echo "PROMO3_FAILED: no baseline (worktree was already at $SHA)"; exit 1; }
$PY textdiff.py runs/promo3_id_old.client.json runs/promo3_id_new.client.json | sed 's/^/    /'; id=${PIPESTATUS[0]}
[ "$id" -eq 0 ] || { echo "PROMO3_FAILED: not identical with the env unset"; exit 1; }
echo "--- 4. SPEED: unset vs GGML_CUDA_FA_MMA_MAX_KV=4096 | $(date +%T)"
for r in a b; do for a in unset kv; do
  l=promo3_${a}_$r; E=""; [ $a = kv ] && E="GGML_CUDA_FA_MMA_MAX_KV=4096"
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $E GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $SERVE 2>&1 | grep -E "^$l|FOREIGN" | cut -c1-100 | sed 's/^/    /'
done; done
for r in a b; do for a in unset kv; do
  l=promo3_pf_${a}_$r; E=""; [ $a = kv ] && E="GGML_CUDA_FA_MMA_MAX_KV=4096"
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $E GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 $NEW/bin/llama-server -m $K2 -ngl 999 -fa on -c 12288 -t 6 \
    --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 -ot exps=CPU --moe-expert-cache 22 -md $HEAD \
    --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 4096 -ubp 2048 > server_$l.log 2>&1 &
  NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  echo -n "    [$a $r] "; python3 longpf.py $l 40000 2>&1
  kill $NP; wait $NP 2>/dev/null
done; done
$PY - <<'PY'
import json
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
S = {a: [L(f"promo3_{a}_{r}") for r in "ab"] for a in ("unset", "kv")}
kinds = list(S["unset"][0]); m = lambda a, k: sum(x[k]["decode_tps"] for x in S[a]) / 2
for k in kinds:
    print(f"  {k:6s} decode unset {m('unset', k):6.2f} | kv4096 {m('kv', k):6.2f} ({100 * (m('kv', k) / m('unset', k) - 1):+5.1f}%)")
P = lambda a: [json.load(open(f"/ai/bench/runs/promo3_pf_{a}_{r}.longpf.json"))["decode_tps"] for r in "ab"]
u, k = P("unset"), P("kv")
print(f"  9.3k decode unset {u[0]:.2f} {u[1]:.2f} | kv4096 {k[0]:.2f} {k[1]:.2f} -> {100 * (sum(k) / sum(u) - 1):+.1f}%")
PY
echo PROMO3_DONE

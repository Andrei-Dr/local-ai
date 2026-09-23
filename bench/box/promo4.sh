#!/bin/bash
# PROMO4: lead B into STABLE — FR-Spec draft vocabulary for the MTP head (LLAMA_MTP_VOCAB_FILE). fr2 (12:34): 49k-token list from the
# model's own eval answers + eval prompts + prose/wiki/code/stdlib: mean decode +6.0% (spread 1.43), acceptance 0.837 vs 0.832 (no
# drop); the target verifies against the full vocabulary. New STABLE commit c75018be7 = 1be3f5f + FR-Spec (ac498ad) + dry-run fix
# (c75018b), pushed as tu116-served-next. Serving artifact: /ai/models/mtp-Qwen3.6-35B-A3B-vocab49k.bin (= fr2_vocab_49152.bin).
# Steps: 1 identity baseline (current STABLE) | 2 ff v2 to c75018be7, rebuild | 3 identity with LLAMA_MTP_VOCAB_FILE unset -> IDENTICAL |
# 4 speed: unset vs set (specbench + 9,279-token prompt), 2 interleaved rounds; both arms with GGML_CUDA_FA_MMA_MAX_KV=4096 (STABLE).
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
VOCAB=/ai/models/mtp-Qwen3.6-35B-A3B-vocab49k.bin
[ -s $VOCAB ] || cp /ai/bench/fr2_vocab_49152.bin $VOCAB || { echo "PROMO4_FAILED: vocab artifact"; exit 1; }
V2=/ai/src/llama.cpp-v2; NEW=$V2/build75; SHA=c75018be7; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
SERVE="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
[ -z "$(git -C $V2 status --porcelain --untracked-files=no)" ] || { echo "PROMO4_REFUSED: $V2 has local changes"; exit 1; }
if [ "$(git -C $V2 rev-parse --short=9 HEAD)" != "$SHA" ]; then
  echo "--- 1. IDENTITY BASELINE (current STABLE $(git -C $V2 rev-parse --short HEAD)) | $(date +%T)"
  env -u LD_LIBRARY_PATH LLAMA_MOE_CACHE_SYNC=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_CUDA_FA_MMA_MAX_KV=4096 ./specbench.sh 999 promo4_id_old $SERVE > /dev/null 2>&1
  echo "--- 2. FF + REBUILD | $(date +%T)"
  git -C $V2 merge -q --ff-only $SHA || { echo "PROMO4_REFUSED: cannot fast-forward $V2 to $SHA"; exit 1; }
fi
cmake --build $NEW -j6 --target llama-server llama-perplexity llama-cli llama-bench test-backend-ops > promo4_build.log 2>&1 || { grep error promo4_build.log | head; echo "PROMO4_FAILED: build"; exit 1; }
strings $NEW/bin/libllama.so* | grep -q LLAMA_MTP_VOCAB_FILE || { echo "PROMO4_FAILED: no LLAMA_MTP_VOCAB_FILE"; exit 1; }
echo "    built $(git -C $V2 rev-parse --short HEAD) | $(date +%T)"
echo "--- 3. IDENTITY (env unset) vs baseline | $(date +%T)"
env -u LD_LIBRARY_PATH LLAMA_MOE_CACHE_SYNC=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_CUDA_FA_MMA_MAX_KV=4096 ./specbench.sh 999 promo4_id_new $SERVE > /dev/null 2>&1
[ -s runs/promo4_id_old.client.json ] || { echo "PROMO4_FAILED: no baseline (worktree was already at $SHA)"; exit 1; }
$PY textdiff.py runs/promo4_id_old.client.json runs/promo4_id_new.client.json | sed 's/^/    /'; id=${PIPESTATUS[0]}
[ "$id" -eq 0 ] || { echo "PROMO4_FAILED: not identical with the env unset"; exit 1; }
echo "--- 4. SPEED: unset vs LLAMA_MTP_VOCAB_FILE=vocab49k | $(date +%T)"
for r in a b; do for a in unset kv; do
  l=promo4_${a}_$r; E=""; [ $a = kv ] && E="LLAMA_MTP_VOCAB_FILE=$VOCAB"
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $E GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_CUDA_FA_MMA_MAX_KV=4096 ./specbench.sh 999 $l $SERVE 2>&1 | grep -E "^$l|FOREIGN" | cut -c1-100 | sed 's/^/    /'
done; done
for r in a b; do for a in unset kv; do
  l=promo4_pf_${a}_$r; E=""; [ $a = kv ] && E="LLAMA_MTP_VOCAB_FILE=$VOCAB"
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $E GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_CUDA_FA_MMA_MAX_KV=4096 GGML_OP_OFFLOAD_MIN_BATCH=32 $NEW/bin/llama-server -m $K2 -ngl 999 -fa on -c 12288 -t 6 \
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
S = {a: [L(f"promo4_{a}_{r}") for r in "ab"] for a in ("unset", "kv")}
kinds = list(S["unset"][0]); m = lambda a, k: sum(x[k]["decode_tps"] for x in S[a]) / 2
for k in kinds:
    print(f"  {k:6s} decode unset {m('unset', k):6.2f} | vocab49k {m('kv', k):6.2f} ({100 * (m('kv', k) / m('unset', k) - 1):+5.1f}%)")
P = lambda a: [json.load(open(f"/ai/bench/runs/promo4_pf_{a}_{r}.longpf.json"))["decode_tps"] for r in "ab"]
u, k = P("unset"), P("kv")
print(f"  9.3k decode unset {u[0]:.2f} {u[1]:.2f} | vocab49k {k[0]:.2f} {k[1]:.2f} -> {100 * (sum(k) / sum(u) - 1):+.1f}%")
PY
echo PROMO4_DONE

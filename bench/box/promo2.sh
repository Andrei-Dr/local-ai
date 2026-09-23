#!/bin/bash
# PROMO2: 0018 default-on in STABLE (tu116-served 8dca9af). prof18 (nsys, 2 captures per arm): host overhead per MoE layer
# (graph launch + stream syncs + copy calls + gaps) 89.5 / 79.4 us -> 62.7 / 71.0 us, stream syncs per launch 5.2 -> 4.6, device
# phase unchanged; ~1.8% of a layer round, below the specbench noise floor (CPU phase alone swings +-25 us) — which is why tu1 /
# tu2 read +3.4 / -2.0%. Exact by construction (tu1 IDENTICAL x4). This job: incremental rebuild of the STABLE worktree, then
# identity new default vs GGML_SCHED_COPY_SYNC=1 (old behavior) under LLAMA_MOE_CACHE_SYNC=1, serving config. Must be IDENTICAL.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
V2=/ai/src/llama.cpp-v2; NEW=$V2/build75; SHA=8dca9aff4
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf; PY=/ai/.venv/bin/python
[ "$(git -C $V2 rev-parse --short=9 HEAD)" = "$SHA" ] || { echo "PROMO2_REFUSED: $V2 not at $SHA"; exit 1; }
cmake --build $NEW -j6 --target llama-server llama-perplexity llama-cli llama-bench test-backend-ops > promo2_build.log 2>&1 \
  || { grep error promo2_build.log | head; echo "PROMO2_FAILED: build"; exit 1; }
strings $NEW/bin/libggml-base.so* | grep -q GGML_SCHED_COPY_SYNC || { echo "PROMO2_FAILED: libggml-base lacks GGML_SCHED_COPY_SYNC"; exit 1; }
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
SERVE="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
echo "--- IDENTITY (LLAMA_MOE_CACHE_SYNC=1, serving config): default (no copy sync) vs GGML_SCHED_COPY_SYNC=1 | $(date +%T)"
env -u LD_LIBRARY_PATH LLAMA_MOE_CACHE_SYNC=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_SCHED_COPY_SYNC=1 ./specbench.sh 999 promo2_id_sync $SERVE 2>&1
env -u LD_LIBRARY_PATH LLAMA_MOE_CACHE_SYNC=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 promo2_id_nosync $SERVE 2>&1
$PY textdiff.py runs/promo2_id_sync.client.json runs/promo2_id_nosync.client.json | sed 's/^/    /'; id=${PIPESTATUS[0]}
[ "$id" -eq 0 ] || { echo "PROMO2_FAILED: not identical"; exit 1; }
echo PROMO2_DONE

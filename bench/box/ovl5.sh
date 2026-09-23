#!/bin/bash
# OVL5: which planned splits change the main model's output? ovl4 (09:08): WITHOUT the MTP draft, off vs plan (mode 2) still
# DIVERGES (code at char 15, reason 205, long 318; edit IDENTICAL; off vs off IDENTICAL) -> the main model itself. (The
# draft-on-GPU arm was void: -otd exps=CUDA0 left the draft's experts in CUDA_Host.) ovl3's tensor dump of one prefill ubatch
# was IDENTICAL off vs plan, so the planned graph that matters is (inferred) not the prompt ubatch.
# Build: t2 at 4431c75 = 1ea8e04 + GGML_SCHED_MOE_PREFETCH_LOG=1 (log each planned pair: graph size, split, first node, ids)
# + GGML_SCHED_MOE_PREFETCH_MIN_IDS=n (plan only splits whose router output has >= n ids). No-draft serving config, identity mode.
#   plan_log   = full plan, logged: WHICH graphs get planned splits (ids count = n_tokens x 8)
#   min2048    = plan only batches >= 256 tokens;   min256 = only >= 32 tokens
#   off vs min2048 / min256 IDENTICAL while off vs plan DIVERGES => the culprit is a planned split in the smaller batches
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; SHA=4431c75
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; PY=/ai/.venv/bin/python
if [ "$(git -C $T2 rev-parse --short=7 HEAD)" != "$SHA" ]; then git -C $T2 merge -q --ff-only $SHA || { echo "OVL5_REFUSED: cannot fast-forward $T2"; exit 1; }; fi
[ -z "$(git -C $T2 status --porcelain --untracked-files=no)" ] || { echo "OVL5_REFUSED: $T2 has local changes"; exit 1; }
cmake --build $NEW -j6 --target llama-server > ovl5_build.log 2>&1 || { grep error ovl5_build.log | head; echo "OVL5_FAILED: build"; exit 1; }
strings $NEW/bin/libggml-base.so* | grep -q GGML_SCHED_MOE_PREFETCH_MIN_IDS || { echo "OVL5_FAILED: diag not in build"; exit 1; }
echo "    built $(git -C $T2 rev-parse --short HEAD) | $(date +%T)"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
NOMTP="-ot exps=CPU --moe-expert-cache 26 -ub 128 -b 2048 -ubp 2048"
arm() { local l=$1 envs=$2; echo "##### $l | env: ${envs:-none} | $(date +%T)"
  env -u LD_LIBRARY_PATH $envs LLAMA_MOE_CACHE_SYNC=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $NOMTP > /dev/null 2>&1
  grep -hE "out of memory|failed to allocate|CUDA error" server_$l.log | head -3 | sed 's/^/    /'; }
arm ovl5_off      ""
arm ovl5_plan_log "GGML_SCHED_MOE_PREFETCH=2 GGML_SCHED_MOE_PREFETCH_LOG=1"
arm ovl5_min2048  "GGML_SCHED_MOE_PREFETCH=2 GGML_SCHED_MOE_PREFETCH_MIN_IDS=2048"
arm ovl5_min256   "GGML_SCHED_MOE_PREFETCH=2 GGML_SCHED_MOE_PREFETCH_MIN_IDS=256"
echo "--- planned splits (plan_log): count by ids / first node of the target / graph size"
grep -o "prefetch plan: .*" server_ovl5_plan_log.log | sed -E 's/split [0-9]+ \(([a-z_]+)-[0-9]+, ids ([0-9-]+), backend ([A-Za-z0-9]+)\) <- split [0-9]+ \(first node ([a-zA-Z_]+)-?[0-9]*\)/target \1 ids \2 \3 <- prev first \4/' \
  | sed -E 's/graph ([0-9]+) nodes ([0-9]+) splits/graph \1n \2s/' | sort | uniq -c | sort -rn | head -25 | sed 's/^/    /'
echo "    total logged pairs: $(grep -c 'prefetch plan:' server_ovl5_plan_log.log)"
d() { echo "  $1 vs $2:"; $PY textdiff.py runs/$1.client.json runs/$2.client.json | sed 's/^/    /'; }
d ovl5_off ovl5_plan_log
d ovl5_off ovl5_min2048
d ovl5_off ovl5_min256
echo OVL5_DONE

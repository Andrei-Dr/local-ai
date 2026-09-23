#!/bin/bash
# OVL6: do CUDA graphs carry the plan's divergence? ovl5 (09:14): only PROMPT graphs get planned splits (routed ids 304 / 448 / 1024 /
# 1960 / 16384; chain gate -> up -> down -> next gate, all CUDA0); decode graphs never do. MIN_IDS=2048 (plan only >= 256-token
# batches) makes code and reason IDENTICAL, long still DIVERGES. Yet ovl3's llama-eval-callback dump of the reason prompt's prefill
# (callback mode = node-by-node compute) was IDENTICAL off vs plan. Hypothesis (inferred): the CUDA graph path (captured/updated split
# graphs on Turing) is what differs. Arms (no draft, identity mode, t2 @ 4431c75):
#   off_ng vs plan_ng  (GGML_CUDA_DISABLE_GRAPHS=1 in both)  IDENTICAL => CUDA graphs carry it
#   off vs off_ng                                            baseline invariance to graphs (expected IDENTICAL)
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; SHA=4431c75
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; PY=/ai/.venv/bin/python
[ "$(git -C $T2 rev-parse --short=7 HEAD)" = "$SHA" ] || { echo "OVL6_REFUSED: $T2 not at $SHA"; exit 1; }
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
NOMTP="-ot exps=CPU --moe-expert-cache 26 -ub 128 -b 2048 -ubp 2048"
arm() { local l=$1 envs=$2; echo "##### $l | env: ${envs:-none} | $(date +%T)"
  env -u LD_LIBRARY_PATH $envs LLAMA_MOE_CACHE_SYNC=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $NOMTP > /dev/null 2>&1
  grep -hE "out of memory|failed to allocate|CUDA error" server_$l.log | head -3 | sed 's/^/    /'; }
arm ovl6_off_ng  "GGML_CUDA_DISABLE_GRAPHS=1"
arm ovl6_plan_ng "GGML_CUDA_DISABLE_GRAPHS=1 GGML_SCHED_MOE_PREFETCH=2"
d() { echo "  $1 vs $2:"; $PY textdiff.py runs/$1.client.json runs/$2.client.json | sed 's/^/    /'; }
d ovl6_off_ng ovl6_plan_ng
d ovl5_off ovl6_off_ng
echo OVL6_DONE

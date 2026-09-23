#!/bin/bash
# OVL11: does a CUDA op fusion carry the plan's layout change into the numbers? ovl10 (09:48): warm-up off still DIVERGES; the
# observer trace shows the prompt pass's ROUTING differs from layer 8 on (layers 0-7 identical), in the server, while callback
# mode (no fusion) was identical (ovl8/ovl9). Suspect: the MoE weighted-reduction fusion (graph_optimize registers alloc deps for
# experts/weights until the fused node) interacting with the hoisted copies in the down split. Arms: t2 @ 20c1581, no draft,
# identity mode, GGML_CUDA_DISABLE_FUSION=1 in both: off_nf vs plan_nf IDENTICAL => a fusion path is layout-sensitive.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf
[ "$(git -C $T2 rev-parse --short=7 HEAD)" = "20c1581" ] || { echo "OVL11_REFUSED: $T2 not at 20c1581"; exit 1; }
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
NOMTP="-ot exps=CPU --moe-expert-cache 26 -ub 128 -b 2048 -ubp 2048"
arm() { local l=$1 envs=$2; echo "##### $l | env: ${envs:-none} | $(date +%T)"
  env -u LD_LIBRARY_PATH $envs LLAMA_MOE_CACHE_SYNC=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $NOMTP > /dev/null 2>&1
  echo "    $(grep -oE 'prompt warm-up: [0-9]+ uploads' server_$l.log | head -1)"; }
arm ovl11_off_nf  "GGML_CUDA_DISABLE_FUSION=1"
arm ovl11_plan_nf "GGML_CUDA_DISABLE_FUSION=1 GGML_SCHED_MOE_PREFETCH=2"
echo "  off_nf vs plan_nf:"; $PY textdiff.py runs/ovl11_off_nf.client.json runs/ovl11_plan_nf.client.json | sed 's/^/    /'
echo OVL11_DONE

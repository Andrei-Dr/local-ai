#!/bin/bash
# OVL4: main model or MTP draft context? ovl3 (09:00): stale expert bytes are NOT the mechanism (off_z vs plan_z still DIVERGE;
# fill 0x00 vs 0xFF IDENTICAL; plain vs zero-filled IDENTICAL -> the served baseline never reads unselected experts); a full
# per-tensor dump of the main model's prefill of the reason prompt is IDENTICAL off vs plan (2,747 tensors). Draft acceptance
# already differs on the first prompt (off 175/204, plan 190/216). The MTP head's experts are host-offloaded too (272.81 MiB
# CUDA_Host), so its context also gets planned splits (its prompt pass runs >= 32 tokens on the GPU).
# Build: t2 at 1ea8e04 (unchanged). Identity arms (LLAMA_MOE_CACHE_SYNC=1), mode 2 = plan only:
#   A  no draft (-md dropped):                  nomtp_off vs nomtp_plan   IDENTICAL => the main model is clean
#   B  draft experts on the GPU (-otd exps=CUDA0, cache 24 in both arms for VRAM): gpud_off vs gpud_plan
#      IDENTICAL => the planned splits of the DRAFT context are the cause; DIVERGES => the main model under speculation
# plus nomtp_off_b (determinism floor without speculation).
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; SHA=1ea8e04ce2af6b8686fd3ef65efa2f0c11bbc53a
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf; PY=/ai/.venv/bin/python
[ "$(git -C $T2 rev-parse HEAD)" = "$SHA" ] || { echo "OVL4_REFUSED: $T2 not at $SHA"; exit 1; }
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
NOMTP="-ot exps=CPU --moe-expert-cache 26 -ub 128 -b 2048 -ubp 2048"
GPUD="-ot exps=CPU --moe-expert-cache 24 -md $HEAD -otd exps=CUDA0 --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
arm() { local l=$1 envs=$2; shift 2; echo "##### $l | env: ${envs:-none} | $(date +%T)"
  env -u LD_LIBRARY_PATH $envs LLAMA_MOE_CACHE_SYNC=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l "$@" > /dev/null 2>&1
  grep -hE "out of memory|failed to allocate|CUDA error|model buffer size" server_$l.log | sed 's/^/    /'
  echo "    $(grep -o 'draft acceptance = [0-9.]* ( *[0-9]* accepted / *[0-9]* generated)' server_$l.log | head -1)"; }
arm ovl4_nomtp_off   ""                          $NOMTP
arm ovl4_nomtp_plan  "GGML_SCHED_MOE_PREFETCH=2" $NOMTP
arm ovl4_nomtp_off_b ""                          $NOMTP
arm ovl4_gpud_off    ""                          $GPUD
arm ovl4_gpud_plan   "GGML_SCHED_MOE_PREFETCH=2" $GPUD
d() { echo "  $1 vs $2:"; $PY textdiff.py runs/$1.client.json runs/$2.client.json | sed 's/^/    /'; }
d ovl4_nomtp_off ovl4_nomtp_off_b
d ovl4_nomtp_off ovl4_nomtp_plan
d ovl4_gpud_off  ovl4_gpud_plan
echo OVL4_DONE

#!/bin/bash
# OVL7: bisect the planned-split divergence by target type, and re-check the "identical prefill" dump with proof the plan engaged.
# ovl6 (09:18): GGML_CUDA_DISABLE_GRAPHS=1 does not change it (off_ng vs plan_ng DIVERGES; off vs off_ng IDENTICAL) -> not CUDA graphs.
# ovl3's dump never proved the plan was active in llama-eval-callback (the log switch came later).
# Build: t2 at b699951 (+ GGML_SCHED_MOE_PREFETCH_ONLY=<substring of the target's first node name>). No draft, identity mode.
#   1 DUMP  code prompt (first specbench prompt, diverges at char 15), off vs plan with GGML_SCHED_MOE_PREFETCH_LOG=1:
#           planned pairs > 0 AND IDENTICAL => the prompt ubatch computes the same; the difference appears after it
#   2 BISECT specbench, plan restricted to ONE target type each: only up (hoisted into gate), only down (into up), only gate
#           (into the previous layer's down split / the first split) -> which pair type changes the output
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; SHA=b699951
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; PY=/ai/.venv/bin/python
if [ "$(git -C $T2 rev-parse --short=7 HEAD)" != "$SHA" ]; then git -C $T2 merge -q --ff-only $SHA || { echo "OVL7_REFUSED: cannot fast-forward $T2"; exit 1; }; fi
[ -z "$(git -C $T2 status --porcelain --untracked-files=no)" ] || { echo "OVL7_REFUSED: $T2 has local changes"; exit 1; }
cmake --build $NEW -j6 --target llama-server llama-eval-callback > ovl7_build.log 2>&1 || { grep error ovl7_build.log | head; echo "OVL7_FAILED: build"; exit 1; }
strings $NEW/bin/libggml-base.so* | grep -q GGML_SCHED_MOE_PREFETCH_ONLY || { echo "OVL7_FAILED: diag not in build"; exit 1; }
echo "    built $(git -C $T2 rev-parse --short HEAD) | $(date +%T)"
echo "--- 1. DUMP (code prompt, one ubatch) | $(date +%T)"
dump() { env -u LD_LIBRARY_PATH $2 GGML_SCHED_MOE_PREFETCH_LOG=1 LLAMA_MOE_CACHE_SYNC=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 \
  $NEW/bin/llama-eval-callback -m $K2 -ngl 999 -fa on -c 4096 -t 6 --load-mode none -ot exps=CPU --moe-expert-cache 26 \
  -ub 128 -b 2048 -ubp 2048 -f /ai/bench/ovl7_prompt.txt > ovl7_dump_$1.log 2>&1
  echo "    dump $1 rc=$? tensors $(grep -c common_debug_cb_eval ovl7_dump_$1.log) planned pairs $(grep -c 'prefetch plan:' ovl7_dump_$1.log) $(grep -m1 -o 'number of input tokens = [0-9]*' ovl7_dump_$1.log)"; }
dump off  ""
dump plan "GGML_SCHED_MOE_PREFETCH=2"
echo "  off vs plan:"; $PY dumpdiff.py ovl7_dump_off.log ovl7_dump_plan.log | sed 's/^/    /'
echo "--- 2. BISECT by target type (specbench, no draft) | $(date +%T)"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
NOMTP="-ot exps=CPU --moe-expert-cache 26 -ub 128 -b 2048 -ubp 2048"
arm() { local l=$1 envs=$2; echo "##### $l | env: ${envs:-none} | $(date +%T)"
  env -u LD_LIBRARY_PATH $envs LLAMA_MOE_CACHE_SYNC=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $NOMTP > /dev/null 2>&1
  echo "    planned pairs logged: $(grep -c 'prefetch plan:' server_$l.log)"; }
arm ovl7_up   "GGML_SCHED_MOE_PREFETCH=2 GGML_SCHED_MOE_PREFETCH_LOG=1 GGML_SCHED_MOE_PREFETCH_ONLY=ffn_moe_up"
arm ovl7_down "GGML_SCHED_MOE_PREFETCH=2 GGML_SCHED_MOE_PREFETCH_LOG=1 GGML_SCHED_MOE_PREFETCH_ONLY=ffn_moe_down"
arm ovl7_gate "GGML_SCHED_MOE_PREFETCH=2 GGML_SCHED_MOE_PREFETCH_LOG=1 GGML_SCHED_MOE_PREFETCH_ONLY=ffn_moe_gate"
d() { echo "  $1 vs $2:"; $PY textdiff.py runs/$1.client.json runs/$2.client.json | sed 's/^/    /'; }
d ovl5_off ovl7_up
d ovl5_off ovl7_down
d ovl5_off ovl7_gate
echo OVL7_DONE

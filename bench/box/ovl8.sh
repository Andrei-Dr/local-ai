#!/bin/bash
# OVL8: async hazard or layout? ovl7 (09:25): eval-callback logged 0 planned pairs (its default log level may drop GGML INFO) -> the
# "identical prefill" dumps of ovl3 / ovl7 are VOID as evidence. Bisection: planning a single target type (only up / only down /
# only gate) keeps code and reason IDENTICAL; only the full chain (a split that is both a prefetcher and a target) breaks them.
# Build: t2 at d75de16 (+ GGML_SCHED_DIAG_SYNC=1: synchronize every backend after each split). No draft, identity mode.
#   off_sync vs plan_sync   IDENTICAL => a cross-split asynchrony hazard (ordering); DIVERGES => the layout itself
#   DUMP (-lv 4)            code prompt, off vs plan: planned pairs must be > 0 for the comparison to count
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; SHA=d75de16
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; PY=/ai/.venv/bin/python
if [ "$(git -C $T2 rev-parse --short=7 HEAD)" != "$SHA" ]; then git -C $T2 merge -q --ff-only $SHA || { echo "OVL8_REFUSED: cannot fast-forward $T2"; exit 1; }; fi
[ -z "$(git -C $T2 status --porcelain --untracked-files=no)" ] || { echo "OVL8_REFUSED: $T2 has local changes"; exit 1; }
cmake --build $NEW -j6 --target llama-server llama-eval-callback > ovl8_build.log 2>&1 || { grep error ovl8_build.log | head; echo "OVL8_FAILED: build"; exit 1; }
strings $NEW/bin/libggml-base.so* | grep -q GGML_SCHED_DIAG_SYNC || { echo "OVL8_FAILED: diag not in build"; exit 1; }
echo "    built $(git -C $T2 rev-parse --short HEAD) | $(date +%T)"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
NOMTP="-ot exps=CPU --moe-expert-cache 26 -ub 128 -b 2048 -ubp 2048"
arm() { local l=$1 envs=$2; echo "##### $l | env: ${envs:-none} | $(date +%T)"
  env -u LD_LIBRARY_PATH $envs LLAMA_MOE_CACHE_SYNC=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $NOMTP > /dev/null 2>&1
  grep -hE "out of memory|failed to allocate|CUDA error" server_$l.log | head -3 | sed 's/^/    /'; }
arm ovl8_off_sync  "GGML_SCHED_DIAG_SYNC=1"
arm ovl8_plan_sync "GGML_SCHED_DIAG_SYNC=1 GGML_SCHED_MOE_PREFETCH=2"
d() { echo "  $1 vs $2:"; $PY textdiff.py runs/$1.client.json runs/$2.client.json | sed 's/^/    /'; }
d ovl8_off_sync ovl8_plan_sync
d ovl5_off ovl8_off_sync
echo "--- DUMP (-lv 4, code prompt) | $(date +%T)"
dump() { env -u LD_LIBRARY_PATH $2 GGML_SCHED_MOE_PREFETCH_LOG=1 LLAMA_MOE_CACHE_SYNC=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 \
  $NEW/bin/llama-eval-callback -m $K2 -ngl 999 -fa on -c 4096 -t 6 --load-mode none -ot exps=CPU --moe-expert-cache 26 \
  -ub 128 -b 2048 -ubp 2048 -lv 4 -f /ai/bench/ovl7_prompt.txt > ovl8_dump_$1.log 2>&1
  echo "    dump $1 rc=$? planned pairs $(grep -c 'prefetch plan:' ovl8_dump_$1.log) $(grep -m1 -o 'number of input tokens = [0-9]*' ovl8_dump_$1.log)"; }
dump off  ""
dump plan "GGML_SCHED_MOE_PREFETCH=2"
echo "  off vs plan:"; $PY dumpdiff.py ovl8_dump_off.log ovl8_dump_plan.log | sed 's/^/    /'
echo OVL8_DONE

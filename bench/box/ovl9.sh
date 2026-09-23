#!/bin/bash
# OVL9: find the decode-path tensor that reads memory it did not write. ovl8 (09:31): GGML_SCHED_DIAG_SYNC=1 (drain after every
# split) still DIVERGES -> layout, not asynchrony; the eval-callback dump WITH the plan engaged (240 pairs) is IDENTICAL for the
# prompt ubatch (all 2,746 tensors before the logits; the logits block is cut by eval-callback's exit crash). Decode graphs are
# never planned (ovl5) -> (inferred) a decode tensor reads compute-buffer bytes left over from the (planned, re-laid-out) prompt
# graph. Setup: code prompt (37 tokens), -ub 33, no prefill mode: ubatch 1 = 33 tokens (>= 32: GPU expert matmuls, planned),
# ubatch 2 = 4 tokens (<= 4: the expert-cache decode chain). dumpdiff names the first differing tensor; index > ubatch-1 count =
# the decode ubatch. off_a vs off_b = determinism floor.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf
[ "$(git -C $T2 rev-parse --short=7 HEAD)" = "d75de16" ] || { echo "OVL9_REFUSED: $T2 not at d75de16"; exit 1; }
dump() { env -u LD_LIBRARY_PATH $2 GGML_SCHED_MOE_PREFETCH_LOG=1 LLAMA_MOE_CACHE_SYNC=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 \
  $NEW/bin/llama-eval-callback -m $K2 -ngl 999 -fa on -c 4096 -t 6 --load-mode none -ot exps=CPU --moe-expert-cache 26 \
  -ub 33 -b 2048 -lv 4 -f /ai/bench/ovl7_prompt.txt > ovl9_dump_$1.log 2>&1
  echo "    dump $1 rc=$? planned pairs $(grep -c 'prefetch plan:' ovl9_dump_$1.log) blocks $(grep -c common_debug_cb_eval ovl9_dump_$1.log) $(grep -m1 -o 'number of input tokens = [0-9]*' ovl9_dump_$1.log)"
  grep -m3 -oE "moe-cache: [^|]{0,80}" ovl9_dump_$1.log | sed 's/^/        /'; }
dump off_a ""
dump plan  "GGML_SCHED_MOE_PREFETCH=2"
dump off_b ""
for p in "off_a off_b" "off_a plan"; do set -- $p; echo "  $1 vs $2:"; $PY dumpdiff.py ovl9_dump_$1.log ovl9_dump_$2.log | sed 's/^/    /'; done
echo "  ubatch boundaries (index of each result_output block, off_a):"
grep common_debug_cb_eval ovl9_dump_off_a.log | grep -n "result_output\|ffn_moe_cache_slots-0 " | head -6 | sed 's/^/    /'
echo OVL9_DONE

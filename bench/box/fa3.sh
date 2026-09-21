#!/bin/bash
# FA3: skip the V work of attention pairs that carry no weight (fa-vskip.patch, on top of FA1). PROF4: at 120k the decode
# attention kernel is 3.66 ms per layer = 53% of GPU kernel time per token, and it is compute bound, not bandwidth bound (~1 G
# multiply-adds per layer on a ~4 TFLOPS card; the bandwidth floor is 0.4 ms). Less WORK is the lever: at depth the softmax is
# sparse, and the kernel still dequantizes V and runs 8 axpys for every position. GGML_CUDA_FA_VEC_VSKIP=1 skips (position,
# head) pairs below 1e-8 of the running maximum; KQ_sum still counts them. It is LOSSY in principle, so it ships only on proof:
#   1. test-backend-ops FLASH_ATTN_EXT under VSKIP=0 and 1 (masked positions exercise the skip path);
#   2. same restored slot, temperature 0, top-10 logprobs of every generated token, VSKIP=0 vs 1 (probcmp.py) at 120k and 239k;
#      two VSKIP=0 runs give the run-to-run floor that drift has to be read against;
#   3. decode t/s per arm (CUDA graphs on). VSKIP=2 (diagnostic build only, WRONG output) times the K walk alone = the ceiling.
# The diagnostic define is removed and the tree rebuilt at the end: the binary left behind has VSKIP (default 0) and no mode 2.
# Runs as ONE queue job, nothing beside it (the build uses all cores).
source /ai/bench/preflight.sh || exit 1
W=/ai/src/llama.cpp-fa1; PB=/ai/bench; B=$W/build75/bin; C=ggml/src/ggml-cuda; SLOTS=/ai/bench/slots
M=/ai/models; Q=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
[ -s $PB/fa-vskip.patch ] && [ -s $PB/fa-gqa-vec.patch ] && [ -d $W/build75 ] || { echo "FA3_REFUSED: patches or the FA1 build tree missing"; exit 1; }
cd $W || exit 1
build() { # $1 = diag | ship | fa1 (fa1 = back to the tree before this job, for a failed gate)
  git checkout -q -- $C/fattn-vec.cuh && patch -s $C/fattn-vec.cuh < $PB/fa-gqa-vec.patch || { echo FA3_PATCH_FAILED; exit 1; }
  [ "$1" = fa1 ] || patch -s $C/fattn-vec.cuh < $PB/fa-vskip.patch || { echo FA3_PATCH_FAILED; exit 1; }
  [ "$1" = diag ] && sed -i '1i #define GGML_CUDA_FA_VEC_DIAG // fa3.sh: timing-only mode 2, removed again before this job ends' $C/fattn-vec.cuh
  local t0=$(date +%s)
  cmake --build build75 -j6 --target llama-server llama-bench test-backend-ops > $PB/build_fa3_$1.log 2>&1 || { grep -E "error" -B2 -A8 $PB/build_fa3_$1.log | head -60; echo "FA3_BUILD_FAILED $1"; exit 1; }
  echo "built ($1) in $(( $(date +%s) - t0 )) s"
}
build diag
cd $PB
for V in 0 1; do
  GGML_CUDA_FA_VEC_VSKIP=$V $B/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0 > $PB/fa3_ops_v$V.log 2>&1; rc=$?
  echo "unit VSKIP=$V: exit $rc | $(sed -E 's/\x1b\[[0-9;]*m//g' $PB/fa3_ops_v$V.log | grep -E 'tests passed' | tail -1 | xargs)"
  [ $rc -eq 0 ] || { sed -E 's/\x1b\[[0-9;]*m//g' $PB/fa3_ops_v$V.log | grep -B1 -E "FAIL|ERR" | grep -E "^ *[A-Z_]+\(" | head -5 | cut -c1-230
                     cd $W && build fa1; echo "FA3_UNIT_FAILED VSKIP=$V (tree rebuilt as FA1)"; exit 1; }
done
export TIMEOUT=3600 GEN=96 NPROBS=10
export EXT=$'<|im_end|>\n<|im_start|>user\nSummarize the text above in ten detailed bullet points.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
arm() { # label vskip slot ctx server-args...
  local T=$1 V=$2 S=$3 CTX=$4; shift 4
  GGML_CUDA_FA_VEC_VSKIP=$V GGML_OP_OFFLOAD_MIN_BATCH=32 $B/llama-server -m $Q -ngl 999 -ot exps=CPU -c $CTX -t 6 --load-mode none --jinja --parallel 1 --port 8099 \
    --cache-ram 0 -lv 4 --slot-save-path $SLOTS -fa on -ctk q4_0 -ctv q4_0 "$@" > server_$T.log 2>&1 &
  local NP=$!
  for i in $(seq 1 200); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || { echo "$T: SERVER DIED: $(grep -iE 'error|out of memory' server_$T.log | tail -1 | cut -c1-140)"; return 1; }; sleep 2; done
  MODE=extend SLOT=$S DROP=0 python3 /ai/bench/slotclient.py $T
  kill $NP; wait $NP 2>/dev/null; sleep 3
}
for D in "131k 131072 --moe-expert-cache 24 -ub 128 -b 256" "262k 262144 --moe-expert-cache 0 -ub 128 -b 256"; do
  set -- $D; N=$1; CTX=$2; shift 2
  [ -s $SLOTS/ctx1b_c$N.slot ] || { echo "no kept slot for $N, skipped"; continue; }
  echo "##### depth $N | $*"
  for A in v0 v0b v1 v2; do
    case $A in v0|v0b) V=0 ;; v1) V=1 ;; v2) V=2 ;; esac
    arm fa3_${N}_$A $V ctx1b_c$N.slot $CTX "$@"
  done
  echo "  floor (VSKIP=0 twice): $(python3 /ai/bench/probcmp.py runs/fa3_${N}_v0.slot.json runs/fa3_${N}_v0b.slot.json)"
  echo "  VSKIP=1 vs 0:          $(python3 /ai/bench/probcmp.py runs/fa3_${N}_v0.slot.json runs/fa3_${N}_v1.slot.json)"
done
cd $W && build ship
GGML_CUDA_FA_VEC_VSKIP=2 $B/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0 > $PB/fa3_ops_ship.log 2>&1
echo "ship build, VSKIP=2 must be inert: exit $? | $(sed -E 's/\x1b\[[0-9;]*m//g' $PB/fa3_ops_ship.log | grep -E 'tests passed' | tail -1 | xargs)"
echo FA3_DONE

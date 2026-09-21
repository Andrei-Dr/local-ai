#!/bin/bash
# FA4: row-owning K walk for the GQA vec kernel (fa-krow.patch on top of FA1; opt-in GGML_CUDA_FA_VEC_KROW=1).
# FA3 result that motivates it: V-skip was a dead end (branches made the kernel 2x SLOWER, and the V walk is only ~1/5 of the
# step anyway); its timing-only arm showed the K walk is ~2/3 of the attention kernel at depth. The K walk splits every dot over
# 32 threads: each 4-element group pays a block-scale load, half->float, 2 float multiplies, and every (position, column) a
# 5-step reduction. KROW gives each thread a whole K row: row loaded once for 8 columns, exact integer block sums, one float
# multiply-add per block, one max-reduction per column per tile. Not lossy (if anything the dot is more exact), but the float
# summation order changes, so outputs are not bit-identical to FA1. Proof:
#   1. test-backend-ops FLASH_ATTN_EXT (vs the CPU reference) under KROW=0 and KROW=1;
#   2. same restored slot, temperature 0, top-10 logprobs per generated token (probcmp.py) at 120k and 239k: KROW=1 vs FA1, read
#      against (a) KROW=1 twice = run-to-run floor and (b) GQA=0 vs FA1 = what an ACCEPTED kernel change does to the logprobs;
#   3. decode t/s per arm, CUDA graphs on.
# One queue job, nothing beside it. A failed unit gate rebuilds the tree as plain FA1.
source /ai/bench/preflight.sh || exit 1
W=/ai/src/llama.cpp-fa1; PB=/ai/bench; B=$W/build75/bin; C=ggml/src/ggml-cuda; SLOTS=/ai/bench/slots
M=/ai/models; Q=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
[ -s $PB/fa-krow.patch ] && [ -s $PB/fa-gqa-vec.patch ] && [ -d $W/build75 ] || { echo "FA4_REFUSED: patches or the FA1 build tree missing"; exit 1; }
cd $W || exit 1
build() { # $1 = krow | fa1
  git checkout -q -- $C/fattn-vec.cuh && patch -s $C/fattn-vec.cuh < $PB/fa-gqa-vec.patch || { echo FA4_PATCH_FAILED; exit 1; }
  [ "$1" = fa1 ] || patch -s $C/fattn-vec.cuh < $PB/fa-krow.patch || { echo FA4_PATCH_FAILED; exit 1; }
  local t0=$(date +%s)
  cmake --build build75 -j6 --target llama-server llama-bench test-backend-ops > $PB/build_fa4_$1.log 2>&1 || { grep -E "error" -B2 -A8 $PB/build_fa4_$1.log | head -60; echo "FA4_BUILD_FAILED $1"; [ "$1" = fa1 ] || build fa1; exit 1; }
  echo "built ($1) in $(( $(date +%s) - t0 )) s"
}
build krow
cd $PB
for V in 0 1; do
  GGML_CUDA_FA_VEC_KROW=$V $B/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0 > $PB/fa4_ops_k$V.log 2>&1; rc=$?
  echo "unit KROW=$V: exit $rc | $(sed -E 's/\x1b\[[0-9;]*m//g' $PB/fa4_ops_k$V.log | grep -E 'tests passed' | tail -1 | xargs)"
  [ $rc -eq 0 ] || { sed -E 's/\x1b\[[0-9;]*m//g' $PB/fa4_ops_k$V.log | grep -B1 -E "FAIL|ERR" | grep -E "^ *[A-Z_]+\(" | head -8 | cut -c1-230
                     cd $W && build fa1; echo "FA4_UNIT_FAILED KROW=$V (tree rebuilt as FA1)"; exit 1; }
done
export TIMEOUT=3600 GEN=96 NPROBS=10
export EXT=$'<|im_end|>\n<|im_start|>user\nSummarize the text above in ten detailed bullet points.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
arm() { # label "ENV=VAL ..." slot ctx server-args...
  local T=$1 E=$2 S=$3 CTX=$4; shift 4
  env $E GGML_OP_OFFLOAD_MIN_BATCH=32 $B/llama-server -m $Q -ngl 999 -ot exps=CPU -c $CTX -t 6 --load-mode none --jinja --parallel 1 --port 8099 \
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
  arm fa4_${N}_gqa0 "GGML_CUDA_FA_VEC_GQA=0"  ctx1b_c$N.slot $CTX "$@"
  arm fa4_${N}_fa1  "GGML_CUDA_FA_VEC_KROW=0" ctx1b_c$N.slot $CTX "$@"
  arm fa4_${N}_k1   "GGML_CUDA_FA_VEC_KROW=1" ctx1b_c$N.slot $CTX "$@"
  arm fa4_${N}_k1b  "GGML_CUDA_FA_VEC_KROW=1" ctx1b_c$N.slot $CTX "$@"
  echo "  floor  (KROW=1 twice): $(python3 /ai/bench/probcmp.py runs/fa4_${N}_k1.slot.json   runs/fa4_${N}_k1b.slot.json)"
  echo "  accepted change (GQA=0 vs FA1): $(python3 /ai/bench/probcmp.py runs/fa4_${N}_fa1.slot.json runs/fa4_${N}_gqa0.slot.json)"
  echo "  KROW=1 vs FA1:         $(python3 /ai/bench/probcmp.py runs/fa4_${N}_fa1.slot.json  runs/fa4_${N}_k1.slot.json)"
done
echo FA4_DONE

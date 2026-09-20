#!/bin/bash
# ARCH1: llama.cpp's own startup banner says our card is mis-served: "devices will have suboptimal performance due to a lack of
# tensor cores: GTX 1650 SUPER. Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and GGML_CUDA_FORCE_MMQ
# to force the use of the Pascal code for Turing." Every mainline number we have is from an arch-75 build, i.e. kernels and
# dispatch tuned for RTX tensor cores the GTX 16xx does not have (FA takes the MMA_F16 path, matmuls prefer cuBLAS GEMM).
# usage: arch1.sh build (CPU only, start right away) | arch1.sh verify (GPU; queue)
# verify = llama-bench A/B, same source tree (fa1 worktree, GQA kernels on), arch 75 vs 61-virtual;80-virtual + FORCE_MMQ:
#   short context pp512 / tg64 (GPU matmul + CPU<->GPU split path), depth 32768 q4_0 KV pp3 / tg32 (attention at depth),
#   depth 32768 F16 KV tg32 (the F16 attention kernel choice). Lossless either way (same math, different kernels).
# Hypothesis: the Pascal paths win on decode (tg) and at depth. Kill: tg64 and tg32 within 3% or worse => keep arch 75.
W=/ai/src/llama.cpp-fa1; SRC=/ai/src/llama.cpp-mainline; PB=/ai/bench
case "${1:-verify}" in
build)
  [ -d $W ] || { echo "ARCH1_REFUSED: run fa1.sh build first (shared worktree)"; exit 1; }
  cd $W || exit 1
  if [ ! -f build6180/CMakeCache.txt ]; then
    EXTRA=$(grep -E "^(GGML_CUDA_[A-Z_]+|LLAMA_CURL|GGML_NATIVE|GGML_OPENMP):BOOL=" $SRC/build75/CMakeCache.txt | grep -v FORCE_MMQ | sed -E 's/^([A-Z_]+):BOOL=(.*)$/-D\1=\2/' | tr '\n' ' ')
    cmake -B build6180 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual" -DGGML_CUDA_FORCE_MMQ=ON $EXTRA > $PB/build_arch1.log 2>&1 || { tail -5 $PB/build_arch1.log; echo ARCH1_CONFIGURE_FAILED; exit 1; }
  fi
  t0=$(date +%s)
  nice -n 5 cmake --build build6180 -j${J:-4} --target llama-server llama-bench >> $PB/build_arch1.log 2>&1 || { grep -E "error" -B2 -A8 $PB/build_arch1.log | head -60; echo ARCH1_BUILD_FAILED; exit 1; }
  echo "built in $(( $(date +%s) - t0 )) s"; echo ARCH1_BUILD_DONE ;;
verify)
  source /ai/bench/preflight.sh || exit 1
  [ -x $W/build6180/bin/llama-bench ] && [ -x $W/build75/bin/llama-bench ] || { echo "ARCH1_REFUSED: both builds needed (fa1.sh build, arch1.sh build)"; exit 1; }
  Q=/ai/models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
  for b in build75 build6180; do
    LB="$W/$b/bin/llama-bench -m $Q -ngl 999 -ot exps=CPU -fa 1 -t 6 -r 2 --load-mode none"
    echo "##### $b"
    env GGML_OP_OFFLOAD_MIN_BATCH=32 $LB -ub 512 -b 512 -p 512 -n 64 2>&1 | grep -E "pp512|tg64|error|lack of tensor" | cut -c1-200
    env GGML_OP_OFFLOAD_MIN_BATCH=32 $LB -ub 2048 -b 2048 -ctk q4_0 -ctv q4_0 -d 32768 -r 1 -p 3 -n 32 2>&1 | grep -E "pp3|tg32|error" | cut -c1-200
    env GGML_OP_OFFLOAD_MIN_BATCH=32 $LB -ub 2048 -b 2048 -d 32768 -r 1 -p 0 -n 32 2>&1 | grep -E "tg32|error" | cut -c1-200
  done
  echo ARCH1_DONE ;;
*) echo "usage: arch1.sh build|verify"; exit 2 ;;
esac

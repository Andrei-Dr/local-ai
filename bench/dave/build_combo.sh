#!/bin/bash
# Build "combo" = e613ef2 + local-ai 0001-0031 + davetha/llama.cpp-mi210 04,06,09,10,11,13-16 (08 superseded upstream, 12 is
# a 2-GPU state-restore fix none of our arms exercise) with Dave's cmake flags. Tree f8130ba. Inside llama-rocm714-rpc:tune.
set -e
cd /w
DF="-DGPU_TARGETS=gfx90a -DGGML_HIP=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON -DGGML_HIP_ROCWMMA_FATTN=OFF -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF"
[ "$(git -C combo rev-parse HEAD^{tree})" = f8130baebc9f3d164eedab9ddf15f1916ee351ea ] || { echo "combo tree mismatch"; exit 1; }
cmake -S combo -B combo/build $DF > combo.cmake.log 2>&1
cmake --build combo/build -j 24 --target llama-server > combo.build.log 2>&1 || { tail -30 combo.build.log; echo BUILD_FAILED; exit 1; }
echo COMBO_BUILD_DONE

#!/bin/bash
# Build (1) "dave": upstream 030ebb558 + davetha/llama.cpp-mi210 patches 04-16 with his cmake flags (his Dockerfile, verbatim
# flags), and (2) "patched-df": our tree (e613ef2 + series) with his HIP flags, to split "his flags" from "his patches".
# Runs inside llama-rocm714-rpc:tune (ROCm 7.14, the toolchain of every other arm).
set -e
cd /w
DF="-DGPU_TARGETS=gfx90a -DGGML_HIP=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON -DGGML_HIP_ROCWMMA_FATTN=OFF -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF"
if [ ! -d dave/.git ]; then git clone -q base dave; fi
cd dave && git checkout -q -f --detach 030ebb558 && git clean -qfdx -e build
for p in /w/dave-mi210/patches/0[4-9]-*.patch /w/dave-mi210/patches/1[0-6]-*.patch; do echo "apply $(basename $p)"; git apply --3way "$p"; done
cd /w
echo "== build dave $(date +%T)"
cmake -S dave -B dave/build $DF > dave.cmake.log 2>&1
cmake --build dave/build -j 24 --target llama-server > dave.build.log 2>&1 || { tail -30 dave.build.log; echo BUILD_FAILED dave; exit 1; }
echo "== build patched-df $(date +%T)"
cmake -S patched -B patched/build-df $DF > patched-df.cmake.log 2>&1
cmake --build patched/build-df -j 24 --target llama-server > patched-df.build.log 2>&1 || { tail -30 patched-df.build.log; echo BUILD_FAILED patched-df; exit 1; }
echo BUILD_DONE $(date +%T)

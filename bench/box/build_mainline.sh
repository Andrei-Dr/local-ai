#!/bin/bash
# Build ggml-org mainline (+ our rebased cache series) as a SECOND tree on the box. Does not touch the fork.
source /ai/bench/preflight.sh || exit 1
set -e
D=/ai/src/llama.cpp-mainline
if [ ! -d $D ]; then git clone -q https://github.com/ggml-org/llama.cpp.git $D; fi
cd $D
git fetch -q origin && git checkout -q e613ef2 2>/dev/null || git checkout -q master
git checkout -q -B moe-cache
git apply --check /ai/bench/mainline-moecache.diff && git apply --whitespace=nowarn /ai/bench/mainline-moecache.diff
git add -A && git -c user.name=ai -c user.email=ai@local commit -q -m "feat: expert cache (rebased onto mainline)"
(time cmake -S . -B build75 -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_CUDA_FORCE_MMQ=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DLLAMA_CURL=OFF && cmake --build build75 -j6 --target llama-server llama-bench) > /ai/bench/build_mainline.log 2>&1 || { grep -E "error" -A4 /ai/bench/build_mainline.log | head -30; echo BUILD_MAINLINE_FAILED; exit 1; }
echo BUILD_MAINLINE_DONE

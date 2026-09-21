#!/bin/bash
# CPU1BENCH: size the CPU1 lever (task-parallel expert FFN) with a microbenchmark before writing the kernel. See src/cpu1bench.cpp.
# Read: "ggml t6" vs "task t6" at pairs 4 (one decode token, ~50% hit) and 12 (MTP verify of 3). GO for CPU1 if task t6 beats
# ggml t6 by >= 1.25x at pairs 12; if ggml already scales ~5x on 6 threads, the CPU phase is kernel-bound and CPU1 is dead
# (then the lever is the q2_K / q3_K dot itself). CPU only, ~3 min, box exclusive (timing).
source /ai/bench/preflight.sh || exit 1
S=/ai/src/llama.cpp-mainline; B=$S/build75/bin; cd /ai/bench
g++ -O2 -std=c++17 -I$S/ggml/include src/cpu1bench.cpp -o cpu1bench -L$B -lggml-cpu -lggml-base -lggml -Wl,-rpath,$B -lpthread 2> build_cpu1bench.log \
  || { head -30 build_cpu1bench.log; echo CPU1BENCH_BUILD_FAILED; exit 1; }
for W in "" "OMP_WAIT_POLICY=active"; do
  echo "##### ${W:-default OMP wait policy}"
  env $W ./cpu1bench ${ITERS:-3000} 2>&1 | grep -vE "^ggml_|^load_backend" | cut -c1-200
done
echo CPU1BENCH_DONE

#!/bin/bash
# MV1 (lead 2): are the dense decode mat-vecs ALU-bound on TU116? pf3 trace: the 4096-row IQ2_S mat-vec (14.8/token) moves ~2.6 MB
# in 82.7 us (~32 GB/s) and the 1024-row one ~17 GB/s, vs 192 GB/s VRAM; the dense mat-vecs cost ~3.6 ms/token vs ~1.3 ms at
# bandwidth. test-backend-ops perf, MUL_MAT on CUDA0, per weight type at the perf-suite shape (m = 4096, k = 14336) and
# n = 1 / 3 / 5 columns (decode / MTP verify) -> achieved GB/s per type = whether the kernel (not the bytes) is the limit.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
V2=/ai/src/llama.cpp-v2/build75
[ -x $V2/bin/test-backend-ops ] || { echo "MV1_FAILED: no test-backend-ops"; exit 1; }
for t in iq2_s iq3_s q2_K q3_K q4_K q4_0 q8_0; do
  for nn in 1 3 5; do
    out=$(env -u LD_LIBRARY_PATH $V2/bin/test-backend-ops perf -o MUL_MAT -b CUDA0 -p "type_a=$t,type_b=f32,m=4096,n=$nn,k=14336," 2>&1 | grep -E "MUL_MAT\(" | head -1)
    echo "  $t n=$nn: ${out:-(no matching case)}" | cut -c1-220
  done
done
echo MV1_DONE

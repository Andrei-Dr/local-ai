#!/bin/bash
# DENSE1: is a dense model's FFN activation energy concentrated enough to treat it "as MoE" (hot neurons on the GPU, cold on the
# CPU, PowerInfer-style)? Zero C++: llama-imatrix records the mean squared input of every matmul, and the input of ffn_down IS the
# per-neuron gated activation. Model: Qwen3.8-27B IQ3_M (dense; 1.38 tok/s on this box = why dense is dead here today).
# Read with bench/imx_heat.py on the pulled file: share of energy in the top 5 / 20 / 50% of neurons per layer. SiLU-gated FFNs
# have no exact zeros, so a FLAT verdict (top 20% < 60% of energy) kills the static split; CONCENTRATED (>= 80%) earns a
# per-token probe next (mean energy is necessary, not sufficient: contextual sparsity needs per-token activations).
source /ai/bench/preflight.sh || exit 1
B=/ai/src/llama.cpp-mainline/build75; M=/ai/models; O=/ai/bench/imx; mkdir -p $O; cd /ai/bench
F=$M/Qwen3.8-27B-i1-IQ3_M.gguf
[ -f $F ] && [ -s corpus/kq_calib.txt ] && [ -x $B/bin/llama-imatrix ] || { echo "DENSE1_REFUSED: model, corpus or llama-imatrix missing"; exit 1; }
t0=$(date +%s)
$B/bin/llama-imatrix -m $F -f corpus/kq_calib.txt -o $O/qwen38_27b_dense.imatrix -c 512 -b 512 --chunks ${CH:-24} -ngl ${NGL:-12} -t 6 > imatrix_dense1.log 2>&1 \
  || { grep -iE "error|out of memory" imatrix_dense1.log | tail -3; tail -3 imatrix_dense1.log; echo DENSE1_FAILED; exit 1; }
echo "imatrix: $(ls -la $O/qwen38_27b_dense.imatrix | awk '{print $5}') bytes in $(( $(date +%s) - t0 )) s | $(grep -E "Final estimate|chunks" imatrix_dense1.log | tail -1 | cut -c1-120)"
echo "magic: $(head -c 4 $O/qwen38_27b_dense.imatrix | od -An -c | tr -s ' ')"
echo DENSE1_DONE

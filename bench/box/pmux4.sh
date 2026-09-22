#!/bin/bash
# PMUX4: is a bigger prefill ubatch numerically equivalent? pmux2's identity check (deterministic cache) was IDENTICAL on code,
# reason and edit (edit went through prefill mode) but the long prompt diverged at char 318 on a near-tie. Two ULP-level causes:
# (a) ubatch boundaries (delta-net chunking, attention tiles), (b) the cache re-ranked by the whole prompt instead of per chunk
# (hit = GPU kernel, miss = CPU kernel). This isolates (a): served build, cache OFF, the same text evaluated at ub 128 (reference
# logits) and ub 2048 / 512 -> KL divergence. Read: mean KLD << the quantization's own (K2 vs Q6: 0.218) and same-top-token
# ~100% => ubatch size is numerically neutral and prefill mode ships; a KLD anywhere near quantization noise => it does not.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench/kld || exit 1
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive; K2=$M/$N-K2-expQ2K-downQ3K.gguf
B=/ai/src/llama.cpp-mainline/build75/bin
PP="$B/llama-perplexity -m $K2 -f /ai/bench/kld/wiki.test.raw -c 2048 -b 2048 --chunks 12 -ngl 999 -ot exps=CPU -t 6 -fa on"
stats() { grep -E "Mean +KLD|Maximum KLD|99\.0% +KLD|Same top p|Mean PPL\(Q\)|out of memory" "$1" | cut -c1-120 | sed 's/^/    /'; }
echo "##### reference ub 128 | $(date +%T)"
GGML_OP_OFFLOAD_MIN_BATCH=32 $PP -ub 128 --kl-divergence-base pmux4_ub128.kld > pmux4_ref.log 2>&1 || { tail -3 pmux4_ref.log; echo PMUX4_FAILED; exit 1; }
grep -E "Final estimate|estimated PPL" pmux4_ref.log | tail -1 | sed 's/^/    /'
for U in 2048 512 128; do
  echo "##### ub $U vs ub 128 | $(date +%T)"
  GGML_OP_OFFLOAD_MIN_BATCH=32 $PP -ub $U --kl-divergence-base pmux4_ub128.kld --kl-divergence > pmux4_ub$U.log 2>&1; stats pmux4_ub$U.log
done
echo "    (ub 128 vs itself = the run-to-run floor of this measurement)"
rm -f pmux4_ub128.kld
echo PMUX4_DONE

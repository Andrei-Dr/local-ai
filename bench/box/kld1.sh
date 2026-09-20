#!/bin/bash
# KLD1: a quality instrument that can SEE quantization damage. Our task gate (GSM8K / HumanEval / MMLU-Pro, thinking off) is
# saturated and small-n; scouts report ~2-bit PTQ passing such sets while losing 20+ pts on hard reasoning. KL divergence of a
# quant's next-token distribution against a high-precision reference on the SAME tokens is far more sensitive (every token is a
# sample, no answer-extraction noise) and costs minutes. Reference = the HauhauCS Q6_K_P source (28.5 GiB > 16 GiB RAM: mmap +
# NVMe streaming, ~10 s per 512-token chunk, experts stream to the GPU as in prefill). Candidates: the served IQ2_M and the K2
# requant (expert gate/up Q2_K, down Q3_K). Text = wikitext-2 test (the published llama.cpp standard), first 40 chunks of 512.
# Read: Mean KLD, "Same top p" (top-1 agreement with Q6), PPL ratio, 99% / max KLD (tail = where reasoning chains break).
# Hypothesis: K2 (12.9 GB, more bits, imatrix from the IQ2_M model) has LOWER KLD than IQ2_M (11 GB) => K2 is the more accurate
# file as well as the faster one. Kill for K2: mean KLD or 99% KLD worse than IQ2_M. The saved Q6 logits are reused by every
# later candidate (Unsloth-recipe replay, better imatrix), so this job is also the screening rig for workstream QX.
source /ai/bench/preflight.sh || exit 1
D=/ai/src/llama.cpp-mainline; B=$D/build75; M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K=/ai/bench/kld; mkdir -p $K; cd $K || exit 1
[ -x $B/bin/llama-perplexity ] || cmake --build $B -j6 --target llama-perplexity > /ai/bench/build_kld1.log 2>&1 \
  || { grep -E "error" -A3 /ai/bench/build_kld1.log | head -20; echo KLD1_BUILD_FAILED; exit 1; }
if [ ! -s wiki.test.raw ]; then
  curl -sL --max-time 300 -o wt2.zip https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip \
    && unzip -oq wt2.zip && cp wikitext-2-raw/wiki.test.raw . || { echo KLD1_DATA_FAILED; exit 1; }
fi
CH=${CH:-40}
PP="$B/bin/llama-perplexity -f $K/wiki.test.raw -c 512 -b 512 -ub 512 --chunks $CH -ngl 999 -ot exps=CPU -t 6 -fa on"
stats() { grep -E "Mean +KLD|Maximum KLD|99\.0% +KLD|99\.9% +KLD|Median +KLD|Same top p|Mean PPL\(Q\)|PPL\(Q\)/PPL\(base\)|Mean +.p|RMS .p|out of memory|error" "$1" | cut -c1-160 | sed 's/^/    /'; }
if [ ! -s q6.kld ]; then
  echo "##### kld1_base_q6 | $(date +%T) | saving reference logits (slow: the file streams from NVMe)"
  GGML_OP_OFFLOAD_MIN_BATCH=32 $PP -m $M/$N-Q6_K_P.gguf --kl-divergence-base $K/q6.kld > base_q6.log 2>&1
  grep -E "Final estimate|out of memory|error" base_q6.log | tail -3 | sed 's/^/    /'
  [ -s q6.kld ] || { echo "    no reference logits written"; tail -5 base_q6.log; echo KLD1_FAILED; exit 1; }
  echo "    reference: $(ls -la q6.kld | awk '{print $5}') bytes, done $(date +%T)"
fi
for cand in IQ2_M K2-expQ2K-downQ3K; do
  f=$M/$N-$cand.gguf; [ -f "$f" ] || { echo "##### kld1_$cand SKIPPED: $f missing"; continue; }
  echo "##### kld1_$cand | $(date +%T) | $(ls -laL $f | awk '{print $5}') bytes"
  GGML_OP_OFFLOAD_MIN_BATCH=32 $PP -m $f --kl-divergence-base $K/q6.kld --kl-divergence > kld_$cand.log 2>&1
  stats kld_$cand.log
done
echo KLD1_DONE

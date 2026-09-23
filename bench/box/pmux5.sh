#!/bin/bash
# PMUX5: non-inferiority of the prefill changes against the TRUTH, not against each other. pmux4: ub 2048 vs ub 128 = mean KLD
# 0.0055, same top 96.9% — small against the quantization's own KLD (K2 vs Q6 0.218) but a 3% top-token flip rate needs the
# right yardstick: each variant's KLD vs the Q6_K_P reference on the same text. Non-inferior = mean KLD within +-2% of the ub-128
# arm's and same-top within 0.5 pts. Arms (cache off, -c 2048, 12 chunks of wikitext): K2 ub128 MMA (today's prefill), K2 ub2048
# MMA (prefill mode), K2 ub2048 dp4a (prefill mode + GGML_CUDA_MMQ_NO_MMA).
# + TU1's float-order switches (ov build after tu1 applied 0011-0013): K2 ub2048 dp4a + FA tile (GGML_CUDA_FA_NO_MMA), K2 ub128 dp4a
# with and without MoE per-expert tile width (GGML_CUDA_MMQ_MOE_EXPERT_COLS). Same yardstick: vs the ub128 MMA arm.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench/kld || exit 1
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; Q6=$M/$N-Q6_K_P.gguf
SERVED=/ai/src/llama.cpp-mainline/build75/bin; OV=/ai/src/llama.cpp-ov/build75/bin
[ -s $Q6 ] || { echo "PMUX5_REFUSED: $Q6 missing"; exit 1; }
COMMON="-f /ai/bench/kld/wiki.test.raw -c 2048 -b 2048 --chunks 12 -ngl 999 -ot exps=CPU -t 6 -fa on"
stats() { grep -E "Mean +KLD|99\.0% +KLD|Same top p|Mean PPL\(Q\)" "$1" | cut -c1-110 | sed 's/^/    /'; }
echo "##### Q6_K_P reference logits (c 2048, ub 512) | $(date +%T)"
GGML_OP_OFFLOAD_MIN_BATCH=32 $SERVED/llama-perplexity -m $Q6 $COMMON -ub 512 --kl-divergence-base pmux5_q6.kld > pmux5_q6.log 2>&1 || { tail -3 pmux5_q6.log; echo PMUX5_FAILED; exit 1; }
arm() { local l=$1 bin=$2 ub=$3 ld=$4 envs=$5; echo "##### $l | env: ${envs:-none} | $(date +%T)"
  env $envs LD_LIBRARY_PATH=$ld GGML_OP_OFFLOAD_MIN_BATCH=32 $bin/llama-perplexity -m $K2 $COMMON -ub $ub --kl-divergence-base pmux5_q6.kld --kl-divergence > pmux5_$l.log 2>&1; stats pmux5_$l.log; }
arm k2_ub128_mma   $SERVED 128  ""
arm k2_ub2048_mma  $SERVED 2048 ""
arm k2_ub2048_dp4a $OV     2048 $OV
if strings $OV/libggml-cuda.so* | grep -q GGML_CUDA_FA_NO_MMA; then
  arm k2_ub2048_dp4a_fatile $OV 2048 $OV "GGML_CUDA_FA_NO_MMA=1"
  arm k2_ub128_dp4a         $OV 128  $OV
  arm k2_ub128_dp4a_expcols $OV 128  $OV "GGML_CUDA_MMQ_MOE_EXPERT_COLS=1"
else
  echo "  (ov build lacks the tu116 switches: tu1 has not run; FA tile / expert-cols arms skipped)"
fi
rm -f pmux5_q6.kld
echo PMUX5_DONE

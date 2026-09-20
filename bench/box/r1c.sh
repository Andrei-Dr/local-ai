#!/bin/bash
# R1C: the one row the R1 keep/kill still lacks. r1 measured SPEED at the shipped footing (cache 30 + MTP n=2: b1.0 = +20-27% decode)
# but quality at cache 48 without MTP; r1b added the b0 baseline at cache 48 (b1.0: GSM8K 98.0 / HE 97.6 vs b0 96.0 / 92.7, +8.9%
# decode; b2.0 loses GSM8K -10) and its cache-48 + MTP speed rows all OOMed (48 slots + the 727 MiB head do not fit). The bias steers
# routing more often when the cache is smaller, so cache-48 quality does not certify cache 30. This job: GSM8K + HumanEval at the
# SHIPPED config (cache 30, MTP n=2, -ub 128) for b0 / b0.5 / b1.0. Gate unchanged: keep a bias only if quality is within 1 sigma
# of the b0 row here. The decode t/s of these rows is a second speed read at the shipped config on quality prompts.
source /ai/bench/preflight.sh || exit 1
D=/ai/src/llama.cpp-mainline
[ "$(git -C $D rev-parse --short=7 HEAD)" = 2582f5c ] || { echo "R1C_REFUSED: $D is not at 2582f5c (r1b leaves it there)"; exit 1; }
cd /ai/bench
export BUILD=$D/build75 QARGS="--sets gsm8k,humaneval"
M=/ai/models; Q=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
for b in 0 0.5 1.0; do t=$(echo $b | tr -d .)
  MODEL=$Q ./qualbench.sh q36_iq2m_c30mtp_r1c_b$t -ngl 999 -ot "exps=CPU" --moe-expert-cache 30 -ub 128 -b 256 \
    -md $M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --moe-expert-cache-bias $b
done
echo R1C_DONE

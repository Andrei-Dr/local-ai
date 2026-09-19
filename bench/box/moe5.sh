#!/bin/bash
# After P3+P4: (a) overlap gain on the best cached configs, (b) MTP on top of the cache, (c) K-quant Gemma files, (d) long generations.
cd /ai/bench
export BUILD=/ai/src/llama.cpp-moecache/build75
M=/ai/models
Q=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
G=$M/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ3_M.gguf
G3=$M/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q3_K_M.gguf
G2=$M/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q2_K_P.gguf
run() { local m=$1 l=$2; shift 2; echo "##### $l | $*"; MODEL=$m OFFLOAD=32 ./specbench.sh 999 "$l" "$@" 2>&1; grep -hE "MoE expert cache enabled|moe-cache: steps" server_$l.log | tail -2 | cut -c1-230 | sed -E 's/^[0-9.]+ +[A-Z] +/    /'; }
echo "########## (a) P4 overlap ##########"
run $G p4_g4_c16_ub128 -ot "exps=CPU" --moe-expert-cache 16 -ub 128 -b 256
run $Q p4_q36_c48      -ot "exps=CPU" --moe-expert-cache 48
run $G p4_g4_c16_t5    -ot "exps=CPU" --moe-expert-cache 16 -ub 128 -b 256 -t 5
run $Q p4_q36_c48_t5   -ot "exps=CPU" --moe-expert-cache 48 -t 5
echo "########## (b) MTP on top of the cache ##########"
for n in 1 2 3; do run $G p4_g4_c12_mtp$n -ot "exps=CPU" --moe-expert-cache 12 -ub 128 -b 256 -md $M/mtp-gemma-4-26B-A4B-it.gguf --spec-type draft-mtp --spec-draft-n-max $n; done
for n in 1 2 3; do run $Q p4_q36_c36_mtp$n -ot "exps=CPU" --moe-expert-cache 36 -ub 128 -b 256 -md $M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max $n; done
echo "########## (c) K-quant Gemma files ##########"
run $G3 p4_g4q3k_off  -ot "exps=CPU"
run $G3 p4_g4q3k_c15  -ot "exps=CPU" --moe-expert-cache 15 -ub 128 -b 256
run $G2 p4_g4q2k_off  -ot "exps=CPU"
run $G2 p4_g4q2k_c19  -ot "exps=CPU" --moe-expert-cache 19 -ub 128 -b 256
echo "########## (d) steady state, 800 generated tokens ##########"
GEN=800 run $G p4_g4_c16_gen800 -ot "exps=CPU" --moe-expert-cache 16 -ub 128 -b 256
GEN=800 run $Q p4_q36_c48_gen800 -ot "exps=CPU" --moe-expert-cache 48
echo MOE5_DONE

#!/bin/bash
# RACE1: is a reasoning chain that hits the token budget a property of the PROBLEM or of the SEED? hq1_iq2m (served IQ2_M file,
# card sampler, one fixed seed per item): finished chains are right 23 of 24 on MATH-L5, and 16 of 40 never finish inside 32k, none
# of them a loop. Rerun ONLY those cut MATH-L5 items with two other seeds (salt 1, salt 2), same file / sampler / budget / config.
# Read (bench/qual/race.py, pre-registered): over the cut items at N = 3 chains, finished >= 50% => RACE_VERDICT seed = running a
# few seeds and taking the first finisher (or retrying a cut chain) is an accuracy lever, next step = its serving form;
# finished <= 20% => problem = dead, the budget / the quantization is the lever; between = mixed, look at first_ok vs any_ok.
# AIME's 10 cut chains are left out on purpose (24 min each, low prior): add them only after a seed verdict here.
# Cost: 16 items x 2 salts x <= 19 min = <= 10 h. Same server config as hq1.sh.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive; F=$M/$N-IQ2_M.gguf
BASE=/ai/bench/qual/results/q36_iq2m_hard.jsonl
[ -f "$F" ] && [ -s "$BASE" ] || { echo "RACE1_REFUSED: model or base results missing"; exit 1; }
grep -q -- "--seed-salt" /ai/bench/qual/qual.py && [ -f /ai/bench/qual/race.py ] || { echo "RACE1_REFUSED: deploy qual.py + race.py first"; exit 1; }
# PINNED (09-23): the served tree is about to be promoted to dp4a MMQ + prefill mode; seed 1 and the first seed-2 items ran on
# this build, so the rest stays on it (a build change would add a second perturbation to a seed experiment).
export BUILD=/ai/src/llama.cpp-mainline/build75
for s in 1 2; do
  export QARGS="--data data_hard --sets math_l5 --think --seed-salt $s --ids-from $BASE --ids-filter cut"
  MODEL=$F ./qualbench.sh q36_iq2m_hard_s$s -ngl 999 -ot "exps=CPU" --moe-expert-cache 24 -ub 128 -b 256 -c 49152
done
/ai/.venv/bin/python /ai/bench/qual/race.py $BASE qual/results/q36_iq2m_hard_s1.jsonl qual/results/q36_iq2m_hard_s2.jsonl --json qual/results/race1.json
echo RACE1_DONE

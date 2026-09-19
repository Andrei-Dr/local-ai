#!/bin/bash
# Longer router traces WITH token ids (for research/scripts/moe-cache-sim/predict.py: is routing predictable from the
# token n-gram, i.e. can drafted tokens drive expert prefetch?). 40k tokens per corpus, Qwen3.6 + Gemma4 Q3_K_M.
source /ai/bench/preflight.sh || exit 1
cd /ai/src/llama.cpp-moecache && python3 /ai/bench/trace_tok.py && cmake --build build75 -j6 --target llama-moe-trace > /ai/bench/build_trace2.log 2>&1 || { echo TRACE2_BUILD_FAILED; exit 1; }
cd /ai/src/llama.cpp-moecache
( find gguf-py tools/server -name "*.py" -o -name "*.ts" | head -40 | xargs cat; find src -maxdepth 1 -name "llama-*.cpp" | head -12 | xargs cat ) | head -c 450000 > /ai/bench/corpus/code_big.txt
[ -f /ai/bench/corpus/wiki.test.raw ] && head -c 450000 /ai/bench/corpus/wiki.test.raw > /ai/bench/corpus/prose_big.txt
B=/ai/src/llama.cpp-moecache/build75/bin
for p in "q36 /ai/models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf" "g4q3k /ai/models/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q3_K_M.gguf"; do set -- $p
  for c in code prose; do
    [ -s /ai/bench/corpus/${c}_big.txt ] || continue
    MOE_TRACE_OUT=/ai/bench/traces/$1_${c}_tok.bin MOE_TRACE_MAX_TOKENS=40000 $B/llama-moe-trace -m $2 -f /ai/bench/corpus/${c}_big.txt -c 512 -b 512 -t 6 -ngl 999 -ot "exps=CPU" --load-mode none > /ai/bench/traces/$1_${c}_tok.log 2>&1
    echo "  trace $1_$c: exit $? bin $(stat -c %s /ai/bench/traces/$1_${c}_tok.bin 2>/dev/null) tok $(stat -c %s /ai/bench/traces/$1_${c}_tok.bin.tok 2>/dev/null)"
  done
done
echo TRACE2_DONE

#!/bin/bash
# Where does MoE decode time go? thread sweep, CUDA-graph A/B, perf profile, then router traces.
cd /ai/bench
B=/ai/src/llama.cpp/build75/bin
Q=/ai/models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
G=/ai/models/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ3_M.gguf
tg() { # tg LABEL MODEL THREADS [env...]
  local l=$1 m=$2 t=$3; shift 3
  local r; r=$(env "$@" $B/llama-bench -m $m -ngl 99 -ot "exps=CPU" -fa 1 -t $t -p 0 -n 64 -r 2 --load-mode none 2>&1 | grep -E "tg64" | sed -E 's/.*tg64 \| *//; s/ *\|.*//')
  echo "  $l t=$t $* : tg64 = $r t/s"
}
echo "########## thread sweep / CUDA graphs ##########"
for t in 2 3 4 5 6; do tg q36 $Q $t A=1; done
tg q36 $Q 6 GGML_CUDA_DISABLE_GRAPHS=1
for t in 3 4 5 6; do tg g4 $G $t A=1; done
tg g4 $G 6 GGML_CUDA_DISABLE_GRAPHS=1
echo "########## perf profile (decode only, -D 20s) ##########"
for p in "q36 $Q" "g4 $G"; do set -- $p
  perf record -q -F 999 -D 20000 -o /ai/bench/perf_$1.data -- $B/llama-bench -m $2 -ngl 99 -ot "exps=CPU" -fa 1 -t 6 -p 0 -n 768 -r 1 --load-mode none >/dev/null 2>&1
  echo "--- $1"; perf report -i /ai/bench/perf_$1.data --stdio --sort dso,symbol 2>/dev/null | grep -vE "^#|^$" | head -22 | cut -c1-150
done
echo "########## router traces ##########"
for p in "q36 $Q" "g4 $G"; do set -- $p; for c in code prose; do
  MOE_TRACE_OUT=/ai/bench/traces/$1_$c.bin MOE_TRACE_MAX_TOKENS=8000 $B/llama-moe-trace -m $2 -f /ai/bench/corpus/$c.txt -c 512 -b 512 -t 6 -ngl 999 -ot "exps=CPU" --load-mode none > /ai/bench/traces/$1_$c.log 2>&1
  echo "  trace $1_$c: exit $? size $(stat -c %s /ai/bench/traces/$1_$c.bin 2>/dev/null)"
done; done
echo PROF1_DONE

#!/bin/bash
# MT1: does a multi-turn conversation reuse its prefix on the STABLE serving config? Qwen3.6 is hybrid (30 Gated DeltaNet layers
# with recurrent state + 10 attention layers): the server can only resume from a context checkpoint of the recurrent state
# (--ctx-checkpoints, default 32 per slot, min spacing 8192 tokens, plus one at the last user message); otherwise it re-processes
# the whole prompt ("forcing full prompt re-processing ... hybrid/recurrent memory"). At ~400 t/s a 9k-token history re-prefilled
# every turn = ~23 s TTFT per turn. Arms (STABLE build, serving flags, -c 12288, cache 22, MTP n=2; mt1.py, 3 turns):
#   default   checkpoints as shipped            | nocp  --ctx-checkpoints 0 (reference: what no reuse costs)
#   short     same as default with a ~2.3k-token first turn (below the 8192 min spacing)
# Read: turns 2-3 prompt_n (processed) vs cache_n (reused), TTFT; plus the server's checkpoint / re-processing log lines.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
V2=/ai/src/llama.cpp-v2/build75
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
run() { # label chars extra-args...
  local l=$1 c=$2; shift 2
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 $V2/bin/llama-server -m $K2 -ngl 999 -fa on -c 12288 -t 6 \
    --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 -lv 4 -ot exps=CPU --moe-expert-cache 22 -md $HEAD \
    --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 4096 -ubp 2048 "$@" > server_$l.log 2>&1 &
  local NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  echo "##### $l | chars $c | $* | $(date +%T)"
  python3 mt1.py $l $c 2>&1 | sed 's/^/    /'
  grep -hoE "forcing full prompt re-processing[^(]*|restored context checkpoint \([^)]*\)|created context checkpoint [0-9]+ of [0-9]+[^\n]{0,60}|erased invalidated context checkpoint" server_$l.log \
    | cut -c1-110 | sort | uniq -c | sed 's/^/    log: /'
  kill $NP; wait $NP 2>/dev/null
}
run mt1_default 40000
run mt1_nocp    40000 --ctx-checkpoints 0
run mt1_short   10000
echo MT1_DONE

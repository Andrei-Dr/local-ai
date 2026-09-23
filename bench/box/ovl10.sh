#!/bin/bash
# OVL10: is the prompt warm-up observer the layout-dependent reader? ovl9 (09:45): in callback mode (node-by-node, no CUDA op
# fusion) a planned prompt ubatch AND the following decode-chain ubatch are IDENTICAL off vs plan. In the server the first prompt's
# warm-up differs: 1016 uploads (off) vs 1017 (plan), then hit rates 51.2 vs 52.4% -> different cached experts -> different GPU vs CPU
# expert split -> different rounding. The observer (moe_warm_obs_op, CPU) reads ggml_cont(selected_experts) of every >= 32-token batch.
# Build: t2 at 20c1581 (+ LLAMA_MOE_WARM_TRACE=1: per call layer / n / checksum / out-of-range ids). No draft, identity mode.
#   A  --moe-expert-cache-warm 0 (no warm-up): off vs plan IDENTICAL => the warm-up input is the reader
#   B  traced: the first differing observer call (layer, n, sum, bad) off vs plan
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; SHA=20c1581
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; PY=/ai/.venv/bin/python
if [ "$(git -C $T2 rev-parse --short=7 HEAD)" != "$SHA" ]; then git -C $T2 merge -q --ff-only $SHA || { echo "OVL10_REFUSED: cannot fast-forward $T2"; exit 1; }; fi
[ -z "$(git -C $T2 status --porcelain --untracked-files=no)" ] || { echo "OVL10_REFUSED: $T2 has local changes"; exit 1; }
cmake --build $NEW -j6 --target llama-server > ovl10_build.log 2>&1 || { grep error ovl10_build.log | head; echo "OVL10_FAILED: build"; exit 1; }
strings $NEW/bin/libllama.so* | grep -q LLAMA_MOE_WARM_TRACE || { echo "OVL10_FAILED: diag not in build"; exit 1; }
echo "    built $(git -C $T2 rev-parse --short HEAD) | $(date +%T)"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
NOMTP="-ot exps=CPU --moe-expert-cache 26 -ub 128 -b 2048 -ubp 2048"
arm() { local l=$1 envs=$2; shift 2; echo "##### $l | env: ${envs:-none} | $* | $(date +%T)"
  env -u LD_LIBRARY_PATH $envs LLAMA_MOE_CACHE_SYNC=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $NOMTP "$@" > /dev/null 2>&1
  echo "    $(grep -oE 'prompt warm-up: [0-9]+ uploads' server_$l.log | head -2 | tr '\n' ' ') $(grep -oE 'hit rate [0-9.]+%' server_$l.log | head -1)"; }
arm ovl10_w0_off  ""                          --moe-expert-cache-warm 0
arm ovl10_w0_plan "GGML_SCHED_MOE_PREFETCH=2" --moe-expert-cache-warm 0
arm ovl10_tr_off  "LLAMA_MOE_WARM_TRACE=1"
arm ovl10_tr_plan "LLAMA_MOE_WARM_TRACE=1 GGML_SCHED_MOE_PREFETCH=2"
d() { echo "  $1 vs $2:"; $PY textdiff.py runs/$1.client.json runs/$2.client.json | sed 's/^/    /'; }
d ovl10_w0_off ovl10_w0_plan
echo "  warm-trace off vs plan (first differing observer call):"
grep -o "moe-warm-trace: .*" server_ovl10_tr_off.log  > ovl10_tr_off.txt
grep -o "moe-warm-trace: .*" server_ovl10_tr_plan.log > ovl10_tr_plan.txt
echo "    calls: $(wc -l < ovl10_tr_off.txt) vs $(wc -l < ovl10_tr_plan.txt) | bad ids off $(awk '{s+=$NF} END {print s}' ovl10_tr_off.txt) plan $(awk '{s+=$NF} END {print s}' ovl10_tr_plan.txt)"
diff <(nl ovl10_tr_off.txt) <(nl ovl10_tr_plan.txt) | head -8 | sed 's/^/    /'
echo OVL10_DONE

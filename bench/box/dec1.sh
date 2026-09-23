#!/bin/bash
# DEC1: decode after a long prompt in prefill mode (open item 1). promo1: specbench long decode 49.1 -> 45.3 / 47.0 t/s on the STABLE
# serving config (both rounds below), yet the 9,279-token run went 47.3 -> 49.8.
# Mechanism (code read): the cache's prompt warm-up counts the routing of large batches and re-ranks the slots at the next step().
# With -b 256 each 256-token batch is followed by a step() that resets the counts => the slots end prefill ranked by the LAST
# batch. In prefill mode step() returns early while the cache is suspended => counts pile up over the WHOLE prompt.
# Fix candidate (patches/warm-tail.patch, env-gated): LLAMA_MOE_WARM_TAIL=H weights each token's routing by 0.5^(age/H).
# Arms on the ov build (same binary; the patch is off unless the env is set), 3 interleaved rounds:
#   P  -b 2048 -ubp 2048                        (prefill mode, whole-prompt warm-up = STABLE today)
#   O  -b 256                                   (no prefill mode: last-batch warm-up)
#   T  -b 2048 -ubp 2048 LLAMA_MOE_WARM_TAIL=128 (prefill mode + tail-weighted warm-up)
# Then the 9,279-token prompt once per arm (-c 12288, cache 22) for the long end.
# Read: the hypothesis holds if long-prompt decode P < O beyond P's and O's spreads and T recovers to O; T becomes a STABLE
# candidate if it also leaves the short prompts unchanged (within spread).
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
SRC=/ai/src/llama.cpp-ov; OV=$SRC/build75
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf; PY=/ai/.venv/bin/python
echo "--- 0. BUILD (apply patches/warm-tail.patch to ov; idempotent) | $(date +%T)"
P=/ai/bench/patches/warm-tail.patch
if git -C $SRC apply -R --check $P 2>/dev/null; then echo "    already applied"; else git -C $SRC apply $P && echo "    applied" || { echo "DEC1_FAILED: patch"; exit 1; }; fi
cmake --build $OV --target llama-server -j6 > dec1_build.log 2>&1 || { grep error dec1_build.log | head; echo "DEC1_FAILED: build"; exit 1; }
strings $OV/bin/libllama.so* | grep -q LLAMA_MOE_WARM_TAIL || { echo "DEC1_FAILED: libllama lacks LLAMA_MOE_WARM_TAIL"; exit 1; }
export LD_LIBRARY_PATH=$OV/bin EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$OV
BASE="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128"
arm() { local l=$1 envs=$2; shift 2; sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  echo "##### $l | env: ${envs:-none} | $*"; env $envs GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $BASE "$@" 2>&1
  grep -hE "prompt warm-up|decode mode: restored" server_$l.log | tail -2 | cut -c1-150 | sed 's/^/    /'; }
echo "--- 1. SPECBENCH (3 rounds P | O | T) | $(date +%T)"
for r in a b c; do
  arm dec1_P_$r ""                      -b 2048 -ubp 2048
  arm dec1_O_$r ""                      -b 256
  arm dec1_T_$r "LLAMA_MOE_WARM_TAIL=128" -b 2048 -ubp 2048
done
$PY - <<'PY'
import json
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
R = {a: [L(f"dec1_{a}_{r}") for r in "abc"] for a in "POT"}
m = lambda a, k: sum(x[k]["decode_tps"] for x in R[a]) / 3
sp = lambda a, k: max(x[k]["decode_tps"] for x in R[a]) - min(x[k]["decode_tps"] for x in R[a])
for k in R["P"][0]:
    print(f"  {k:6s} decode P {m('P', k):6.2f} ({sp('P', k):.2f}) | O {m('O', k):6.2f} ({sp('O', k):.2f}) | T {m('T', k):6.2f} ({sp('T', k):.2f})"
          f" || P vs O {100 * (m('P', k) / m('O', k) - 1):+5.1f}% | T vs O {100 * (m('T', k) / m('O', k) - 1):+5.1f}% | T vs P {100 * (m('T', k) / m('P', k) - 1):+5.1f}%")
PY
echo "--- 2. 9,279-TOKEN PROMPT (-c 12288, cache 22) | $(date +%T)"
for a in P O T; do
  case $a in P) E=""; X="-b 4096 -ubp 2048";; O) E=""; X="-b 256";; T) E="LLAMA_MOE_WARM_TAIL=128"; X="-b 4096 -ubp 2048";; esac
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env $E GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 $OV/bin/llama-server -m $K2 -ngl 999 -fa on -c 12288 -t 6 \
    --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 -ot exps=CPU --moe-expert-cache 22 -md $HEAD \
    --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 $X > server_dec1_pf_$a.log 2>&1 &
  NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  echo -n "    [$a] "; python3 longpf.py dec1_pf_$a 40000 2>&1
  kill $NP; wait $NP 2>/dev/null
done
unset LD_LIBRARY_PATH
echo DEC1_DONE

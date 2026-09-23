#!/bin/bash
# DEC3: dec2 re-run on a clean box. dec2 (08:47) was contaminated: kcompactd0 took 54-75% of a core in 4 of 10 specbench arms,
# 3 of them T arms (T_b T_c T_d; P_d) -> the P vs T comparison is biased against T and is not evidence either way.
# Fix (preflight 08:50): vm.compaction_proactiveness=0, vm.watermark_boost_factor=0. Same arms, same pre-registered rule;
# additionally the verdict counts only if NO arm carries a FOREIGN CPU flag (else re-run, don't read).
#
# (dec2 header follows)
# DEC2 (follow-up to dec1, 08:11):
# dec1: in-build, prefill mode (P) vs the old -b 256 path (O): long 2.2k +2.9% (promo1's -6% did NOT reproduce), short prompts -3.2..-4.5%
# (inside spreads); tail-weighted warm-up (T) recovered code fully / reason partly, 9,279-token decode P 46.8 O 49.1 T 50.2 (1 run).
# This job: P vs T only, 5 interleaved rounds + the 9,279-token prompt x3 each. Pre-registered: T WINS if its 4-prompt mean decode
# beats P's by more than P's mean spread AND T >= P on the 9,279-token decode in >= 2 of 3 rounds; else NOT PROVEN (T stays off).
#
# (dec1 header follows)
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
if git -C $SRC apply -R --check $P 2>/dev/null; then echo "    already applied"; else git -C $SRC apply $P && echo "    applied" || { echo "DEC3_FAILED: patch"; exit 1; }; fi
cmake --build $OV --target llama-server -j6 > dec3_build.log 2>&1 || { grep error dec3_build.log | head; echo "DEC3_FAILED: build"; exit 1; }
strings $OV/bin/libllama.so* | grep -q LLAMA_MOE_WARM_TAIL || { echo "DEC3_FAILED: libllama lacks LLAMA_MOE_WARM_TAIL"; exit 1; }
export LD_LIBRARY_PATH=$OV/bin EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$OV
BASE="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128"
arm() { local l=$1 envs=$2; shift 2; sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  echo "##### $l | env: ${envs:-none} | $*"; env $envs GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $BASE "$@" 2>&1
  grep -hE "prompt warm-up|decode mode: restored" server_$l.log | tail -2 | cut -c1-150 | sed 's/^/    /'; }
echo "--- 1. SPECBENCH (5 rounds P | T) | $(date +%T)"
for r in a b c d e; do
  arm dec3_P_$r ""                        -b 2048 -ubp 2048
  arm dec3_T_$r "LLAMA_MOE_WARM_TAIL=128" -b 2048 -ubp 2048
done
echo "--- 2. 9,279-TOKEN PROMPT x3 (-c 12288, cache 22) | $(date +%T)"
for r in a b c; do for a in P T; do
  E=""; [ $a = T ] && E="LLAMA_MOE_WARM_TAIL=128"
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env $E GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 $OV/bin/llama-server -m $K2 -ngl 999 -fa on -c 12288 -t 6 \
    --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 -ot exps=CPU --moe-expert-cache 22 -md $HEAD \
    --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 4096 -ubp 2048 > server_dec3_pf_${a}_$r.log 2>&1 &
  NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  echo -n "    [$a $r] "; python3 longpf.py dec3_pf_${a}_$r 40000 2>&1
  kill $NP; wait $NP 2>/dev/null
done; done
unset LD_LIBRARY_PATH
$PY - <<'PY2'
import json
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
R = {a: [L(f"dec3_{a}_{r}") for r in "abcde"] for a in "PT"}
kinds = list(R["P"][0])
m = lambda a, k: sum(x[k]["decode_tps"] for x in R[a]) / 5
sp = lambda a, k: max(x[k]["decode_tps"] for x in R[a]) - min(x[k]["decode_tps"] for x in R[a])
for k in kinds:
    print(f"  {k:6s} P {m('P', k):6.2f} ({sp('P', k):.2f}) | T {m('T', k):6.2f} ({sp('T', k):.2f}) | T vs P {100 * (m('T', k) / m('P', k) - 1):+5.1f}%")
MP = sum(m("P", k) for k in kinds) / 4; MT = sum(m("T", k) for k in kinds) / 4; SP = sum(sp("P", k) for k in kinds) / 4
pf = {a: [json.load(open(f"/ai/bench/runs/dec3_pf_{a}_{r}.longpf.json"))["decode_tps"] for r in "abc"] for a in "PT"}
wins = sum(t >= p for p, t in zip(pf["P"], pf["T"]))
print(f"  9,279-token decode P {pf['P']} | T {pf['T']} -> T >= P in {wins}/3")
ok = (MT - MP) > SP and wins >= 2
print(f"  MEAN P {MP:.2f} | T {MT:.2f} ({100 * (MT / MP - 1):+.1f}%, P spread {SP:.2f}) -> {'T WINS' if ok else 'NOT PROVEN'}")
PY2
echo DEC3_DONE

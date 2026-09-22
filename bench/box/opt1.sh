#!/bin/bash
# OPT1: the untested zero-code levers on the promoted build (0007-0010), one job, base interleaved to catch drift.
# Facts behind each arm (2026-09-23):
#   t5      -t 5: the CPU miss phase is memory bound at 6 threads (cpu1bench2); a 6th spinning worker may only steal the core the
#           CUDA-driving / HTTP threads need (lat1 showed the spinning workers are load-bearing, never tested the count).
#   pin     OMP_PROC_BIND=close OMP_PLACES=cores: one worker per PHYSICAL core (siblings are 0/6 .. 5/11); today placement is luck.
#   nopin   GGML_CUDA_NO_PINNED=1: the 11.3 GB expert buffer comes from cudaMallocHost = 4 KB pages (AnonHugePages 172 MB of
#           ~11 GB RSS); a plain allocation gets THP 2 MB pages => fewer page walks in the memory-bound CPU phase. Confound:
#           cache uploads become pageable (lat1: pageable copies already run near the PCIe ceiling). The arm logs AnonHugePages.
#   mtp3    --spec-draft-n-max 3: MTP acceptance 0.84-0.98 and verify got cheaper with 0007-0009; n 2 was tuned before.
#   ub256   -ub 256 -b 512 at cache 26: prefill is ~45 s of the long prompt's 51 s wall; lat1's ub256 row ran at cache 28.
#   turns   multi-turn prefix reuse, thinking on (turns.py): how much of each turn's prompt is re-processed.
# Read: an arm is a win when its 4-prompt mean decode beats the base mean by more than the base spread (max-min of the 3
# bases) AND no prompt drops by more than that spread. Texts are not compared here (thread count / pinning / allocation do
# not change the math; mtp3 is lossless speculation; ub changes prefill batching only) — a winner gets its identity check
# under LLAMA_MOE_CACHE_SYNC=1 before it ships.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf; PY=/ai/.venv/bin/python
export BUILD=/ai/src/llama.cpp-mainline/build75 EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32
strings $BUILD/bin/libllama.so.0 | grep -q LLAMA_MOE_CACHE_SYNC || { echo "OPT1_REFUSED: served build is not the promoted one"; exit 1; }
ARGS="-ot exps=CPU --moe-expert-cache 26 -ub 128 -b 256 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2"
thp() { # sample the server's huge-page coverage once it is up and warm
  sleep 90; local p; p=$(pgrep -f "llama-server .*--port 8099" | head -1)
  [ -n "$p" ] && echo "    [$1] $(grep -E "^(Rss|Anonymous|AnonHugePages)" /proc/$p/smaps_rollup | tr -s ' ' | tr '\n' ' ')"
}
arm() { # label "ENV=.. ..." extra-args...
  local l=$1 envs=$2; shift 2
  echo "##### $l | env: ${envs:-none} | extra: $*"
  thp $l &
  env $envs ./specbench.sh 999 $l $ARGS "$@" 2>&1
  wait
  grep -hE "moe-cache: steps|statistics +draft|out of memory|compute buffer size" server_$l.log | tail -3 | cut -c1-180 | sed 's/^/    /'
}
arm opt1_base_a  ""
arm opt1_t5      ""                                    -t 5
arm opt1_pin     "OMP_PROC_BIND=close OMP_PLACES=cores"
arm opt1_nopin   "GGML_CUDA_NO_PINNED=1"
arm opt1_base_b  ""
arm opt1_mtp3    ""                                    --spec-draft-n-max 3
arm opt1_ub256   ""                                    -ub 256 -b 512
arm opt1_base_c  ""
echo "##### opt1_turns (multi-turn prefix reuse, thinking on)"
GGML_OP_OFFLOAD_MIN_BATCH=32 $BUILD/bin/llama-server -m $K2 -ngl 999 -fa on -c 16384 -t 6 --load-mode none --jinja --parallel 1 \
  --port 8099 --cache-ram 0 $ARGS > server_opt1_turns.log 2>&1 &
NP=$!
for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
python3 turns.py opt1_turns 2>&1 | sed 's/^/    /'
grep -hiE "checkpoint|erased invalidated|forcing full prompt|restored context" server_opt1_turns.log | tail -6 | cut -c1-180 | sed 's/^/    /'
kill $NP; wait $NP 2>/dev/null
echo "--- READ (decode t/s, 4-prompt mean vs the 3 bases; prefill t/s for the long prompt)"
$PY - <<'PY'
import json
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
bases = [L(b) for b in ("opt1_base_a", "opt1_base_b", "opt1_base_c")]
kinds = list(bases[0])
bmean = {k: sum(b[k]["decode_tps"] for b in bases) / 3 for k in kinds}
bspread = {k: max(b[k]["decode_tps"] for b in bases) - min(b[k]["decode_tps"] for b in bases) for k in kinds}
bm = sum(bmean.values()) / len(kinds)
bs = sum(bspread.values()) / len(kinds)
bpf = sum(b["long"]["prefill_tps"] for b in bases) / 3
print(f"  base      mean decode {bm:6.2f} (spread {bs:.2f}) | long prefill {bpf:6.1f}")
for arm in ("opt1_t5", "opt1_pin", "opt1_nopin", "opt1_mtp3", "opt1_ub256"):
    try:
        a = L(arm)
    except Exception as e:
        print(f"  {arm:9s} NO RESULT ({e})"); continue
    m = sum(a[k]["decode_tps"] for k in kinds) / len(kinds)
    worst = min(a[k]["decode_tps"] - bmean[k] + bspread[k] for k in kinds)
    win = (m - bm) > bs and worst >= 0
    per = " ".join(f"{k} {100 * (a[k]['decode_tps'] / bmean[k] - 1):+.1f}%" for k in kinds)
    print(f"  {arm:9s} mean decode {m:6.2f} ({100 * (m / bm - 1):+5.1f}%) | {per} | long prefill {a['long']['prefill_tps']:6.1f} "
          f"({100 * (a['long']['prefill_tps'] / bpf - 1):+.0f}%) | {'WIN' if win else 'no'}")
PY
echo OPT1_DONE

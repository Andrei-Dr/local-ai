#!/bin/bash
# HAND2: the three HAND1 levers stacked on the served build (llama.cpp-ov, branch shexp-overlap = moe-cache + 3 commits):
#   0007 the qwen35moe shared expert built INSIDE the MoE expert-cache split (runs on the GPU while the CPU does the misses);
#   0008 CUDA concat: flat kernel for narrow non-contiguous rows (delta-net conv state: 53 us per GDN layer in hand1's trace);
#   0009 ggml-backend sched: host->device split inputs enqueued on the split's stream (the 196 KB expert-output upload stops
#        blocking the host for the transfer before the ~20 us graph launch).
# hand1 per layer: 473 us CPU experts + 458 us serial GPU phase + 84 us handoff; the levers target ~2 + ~1.5 + ~0.7 ms of a
# ~58 ms round. All are bit-identical by construction (same kernels / data movement, reordered or re-launched).
# Gates, in order:
#   0. UNIT: test-backend-ops -o CONCAT, CUDA vs the CPU reference (incl. the transposed-b delta-net shapes); any FAIL stops.
#   1. IDENTITY: temp-0 texts byte-identical to base; base_a vs base_b is the determinism floor (base diverging => UNDECIDED).
#   2. MECHANISM: short nsys run per build, hand1_phases.py: device phase and short syncs must shrink.
#   3. SPEED: decode t/s ABAB at the prof2 config. GO = identity holds AND ov >= base +1.5% on the 4-prompt mean AND no prompt
#      below base by more than the base_a/base_b spread.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
BASE=/ai/src/llama.cpp-mainline/build75; OV=/ai/src/llama.cpp-ov/build75; PY=/ai/.venv/bin/python
for f in $K2 $HEAD $OV/bin/libllama.so.0 $OV/bin/test-backend-ops /ai/bench/hand1_phases.py /ai/bench/textdiff.py; do [ -e $f ] || { echo "HAND2_REFUSED: $f missing"; exit 1; }; done
[ "$(git -C /ai/src/llama.cpp-ov rev-parse --abbrev-ref HEAD)" = shexp-overlap ] || { echo "HAND2_REFUSED: ov tree not on shexp-overlap"; exit 1; }
echo "--- 0. UNIT (test-backend-ops CONCAT, CUDA0 vs CPU)"
$OV/bin/test-backend-ops -o CONCAT -b CUDA0 > hand2_unit.log 2>&1; urc=$?
grep -E "tests passed|FAIL" hand2_unit.log | tail -3 | sed 's/^/    /'
[ $urc -eq 0 ] && grep -q "tests passed" hand2_unit.log || { echo "HAND2_FAILED: unit"; exit 1; }
$OV/bin/test-backend-ops perf -o CONCAT -b CUDA0 2>&1 | grep -E "CONCAT" | cut -c1-150 | sed 's/^/    perf /'
ARGS="-ot exps=CPU --moe-expert-cache 26 -ub 128 -b 256 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32
arm() { # label build ldpath
  local l=$1; export BUILD=$2
  if [ -n "$3" ]; then export LD_LIBRARY_PATH=$3; else unset LD_LIBRARY_PATH; fi
  echo "##### $l | libllama: $(ldd $BUILD/bin/llama-server | grep -oE "(libllama|libggml-cuda|libggml-base)\.so\.0 => [^ ]+" | tr "\n" " ")"
  ./specbench.sh 999 $l $ARGS 2>&1
  grep -hE "moe-cache: steps|statistics +draft|out of memory" server_$l.log | tail -2 | cut -c1-200 | sed 's/^/    /'
}
arm hand2_base_a $BASE ""
arm hand2_ov_a   $OV   $OV/bin
arm hand2_base_b $BASE ""
arm hand2_ov_b   $OV   $OV/bin
unset LD_LIBRARY_PATH
echo "--- 1. IDENTITY (temp 0, byte-exact)"
echo "  floor base_a vs base_b:"; $PY textdiff.py runs/hand2_base_a.client.json runs/hand2_base_b.client.json | sed 's/^/    /'; floor=${PIPESTATUS[0]}
echo "  base_a vs ov_a:";         $PY textdiff.py runs/hand2_base_a.client.json runs/hand2_ov_a.client.json   | sed 's/^/    /'; id1=${PIPESTATUS[0]}
echo "  base_b vs ov_b:";         $PY textdiff.py runs/hand2_base_b.client.json runs/hand2_ov_b.client.json   | sed 's/^/    /'; id2=${PIPESTATUS[0]}
if [ $floor -ne 0 ]; then IDV=UNDECIDED; elif [ $id1 -eq 0 ] && [ $id2 -eq 0 ]; then IDV=IDENTICAL; else IDV=DIVERGES; fi
echo "  identity: $IDV"
echo "--- 2. MECHANISM (nsys API trace, graphs on, same config, GEN 300 on one prompt)"
for b in base ov; do
  if [ $b = ov ]; then BLD=$OV; export LD_LIBRARY_PATH=$OV/bin; else BLD=$BASE; unset LD_LIBRARY_PATH; fi
  rm -f hand2_nsys_$b.nsys-rep hand2_nsys_$b.sqlite
  GGML_OP_OFFLOAD_MIN_BATCH=32 nsys profile -t cuda --sample=none --cpuctxsw=none -f true -o /ai/bench/hand2_nsys_$b \
    $BLD/bin/llama-server -m $K2 -ngl 999 -fa on -c 4096 -t 6 --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 $ARGS \
    > server_hand2_nsys_$b.log 2>&1 &
  NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  EDIT=0 LONG=0 OUT=/tmp python3 specclient.py hand2_nsys_$b x > /dev/null 2>&1
  pkill -INT -f "llama-server .*--port 8099"; wait $NP 2>/dev/null; sleep 3
  # under the queue service nsys cannot find its own importer (it lives in /usr/lib/nsight-systems/host-linux-x64, the binary in
  # the target dir): it leaves a raw .qdstrm, which the importer turns into the report
  [ -s hand2_nsys_$b.nsys-rep ] || /usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i hand2_nsys_$b.qdstrm > /dev/null 2>&1
  nsys export --type sqlite -f true --output hand2_nsys_$b.sqlite hand2_nsys_$b.nsys-rep > /dev/null 2>&1
  echo "  $b:"; $PY hand1_phases.py hand2_nsys_$b.sqlite --skip-s 0 2>&1 | sed 's/^/    /'
done
unset LD_LIBRARY_PATH
echo "--- 3. SPEED (decode t/s per prompt: base_a base_b | ov_a ov_b)"
$PY - "$IDV" <<'PY'
import json, sys
L = lambda l: {r["prompt"]: r["decode_tps"] for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
ba, bb, oa, ob = (L(x) for x in ("hand2_base_a", "hand2_base_b", "hand2_ov_a", "hand2_ov_b"))
worst_ok, rb, ro = True, [], []
for k in ba:
    b, o = (ba[k] + bb[k]) / 2, (oa[k] + ob[k]) / 2
    spread = abs(ba[k] - bb[k])
    rb.append(b); ro.append(o)
    bad = o < b - spread
    worst_ok &= not bad
    print(f"  {k:6s} base {ba[k]:6.2f} {bb[k]:6.2f} | ov {oa[k]:6.2f} {ob[k]:6.2f} | {100 * (o / b - 1):+5.1f}%{'  BELOW base by more than the base spread' if bad else ''}")
gain = 100 * (sum(ro) / sum(rb) - 1)
idv = sys.argv[1]
verdict = "GO" if idv == "IDENTICAL" and gain >= 1.5 and worst_ok else ("NO-GO" if idv == "DIVERGES" or gain < 1.5 or not worst_ok else "UNDECIDED")
print(f"HAND2 mean decode {gain:+.1f}% | identity {idv} | verdict {verdict}")
PY
echo HAND2_DONE

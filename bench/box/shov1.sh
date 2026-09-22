#!/bin/bash
# SHOV1 (HAND1 lever 1): the qwen35moe shared expert built INSIDE the MoE expert-cache split, so it runs on the GPU while the
# CPU computes the expert misses instead of after them (llama.cpp-ov, branch shexp-overlap = moe-cache + 1 commit, libllama only).
# hand1 measured per layer 473 us CPU experts + 458 us serial GPU phase + 84 us handoff; the shared expert is ~55-60 us of the
# serial phase => expected ~2 ms of a ~58 ms round (~3-4% decode t/s).
# Gates, in order:
#   1. IDENTITY: same kernels on the same inputs, only reordered => temp-0 texts must be byte-identical to base. base_a vs base_b
#      is the determinism floor: if base itself diverges run to run, identity is UNDECIDED (not a pass).
#   2. MECHANISM: short nsys run of each build, hand1_phases.py: the device phase (sync after the B launch) must shrink.
#   3. SPEED: decode t/s, ABAB (base, ov, base, ov) at the prof2 config. GO = identity holds AND ov mean decode >= base +1.5% on
#      the 4-prompt mean AND no prompt below base by more than the base_a/base_b spread.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
BASE=/ai/src/llama.cpp-mainline/build75; OV=/ai/src/llama.cpp-ov/build75; PY=/ai/.venv/bin/python
for f in $K2 $HEAD $OV/bin/libllama.so.0 /ai/bench/hand1_phases.py /ai/bench/textdiff.py; do [ -e $f ] || { echo "SHOV1_REFUSED: $f missing"; exit 1; }; done
[ "$(git -C /ai/src/llama.cpp-ov rev-parse --abbrev-ref HEAD)" = shexp-overlap ] || { echo "SHOV1_REFUSED: ov tree not on shexp-overlap"; exit 1; }
ARGS="-ot exps=CPU --moe-expert-cache 26 -ub 128 -b 256 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32
arm() { # label build ldpath
  local l=$1; export BUILD=$2
  if [ -n "$3" ]; then export LD_LIBRARY_PATH=$3; else unset LD_LIBRARY_PATH; fi
  echo "##### $l | libllama: $(ldd $BUILD/bin/llama-server | grep -oE 'libllama\.so\.0 => [^ ]+')"
  ./specbench.sh 999 $l $ARGS 2>&1
  grep -hE "moe-cache: steps|statistics +draft|out of memory" server_$l.log | tail -2 | cut -c1-200 | sed 's/^/    /'
}
arm shov1_base_a $BASE ""
arm shov1_ov_a   $OV   $OV/bin
arm shov1_base_b $BASE ""
arm shov1_ov_b   $OV   $OV/bin
unset LD_LIBRARY_PATH
echo "--- 1. IDENTITY (temp 0, byte-exact)"
echo "  floor base_a vs base_b:"; $PY textdiff.py runs/shov1_base_a.client.json runs/shov1_base_b.client.json | sed 's/^/    /'; floor=${PIPESTATUS[0]}
echo "  base_a vs ov_a:";         $PY textdiff.py runs/shov1_base_a.client.json runs/shov1_ov_a.client.json   | sed 's/^/    /'; id1=${PIPESTATUS[0]}
echo "  base_b vs ov_b:";         $PY textdiff.py runs/shov1_base_b.client.json runs/shov1_ov_b.client.json   | sed 's/^/    /'; id2=${PIPESTATUS[0]}
if [ $floor -ne 0 ]; then IDV=UNDECIDED; elif [ $id1 -eq 0 ] && [ $id2 -eq 0 ]; then IDV=IDENTICAL; else IDV=DIVERGES; fi
echo "  identity: $IDV"
echo "--- 2. MECHANISM (nsys API trace, graphs on, same config, GEN 300 on one prompt)"
for b in base ov; do
  if [ $b = ov ]; then BLD=$OV; export LD_LIBRARY_PATH=$OV/bin; else BLD=$BASE; unset LD_LIBRARY_PATH; fi
  rm -f shov1_nsys_$b.nsys-rep shov1_nsys_$b.sqlite
  GGML_OP_OFFLOAD_MIN_BATCH=32 nsys profile -t cuda --sample=none --cpuctxsw=none -f true -o /ai/bench/shov1_nsys_$b \
    $BLD/bin/llama-server -m $K2 -ngl 999 -fa on -c 4096 -t 6 --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 $ARGS \
    > server_shov1_nsys_$b.log 2>&1 &
  NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  EDIT=0 LONG=0 OUT=/tmp python3 specclient.py shov1_nsys_$b x > /dev/null 2>&1
  pkill -INT -f "llama-server .*--port 8099"; wait $NP 2>/dev/null; sleep 3
  nsys export --type sqlite --output shov1_nsys_$b.sqlite shov1_nsys_$b.nsys-rep > /dev/null 2>&1
  echo "  $b:"; $PY hand1_phases.py shov1_nsys_$b.sqlite --skip-s 0 2>&1 | sed 's/^/    /'
done
unset LD_LIBRARY_PATH
echo "--- 3. SPEED (decode t/s per prompt: base_a base_b | ov_a ov_b)"
$PY - "$IDV" <<'PY'
import json, sys
L = lambda l: {r["prompt"]: r["decode_tps"] for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
ba, bb, oa, ob = (L(x) for x in ("shov1_base_a", "shov1_base_b", "shov1_ov_a", "shov1_ov_b"))
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
print(f"SHOV1 mean decode {gain:+.1f}% | identity {idv} | verdict {verdict}")
PY
echo SHOV1_DONE

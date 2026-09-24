#!/bin/bash
# SPEC3: the waterfall of a speculative step (research/spec-design-2026-09-24.md section 10): how much of each extra draft
# position is the serial MTP draft (GPU, CPU idle) and how much GPU idle the verify leaves, before building M1 (draft inside the
# verify's CPU window).
# TEST build = /ai/src/llama.cpp-t4 at spec3 11d392591 = spec2 e16838c6d (STABLE source + dump instrumentation + P1, inert at
#   n <= 3) + CUDA split timing (backend procs, off by default) + step timing in LLAMA_SPEC_DUMP: the draft phase and each depth's
#   llama_decode + synchronize, the MTP catch-up decode, the verify llama_decode + synchronize (host clock), and the verify's GPU
#   time = the summed stream time between CUDA events recorded at the start and end of every graph_compute of the verify (kernel
#   time plus launch gaps inside a split; the scheduler's input copies before a split are outside).
# PRE-REGISTERED (copied verbatim from the design doc, section 10, before any code):
#   spec3 (measure before building M1):
#   - Instrument (behind LLAMA_SPEC_DUMP): wall time of the draft phase per step and per depth, of the verify llama_decode, and the
#     GPU busy fraction inside the verify (CUDA events around the graph vs the host wait).
#   - Arms: STABLE n1 / n2 / n3, SYNC off, the 3 prompts, 400 tokens, 2 passes.
#   - H4: the draft phase is >= 30% of the marginal step cost per extra position (i.e. >= 2.8 ms of the ~9.2 ms per depth). If < 15%,
#     M1 is dead (nothing to hide); between: report and decide.
#   - H5: GPU idle inside the verify at n3 >= the n3 draft phase time (the window can hold the draft). If not, M1 hides only part.
# ARMS (STABLE config at -c 4096 via stable_server; the 3 prompts of spec2; greedy; 400 tokens; thinking off; -lv 4; logs kept):
#   R0 (identity, first, LLAMA_MOE_CACHE_SYNC=1): I_S3 = STABLE build n3, dumps off | I_T3 = TEST build n3, LLAMA_SPEC_DUMP on.
#      Token ids IDENTICAL on all 3 prompts AND draft_n / draft_n_accepted EXACTLY equal (spec0's R0 + R0b), else SPEC3 FAIL, stop:
#      the timed build must behave as STABLE (the dump makes the MTP draft sample on the CPU and adds synchronizes).
#   Timing arms, SYNC off, order A B C D D C B A (2 passes): A T1 | B T2 | C T3 = TEST build n 1 / 2 / 3 with LLAMA_SPEC_DUMP on
#      ("STABLE n1-n3" of the doc: the dump exists only on the TEST build; P1 is inert at n <= 3, spec2 R0 clause 1) | D S3 = STABLE
#      build n3, dumps off: T3 vs S3 decode t/s = the price of the instrumentation itself (printed, no rule).
# OPERATIONAL DEFINITIONS (bench/box/spec3_analyze.py, deployed as /ai/bench/spec3_analyze.py): per arm, means over the steps that
#   drafted exactly n tokens; marginal cost per position = least-squares slope over n = 1..3 of the mean step time (host time
#   between consecutive steps of a task) and of the mean draft phase; H4 ratio = draft slope / step slope; H5: at n3, mean(verify
#   wall - verify GPU) vs mean draft phase. Void: a dead server, OOM, a FOREIGN CPU flag, or an empty dump -> that pass is void;
#   H4 / H5 need every T arm with timed steps, else INCONCLUSIVE.
# RUNTIME (est.): incremental build ~5 min (ggml-cuda.cu, common, server) | R0 2 x ~1.5 = 3 | timing 8 x ~1.3 = 11 => ~20 min.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench || exit 1
source /ai/bench/stable.sh || { echo "SPEC3_FAILED: stable.sh"; exit 1; }
PY=/ai/.venv/bin/python
T4=/ai/src/llama.cpp-t4; B4=$T4/build75; SHA=11d392591
OUT=/ai/bench/runs/spec3; mkdir -p $OUT/logs
export STABLE_LOGDIR=$OUT/logs
GEN=400

echo "--- BUILD $SHA in $T4 | $(date +%T)"
[ -d $T4 ] || { echo "SPEC3_REFUSED: $T4 missing"; exit 1; }
[ -z "$(git -C $T4 status --porcelain --untracked-files=no)" ] || { echo "SPEC3_REFUSED: $T4 has local changes"; exit 1; }
if [ "$(git -C $T4 rev-parse --short=9 HEAD)" != "$SHA" ]; then
  git -C $T4 checkout -q --detach $SHA || { echo "SPEC3_REFUSED: cannot check out $SHA (is branch spec3 in the box repo?)"; exit 1; }
fi
[ -f $B4/CMakeCache.txt ] || { echo "SPEC3_REFUSED: $B4 not configured"; exit 1; }
cmake --build $B4 -j6 --target llama-server > spec3_build.log 2>&1 || { grep error spec3_build.log | head; echo "SPEC3_FAILED: build"; exit 1; }
[ "$(git -C $T4 rev-parse --short=9 HEAD)" = "$SHA" ] || { echo "SPEC3_REFUSED: $T4 moved off $SHA"; exit 1; }
cat $B4/bin/lib*.so* 2>/dev/null | grep -qa ggml_backend_cuda_split_timing_take || { echo "SPEC3_FAILED: build lacks the split timing"; exit 1; }
echo "    built $(git -C $T4 rev-parse --short=9 HEAD) | $(date +%T)"

voidlog() { grep -hoE 'could not re-allocate|out of memory' "$OUT/logs/server_$1.log" 2>/dev/null | head -1; }
health() { # pid: wait for /health, 0 = up
  local _
  for _ in $(seq 1 200); do curl -sf localhost:8099/health >/dev/null 2>&1 && return 0; kill -0 "$1" 2>/dev/null || return 1; sleep 2; done
  return 1
}
settle() { sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2; }
stop_server() { kill "$STABLE_PID" 2>/dev/null; wait "$STABLE_PID" 2>/dev/null; }

client() {
  $PY - "$1" "$OUT" "$GEN" <<'PY'
import json, sys, urllib.request
label, out, gen = sys.argv[1], sys.argv[2], int(sys.argv[3])
PROMPTS = [  # spec2.sh's three prompts, verbatim
    ("code",   "Write a Python class implementing an LRU cache with get and put in O(1), with type hints and a short docstring for each method."),
    ("reason", "A train leaves at 3pm going 60 mph. A second leaves the same station at 4pm going 80 mph on the same track. When does the second catch the first? Show the algebra step by step."),
    ("prose",  "Write an essay of about 600 words on how the printing press changed literacy, religion and science in Europe between 1450 and 1650."),
]
def post(path, body):
    req = urllib.request.Request("http://localhost:8099" + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))
def ids_of(q):
    prompt = post("/apply-template", {"messages": [{"role": "user", "content": q}], "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
    return post("/tokenize", {"content": prompt, "add_special": True, "parse_special": True})["tokens"]
post("/completion", {"prompt": ids_of(PROMPTS[0][1]), "n_predict": 16, "temperature": 0, "cache_prompt": False})  # warm-up
rows = []
for kind, q in PROMPTS:
    d = post("/completion", {"prompt": ids_of(q), "n_predict": gen, "temperature": 0, "return_tokens": True, "cache_prompt": False})
    t = d.get("timings", {})
    rows.append({"prompt": kind, "tokens": d["tokens"], "decode_tps": t.get("predicted_per_second"),
                 "draft_n": t.get("draft_n") or 0, "draft_n_accepted": t.get("draft_n_accepted") or 0})
    r = rows[-1]
    print(f"    {label} {kind}: {len(r['tokens'])} tok, {r['decode_tps']:.2f} t/s, draft {r['draft_n']} accepted {r['draft_n_accepted']}",
          flush=True)
json.dump(rows, open(f"{out}/{label}.json", "w"))
PY
}

# run LABEL BUILD N SYNC DUMP: one server + client; a void pass leaves no $OUT/LABEL.json
run() {
  local l=$1 b=$2 n=$3 sync=$4 dump=$5 v
  rm -f $OUT/$l.json $OUT/$l.jsonl runs/$l.mon.json; settle
  ( export BUILD=$b
    [ "$sync" = 1 ] && export LLAMA_MOE_CACHE_SYNC=1
    [ "$dump" = 1 ] && export LLAMA_SPEC_DUMP=$OUT/$l.jsonl
    stable_server 4096 $l --spec-draft-n-max $n -lv 4
    if health $STABLE_PID; then
      python3 mon.py start $l; client $l; python3 mon.py stop $l | grep FOREIGN
    else echo "    $l: SERVER DIED $(voidlog $l)"; fi
    stop_server )
  grep -hoE 'speculative dump -> .*' $OUT/logs/server_$l.log | head -1 | sed "s/^/    $l: /"
  v=$(voidlog $l)
  grep -q '"foreign_cpu_flag": true' runs/$l.mon.json 2>/dev/null && v="${v:+$v, }FOREIGN CPU"
  [ "$dump" = 1 ] && [ ! -s $OUT/$l.jsonl ] && v="${v:+$v, }empty dump"
  [ -n "$v" ] && { echo "    $l: VOID ($v)"; rm -f $OUT/$l.json; }
  return 0
}

echo "--- R0 IDENTITY (T = 0, SYNC=1): I_S3 STABLE n3 dumps off | I_T3 TEST n3 dump on | $(date +%T)"
run spec3_I_S3 "$STABLE_BUILD" 3 1 0
run spec3_I_T3 "$B4" 3 1 1
$PY - "$OUT" <<'PY' || { echo "SPEC3_VERDICT FAIL (R0)"; echo SPEC3_DONE; exit 0; }
import json, os, sys
out = sys.argv[1]
runs = {}
for a in ("S3", "T3"):
    p = f"{out}/spec3_I_{a}.json"
    if not os.path.exists(p):
        print(f"    R0 FAIL: identity arm {a} void"); sys.exit(1)
    runs[a] = {r["prompt"]: r for r in json.load(open(p))}
ok = True
for k, s in runs["S3"].items():
    t = runs["T3"].get(k, {"tokens": [], "draft_n": None, "draft_n_accepted": None})
    same_ids = s["tokens"] == t["tokens"]
    same_d = (s["draft_n"], s["draft_n_accepted"]) == (t["draft_n"], t["draft_n_accepted"])
    print(f"    R0 {k}: ids {'IDENTICAL' if same_ids else 'DIFFER'} ({len(s['tokens'])} tokens) | draft / accepted"
          f" {s['draft_n']} / {s['draft_n_accepted']} vs {t['draft_n']} / {t['draft_n_accepted']} {'EQUAL' if same_d else 'DIFFERENT'}")
    ok &= same_ids and same_d
sys.exit(0 if ok else 1)
PY

echo "--- TIMING (SYNC off): order A B C D D C B A (A T1, B T2, C T3 dump on; D S3 STABLE dump off) | $(date +%T)"
i=0
for a in A B C D D C B A; do
  i=$((i + 1))
  case $a in
    A) run spec3_T1_$i "$B4" 1 0 1;;
    B) run spec3_T2_$i "$B4" 2 0 1;;
    C) run spec3_T3_$i "$B4" 3 0 1;;
    D) run spec3_S3_$i "$STABLE_BUILD" 3 0 0;;
  esac
done

echo "--- ANALYSIS | $(date +%T)"
$PY /ai/bench/spec3_analyze.py $OUT --json $OUT/spec3_verdict.json 2>&1 | tee $OUT/spec3_verdict.txt | sed 's/^/  /'
echo SPEC3_DONE

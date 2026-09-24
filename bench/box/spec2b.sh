#!/bin/bash
# SPEC2B: diagnosis of spec2's R0 failure (research/spec-design-2026-09-24.md section 9): is TEST n4's greedy divergence from n3
# batch-shape rounding (a 5-token verify batch vs a 4-token one) or P1 (the expert-cache chain serving 5-token batches) computing
# wrong? Then, only if P1 clears, spec2's speed arms.
# BUILD: /ai/src/llama.cpp-t4 at spec2 e16838c6d (as spec2; rebuilt only if the tree moved).
# SUBSTITUTION (stated before the run): every arm runs on the TEST build, because the target top-10 logits come from the spec0
#   instrumentation (LLAMA_SPEC_DUMP), which STABLE's build lacks. The "STABLE" arms S1 / S2 / S3 are the TEST build at draft
#   n 1 / 2 / 3: their verify batches are 2 / 3 / 4 tokens, never above 4, so P1 is inert there (the cap only differs from the
#   old fixed 4 for batches of 5+), and spec2's R0 clause 1 measured TEST n3 = STABLE n3 token ids IDENTICAL on all 3 prompts.
#   T3 = the same build and flags as S3 (a determinism check); T4 = TEST n4 (5-token batches: P1 active).
# PRE-REGISTERED (copied verbatim from the design doc, section 9, before any code):
#   spec2b (T = 0, SYNC=1, the same 3 prompts, 400 tokens; LLAMA_SPEC_DUMP on for the target top-10 logits):
#   - D1 (does batch shape alone flip greedy tokens on STABLE?): STABLE n1, n2, n3 pairwise token ids. Batches 2 / 3 / 4, all
#     inside the unpatched cache window.
#   - D2 (does P1 add error beyond batch shape?): per prompt, over the verify positions both runs computed before their first
#     divergence, max |delta logit| over the target's top-10 between (a) STABLE n2 vs STABLE n3 and (b) TEST n4 vs TEST n3; and
#     the target's top-1 minus top-2 logit margin at each first-divergence position.
#   - Verdict: P1 is exact-equivalent ("batch-shape class") if D1 shows at least one flip OR every TEST n4 flip sits at a margin
#     <= the largest margin at which a D1 flip or (a) difference occurs, AND max|delta|(b) <= 2 x max|delta|(a). Otherwise P1 is
#     SUSPECT and goes to a KLD-vs-Q6 test before any speed claim.
#   - If P1 clears: rerun spec2's speed arms (R1-R3 unchanged) with R0 reduced to its first clause (TEST n3 = STABLE n3).
# OPERATIONAL DEFINITIONS: bench/box/spec2b_analyze.py (deployed as /ai/bench/spec2b_analyze.py) — which positions count, how a
#   flip's margin is taken (the larger of the two runs' margins), and the reading of "(a) difference" when D1 has no flip
#   (max|delta|(a)). Every arm: LLAMA_MOE_CACHE_SYNC=1, LLAMA_SPEC_DUMP on (it makes the MTP draft sample on the CPU; spec0's R0b
#   measured identical draft / accepted counts either way), -lv 4 (the log shows the cache-chain cap), server logs kept in
#   runs/spec2b/logs. Void: a dead server, OOM, a FOREIGN CPU flag or an empty dump -> SPEC2B INCONCLUSIVE, stop.
# SPEED PHASE (only after "BATCH-SHAPE CLASS"): SPEC2_R0=clause1 SPEC2_OUT=runs/spec2b/speed bash spec2.sh = spec2's arms, order,
#   R1-R3 verbatim, R0 = TEST n3 vs STABLE n3 only.
# RUNTIME (est.): diagnosis 5 x ~1.3 = 7 min | speed phase (if it runs) R0 2 x ~1.5 + 14 x ~1.2 = 20 min => ~8 or ~28 min.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench || exit 1
source /ai/bench/stable.sh || { echo "SPEC2B_FAILED: stable.sh"; exit 1; }
PY=/ai/.venv/bin/python
T4=/ai/src/llama.cpp-t4; B4=$T4/build75; SHA=e16838c6d
OUT=/ai/bench/runs/spec2b; mkdir -p $OUT/logs
export STABLE_LOGDIR=$OUT/logs
GEN=400

echo "--- BUILD $SHA in $T4 | $(date +%T)"
[ -d $T4 ] || { echo "SPEC2B_REFUSED: $T4 missing"; exit 1; }
[ -z "$(git -C $T4 status --porcelain --untracked-files=no)" ] || { echo "SPEC2B_REFUSED: $T4 has local changes"; exit 1; }
if [ "$(git -C $T4 rev-parse --short=9 HEAD)" != "$SHA" ]; then
  git -C $T4 checkout -q --detach $SHA || { echo "SPEC2B_REFUSED: cannot check out $SHA"; exit 1; }
fi
cmake --build $B4 -j6 --target llama-server > spec2b_build.log 2>&1 || { grep error spec2b_build.log | head; echo "SPEC2B_FAILED: build"; exit 1; }
[ "$(git -C $T4 rev-parse --short=9 HEAD)" = "$SHA" ] || { echo "SPEC2B_REFUSED: $T4 moved off $SHA"; exit 1; }
cat $B4/bin/lib*.so* 2>/dev/null | grep -qa ggml_backend_cuda_mmvq_mmid_max_batch || { echo "SPEC2B_FAILED: build lacks P1"; exit 1; }
cat $B4/bin/lib*.so* 2>/dev/null | grep -qa LLAMA_SPEC_DUMP || { echo "SPEC2B_FAILED: build lacks the dump"; exit 1; }
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

echo "--- DIAGNOSIS (T = 0, SYNC=1, dump on, TEST build): S1 S2 S3 T3 T4 | $(date +%T)"
for a in S1:1 S2:2 S3:3 T3:3 T4:4; do
  arm=${a%%:*}; n=${a##*:}; l=spec2b_$arm
  rm -f $OUT/$l.json $OUT/$l.jsonl runs/$l.mon.json; settle
  ( export BUILD=$B4 LLAMA_MOE_CACHE_SYNC=1 LLAMA_SPEC_DUMP=$OUT/$l.jsonl
    stable_server 4096 $l --spec-draft-n-max $n -lv 4
    if health $STABLE_PID; then
      python3 mon.py start $l; client $l; python3 mon.py stop $l | grep FOREIGN
    else echo "    $l: SERVER DIED $(voidlog $l)"; fi
    stop_server )
  echo "    $l: cache chain $(grep -oE 'serves batches of up to [0-9]+-[0-9]+ tokens' $OUT/logs/server_$l.log | head -1 || true)"
  v=$(voidlog $l)
  grep -q '"foreign_cpu_flag": true' runs/$l.mon.json 2>/dev/null && v="${v:+$v, }FOREIGN CPU"
  [ -s $OUT/$l.json ] || v="${v:+$v, }no client output"
  [ -s $OUT/$l.jsonl ] || v="${v:+$v, }empty dump"
  [ -n "$v" ] && { echo "    $l: VOID ($v)"; echo "SPEC2B_VERDICT INCONCLUSIVE (void arm $arm)"; echo SPEC2B_DONE; exit 0; }
done

echo "--- D1 / D2 + VERDICT | $(date +%T)"
$PY /ai/bench/spec2b_analyze.py $OUT --json $OUT/spec2b_verdict.json 2>&1 | tee $OUT/spec2b_verdict.txt | sed 's/^/  /'
if grep -q "SPEC2B_VERDICT BATCH-SHAPE CLASS" $OUT/spec2b_verdict.txt; then
  echo "--- P1 CLEARS: spec2 speed arms, R0 = clause 1 (TEST n3 vs STABLE n3) | $(date +%T)"
  SPEC2_R0=clause1 SPEC2_OUT=$OUT/speed bash /ai/bench/spec2.sh 2>&1 | sed 's/^/  /'
else
  echo "    P1 not cleared (SUSPECT or no verdict): stop; next = the KLD-vs-Q6 test (design section 9)"
fi
echo SPEC2B_DONE

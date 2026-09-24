#!/bin/bash
# SPEC2: speculation Phase 2 = direct box measurement (research/spec-design-2026-09-24.md section 8), replacing the simulator gate.
# TEST build = /ai/src/llama.cpp-t4 at spec2 e16838c6d = spec0 7488986d5 (STABLE source + the dump instrumentation, off unless its
#   env switches are set) + P1: the expert-cache chain's token cap = the device's MMVQ mul_mat_id window for the layer's expert
#   types, clamped to [4, 8] (was a fixed 4; K2q6 on Turing: Q2_K 7, Q3_K 5 -> 5), so a 5-token verify (n = 4) keeps the cache.
# PRE-REGISTERED (copied verbatim from the design doc, section 8, before any code or job existed):
#   - **P1 patch (0034 candidate):** the cache chain's token cap = min over the layer's expert tensors of
#     get_mmvq_mmid_max_batch(type, cc) (host-side query), capped at 8; unchanged behavior for batches <= 4.
#   - Arms, STABLE config at -c 4096, specbench code + reason + the spec0 prose prompt, greedy 400 tokens, order ABC..CBA,
#     2 passes: S3 = STABLE n3 (baseline); S2 = STABLE n2; P3a / P3b = STABLE n3 --spec-draft-p-min 0.6 / 0.75;
#     T3 = TEST (P1) n3; T4 = TEST n4; T4p = TEST n4 --spec-draft-p-min 0.6.
#   - R0 identity: TEST vs STABLE at n3, LLAMA_MOE_CACHE_SYNC=1, T = 0, token ids IDENTICAL on all prompts (batches <= 4 must be
#     untouched); and T4 vs S3 token ids IDENTICAL (speculation length cannot change greedy output; a difference = the batch-5
#     cache chain computes wrong) -> else FAIL and stop.
#   - R1 (P1 worth it): T4 or T4p beats S3 by more than the S3 spread on the 3-prompt mean decode t/s, AND T3 is within the S3
#     spread of S3 (no regression at n3).
#   - R2 (confidence stop): P3a or P3b beats S3 by more than the S3 spread on the 3-prompt mean -> recommend that p-min (a
#     serving flag: Andrei's call via STABLE promotion rules). Prose reported separately (its 44% acceptance is where a stop
#     should pay).
#   - R3 (depth): S2 vs S3 reported; no rule (informational for the draft-length default).
# OPERATIONAL DEFINITIONS:
#   - R0 runs first, on its own servers with LLAMA_MOE_CACHE_SYNC=1 (I_S3 = STABLE n3, I_T3 = TEST n3, I_T4 = TEST n4); the speed
#     arms run without SYNC. A void identity arm = FAIL (an identity test never passes by default).
#   - Speed order A..G G..A with A S3, B S2, C P3a, D P3b, E T3, F T4, G T4p (14 servers, one pass = one server per arm).
#     Per server: one unrecorded 16-token warm-up, then the 3 prompts; an arm's value = the 3-prompt mean decode t/s of one pass;
#     mean = over its 2 passes; spread(S3) = |S3 pass 1 - S3 pass 2|.
#     "beats S3 by more than the S3 spread" = mean(X) - mean(S3) > spread(S3); "T3 within the S3 spread" = mean(T3) >=
#     mean(S3) - spread(S3).
#   - Void (as spec1cal): a dead server, OOM in its log, or a mon.py FOREIGN CPU flag -> that pass is void; a rule that needs a
#     void pass is INCONCLUSIVE, never a pass.
#   - Also printed (no rule): the cache chain cap each build logs (TEST must say 5 for K2q6), and whether the speed arms' greedy
#     token ids agree with S3 (S2 vs S3 separates "a different verify batch shape changes greedy output" from an R0 T4 failure).
# RUNTIME (est.): incremental build ~6 min (ggml-cuda.cu + libllama) | identity 3 x ~1.5 = 5 | speed 14 x ~1.2 = 17 => ~28 min.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench || exit 1
source /ai/bench/stable.sh || { echo "SPEC2_FAILED: stable.sh"; exit 1; }
PY=/ai/.venv/bin/python
SRC=/ai/src/llama.cpp-mainline; T4=/ai/src/llama.cpp-t4; B4=$T4/build75; SHA=e16838c6d
OUT=/ai/bench/runs/spec2; mkdir -p $OUT
GEN=400

echo "--- BUILD $SHA in $T4 | $(date +%T)"
[ -d $T4 ] || { echo "SPEC2_REFUSED: $T4 missing (spec0 creates it)"; exit 1; }
[ -z "$(git -C $T4 status --porcelain --untracked-files=no)" ] || { echo "SPEC2_REFUSED: $T4 has local changes"; exit 1; }
if [ "$(git -C $T4 rev-parse --short=9 HEAD)" != "$SHA" ]; then
  git -C $T4 checkout -q --detach $SHA || { echo "SPEC2_REFUSED: cannot check out $SHA (is branch spec2 in $SRC?)"; exit 1; }
fi
[ -f $B4/CMakeCache.txt ] || { echo "SPEC2_REFUSED: $B4 not configured"; exit 1; }
cmake --build $B4 -j6 --target llama-server > spec2_build.log 2>&1 || { grep error spec2_build.log | head; echo "SPEC2_FAILED: build"; exit 1; }
[ "$(git -C $T4 rev-parse --short=9 HEAD)" = "$SHA" ] || { echo "SPEC2_REFUSED: $T4 moved off $SHA"; exit 1; }
cat $B4/bin/lib*.so* 2>/dev/null | grep -qa ggml_backend_cuda_mmvq_mmid_max_batch || { echo "SPEC2_FAILED: build lacks P1"; exit 1; }
echo "    built $(git -C $T4 rev-parse --short=9 HEAD) | $(date +%T)"

voidlog() { grep -hoE 'could not re-allocate|out of memory' "server_$1.log" 2>/dev/null | head -1; }
health() { # pid: wait for /health, 0 = up
  local _
  for _ in $(seq 1 200); do curl -sf localhost:8099/health >/dev/null 2>&1 && return 0; kill -0 "$1" 2>/dev/null || return 1; sleep 2; done
  return 1
}
settle() { sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2; }
stop_server() { kill "$STABLE_PID" 2>/dev/null; wait "$STABLE_PID" 2>/dev/null; }

# client LABEL: warm-up, then code + reason + prose, greedy, GEN tokens, via /completion on the templated prompt ids (spec0's way:
# exact token ids). Writes $OUT/LABEL.json.
client() {
  $PY - "$1" "$OUT" "$GEN" <<'PY'
import json, sys, urllib.request
label, out, gen = sys.argv[1], sys.argv[2], int(sys.argv[3])
PROMPTS = [  # code + reason = bench/box/specclient.py, verbatim; prose = spec0's prose prompt, verbatim
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
                 "predicted_n": t.get("predicted_n"), "draft_n": t.get("draft_n") or 0,
                 "draft_n_accepted": t.get("draft_n_accepted") or 0})
    r = rows[-1]
    print(f"    {label} {kind}: {r['predicted_n']} tok, {r['decode_tps']:.2f} t/s, draft {r['draft_n']} accepted {r['draft_n_accepted']}",
          flush=True)
json.dump(rows, open(f"{out}/{label}.json", "w"))
PY
}

# run LABEL BUILD N PMIN SYNC: one server + client; void -> $OUT/LABEL.json removed
run() {
  local l=$1 b=$2 n=$3 pmin=$4 sync=$5 v
  rm -f $OUT/$l.json runs/$l.mon.json; settle
  ( export BUILD=$b
    [ "$sync" = 1 ] && export LLAMA_MOE_CACHE_SYNC=1
    # shellcheck disable=SC2046  # optional flag pair
    stable_server 4096 $l --spec-draft-n-max $n $([ -n "$pmin" ] && echo --spec-draft-p-min $pmin)
    if health $STABLE_PID; then
      python3 mon.py start $l; client $l; python3 mon.py stop $l | grep FOREIGN
    else echo "    $l: SERVER DIED $(voidlog $l)"; fi
    stop_server )
  echo "    $l: cache chain $(grep -o 'serves batches of up to [0-9-]* tokens' server_$l.log | head -1 || true)"
  v=$(voidlog $l)
  grep -q '"foreign_cpu_flag": true' runs/$l.mon.json 2>/dev/null && v="${v:+$v, }FOREIGN CPU"
  [ -n "$v" ] && { echo "    $l: VOID ($v)"; rm -f $OUT/$l.json; }
  return 0
}

echo "--- R0 IDENTITY (T = 0, LLAMA_MOE_CACHE_SYNC=1): I_S3 STABLE n3 | I_T3 TEST n3 | I_T4 TEST n4 | $(date +%T)"
run spec2_I_S3 "$STABLE_BUILD" 3 "" 1
run spec2_I_T3 "$B4" 3 "" 1
run spec2_I_T4 "$B4" 4 "" 1
$PY - "$OUT" <<'PY' || { echo "SPEC2_VERDICT FAIL (R0)"; echo SPEC2_DONE; exit 0; }
import json, os, sys
out = sys.argv[1]
runs = {}
for a in ("S3", "T3", "T4"):
    p = f"{out}/spec2_I_{a}.json"
    if not os.path.exists(p):
        print(f"    R0 FAIL: identity arm {a} void"); sys.exit(1)
    runs[a] = {r["prompt"]: r["tokens"] for r in json.load(open(p))}
ok = True
for x, y in (("S3", "T3"), ("S3", "T4")):
    for k, a in runs[x].items():
        b = runs[y].get(k, [])
        if a == b:
            print(f"    R0 {y} vs {x} {k}: IDENTICAL ({len(a)} tokens)")
        else:
            i = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
            print(f"    R0 FAIL {y} vs {x} {k}: token ids differ at {i} ({len(a)} vs {len(b)} tokens)"); ok = False
sys.exit(0 if ok else 1)
PY

echo "--- SPEED: order A..G G..A (A S3, B S2, C P3a, D P3b, E T3, F T4, G T4p) | $(date +%T)"
i=0
for a in A B C D E F G G F E D C B A; do
  i=$((i + 1))
  case $a in
    A) run spec2_S3_$i  "$STABLE_BUILD" 3 ""   0;;
    B) run spec2_S2_$i  "$STABLE_BUILD" 2 ""   0;;
    C) run spec2_P3a_$i "$STABLE_BUILD" 3 0.6  0;;
    D) run spec2_P3b_$i "$STABLE_BUILD" 3 0.75 0;;
    E) run spec2_T3_$i  "$B4"           3 ""   0;;
    F) run spec2_T4_$i  "$B4"           4 ""   0;;
    G) run spec2_T4p_$i "$B4"           4 0.6  0;;
  esac
done

echo "--- VERDICTS | $(date +%T)"
$PY - "$OUT" <<'PY'
import glob, json, os, re, statistics as st, sys
out = sys.argv[1]
arms = {}
for f in sorted(glob.glob(f"{out}/spec2_*_*.json")):
    m = re.match(r"spec2_(S3|S2|P3a|P3b|T3|T4|T4p)_(\d+)\.json$", os.path.basename(f))
    if m:
        arms.setdefault(m.group(1), []).append(json.load(open(f)))
def mean3(run):
    return st.mean(r["decode_tps"] for r in run)
def row(a):
    runs = arms.get(a, [])
    return [round(mean3(r), 2) for r in runs] if len(runs) == 2 else None
def prose(a):
    return [round(next(r["decode_tps"] for r in run if r["prompt"] == "prose"), 2) for run in arms.get(a, [])]
for a in ("S3", "S2", "P3a", "P3b", "T3", "T4", "T4p"):
    runs = arms.get(a, [])
    acc = [f"{sum(r['draft_n_accepted'] for r in run)}/{sum(r['draft_n'] for r in run)}" for run in runs]
    print(f"  {a:4s} 3-prompt mean t/s per pass {row(a) or [round(mean3(r), 2) for r in runs]} ({len(runs)} of 2 passes)"
          f" | prose {prose(a)} | accepted/drafted {acc}")
s3 = row("S3")
if s3 is None:
    print("  R1 / R2 / R3: INCONCLUSIVE (S3 void)"); sys.exit(0)
m3, spread = st.mean(s3), abs(s3[0] - s3[1])
print(f"  S3 mean {m3:.2f} t/s, spread {spread:.2f}")
def beats(a):
    r = row(a)
    if r is None:
        return None
    return st.mean(r) - m3 > spread
def gain(a):
    r = row(a)
    return "void" if r is None else f"{st.mean(r):.2f} t/s ({100 * (st.mean(r) / m3 - 1):+.1f}%)"
b4, b4p, t3 = beats("T4"), beats("T4p"), row("T3")
if t3 is None or (b4 is None and b4p is None):
    r1 = "INCONCLUSIVE (void pass)"
else:
    no_reg = st.mean(t3) >= m3 - spread
    r1 = "PASS (P1 worth it)" if (b4 or b4p) and no_reg else f"FAIL (T4/T4p beat S3 by more than the spread: {bool(b4 or b4p)}; T3 no regression: {no_reg})"
print(f"  R1 P1: T4 {gain('T4')} | T4p {gain('T4p')} | T3 {gain('T3')} -> {r1}")
ba, bb = beats("P3a"), beats("P3b")
if ba is None and bb is None:
    r2 = "INCONCLUSIVE (void pass)"
else:
    win = [n for n, b in (("p-min 0.6", ba), ("p-min 0.75", bb)) if b]
    r2 = f"PASS: recommend {' / '.join(win)} (a serving flag: Andrei's call)" if win else "FAIL (neither beats S3 by more than the spread)"
print(f"  R2 confidence stop: P3a {gain('P3a')} | P3b {gain('P3b')} -> {r2}")
print(f"  R2 prose only: S3 {prose('S3')} | P3a {prose('P3a')} | P3b {prose('P3b')} | T4p {prose('T4p')}")
print(f"  R3 depth (no rule): S2 {gain('S2')} vs S3 {m3:.2f}")
base = {r["prompt"]: r["tokens"] for r in arms["S3"][0]}
for a in ("S2", "P3a", "P3b", "T3", "T4", "T4p"):
    for run in arms.get(a, [])[:1]:
        same = [k for k in base if next(r["tokens"] for r in run if r["prompt"] == k) == base[k]]
        print(f"  (info) greedy ids {a} vs S3 pass 1: identical on {len(same)} of {len(base)} prompts {same}")
PY
echo SPEC2_DONE

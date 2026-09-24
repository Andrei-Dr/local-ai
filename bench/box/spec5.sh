#!/bin/bash
# SPEC5: P2 = skip the MoE cache chain's dummy-slot mat-vecs (research/spec-design-2026-09-24.md section 12).
# PRE-REGISTERED (copied from section 12, 2026-09-25, before the run):
# - R0 identity (T = 0, SYNC=1, the 3 spec2 prompts, 400 tokens): TEST vs STABLE at n3 AND at n2: token ids IDENTICAL and
#   draft / accepted counts EQUAL, else FAIL and stop.
# - Speed (no SYNC; order A B C D D C B A; A S3 = STABLE n3, B T3 = TEST n3, C S2 = STABLE n2, D T2 = TEST n2): 3-prompt mean
#   decode t/s per pass. R1: P2 WINS if T3 - S3 > spread(S3) OR T2 - S2 > spread(S2), AND neither TEST arm is below its STABLE arm
#   by more than that arm's spread. Otherwise NOT PROVEN.
# - Report: T2 vs T3.
# TEST = /ai/src/llama.cpp-t4 at ba0d5f708 (branch spec5 = spec3 + P2). Void: dead server, OOM, FOREIGN CPU.
# RUNTIME (est.): incremental build ~6 min (ggml-cuda.cu + libllama) | identity 4 x ~1.5 = 6 | speed 8 x ~1.2 = 10 => ~22 min.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench || exit 1
source /ai/bench/stable.sh || { echo "SPEC5_FAILED: stable.sh"; exit 1; }
PY=/ai/.venv/bin/python
SRC=/ai/src/llama.cpp-mainline; T4=/ai/src/llama.cpp-t4; B4=$T4/build75; SHA=ba0d5f708
OUT=/ai/bench/runs/spec5; mkdir -p $OUT/logs
export STABLE_LOGDIR=$OUT/logs  # server logs kept with the results
GEN=400

echo "--- BUILD $SHA in $T4 | $(date +%T)"
[ -d $T4 ] || { echo "SPEC5_REFUSED: $T4 missing (spec0 creates it)"; exit 1; }
[ -z "$(git -C $T4 status --porcelain --untracked-files=no)" ] || { echo "SPEC5_REFUSED: $T4 has local changes"; exit 1; }
if [ "$(git -C $T4 rev-parse --short=9 HEAD)" != "$SHA" ]; then
  git -C $T4 checkout -q --detach $SHA || { echo "SPEC5_REFUSED: cannot check out $SHA (is branch spec5 in $SRC?)"; exit 1; }
fi
[ -f $B4/CMakeCache.txt ] || { echo "SPEC5_REFUSED: $B4 not configured"; exit 1; }
cmake --build $B4 -j6 --target llama-server > spec5_build.log 2>&1 || { grep error spec5_build.log | head; echo "SPEC5_FAILED: build"; exit 1; }
[ "$(git -C $T4 rev-parse --short=9 HEAD)" = "$SHA" ] || { echo "SPEC5_REFUSED: $T4 moved off $SHA"; exit 1; }
cat $B4/bin/lib*.so* 2>/dev/null | grep -qa ggml_backend_cuda_mmvq_mmid_max_batch || { echo "SPEC5_FAILED: build lacks P1"; exit 1; }
echo "    built $(git -C $T4 rev-parse --short=9 HEAD) | $(date +%T)"

voidlog() { grep -hoE 'could not re-allocate|out of memory' "$OUT/logs/server_$1.log" 2>/dev/null | head -1; }
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
    # -lv 4: llama.cpp's own INFO lines (the cache-chain cap) only reach the log at verbosity 4, as in specbench.sh
    stable_server 4096 $l --spec-draft-n-max $n $([ -n "$pmin" ] && echo --spec-draft-p-min $pmin) -lv 4
    if health $STABLE_PID; then
      python3 mon.py start $l; client $l; python3 mon.py stop $l | grep FOREIGN
    else echo "    $l: SERVER DIED $(voidlog $l)"; fi
    stop_server )
  echo "    $l: cache chain $(grep -oE 'serves batches of up to [0-9]+-[0-9]+ tokens' $OUT/logs/server_$l.log | head -1 || true)"
  v=$(voidlog $l)
  grep -q '"foreign_cpu_flag": true' runs/$l.mon.json 2>/dev/null && v="${v:+$v, }FOREIGN CPU"
  [ -n "$v" ] && { echo "    $l: VOID ($v)"; rm -f $OUT/$l.json; }
  return 0
}

echo "--- R0 IDENTITY (T = 0, LLAMA_MOE_CACHE_SYNC=1): STABLE vs TEST at n3 and n2 | $(date +%T)"
run spec5_I_S3 "$STABLE_BUILD" 3 "" 1
run spec5_I_T3 "$B4" 3 "" 1
run spec5_I_S2 "$STABLE_BUILD" 2 "" 1
run spec5_I_T2 "$B4" 2 "" 1
$PY - "$OUT" <<'PY' || { echo "SPEC5_VERDICT FAIL (R0)"; echo SPEC5_DONE; exit 0; }
import json, os, sys
out = sys.argv[1]; ok = True
for s, t in (("S3", "T3"), ("S2", "T2")):
    runs = {}
    for a in (s, t):
        p = f"{out}/spec5_I_{a}.json"
        if not os.path.exists(p):
            print(f"    R0 FAIL: identity arm {a} void"); sys.exit(1)
        runs[a] = {r["prompt"]: r for r in json.load(open(p))}
    for k, x in runs[s].items():
        y = runs[t].get(k)
        same = y is not None and x["tokens"] == y["tokens"]
        cnt = y is not None and (x["draft_n"], x["draft_n_accepted"]) == (y["draft_n"], y["draft_n_accepted"])
        print(f"    R0 {t} vs {s} {k}: ids {'IDENTICAL' if same else 'DIFFER'} | draft / accepted "
              f"{x['draft_n']} / {x['draft_n_accepted']} vs {y['draft_n'] if y else '-'} / {y['draft_n_accepted'] if y else '-'} "
              f"{'EQUAL' if cnt else 'DIFFER'}")
        ok = ok and same and cnt
sys.exit(0 if ok else 1)
PY

echo "--- SPEED: order A B C D D C B A (A S3, B T3, C S2, D T2) | $(date +%T)"
i=0
for a in A B C D D C B A; do
  i=$((i + 1))
  case $a in
    A) run spec5_S3_$i "$STABLE_BUILD" 3 "" 0;;
    B) run spec5_T3_$i "$B4"           3 "" 0;;
    C) run spec5_S2_$i "$STABLE_BUILD" 2 "" 0;;
    D) run spec5_T2_$i "$B4"           2 "" 0;;
  esac
done

echo "--- VERDICTS | $(date +%T)"
$PY - "$OUT" <<'PY'
import glob, json, os, re, statistics as st, sys
out = sys.argv[1]; arms = {}
for f in sorted(glob.glob(f"{out}/spec5_*_*.json")):
    m = re.match(r"spec5_(S3|T3|S2|T2)_(\d+)\.json$", os.path.basename(f))
    if m:
        arms.setdefault(m.group(1), []).append(json.load(open(f)))
def row(a):
    runs = arms.get(a, [])
    return [round(st.mean(r["decode_tps"] for r in run), 2) for run in runs] if len(runs) == 2 else None
for a in ("S3", "T3", "S2", "T2"):
    print(f"  {a} 3-prompt mean t/s per pass {row(a)} | per prompt pass 1 "
          f"{[(r['prompt'], round(r['decode_tps'], 2)) for r in arms[a][0]] if arms.get(a) else '-'}")
pairs = []
for s, t in (("S3", "T3"), ("S2", "T2")):
    rs, rt = row(s), row(t)
    if rs is None or rt is None:
        pairs.append(None); print(f"  {t} vs {s}: INCONCLUSIVE (void pass)"); continue
    ms, mt, sp = st.mean(rs), st.mean(rt), abs(rs[0] - rs[1])
    pairs.append((mt - ms > sp, mt >= ms - sp))
    print(f"  {t} vs {s}: {mt:.2f} vs {ms:.2f} t/s ({100 * (mt / ms - 1):+.1f}%), spread {sp:.2f}")
if None in pairs:
    print("  R1: INCONCLUSIVE")
else:
    win = any(w for w, _ in pairs) and all(n for _, n in pairs)
    print(f"  R1 P2: {'WINS' if win else 'NOT PROVEN'} (beats beyond spread: {[w for w, _ in pairs]}; no regression: {[n for _, n in pairs]})")
t2, t3 = row("T2"), row("T3")
if t2 and t3:
    print(f"  (report) T2 vs T3: {st.mean(t2):.2f} vs {st.mean(t3):.2f} t/s")
PY
echo SPEC5_DONE

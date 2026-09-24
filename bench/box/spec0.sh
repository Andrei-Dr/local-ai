#!/bin/bash
# SPEC0: speculation Phase 0 (research/spec-design-2026-09-24.md section 5, P0b): measure before building any tree / multi-draft.
# TEST build = /ai/src/llama.cpp-t4 at spec0 7488986d5 = STABLE source (tu116-served f5ddca176) + three measurement-only commits:
#   LLAMA_SPEC_DUMP=<jsonl>        per verify step: draft top-10 per depth, target top-10 per verified position (full-vocab softmax
#                                  at T = 1 of the raw logits, plus log-sum-exp at the sampler temperature), draft / sampled tokens,
#                                  accepted length, sampler settings. While on, the MTP draft samples on the CPU (a backend top-k
#                                  leaves no full-vocab logits on the host); its top-1 is the same argmax.
#   LLAMA_SPEC_ROUTE_DUMP=<bin>    the selected experts of every verify-batch token (rejected ones included), from the expert
#                                  cache's CPU routing observer (read-only chain), keyed to the same step index.
#   llama-spec-sibling             teacher-forced routing of the target's depth-1 alternatives, forked from the prefix with seq_cp.
# PRE-REGISTERED (copied verbatim from the design doc, section 5, before any run):
#   - H1 (per-depth collapse): acceptance at depth 2 and 3 conditional on depth-1 acceptance is < 0.8 x depth 1.
#     If FALSE (no collapse) -> L1 is demoted.
#   - H2 (depth-1 alternatives): P(target token in draft top-2..4 | draft top-1 rejected) >= 0.30 at T = 0.
#     If < 0.15 -> L3 / L4 are dead at T = 0 (they stay open for T > 0 only if the same number >= 0.30 under the card sampler).
#   - H3 (sibling overlap): mean NEW experts (not in the accepted path's set nor in the LRU) for a depth-1 sibling <= 2.0 per
#     layer (a full token costs ~4). If >= 3.5 -> L4 collapses to L3 (width is never cheap).
#   (The offline simulator over the dump, "a lever proceeds only at >= +5% tokens/s over chain n = 3", is a later step, not here.)
# OPERATIONAL DEFINITIONS (research/scripts/spec0_analyze.py, deployed as /ai/bench/spec0_analyze.py, holds the same text):
#   H1 depth d = steps that drafted >= d tokens with depths 1..d-1 accepted; TRUE if depth 2 AND depth 3 < 0.8 x depth 1, FALSE if
#      both >= 0.8 x depth 1, else MIXED. Judged at T = 0; the card-sampler value is printed beside it.
#   H2 over steps whose depth-1 draft token was rejected: is the target's token (tgt_tok[0]) at rank 2..4 of the draft's top-10?
#   H3 at every 4th generated position p of the T = 0 texts, alternatives = the target's top-4 at p minus the path token (3 per p);
#      batch = path tokens p-1..p+2 (a chain n = 3 verify), LRU = 21 slots per layer replayed over the path before p-1; NEW = the
#      sibling's experts outside batch U LRU, averaged over the 40 layers and all siblings.
# ARMS (STABLE config at -c 4096 via stable_server; workloads = specbench's code + reason prompts + one prose prompt, 768 tokens
#   each, thinking off, /completion on the templated prompt tokens so the exact token ids are kept):
#   0. IDENTITY, T = 0, LLAMA_MOE_CACHE_SYNC=1: S = STABLE build | O = TEST build, dumps unset | D = TEST build, both dumps on.
#      D's dump is also the T = 0 data set (SYNC only fixes when cache uploads publish; drafts and verdicts do not depend on it).
#   1. CARD SAMPLER: TEST build, dumps on, Qwen3.6-35B-A3B card instruct / non-thinking mode (HF README read 2026-09-24):
#      temperature 0.7, top_p 0.80, top_k 20, min_p 0, presence_penalty 1.5, repetition_penalty 1.0; seeds 1 and 2.
#   2. SIBLINGS: llama-spec-sibling over prompt + D's generation per prompt (no expert cache: routing does not depend on it).
# RULES:
#   R0 IDENTITY: the generated token ids of O equal S (the TEST build with the switches unset is the STABLE build) AND D equal O
#      (the switches change no output), for all three prompts; else SPEC0 FAIL (R0) and stop. A void arm (dead server, OOM, empty
#      dump) = FAIL too: an identity test never passes by default.
#   H1-H3 verdicts as printed by spec0_analyze.py; routing alignment (every route record has n_draft + 1 tokens) must be 0
#   misaligned, else the H3-side numbers from the route dump are void (H3 itself comes from the sibling tool).
# RUNTIME (est.): build ~35 min (fresh worktree, CUDA, FA_ALL_QUANTS; incremental when rerun) | identity 3 x ~3 = 9 | card ~4 |
#   siblings 3 x ~1.5 + load = 6 | analysis 1 => ~20 min + the build.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench || exit 1
source /ai/bench/stable.sh || { echo "SPEC0_FAILED: stable.sh"; exit 1; }
PY=/ai/.venv/bin/python
SRC=/ai/src/llama.cpp-mainline; T4=/ai/src/llama.cpp-t4; B4=$T4/build75; SHA=7488986d5
OUT=/ai/bench/runs/spec0; mkdir -p $OUT
GEN=768

echo "--- BUILD $SHA in $T4 | $(date +%T)"
if [ ! -d $T4 ]; then
  git -C $SRC worktree add -q --detach $T4 $SHA || { echo "SPEC0_REFUSED: cannot create $T4 at $SHA (is branch spec0 in $SRC?)"; exit 1; }
fi
[ -z "$(git -C $T4 status --porcelain --untracked-files=no)" ] || { echo "SPEC0_REFUSED: $T4 has local changes"; exit 1; }
if [ "$(git -C $T4 rev-parse --short=9 HEAD)" != "$SHA" ]; then git -C $T4 checkout -q --detach $SHA || { echo "SPEC0_REFUSED: cannot check out $SHA"; exit 1; }; fi
if [ ! -f $B4/CMakeCache.txt ]; then
  # shellcheck disable=SC2086  # a flag list
  cmake -S $T4 -B $B4 $STABLE_CMAKE_FLAGS > spec0_build.log 2>&1 || { tail -5 spec0_build.log; echo "SPEC0_FAILED: configure"; exit 1; }
fi
cmake --build $B4 -j6 --target llama-server llama-spec-sibling >> spec0_build.log 2>&1 || { grep error spec0_build.log | head; echo "SPEC0_FAILED: build"; exit 1; }
[ "$(git -C $T4 rev-parse --short=9 HEAD)" = "$SHA" ] || { echo "SPEC0_REFUSED: $T4 moved off $SHA"; exit 1; }
cat $B4/bin/llama-server $B4/bin/lib*.so* 2>/dev/null | grep -qa LLAMA_SPEC_ROUTE_DUMP || { echo "SPEC0_FAILED: build lacks the dump"; exit 1; }
[ -x $B4/bin/llama-spec-sibling ] || { echo "SPEC0_FAILED: no llama-spec-sibling"; exit 1; }
echo "    built $(git -C $T4 rev-parse --short=9 HEAD) | $(date +%T)"

voidlog() { grep -hoE 'could not re-allocate|out of memory' "server_$1.log" 2>/dev/null | head -1; }
health() { # pid: wait for /health, 0 = up
  local _
  for _ in $(seq 1 200); do curl -sf localhost:8099/health >/dev/null 2>&1 && return 0; kill -0 "$1" 2>/dev/null || return 1; sleep 2; done
  return 1
}
settle() { sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2; }
stop_server() { kill "$STABLE_PID" 2>/dev/null; wait "$STABLE_PID" 2>/dev/null; }

# client LABEL MODE: MODE t0 = greedy, one pass; card = the card sampler, seeds 1 and 2. Writes $OUT/LABEL.json (per prompt and
# seed: prompt ids, generated ids, text) and, for t0, $OUT/LABEL_<prompt>.i32 (prompt + generation) + .start (prompt length).
client() {
  $PY - "$1" "$2" "$OUT" "$GEN" <<'PY'
import json, struct, sys, urllib.request
label, mode, out, gen = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
PROMPTS = [  # code + reason = bench/box/specclient.py's specbench prompts, verbatim; prose added for a third text shape
    ("code",   "Write a Python class implementing an LRU cache with get and put in O(1), with type hints and a short docstring for each method."),
    ("reason", "A train leaves at 3pm going 60 mph. A second leaves the same station at 4pm going 80 mph on the same track. When does the second catch the first? Show the algebra step by step."),
    ("prose",  "Write an essay of about 600 words on how the printing press changed literacy, religion and science in Europe between 1450 and 1650."),
]
CARD = {"temperature": 0.7, "top_p": 0.80, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repeat_penalty": 1.0}
def post(path, body):
    req = urllib.request.Request("http://localhost:8099" + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))
rows = []
for seed in ([0] if mode == "t0" else [1, 2]):
    for kind, q in PROMPTS:
        prompt = post("/apply-template", {"messages": [{"role": "user", "content": q}],
                                          "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
        ids = post("/tokenize", {"content": prompt, "add_special": True, "parse_special": True})["tokens"]
        body = {"prompt": ids, "n_predict": gen, "return_tokens": True, "cache_prompt": False, "seed": seed}
        body.update({"temperature": 0} if mode == "t0" else CARD)
        d = post("/completion", body)
        rows.append({"prompt": kind, "seed": seed, "prompt_ids": ids, "tokens": d["tokens"], "text": d["content"],
                     "timings": d.get("timings", {})})
        t = d.get("timings", {})
        print(f"    {label} {kind} seed {seed}: {len(ids)} prompt tok, {len(d['tokens'])} generated,"
              f" draft {t.get('draft_n')} accepted {t.get('draft_n_accepted')}", flush=True)
        if mode == "t0":
            with open(f"{out}/{label}_{kind}.i32", "wb") as f:
                f.write(struct.pack(f"<{len(ids) + len(d['tokens'])}i", *ids, *d["tokens"]))
            open(f"{out}/{label}_{kind}.start", "w").write(str(len(ids)))
json.dump(rows, open(f"{out}/{label}.json", "w"))
PY
}

# arm LABEL BUILD MODE [dump]: one server, one client pass
arm() {
  local l=$1 b=$2 mode=$3 dump=$4
  rm -f $OUT/$l.json $OUT/$l.jsonl $OUT/$l.route; settle
  ( export BUILD=$b
    [ "$mode" = t0 ] && export LLAMA_MOE_CACHE_SYNC=1
    [ -n "$dump" ] && export LLAMA_SPEC_DUMP=$OUT/$l.jsonl LLAMA_SPEC_ROUTE_DUMP=$OUT/$l.route
    stable_server 4096 $l
    if health $STABLE_PID; then client $l $mode; else echo "    $l: SERVER DIED $(voidlog $l)"; fi
    stop_server )
  echo "    $l: $(voidlog $l) dump $(wc -l < $OUT/$l.jsonl 2>/dev/null || echo -) steps, route $(stat -c %s $OUT/$l.route 2>/dev/null || echo -) B"
}

echo "--- 0. IDENTITY T = 0, LLAMA_MOE_CACHE_SYNC=1: S (STABLE) | O (TEST, dumps unset) | D (TEST, dumps on) | $(date +%T)"
arm spec0_S "$STABLE_BUILD" t0
arm spec0_O "$B4" t0
arm spec0_D "$B4" t0 dump
$PY - "$OUT" <<'PY' || { echo "SPEC0_VERDICT FAIL (R0)"; echo SPEC0_DONE; exit 0; }
import json, os, sys
out = sys.argv[1]
runs = {}
for a in "SOD":
    p = f"{out}/spec0_{a}.json"
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        print(f"    R0 FAIL: arm {a} void (no client output)"); sys.exit(1)
    runs[a] = {r["prompt"]: r["tokens"] for r in json.load(open(p))}
if not os.path.getsize(f"{out}/spec0_D.jsonl") or not os.path.getsize(f"{out}/spec0_D.route"):
    print("    R0 FAIL: D wrote an empty dump"); sys.exit(1)
ok = True
for x, y in (("S", "O"), ("O", "D")):
    for k in runs[x]:
        a, b = runs[x][k], runs[y].get(k, [])
        if a != b:
            i = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
            print(f"    R0 FAIL {x} vs {y} {k}: token ids differ at {i} ({len(a)} vs {len(b)} tokens)"); ok = False
        else:
            print(f"    R0 {x} vs {y} {k}: IDENTICAL ({len(a)} tokens)")
sys.exit(0 if ok else 1)
PY

echo "--- 1. CARD SAMPLER (T 0.7, top_p 0.8, top_k 20, presence 1.5), seeds 1 2, dumps on | $(date +%T)"
arm spec0_C "$B4" card dump

echo "--- 2. SIBLINGS over prompt + D's generation (every 4th generated position, 3 alternatives) | $(date +%T)"
SIBS=()
for k in code reason prose; do
  [ -s $OUT/spec0_D_$k.i32 ] || { echo "    sib $k: no token file"; continue; }
  settle
  SIB_OUT=$OUT/sib_$k.bin SIB_TOKENS=$OUT/spec0_D_$k.i32 SIB_START=$(cat $OUT/spec0_D_$k.start) SIB_EVERY=4 SIB_N_ALT=3 SIB_MAX_TOKENS=4096 \
    env -u LD_LIBRARY_PATH $B4/bin/llama-spec-sibling -m "$STABLE_MODEL_PATH" -c 4096 -b 512 -ub 128 -ngl 999 -fa on -t 6 \
    -ot exps=CPU --load-mode none > $OUT/sib_$k.log 2>&1
  echo "    sib $k: exit $? | $(grep -o 'spec-sibling: .*' $OUT/sib_$k.log | tail -1)"
  [ -s $OUT/sib_$k.bin ] && SIBS+=("$OUT/sib_$k.bin")
done

echo "--- 3. ANALYSIS | $(date +%T)"
$PY /ai/bench/spec0_analyze.py --t0 $OUT/spec0_D.jsonl --route $OUT/spec0_D.route --card $OUT/spec0_C.jsonl \
  ${SIBS[@]:+--sib "${SIBS[@]}"} --slots 21 --json $OUT/spec0_verdicts.json 2>&1 | sed 's/^/  /'
echo SPEC0_DONE

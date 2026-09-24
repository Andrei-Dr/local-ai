#!/bin/bash
# LEAD2: two follow-ups to lead1 (TEST = t3 e6961075b = STABLE + 0032 GQA + 0033 KROW).
# (A) lead1 R2 measured decode after the 9.3k prompt at S [51.59, 52.51] vs T [54.79, 54.69] (+5.2%) at the STABLE config, where
#     F16 KV reaches neither patch. Code read (t3 fattn.cu, ggml_cuda_get_best_fattn_kernel): with GGML_CUDA_FA_MMA_MAX_KV=4096, a
#     1-token F16 FA op at KV >= 4096 on cc 7.5 has gqa_opt_applies = true, so it skips the vec kernel and runs the TILE kernel;
#     below 4096 it runs MMA. Neither patch touches tile or MMA, and vec_gqa_applies() returns false for F16 without
#     GGML_CUDA_FA_VEC_GQA_F16. So the dispatch is unchanged and no mechanism exists. STABLE's usual value here is 54.99-55.4 (stable1):
#     lead1's S arm ran LOW, and S always ran first in each round (S T, S T) = an order confound.
#     HYPOTHESIS HA: the +5.2% is an order / run artifact; with counterbalanced order (S T T S S T T S) T and S tie.
# (B) Long-context KV type (Andrei's decision; we measure): lead1's exploratory arm decoded at 32k with F16 KV + cache 6 at 34.1 t/s,
#     vs q4_0 KV + both patches + cache 12 at 32.5. q4_0 costs a dequant per K/V element on a card with no fast path for it; F16
#     costs 3.6x the KV bytes = fewer expert-cache slots (VRAM trades 1:1 against slots).
#     HYPOTHESIS HB: on TEST without speculation, F16 KV decodes at least as fast as q4_0 KV at 32k; at 64k F16 may not fit.
# ARMS (TEST build unless noted; no MTP in B):
#   A1 identity at 9.3k: S vs T, stable_server 12288, LLAMA_MOE_CACHE_SYNC=1, the 9,279-token prompt, 128 greedy tokens, text saved.
#   A2 speed at 9.3k: stable_server 12288 (STABLE config), order S T T S S T T S (4 runs per arm), longpf 40000.
#   B  32k (138000 chars, -c 36864): Q = -ctk/-ctv q4_0 cache 12 (lead1-proven fit), F = F16 KV cache 6 (lead1-proven fit);
#      order F Q Q F.  64k (276000 chars, -c 69632): Q = q4_0 cache 12 (lead1-proven); F = F16 KV at the largest cache in 4, 2, 0
#      that completes the request (probe in the first F run, the rest reuse it); order Q F F Q.
# PRE-REGISTERED RULES (written before any run):
#   RA1: T text IDENTICAL to S, else LEAD2 FAIL (A1) and stop: a changed output on the served path blocks 0032/0033.
#   RA2: decode after 9.3k, n=4 each, spread = max - min of an arm. ARTIFACT (HA holds) if min(T) <= max(S) (ranges overlap);
#        REPRODUCED if min(T) > max(S) -> an unexplained speed change on an unchanged dispatch = open item, no promotion until
#        explained. Either way R2 (no regression) is re-judged: median(T) >= median(S) - spread(S). Prefill printed, same rule.
#   RB (per depth, decode after the prompt, n=2 each): F16 WIN if min(F) > max(Q); q4_0 WIN if min(Q) > max(F); else TIE, and a tie
#        goes to the higher precision (F16). 64k with no F16 cache that completes -> F16 INFEASIBLE at 64k (a finding). Prefill
#        printed per arm. This is a recommendation input only: the long-context config is Andrei's call.
#   Void: "out of memory" / "could not re-allocate" in a server log, a dead server, or a FOREIGN CPU flag. A void arm makes its
#        comparison INCONCLUSIVE, never a pass (the 64k F probe is the one place a void is expected and handled).
#   Quality: A is an identity test; in B, F16 is the higher precision than q4_0, so a KLD run cannot move the recommendation.
# RUNTIME (est.): A1 6 min | A2 8 x ~2.5 = 20 | B 32k 4 x ~3 = 12, 64k 4 x ~4 + probe ~8 = 24 => ~65 min. Needs the box quiet.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench || exit 1
source /ai/bench/stable.sh || { echo "LEAD2_FAILED: stable.sh"; exit 1; }
PY=/ai/.venv/bin/python
T3=/ai/src/llama.cpp-t3/build75
[ -x $T3/bin/llama-server ] || { echo "LEAD2_REFUSED: TEST build missing ($T3)"; exit 1; }
[ "$(git -C /ai/src/llama.cpp-t3 rev-parse --short=9 HEAD)" = "e6961075b" ] || { echo "LEAD2_REFUSED: t3 is not at e6961075b"; exit 1; }
voidlog() { grep -hoE 'could not re-allocate|out of memory' "server_$1.log" 2>/dev/null | head -1; }
health() { # pid: wait for /health, 0 = up
  local _
  for _ in $(seq 1 200); do curl -sf localhost:8099/health >/dev/null 2>&1 && return 0; kill -0 "$1" 2>/dev/null || return 1; sleep 2; done
  return 1
}
# depth_server CTX LABEL [flags]: STABLE's -c 12288 flag set without speculation, then -c CTX and the arm's flags (as lead1).
depth_server() {
  local ctx=$1 label=$2 a out=() skip=0
  shift 2
  for a in $(stable_args 12288); do
    if [ $skip = 1 ]; then skip=0; continue; fi
    case $a in -md|--spec-type|--spec-draft-n-max) skip=1; continue;; esac
    out+=("$a")
  done
  ( stable_export_env; unset LLAMA_MTP_VOCAB_FILE; exec env -u LD_LIBRARY_PATH "${BUILD:-$STABLE_BUILD}/bin/llama-server" -m "$STABLE_MODEL_PATH" \
      -c "$ctx" --jinja --parallel 1 --port 8099 "${out[@]}" "$@" ) > "server_$label.log" 2>&1 &
  STABLE_PID=$!
}
settle() { sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2; }
stop_server() { kill "$STABLE_PID" 2>/dev/null; wait "$STABLE_PID" 2>/dev/null; }
build_of() { case $1 in S) echo "$STABLE_BUILD";; *) echo "$T3";; esac; }

echo "--- A1. IDENTITY at 9.3k, STABLE config, LLAMA_MOE_CACHE_SYNC=1 | $(date +%T)"
for a in S T; do
  l=lead2_id_$a; settle
  ( export BUILD=$(build_of $a) LLAMA_MOE_CACHE_SYNC=1; stable_server 12288 $l
    health $STABLE_PID || echo "    $l: SERVER DIED $(voidlog $l)"
    $PY - "$l" <<'PY'
import json, sys, urllib.request
l = sys.argv[1]
doc = open("/ai/bench/corpus/prose_big.txt", encoding="utf-8").read()[:40000]
body = json.dumps({"messages": [{"role": "user", "content": "Summarize the following text in five bullet points.\n\n" + doc}],
                   "temperature": 0, "max_tokens": 128, "chat_template_kwargs": {"enable_thinking": False}}).encode()
d = json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:8099/v1/chat/completions", body,
                                                            {"Content-Type": "application/json"}), timeout=3600))
text = d["choices"][0]["message"]["content"]
json.dump({"text": text, "timings": d.get("timings", {})}, open(f"/ai/bench/runs/{l}.text.json", "w"))
print(f"    {l}: {d['timings']['prompt_n']} prompt tok, {d['timings']['predicted_n']} generated")
PY
    stop_server )
done
$PY - <<'PY' || { echo "LEAD2_VERDICT FAIL (RA1)"; echo LEAD2_DONE; exit 0; }
import json, sys
s, t = (json.load(open(f"/ai/bench/runs/lead2_id_{a}.text.json"))["text"] for a in "ST")
if s != t:
    i = next((k for k in range(min(len(s), len(t))) if s[k] != t[k]), min(len(s), len(t)))
    print(f"    RA1 FAIL: texts differ at char {i}: S {s[i:i+60]!r} | T {t[i:i+60]!r}"); sys.exit(1)
print(f"    RA1 PASS: IDENTICAL ({len(s)} chars)")
PY

echo "--- A2. 9.3k speed at the STABLE config, order S T T S S T T S | $(date +%T)"
i=0
for a in S T T S S T T S; do
  i=$((i + 1)); l=lead2_pf_${a}_$i; settle
  ( export BUILD=$(build_of $a); stable_server 12288 $l; health $STABLE_PID || echo "    $l: SERVER DIED"
    echo -n "    "; python3 longpf.py $l 40000 2>&1 | tail -1 | tr -d '\n'; echo " | $(voidlog $l)"
    stop_server )
done

echo "--- B. KV type at depth on TEST, no speculation | $(date +%T)"
FC64=""
for D in "32k 36864 138000 F_Q_Q_F" "64k 69632 276000 Q_F_F_Q"; do
  set -- $D; N=$1; CTX=$2; CH=$3; i=0
  for a in ${4//_/ }; do
    i=$((i + 1)); l=lead2_d${N}_${a}_$i
    if [ $a = Q ]; then flags="-ctk q4_0 -ctv q4_0 --moe-expert-cache 12"; caches=12
    elif [ $N = 32k ]; then caches=6
    else caches=${FC64:-4 2 0}; fi
    for c in $caches; do
      [ $a = F ] && flags="--moe-expert-cache $c"
      rm -f runs/$l.longpf.json runs/$l.cache; settle
      ( export BUILD=$T3; depth_server $CTX $l $flags
        health $STABLE_PID || echo "    $l (cache $c): SERVER DIED $(grep -iE 'error|out of memory' server_$l.log | tail -1 | cut -c1-120)"
        echo -n "    [cache $c] "; python3 longpf.py $l $CH 2>&1 | tail -1 | tr -d '\n'; echo " | $(voidlog $l)"
        stop_server )
      if [ -s runs/$l.longpf.json ] && [ -z "$(voidlog $l)" ]; then
        [ $a = F ] && [ $N = 64k ] && FC64=$c
        echo "$c" > runs/$l.cache; break
      fi
      rm -f runs/$l.longpf.json
    done
  done
  [ $N = 64k ] && [ -z "$FC64" ] && echo "    64k F16: no cache in 4 2 0 completed -> F16 INFEASIBLE at 64k"
done

$PY - <<'PY'
import json, os, re, statistics as st
R = "/ai/bench/runs"
def row(l):
    f, p = f"/ai/bench/server_{l}.log", f"{R}/{l}.longpf.json"
    if not os.path.exists(f) or not os.path.exists(p) or re.search("could not re-allocate|out of memory", open(f, errors="ignore").read()):
        return None
    return json.load(open(p))
# RA2
order = "S T T S S T T S".split()
A = {"S": [], "T": []}
for i, a in enumerate(order, 1):
    A[a].append(row(f"lead2_pf_{a}_{i}"))
if any(x is None for v in A.values() for x in v):
    print("  RA2 INCONCLUSIVE (void arm)")
else:
    for f in ("decode_tps", "prefill_tps"):
        s, t = [round(x[f], 2) for x in A["S"]], [round(x[f], 2) for x in A["T"]]
        rep = "REPRODUCED (T faster beyond the ranges: open item)" if min(t) > max(s) else "ARTIFACT (ranges overlap)"
        r2 = "PASS" if st.median(t) >= st.median(s) - (max(s) - min(s)) else "FAIL"
        print(f"  RA2 9.3k {f}: S {s} T {t} | median T/S {100 * (st.median(t) / st.median(s) - 1):+.1f}% | lead1 gain: {rep} | R2 no-regression {r2}")
# RB
for N, seq in (("32k", "F Q Q F"), ("64k", "Q F F Q")):
    d = {"F": [], "Q": []}; c = {"F": set(), "Q": set()}
    for i, a in enumerate(seq.split(), 1):
        l = f"lead2_d{N}_{a}_{i}"; x = row(l); d[a].append(x)
        if os.path.exists(f"{R}/{l}.cache"):
            c[a].add(open(f"{R}/{l}.cache").read().strip())
    if N == "64k" and all(x is None for x in d["F"]):
        q = [round(x["decode_tps"], 2) for x in d["Q"] if x]
        print(f"  RB 64k: F16 INFEASIBLE (no cache in 4/2/0 completed) | q4_0 decode {q}"); continue
    if any(x is None for v in d.values() for x in v):
        print(f"  RB {N}: INCONCLUSIVE (void arm)"); continue
    f, q = [round(x["decode_tps"], 2) for x in d["F"]], [round(x["decode_tps"], 2) for x in d["Q"]]
    v = "F16 WIN" if min(f) > max(q) else ("q4_0 WIN" if min(q) > max(f) else "TIE -> F16 (higher precision)")
    fp, qp = st.median(x["prefill_tps"] for x in d["F"]), st.median(x["prefill_tps"] for x in d["Q"])
    print(f"  RB {N}: decode F16 (cache {'/'.join(sorted(c['F']))}) {f} vs q4_0 (cache 12) {q} | F/Q {100 * (st.median(f) / st.median(q) - 1):+.1f}%"
          f" -> {v} | prefill F16 {fp:.0f} q4_0 {qp:.0f}")
PY
echo LEAD2_DONE

#!/bin/bash
# LEAD1: patches 0032 (GGML_CUDA_FA_VEC_GQA, default 1) + 0033 (GGML_CUDA_FA_VEC_KROW, default on) = branch tu116-leads, TEST tree
# /ai/src/llama.cpp-t3/build75 (= STABLE d0fd493 + the two commits; nothing else).
# WHAT THEY TOUCH: the FlashAttention vec kernel for ONE query token, head size 256, GQA ratio % 8 == 0, and a QUANTIZED KV cache
# (K/V q4_0, q8_0, q8_0/q5_1; KROW: K q4_0 / q8_0). STABLE serves F16 KV -> neither patch is on its decode path (F16 is only routed
# with GGML_CUDA_FA_VEC_GQA_F16=1), and STABLE's MTP verify batches (2-4 tokens) never take it either. So the win can only show
# where long context is served: q4_0 KV at depth, without speculation (ctx1c: MTP is a net loss at depth).
# HYPOTHESES: (H1) at the STABLE config the TEST build is byte-identical (dispatch unchanged for F16 KV) and not slower;
# (H2) at depth with q4_0 KV, no speculation, decode is faster: GQA alone (KROW=0) over STABLE, KROW over GQA alone
# (fa1: +10% at 131k; fa4: +17% at 120k, +22% at 239k, on IQ2_M); prefill unchanged (prefill never runs the vec kernel).
# ARMS (a/b interleaved, every arm once per round): S = STABLE build, T = TEST build (defaults), G = TEST + GGML_CUDA_FA_VEC_KROW=0.
#   0. unit gate: test-backend-ops -o FLASH_ATTN_EXT (TEST build) under defaults, KROW=0, GQA=0 (vs the CPU reference).
#   1. identity at the STABLE config: `STABLE=1 specbench.sh`, LLAMA_MOE_CACHE_SYNC=1, S vs T, textdiff.
#   2. STABLE config speed: specbench (-c 4096, EDIT+LONG, GEN 300) S/T x 2 rounds; 9,279-token prompt (longpf 40000) + 128-token
#      decode after it at -c 12288 via stable_server, S/T x 2 rounds.
#   3. depth, q4_0 KV, no speculation (STABLE flags at -c 12288 minus -md / --spec-*, then -c, -ctk/-ctv q4_0, cache 12):
#      longpf at ~32k (138000 chars, -c 36864) and ~64k (276000 chars, -c 69632); S/G/T x 2 rounds per depth.
#   4. EXPLORATORY (no verdict, informs a possible F16 lead): 32k depth with F16 KV, no speculation, cache 6 (F16 KV is 3.6x the
#      q4_0 bytes), T vs T+GGML_CUDA_FA_VEC_GQA_F16=1.
# PRE-REGISTERED RULES (written before any run):
#   R0 unit: every FLASH_ATTN_EXT case passes in all three modes, else LEAD1_UNIT_FAILED and stop.
#   R1 identity: T texts IDENTICAL to S for every specbench prompt, else LEAD1 FAIL (the "inert on F16" claim is false) and stop.
#   R2 no regression at the STABLE config: T mean specbench decode >= S mean - S spread (|a - b| averaged over prompts), and T median
#      9.3k decode-after and prefill >= S median - S spread. FAIL => neither patch ships as a default.
#   R3 depth (per depth, decode-after median of the 2 rounds; spread = |a - b| of the arm):
#      0032 PASS if min(G) > max(S) at both depths (faster beyond the run-to-run spread); KILL if median(G) < median(S) at either.
#      0033 PASS if min(T) > max(G) at both depths; KILL if median(T) < median(G) at either.
#      Prefill at depth: |median(T) / median(S) - 1| <= 5% (it runs no vec kernel; more = a harness or box problem, note it).
#   Any arm with "out of memory" / "could not re-allocate" in its server log, a dead server, or a FOREIGN CPU flag is void; a
#   voided comparison is INCONCLUSIVE (rerun), never a pass.
#   Quality: 0033 is non-inferior at the task level (lq2, 4k-128k); 0032 gave a byte-identical reply at 131k (fa1). Neither changes
#   prefill, so a perplexity / KLD run cannot see them (llama-perplexity never decodes one token at a time): no KLD arm.
# RUNTIME (est.): build check + unit 10 min | identity 8 | STABLE speed 26 | depth 32k ~24, 64k ~54 | exploratory 16 => ~2.3 h.
# Needs the box quiet (no other GPU job). Not a promotion: a PASS makes 0032/0033 candidates; promotion is its own step.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench || exit 1
source /ai/bench/stable.sh || { echo "LEAD1_FAILED: stable.sh"; exit 1; }
PY=/ai/.venv/bin/python
T3=/ai/src/llama.cpp-t3/build75
[ -x $T3/bin/llama-server ] && [ -x $T3/bin/test-backend-ops ] || { echo "LEAD1_REFUSED: TEST build missing ($T3)"; exit 1; }
strings $T3/bin/libggml-cuda.so* | grep -q GGML_CUDA_FA_VEC_KROW || { echo "LEAD1_REFUSED: $T3 has no GGML_CUDA_FA_VEC_KROW"; exit 1; }
[ "$(git -C /ai/src/llama.cpp-t3 rev-parse --short=9 HEAD)" = "e6961075b" ] || { echo "LEAD1_REFUSED: t3 is not at e6961075b"; exit 1; }
voidlog() { grep -hoE 'could not re-allocate|out of memory' "server_$1.log" 2>/dev/null | head -1; }
health() { # pid: wait for /health, 0 = up
  local _
  for _ in $(seq 1 200); do curl -sf localhost:8099/health >/dev/null 2>&1 && return 0; kill -0 "$1" 2>/dev/null || return 1; sleep 2; done
  return 1
}
# depth_server CTX LABEL [flags]: the STABLE -c 12288 flag set without speculation (-md FILE, --spec-type X, --spec-draft-n-max N
# dropped), then -c CTX and the arm's flags. Not a STABLE-mode run (no LEDGER_STABLE): it is the long-context config under test.
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
arm_env() { # arm -> BUILD + env assignments (printed for eval)
  case $1 in S) echo "BUILD=$STABLE_BUILD";; T) echo "BUILD=$T3";; G) echo "BUILD=$T3 GGML_CUDA_FA_VEC_KROW=0";;
             F) echo "BUILD=$T3 GGML_CUDA_FA_VEC_GQA_F16=1";; esac
}

echo "--- 0. UNIT GATE (TEST build) | $(date +%T)"
for M in "" "GGML_CUDA_FA_VEC_KROW=0" "GGML_CUDA_FA_VEC_GQA=0"; do
  n=${M:-default}; n=${n//=/_}
  env -u LD_LIBRARY_PATH $M $T3/bin/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0 > lead1_ops_$n.log 2>&1; rc=$?
  echo "    ${M:-defaults}: exit $rc | $(sed -E 's/\x1b\[[0-9;]*m//g' lead1_ops_$n.log | grep -E 'tests passed' | tail -1 | xargs)"
  [ $rc -eq 0 ] || { sed -E 's/\x1b\[[0-9;]*m//g' lead1_ops_$n.log | grep -B1 -E "FAIL|ERR" | grep -E "^ *[A-Z_]+\(" | head -8 | cut -c1-200
                     echo "LEAD1_UNIT_FAILED ${M:-defaults}"; exit 1; }
done

echo "--- 1. IDENTITY at the STABLE config (LLAMA_MOE_CACHE_SYNC=1) | $(date +%T)"
export EDIT=1 LONG=1 GEN=300
for a in S T; do
  settle
  ( eval "export $(arm_env $a)"; LLAMA_MOE_CACHE_SYNC=1 STABLE=1 ./specbench.sh 999 lead1_id_$a ) > /dev/null 2>&1
  [ -s runs/lead1_id_$a.client.json ] || { echo "LEAD1_FAILED: identity run $a produced nothing ($(voidlog lead1_id_$a))"; exit 1; }
done
$PY textdiff.py runs/lead1_id_S.client.json runs/lead1_id_T.client.json | sed 's/^/    /'; id=${PIPESTATUS[0]}
[ "$id" -eq 0 ] || { echo "    R1 FAIL: TEST is not identical to STABLE at the STABLE config"; echo "LEAD1_VERDICT FAIL (R1)"; echo LEAD1_DONE; exit 0; }
echo "    R1 PASS"

echo "--- 2. STABLE config speed | $(date +%T)"
for r in a b; do for a in S T; do
  l=lead1_spec_${a}_$r; settle
  echo "##### $l | $(date +%T)"
  ( eval "export $(arm_env $a)"; STABLE=1 ./specbench.sh 999 $l ) 2>&1 | grep -E "^$l|FOREIGN|DIED" | cut -c1-110 | sed 's/^/    /'
  echo "    $(grep -oE 'hit rate [0-9.]+%' server_$l.log | tail -1) $(voidlog $l)"
done; done
for r in a b; do for a in S T; do
  l=lead1_pf_${a}_$r; settle
  ( eval "export $(arm_env $a)"; stable_server 12288 $l; health $STABLE_PID || echo "    $l: SERVER DIED"
    echo -n "    "; python3 longpf.py $l 40000 2>&1 | tail -1 | tr -d '\n'; echo " | $(voidlog $l)"
    kill $STABLE_PID; wait $STABLE_PID 2>/dev/null )
done; done

echo "--- 3. DEPTH, q4_0 KV, no speculation, cache 12 | $(date +%T)"
for D in "32k 36864 138000" "64k 69632 276000"; do
  set -- $D; N=$1; CTX=$2; CH=$3
  for r in a b; do for a in S G T; do
    l=lead1_d${N}_${a}_$r; settle
    ( eval "export $(arm_env $a)"; depth_server $CTX $l -ctk q4_0 -ctv q4_0 --moe-expert-cache 12
      health $STABLE_PID || echo "    $l: SERVER DIED $(grep -iE 'error|out of memory' server_$l.log | tail -1 | cut -c1-120)"
      echo -n "    "; python3 longpf.py $l $CH 2>&1 | tail -1 | tr -d '\n'; echo " | $(voidlog $l)"
      kill $STABLE_PID; wait $STABLE_PID 2>/dev/null )
  done; done
done

echo "--- 4. EXPLORATORY: 32k F16 KV, no speculation, cache 6: T vs T + GQA_F16=1 | $(date +%T)"
for r in a b; do for a in T F; do
  l=lead1_f16_${a}_$r; settle
  ( eval "export $(arm_env $a)"; depth_server 36864 $l --moe-expert-cache 6
    health $STABLE_PID || echo "    $l: SERVER DIED $(grep -iE 'error|out of memory' server_$l.log | tail -1 | cut -c1-120)"
    echo -n "    "; python3 longpf.py $l 138000 2>&1 | tail -1 | tr -d '\n'; echo " | $(voidlog $l)"
    kill $STABLE_PID; wait $STABLE_PID 2>/dev/null )
done; done

$PY - <<'PY'
import json, os, re, statistics as st
R = "/ai/bench/runs"
def void(l):
    f = f"/ai/bench/server_{l}.log"
    return not os.path.exists(f) or bool(re.search("could not re-allocate|out of memory", open(f, errors="ignore").read()))
def spec(l):
    p = f"{R}/{l}.client.json"
    return None if void(l) or not os.path.exists(p) else {x["prompt"]: x for x in json.load(open(p))["rows"]}
def lpf(l):
    p = f"{R}/{l}.longpf.json"
    return None if void(l) or not os.path.exists(p) else json.load(open(p))
r2 = []
# R2 specbench
S, T = [spec(f"lead1_spec_S_{r}") for r in "ab"], [spec(f"lead1_spec_T_{r}") for r in "ab"]
if None in S + T:
    print("  R2 specbench INCONCLUSIVE (void arm)"); r2.append(None)
else:
    kinds = list(S[0]); m = lambda X: st.mean(x[k]["decode_tps"] for x in X for k in kinds)
    sp = st.mean(abs(S[0][k]["decode_tps"] - S[1][k]["decode_tps"]) for k in kinds)
    ok = m(T) >= m(S) - sp
    print(f"  R2 specbench decode S {m(S):.2f} T {m(T):.2f} ({100 * (m(T) / m(S) - 1):+.1f}%, S spread {sp:.2f}) -> {'PASS' if ok else 'FAIL'}")
    r2.append(ok)
P = {a: [lpf(f"lead1_pf_{a}_{r}") for r in "ab"] for a in "ST"}
if any(x is None for v in P.values() for x in v):
    print("  R2 9.3k INCONCLUSIVE (void arm)"); r2.append(None)
else:
    for f in ("decode_tps", "prefill_tps"):
        s, t = [round(x[f], 2) for x in P["S"]], [round(x[f], 2) for x in P["T"]]
        ok = st.median(t) >= st.median(s) - abs(s[0] - s[1])
        print(f"  R2 9.3k {f}: S {s} T {t} ({100 * (st.median(t) / st.median(s) - 1):+.1f}%) -> {'PASS' if ok else 'FAIL'}")
        r2.append(ok)
# R3 depth
res = {"0032": [], "0033": []}
for N in ("32k", "64k"):
    D = {a: [lpf(f"lead1_d{N}_{a}_{r}") for r in "ab"] for a in "SGT"}
    if any(x is None for v in D.values() for x in v):
        print(f"  R3 {N}: INCONCLUSIVE (void arm)"); res["0032"].append(None); res["0033"].append(None); continue
    d = {a: [round(x["decode_tps"], 2) for x in D[a]] for a in D}; p = {a: [x["prefill_tps"] for x in D[a]] for a in D}
    md = {a: st.median(d[a]) for a in d}
    g = "PASS" if min(d["G"]) > max(d["S"]) else ("KILL" if md["G"] < md["S"] else "NOT PROVEN")
    k = "PASS" if min(d["T"]) > max(d["G"]) else ("KILL" if md["T"] < md["G"] else "NOT PROVEN")
    res["0032"].append(g); res["0033"].append(k)
    pf = 100 * (st.median(p["T"]) / st.median(p["S"]) - 1)
    print(f"  R3 {N} (prompt {D['S'][0]['prompt_n']} tok) decode S {d['S']} G {d['G']} T {d['T']} | G/S {100 * (md['G'] / md['S'] - 1):+.1f}% -> 0032 {g}"
          f" | T/G {100 * (md['T'] / md['G'] - 1):+.1f}% -> 0033 {k} | T/S {100 * (md['T'] / md['S'] - 1):+.1f}% | prefill T/S {pf:+.1f}%"
          + (" (PREFILL MOVED > 5%: note it)" if abs(pf) > 5 else ""))
for p, v in res.items():
    s = "INCONCLUSIVE" if None in v else ("KILL" if "KILL" in v else ("PASS" if all(x == "PASS" for x in v) else "NOT PROVEN"))
    print(f"  R3 {p}: {s}")
E = {a: [lpf(f"lead1_f16_{a}_{r}") for r in "ab"] for a in "TF"}
if all(x is not None for v in E.values() for x in v):
    e = {a: [round(x["decode_tps"], 2) for x in E[a]] for a in E}
    print(f"  EXPLORATORY 32k F16 decode T {e['T']} T+GQA_F16 {e['F']} ({100 * (st.median(e['F']) / st.median(e['T']) - 1):+.1f}%; no verdict)")
else:
    print("  EXPLORATORY: void arm")
R2 = "FAIL" if False in r2 else ("INCONCLUSIVE" if None in r2 else "PASS")
print(f"  LEAD1_VERDICT R1 PASS | R2 {R2} | R3 0032 / 0033 above (a patch ships as a default only with R2 PASS and its R3 PASS)")
PY
echo LEAD1_DONE

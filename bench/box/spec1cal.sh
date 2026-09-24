#!/bin/bash
# SPEC1CAL: calibration run for the speculation simulator (research/spec-design-2026-09-24.md section 7, Phase 1).
# The simulator (bench/box/spec1_sim.py) prices a verify step as t_step = t_fixed + t_draft x depth + c x (NEW experts over the
# batch) and fits t_fixed / t_draft / c on this run: STABLE (build, model, flags, env) with the draft length set to 1, 2, 3, 4 and
# with speculation off. No new build, no dump switches.
# PRE-REGISTERED (copied verbatim from the design doc, section 7, before any run):
#   Step time model t_step = t_fixed + t_draft x depth + c x (sum of NEW experts over the batch), with t_fixed / t_draft / c fitted
#   on the box from a short calibration run (STABLE, draft n = 1..4 and no speculation, specbench prompts) and reported with their
#   fit error.
#   Gate: a policy goes to a box prototype only if its predicted tokens/s beats STABLE's chain n = 3 by >= 5% on the held-out dump
#   at BOTH T = 0 and the card sampler, AND the calibration fit predicts STABLE's measured n = 1..4 speeds within 3%.
# ARMS (A..E, each a fresh server; -c 4096 STABLE config): A n = 1 | B n = 2 | C n = 3 (= STABLE) | D n = 4 | E no speculation
#   (STABLE flags without -md / --spec-type / --spec-draft-n-max, LLAMA_MTP_VOCAB_FILE unset). Two passes in the order
#   A B C D E E D C B A. Per server: one unrecorded 16-token warm-up, then specbench's code + reason prompts (bench/box/specclient.py,
#   verbatim), greedy, 400 tokens, thinking off, via /v1/chat/completions as specbench does.
# RECORDED per prompt (runs/spec1cal/LABEL.json): predicted_n, predicted_ms (decode only), draft_n, draft_n_accepted, and the verify
#   steps of that request (delta of llamacpp:spec_decode_num_drafts_total on /metrics, --metrics added: an endpoint only); for E
#   the steps are the decoded tokens. The simulator reads these; step time = predicted_ms / steps.
# RULE (the 3% fit rule above, applied by spec1_sim.py --cal): per arm n = 1..4, |predicted - measured| decode t/s <= 3% of measured,
#   all four, else the fit FAILS and no policy can pass the gate. Void: a dead server, OOM in a log or a FOREIGN CPU flag -> the arm
#   is missing and the fit is INCONCLUSIVE.
# RUNTIME (est.): 10 servers x (~45 s load + warm-up + 2 x 400 tokens at 40-50 t/s ~ 20 s) + settle ~ 12-15 min.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench || exit 1
source /ai/bench/stable.sh || { echo "SPEC1CAL_FAILED: stable.sh"; exit 1; }
PY=/ai/.venv/bin/python
OUT=/ai/bench/runs/spec1cal; mkdir -p $OUT
GEN=400

voidlog() { grep -hoE 'could not re-allocate|out of memory' "server_$1.log" 2>/dev/null | head -1; }
health() { # pid: wait for /health, 0 = up
  local _
  for _ in $(seq 1 200); do curl -sf localhost:8099/health >/dev/null 2>&1 && return 0; kill -0 "$1" 2>/dev/null || return 1; sleep 2; done
  return 1
}
settle() { sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2; }
stop_server() { kill "$STABLE_PID" 2>/dev/null; wait "$STABLE_PID" 2>/dev/null; }
# nospec_server LABEL: STABLE's -c 4096 flags without speculation (as lead2's depth_server)
nospec_server() {
  local label=$1 a out=() skip=0
  for a in $(stable_args 4096); do
    if [ $skip = 1 ]; then skip=0; continue; fi
    case $a in -md|--spec-type|--spec-draft-n-max) skip=1; continue;; esac
    out+=("$a")
  done
  ( stable_export_env; unset LLAMA_MTP_VOCAB_FILE; export LEDGER_STABLE=$STABLE_STABLE_SINCE
    exec env -u LD_LIBRARY_PATH "$STABLE_BUILD/bin/llama-server" -m "$STABLE_MODEL_PATH" \
      -c 4096 --jinja --parallel 1 --port 8099 --metrics "${out[@]}" ) > "server_$label.log" 2>&1 &
  STABLE_PID=$!
}

client() {
  $PY - "$1" "$OUT" "$GEN" <<'PY'
import json, re, sys, urllib.request
label, out, gen = sys.argv[1], sys.argv[2], int(sys.argv[3])
PROMPTS = [  # bench/box/specclient.py PROMPTS, verbatim
    ("code",   "Write a Python class implementing an LRU cache with get and put in O(1), with type hints and a short docstring for each method."),
    ("reason", "A train leaves at 3pm going 60 mph. A second leaves the same station at 4pm going 80 mph on the same track. When does the second catch the first? Show the algebra step by step."),
]
def chat(q, n):
    body = json.dumps({"messages": [{"role": "user", "content": q}], "temperature": 0, "max_tokens": n,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request("http://localhost:8099/v1/chat/completions", body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))
def steps_total():
    txt = urllib.request.urlopen("http://localhost:8099/metrics", timeout=60).read().decode()
    m = re.search(r"^llamacpp:spec_decode_num_drafts_total (\S+)$", txt, re.M)
    return float(m.group(1)) if m else 0.0
chat(PROMPTS[0][1], 16)  # warm-up, not recorded
rows = []
for kind, q in PROMPTS:
    s0 = steps_total()
    d = chat(q, gen)
    s1 = steps_total()
    t = d.get("timings", {})
    rows.append({"prompt": kind, "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
                 "draft_n": t.get("draft_n") or 0, "draft_n_accepted": t.get("draft_n_accepted") or 0,
                 "steps": int(round(s1 - s0)), "decode_tps": t.get("predicted_per_second")})
    r = rows[-1]
    print(f"    {label} {kind}: {r['predicted_n']} tok, {r['decode_tps']:.2f} t/s, draft {r['draft_n']} accepted"
          f" {r['draft_n_accepted']}, verify steps {r['steps']}", flush=True)
json.dump(rows, open(f"{out}/{label}.json", "w"))
PY
}

echo "--- SPEC1CAL: order A B C D E E D C B A (A..D = draft n 1..4, E = no speculation) | $(date +%T)"
i=0
for a in A B C D E E D C B A; do
  i=$((i + 1))
  case $a in A) n=1;; B) n=2;; C) n=3;; D) n=4;; E) n=0;; esac
  l=spec1cal_n${n}_$i; rm -f $OUT/$l.json; settle
  ( if [ $n = 0 ]; then nospec_server $l; else stable_server 4096 $l --spec-draft-n-max $n --metrics; fi
    if health $STABLE_PID; then
      python3 mon.py start $l; client $l; python3 mon.py stop $l | grep FOREIGN
    else echo "    $l: SERVER DIED $(voidlog $l)"; fi
    stop_server )
  v=$(voidlog $l)
  grep -q '"foreign_cpu_flag": true' runs/$l.mon.json 2>/dev/null && v="${v:+$v, }FOREIGN CPU"
  [ -n "$v" ] && { echo "    $l: VOID ($v)"; rm -f $OUT/$l.json; }
done
echo "    results: $(ls $OUT/*.json 2>/dev/null | wc -l) of 10 arms | fit: spec1_sim.py --cal $OUT"
echo SPEC1CAL_DONE

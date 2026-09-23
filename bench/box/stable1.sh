#!/bin/bash
# STABLE1: end-to-end check of the STABLE-mode harness (bench/box/stable.sh, specbench.sh STABLE=1) against stable/stable.env.
# PRE-REGISTERED: PASS if (1) both runs complete with no OOM / slot re-allocation failure, (2) the ledger records
# stable_since = stable.env's STABLE_SINCE and args containing `stable_args 4096`, (3) the live 9.3k server's argv contains
# `stable_args 12288`, and (4) speed is within 5% of the k2q6b medians (specbench n3 mean 63.06 at cache 21; 9.3k decode 55.4
# at cache 18). A FAIL on (1)-(3) is a harness bug; on (4) alone, box state (record it, rerun).
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
source /ai/bench/stable.sh || { echo "STABLE1_FAILED: stable.sh"; exit 1; }
export EDIT=1 LONG=1 GEN=300
echo "##### stable1_spec | $(date +%T)"
STABLE=1 ./specbench.sh 999 stable1_spec 2>&1 | grep -E "^stable1_spec|FOREIGN|DIED" | cut -c1-110 | sed 's/^/    /'
echo "    $(grep -oE 'hit rate [0-9.]+%' server_stable1_spec.log | tail -1) $(grep -hoE 'could not re-allocate|out of memory' server_stable1_spec.log | head -1)"
echo "##### stable1_pf | $(date +%T)"
sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
stable_server 12288 stable1_pf
for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $STABLE_PID 2>/dev/null || break; sleep 2; done
ARGV=$(tr '\0' ' ' < /proc/$(pgrep -f 'llama-server.*--port 8099' | head -1)/cmdline 2>/dev/null)
echo -n "    "; python3 longpf.py stable1_pf 40000 2>&1 | tail -1 | tr -d '\n'; echo " | $(grep -hoE 'could not re-allocate|out of memory' server_stable1_pf.log | head -1)"
kill $STABLE_PID; wait $STABLE_PID 2>/dev/null
A4=$(stable_args 4096); A12=$(stable_args 12288)
/ai/.venv/bin/python - "$A4" "$A12" "$ARGV" "$STABLE_STABLE_SINCE" <<'PY'
import json, statistics as st, sys
a4, a12, argv, since = sys.argv[1:5]
recs = [json.loads(l) for l in open("/ai/bench/ledger.jsonl")]
spec = [r for r in recs if r.get("label") == "stable1_spec"][-1]
pf = [r for r in recs if r.get("label") == "stable1_pf"]
ok = []
ok.append(("completed", bool(spec.get("completed")) and bool(pf and pf[-1].get("completed"))))
ok.append(("stable_since", spec.get("stable_since") == since))
ok.append(("specbench args", a4 in (spec.get("args") or "")))
ok.append(("9.3k server argv", a12 in argv))
d = st.mean(p["decode_tps"] for p in spec["prompts"]) if spec.get("prompts") else 0
d93 = pf[-1]["longpf"]["decode_tps"] if pf and pf[-1].get("longpf") else 0
ok.append((f"speed specbench {d:.2f} vs 63.06", d >= 63.06 * 0.95))
ok.append((f"speed 9.3k {d93:.2f} vs 55.4", d93 >= 55.4 * 0.95))
for k, v in ok:
    print(f"  {'PASS' if v else 'FAIL'} {k}")
print("  STABLE1_VERDICT " + ("PASS" if all(v for _, v in ok) else "FAIL"))
PY
echo STABLE1_DONE

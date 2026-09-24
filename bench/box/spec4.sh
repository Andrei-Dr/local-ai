#!/bin/bash
# SPEC4: which GPU ops grow with the verify's token count (research/spec-design-2026-09-24.md section 11). spec3 found the verify's
# GPU busy time grows ~5.2 ms per draft position (41% of the marginal step cost); this attributes it per op class with nsys.
# BUILD: STABLE (/ai/src/llama.cpp-v2/build75 via stable.sh), no dump, no new build.
# PRE-REGISTERED (copied verbatim from the design doc, section 11, before any code):
#   spec4: attribute the verify's GPU time per op with nsys (on the box: /usr/bin/nsys), STABLE build (no dump), n1 and n3, the code
#   prompt, 200 tokens each; per-kernel time summed per verify step, grouped by op class (GDN / gated delta rule + conv, cache-chain
#   mul_mat_id, dense mul_mat, flash attention, norms / elementwise, copies), and the CUDA API launch time on the host.
#   - H6: the op classes whose GPU time grows from n1 to n3 account for >= 80% of the busy growth (attribution complete).
#   - H7: the GDN layers are >= 50% of the per-position GPU growth -> the chunked / parallel GDN form for small batches becomes the top
#     lead (it serves the verify AND prefill). If < 25%, GDN is not the target; the top grower is.
#   - H8: host launch time per verify grows >= 1.5 ms per position -> launch overhead is a real share (CUDA graphs / fusion lead).
# METHOD (as att1, the nsys precedent here): nsys profile -t cuda --sample=none --cpuctxsw=none around llama-server, with
#   GGML_CUDA_DISABLE_GRAPHS=1 because nsys 2022.4 cannot see kernels inside CUDA graphs (hand1_phases.py); the served build uses
#   graphs, so launch time here is an upper bound and t/s under nsys is lower than served (printed next to spec3's S3). STABLE
#   config at -c 4096 (stable_args) with --spec-draft-n-max 1 or 3 and -lv 4. Client: a 16-token warm-up request, a 3 s pause, then
#   the measured request (the code prompt, greedy, 200 tokens, /completion on the templated ids). Server stopped with SIGINT so nsys
#   finalizes; .qdstrm imported if needed; exported to sqlite.
#   Verify steps (bench/box/spec4_analyze.py, deployed as /ai/bench/spec4_analyze.py): kernels after the pause; target = streams
#   with gated_delta_net, draft = streams with flash attention and no GDN; a step = the target kernels between two draft-stream
#   bursts; the first group (the prompt pass) dropped. The name -> class map is explicit in the analyzer; unmapped kernels print.
#   Void: a dead server, OOM, a FOREIGN CPU flag, no sqlite -> SPEC4 INCONCLUSIVE.
# RUNTIME (est.): 2 arms x (~90 s load under nsys + ~10 s generation + ~60 s export) ~ 6 min, + analysis ~1 min => ~8 min.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench || exit 1
source /ai/bench/stable.sh || { echo "SPEC4_FAILED: stable.sh"; exit 1; }
PY=/ai/.venv/bin/python
NSYS=/usr/bin/nsys; IMP=/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter
OUT=/ai/bench/runs/spec4; mkdir -p $OUT
[ -x $NSYS ] || { echo "SPEC4_FAILED: $NSYS missing"; exit 1; }

voidlog() { grep -hoE 'could not re-allocate|out of memory' "$OUT/server_$1.log" 2>/dev/null | head -1; }
health() { # pid: wait for /health, 0 = up
  local _
  for _ in $(seq 1 300); do curl -sf localhost:8099/health >/dev/null 2>&1 && return 0; kill -0 "$1" 2>/dev/null || return 1; sleep 2; done
  return 1
}
settle() { sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2; }

# nsys_server LABEL N: STABLE -c 4096 flags + --spec-draft-n-max N under nsys, CUDA graphs off
nsys_server() {
  local l=$1 n=$2
  # shellcheck disable=SC2046  # stable_args is a flag list
  ( stable_export_env; export GGML_CUDA_DISABLE_GRAPHS=1
    exec env -u LD_LIBRARY_PATH $NSYS profile -t cuda --sample=none --cpuctxsw=none -f true -o $OUT/$l \
      "$STABLE_BUILD/bin/llama-server" -m "$STABLE_MODEL_PATH" -c 4096 --jinja --parallel 1 --port 8099 $(stable_args 4096) \
      --spec-draft-n-max $n -lv 4 ) > "$OUT/server_$l.log" 2>&1 &
  NP=$!
}

client() {
  $PY - "$1" "$OUT" <<'PY'
import json, sys, time, urllib.request
label, out = sys.argv[1], sys.argv[2]
CODE = "Write a Python class implementing an LRU cache with get and put in O(1), with type hints and a short docstring for each method."
def post(path, body):
    req = urllib.request.Request("http://localhost:8099" + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))
prompt = post("/apply-template", {"messages": [{"role": "user", "content": CODE}], "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
ids = post("/tokenize", {"content": prompt, "add_special": True, "parse_special": True})["tokens"]
post("/completion", {"prompt": ids, "n_predict": 16, "temperature": 0, "cache_prompt": False})  # warm-up
time.sleep(3)  # the pause spec4_analyze.py uses to find the measured request
d = post("/completion", {"prompt": ids, "n_predict": 200, "temperature": 0, "return_tokens": True, "cache_prompt": False})
t = d.get("timings", {})
row = {"prompt": "code", "decode_tps": t.get("predicted_per_second"), "predicted_n": t.get("predicted_n"),
       "draft_n": t.get("draft_n") or 0, "draft_n_accepted": t.get("draft_n_accepted") or 0}
print(f"    {label}: {row['predicted_n']} tok, {row['decode_tps']:.2f} t/s under nsys (graphs off), draft {row['draft_n']}"
      f" accepted {row['draft_n_accepted']}", flush=True)
json.dump(row, open(f"{out}/{label}.json", "w"))
PY
}

for n in 1 3; do
  l=spec4_n$n
  echo "--- PROFILE n$n | $(date +%T)"
  rm -f $OUT/$l.nsys-rep $OUT/$l.qdstrm $OUT/$l.sqlite $OUT/$l.json runs/$l.mon.json; settle
  nsys_server $l $n
  if health $NP; then
    python3 mon.py start $l; client $l; python3 mon.py stop $l | grep FOREIGN
  else echo "    $l: SERVER DIED $(voidlog $l)"; fi
  pkill -INT -f "llama-server .*--port 8099"; wait $NP 2>/dev/null; sleep 3
  [ -s $OUT/$l.nsys-rep ] || { [ -f $OUT/$l.qdstrm ] && $IMP -i $OUT/$l.qdstrm > /dev/null 2>&1; }
  [ -s $OUT/$l.nsys-rep ] && $NSYS export --type sqlite -f true --output $OUT/$l.sqlite $OUT/$l.nsys-rep > /dev/null 2>&1
  rm -f $OUT/$l.qdstrm
  v=$(voidlog $l)
  grep -q '"foreign_cpu_flag": true' runs/$l.mon.json 2>/dev/null && v="${v:+$v, }FOREIGN CPU"
  [ -s $OUT/$l.json ] || v="${v:+$v, }no client output"
  [ -s $OUT/$l.sqlite ] || v="${v:+$v, }no sqlite"
  [ -n "$v" ] && { echo "    $l: VOID ($v)"; echo "SPEC4_VERDICT INCONCLUSIVE (void arm n$n)"; echo SPEC4_DONE; exit 0; }
  echo "    $l: sqlite $(stat -c %s $OUT/$l.sqlite) bytes"
done

echo "--- t/s under nsys vs spec3's S3 (STABLE n3, no nsys, graphs on, code prompt) | $(date +%T)"
$PY - "$OUT" <<'PY'
import glob, json, statistics as st, sys
out = sys.argv[1]
s3 = [r["decode_tps"] for f in glob.glob("/ai/bench/runs/spec3/spec3_S3_*.json") for r in json.load(open(f)) if r["prompt"] == "code"]
for n in (1, 3):
    r = json.load(open(f"{out}/spec4_n{n}.json"))
    ref = f"spec3 S3 code {st.mean(s3):.2f} t/s -> {100 * (r['decode_tps'] / st.mean(s3) - 1):+.1f}%" if s3 and n == 3 else "(no spec3 reference at n1)"
    print(f"  n{n}: {r['decode_tps']:.2f} t/s under nsys | {ref}")
PY
echo "--- ANALYSIS | $(date +%T)"
$PY /ai/bench/spec4_analyze.py $OUT/spec4_n1.sqlite $OUT/spec4_n3.sqlite --json $OUT/spec4_verdict.json 2>&1 | tee $OUT/spec4_verdict.txt | sed 's/^/  /'
echo SPEC4_DONE

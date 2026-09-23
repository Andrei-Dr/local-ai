#!/bin/bash
# OVL13: why does DECODE slow down after a long prompt when the upload overlap is on? ovl12 (11:15): prefill +18.2% (ubp 2048,
# 402 -> 476 t/s) / +10.3% (ubp 4096, 447 -> 493), prefill KLD identical to 6 decimals; but decode after the 9.3k prompt 49.2-50.1
# -> 44.6-47.9 (-6..-9%), peak VRAM +156 MiB. Decode graphs are never planned (ovl5), so the cost must come from state the overlap
# leaves behind (second backend instance / events / sync of the copy stream on every sched synchronize / a larger compute buffer)
# or from the prefill -> decode transition. Same build (t2 combo), GGML_CUDA_DISABLE_GRAPHS=1 nsys of decode after the 9,279-token
# prompt, off vs on (ubp 2048) -> decprof + h2dov; plus specbench (short prompts: the overlap engages only on the long one) on/off.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
IMP=/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter
[ "$(git -C $T2 rev-parse --abbrev-ref HEAD)" = "combo" ] || { echo "OVL13_REFUSED: $T2 not on combo"; exit 1; }
echo "    t2 at $(git -C $T2 rev-parse --short HEAD)"
for a in off on; do
  l=ovl13_nsys_$a; E=""; [ $a = on ] && E="GGML_SCHED_MOE_PREFETCH=1"
  rm -f $l.nsys-rep $l.qdstrm $l.sqlite
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $E GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 \
    nsys profile -t cuda --sample=none --cpuctxsw=none -f true -o /ai/bench/$l $NEW/bin/llama-server -m $K2 -ngl 999 -fa on \
    -c 12288 -t 6 --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 -lv 4 -ot exps=CPU --moe-expert-cache 22 -md $HEAD \
    --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 4096 -ubp 2048 > server_$l.log 2>&1 &
  NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  echo -n "    "; python3 longpf.py $l 40000 2>&1
  pkill -INT -f "llama-server .*--port 8099"; wait $NP 2>/dev/null; sleep 3
  [ -s $l.nsys-rep ] || $IMP -i $l.qdstrm > /dev/null 2>&1
  nsys export --type sqlite -f true --output $l.sqlite $l.nsys-rep > /dev/null 2>&1
  $PY decprof.py $l.sqlite $l 2>&1 | head -12 | sed 's/^/    /'
  $PY h2dov.py $l.sqlite $l 2>&1 | sed 's/^/    /'
  grep -oE "hit rate [0-9.]+%|decode mode: restored [^\n]{0,80}|could not re-allocate[^\n]{0,60}|prompt warm-up: [0-9]+ uploads" server_$l.log | head -4 | sed 's/^/    /'
  rm -f $l.qdstrm
done
echo "--- specbench on/off (2 rounds) | $(date +%T)"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
SERVE="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
for r in a b; do for a in off on; do
  l=ovl13_${a}_$r; E=""; [ $a = on ] && E="GGML_SCHED_MOE_PREFETCH=1"
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $E GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $SERVE 2>&1 | grep -E "^$l|FOREIGN|DIED" | cut -c1-100 | sed 's/^/    /'
done; done
$PY - <<'PY'
import json
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
S = {a: [L(f"ovl13_{a}_{r}") for r in "ab"] for a in ("off", "on")}
kinds = list(S["off"][0]); m = lambda a, k, f="decode_tps": sum(x[k][f] for x in S[a]) / 2
for k in kinds:
    print(f"  {k:6s} decode off {m('off', k):6.2f} on {m('on', k):6.2f} ({100 * (m('on', k) / m('off', k) - 1):+5.1f}%) | prefill off {m('off', k, 'prefill_tps'):6.1f} on {m('on', k, 'prefill_tps'):6.1f}")
PY
echo OVL13_DONE

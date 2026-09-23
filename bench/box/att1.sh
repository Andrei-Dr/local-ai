#!/bin/bash
# ATT1: why does decode slow down after a long prompt? pfprof1's trace (decode after the 9,279-token prompt, MTP n=2): flash attention
# 630 us per launch = 3.17 of 23.2 ms per token; one layer's KV at 9.3k = 19 MB -> 99 us at 192 GB/s, i.e. the kernel runs at ~30 GB/s
# (~6x off the bandwidth floor). The kernel is flash_attn_ext_f16 (MMA): 0019 keeps MMA below 32 query tokens (tu1: tile everywhere
# cost short-context decode -3.9%), and MTP verify batches are 3 tokens. Hypothesis (inferred): at long KV the MMA decode kernel is
# the slow librarian; the tile/vec path (GGML_CUDA_FA_NO_MMA=1) may scan faster. Build: STABLE (/ai/src/llama.cpp-v2/build75).
#   1 PROFILE (nsys, graphs off) decode after 9.3k: served vs NO_MMA -> FA us/launch, ms/token (decprof.py)
#   2 TIMING  (graphs on) 9.3k prompt + 128 tok decode, served vs NO_MMA interleaved x2; plus the same at a ~2.3k prompt (10k chars)
#   3 NO-DRAFT reference: decode after 9.3k without MTP (1-token passes -> FA vec kernel), served env
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
V2=/ai/src/llama.cpp-v2/build75; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
IMP=/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter
SERVE="-ot exps=CPU --moe-expert-cache 22 -ub 128 -b 4096 -ubp 2048"
MTP="-md $HEAD --spec-type draft-mtp --spec-draft-n-max 2"
start() { # label envs extra-args... ; sets NP
  local l=$1 envs=$2; shift 2
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $envs GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 $PFX $V2/bin/llama-server -m $K2 -ngl 999 -fa on \
    -c 12288 -t 6 --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 -lv 4 $SERVE "$@" > server_$l.log 2>&1 &
  NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done; }
acc() { grep -o "draft acceptance = [0-9.]* ( *[0-9]* accepted / *[0-9]* generated)" server_$1.log | tail -1; }
echo "--- 1. PROFILE decode after 9.3k (nsys, graphs off) | $(date +%T)"
for a in served nomma; do
  l=att1_nsys_$a; E=""; [ $a = nomma ] && E="GGML_CUDA_FA_NO_MMA=1"
  rm -f $l.nsys-rep $l.qdstrm $l.sqlite
  PFX="nsys profile -t cuda --sample=none --cpuctxsw=none -f true -o /ai/bench/$l" start $l "$E GGML_CUDA_DISABLE_GRAPHS=1" $MTP
  echo -n "    "; python3 longpf.py $l 40000 2>&1
  pkill -INT -f "llama-server .*--port 8099"; wait $NP 2>/dev/null; sleep 3
  [ -s $l.nsys-rep ] || $IMP -i $l.qdstrm > /dev/null 2>&1
  nsys export --type sqlite -f true --output $l.sqlite $l.nsys-rep > /dev/null 2>&1
  $PY decprof.py $l.sqlite $l 2>&1 | grep -E "^att1|fa:|FLASH|mmvq" | sed 's/^/    /'
  rm -f $l.qdstrm
done
PFX=""
echo "--- 2. TIMING (graphs on, MTP n=2): served vs NO_MMA, 2 rounds, 9.3k and ~2.3k prompts | $(date +%T)"
for r in a b; do for c in 40000 10000; do for a in served nomma; do
  l=att1_${a}_${c}_$r; E=""; [ $a = nomma ] && E="GGML_CUDA_FA_NO_MMA=1"
  start $l "$E" $MTP
  echo -n "    "; python3 longpf.py $l $c 2>&1 | tr -d '\n'; echo " | $(acc $l)"
  kill $NP; wait $NP 2>/dev/null
done; done; done
echo "--- 3. NO-DRAFT decode after 9.3k (served env) | $(date +%T)"
start att1_nodraft "" 
echo -n "    "; python3 longpf.py att1_nodraft 40000 2>&1
kill $NP; wait $NP 2>/dev/null
$PY - <<'PY'
import json
L = lambda l: json.load(open(f"/ai/bench/runs/{l}.longpf.json"))
for c, name in ((40000, "9.3k"), (10000, "~2.3k")):
    s = [L(f"att1_served_{c}_{r}")["decode_tps"] for r in "ab"]; n = [L(f"att1_nomma_{c}_{r}")["decode_tps"] for r in "ab"]
    ps = [L(f"att1_served_{c}_{r}")["prefill_tps"] for r in "ab"]; pn = [L(f"att1_nomma_{c}_{r}")["prefill_tps"] for r in "ab"]
    ms, mn = sum(s) / 2, sum(n) / 2
    print(f"  {name}: decode served {s[0]:.2f} {s[1]:.2f} | NO_MMA {n[0]:.2f} {n[1]:.2f} -> {100 * (mn / ms - 1):+5.1f}% (served spread {abs(s[0] - s[1]):.2f})"
          f" | prefill {sum(ps) / 2:.1f} vs {sum(pn) / 2:.1f}")
PY
echo ATT1_DONE

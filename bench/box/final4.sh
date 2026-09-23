#!/bin/bash
# FINAL4: the morning table with the full night stack (supersedes final3), one interleaved job on one box state (compaction sysctls off, preflight clean):
#   LEGACY  /ai/src/llama.cpp-mainline/build75, the pre-promotion served config (-b 256, no prefill mode, cache 26)
#   STABLE0 /ai/src/llama.cpp-v2/build75 with this morning's serving env (promo1/promo2: FA tile >= 32, -b 2048 -ubp 2048)
#   STABLE1 same binary + tonight's promotions: FA_MMA_MAX_KV=4096 (promo3) + 49k draft vocab (promo4) + upload overlap (promo5), MTP n2
#   STABLE2 = STABLE1 with --spec-draft-n-max 3 (mtpn1: +4.1% specbench, -2.9% at 9.3k inside noise)
# Specbench (code / reason / edit / long 2.2k) and the 9,279-token prompt (-c 12288, cache 22), 2 rounds, order rotated per round.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
OLD=/ai/src/llama.cpp-mainline/build75; V2=/ai/src/llama.cpp-v2/build75; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf; VOCAB=$M/mtp-Qwen3.6-35B-A3B-vocab49k.bin
[ -s $VOCAB ] || { echo "FINAL4_REFUSED: no $VOCAB (promo4 first)"; exit 1; }
strings $V2/bin/libggml-base.so* | grep -q GGML_SCHED_MOE_PREFETCH || { echo "FINAL4_REFUSED: STABLE lacks the overlap (promo5 first)"; exit 1; }
MTP="-md $HEAD --spec-type draft-mtp --spec-draft-n-max 2"
cfg() { # arm -> BUILD, ENV, specbench args, 9.3k args
  case $1 in
    LEGACY)  B=$OLD; E="";                                                   S="-ot exps=CPU --moe-expert-cache 26 $MTP -ub 128 -b 256";             P="-ot exps=CPU --moe-expert-cache 22 $MTP -ub 128 -b 256";;
    STABLE0) B=$V2;  E="GGML_CUDA_FA_TILE_MIN_BATCH=32";                     S="-ot exps=CPU --moe-expert-cache 26 $MTP -ub 128 -b 2048 -ubp 2048"; P="-ot exps=CPU --moe-expert-cache 22 $MTP -ub 128 -b 4096 -ubp 2048";;
    STABLE1) B=$V2;  E="GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_CUDA_FA_MMA_MAX_KV=4096 LLAMA_MTP_VOCAB_FILE=$VOCAB GGML_SCHED_MOE_PREFETCH=1"
             S="-ot exps=CPU --moe-expert-cache 26 $MTP -ub 128 -b 2048 -ubp 2048"; P="-ot exps=CPU --moe-expert-cache 22 $MTP -ub 128 -b 4096 -ubp 2048";;
    STABLE2) B=$V2;  E="GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_CUDA_FA_MMA_MAX_KV=4096 LLAMA_MTP_VOCAB_FILE=$VOCAB GGML_SCHED_MOE_PREFETCH=1"
             M3="-md $HEAD --spec-type draft-mtp --spec-draft-n-max 3"
             S="-ot exps=CPU --moe-expert-cache 26 $M3 -ub 128 -b 2048 -ubp 2048"; P="-ot exps=CPU --moe-expert-cache 22 $M3 -ub 128 -b 4096 -ubp 2048";;
  esac; }
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32
for r in a b; do
  [ $r = a ] && ORDER="LEGACY STABLE0 STABLE1 STABLE2" || ORDER="STABLE2 STABLE1 STABLE0 LEGACY"
  for a in $ORDER; do
    cfg $a; l=final4_${a}_$r
    sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
    echo "##### $l | $B | ${E:-no env} | $(date +%T)"
    BUILD=$B env -u LD_LIBRARY_PATH $E ./specbench.sh 999 $l $S 2>&1 | grep -E "^$l|FOREIGN|DIED" | cut -c1-110 | sed 's/^/    /'
    l=final4_pf_${a}_$r
    sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
    env -u LD_LIBRARY_PATH $E GGML_OP_OFFLOAD_MIN_BATCH=32 $B/bin/llama-server -m $K2 -ngl 999 -fa on -c 12288 -t 6 --load-mode none --jinja \
      --parallel 1 --port 8099 --cache-ram 0 $P > server_$l.log 2>&1 &
    NP=$!
    for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
    echo -n "    "; python3 longpf.py $l 40000 2>&1
    kill $NP; wait $NP 2>/dev/null
  done
done
$PY - <<'PY'
import json
A = ("LEGACY", "STABLE0", "STABLE1", "STABLE2")
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
S = {a: [L(f"final4_{a}_{r}") for r in "ab"] for a in A}
P = {a: [json.load(open(f"/ai/bench/runs/final4_pf_{a}_{r}.longpf.json")) for r in "ab"] for a in A}
kinds = list(S["LEGACY"][0])
m = lambda a, k, f: sum(x[k][f] for x in S[a]) / 2
pm = lambda a, f: sum(x[f] for x in P[a]) / 2
rows = [(f"decode {k}", lambda a, k=k: m(a, k, "decode_tps")) for k in kinds] + \
       [(f"prefill {k}", lambda a, k=k: m(a, k, "prefill_tps")) for k in kinds] + \
       [("prefill 9.3k", lambda a: pm(a, "prefill_tps")), ("decode after 9.3k", lambda a: pm(a, "decode_tps")),
        ("wall 9.3k prompt+128 (s)", lambda a: pm(a, "wall_s"))]
print("| metric | LEGACY | STABLE (this morning) | STABLE (now, n2) | STABLE (now, n3) | best now vs LEGACY |")
print("|---|---|---|---|---|---|")
for name, f in rows:
    v = [f(a) for a in A]
    best = min(v[2], v[3]) if name.startswith("wall") else max(v[2], v[3])
    ratio = (v[0] / best) if name.startswith("wall") else (best / v[0])
    print(f"| {name} | {v[0]:.1f} | {v[1]:.1f} | {v[2]:.1f} | {v[3]:.1f} | {ratio:.2f}x |")
PY
echo FINAL4_DONE

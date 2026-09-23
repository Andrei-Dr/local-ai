#!/bin/bash
# PF4 (leads A + D): window-mode pre-gated prefetch (880f74b) — L+2's predicted misses issued at layer L's tail on a copy stream,
# in the DDR4-idle window (device runs L's combine + L+1's attention), waited for at L+1's tail. pf3: same-step prefetch put 94%
# of its DMA inside the host window and lost 13%. D: once A shrinks the host window the device phase is the critical side, so the
# FA tile kernel (NO_MMA: 1.9x per launch at 9.3k, att1) should start to pay — the harmony test.
# PRE-REGISTERED (A): win arm WINS if its 4-prompt mean decode beats off by more than off's spread; mechanism confirmed if the
# nsys arm shows H2D inside idle gaps < 30% (pf3 step: 94%, off: 36-43%) and idle ms/token below off's (6.32).
# PRE-REGISTERED (D): win+NO_MMA beats win alone by more than off's spread while NO_MMA alone vs off stays within spread (att1).
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; SHA=880f74b; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
IMP=/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter
[ -z "$(git -C $T2 status --porcelain --untracked-files=no)" ] || { echo "PF4_REFUSED: $T2 has local changes"; exit 1; }
[ "$(git -C $T2 rev-parse --abbrev-ref HEAD)" = "pregate-prefetch" ] || { echo "PF4_REFUSED: $T2 not on pregate-prefetch"; exit 1; }
if [ "$(git -C $T2 rev-parse --short=7 HEAD)" != "$SHA" ]; then git -C $T2 merge -q --ff-only $SHA || { echo "PF4_REFUSED: cannot fast-forward to $SHA"; exit 1; }; fi
cmake --build $NEW -j6 --target llama-server llama-perplexity > pf4_build.log 2>&1 || { grep error pf4_build.log | head; echo "PF4_FAILED: build"; exit 1; }
strings $NEW/bin/libllama.so* | grep -q LLAMA_MOE_PREFETCH_MODE || { echo "PF4_FAILED: build lacks 880f74b"; exit 1; }
echo "    built $(git -C $T2 rev-parse --short HEAD) | $(date +%T)"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
SERVE="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
cst() { grep -oE "hit rate [0-9.]+%|prefetch [0-9]+ \([^)]*\)|pre-gated prefetch on: [a-z]+ mode|prefetch disabled[^\n]*|GGML_ASSERT|ABORT|not waited" server_$1.log | tail -3 | tr '\n' ' '; }
echo "--- 1. SPEED (specbench, 2 interleaved rounds) | $(date +%T)"
A=(off w2 w4 nm w4nm)
for r in a b; do for a in "${A[@]}"; do
  l=pf4_${a}_$r
  case $a in off) E="";; w2) E="LLAMA_MOE_PREFETCH=2 LLAMA_MOE_PREFETCH_TOPK=8";; w4) E="LLAMA_MOE_PREFETCH=4 LLAMA_MOE_PREFETCH_TOPK=8";;
    nm) E="GGML_CUDA_FA_NO_MMA=1";; w4nm) E="LLAMA_MOE_PREFETCH=4 LLAMA_MOE_PREFETCH_TOPK=8 GGML_CUDA_FA_NO_MMA=1";; esac
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  echo "##### $l | ${E:-off} | $(date +%T)"
  env -u LD_LIBRARY_PATH $E GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $SERVE 2>&1 | grep -E "^$l|FOREIGN|DIED" | cut -c1-100 | sed 's/^/    /'
  echo "    $(cst $l)"
done; done
$PY - <<'PY'
import json
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
A = ("off", "w2", "w4", "nm", "w4nm")
R = {a: [L(f"pf4_{a}_{r}") for r in "ab"] for a in A}
kinds = list(R["off"][0])
m = lambda a, k: sum(x[k]["decode_tps"] for x in R[a]) / 2
for k in kinds:
    print(f"  {k:6s} " + " | ".join(f"{a} {m(a, k):6.2f} ({100 * (m(a, k) / m('off', k) - 1):+5.1f}%)" for a in A))
M = {a: sum(m(a, k) for k in kinds) / len(kinds) for a in A}
sp = sum(abs(R["off"][0][k]["decode_tps"] - R["off"][1][k]["decode_tps"]) for k in kinds) / len(kinds)
print(f"  MEAN off {M['off']:.2f} (spread {sp:.2f}) | " + " | ".join(f"{a} {M[a]:.2f} ({100 * (M[a] / M['off'] - 1):+.1f}%)" for a in A[1:]))
best_w = max(("w2", "w4"), key=lambda a: M[a])
print(f"  A: {best_w} {'WINS' if M[best_w] - M['off'] > sp else 'NOT PROVEN'} | D: w4nm vs w4 {M['w4nm'] - M['w4']:+.2f} -> "
      f"{'HARMONY (WIN)' if M['w4nm'] - M['w4'] > sp and abs(M['nm'] - M['off']) <= sp else 'NOT PROVEN'} (nm vs off {M['nm'] - M['off']:+.2f})")
PY
echo "--- 2. LONG (9,279-token prompt, cache 22): off vs w4 vs w4+NO_MMA | $(date +%T)"
for r in a b; do for a in off w4 w4nm; do
  l=pf4_pf_${a}_$r
  case $a in off) E="";; w4) E="LLAMA_MOE_PREFETCH=4 LLAMA_MOE_PREFETCH_TOPK=8";; w4nm) E="LLAMA_MOE_PREFETCH=4 LLAMA_MOE_PREFETCH_TOPK=8 GGML_CUDA_FA_NO_MMA=1";; esac
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $E GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 $NEW/bin/llama-server -m $K2 -ngl 999 -fa on -c 12288 -t 6 \
    --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 -lv 4 -ot exps=CPU --moe-expert-cache 22 -md $HEAD \
    --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 4096 -ubp 2048 > server_$l.log 2>&1 &
  NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  echo -n "    [$a $r] "; python3 longpf.py $l 40000 2>&1 | tr -d '\n'; echo " | $(cst $l)"
  kill $NP; wait $NP 2>/dev/null
done; done
echo "--- 3. MECHANISM (nsys, graphs off, 9.3k decode, w4) | $(date +%T)"
l=pf4_nsys_w4
rm -f $l.nsys-rep $l.qdstrm $l.sqlite
sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
env -u LD_LIBRARY_PATH LLAMA_MOE_PREFETCH=4 LLAMA_MOE_PREFETCH_TOPK=8 GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 \
  nsys profile -t cuda --sample=none --cpuctxsw=none -f true -o /ai/bench/$l $NEW/bin/llama-server -m $K2 -ngl 999 -fa on \
  -c 12288 -t 6 --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 -lv 4 -ot exps=CPU --moe-expert-cache 22 -md $HEAD \
  --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 4096 -ubp 2048 > server_$l.log 2>&1 &
NP=$!
for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
echo -n "    "; python3 longpf.py $l 40000 2>&1
pkill -INT -f "llama-server .*--port 8099"; wait $NP 2>/dev/null; sleep 3
[ -s $l.nsys-rep ] || $IMP -i $l.qdstrm > /dev/null 2>&1
nsys export --type sqlite -f true --output $l.sqlite $l.nsys-rep > /dev/null 2>&1
$PY decprof.py $l.sqlite $l 2>&1 | head -4 | sed 's/^/    /'
$PY h2dov.py $l.sqlite $l 2>&1 | sed 's/^/    /'
echo "    (off reference pf3_nsys_off: idle 6.32 ms/tok, H2D inside gaps 43% / fills 36%)"
rm -f $l.qdstrm
echo PF4_DONE

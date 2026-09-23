#!/bin/bash
# PF1: pre-gated expert prefetch with same-step publish (branch pregate-prefetch bb65a4d = STABLE tu116-served + 1 commit).
# pg1/pg2: layer L+1's router on layer L's MoE input recalls 91% of L+1's top-8 (top-16), covers 70% of the misses with a top-8
# prediction at 68% useful uploads. Ledger (research/design-harmony-ledger.md): the decode critical path is the CPU miss phase
# (~0.42-0.47 ms per layer per pass on DDR4); an expert costs ~42 us on the CPU vs ~92 us over PCIe, run in parallel => balance
# at ~2-3 useful uploads per layer => ~-24% on the miss phase (minus DDR4 contention). TEST tree: t2 worktree switched to
# pregate-prefetch (STABLE build flags). Gates:
#   A INERT     prefetch unset on t2 vs the STABLE binary, identity mode -> must be IDENTICAL (the code is off unless the env is set)
#   B SPEED     specbench serving config, MTP n=2, 2 interleaved rounds: off | budget 2 | 3 | 4 (top-8 prediction)
#   C LONG      9,279-token prompt + 128-token decode, -c 12288 cache 22: off vs budget 3, 2 rounds
# Quality: prefetch only changes WHICH experts are cache hits (device vs host rounding, the cache's accepted class); a
# decode-path KLD job follows if B/C show a gain.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; V2=/ai/src/llama.cpp-v2/build75; SHA=bb65a4d; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
[ -z "$(git -C $T2 status --porcelain --untracked-files=no)" ] || { echo "PF1_REFUSED: $T2 has local changes"; exit 1; }
if [ "$(git -C $T2 rev-parse --short=7 HEAD)" != "$SHA" ]; then git -C $T2 checkout -q pregate-prefetch || { echo "PF1_REFUSED: checkout"; exit 1; }; fi
[ "$(git -C $T2 rev-parse --short=7 HEAD)" = "$SHA" ] || { echo "PF1_REFUSED: $T2 not at $SHA"; exit 1; }
echo "--- 0. BUILD | $(date +%T)"
cmake --build $NEW -j6 --target llama-server > pf1_build.log 2>&1 || { grep error pf1_build.log | head; echo "PF1_FAILED: build"; exit 1; }
strings $NEW/bin/libllama.so* | grep -q LLAMA_MOE_PREFETCH || { echo "PF1_FAILED: no LLAMA_MOE_PREFETCH in libllama"; exit 1; }
echo "    built $(git -C $T2 rev-parse --short HEAD) | $(date +%T)"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32
SERVE="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
cst() { grep -oE "hit rate [0-9.]+%|prefetch [0-9]+ \([0-9.]+ MiB/step\)|pre-gated prefetch on[^(]*" server_$1.log | tail -3 | tr '\n' ' '; }
echo "--- A. INERT (identity mode): STABLE binary vs t2 with prefetch unset | $(date +%T)"
BUILD=$V2 env -u LD_LIBRARY_PATH LLAMA_MOE_CACHE_SYNC=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 pf1_id_stable $SERVE > /dev/null 2>&1
BUILD=$NEW env -u LD_LIBRARY_PATH LLAMA_MOE_CACHE_SYNC=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 pf1_id_t2off $SERVE > /dev/null 2>&1
$PY textdiff.py runs/pf1_id_stable.client.json runs/pf1_id_t2off.client.json | sed 's/^/    /'; id=${PIPESTATUS[0]}
[ "$id" -eq 0 ] || { echo "PF1_FAILED: not inert"; exit 1; }
echo "--- B. SPEED (specbench, 2 interleaved rounds) | $(date +%T)"
for r in a b; do for b in 0 2 3 4; do
  l=pf1_b${b}_$r; E=""; [ $b -gt 0 ] && E="LLAMA_MOE_PREFETCH=$b LLAMA_MOE_PREFETCH_TOPK=8"
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  echo "##### $l | ${E:-off} | $(date +%T)"
  BUILD=$NEW env -u LD_LIBRARY_PATH $E GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $SERVE 2>&1 | grep -E "^$l|FOREIGN" | cut -c1-100 | sed 's/^/    /'
  echo "    $(cst $l)"
done; done
$PY - <<'PY'
import json
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
R = {b: [L(f"pf1_b{b}_{r}") for r in "ab"] for b in (0, 2, 3, 4)}
kinds = list(R[0][0])
m = lambda b, k: sum(x[k]["decode_tps"] for x in R[b]) / 2
for k in kinds:
    print(f"  {k:6s} off {m(0, k):6.2f} | b2 {m(2, k):6.2f} ({100 * (m(2, k) / m(0, k) - 1):+5.1f}%) | b3 {m(3, k):6.2f} ({100 * (m(3, k) / m(0, k) - 1):+5.1f}%) | b4 {m(4, k):6.2f} ({100 * (m(4, k) / m(0, k) - 1):+5.1f}%)")
M = {b: sum(m(b, k) for k in kinds) / len(kinds) for b in R}
sp = sum(abs(R[0][0][k]["decode_tps"] - R[0][1][k]["decode_tps"]) for k in kinds) / len(kinds)
print(f"  MEAN off {M[0]:.2f} (spread {sp:.2f}) | b2 {M[2]:.2f} ({100 * (M[2] / M[0] - 1):+.1f}%) | b3 {M[3]:.2f} ({100 * (M[3] / M[0] - 1):+.1f}%) | b4 {M[4]:.2f} ({100 * (M[4] / M[0] - 1):+.1f}%)")
PY
echo "--- C. LONG (9,279-token prompt, -c 12288, cache 22): off vs budget 3 | $(date +%T)"
for r in a b; do for b in 0 3; do
  l=pf1_pf_b${b}_$r; E=""; [ $b -gt 0 ] && E="LLAMA_MOE_PREFETCH=$b LLAMA_MOE_PREFETCH_TOPK=8"
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $E GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 $NEW/bin/llama-server -m $K2 -ngl 999 -fa on -c 12288 -t 6 \
    --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 -lv 4 -ot exps=CPU --moe-expert-cache 22 -md $HEAD \
    --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 4096 -ubp 2048 > server_$l.log 2>&1 &
  NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  echo -n "    [b$b $r] "; python3 longpf.py $l 40000 2>&1 | tr -d '\n'; echo " | $(cst $l)"
  kill $NP; wait $NP 2>/dev/null
done; done
echo PF1_DONE

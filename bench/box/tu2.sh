#!/bin/bash
# TU2: confirm the two decode-side TU116 results cleanly, and the split FA switch.
# Facts (tu1, 06:17): 0014 (no host sync before stream-ordered copies) decode +3.4% mean (code +3.9, reason +6.2, edit +3.4,
# long -0.4) and IDENTICAL texts on all 4 prompts — but every arm was flagged FOREIGN CPU (kcompactd0, THP compaction), with
# sync-arm spreads up to 5.5 t/s. FA tile (GGML_CUDA_FA_NO_MMA=1): prefill +29% (347.5 -> 448.9 t/s) but decode -3.9%.
# 0015 GGML_CUDA_FA_TILE_MIN_BATCH=N: tile kernel only for batches of N+ query tokens -> prefill gets tile, decode / MTP verify
# (1-3 tokens) keeps MMA.
# Method: compact memory before every arm (front-loads what kcompactd did during tu1's arms), three interleaved rounds of
# D default (0014 on) | S GGML_SCHED_SYNC_BEFORE_COPY=1 (0014 off) | T GGML_CUDA_FA_TILE_MIN_BATCH=32; then long prefill for T.
# Read: 0014 is a WIN when mean(D) - mean(S) > the larger of the D and S spreads (max-min over rounds) and no prompt drops beyond
# its spread. T must equal D on decode within D's spread (decode batches < 32 never reach the tile path) and match tu1's
# FA-tile prefill (~449 t/s).
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf; PY=/ai/.venv/bin/python
SRC=/ai/src/llama.cpp-ov; OV=$SRC/build75
echo "--- 0. BUILD (apply research/patches/tu116; idempotent) | $(date +%T)"
for p in /ai/bench/patches/tu116/*.patch; do
  if git -C $SRC apply -R --check "$p" 2>/dev/null; then echo "    already applied: $(basename $p)"
  else git -C $SRC apply "$p" && echo "    applied: $(basename $p)" || { echo "TU2_FAILED: patch $(basename $p) does not apply"; exit 1; }; fi
done
cmake --build $OV --target llama-server test-backend-ops -j6 > tu2_build.log 2>&1 || { tail -20 tu2_build.log; echo "TU2_FAILED: build"; exit 1; }
strings $OV/bin/libggml-cuda.so* | grep -q GGML_CUDA_FA_TILE_MIN_BATCH || { echo "TU2_FAILED: libggml-cuda lacks 0015"; exit 1; }
export LD_LIBRARY_PATH=$OV/bin
GGML_CUDA_FA_TILE_MIN_BATCH=32 $OV/bin/test-backend-ops -o FLASH_ATTN_EXT -b CUDA0 > tu2_unit.log 2>&1 || { grep FAIL tu2_unit.log | head -5; echo "TU2_FAILED: unit"; exit 1; }
echo "    unit (FA_TILE_MIN_BATCH=32 FLASH_ATTN_EXT): $(grep -E "tests passed" tu2_unit.log | tail -1)"
compact() { sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2; }
echo "--- 1. DECODE (specbench, ov build, prefill mode on; 3 rounds D | S | T, memory compacted before each arm) | $(date +%T)"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$OV
ARGS="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
arm() { local l=$1 envs=$2; shift 2; compact; echo "##### $l | env: ${envs:-none}"; env $envs ./specbench.sh 999 $l $ARGS "$@" 2>&1; }
for r in a b c; do
  arm tu2_D_$r ""
  arm tu2_S_$r "GGML_SCHED_SYNC_BEFORE_COPY=1"
  arm tu2_T_$r "GGML_CUDA_FA_TILE_MIN_BATCH=32"
done
$PY - <<'PY'
import json
L = lambda l: {r["prompt"]: r for r in json.load(open(f"/ai/bench/runs/{l}.client.json"))["rows"]}
R = {a: [L(f"tu2_{a}_{r}") for r in "abc"] for a in "DST"}
kinds = list(R["D"][0])
mean = lambda a, k: sum(x[k]["decode_tps"] for x in R[a]) / 3
spread = lambda a, k: max(x[k]["decode_tps"] for x in R[a]) - min(x[k]["decode_tps"] for x in R[a])
for k in kinds:
    print(f"  {k:6s} D {mean('D', k):6.2f} (spread {spread('D', k):.2f}) | S {mean('S', k):6.2f} ({spread('S', k):.2f}) -> 0014 {100 * (mean('D', k) / mean('S', k) - 1):+5.1f}%"
          f" | T {mean('T', k):6.2f} ({spread('T', k):.2f}) -> T vs D {100 * (mean('T', k) / mean('D', k) - 1):+5.1f}%")
M = {a: sum(mean(a, k) for k in kinds) / len(kinds) for a in "DST"}
SP = {a: sum(spread(a, k) for k in kinds) / len(kinds) for a in "DST"}
worst = min(mean("D", k) - mean("S", k) + max(spread("D", k), spread("S", k)) for k in kinds)
win = (M["D"] - M["S"]) > max(SP["D"], SP["S"]) and worst >= 0
print(f"  MEAN   D {M['D']:6.2f} | S {M['S']:6.2f} | T {M['T']:6.2f} || 0014 {100 * (M['D'] / M['S'] - 1):+5.1f}% (spreads D {SP['D']:.2f} S {SP['S']:.2f}) -> {'WIN' if win else 'not proven'}"
      f" | T vs D {100 * (M['T'] / M['D'] - 1):+5.1f}% ({'same' if abs(M['T'] - M['D']) <= SP['D'] else 'DIFFERENT'})")
PY
echo "--- 2. LONG-PROMPT PREFILL with T (9,279 tok; tu1: MMA 347.5, FA tile everywhere 448.9) | $(date +%T)"
compact
GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 $OV/bin/llama-server -m $K2 -ngl 999 -fa on -c 12288 -t 6 --load-mode none \
  --jinja --parallel 1 --port 8099 --cache-ram 0 -ot exps=CPU --moe-expert-cache 22 -md $HEAD --spec-type draft-mtp \
  --spec-draft-n-max 2 -ub 128 -b 4096 -ubp 4096 > server_tu2_pf.log 2>&1 &
NP=$!
for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
echo -n "    "; python3 longpf.py tu2_pf_T 40000 2>&1
kill $NP; wait $NP 2>/dev/null
unset LD_LIBRARY_PATH
echo TU2_DONE

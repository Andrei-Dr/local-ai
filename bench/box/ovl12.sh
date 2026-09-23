#!/bin/bash
# OVL12 (lead E): upload/compute overlap, re-framed. ovl2-11: the overlap's divergence is not a race, not stale bytes, not CUDA
# graphs, and vanishes with GGML_CUDA_DISABLE_FUSION=1. The MoE weighted-reduction and top-k MoE fusions are only taken when
# ggml_cuda_check_fusion_memory_ranges passes = an ADDRESS test; the hoist moves buffers -> a verdict can flip -> fused vs unfused
# kernels (same math, different rounding) -> routing ties differ from layer 8 on (ovl10). If confirmed, the overlap is the FA-tile
# class (non-bit-exact, KLD-gated), not a correctness bug. Build: t2 switched to branch combo 3858832 (window prefetch + overlap +
# GGML_CUDA_FUSION_LOG).
#   1 VERDICTS  no draft, identity mode, off vs plan (mode 2): the fusion verdict counts differ => mechanism confirmed
#   2 KLD       llama-perplexity -c 2048 6 chunks -b 2048 -ub 2048 (every batch >= 8 tokens/expert: overlap engaged), cache off,
#               vs pfkld_q6.kld: PRE-REGISTERED on PASSES if mean KLD within 0.5% of off and same-top within 0.5 points
#   3 SPEED     9,279-token prompt, -c 12288 cache 22, ubp 2048 and 4096, off/on x 2 rounds + peak VRAM
#               PRE-REGISTERED: on WINS at a ubp if its mean prefill beats off by more than off's spread
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; SHA=3858832; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
[ -z "$(git -C $T2 status --porcelain --untracked-files=no)" ] || { echo "OVL12_REFUSED: $T2 has local changes"; exit 1; }
[ -s pfkld_q6.kld ] || { echo "OVL12_REFUSED: no pfkld_q6.kld (pfkld must run first)"; exit 1; }
if [ "$(git -C $T2 rev-parse --short=7 HEAD)" != "$SHA" ]; then git -C $T2 checkout -q combo 2>/dev/null || git -C $T2 checkout -q -b combo origin/combo 2>/dev/null || git -C $T2 checkout -q $SHA; fi
[ "$(git -C $T2 rev-parse --short=7 HEAD)" = "$SHA" ] || { echo "OVL12_REFUSED: $T2 not at $SHA ($(git -C $T2 rev-parse --short HEAD))"; exit 1; }
cmake --build $NEW -j6 --target llama-server llama-perplexity > ovl12_build.log 2>&1 || { grep error ovl12_build.log | head; echo "OVL12_FAILED: build"; exit 1; }
strings $NEW/bin/libggml-cuda.so* | grep -q GGML_CUDA_FUSION_LOG || { echo "OVL12_FAILED: build lacks the fusion log"; exit 1; }
echo "    built $(git -C $T2 rev-parse --short HEAD) | $(date +%T)"
echo "--- 1. VERDICTS | $(date +%T)"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
NOMTP="-ot exps=CPU --moe-expert-cache 26 -ub 128 -b 2048 -ubp 2048"
for a in off plan; do
  l=ovl12_v_$a; E=""; [ $a = plan ] && E="GGML_SCHED_MOE_PREFETCH=2"
  env -u LD_LIBRARY_PATH $E GGML_CUDA_FUSION_LOG=1 LLAMA_MOE_CACHE_SYNC=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $NOMTP > /dev/null 2>&1
  echo "    $a: $(grep -o 'cuda-fusion: .*' server_$l.log | head -1)"
  echo "    $a: $(grep -o 'cuda-fusion: .*' server_$l.log | tail -1)"
done
$PY textdiff.py runs/ovl12_v_off.client.json runs/ovl12_v_plan.client.json | sed 's/^/    /'
echo "--- 2. KLD (prefill batches, cache off) | $(date +%T)"
COMMON="-f /ai/bench/kld/wiki.test.raw -c 2048 --chunks 6 -ngl 999 -ot exps=CPU -t 6 -fa on -b 2048 -ub 2048"
stats() { grep -E "Mean +KLD|99\.0% +KLD|Same top p" "$1" | cut -c1-100 | sed 's/^/    /'; }
for a in off on; do
  E=""; [ $a = on ] && E="GGML_SCHED_MOE_PREFETCH=1"
  env -u LD_LIBRARY_PATH $E GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 $NEW/bin/llama-perplexity -m $K2 $COMMON \
    --kl-divergence-base pfkld_q6.kld --kl-divergence > ovl12_kld_$a.log 2>&1 || { tail -3 ovl12_kld_$a.log; echo "OVL12_FAILED: kld $a"; exit 1; }
  echo "  $a:"; stats ovl12_kld_$a.log
done
$PY - <<'PY'
import re
g = lambda a: (lambda t: (float(re.search(r"Mean +KLD: +([\d.]+)", t).group(1)), float(re.search(r"Same top p: +([\d.]+)", t).group(1))))(open(f"/ai/bench/ovl12_kld_{a}.log").read())
o, n = g("off"), g("on")
rel = 100 * (n[0] / o[0] - 1)
print(f"  KLD off {o[0]:.6f} top {o[1]:.3f} | on {n[0]:.6f} top {n[1]:.3f} -> {rel:+.2f}% KLD -> {'PASS' if (abs(rel) <= 0.5 or n[0] <= o[0]) and n[1] >= o[1] - 0.5 else 'FAIL'}")
PY
echo "--- 3. SPEED (9,279 tok, -c 12288, cache 22) | $(date +%T)"
vram_peak() { local m=0 v; while kill -0 $1 2>/dev/null; do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$v" -gt "$m" ] && m=$v; echo $m > /tmp/ovl12_vram; sleep 0.5; done; }
for u in 2048 4096; do for r in a b; do for a in off on; do
  l=ovl12_u${u}_${a}_$r; E=""; [ $a = on ] && E="GGML_SCHED_MOE_PREFETCH=1"
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $E GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 $NEW/bin/llama-server -m $K2 -ngl 999 -fa on \
    -c 12288 -t 6 --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 -ot exps=CPU --moe-expert-cache 22 -md $HEAD \
    --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 4096 -ubp $u > server_$l.log 2>&1 &
  NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  vram_peak $NP & VP=$!
  echo -n "    [$l] "; python3 longpf.py $l 40000 2>&1 | tr -d '\n'
  kill $NP; wait $NP 2>/dev/null; kill $VP 2>/dev/null; wait $VP 2>/dev/null
  echo " | peak VRAM $(cat /tmp/ovl12_vram) MiB $(grep -hE 'prefetch disabled|out of memory|failed to allocate' server_$l.log | head -1 | cut -c1-80)"
done; done; done
$PY - <<'PY'
import json
L = lambda l: json.load(open(f"/ai/bench/runs/{l}.longpf.json"))
for u in (2048, 4096):
    off = [L(f"ovl12_u{u}_off_{r}")["prefill_tps"] for r in "ab"]; on = [L(f"ovl12_u{u}_on_{r}")["prefill_tps"] for r in "ab"]
    mo, mn = sum(off) / 2, sum(on) / 2; sp = abs(off[0] - off[1])
    print(f"  ubp {u}: prefill off {off[0]:.1f} {off[1]:.1f} | on {on[0]:.1f} {on[1]:.1f} -> {100 * (mn / mo - 1):+5.1f}% (off spread {sp:.1f}) -> {'WIN' if mn - mo > sp else 'NOT PROVEN'}")
PY
echo OVL12_DONE

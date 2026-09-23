#!/bin/bash
# OVL1: upload/compute overlap for prefill (open item 2; branch upload-overlap = STABLE tu116-served 8dca9af + 1 commit,
# research/patches/experimental/upload-overlap.patch). GGML_SCHED_MOE_PREFETCH=1: each host expert-weight upload is issued by the
# previous GPU split on a second CUDA stream while that split computes; the input copy is hoisted in the graph so ggml-alloc keeps
# its region apart. Facts: pfprof1 had 3.4 s of H2D per 9,279-token prompt with 0.2 s hidden under compute (at 348 t/s); STABLE
# now prefills that prompt at 403 t/s, so the serialized upload is ~15% of the window.
# Build: a fresh TEST worktree /ai/src/llama.cpp-t2 (STABLE flags, -DGGML_CUDA_MMQ_NO_MMA=ON). STABLE and LEGACY are untouched.
# Gates: 1 IDENTITY (LLAMA_MOE_CACHE_SYNC=1, serving config, specbench: the long prompt runs one 2k ubatch = 16 tokens/expert, so
# prefetch engages) prefetch on vs off -> must be IDENTICAL | 2 SPEED 9,279-token prompt at ubp 2048 and 4096, off/on x 2 rounds
# | 3 MECHANISM one nsys capture with prefetch on -> pfprof.py: H2D hidden under compute should rise from ~0.2 s toward the H2D total
# | peak VRAM and any "prefetch disabled" / OOM line.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
REPO=/ai/src/llama.cpp-mainline; T2=/ai/src/llama.cpp-t2; BR=upload-overlap; SHA=a6bd2dbc56e05a8d91d5df9d137060c6d4a97e8b; NEW=$T2/build75
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf; PY=/ai/.venv/bin/python
echo "--- 0. WORKTREE + BUILD | $(date +%T)"
[ -d $T2 ] || git -C $REPO worktree add -q $T2 $BR || { echo "OVL1_FAILED: worktree"; exit 1; }
if [ "$(git -C $T2 rev-parse HEAD)" != "$SHA" ]; then git -C $T2 merge -q --ff-only $SHA || { echo "OVL1_REFUSED: $T2 cannot fast-forward to $SHA"; exit 1; }; fi
[ -z "$(git -C $T2 status --porcelain --untracked-files=no)" ] || { echo "OVL1_REFUSED: $T2 has local changes"; exit 1; }
[ -f $NEW/CMakeCache.txt ] || cmake -S $T2 -B $NEW -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA_FORCE_MMQ=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_MMQ_NO_MMA=ON > ovl1_cmake.log 2>&1 || { tail ovl1_cmake.log; echo "OVL1_FAILED: cmake"; exit 1; }
cmake --build $NEW -j6 --target llama-server > ovl1_build.log 2>&1 || { grep error ovl1_build.log | head; echo "OVL1_FAILED: build"; exit 1; }
strings $NEW/bin/libggml-base.so* | grep -q GGML_SCHED_MOE_PREFETCH || { echo "OVL1_FAILED: no GGML_SCHED_MOE_PREFETCH"; exit 1; }
echo "    built $(git -C $T2 rev-parse --short HEAD) | $(date +%T)"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
SERVE="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
echo "--- 1. IDENTITY (LLAMA_MOE_CACHE_SYNC=1): prefetch off vs on | $(date +%T)"
for a in off on; do
  E=""; [ $a = on ] && E="GGML_SCHED_MOE_PREFETCH=1"
  env -u LD_LIBRARY_PATH $E LLAMA_MOE_CACHE_SYNC=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 ovl1_id_$a $SERVE 2>&1
  grep -hE "prefetch disabled|out of memory|failed to allocate" server_ovl1_id_$a.log | head -3 | sed 's/^/    /'
done
$PY textdiff.py runs/ovl1_id_off.client.json runs/ovl1_id_on.client.json | sed 's/^/    /'; id=${PIPESTATUS[0]}
[ "$id" -eq 0 ] || { echo "OVL1_FAILED: prefetch changes the output"; exit 1; }
vram_peak() { local m=0 v; while kill -0 $1 2>/dev/null; do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$v" -gt "$m" ] && m=$v; echo $m > /tmp/ovl1_vram; sleep 0.5; done; }
pf() { # label envs ubp
  local l=$1 envs=$2 u=$3
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $envs GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 $NEW/bin/llama-server -m $K2 -ngl 999 -fa on \
    -c 12288 -t 6 --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 -ot exps=CPU --moe-expert-cache 22 -md $HEAD \
    --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 4096 -ubp $u > server_$l.log 2>&1 &
  local NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  vram_peak $NP & local VP=$!
  echo -n "    [$l] "; python3 longpf.py $l 40000 2>&1
  kill $NP; wait $NP 2>/dev/null; kill $VP 2>/dev/null; wait $VP 2>/dev/null
  echo "        peak VRAM $(cat /tmp/ovl1_vram) MiB $(grep -hE "prefetch disabled|out of memory|failed to allocate|could not re-allocate" server_$l.log | head -1 | cut -c1-100)"
}
echo "--- 2. SPEED (9,279 tok, -c 12288, cache 22; off/on interleaved, 2 rounds per ubp) | $(date +%T)"
for u in 2048 4096; do for r in a b; do
  pf ovl1_u${u}_off_$r "" $u
  pf ovl1_u${u}_on_$r  "GGML_SCHED_MOE_PREFETCH=1" $u
done; done
$PY - <<'PY'
import json
L = lambda l: json.load(open(f"/ai/bench/runs/{l}.longpf.json"))
for u in (2048, 4096):
    off = [L(f"ovl1_u{u}_off_{r}")["prefill_tps"] for r in "ab"]; on = [L(f"ovl1_u{u}_on_{r}")["prefill_tps"] for r in "ab"]
    mo, mn = sum(off) / 2, sum(on) / 2
    print(f"  ubp {u}: prefill off {off[0]:.1f} {off[1]:.1f} | on {on[0]:.1f} {on[1]:.1f} -> {100 * (mn / mo - 1):+5.1f}% (off spread {abs(off[0] - off[1]):.1f})")
PY
echo "--- 3. MECHANISM (nsys, prefetch on, ubp 4096, graphs off) | $(date +%T)"
rm -f ovl1_nsys.nsys-rep ovl1_nsys.qdstrm ovl1_nsys.sqlite
env -u LD_LIBRARY_PATH GGML_SCHED_MOE_PREFETCH=1 GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 \
  nsys profile -t cuda --sample=none --cpuctxsw=none -f true -o /ai/bench/ovl1_nsys $NEW/bin/llama-server -m $K2 -ngl 999 -fa on \
  -c 12288 -t 6 --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 -ot exps=CPU --moe-expert-cache 22 -md $HEAD \
  --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 4096 -ubp 4096 > server_ovl1_nsys.log 2>&1 &
NP=$!
for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
echo -n "    "; python3 longpf.py ovl1_nsys 40000 2>&1
pkill -INT -f "llama-server .*--port 8099"; wait $NP 2>/dev/null; sleep 3
[ -s ovl1_nsys.nsys-rep ] || /usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i ovl1_nsys.qdstrm > /dev/null 2>&1
nsys export --type sqlite -f true --output ovl1_nsys.sqlite ovl1_nsys.nsys-rep > /dev/null 2>&1
$PY pfprof.py ovl1_nsys.sqlite 2>&1 | head -8 | sed 's/^/    /'
rm -f ovl1_nsys.qdstrm
echo OVL1_DONE

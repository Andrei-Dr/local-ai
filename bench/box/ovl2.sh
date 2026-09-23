#!/bin/bash
# OVL2: why does GGML_SCHED_MOE_PREFETCH=1 change the output? (ovl1 08:21: IDENTITY off vs on -> code / long / reason DIVERGE at
# chars 78-221, edit identical, under LLAMA_MOE_CACHE_SYNC=1 in the serving config; the adversarial review found no hazard.)
# Build: TEST worktree /ai/src/llama.cpp-t2 fast-forwarded to dfbdf14 (adds GGML_SCHED_MOE_PREFETCH=2 = hoist the copies but
# upload in stream order, no second stream). Arms (specbench, same config, LLAMA_MOE_CACHE_SYNC=1):
#   off_a, off_b     determinism floor of the build (must be IDENTICAL, as promo1/promo2 were)
#   on_a,  on_b      prefetch; on_a vs on_b DIVERGING => a race (timing-dependent), IDENTICAL => a deterministic difference
#   plan             mode 2: layout change only; plan vs off DIVERGING => the hoist / allocation is wrong, not the second stream
#   on_block         prefetch with CUDA_LAUNCH_BLOCKING=1 (every launch/copy completes before the host moves on): IDENTICAL to off
#                    => a synchronization (event) bug; DIVERGING => wrong data or wrong region, independent of timing
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; SHA=dfbdf1400cf51ea432f495cc6be7cea5d0c174de
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf; PY=/ai/.venv/bin/python
if [ "$(git -C $T2 rev-parse HEAD)" != "$SHA" ]; then git -C $T2 merge -q --ff-only $SHA || { echo "OVL2_REFUSED: cannot fast-forward $T2"; exit 1; }; fi
cmake --build $NEW -j6 --target llama-server > ovl2_build.log 2>&1 || { grep error ovl2_build.log | head; echo "OVL2_FAILED: build"; exit 1; }
echo "    built $(git -C $T2 rev-parse --short HEAD) | $(date +%T)"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
SERVE="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
arm() { local l=$1 envs=$2; echo "##### $l | env: ${envs:-none} | $(date +%T)"
  env -u LD_LIBRARY_PATH $envs LLAMA_MOE_CACHE_SYNC=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $SERVE > /dev/null 2>&1
  grep -hE "prefetch disabled|out of memory|failed to allocate|CUDA error" server_$l.log | head -3 | sed 's/^/    /'; }
arm ovl2_off_a    ""
arm ovl2_on_a     "GGML_SCHED_MOE_PREFETCH=1"
arm ovl2_plan     "GGML_SCHED_MOE_PREFETCH=2"
arm ovl2_off_b    ""
arm ovl2_on_b     "GGML_SCHED_MOE_PREFETCH=1"
arm ovl2_on_block "GGML_SCHED_MOE_PREFETCH=1 CUDA_LAUNCH_BLOCKING=1"
d() { echo "  $1 vs $2:"; $PY textdiff.py runs/ovl2_$1.client.json runs/ovl2_$2.client.json | sed 's/^/    /'; }
d off_a off_b
d on_a on_b
d off_a on_a
d off_a plan
d off_a on_block
echo OVL2_DONE

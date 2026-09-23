#!/bin/bash
# OVL3: is the ovl2 divergence stale memory in the partially uploaded expert tensor?
# ovl2 (08:42): on_a == on_b and off_a == off_b (deterministic); off vs on, off vs plan (mode 2 = hoist only, no second stream)
# and off vs on_block (CUDA_LAUNCH_BLOCKING) all DIVERGE at the same chars (code 221, long 96, reason 78; edit IDENTICAL)
# => not a race, not the second stream, not event sync: the layout change alone changes the output.
# Hypothesis: below 8 routed tokens per expert the sched uploads only the used experts; the rest of the device copy keeps
# whatever the region held before, which depends on the layout. If any kernel reads those bytes, the output depends on garbage
# (in the baseline too). Build: t2 at 1ea8e04 = dfbdf14 + GGML_SCHED_MOE_DIAG_FILL=<byte> (fill the copy before the partial upload).
# Read (pre-registered):
#   off_plain vs ovl2_off_a   IDENTICAL = the diag commit is inert when unset (build sanity)
#   off_z vs plan_z           IDENTICAL => stale expert bytes ARE the mechanism; DIVERGES => look elsewhere (another stale read)
#   off_z vs off_n            DIVERGES => a kernel reads experts the router did not select (fill 0x00 vs 0xFF = NaN scales)
#   off_plain vs off_z        DIVERGES => the served baseline's output already depends on those bytes
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; SHA=1ea8e04ce2af6b8686fd3ef65efa2f0c11bbc53a
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf; PY=/ai/.venv/bin/python
if [ "$(git -C $T2 rev-parse HEAD)" != "$SHA" ]; then git -C $T2 merge -q --ff-only $SHA || { echo "OVL3_REFUSED: cannot fast-forward $T2"; exit 1; }; fi
[ -z "$(git -C $T2 status --porcelain --untracked-files=no)" ] || { echo "OVL3_REFUSED: $T2 has local changes"; exit 1; }
cmake --build $NEW -j6 --target llama-server > ovl3_build.log 2>&1 || { grep error ovl3_build.log | head; echo "OVL3_FAILED: build"; exit 1; }
strings $NEW/bin/libggml-base.so* | grep -q GGML_SCHED_MOE_DIAG_FILL || { echo "OVL3_FAILED: no GGML_SCHED_MOE_DIAG_FILL"; exit 1; }
echo "    built $(git -C $T2 rev-parse --short HEAD) | $(date +%T)"
export EDIT=1 LONG=1 GEN=300 MODEL=$K2 OFFLOAD=32 BUILD=$NEW
SERVE="-ot exps=CPU --moe-expert-cache 26 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 2048 -ubp 2048"
arm() { local l=$1 envs=$2; echo "##### $l | env: ${envs:-none} | $(date +%T)"
  env -u LD_LIBRARY_PATH $envs LLAMA_MOE_CACHE_SYNC=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 ./specbench.sh 999 $l $SERVE > /dev/null 2>&1
  grep -hE "out of memory|failed to allocate|CUDA error|nan" server_$l.log | head -3 | sed 's/^/    /'; }
arm ovl3_off_plain ""
arm ovl3_off_z     "GGML_SCHED_MOE_DIAG_FILL=0"
arm ovl3_plan_z    "GGML_SCHED_MOE_PREFETCH=2 GGML_SCHED_MOE_DIAG_FILL=0"
arm ovl3_off_n     "GGML_SCHED_MOE_DIAG_FILL=255"
d() { echo "  $1 vs $2:"; $PY textdiff.py runs/$1.client.json runs/$2.client.json | sed 's/^/    /'; }
d ovl3_off_plain ovl2_off_a
d ovl3_off_z ovl3_plan_z
d ovl3_off_z ovl3_off_n
d ovl3_off_plain ovl3_off_z
echo OVL3_DONE

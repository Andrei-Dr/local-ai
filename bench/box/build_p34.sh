#!/bin/bash
# Apply P3 (multi-token cache path), P4 (scheduler barrier / overlap) and mainline #28549 on the moe-cache worktree,
# one commit each, then a full rebuild (ggml.h changes). Refuses to run next to a resident model.
source /ai/bench/preflight.sh || exit 1
set -e
cd /ai/src/llama.cpp-moecache
ci() { git add -A && git -c user.name="$(git log -1 --format=%an)" -c user.email="$(git log -1 --format=%ae)" commit -q -m "$1"; }
git apply --whitespace=nowarn /ai/bench/moecache-p3.diff
ci "feat: serve 2-4 token batches from the MoE expert cache (speculative verify)"
git apply --whitespace=nowarn /ai/bench/moecache-p4.diff
ci "perf: overlap host expert misses with device cache hits via a scheduler barrier

- GGML_TENSOR_FLAG_SCHED_BARRIER starts a new scheduler split; when the next split runs on another
  backend and takes no input from it, that split's inputs are copied before the barrier split is enqueued
- build and expand the device-side cache chain before the host chain; device copies of per-expert scales
- routing observation callback takes the layer index instead of a weight name
- skip empty ids when offloading selected experts (ggml-org/llama.cpp#28739)"
if git apply --check /ai/bench/pr28549.diff 2>/dev/null; then
  git apply --whitespace=nowarn /ai/bench/pr28549.diff
  ci "perf: separate graph result arenas for batches with and without outputs (ggml-org/llama.cpp#28549)"
else
  echo "pr28549 does not apply cleanly - skipped"
fi
git log --oneline -5
(time cmake --build build75 -j6 --target llama-server llama-bench llama-cli test-backend-ops) > /ai/bench/build_p34.log 2>&1 || { grep -E "error" -A4 /ai/bench/build_p34.log | head -40; echo BUILD_P34_FAILED; exit 1; }
tail -3 /ai/bench/build_p34.log
echo BUILD_P34_DONE

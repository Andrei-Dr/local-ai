#!/bin/bash
# PROF2: where does a decode ROUND go NOW? The only profile we have (prof1) is llama-bench at k=1, IQ2_M, no expert cache, no MTP,
# before the P4 overlap and before K2. The shipped-candidate round is different in every term: K2 experts (cheaper CPU matvec),
# cache 26 + MTP n=2 (3-token verify batches), host misses overlapped with device hits. SPEC 4.1 model says a ~50 ms K2 round is
# ~21 ms "fixed" + ~5 ms draft + ~23 ms CPU expert matvecs; nobody has measured what the 21 ms is (GPU compute vs CPU<->GPU
# sync per MoE layer vs PCIe staging vs threadpool barriers). This job measures it so the next C++ goes at the biggest term.
#   A. perf (CPU side, server under real load): symbols + per-thread split. vec_dot share = compute; futex/barrier/sched = waiting;
#      libcuda busy-wait = CPU stalled on the GPU; memcpy = staging.
#   B. nsys (GPU side): kernel time per round, H2D/D2H count + bytes + time, cudaStreamSynchronize / cudaMemcpy API time.
# Read: if GPU kernels + API sync >> CPU vec_dot, the lever is fewer/larger device steps (graph splits, sync points); if vec_dot
# still dominates, it is the miss kernel / hit rate; if futex/barrier is large, it is the threadpool granularity.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive; BUILD=/ai/src/llama.cpp-mainline/build75
F=$M/$N-K2-expQ2K-downQ3K.gguf; [ -f "$F" ] || F=$M/$N-IQ2_M.gguf
ARGS="-m $F -ngl 999 -ot exps=CPU -fa on -c 4096 -t 6 --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 -lv 4
  --moe-expert-cache 26 -ub 128 -b 256 -md $M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 2"
up() { local i; for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && return 0; kill -0 $1 2>/dev/null || return 1; sleep 2; done; return 1; }
echo "model: $F"
echo "########## A. perf, server attached, warm cache ##########"
GGML_OP_OFFLOAD_MIN_BATCH=32 $BUILD/bin/llama-server $ARGS > server_prof2a.log 2>&1 &
PID=$!; up $PID || { echo "    server died: $(grep -iE 'error|out of memory' server_prof2a.log | tail -1 | cut -c1-140)"; echo PROF2_FAILED; exit 1; }
GEN=300 python3 /ai/bench/specclient.py prof2_warm "warm" > /dev/null 2>&1          # fills the expert cache
perf record -q -F 1999 -p $PID -o /ai/bench/perf_prof2.data -- sleep 45 > /dev/null 2>&1 &
PP=$!; sleep 1
GEN=600 python3 /ai/bench/specclient.py prof2_perf "perf" 2>&1 | tail -6
wait $PP 2>/dev/null
grep -hE "statistics +draft|moe-cache: steps" server_prof2a.log | tail -2 | cut -c1-220
kill $PID; wait $PID 2>/dev/null
echo "--- by dso,symbol"; perf report -i /ai/bench/perf_prof2.data --stdio --sort dso,symbol 2>/dev/null | grep -vE "^#|^$" | head -28 | cut -c1-160
echo "--- by thread"; perf report -i /ai/bench/perf_prof2.data --stdio --sort comm,tid 2>/dev/null | grep -vE "^#|^$" | head -12 | cut -c1-120
echo "########## B. nsys, GPU timeline ##########"
rm -f /ai/bench/prof2_nsys.*
GGML_OP_OFFLOAD_MIN_BATCH=32 nsys profile -t cuda --sample=none --cpuctxsw=none -f true -o /ai/bench/prof2_nsys \
  $BUILD/bin/llama-server $ARGS > server_prof2b.log 2>&1 &
NP=$!
for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
GEN=300 python3 /ai/bench/specclient.py prof2_nsys "nsys" 2>&1 | tail -5
pkill -INT -f "llama-server .*--port 8099"; wait $NP 2>/dev/null; sleep 3
R=$(ls /ai/bench/prof2_nsys.nsys-rep 2>/dev/null)
if [ -n "$R" ]; then
  for rep in cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_gpu_mem_size_sum cuda_api_sum; do
    echo "--- $rep"; nsys stats --report $rep --format table "$R" 2>/dev/null | grep -vE "^$|Processing|Generating|SQLite" | head -16 | cut -c1-200
  done
else echo "    no nsys report written: $(tail -3 server_prof2b.log | cut -c1-160)"; fi
rm -f /ai/bench/prof2_nsys.sqlite
echo PROF2_DONE

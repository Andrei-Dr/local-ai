#!/bin/bash
# PFPROF1: where does a long-prompt PREFILL spend its time? Floor ~0.8-1.0 ms/token (~8 GFLOP/token on ~19 TOPS dp4a / 9.6 TFLOPS
# fp16, 50% efficiency) + 0.22 ms/token of expert upload at ub 4096; measured 6.0 ms/token (served prefill mode, ubp 4096).
# One nsys capture per arm of one 9,279-token prompt (longpf.py), CUDA graphs OFF so every kernel is visible (nsys 2022.4 cannot
# see inside graphs; prefill batches are not graph-captured anyway). Arms: served MMA build in prefill mode (llama.cpp-ov is the
# dp4a variant now, so the MMA arm uses a prefill-mode build without the local define — see below), and the dp4a variant.
# Output per arm: time per kernel class (MMQ / MMVQ / FA / delta-net / norm+elementwise / copies), H2D bytes + time, overlap of
# upload and compute, GPU idle inside the prefill window, and the draft (-md) context's share (kernels after the main prefill).
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf; PY=/ai/.venv/bin/python
OV=/ai/src/llama.cpp-ov/build75
IMP=/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter
BASEARGS="-ot exps=CPU --moe-expert-cache 22 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 4096 -ubp 4096"
cap() { # label libdir
  local l=$1 lib=$2
  echo "##### $l | $(date +%T)"
  rm -f $l.nsys-rep $l.qdstrm $l.sqlite
  LD_LIBRARY_PATH=$lib GGML_CUDA_DISABLE_GRAPHS=1 GGML_OP_OFFLOAD_MIN_BATCH=32 nsys profile -t cuda --sample=none --cpuctxsw=none -f true \
    -o /ai/bench/$l $OV/bin/llama-server -m $K2 -ngl 999 -fa on -c 12288 -t 6 --load-mode none --jinja --parallel 1 --port 8099 \
    --cache-ram 0 $BASEARGS > server_$l.log 2>&1 &
  local NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  python3 longpf.py $l 40000 2>&1 | sed 's/^/    /'
  pkill -INT -f "llama-server .*--port 8099"; wait $NP 2>/dev/null; sleep 3
  [ -s $l.nsys-rep ] || $IMP -i $l.qdstrm > /dev/null 2>&1
  nsys export --type sqlite -f true --output $l.sqlite $l.nsys-rep > /dev/null 2>&1
  $PY pfprof.py $l.sqlite 2>&1 | sed 's/^/    /'
}
cap pfprof1_dp4a $OV/bin
echo PFPROF1_DONE

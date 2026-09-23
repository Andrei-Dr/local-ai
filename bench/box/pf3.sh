#!/bin/bash
# PF3: WHY is prefetch slower? pf2 (10:20): moving the copy issue to a helper thread cut the head op 19.5 -> 2.3 us/layer and
# changed NOTHING (b4 async -13.3%, inline -13.1%; hit rate 48 -> 67%). So the DMA itself (41.5 MiB/step) costs more than the ~35%
# fewer CPU misses save. nsys (graphs off) of the decode after the 9,279-token prompt, off vs async budget 4: decprof (ms/token by
# kernel class, GPU idle) + h2dov (H2D inside the CPU-bound idle gaps) + CPU-side view: host experts per token (osrt not traced;
# the idle gaps are the host phases). t2 @ f43cd33.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
T2=/ai/src/llama.cpp-t2; NEW=$T2/build75; PY=/ai/.venv/bin/python
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
IMP=/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter
[ "$(git -C $T2 rev-parse --short=7 HEAD)" = "f43cd33" ] || { echo "PF3_REFUSED: $T2 not at f43cd33"; exit 1; }
for a in off b4; do
  l=pf3_nsys_$a; E=""; [ $a = b4 ] && E="LLAMA_MOE_PREFETCH=4 LLAMA_MOE_PREFETCH_TOPK=8"
  rm -f $l.nsys-rep $l.qdstrm $l.sqlite
  sync; echo 1 > /proc/sys/vm/compact_memory; sleep 2
  env -u LD_LIBRARY_PATH $E GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_OP_OFFLOAD_MIN_BATCH=32 \
    nsys profile -t cuda --sample=none --cpuctxsw=none -f true -o /ai/bench/$l $NEW/bin/llama-server -m $K2 -ngl 999 -fa on \
    -c 12288 -t 6 --load-mode none --jinja --parallel 1 --port 8099 --cache-ram 0 -lv 4 -ot exps=CPU --moe-expert-cache 22 -md $HEAD \
    --spec-type draft-mtp --spec-draft-n-max 2 -ub 128 -b 4096 -ubp 2048 > server_$l.log 2>&1 &
  NP=$!
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  echo -n "    "; python3 longpf.py $l 40000 2>&1
  pkill -INT -f "llama-server .*--port 8099"; wait $NP 2>/dev/null; sleep 3
  [ -s $l.nsys-rep ] || $IMP -i $l.qdstrm > /dev/null 2>&1
  nsys export --type sqlite -f true --output $l.sqlite $l.nsys-rep > /dev/null 2>&1
  $PY decprof.py $l.sqlite $l 2>&1 | head -9 | sed 's/^/    /'
  $PY h2dov.py $l.sqlite $l 2>&1 | sed 's/^/    /'
  grep -oE "hit rate [0-9.]+%|prefetch [0-9]+ \([^)]*\)" server_$l.log | tail -2 | tr '\n' ' ' | sed 's/^/    /'; echo
  rm -f $l.qdstrm
done
echo PF3_DONE

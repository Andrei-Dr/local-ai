#!/bin/bash
# PMUX3: prefill mode on a LONG prompt (~10k tokens), where the big ubatch can actually spread the per-ubatch expert upload:
# ctx1d measured 362.7 t/s on 26.8k tokens with a dedicated ub-4096 cache-0 prefill server; a 2.2k prompt caps any ubatch at
# 2.2k (pmux1: 155.6 t/s). Config: -c 12288 and cache 22 so 12k of F16 KV + the MTP head fit beside the slots (a prefill
# measurement; decode after it is compared like for like). Arms: served ub128, prefill mode ubp 2048, ubp 4096 (-b 4096).
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
M=/ai/models; N=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
K2=$M/$N-K2-expQ2K-downQ3K.gguf; HEAD=$M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
SERVED=/ai/src/llama.cpp-mainline/build75; OV=/ai/src/llama.cpp-ov/build75
BASEARGS="-ot exps=CPU --moe-expert-cache 22 -md $HEAD --spec-type draft-mtp --spec-draft-n-max 2"
arm() { # label build extra...
  local l=$1 b=$2; shift 2
  if [ $b = ov ]; then BLD=$OV; export LD_LIBRARY_PATH=$OV/bin; else BLD=$SERVED; unset LD_LIBRARY_PATH; fi
  echo "##### $l | $b | $*"
  GGML_OP_OFFLOAD_MIN_BATCH=32 $BLD/bin/llama-server -m $K2 -ngl 999 -fa on -c 12288 -t 6 --load-mode none --jinja --parallel 1 \
    --port 8099 --cache-ram 0 $BASEARGS "$@" > server_$l.log 2>&1 &
  local NP=$! m=0 v
  for i in $(seq 1 150); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; kill -0 $NP 2>/dev/null || break; sleep 2; done
  ( while kill -0 $NP 2>/dev/null; do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$v" -gt "$m" ] && m=$v; echo $m > /tmp/pmux3_vram.$l; sleep 1; done ) &
  python3 longpf.py $l 40000 2>&1 | sed 's/^/    /'
  python3 longpf.py ${l}_2 40000 2>&1 | sed 's/^/    /'      # same prompt again: prefix reuse + the second switch
  kill $NP; wait $NP 2>/dev/null
  echo "    peak VRAM $(cat /tmp/pmux3_vram.$l 2>/dev/null) MiB"
  grep -hE "prefill mode: released|after the prefill|decode mode: restored|could not re-allocate|out of memory|failed to allocate" server_$l.log | cut -c1-160 | sed 's/^/    /'
}
arm pmux3_base     served -ub 128 -b 256
arm pmux3_ubp2048  ov     -ub 128 -b 4096 -ubp 2048
arm pmux3_ubp4096  ov     -ub 128 -b 4096 -ubp 4096
unset LD_LIBRARY_PATH
echo PMUX3_DONE

#!/bin/bash
# Repro of dave1 s3_c (patched, DSV4 REAP 145B, cache 96, -ubp 2048) on renderD128 only; keeps the server log.
W=/mnt/llm-storage/localai; IMG=llama-rocm714-rpc:tune
DSV4=/models/dsv4-reap145-gguf/DSV4-Flash-Vision-Exp-REAP-145B-MXFP4-Q8_0.gguf
cd $W/bench; docker rm -f localai-srv >/dev/null 2>&1
docker run -d --name localai-srv --network host --cpuset-cpus 0-23 --device /dev/kfd --device /dev/dri/renderD128 --group-add 44 --group-add 991 \
  --user "$(id -u):$(id -g)" -e GGML_CUDA_FA_TILE_MIN_BATCH=32 -e GGML_SCHED_MOE_PREFETCH=1 -e GGML_OP_OFFLOAD_MIN_BATCH=32 \
  -v $W:/w -v /mnt/llm-storage:/models:ro --entrypoint /w/patched/build/bin/llama-server $IMG \
  -fa on -c 8192 -t 24 --parallel 1 --port 8099 --host 127.0.0.1 --jinja -ngl 999 -ot exps=CPU --moe-expert-cache 96 -ub 128 -b 4096 -ubp 2048 -m $DSV4 >/dev/null
for _ in $(seq 1 300); do curl -sf localhost:8099/health >/dev/null 2>&1 && break; sleep 2; done
OUT=/tmp EDIT=1 LONG=1 GEN=300 python3 specclient.py s3repro - 2>&1 | tail -3
docker logs localai-srv > runs/s3repro.server.log 2>&1
echo "running=$(docker inspect -f '{{.State.Running}} exit={{.State.ExitCode}}' localai-srv)"
docker rm -f localai-srv >/dev/null 2>&1
grep -nE "prompt|n_tokens|ubatch|prefill|abort|error|assert|GGML_|failed|Segmentation|core dumped" runs/s3repro.server.log | grep -v "^.*print_info" | tail -25

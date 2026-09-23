#!/bin/bash
# DAVE2: re-run of dave1 S3 after fix 0031 (window KV caches sized for the prefill-mode ubatch). dave1's patched S3 arms died on
# the long prompt: DeepSeek-V4's sliding-window KV cache was sized for -ub 128 while prefill mode split at -ubp 2048
# (find_slot n_tokens 1348 > size 256 -> smaller-batch retries -> GPU memory fault).
# PRE-REGISTERED (written before any run):
#  S3 DSV4 REAP 145B on ONE MI210 (renderD128), experts offloaded, -c 12288 (dave1's 8192 rejected the 9.3k prompt): base -ot
#     exps=CPU vs patched --moe-expert-cache 96 + prefill mode, 2 reps each, a/b interleaved.
#  Fix gate: every patched arm completes all 4 prompts and the 9.3k prefill with no find_slot error in its server log.
#  Speed: WIN per metric (decode mean over 4 prompts, 9.3k prefill, decode after 9.3k) if patched beats base by > base's a/b
#  spread. Texts: sha256 per prompt vs base (the cache computes hits on the GPU = different rounding; differing = "not proven").
set -u
W=/mnt/llm-storage/localai; IMG=llama-rocm714-rpc:tune; A=renderD128
DSV4=/models/dsv4-reap145-gguf/DSV4-Flash-Vision-Exp-REAP-145B-MXFP4-Q8_0.gguf
cd $W/bench || exit 1; mkdir -p runs/logs
arm() { # arm LABEL TREE "docker env" "server args"
  local label=$1 tree=$2 envs=$3 args=$4
  docker rm -f localai-srv >/dev/null 2>&1
  echo "##### $label | $(date +%T)"
  # shellcheck disable=SC2086
  docker run -d --name localai-srv --network host --cpuset-cpus 0-23 --device /dev/kfd --device /dev/dri/$A --group-add 44 --group-add 991 \
    --user "$(id -u):$(id -g)" $envs -v $W:/w -v /mnt/llm-storage:/models:ro --entrypoint /w/$tree/build/bin/llama-server $IMG \
    -fa on -c 12288 -t 24 --parallel 1 --port 8099 --host 127.0.0.1 --jinja $args >/dev/null
  for _ in $(seq 1 300); do curl -sf localhost:8099/health >/dev/null 2>&1 && break
    [ "$(docker inspect -f '{{.State.Running}}' localai-srv 2>/dev/null)" = true ] || { echo "    $label: SERVER DIED"; break; }
    sleep 2; done
  local vram="$(( $(cat /sys/class/drm/$A/device/mem_info_vram_used) / 1048576 ))MiB"
  OUT=$W/bench/runs EDIT=1 LONG=1 GEN=300 python3 specclient.py "$label" "$vram" 2>&1 | grep -E "^$label|Error" | cut -c1-120 | sed 's/^/    /'
  echo -n "    "; OUT=$W/bench/runs CORPUS=$W/bench/corpus/prose_big.txt python3 longpf.py "${label}_pf" 40000 2>&1 | tail -1
  docker logs localai-srv > runs/logs/$label.server.log 2>&1
  echo "    running=$(docker inspect -f '{{.State.Running}}' localai-srv) find_slot_errors=$(grep -c 'find_slot: n_tokens' runs/logs/$label.server.log) $(grep -oE 'hit rate [0-9.]+%' runs/logs/$label.server.log | tail -1) vram $vram"
  docker rm -f localai-srv >/dev/null 2>&1
}
for r in a b; do
  arm d2_s3_base_$r  base    "" "-ngl 999 -ot exps=CPU -ub 512 -b 2048 -m $DSV4"
  arm d2_s3_c_$r     patched "-e GGML_CUDA_FA_TILE_MIN_BATCH=32 -e GGML_SCHED_MOE_PREFETCH=1 -e GGML_OP_OFFLOAD_MIN_BATCH=32" "-ngl 999 -ot exps=CPU --moe-expert-cache 96 -ub 128 -b 4096 -ubp 2048 -m $DSV4"
done
echo "patched tree: $(git -C $W/patched rev-parse HEAD^{tree})"
echo DAVE2_DONE

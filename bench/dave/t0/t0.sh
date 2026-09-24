#!/bin/bash
# T0: training go/no-go + student step throughput on Dave's 2x MI210 (gfx90a), gating T5 (SPEC 7.1, 7.5).
# PRE-REGISTERED (written before any run):
#  Student = Qwen3.6-35B-A3B architecture (transformers 5.16.1 qwen3_5_moe_text: 40 layers = 30 Gated DeltaNet + 10 gated
#   full attention, 256 experts top-8 + shared expert, hidden 2048, vocab 248320), no MTP head. Config source: the
#   text_config of /mnt/llm-storage/localai/models/qwen36-gptq-int4/config.json (palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4,
#   copied to qwen36_src_config.json; the quantization block is not used). Random BF16 init per tensor from the model's own
#   init statistics (throughput does not depend on values; routing on random routers is near-uniform, a trained router is
#   more skewed: noted, not corrected). Image local/vllm-mi210:rocm10-mi210.7-aiter-jitwarm (torch 2.12.0+rocm10,
#   transformers 5.16.1, no deepspeed). flash-linear-attention 0.5.2 + fla-core pip-installed into /w/t0/pylib only
#   (--no-deps; transformers picks it up at import for chunk_gated_delta_rule). causal_conv1d is not installed (CUDA/HIP
#   extension build): torch conv1d fallback, recorded. Experts: transformers experts_implementation=grouped_mm.
#   Attention: sdpa. FSDP2 fully_shard per decoder layer + root, world size 2 (one rank per MI210), micro-batch 1 per rank,
#   seq 2048, gradient checkpointing (non-reentrant) on. Loss = next-token CE on random ids (the T5 KD loss has the same
#   shape of cost: one [seq, vocab] logits tensor). tok/s = 2 ranks x 2048 / median step time.
#  T0a GO/NO-GO: GO iff a forward + backward + optimizer step completes on both MI210s in BF16 for the full 40-layer student
#   with finite loss and finite grad norm, in regime (i). Kernel paths recorded three ways: static dispatch of
#   chunk_gated_delta_rule / causal_conv1d_fn (fla vs torch), a runtime counter on transformers' _can_use_grouped_mm
#   (grouped_mm vs per-expert fallback), and a torch.profiler step (top device kernels). If the fla path fails, the run
#   is repeated once with fla off (torch GDN fallback) and that is reported as the path T5 would have; if that also fails:
#   NO-GO, stop, report the op, error and whether a fallback exists.
#  T0b (i) routers (40 x mlp.gate) + decoder block 39 (full attention + its experts) trainable, rest frozen, AdamW, params on
#   GPU. (ii) all parameters trainable, FSDP2 CPUOffloadPolicy (sharded params, grads, optimizer on host, optimizer step on
#   CPU). Optimizer for (ii) = torch Adafactor, chosen BEFORE running on host RAM: bf16 params 70 GB + grads 70 GB +
#   AdamW's two states 140 GB = 280 GB > ~200 GB MemAvailable on a shared box; Adafactor's factored state is ~0. If
#   Adafactor fails on the DTensor shards, SGD (no state) is the fallback and is reported as the lower bound on optimizer
#   time. 2 warm-up steps then >= 5 timed steps; report median step (fwd / bwd / opt split), max-min spread, peak VRAM
#   (torch max reserved) per GPU, host RSS high-water per rank, system MemAvailable min.
#  Guards: container --memory 190g (the cgroup OOM kills OUR job, never Dave's) + in-process abort if MemAvailable < 24 GiB.
#   CPU pinned to 24-47. Only renderD128 + renderD131 are passed in. bonsai-ablpq2 is stopped only for the 2-GPU runs and
#   restarted by an EXIT trap (success, failure or signal). Per-run timeouts 15 min (i) / 25 min (ii); whole script 55 min.
#  Re-plan threshold (SPEC 7.1): a trainable-step rate below ~100 tok/s puts a 10M-token run past a day.
#  Output: ETA table hours per 10M / 100M tokens = N / tok_s / 3600 for (i) and (ii).
#  T0c (teacher cost) is not measured here: it comes from q38fn serving on the R9700s (~40-65 t/s decode per request, 4
#   concurrent, production numbers).
# Usage: t0.sh smoke   (renderD128 only, world size 1, 8 layers: validates the code, bonsai untouched)
#        t0.sh run     (both MI210s, the measurement)
set -u
MODE=${1:?smoke|run}
W=/mnt/llm-storage/localai; T=$W/t0
IMG=local/vllm-mi210:rocm10-mi210.7-aiter-jitwarm
mkdir -p $T/results $T/triton-cache
cd $T || exit 1

tr() { # tr LABEL NPROC DEVICES TIMEOUT GDN(fla|torch) ARGS...
  local label=$1 np=$2 devs=$3 to=$4 gdn=$5; shift 5
  local pp=""; [ "$gdn" = fla ] && pp=/w/t0/pylib
  echo "##### $label | np=$np gdn=$gdn | $(date +%T)"
  # shellcheck disable=SC2086
  timeout "$to" docker run --rm --name t0-$label --cpuset-cpus 24-47 --memory 190g --device /dev/kfd $devs \
    --group-add 44 --group-add 991 --security-opt seccomp=unconfined --ipc host -v $W:/w \
    -e PYTHONPATH=$pp -e TRITON_CACHE_DIR=/w/t0/triton-cache -e OMP_NUM_THREADS=$((24 / np)) \
    -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
    --entrypoint torchrun $IMG --standalone --nproc_per_node "$np" /w/t0/t0.py --label "$label" "$@" 2>&1 \
    | grep -vE "Model config|UserWarning|warnings.warn" | tee $T/results/$label.log
  local rc=${PIPESTATUS[0]}; docker rm -f t0-$label >/dev/null 2>&1; echo "##### $label rc=$rc | $(date +%T)"; return "$rc"
}
A="--device /dev/dri/renderD128"; AB="--device /dev/dri/renderD128 --device /dev/dri/renderD131"
vram() { for d in 128 131; do printf "renderD%s used=%.1fGiB busy=%s%% | " $d \
  "$(echo "$(cat /sys/class/drm/renderD$d/device/mem_info_vram_used)/2^30" | bc -l)" "$(cat /sys/class/drm/renderD$d/device/gpu_busy_percent)"; done; echo; }

case $MODE in
smoke)
  vram
  tr smoke_frozen 1 "$A" 10m fla --regime frozen --layers 8 --steps 2 --warmup 1 --profile
  tr smoke_full 1 "$A" 10m fla --regime full --layers 8 --steps 2 --warmup 1
  ;;
run)
  restore() { docker start bonsai-ablpq2 >/dev/null && echo "bonsai-ablpq2 restarted: $(docker inspect -f '{{.State.Status}}' bonsai-ablpq2)"; }
  trap restore EXIT
  vram; echo "MemAvailable $(awk '/MemAvailable/{printf "%.1f GiB",$2/2^20}' /proc/meminfo)"
  docker stop bonsai-ablpq2 >/dev/null && echo "bonsai-ablpq2 stopped $(date +%T)"
  sleep 3; vram
  G=fla
  if ! tr t0_i_frozen 2 "$AB" 15m fla --regime frozen --steps 6 --profile; then
    if grep -qE "fla|triton|chunk_gated" $T/results/t0_i_frozen.log; then
      G=torch; tr t0_i_frozen_torchgdn 2 "$AB" 15m torch --regime frozen --steps 6 --profile || { echo "T0a NO-GO"; exit 1; }
    else echo "T0a NO-GO"; exit 1; fi
  fi
  tr t0_ii_full 2 "$AB" 25m $G --regime full --steps 5 \
    || { grep -q "Adafactor\|_single_tensor_adafactor\|_foreach_adafactor" $T/results/t0_ii_full.log \
         && tr t0_ii_full_sgd 2 "$AB" 25m $G --regime full --steps 5 --opt SGD; }
  vram
  ;;
esac
docker run --rm -v $W:/w --entrypoint chown $IMG -R "$(id -u):$(id -g)" /w/t0

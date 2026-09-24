#!/bin/bash
# DAVE4: vLLM vs llama.cpp for Qwen3.6-35B-A3B fully resident on ONE MI210 (renderD128), single user.
#   vllm     = Dave's image local/vllm-mi210:rocm10-mi210.7-aiter-jitwarm (vLLM 0.28.1rc0+mi210.7, AITER), checkpoint
#              palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4 (experts INT4 g128, rest BF16, 24.4 GB, MTP head BF16), cudagraph
#              FULL_DECODE_ONLY (Dave's stack: PIECEWISE under spec decoding costs ~60% decode)
#   ours     = e613ef2 + our series 0001-0031, Dave's cmake flags (build-df), HauhauCS Q4_K_M + MTP Q4_0 head, draft vocab
#   dave     = 030ebb558 + davetha/llama.cpp-mi210 04-16, his config (no MTP: his base predates it)
# PRE-REGISTERED (written before any run):
#  Arms (2 reps, interleaved): vllm_mtp3, vllm_nomtp, ours_mtp3, dave. Every arm is timed by the SAME client (oaiclient.py:
#  streamed tokens, client clock) on the same prompts (specclient's 4 + a ~9.3k-token longpf row); one warm-up request per
#  server start before recording. Metrics: decode mean over the 4 prompts; longpf prefill (prompt / TTFT); longpf decode.
#  WIN rule: mean beats the other arm's by more than the larger a/b spread.
#  Caveat recorded up front: the checkpoints differ (GPTQ-Int4 of base Qwen3.6 vs Q4_K_M of the HauhauCS finetune; ~4.4 vs
#  ~4.8 bits/weight), so this is engine+format vs engine+format, and texts are not compared.
set -u
W=/mnt/llm-storage/localai; A=renderD128
VIMG=local/vllm-mi210:rocm10-mi210.7-aiter-jitwarm; LIMG=llama-rocm714-rpc:tune
QWEN=/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf
HEAD="-md /w/models/mtp-Qwen3.6-35B-A3B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 3"
OURS_ENV="-e GGML_CUDA_FA_TILE_MIN_BATCH=32 -e GGML_CUDA_FA_MMA_MAX_KV=4096 -e GGML_SCHED_MOE_PREFETCH=1 -e GGML_OP_OFFLOAD_MIN_BATCH=32 -e LLAMA_MTP_VOCAB_FILE=/w/models/mtp-Qwen3.6-35B-A3B-vocab49k.bin"
cd $W/bench || exit 1; mkdir -p runs/logs $W/vcache
up() { # wait for :8099, fail fast if the container died
  for _ in $(seq 1 600); do curl -sf localhost:8099/health >/dev/null 2>&1 && return 0
    [ "$(docker inspect -f '{{.State.Running}}' localai-srv 2>/dev/null)" = true ] || return 1; sleep 2; done; return 1; }
measure() { # measure LABEL MODEL_NAME
  curl -s localhost:8099/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$2\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"max_tokens\":16,\"temperature\":0}" >/dev/null
  OUT=$W/bench/runs MODEL_NAME=$2 EDIT=1 LONG=1 LONGPF=40000 CORPUS=$W/bench/corpus/prose_big.txt GEN=300 python3 oaiclient.py "$1" 2>&1 | sed 's/^/    /'
  docker logs localai-srv > runs/logs/$1.server.log 2>&1; docker rm -f localai-srv >/dev/null 2>&1; }
vllm() { # vllm LABEL [extra args...]
  local label=$1; shift; docker rm -f localai-srv >/dev/null 2>&1; echo "##### $label | $(date +%T)"
  docker run -d --name localai-srv --network host --ipc host --cpuset-cpus 0-23 --device /dev/kfd --device /dev/dri/$A \
    --group-add 44 --group-add 991 --security-opt seccomp=unconfined -e HSA_NO_SCRATCH_RECLAIM=1 \
    -v $W:/w -v $W/vcache:/cache --entrypoint /usr/local/bin/mi210-entrypoint $VIMG \
    serve /w/models/qwen36-gptq-int4 --served-model-name m --port 8099 --host 127.0.0.1 --max-model-len 16384 --max-num-seqs 1 \
    --gpu-memory-utilization 0.90 --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' "$@" >/dev/null
  up && measure "$label" m || { echo "    $label: SERVER FAILED: $(docker logs localai-srv 2>&1 | grep -iE 'error|exception' | tail -3 | cut -c1-200)"; docker logs localai-srv > runs/logs/$label.server.log 2>&1; docker rm -f localai-srv >/dev/null 2>&1; }; }
llama() { # llama LABEL TREE "env" "args"
  local label=$1; docker rm -f localai-srv >/dev/null 2>&1; echo "##### $label | $(date +%T)"
  # shellcheck disable=SC2086
  docker run -d --name localai-srv --network host --cpuset-cpus 0-23 --device /dev/kfd --device /dev/dri/$A --group-add 44 --group-add 991 \
    --user "$(id -u):$(id -g)" $3 -v $W:/w -v /mnt/llm-storage:/models:ro --entrypoint /w/$2/bin/llama-server $LIMG \
    -fa on -c 16384 -t 24 --parallel 1 --port 8099 --host 127.0.0.1 --jinja $4 >/dev/null
  up && measure "$label" m || { echo "    $label: SERVER FAILED"; docker rm -f localai-srv >/dev/null 2>&1; }; }
for r in a b; do
  vllm  d4_vllm_mtp3_$r  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
  vllm  d4_vllm_nomtp_$r
  llama d4_ours_mtp3_$r  patched/build-df "$OURS_ENV" "-ngl 999 -ub 512 -b 2048 -m $QWEN $HEAD"
  llama d4_dave_$r       dave/build "-e HSA_NO_SCRATCH_RECLAIM=1" "-ngl 999 -ub 2048 -b 4096 -m $QWEN"
done
docker run --rm -v $W:/w --entrypoint chown $LIMG -R "$(id -u):$(id -g)" /w/vcache
python3 - <<'PY'
import json, os, statistics as st
R = "/mnt/llm-storage/localai/bench/runs"
for a in ("vllm_mtp3", "vllm_nomtp", "ours_mtp3", "dave"):
    runs = [json.load(open(f"{R}/d4_{a}_{r}.oai.json")) for r in "ab" if os.path.exists(f"{R}/d4_{a}_{r}.oai.json")]
    if not runs: print(f"  {a:10s} no data"); continue
    rows = [{q["prompt"]: q for q in x["rows"]} for x in runs]
    per = [st.mean(x[k]["decode_tps"] for k in ("code", "reason", "edit", "long")) for x in rows]
    pf = [x["longpf"]["prefill_tps"] for x in rows]; pd = [x["longpf"]["decode_tps"] for x in rows]
    print(f"  {a:10s} decode {st.mean(per):7.2f} (spread {max(per)-min(per):5.2f}) | " +
          " ".join(f"{k} {st.mean(x[k]['decode_tps'] for x in rows):6.1f}" for k in ("code", "reason", "edit", "long")) +
          f" | longpf {rows[0]['longpf']['prompt_tokens']} tok prefill {st.mean(pf):7.1f} (spread {max(pf)-min(pf):5.1f}) decode-after {st.mean(pd):6.2f}")
PY
echo DAVE4_DONE

#!/bin/bash
# DAVE5: long-prompt prefill sweep. dave3 showed Dave's build reading a 9.3k prompt at ~4170 t/s (2.2 s): too short to separate
# builds. Here: 32k / 64k / 128k-token prompts, one server start per arm, every prompt timed by oaiclient.py (client clock,
# prefill = prompt tokens / time to first token, decode = the 128 tokens after it), same corpus prefix for every arm.
# PRE-REGISTERED (written before any run):
#  R  resident, Qwen3.6 on one MI210 (renderD128), -c 135168: stock (-ub 512), dave (his -ub 2048 -b 4096), ours (-ub 512 -b 2048),
#     ours -ub 2048 -b 4096 (isolates batch size from code), vllm GPTQ-Int4 (max-model-len 135168, no MTP: prefill only).
#  O  experts in RAM (-ot exps=CPU, pinned), same model: stock, dave, ours prefill mode (-ub 128 -ubp 2048 -b 4096, cache 128).
#  COMBO arms (added before any run): R combo -ub 2048 -b 4096; O combo prefill mode (cache 128) and O combo -ub 2048 -b 4096.
#  2 reps per arm (a/b interleaved). Metric per depth: prefill t/s, then decode t/s after it. WIN rule as dave3 (beat the other
#  arm by more than the larger a/b spread). Corpus = prose_big.txt + wiki.test.raw (prose alone ends near 105k tokens).
set -u
W=/mnt/llm-storage/localai; A=renderD128; LIMG=llama-rocm714-rpc:tune; VIMG=local/vllm-mi210:rocm10-mi210.7-aiter-jitwarm
QWEN=/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf
OURS_ENV="-e GGML_CUDA_FA_TILE_MIN_BATCH=32 -e GGML_CUDA_FA_MMA_MAX_KV=4096 -e GGML_SCHED_MOE_PREFETCH=1 -e GGML_OP_OFFLOAD_MIN_BATCH=32"
CTX=135168; CORP=$W/bench/corpus/long_mix.txt
cd $W/bench || exit 1; mkdir -p runs/logs
[ -s $CORP ] || cat corpus/prose_big.txt /mnt/llm-storage/wiki.test.raw > $CORP
up() { for _ in $(seq 1 600); do curl -sf localhost:8099/health >/dev/null 2>&1 && return 0
  [ "$(docker inspect -f '{{.State.Running}}' localai-srv 2>/dev/null)" = true ] || return 1; sleep 2; done; return 1; }
sweep() { # sweep LABEL MODEL_NAME
  curl -s localhost:8099/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$2\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"max_tokens\":16,\"temperature\":0}" >/dev/null
  for c in 140000 280000 560000; do
    OUT=$W/bench/runs MODEL_NAME=$2 LONGPF=$c LONGPF_ONLY=1 CORPUS=$CORP python3 oaiclient.py "${1}_$c" 2>&1 | sed 's/^/    /'
  done
  docker logs localai-srv > runs/logs/$1.server.log 2>&1; docker rm -f localai-srv >/dev/null 2>&1; }
llama() { # llama LABEL TREE "env" "args"
  docker rm -f localai-srv >/dev/null 2>&1; echo "##### $1 | $(date +%T)"
  # shellcheck disable=SC2086
  docker run -d --name localai-srv --network host --cpuset-cpus 0-23 --device /dev/kfd --device /dev/dri/$A --group-add 44 --group-add 991 \
    --user "$(id -u):$(id -g)" $3 -v $W:/w -v /mnt/llm-storage:/models:ro --entrypoint /w/$2/bin/llama-server $LIMG \
    -fa on -c $CTX -t 24 --parallel 1 --port 8099 --host 127.0.0.1 --jinja --cache-ram 0 $4 >/dev/null
  up && sweep "$1" m || { echo "    $1: SERVER FAILED"; docker logs localai-srv > runs/logs/$1.server.log 2>&1; docker rm -f localai-srv >/dev/null 2>&1; }; }
vllm() { # vllm LABEL
  docker rm -f localai-srv >/dev/null 2>&1; echo "##### $1 | $(date +%T)"
  docker run -d --name localai-srv --network host --ipc host --cpuset-cpus 0-23 --device /dev/kfd --device /dev/dri/$A \
    --group-add 44 --group-add 991 --security-opt seccomp=unconfined -e HSA_NO_SCRATCH_RECLAIM=1 \
    -v $W:/w -v $W/vcache:/cache --entrypoint /usr/local/bin/mi210-entrypoint $VIMG \
    serve /w/models/qwen36-gptq-int4 --served-model-name m --port 8099 --host 127.0.0.1 --max-model-len $CTX --max-num-seqs 1 \
    --gpu-memory-utilization 0.90 --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' --no-enable-prefix-caching >/dev/null
  up && sweep "$1" m || { echo "    $1: SERVER FAILED: $(docker logs localai-srv 2>&1 | grep -iE 'error|exception' | tail -2 | cut -c1-200)"; docker logs localai-srv > runs/logs/$1.server.log 2>&1; docker rm -f localai-srv >/dev/null 2>&1; }; }
for r in a b; do
  llama d5_R_stock_$r  base/build        ""                         "-ngl 999 -ub 512 -b 2048 -m $QWEN"
  llama d5_R_dave_$r   dave/build        "-e HSA_NO_SCRATCH_RECLAIM=1" "-ngl 999 -ub 2048 -b 4096 -m $QWEN"
  llama d5_R_ours_$r   patched/build-df  "$OURS_ENV"                "-ngl 999 -ub 512 -b 2048 -m $QWEN"
  llama d5_R_oursub_$r patched/build-df  "$OURS_ENV"                "-ngl 999 -ub 2048 -b 4096 -m $QWEN"
  llama d5_R_combo_$r  combo/build       "$OURS_ENV"                "-ngl 999 -ub 2048 -b 4096 -m $QWEN"
  vllm  d5_R_vllm_$r
done
for r in a b; do
  llama d5_O_stock_$r  base/build        ""                         "-ngl 999 -ot exps=CPU -lm none -ub 512 -b 2048 -m $QWEN"
  llama d5_O_dave_$r   dave/build        "-e HSA_NO_SCRATCH_RECLAIM=1" "-ngl 999 -ot exps=CPU --no-mmap -ub 2048 -b 4096 -m $QWEN"
  llama d5_O_ours_$r   patched/build-df  "$OURS_ENV"                "-ngl 999 -ot exps=CPU -lm none --moe-expert-cache 128 -ub 128 -b 4096 -ubp 2048 -m $QWEN"
  llama d5_O_comboc_$r combo/build       "$OURS_ENV"                "-ngl 999 -ot exps=CPU -lm none --moe-expert-cache 128 -ub 128 -b 4096 -ubp 2048 -m $QWEN"
  llama d5_O_combo_$r  combo/build       "$OURS_ENV"                "-ngl 999 -ot exps=CPU -lm none -ub 2048 -b 4096 -m $QWEN"
done
docker run --rm -v $W:/w --entrypoint chown $LIMG -R "$(id -u):$(id -g)" /w/vcache
python3 - <<'PY'
import json, os, statistics as st
R = "/mnt/llm-storage/localai/bench/runs"
for s, arms in (("R", ("stock", "dave", "ours", "oursub", "combo", "vllm")), ("O", ("stock", "dave", "ours", "comboc", "combo"))):
    print(f"== {s}")
    for a in arms:
        cells = []
        for c in (140000, 280000, 560000):
            rows = [json.load(open(p))["rows"][0] for p in (f"{R}/d5_{s}_{a}_{r}_{c}.oai.json" for r in "ab") if os.path.exists(p)]
            if not rows: cells.append(f"{c//1000}k-chars: -"); continue
            pf = [x["prefill_tps"] for x in rows]; dc = [x["decode_tps"] for x in rows]
            cells.append(f"{rows[0]['prompt_tokens']//1000}k tok pf {st.mean(pf):7.1f}±{(max(pf)-min(pf)):5.1f} dec {st.mean(dc):6.1f}")
        print(f"  {a:7s} " + " | ".join(cells))
PY
echo DAVE5_DONE

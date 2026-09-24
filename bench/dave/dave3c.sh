#!/bin/bash
# DAVE3C: like-for-like with Dave's build: NO draft model (his base predates MTP; in dave3/3b our arms ran MTP, whose head
# also reads the whole prompt, so their prefill carried an extra cost Dave's did not). Every arm here: no -md, Dave's batch
# config -ub 2048 -b 4096, no expert cache. S1 resident + S2 experts in RAM, 2 reps: dave, ours, combo. Pre-registered, same
# WIN rule as dave3. (dave3b header follows)
# DAVE3B: dave3 + a fourth build, COMBO = ours 0001-0031 + Dave's 04,06,09,10,11,13-16 (tree f8130ba;
# 08 superseded upstream, 12 = 2-GPU state restore, not exercised), Dave's cmake flags. Only the combo arms run, plus ONE
# stock arm per scenario as the drift anchor (dave3 already holds stock / dave / ours). Written before any run.
#  combo arms: S1 +MTP -ub 512 -b 2048 and +MTP -ub 2048 -b 4096 | S2 +MTP cache 128 + prefill mode, and +MTP no cache
#  -ub 2048 -b 4096 | S3 cache 96 + prefill mode, and no cache -ub 2048 -b 4096 | S4 -sm layer -ub 2048 -b 4096.
#  Rule as dave3 (beat the other build by more than the larger a/b spread), valid only if the stock anchor lands inside
#  dave3's stock spread +- 2% (else the session drifted and combo is compared to the anchor only).
# (dave3 header follows)
# DAVE3: four llama.cpp builds head to head on the MI210s, each at its recommended config.
#   stock    = ggml-org e613ef2, default HIP flags (dave1's base)
#   dave     = ggml-org 030ebb558 + davetha/llama.cpp-mi210 patches 04-16, his cmake flags (MMQ_MFMA, HIP graphs, NO_VMM),
#              his runtime config (-ub 2048 -b 4096, HSA_NO_SCRATCH_RECLAIM=1, GGML_CUDA_REGISTER_HOST=1 on 2 GPUs). His base
#              predates MTP drafting, so his arms run without speculative decoding.
#   ours     = e613ef2 + our series 0001-0031 built with HIS cmake flags (build-df; dave1 used default flags)
# PRE-REGISTERED (written before any run):
#  S1 Qwen3.6-35B-A3B Q4_K_M resident on one MI210 (renderD128). Arms: stock+MTP, dave, ours+MTP (-ub 512), ours+MTP -ub 2048,
#     ours no-MTP (engine-only control vs dave). Metrics: decode mean over 4 prompts, 9.3k prefill, decode after 9.3k.
#  S2 same model, experts in RAM, all arms -lm none / --no-mmap (pinned host experts; upstream's own warning for -ot CPU):
#     stock+MTP, dave, ours+MTP no cache (+ prefill mode), ours+MTP cache 128 (+ prefill mode).
#  S3 DeepSeek-V4 REAP 145B one MI210, experts in RAM, no-mmap: stock, dave, ours cache 96 + prefill mode.
#  S4 same 145B on BOTH MI210s (-sm layer; Dave's Bonsai container stopped for S4 only, restarted after): stock, dave, ours.
#  Rule: a build WINS a metric vs another if its mean beats the other's by more than the larger of the two a/b spreads.
#  Texts: sha256 vs stock per prompt; builds with different kernels are expected to differ -> speed only, accuracy not claimed.
set -u
W=/mnt/llm-storage/localai; IMG=llama-rocm714-rpc:tune; A=renderD128; B=renderD131
QWEN=/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf
DSV4=/models/dsv4-reap145-gguf/DSV4-Flash-Vision-Exp-REAP-145B-MXFP4-Q8_0.gguf
HEAD="-md /w/models/mtp-Qwen3.6-35B-A3B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 3"
OURS_ENV="-e GGML_CUDA_FA_TILE_MIN_BATCH=32 -e GGML_CUDA_FA_MMA_MAX_KV=4096 -e GGML_SCHED_MOE_PREFETCH=1 -e GGML_OP_OFFLOAD_MIN_BATCH=32 -e LLAMA_MTP_VOCAB_FILE=/w/models/mtp-Qwen3.6-35B-A3B-vocab49k.bin"
DAVE_ENV="-e HSA_NO_SCRATCH_RECLAIM=1"
declare -A BIN=([stock]=base/build [dave]=dave/build [ours]=patched/build-df [combo]=combo/build)
cd $W/bench || exit 1; mkdir -p runs/logs
arm() { # arm LABEL BUILD "devices" "docker env" CTX "server args"
  local label=$1 b=$2 devs=$3 envs=$4 ctx=$5 args=$6 d dargs=""
  for d in $devs; do dargs="$dargs --device /dev/dri/$d"; done
  docker rm -f localai-srv >/dev/null 2>&1
  echo "##### $label | $(date +%T)"
  # shellcheck disable=SC2086
  docker run -d --name localai-srv --network host --cpuset-cpus 0-23 --device /dev/kfd $dargs --group-add 44 --group-add 991 \
    --user "$(id -u):$(id -g)" $envs -v $W:/w -v /mnt/llm-storage:/models:ro --entrypoint /w/${BIN[$b]}/bin/llama-server $IMG \
    -fa on -c $ctx -t 24 --parallel 1 --port 8099 --host 127.0.0.1 --jinja $args >/dev/null
  for _ in $(seq 1 450); do curl -sf localhost:8099/health >/dev/null 2>&1 && break
    [ "$(docker inspect -f '{{.State.Running}}' localai-srv 2>/dev/null)" = true ] || break
    sleep 2; done
  if curl -sf localhost:8099/health >/dev/null 2>&1; then
    local vram; vram=$(for d in $devs; do echo -n "$(( $(cat /sys/class/drm/$d/device/mem_info_vram_used) / 1048576 ))MiB "; done)
    OUT=$W/bench/runs EDIT=1 LONG=1 GEN=300 python3 specclient.py "$label" "$vram" 2>&1 | grep -E "^$label|Error" | cut -c1-120 | sed 's/^/    /'
    echo -n "    "; OUT=$W/bench/runs CORPUS=$W/bench/corpus/prose_big.txt python3 longpf.py "${label}_pf" 40000 2>&1 | tail -1
  else echo "    $label: SERVER DID NOT COME UP"; fi
  docker logs localai-srv > runs/logs/$label.server.log 2>&1
  echo "    running=$(docker inspect -f '{{.State.Running}}' localai-srv) $(grep -oE 'hit rate [0-9.]+%' runs/logs/$label.server.log | tail -1)"
  docker rm -f localai-srv >/dev/null 2>&1
}
for r in a b; do
  arm d3c_s1_dave_$r   dave  "$A" "$DAVE_ENV" 12288 "-ngl 999 -ub 2048 -b 4096 -m $QWEN"
  arm d3c_s1_ours_$r   ours  "$A" "$OURS_ENV" 12288 "-ngl 999 -ub 2048 -b 4096 -m $QWEN"
  arm d3c_s1_combo_$r  combo "$A" "$OURS_ENV" 12288 "-ngl 999 -ub 2048 -b 4096 -m $QWEN"
done
for r in a b; do
  arm d3c_s2_dave_$r   dave  "$A" "$DAVE_ENV" 12288 "-ngl 999 -ot exps=CPU --no-mmap -ub 2048 -b 4096 -m $QWEN"
  arm d3c_s2_ours_$r   ours  "$A" "$OURS_ENV" 12288 "-ngl 999 -ot exps=CPU -lm none -ub 2048 -b 4096 -m $QWEN"
  arm d3c_s2_combo_$r  combo "$A" "$OURS_ENV" 12288 "-ngl 999 -ot exps=CPU -lm none -ub 2048 -b 4096 -m $QWEN"
done
python3 - <<'PY'
import json, os, statistics as st
R = "/mnt/llm-storage/localai/bench/runs"
def load(p): return json.load(open(p)) if os.path.exists(p) else None
for s in ("s1", "s2"):
    print(f"== {s} (no draft model, -ub 2048 -b 4096)")
    for a in ("dave", "ours", "combo"):
        runs = [x for x in (load(f"{R}/d3c_{s}_{a}_{r}.client.json") for r in "ab") if x]
        pfs = [x for x in (load(f"{R}/d3c_{s}_{a}_{r}_pf.longpf.json") for r in "ab") if x]
        if not runs: print(f"  {a:6s} no data"); continue
        per = [st.mean(q["decode_tps"] for q in x["rows"]) for x in runs]
        pf = [x["prefill_tps"] for x in pfs]; af = [x["decode_tps"] for x in pfs]
        print(f"  {a:6s} decode {st.mean(per):6.1f} (spread {max(per)-min(per):4.1f}) | pf9.3k {st.mean(pf):7.1f} (spread {max(pf)-min(pf):5.1f}) after {st.mean(af):6.1f}")
PY
echo DAVE3C_DONE

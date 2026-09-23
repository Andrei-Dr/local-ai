#!/bin/bash
# DAVE1: stock llama.cpp (e613ef2) vs e613ef2 + local-ai series 0001-0030 on a friend's box: AMD Instinct MI210 64 GB (gfx90a,
# HBM2e 1.6 TB/s, matrix cores), EPYC 74F3 24C, 512 GB DDR4-8ch, PCIe 4.0. Runs on the host; each arm is a llama-server
# container that sees ONE MI210 render node (two for S4), pinned to cores 0-23.
# PRE-REGISTERED (written before any run):
#  S1 resident, Qwen3.6-35B-A3B HauhauCS Q4_K_M (21 GB) fully on one MI210, MTP n3: patched vs base = the series' effect when
#     VRAM is plentiful. Expect: decode +0..8% (FR-Spec draft vocab is the only live lever), prefill ~0. WIN if decode mean over
#     4 prompts beats base by > the base a/b spread.
#  S2 experts in RAM, same model, -ot exps=CPU: base (experts always on the CPU) vs patched --moe-expert-cache 64 / 128 + prefill
#     mode -ubp 2048. Expect: decode >> base (cache hits on HBM), prefill 9.3k several x base. WIN rule as S1, per metric.
#  S3 beefy, DeepSeek-V4-Flash REAP 145B MXFP4/Q8_0 (82.7 GB) on ONE MI210 (does not fit: experts must offload): base -ot exps=CPU
#     vs patched --moe-expert-cache N. First question = does the cache path run on this architecture at all (record, not assumed).
#  S4 the same 145B across BOTH MI210s, fully resident (-sm layer): base vs patched = the no-offload ceiling on this box.
#  Accuracy: temp 0; per prompt, text sha256 of patched vs base is reported (identical / differs); a speed WIN whose texts differ
#  is only "not proven" until a KLD run.
set -u
W=/mnt/llm-storage/localai; IMG=llama-rocm714-rpc:tune
A=renderD128; B=renderD131            # MI210 c3:00.0 and 86:00.0 (renderD129/130 = Dave's R9700s, never touched)
QWEN=/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf
DSV4=/models/dsv4-reap145-gguf/DSV4-Flash-Vision-Exp-REAP-145B-MXFP4-Q8_0.gguf
HEAD="-md /w/models/mtp-Qwen3.6-35B-A3B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 3"
STABLE_ENV="-e GGML_CUDA_FA_TILE_MIN_BATCH=32 -e GGML_CUDA_FA_MMA_MAX_KV=4096 -e GGML_SCHED_MOE_PREFETCH=1 -e GGML_OP_OFFLOAD_MIN_BATCH=32 -e LLAMA_MTP_VOCAB_FILE=/w/models/mtp-Qwen3.6-35B-A3B-vocab49k.bin"
cd $W/bench || exit 1
arm() { # arm LABEL TREE "devices" "docker env" CTX "server args" [longpf]
  local label=$1 tree=$2 devs=$3 envs=$4 ctx=$5 args=$6 lp=${7:-1} d dargs=""
  for d in $devs; do dargs="$dargs --device /dev/dri/$d"; done
  docker rm -f localai-srv >/dev/null 2>&1
  echo "##### $label | $(date +%T)"
  # shellcheck disable=SC2086
  docker run -d --name localai-srv --network host --cpuset-cpus 0-23 --device /dev/kfd $dargs --group-add 44 --group-add 991 \
    --user "$(id -u):$(id -g)" $envs -v $W:/w -v /mnt/llm-storage:/models:ro --entrypoint /w/$tree/build/bin/llama-server $IMG \
    -fa on -c $ctx -t 24 --parallel 1 --port 8099 --host 127.0.0.1 --jinja $args >/dev/null
  for _ in $(seq 1 300); do curl -sf localhost:8099/health >/dev/null 2>&1 && break
    [ "$(docker inspect -f '{{.State.Running}}' localai-srv 2>/dev/null)" = true ] || { echo "    $label: SERVER DIED: $(docker logs localai-srv 2>&1 | grep -iE 'error|fail|out of memory|unknown|abort' | tail -2 | tr '\n' ' ' | cut -c1-220)"; return 1; }
    sleep 2; done
  local vram; vram=$(for d in $devs; do echo -n "$(( $(cat /sys/class/drm/$d/device/mem_info_vram_used) / 1048576 ))MiB "; done)
  OUT=$W/bench/runs EDIT=1 LONG=1 GEN=300 python3 specclient.py "$label" "$vram" 2>&1 | grep -E "^$label" | cut -c1-120 | sed 's/^/    /'
  [ "$lp" = 1 ] && { echo -n "    "; OUT=$W/bench/runs CORPUS=$W/bench/corpus/prose_big.txt python3 longpf.py "${label}_pf" 40000 2>&1 | tail -1; }
  echo "    $(docker logs localai-srv 2>&1 | grep -oE 'hit rate [0-9.]+%' | tail -1) vram $vram"
  docker rm -f localai-srv >/dev/null 2>&1
}
for r in a b; do
  arm s1_base_$r  base    "$A" "" 12288 "-ngl 999 -ub 512 -b 2048 -m $QWEN $HEAD"
  arm s1_patch_$r patched "$A" "$STABLE_ENV" 12288 "-ngl 999 -ub 512 -b 2048 -m $QWEN $HEAD"
done
for r in a b; do
  arm s2_base_$r    base    "$A" "" 12288 "-ngl 999 -ot exps=CPU -ub 512 -b 2048 -m $QWEN $HEAD"
  arm s2_c64_$r     patched "$A" "$STABLE_ENV" 12288 "-ngl 999 -ot exps=CPU --moe-expert-cache 64 -ub 128 -b 4096 -ubp 2048 -m $QWEN $HEAD"
  arm s2_c128_$r    patched "$A" "$STABLE_ENV" 12288 "-ngl 999 -ot exps=CPU --moe-expert-cache 128 -ub 128 -b 4096 -ubp 2048 -m $QWEN $HEAD"
done
for r in a b; do
  arm s3_base_$r  base    "$A" "" 8192 "-ngl 999 -ot exps=CPU -ub 512 -b 2048 -m $DSV4"
  arm s3_c_$r     patched "$A" "-e GGML_CUDA_FA_TILE_MIN_BATCH=32 -e GGML_SCHED_MOE_PREFETCH=1 -e GGML_OP_OFFLOAD_MIN_BATCH=32" 8192 "-ngl 999 -ot exps=CPU --moe-expert-cache 96 -ub 128 -b 4096 -ubp 2048 -m $DSV4"
done
arm s4_base   base    "$A $B" "" 8192 "-ngl 999 -sm layer -ub 512 -b 2048 -m $DSV4"
arm s4_patch  patched "$A $B" "-e GGML_CUDA_FA_TILE_MIN_BATCH=32 -e GGML_OP_OFFLOAD_MIN_BATCH=32" 8192 "-ngl 999 -sm layer -ub 512 -b 2048 -m $DSV4"
python3 - <<'PY'
import json, glob, os, statistics as st
R = "/mnt/llm-storage/localai/bench/runs"
def spec(l):
    p = f"{R}/{l}.client.json"
    return {r["prompt"]: r for r in json.load(open(p))["rows"]} if os.path.exists(p) else None
def lp(l):
    p = f"{R}/{l}_pf.longpf.json"
    return json.load(open(p)) if os.path.exists(p) else None
for s, arms in (("s1", ("base", "patch")), ("s2", ("base", "c64", "c128")), ("s3", ("base", "c")), ("s4", ("base", "patch"))):
    rs = ("_a", "_b") if s != "s4" else ("",)
    print(f"== {s}")
    base_txt = {}
    for a in arms:
        runs = [spec(f"{s}_{a}{r}") for r in rs]
        runs = [x for x in runs if x]
        if not runs:
            print(f"  {a}: no data"); continue
        kinds = list(runs[0])
        m = st.mean(st.mean(x[k]["decode_tps"] for k in kinds) for x in runs)
        spread = (max(st.mean(x[k]["decode_tps"] for k in kinds) for x in runs) - min(st.mean(x[k]["decode_tps"] for k in kinds) for x in runs)) if len(runs) > 1 else 0
        pfs = [lp(f"{s}_{a}{r}") for r in rs]; pfs = [x for x in pfs if x]
        txt = {k: runs[0][k]["text_sha256"] for k in kinds}
        if a == "base":
            base_txt = txt
        same = sum(txt[k] == base_txt.get(k) for k in kinds)
        pf = f"prefill 9.3k {st.mean(x['prefill_tps'] for x in pfs):.0f} decode-after {st.mean(x['decode_tps'] for x in pfs):.1f}" if pfs else "no 9.3k"
        print(f"  {a:6s} decode {m:6.2f} (spread {spread:.2f}) | " + " ".join(f"{k} {st.mean(x[k]['decode_tps'] for x in runs):.1f}" for k in kinds) + f" | {pf} | texts = base {same}/{len(kinds)}")
PY
echo DAVE1_DONE

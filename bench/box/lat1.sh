#!/bin/bash
# LAT1: zero-code latency / transfer levers read off the telemetry (2026-09-20): during decode the GPU shows 70-99% "util" at only
# 45-65 W of 100 W (= busy-waiting on tiny kernels, syncs and copies, not computing), host->device peaks at 3-7 GB/s (pageable
# memory; PCIe 3.0 x16 can do ~12), and PREFILL runs at ~39 tok/s on Qwen (every 128-token ubatch streams ~all experts, ~10 GB,
# over that slow path) = ~48 s to first token on a 1.9k-token prompt. One variable per row, same binary, Qwen c30 + MTP n=2.
# Prompts: code, reason, edit (377 tok), long (W4, ~1.9k tok) => prefill_tps per prompt is in the ledger row.
source /ai/bench/preflight.sh || exit 1
cd /ai/bench
export BUILD=/ai/src/llama.cpp-mainline/build75 EDIT=1 LONG=1 GEN=200
M=/ai/models; Q=$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
QH="-md $M/mtp-Qwen3.6-35B-A3B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 2"
run() { local envs=$1 off=$2 l=$3; shift 3; echo "##### $l | env[$envs] OFFLOAD=$off | $*"; env $envs MODEL=$Q OFFLOAD=$off ./specbench.sh 999 "$l" -ot exps=CPU "$@" 2>&1; grep -hE "MoE expert cache enabled|moe-cache: steps|out of memory|pinned|register" server_$l.log | sort -u | tail -4 | cut -c1-200 | sed -E 's/^[0-9.]+ +[A-Z] +/    /'; python3 - "$l" <<'PY'
import json,sys
try:
    t=json.load(open(f"/ai/bench/runs/{sys.argv[1]}.mon.json")); print("    telemetry:", {k:t.get(k) for k in ("gpu_util_avg","power_w_avg","pcie_rx_gbs_max","dram_read_gbs_avg","cpu_busy_pct","mem_avail_mib_min","swap_used_mib_max")})
except Exception as e: print("    telemetry: n/a", e)
PY
}
C30="--moe-expert-cache 30 -ub 128 -b 256"
run "A=0" 32 lat_base                 $C30 $QH
# 1. are CUDA graphs doing anything for us? (they are keyed per split; mul_mat_id fallbacks and batch > 1 can disable them)
run "GGML_CUDA_DISABLE_GRAPHS=1" 32 lat_nographs $C30 $QH
# 2. pinned host expert buffers: async DMA at full PCIe speed for prefill streaming and cache uploads (watch mem_avail_mib_min)
run "GGML_CUDA_REGISTER_HOST=1" 32 lat_pinned    $C30 $QH
# 3. prefill: bytes streamed per prompt token fall with the ubatch size; costs compute-buffer VRAM (fewer slots)
run "A=0" 32 lat_ub256_c28            --moe-expert-cache 28 -ub 256 -b 512 $QH
run "A=0" 32 lat_ub512_c24            --moe-expert-cache 24 -ub 512 -b 512 $QH
# 4. where should prompts be computed? offload threshold: never (CPU prefill, no streaming) / 32 (today) / 8
run "A=0" 99999 lat_cpu_prefill       $C30 $QH
run "A=0" 8 lat_offload8              $C30 $QH
# 5. OpenMP workers spin while the GPU works (35% of CPU samples were libgomp spin): does it starve the CUDA-driving thread?
run "OMP_WAIT_POLICY=PASSIVE" 32 lat_omp_passive $C30 $QH
run "GOMP_SPINCOUNT=1000" 32 lat_spin1k          $C30 $QH
run "A=0" 32 lat_prio2                $C30 $QH --prio 2
echo LAT1_DONE

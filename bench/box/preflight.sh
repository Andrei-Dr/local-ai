#!/bin/bash
# Sourced by every bench script. Enforces the host perf settings, refuses to measure on a busy or
# memory-starved box (16 GB RAM: a resident no-mmap model + a build wedged it once), logs the state.
# env: PREFLIGHT_MIN_MB (default 12500), PREFLIGHT_ALLOW_BUSY=1 to skip the busy check.
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  [ "$(cat $g)" = performance ] || echo performance > $g
done
grep -q '\[always\]' /sys/kernel/mm/transparent_hugepage/enabled || echo always > /sys/kernel/mm/transparent_hugepage/enabled
if [ -z "$PREFLIGHT_ALLOW_BUSY" ]; then
  busy=$(pgrep -x -l 'llama-server|llama-bench|llama-cli|llama-moe-trace|llama-perplexity|nvcc|cicc|cc1plus|ld' | tr '\n' ' ')
  [ -n "$busy" ] && { echo "PREFLIGHT REFUSED: box is busy: $busy"; exit 1; }
fi
avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
[ "$avail" -lt "${PREFLIGHT_MIN_MB:-12500}" ] && { echo "PREFLIGHT REFUSED: only $avail MiB RAM available (< ${PREFLIGHT_MIN_MB:-12500})"; exit 1; }
echo "    preflight: governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) thp=$(grep -o '\[[a-z]*\]' /sys/kernel/mm/transparent_hugepage/enabled) mem_avail=${avail}MiB swap_used=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo)MiB gpu=$(nvidia-smi --query-gpu=memory.used,temperature.gpu,pstate --format=csv,noheader | tr -d ' ')"

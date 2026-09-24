#!/usr/bin/env bash
# Set the R9700 (PCI device 0x7551) power limits: sustained cap, and - once tuned - a top-clock cap and a voltage offset.
#
# WHY. The R9700s serve the q38fn vLLM. The sustained cap (power1_cap) holds the ~1 s average at 210 W, but the SMU lets short
# bursts through above it (gpu-temp showed up to ~300 W). There is no sysfs knob for the burst limit; what bounds bursts is the
# clock/voltage the card is allowed to reach, which needs OverDrive: amdgpu.ppfeaturemask=0xfff7ffff on the kernel command line
# (= the driver default 0xfff7bfff + the OverDrive bit 0x4000), set in /etc/default/grub.d/99-amdgpu.cfg.
#
# KNOBS (env, or edit the unit). Empty = do not touch.
#   R9700_POWER_CAP_WATTS  sustained cap, default 210
#   R9700_SCLK_MAX_MHZ     top GFX clock (pp_od_clk_voltage "s 1 <MHz>"), empty until tuned
#   R9700_VOLT_OFFSET_MV   voltage offset (pp_od_clk_voltage "vo <mV>", negative = undervolt), empty until tuned
# Matches cards by vendor 0x1002 + device 0x7551 (hwmon numbering is not stable; the MI210s are 0x740f and are never touched).
# Every write is read back; a value that did not apply is a failure.
set -uo pipefail
WATTS="${1:-${R9700_POWER_CAP_WATTS:-210}}"; SCLK="${R9700_SCLK_MAX_MHZ:-}"; VOFF="${R9700_VOLT_OFFSET_MV:-}"
found=0; failed=0
for d in /sys/class/drm/card*/device; do
    [ "$(cat "$d/vendor" 2>/dev/null)" = "0x1002" ] && [ "$(cat "$d/device" 2>/dev/null)" = "0x7551" ] || continue
    found=$((found + 1)); pci=$(basename "$(readlink -f "$d")")
    hw=$(ls -d "$d"/hwmon/hwmon* 2>/dev/null | head -1)
    [ -n "$hw" ] || { echo "$pci: no hwmon" >&2; failed=1; continue; }
    echo $((WATTS * 1000000)) > "$hw/power1_cap"
    got=$(( $(cat "$hw/power1_cap") / 1000000 ))
    [ "$got" = "$WATTS" ] && echo "$pci: power cap $got W" || { echo "$pci: power cap wanted $WATTS W, reads $got W" >&2; failed=1; }
    if [ -n "$SCLK$VOFF" ]; then
        od="$d/pp_od_clk_voltage"
        [ -w "$od" ] || { echo "$pci: no OverDrive ($od missing: is amdgpu.ppfeaturemask set?)" >&2; failed=1; continue; }
        [ -n "$SCLK" ] && echo "s 1 $SCLK" > "$od"
        [ -n "$VOFF" ] && echo "vo $VOFF" > "$od"
        echo c > "$od" || { echo "$pci: OverDrive commit rejected" >&2; failed=1; continue; }
        echo "$pci: OverDrive now:"; sed 's/^/    /' "$od"
    fi
done
[ "$found" -gt 0 ] || { echo "no R9700 (0x7551) found" >&2; exit 1; }
exit $failed

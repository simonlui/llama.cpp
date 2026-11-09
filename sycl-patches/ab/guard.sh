#!/usr/bin/env bash
# Benchmark guard for the Arc A770 (i915 or xe kernel driver), after the bench-guard of
# kainlan/llama.cpp-intel-optimizations, which only reads xe sysfs files.
#
#   guard.sh --stamp          print the setup a result depends on (driver stack, clocks, governor, oneDNN,
#                             temperature, other GPU clients); exit 0
#   guard.sh --wait SECONDS   first wait (up to AB_COOL_WAIT, default 900 s) until the GPU is below
#                             AB_TEMP_START (60 C) and no other process uses an engine above AB_BUSY_MAX (5%):
#                             exit 3 with the reason on stderr when it does not get there. Then wait up to SECONDS
#                             until the GPU is idle, not throttled and no other llama process runs: exit 1 with
#                             the reason when SECONDS pass first (a display on the card can keep it awake), else 0
#   guard.sh --temp           print the GPU temperature in C (nothing when the driver exposes none)
#   guard.sh --clients        print other processes' engine use of the card over one second
#   guard.sh --faults SINCE   count GPU hangs and resets in the kernel log since SINCE (a `date` string)
#
# The card is the first Intel display device with PCI id 0x56xx (DG2) under /sys/class/drm, or AB_CARD
# (for example /sys/class/drm/card1). Run --stamp with every set of numbers: a compute-runtime update alone
# moved the fork's B70 decode by 18%. The temperature and client gates follow an A770 flash-attention study:
# a render job using 10-35% of the engine slowed both sides of its pairs by about 10% at the same clock,
# desktop clients at 3-4% did not, and its runs start below 60 C and stop at 80 C.
set -uo pipefail

find_card() {
    if [ -n "${AB_CARD:-}" ]; then echo "$AB_CARD"; return; fi
    local c
    for c in /sys/class/drm/card[0-9]*; do
        [[ "$c" =~ card[0-9]+$ ]] || continue
        [ "$(cat "$c/device/vendor" 2>/dev/null)" = "0x8086" ] || continue
        case "$(cat "$c/device/device" 2>/dev/null)" in 0x56*) echo "$c"; return;; esac
    done
}

CARD="$(find_card)"
DRIVER="$( [ -n "$CARD" ] && basename "$(readlink -f "$CARD/device/driver")" 2>/dev/null)"

act_freq() {  # MHz the GPU runs at now; 0 when it is idle (RC6)
    case "$DRIVER" in
        i915) cat "$CARD/gt_act_freq_mhz" 2>/dev/null || cat "$CARD/gt/gt0/rps_act_freq_mhz" 2>/dev/null;;
        xe)   cat "$CARD"/device/tile0/gt0/freq0/act_freq 2>/dev/null;;
    esac
}
throttled() {  # 1 when a throttle reason is active
    case "$DRIVER" in
        i915) cat "$CARD/gt/gt0/throttle_reason_status" 2>/dev/null || echo 0;;
        xe)   cat "$CARD"/device/tile0/gt0/freq0/throttle/status 2>/dev/null || echo 0;;
    esac
}
others() { pgrep -a '^llama-' 2>/dev/null || true; }  # comm is cut at 15 characters, so match the prefix
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDEV="$( [ -n "$CARD" ] && basename "$(readlink -f "$CARD/device")" 2>/dev/null)"
gpu_temp() {  # hottest hwmon sensor of the card in whole C; empty when there is none
    local f t max=""
    for f in "$CARD"/device/hwmon/hwmon*/temp*_input; do
        [ -r "$f" ] || continue
        t=$(( $(cat "$f" 2>/dev/null || echo 0) / 1000 ))
        if [ -z "$max" ] || [ "$t" -gt "$max" ]; then max=$t; fi
    done
    echo "$max"
}
clients() {  # "max <percent> <engine> <pid> <comm>", then one line per busy client (drm_clients.py)
    [ -n "$PDEV" ] || { echo "max 0.0 - - -"; return; }
    python3 -I "$HERE/drm_clients.py" --pdev "$PDEV" --interval "${1:-1}" --proc "${AB_PROC:-/proc}" 2>/dev/null ||
        echo "max 0.0 - - -"
}
onednn() {  # oneDNN headers of the environment and the libdnnl the A/B build loads
    local h="${DNNLROOT:-}/include/oneapi/dnnl/dnnl_version.h" lib
    if [ -f "$h" ]; then
        printf '%s' "$(awk '/define DNNL_VERSION_(MAJOR|MINOR|PATCH)/ { printf "%s%s", s, $3; s = "." }' "$h") (DNNLROOT=$DNNLROOT)"
    fi
    if [ -n "${AB_BUILD:-}" ] && [ -x "$AB_BUILD/bin/llama-bench" ]; then
        lib="$(ldd "$AB_BUILD/bin/llama-bench" 2>/dev/null | awk '/libdnnl/ { print $3; exit }')"
        [ -n "$lib" ] && printf ' loaded=%s' "$(readlink -f "$lib")"
    fi
    echo
}

stamp() {
    echo "date:      $(date -Is)"
    echo "kernel:    $(uname -r)"
    echo "card:      ${CARD:-not found} driver=${DRIVER:-?}${CARD:+ pci=$(basename "$(readlink -f "$CARD/device")")}"
    if [ "$DRIVER" = i915 ]; then
        echo "clocks:    min=$(cat "$CARD/gt_min_freq_mhz" 2>/dev/null) max=$(cat "$CARD/gt_max_freq_mhz" 2>/dev/null) RP0=$(cat "$CARD/gt_RP0_freq_mhz" 2>/dev/null) MHz"
    fi
    echo "governor:  $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo ?)"
    echo "gpu temp:  $(gpu_temp) C"
    echo "clients:   $(clients 1 | paste -sd ';')"
    echo "onednn:    $(onednn)"
    if command -v sycl-ls >/dev/null; then
        echo "sycl-ls:   $(sycl-ls 2>/dev/null | grep -i 'level_zero:gpu' | head -2 | tr -s ' ' | paste -sd ';')"
    fi
    if command -v icpx >/dev/null; then
        echo "icpx:      $(icpx --version 2>/dev/null | head -1)"
    fi
    if command -v dpkg-query >/dev/null; then
        echo "packages:  $(dpkg-query -W -f='${Package}=${Version} ' 'intel-igc-core*' 'intel-opencl-icd' 'libze-intel-gpu1' 'libze1' 'intel-level-zero-gpu' 2>/dev/null)"
    elif command -v rpm >/dev/null; then
        echo "packages:  $(rpm -q intel-igc-core intel-opencl intel-level-zero-gpu level-zero 2>/dev/null | grep -v 'not installed' | tr '\n' ' ')"
    fi
    local repo
    repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    echo "tree:      $(git -C "$repo" describe --always --dirty 2>/dev/null) ($(git -C "$repo" diff --stat 2>/dev/null | tail -1 | sed 's/^ *//'))"
}

case "${1:-}" in
    --stamp)
        stamp
        ;;
    --temp)
        gpu_temp
        ;;
    --clients)
        clients 1
        ;;
    --wait)
        limit="${2:-60}"
        [ -n "$CARD" ] || { echo "guard: no DG2 card found under /sys/class/drm (set AB_CARD)" >&2; exit 1; }
        temp_start="${AB_TEMP_START:-60}"; busy_max="${AB_BUSY_MAX:-5}"; cool="${AB_COOL_WAIT:-900}"
        waited=0
        while :; do  # hard conditions: a hot card or another GPU job makes the next number worthless
            tc="$(gpu_temp)"; cl="$(clients 1)"
            read -r _ pct eng pid comm <<< "$(head -1 <<< "$cl")"
            hot=0; [ -n "$tc" ] && [ "$tc" -ge "$temp_start" ] && hot=1
            busy=$(awk -v p="$pct" -v m="$busy_max" 'BEGIN { print (p > m) ? 1 : 0 }')
            [ "$hot" = 0 ] && [ "$busy" = 0 ] && break
            if [ "$waited" -ge "$cool" ]; then
                echo "guard: after ${cool}s temp=${tc:-?}C (start below ${temp_start}C), busiest other client ${comm} (pid ${pid}) at ${pct}% of ${eng} (limit ${busy_max}%)" >&2
                exit 3
            fi
            sleep 4; waited=$((waited + 5))
        done
        waited=0
        while :; do
            f="$(act_freq)"; t="$(throttled)"; o="$(others)"
            if [ "${f:-0}" = 0 ] && [ "${t:-0}" = 0 ] && [ -z "$o" ]; then exit 0; fi
            if [ "$waited" -ge "$limit" ]; then
                # a display on this card can keep it from idling: say so rather than refuse to measure
                echo "guard: after ${limit}s act_freq=${f:-?}MHz throttle=${t:-?} others=[${o//$'\n'/; }]" >&2
                exit 1
            fi
            sleep 2; waited=$((waited + 2))
        done
        ;;
    --faults)
        since="${2:-1 hour ago}"
        if journalctl -k --since "$since" -q >/dev/null 2>&1; then
            journalctl -k --since "$since" -q 2>/dev/null | grep -ciE 'gpu hang|i915.*reset|xe.*(reset|timedout)|guc.*(fail|err)'
        else
            echo "guard: cannot read the kernel log (journalctl -k); check dmesg by hand" >&2
            exit 2
        fi
        ;;
    *)
        awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
        exit 2
        ;;
esac

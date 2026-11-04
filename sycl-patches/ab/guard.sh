#!/usr/bin/env bash
# Benchmark guard for the Arc A770 (i915 or xe kernel driver), after the bench-guard of
# kainlan/llama.cpp-intel-optimizations, which only reads xe sysfs files.
#
#   guard.sh --stamp          print the setup a result depends on (driver stack, clocks, governor); exit 0
#   guard.sh --wait SECONDS   wait until the GPU is idle and not throttled and no other llama process runs;
#                             exit 0 when it is, 1 (with the reason on stderr) when SECONDS pass first
#   guard.sh --faults SINCE   count GPU hangs and resets in the kernel log since SINCE (a `date` string)
#
# The card is the first Intel display device with PCI id 0x56xx (DG2) under /sys/class/drm, or AB_CARD
# (for example /sys/class/drm/card1). Run --stamp with every set of numbers: a compute-runtime update alone
# moved the fork's B70 decode by 18%.
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

stamp() {
    echo "date:      $(date -Is)"
    echo "kernel:    $(uname -r)"
    echo "card:      ${CARD:-not found} driver=${DRIVER:-?}${CARD:+ pci=$(basename "$(readlink -f "$CARD/device")")}"
    if [ "$DRIVER" = i915 ]; then
        echo "clocks:    min=$(cat "$CARD/gt_min_freq_mhz" 2>/dev/null) max=$(cat "$CARD/gt_max_freq_mhz" 2>/dev/null) RP0=$(cat "$CARD/gt_RP0_freq_mhz" 2>/dev/null) MHz"
    fi
    echo "governor:  $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo ?)"
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
    --wait)
        limit="${2:-60}"
        [ -n "$CARD" ] || { echo "guard: no DG2 card found under /sys/class/drm (set AB_CARD)" >&2; exit 1; }
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
        sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac

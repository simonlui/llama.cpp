#!/usr/bin/env bash
# List the Intel GPU architecture gates in ggml-sycl at a git ref and compare them with the
# snapshot taken at the last refresh, so new Battlemage/Lunar Lake-only paths are noticed.
#
#   bash sycl-patches/check_arch_gates.sh [REF]          # default REF: upstream/master
#   bash sycl-patches/check_arch_gates.sh --update [REF] # rewrite the snapshot after review
#
# Each gate line is tagged HW when its file uses DPAS, joint_matrix, ESIMD or 2D block loads:
# such a gate usually guards sub-group-16 DPAS or Xe2-only loads, and adding DG2 to it can
# compile cleanly yet run wrong. Untagged gates usually only pick between kernels that are
# built for every device (tuning gates), see GGML_SYCL_DG2_BMG_TUNING in the candidates.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAPSHOT="$SCRIPT_DIR/arch_gates.txt"

update=0
if [ "${1:-}" = "--update" ]; then
    update=1
    shift
fi
REF="${1:-upstream/master}"
cd "$SCRIPT_DIR/.."
git rev-parse --verify --quiet "$REF^{commit}" >/dev/null || { echo "ERROR: unknown ref $REF"; exit 1; }

SRC=ggml/src/ggml-sycl
GATE_RE='gpu_arch::intel_gpu_[a-z0-9_]+|xmx_supports_arch|GGML_SYCL_XMX_AOT_SG|ext_oneapi_architecture_is'
HW_RE='xmx::dpas|joint_matrix|"dpas\.|load_2d|ext::intel::esimd'

current=$(mktemp)
trap 'rm -f "$current"' EXIT

git grep -nE "$GATE_RE" "$REF" -- "$SRC" ':!'"$SRC"'/sycl_hw.cpp' |
    while IFS= read -r hit; do
        rest=${hit#"$REF:"}
        file=${rest%%:*}
        rest=${rest#*:}
        text=${rest#*:}
        # skip enum and comment-only lines
        case "$(echo "$text" | sed 's/^[[:space:]]*//')" in
            //*|\**) continue ;;
        esac
        tag="  "
        if git grep -qE "$HW_RE" "$REF" -- "$file"; then
            tag="HW"
        fi
        printf '%s %s: %s\n' "$tag" "${file#"$SRC"/}" "$(echo "$text" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//; s/[[:space:]]\+/ /g')"
    done | sort -u > "$current"

if [ "$update" -eq 1 ]; then
    {
        echo "# ggml-sycl architecture gates at $(git rev-parse --short "$REF") ($(git log -1 --format=%cs "$REF"))"
        echo "# regenerate: bash sycl-patches/check_arch_gates.sh --update [REF]"
        cat "$current"
    } > "$SNAPSHOT"
    echo "Snapshot updated: $(grep -vc '^#' "$SNAPSHOT") gate lines at $REF."
    exit 0
fi

if [ ! -e "$SNAPSHOT" ]; then
    echo "No snapshot yet; current gates at $REF:"
    cat "$current"
    exit 0
fi

new=$(comm -13 <(grep -v '^#' "$SNAPSHOT" | sort -u) "$current" || true)
gone=$(comm -23 <(grep -v '^#' "$SNAPSHOT" | sort -u) "$current" || true)
echo "Architecture gates at $REF vs snapshot ($(head -1 "$SNAPSHOT" | sed 's/^# ggml-sycl architecture gates at //')):"
if [ -z "$new" ] && [ -z "$gone" ]; then
    echo "  no change"
    exit 0
fi
if [ -n "$new" ]; then
    echo "NEW (classify each as tuning or HW before widening it to DG2):"
    echo "$new" | sed 's/^/  + /'
fi
if [ -n "$gone" ]; then
    echo "REMOVED or changed:"
    echo "$gone" | sed 's/^/  - /'
fi

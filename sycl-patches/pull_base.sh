#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly TARGET_DIR="$SCRIPT_DIR"
readonly API_URL="https://api.github.com/repos/aicss-genai/llama.cpp/contents/aicss-genai/patches?ref=aicss-genai-fork"

echo "Checking for base patches from aicss-genai..."

# -f fails on HTTP errors instead of parsing an error page as a listing
RESPONSE=$(curl -sfS "$API_URL")

if jq -e 'type == "object" and .message' <<< "$RESPONSE" >/dev/null; then
    echo "ERROR: GitHub API returned an error: $(jq -r '.message' <<< "$RESPONSE")"
    exit 1
fi

# Pinned patches are hands-off for this script: never downloaded, never
# overwritten, never deleted. The file on disk is the carry decision:
#   file present -> carried as a locally rebased or local-only patch
#   file absent  -> intentionally not carried (retired / merged upstream)
# The fork's file listing stays the cleanup authority for unpinned names.
declare -A PINNED=(
    # carried: locally rebased or local-only, file exists
    [0002-sycl-remove-dpct-capability-checks.patch]="locally regenerated with unique reverse context so apply/unapply is deterministic; re-anchored on dde879bc onto upstream's rewritten dequant and MMQ launchers (same call sites, checks dropped); on fac3a894 the dmmv hunks drop the check from the two launcher templates that merged PR #31325 introduced"
    [0004-sycl-relax-sum-mean-contiguous.patch]="locally rebased, GROUP_NORM gate only: GROUP_NORM gets its own full-contiguous gate (its kernel isn't stride-generic, unlike NORM/L2_NORM/RMS_NORM which upstream #25044 made stride-generic). The SUM relaxation was dropped on dde879bc: ggml_sycl_op_sum sums ne consecutive floats, so a row-contiguous view with gaps (upstream #28851 test_sum slice=true) would compute wrong data. File name kept so the pin still matches"
    [0007-sycl-barrier-local-space-fence.patch]="locally rebased: ggml-sycl.cpp k_copy_src1_to_contiguous hunk dropped (barrier removed upstream by PR-23142); pinned because the fork sync used to re-download the fork's full copy and clobber the rebase"
    [0008-sycl-native-sycl-shuffles.patch]="locally rebased: obsolete ROLL/ARANGE supports_op relocation dropped; on fac3a894 the dmmv hunks were regenerated after merged PR #31325: all 12 remaining dpct::permute_sub_group_by_xor calls in dmmv.cpp become select_from_group"
    [0014-sycl-getrows-ungate-qk-a770-epsilon.patch]="LOCAL-ONLY patch (not on the fork): re-gates Q2_K/Q4_K/Q5_K GET_ROWS unsupported on A770 after PR-23710 merged; pinned so cleanup does not delete it (the fork API never returns it)"
    [0015-sycl-concat-gate-supported-types.patch]="LOCAL-ONLY patch (not on the fork): gates CONCAT supports_op to the types concat.cpp actually implements (F32/F16/BF16/I32/I16/I64/I8, added by upstream #24363/#24011) after upstream #25247 added quantized-type CONCAT test cases that hit concat.cpp's GGML_ASSERT(false) default and abort; replaces an earlier, narrower 0015 (F32/I32-only) that predated the scalar-type support"
    [0017-sycl-gate-unimplemented-ternary-types.patch]="LOCAL-ONLY patch (not on the fork): the SYCL backend implements no kernel for TQ1_0/TQ2_0 anywhere, but supports_op only checks operand types per op, so test-backend-ops' all_types entry for TQ2_0 aborts in set_rows.cpp, cpy.cpp and mmvq.cpp in turn; one central type gate replaces what would otherwise be three per-op gates"
    # not carried: merged upstream or superseded, no file
    [0001-sycl-battlemage-optimizations.patch]="merged upstream via PR #22147/#22152/#22153/#22156"
    [0003-sycl-pad-non-contiguous-stride.patch]="merged upstream as PR #22148"
    [0005-sycl-rms-norm-mul-fusion.patch]="complete RMS_NORM+MUL path merged upstream as de699957b"
    [0006-sycl-add-new-ops.patch]="merged upstream as PR #22149"
    [0010-sycl-small-matmul-onemkl-direct.patch]="merged upstream as PR #22150"
    [0011-sycl-fuse-unary-mul.patch]="superseded by the richer upstream PR-26411 implementation"
    [0012-sycl-extend-graph-fusion.patch]="superseded by upstream graph changes / disabled locally"
)

PATCH_DATA=$(jq -r '.[] | select(.name | endswith(".patch")) | "\(.name)\t\(.download_url)"' <<< "$RESPONSE")
if [ -z "$PATCH_DATA" ]; then
    echo "ERROR: the fork listing returned no patches; refusing to run cleanup."
    exit 1
fi

# the =() initializers matter: `set -u` treats a declared-but-empty
# associative array as unbound when its count is taken
declare -A KEEP_PATCHES=() LISTED_PATCHES=()

while IFS=$'\t' read -r name url; do
    [ -z "$name" ] && continue
    LISTED_PATCHES["$name"]=1

    if [[ ${PINNED[$name]+_} ]]; then
        echo "Pinned $name (${PINNED[$name]})"
        continue
    fi

    KEEP_PATCHES["$name"]=1
    echo "Syncing $name..."
    # -z downloads only when the fork copy is newer than the local file
    curl -sfSL -z "$TARGET_DIR/$name" -o "$TARGET_DIR/$name" "$url"
done <<< "$PATCH_DATA"

# Protect every pinned file, including local-only patches the fork never
# lists: without this, cleanup would delete them as "stale" on every sync.
for name in "${!PINNED[@]}"; do
    KEEP_PATCHES["$name"]=1
done

# Cleanup: local 0*.patch files that are neither pinned nor on the fork
echo "Cleaning up stale base patches..."
shopt -s nullglob
for local_patch in "$TARGET_DIR"/0*.patch; do
    base_name=$(basename "$local_patch")
    if [[ ! ${KEEP_PATCHES[$base_name]+_} ]]; then
        echo "  Deleting stale patch: $base_name"
        rm "$local_patch"
    fi
done
shopt -u nullglob

# Staleness notes: a pinned name with no local file and no fork listing is
# fully retired and no longer needs its entry.
for name in "${!PINNED[@]}"; do
    if [[ ! ${LISTED_PATCHES[$name]+_} ]] && [ ! -e "$TARGET_DIR/$name" ]; then
        echo "NOTE: pinned $name is neither on the fork nor local; its PINNED entry can be removed."
    fi
done

echo "Base patches sync complete."

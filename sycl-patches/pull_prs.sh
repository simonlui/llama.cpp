#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly TARGET_DIR="$SCRIPT_DIR"
readonly UPSTREAM_REPO="ggml-org/llama.cpp"

# Look back ~2 months to catch all in-flight SYCL PRs. An open PR that ages
# out of this window is safe: the cleanup pass verifies the real PR state
# before it deletes anything.
SINCE_DATE=$(date -d '60 days ago' -u +"%Y-%m-%d")
readonly SINCE_DATE
REFRESH_DATE=$(date -u +"%Y%m%d")
readonly REFRESH_DATE
readonly RETIRED_PATCH_DIR="$SCRIPT_DIR/../patch-backups/upstream-refresh-$REFRESH_DATE/$(basename "$SCRIPT_DIR")"
echo "Searching for SYCL PRs created since $SINCE_DATE..."

# Use authenticated `gh` for all GitHub queries. The old path (raw search API
# + strict jq) aborted on unescaped control characters in PR titles/bodies;
# `gh` parses and re-serializes the JSON itself, so control chars are handled.
if ! command -v gh >/dev/null 2>&1; then
    echo "ERROR: the 'gh' CLI is required (and must be authenticated) to search for PRs."
    exit 1
fi

# Pinned PRs are never downloaded or overwritten. Their current state still
# controls carry: open non-drafts remain local, while merged, closed, and
# draft patches are archived by the cleanup pass.
declare -A PINNED=(
    # carried: locally re-anchored patches that must not be overwritten by a force-push
    [29506]="re-anchored on upstream's sycl-f16-footnote anchor in docs/backend/SYCL.md; PR footnote 2 kept"
    [29672]="re-anchored next to upstream GGML_TYPE_F8_E4M3 (43): PTQ1_0 stays 143, both ftypes and gguf-py entries kept; x86 q2_0 generic alias dropped as upstream did"
    [30206]="re-anchored: test_rope_set_rows keeps upstream n_dims and the PR's yarn/truncate; the PR's yarn call site passes n_dims=128 explicitly so the bools do not bind to n_dims"
    [31285]="re-anchored after PR-30606: the reworked PR now edits the docs/backend/SYCL.md env table, where 30606 adds GGML_SYCL_FA_TILE_DEBUG in its context"
    [31460]="re-anchored after PR-29672: the type whitelist replaces the per-op TQ1_0/TQ2_0/PTQ1_0 blocklists, and PTQ1_0 stays unsupported"
    [31536]="re-anchored on fac3a894: its mask-broadcast test struct sits before upstream's new test_flash_attn_ext_kv_rows (#29510) and its test cases after PR-30606's GQA decode loop"
    [31542]="re-anchored after PR-31536: its gpt-oss / Flint sink test cases follow 31536's, and its perf cases precede the q8_0 long-context block"
)

# Explicit exclusions are distinct from pinned carries and automatic draft or
# merged/closed retirement. Keep the reason here so each open PR is auditable.
declare -A SKIP_PRS=(
    [29676]="CUDA support for PTQ1_0 on top of PR-29672; no SYCL kernel change; not SYCL-primary"
    [30559]="CUDA IQ4_NL GET_ROWS; SYCL only gains a supports_op line; not SYCL-primary"
    [30715]="new JQ*_K quantization types for CPU (2k lines); SYCL only gates the types out; not SYCL-primary"
    [30743]="cross-backend argsort/top-k stride fix (CPU, CUDA, Metal, OpenCL); SYCL only gates; not SYCL-primary"
    [30967]="docs only (README, build.md, SYCL.md); no SYCL backend change"
    [31066]="nix packaging only; no SYCL backend change"
    [31139]="cross-backend max-allocation-size reporting and MoE cache groups across 25 files; the SYCL buffer types get a NULL get_max_alloc_size, so nothing changes on SYCL"
    [31236]="changes the CPU reference MXFP4/NVFP4 rounding and gguf-py quantizers; not SYCL-primary"
    [31540]="CI, Dockerfile and a docs changelog only: oneAPI 2026.2, oneDNN 2026.1 and Level Zero 1.34.0 in the SYCL builds; no backend change"
    [21067]="cross-backend prefetch; not SYCL-primary"
    [21160]="cross-backend profiler; not SYCL-primary"
    [22098]="UMA-only allocation path; not applicable to the target dGPU"
    [25032]="cross-backend tie-breaking change; not SYCL-primary"
    [25206]="cross-backend RWKV fusion; not SYCL-primary"
    [25214]="draft server sleep change; not SYCL-primary"
    [25271]="cross-backend sleep/reset infrastructure"
    [26283]="cross-backend suffix-decoding feature; no SYCL backend change and outside this SYCL performance stack"
)

# Fetch open, SYCL-labeled PRs (filters out our own fork's author). Capture
# the command status explicitly: process substitution hides a failing producer
# from `set -e`, which could otherwise turn a network/API outage into an empty
# result and delete every unpinned local patch during cleanup.
if ! PR_SEARCH_OUTPUT=$(gh search prs --repo "$UPSTREAM_REPO" --state open --label SYCL \
    --created ">=$SINCE_DATE" -L 100 --json number,author,isDraft,title \
    --jq '.[] | select(.author.login != "aicss-genai") | [.number, .isDraft, .title] | @tsv'); then
    echo "ERROR: failed to query upstream PRs; leaving local patches untouched."
    exit 1
fi
if [ -z "$PR_SEARCH_OUTPUT" ]; then
    echo "NOTE: the search returned no PRs; only the cleanup pass will run."
fi
mapfile -t PR_RECORDS <<< "$PR_SEARCH_OUTPUT"

# the =() initializers matter: `set -u` treats a declared-but-empty
# associative array as unbound when its count is taken
declare -A KEEP_PRS=() SEEN_PRS=() DRAFT_PRS=()

for record in "${PR_RECORDS[@]}"; do
    IFS=$'\t' read -r pr is_draft title <<< "$record"
    [ -z "${pr:-}" ] && continue
    SEEN_PRS["$pr"]=1
    filename="PR-$pr.patch"

    # Draft status takes precedence over local pinning. A pinned patch is a
    # local re-anchor, not an exemption from the draft skip policy.
    if [[ "$is_draft" == "true" ]]; then
        DRAFT_PRS["$pr"]=1
        echo "Skipping PR-$pr (draft): $title"
        continue
    fi

    if [[ ${SKIP_PRS[$pr]+_} ]]; then
        echo "Skipping PR-$pr (explicit): ${SKIP_PRS[$pr]}"
        continue
    fi

    if [[ ${PINNED[$pr]+_} ]]; then
        echo "Pinned PR-$pr (${PINNED[$pr]})"
        KEEP_PRS["$filename"]=1
        continue
    fi

    KEEP_PRS["$filename"]=1
    echo "Syncing $filename ($title)..."
    # -z downloads only when upstream is newer than the local file
    curl -sfSL -z "$TARGET_DIR/$filename" -o "$TARGET_DIR/$filename" \
        "https://github.com/$UPSTREAM_REPO/pull/$pr.patch"
done

# Collect the PRs whose real state must be checked: deletion candidates
# (local files that are neither pinned nor in the query result) and pinned
# entries the query did not return.
declare -A CHECK_PRS=()
STALE_FILES=()
shopt -s nullglob
for local_patch in "$TARGET_DIR"/PR-*.patch; do
    base_name=$(basename "$local_patch")
    if [[ ! ${KEEP_PRS[$base_name]+_} ]]; then
        pr="${base_name#PR-}"
        pr="${pr%.patch}"
        if [[ ! "$pr" =~ ^[0-9]+$ ]]; then
            echo "WARNING: $base_name does not name a PR number; keeping it."
            continue
        fi
        STALE_FILES+=("$local_patch")
        CHECK_PRS["$pr"]=1
    fi
done
shopt -u nullglob
for pr in "${!PINNED[@]}"; do
    if [[ ! ${SEEN_PRS[$pr]+_} ]]; then
        CHECK_PRS["$pr"]=1
    fi
done

# One batched GraphQL call resolves every state. If it fails, no state is
# recorded and the passes below keep everything: an outage can never delete a
# live patch.
declare -A PR_STATE=() PR_DRAFT=()
if [ "${#CHECK_PRS[@]}" -gt 0 ]; then
    graphql_query="query { repository(owner: \"${UPSTREAM_REPO%%/*}\", name: \"${UPSTREAM_REPO#*/}\") {"
    for pr in "${!CHECK_PRS[@]}"; do
        graphql_query+=" pr$pr: pullRequest(number: $pr) { state isDraft }"
    done
    graphql_query+=" } }"
    if STATE_JSON=$(gh api graphql -f query="$graphql_query" 2>/dev/null); then
        while IFS=$'\t' read -r key state is_draft; do
            [ -z "$key" ] && continue
            PR_STATE["${key#pr}"]="$state"
            PR_DRAFT["${key#pr}"]="$is_draft"
        done < <(jq -r '.data.repository | to_entries[] | [.key, (.value.state // "UNKNOWN"), (.value.isDraft // "UNKNOWN")] | @tsv' <<< "$STATE_JSON")
    else
        echo "WARNING: could not verify PR states; keeping all local patches this run."
    fi
fi

# Cleanup: retire a stale patch only when its PR has finished or is a draft.
# Archive it first. An open PR outside the query window is kept.
echo "Cleaning up stale PR patches..."
archive_patch() {
    local patch_path="$1" base_name="$2" hash archive_path
    mkdir -p "$RETIRED_PATCH_DIR"
    hash=$(sha256sum -- "$patch_path" | awk '{print substr($1, 1, 12)}')
    archive_path="$RETIRED_PATCH_DIR/${base_name%.patch}-$hash.patch"
    if [ -e "$archive_path" ]; then
        if cmp -s -- "$patch_path" "$archive_path"; then
            rm -- "$patch_path"
            return
        fi
        echo "  ERROR: backup path already exists with different content: $archive_path"
        return 1
    fi
    mv -- "$patch_path" "$archive_path"
    echo "  Archived as $archive_path"
}

for local_patch in "${STALE_FILES[@]+"${STALE_FILES[@]}"}"; do
    base_name=$(basename "$local_patch")
    pr="${base_name#PR-}"
    pr="${pr%.patch}"
    state="${PR_STATE[$pr]:-UNKNOWN}"
    case "$state" in
        MERGED|CLOSED)
            echo "  Retiring $base_name (PR state: $state)"
            archive_patch "$local_patch" "$base_name"
            ;;
        OPEN)
            if [[ ${DRAFT_PRS[$pr]+_} || "${PR_DRAFT[$pr]:-UNKNOWN}" == "true" ]]; then
                echo "  Retiring $base_name (PR is now a draft)"
                archive_patch "$local_patch" "$base_name"
            elif [[ ${SKIP_PRS[$pr]+_} ]]; then
                echo "  Retiring $base_name (explicit skip: ${SKIP_PRS[$pr]})"
                archive_patch "$local_patch" "$base_name"
            else
                echo "  WARNING: PR-$pr is still open but outside the query window; keeping $base_name. Pin it or widen the window."
            fi
            ;;
        *)
            echo "  WARNING: could not determine the state of PR-$pr; keeping $base_name."
            ;;
    esac
done

# Staleness notes: a pinned PR that merged or closed no longer needs its entry.
for pr in "${!PINNED[@]}"; do
    [[ ${SEEN_PRS[$pr]+_} ]] && continue
    state="${PR_STATE[$pr]:-UNKNOWN}"
    case "$state" in
        MERGED|CLOSED)
            if [ -e "$TARGET_DIR/PR-$pr.patch" ]; then
                echo "NOTE: pinned PR-$pr is $state; delete PR-$pr.patch and its PINNED entry by hand."
            else
                echo "NOTE: pinned PR-$pr is $state; its PINNED entry can be removed."
            fi
            ;;
        OPEN)
            if [[ "${PR_DRAFT[$pr]:-UNKNOWN}" == "true" ]]; then
                echo "NOTE: pinned PR-$pr is now a draft; its patch is not carried."
            fi
            # Open PRs outside the query window remain carried unless drafted.
            ;;
        *)
            echo "NOTE: could not determine the state of pinned PR-$pr."
            ;;
    esac
done

echo "PR patches sync complete."

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly TARGET_DIR="$SCRIPT_DIR"
readonly UPSTREAM_REPO="ggml-org/llama.cpp"

# Look back far enough to catch every open Intel Vulkan PR. Unlike the SYCL
# stack this query is narrowed by the title filter below, so the candidate set
# is small and a wide window is cheap. An open PR that ages out of the window
# is safe: the cleanup pass verifies the real PR state before it deletes
# anything.
SINCE_DATE=$(date -d '365 days ago' -u +"%Y-%m-%d")
readonly SINCE_DATE
REFRESH_DATE=$(date -u +"%Y%m%d")
readonly REFRESH_DATE
readonly RETIRED_PATCH_DIR="$SCRIPT_DIR/../patch-backups/upstream-refresh-$REFRESH_DATE/$(basename "$SCRIPT_DIR")"
echo "Searching for Vulkan PRs with 'Intel' in the title created since $SINCE_DATE..."

# Use authenticated `gh` for all GitHub queries. `gh` parses and re-serializes
# the JSON itself, so unescaped control characters in PR titles/bodies (which
# make strict jq abort under `set -euo pipefail`) are handled safely.
if ! command -v gh >/dev/null 2>&1; then
    echo "ERROR: the 'gh' CLI is required (and must be authenticated) to search for PRs."
    exit 1
fi

# Pinned PRs are hands-off for this script: never downloaded, never
# overwritten, never deleted. The patch file on disk is the carry decision:
#   file present -> carried as a locally maintained/re-anchored patch
#   file absent  -> intentionally not carried
# When a pinned PR merges or closes, the cleanup pass prints a NOTE so the
# entry (and any file) can be retired by hand.
declare -A PINNED=(
    # carried: locally collapsed, file exists
    [29357]="collapsed the 3-commit series into one cumulative diff: commit 3 edits flash_attn_prefill.comp, which commit 1 creates, and git apply -R of the series leaves that new file behind"
    # not carried: intentionally out of scope, no patch file
    [30641]="q8/q4 KV for the Xe FA decode kernel; merges cleanly with upstream but rewrites the same Xe FA dispatch block and decode-shader K/V offsets as carried PR-29357, so the two need a semantic merge"
    # Intel-titled drafts, including 24408 and 26829, are auto-skipped below.
)

# Scope is defined by two authoritative filters, per the user's request:
#   1. the `Vulkan` GitHub label, and
#   2. the literal token "Intel" appearing in the PR title.
# This differs deliberately from the SYCL stack, which does NOT use title text
# as a scope test. The title filter is applied client-side via --jq below.
# Capture the command status explicitly: process substitution hides a failing
# producer from `set -e`, which could otherwise turn a network/API outage into
# an empty result and delete every unpinned local patch during cleanup.
if ! PR_SEARCH_OUTPUT=$(gh search prs --repo "$UPSTREAM_REPO" --state open --label Vulkan \
    --created ">=$SINCE_DATE" -L 200 --json number,isDraft,title \
    --jq '.[] | select(.title | test("Intel"; "i")) | [.number, .isDraft, .title] | @tsv'); then
    echo "ERROR: failed to query upstream Vulkan PRs; leaving local patches untouched."
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

    if [[ ${PINNED[$pr]+_} ]]; then
        echo "Pinned PR-$pr (${PINNED[$pr]})"
        KEEP_PRS["$filename"]=1
        continue
    fi

    # Draft status is a first-class automatic skip reason, mirroring the SYCL
    # stack: drafts are not downloaded or applied. A patch carried before its
    # PR became a draft must leave the active stack.
    if [[ "$is_draft" == "true" ]]; then
        DRAFT_PRS["$pr"]=1
        echo "Skipping PR-$pr (draft): $title"
        continue
    fi

    KEEP_PRS["$filename"]=1
    echo "Syncing $filename ($title)..."
    # -z downloads only when upstream is newer than the local file
    curl -sfSL -z "$TARGET_DIR/$filename" -o "$TARGET_DIR/$filename" \
        "https://github.com/$UPSTREAM_REPO/pull/$pr.patch"
done

# Protect every pinned patch file, including pinned PRs older than the query
# window: the query cannot see them, the stack still carries them.
for pr in "${!PINNED[@]}"; do
    KEEP_PRS["PR-$pr.patch"]=1
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
declare -A PR_STATE=()
if [ "${#CHECK_PRS[@]}" -gt 0 ]; then
    graphql_query="query { repository(owner: \"${UPSTREAM_REPO%%/*}\", name: \"${UPSTREAM_REPO#*/}\") {"
    for pr in "${!CHECK_PRS[@]}"; do
        graphql_query+=" pr$pr: pullRequest(number: $pr) { state }"
    done
    graphql_query+=" } }"
    if STATE_JSON=$(gh api graphql -f query="$graphql_query" 2>/dev/null); then
        while IFS=$'\t' read -r key state; do
            [ -z "$key" ] && continue
            PR_STATE["${key#pr}"]="$state"
        done < <(jq -r '.data.repository | to_entries[] | [.key, (.value.state // "UNKNOWN")] | @tsv' <<< "$STATE_JSON")
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
            if [[ ${DRAFT_PRS[$pr]+_} ]]; then
                echo "  Retiring $base_name (PR is now a draft)"
                archive_patch "$local_patch" "$base_name"
            else
                echo "  WARNING: PR-$pr is still open but outside the query window/title filter; keeping $base_name. Pin it or widen the window."
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
            # open but outside the query window; expected for old carried PRs
            ;;
        *)
            echo "NOTE: could not determine the state of pinned PR-$pr."
            ;;
    esac
done

echo "PR patches sync complete."

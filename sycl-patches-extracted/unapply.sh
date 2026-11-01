#!/usr/bin/env bash
# Reverse the full SYCL patch stack, in the exact inverse order of apply.sh:
#   1. Local repairs: sycl-patches-extracted/ZZ-LOCAL-*.patch
#   2. Extracted:     sycl-patches-extracted/0*.patch
#   3. Upstream PRs:  sycl-patches/PR-*.patch
#   4. Fork patches:  sycl-patches/0*.patch
# Each group is reversed in reverse-sorted order.
# Already-reversed patches are skipped safely; real reverse conflicts stop the
# run with diagnostics.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${1:-$(dirname "$SCRIPT_DIR")}"
# Fork/PR patches live next to this script (sibling dir), not under the target.
FORK_DIR="$(dirname "$SCRIPT_DIR")/sycl-patches"

cd "$LLAMA_DIR"

GIT_DIR="$(git rev-parse --absolute-git-dir)"
STATE_DIR="$GIT_DIR/llama-patch-stack-state"
STATE_FILE="$STATE_DIR/sycl-patches-extracted.untracked"

# Collect each group. `shopt -s nullglob` makes empty matches expand to an
# empty array instead of the literal pattern, which keeps `set -u` happy.
# Globs expand in sorted order; reverse each group so unapply order is the
# inverse of apply order.
shopt -s nullglob
FORK_PATCHES=("$FORK_DIR"/0*.patch)
UPSTREAM_PATCHES=("$FORK_DIR"/PR-*.patch)
EXTRACTED_PATCHES=("$SCRIPT_DIR"/0*.patch)
LOCAL_PATCHES=("$SCRIPT_DIR"/ZZ-LOCAL-*.patch)
shopt -u nullglob

reverse_in_place() {
    local -n arr="$1"
    local i j tmp
    for ((i = 0, j = ${#arr[@]} - 1; i < j; i++, j--)); do
        tmp="${arr[i]}"
        arr[i]="${arr[j]}"
        arr[j]="$tmp"
    done
}

reverse_in_place FORK_PATCHES
reverse_in_place UPSTREAM_PATCHES
reverse_in_place EXTRACTED_PATCHES
reverse_in_place LOCAL_PATCHES

reverse_group() {
    local label="$1"
    shift
    local patches=("$@")
    if [ "${#patches[@]}" -eq 0 ]; then
        echo ""
        echo "=== $label: no patches found ==="
        return
    fi
    echo ""
    echo "=== Reversing $label patches (${#patches[@]}) ==="
    local patch_path patch_name reverse_args existing_path unresolved
    for patch_path in "${patches[@]}"; do
        patch_name=$(basename "$patch_path")
        reverse_args=()
        while IFS= read -r existing_path; do
            [ -z "$existing_path" ] || reverse_args+=("--exclude=$existing_path")
        done < <(patch_untracked_files_to_protect "$patch_path")
        echo "Reversing $patch_name..."
        if git apply --reverse --check "${reverse_args[@]}" "$patch_path" >/dev/null 2>&1; then
            git apply --reverse "${reverse_args[@]}" "$patch_path"
            echo "  OK"
        elif git apply --check "${reverse_args[@]}" "$patch_path" >/dev/null 2>&1; then
            echo "  SKIP (not applied)"
        elif git apply --reverse --3way "${reverse_args[@]}" "$patch_path"; then
            # --3way implies --index; restore worktree-only semantics after
            # each fallback so subsequent reverse patches use the same base.
            git reset --quiet -- .
            echo "  OK (three-way)"
        else
            echo "  ERROR (reverse patch conflict)"
            echo "  Unresolved/affected paths:"
            git status --short --untracked-files=all
            unresolved=$(git diff --name-only --diff-filter=U)
            if [ -n "$unresolved" ]; then
                echo "  Unresolved index paths:"
                printf '%s\n' "$unresolved"
            fi
            return 1
        fi
    done
}

patch_untracked_files_to_protect() {
    local patch_path="$1"
    awk '/^diff --git / { path = $4; sub(/^b\//, "", path); next }
         /^new file mode / { print path }' "$patch_path" |
        while IFS= read -r path; do
            if [ -e "$path" ] && ! git ls-files --error-unmatch -- "$path" >/dev/null 2>&1; then
                # A missing state file is the compatibility fallback for a
                # stack applied by an older script: preserve every existing
                # untracked file rather than risking user data.
                if [ ! -e "$STATE_FILE" ] || grep -Fzxq -- "$path" "$STATE_FILE"; then
                    printf '%s\n' "$path"
                fi
            fi
        done
}


reverse_group "Local repairs" "${LOCAL_PATCHES[@]+"${LOCAL_PATCHES[@]}"}"
reverse_group "Extracted" "${EXTRACTED_PATCHES[@]+"${EXTRACTED_PATCHES[@]}"}"
reverse_group "Upstream" "${UPSTREAM_PATCHES[@]+"${UPSTREAM_PATCHES[@]}"}"
reverse_group "Fork" "${FORK_PATCHES[@]+"${FORK_PATCHES[@]}"}"

rm -f "$STATE_FILE"
rmdir "$STATE_DIR" 2>/dev/null || true

echo ""
echo "Patch reversal complete."

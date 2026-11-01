#!/usr/bin/env bash
# Apply the full SYCL patch stack in dependency order:
#   1. Fork patches:  sycl-patches/0*.patch       (the author's experimental SYCL work)
#   2. Upstream PRs:  sycl-patches/PR-*.patch     (pulled from upstream llama.cpp)
#   3. Extracted:     sycl-patches-extracted/0*.patch (isolated fork patches)
#   4. Local repairs: sycl-patches-extracted/ZZ-LOCAL-*.patch
# Already-applied patches are skipped; real conflicts stop with diagnostics.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${1:-$(dirname "$SCRIPT_DIR")}"
# Fork/PR patches live next to this script (sibling dir), not under the apply
# target -- resolve them relative to SCRIPT_DIR so the stack can be applied onto
# any checkout/worktree, not just the repo that holds the patches.
FORK_DIR="$(dirname "$SCRIPT_DIR")/sycl-patches"

cd "$LLAMA_DIR"

GIT_DIR="$(git rev-parse --absolute-git-dir)"
STATE_DIR="$GIT_DIR/llama-patch-stack-state"
STATE_FILE="$STATE_DIR/sycl-patches-extracted.untracked"

# Collect each group. `shopt -s nullglob` makes empty matches expand to an
# empty array instead of the literal pattern, which keeps `set -u` happy.
# Globs expand in sorted order, so file-name order == apply order.
shopt -s nullglob
FORK_PATCHES=("$FORK_DIR"/0*.patch)
UPSTREAM_PATCHES=("$FORK_DIR"/PR-*.patch)
EXTRACTED_PATCHES=("$SCRIPT_DIR"/0*.patch)
LOCAL_PATCHES=("$SCRIPT_DIR"/ZZ-LOCAL-*.patch)
shopt -u nullglob

apply_group() {
    local label="$1"
    shift
    local patches=("$@")
    if [ "${#patches[@]}" -eq 0 ]; then
        echo ""
        echo "=== $label: no patches found ==="
        return
    fi
    echo ""
    echo "=== $label patches (${#patches[@]}) ==="
    local patch_path patch_name apply_args existing_path unresolved
    for patch_path in "${patches[@]}"; do
        patch_name=$(basename "$patch_path")
        apply_args=()
        while IFS= read -r existing_path; do
            [ -z "$existing_path" ] || apply_args+=("--exclude=$existing_path")
        done < <(patch_new_untracked_files "$patch_path")
        echo "Applying $patch_name..."
        if git apply --check "${apply_args[@]}" "$patch_path" >/dev/null 2>&1; then
            git apply "${apply_args[@]}" "$patch_path"
            echo "  OK"
        elif git apply --reverse --check "$patch_path" >/dev/null 2>&1; then
            echo "  SKIP (already applied)"
        elif git apply --3way "${apply_args[@]}" "$patch_path"; then
            # --3way implies --index; keep this script worktree-only so later
            # patches and the inverse unapply see one consistent index.
            git reset --quiet -- .
            echo "  OK (three-way)"
        else
            echo "  ERROR (patch conflict)"
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

patch_new_untracked_files() {
    local patch_path="$1"
    awk '/^diff --git / { path = $4; sub(/^b\//, "", path); next }
         /^new file mode / { print path }' "$patch_path" |
        while IFS= read -r path; do
            if [ -e "$path" ] && ! git ls-files --error-unmatch -- "$path" >/dev/null 2>&1; then
                printf '%s\n' "$path"
            fi
        done
}

snapshot_untracked_files() {
    if [ -e "$STATE_FILE" ]; then
        return
    fi
    mkdir -p "$STATE_DIR"
    git ls-files --others --exclude-standard -z > "$STATE_FILE"
}

snapshot_untracked_files

apply_group "Fork" "${FORK_PATCHES[@]+"${FORK_PATCHES[@]}"}"
apply_group "Upstream" "${UPSTREAM_PATCHES[@]+"${UPSTREAM_PATCHES[@]}"}"
apply_group "Extracted" "${EXTRACTED_PATCHES[@]+"${EXTRACTED_PATCHES[@]}"}"
apply_group "Local repairs" "${LOCAL_PATCHES[@]+"${LOCAL_PATCHES[@]}"}"

echo ""
echo "Patch application complete."

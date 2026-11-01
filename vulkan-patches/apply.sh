#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${1:-$(dirname "$SCRIPT_DIR")}"

cd "$LLAMA_DIR"

GIT_DIR="$(git rev-parse --absolute-git-dir)"
STATE_DIR="$GIT_DIR/llama-patch-stack-state"
STATE_FILE="$STATE_DIR/vulkan-patches.untracked"

# Globs expand in sorted order, so file-name order == apply order; nullglob
# makes an empty match a truly empty array.
shopt -s nullglob
PATCHES=("$SCRIPT_DIR"/*.patch)
shopt -u nullglob

if [ ${#PATCHES[@]} -eq 0 ]; then
    echo "No patches found in $SCRIPT_DIR"
    exit 0
fi

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

for patch_path in "${PATCHES[@]}"; do
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
        exit 1
    fi
done

echo ""
echo "Patch application complete."

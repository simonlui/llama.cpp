#!/usr/bin/env bash
# Reverse the opt-in SYCL candidates (C90..C10) before sycl-patches-extracted/unapply.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${1:-$(dirname "$SCRIPT_DIR")}"

cd "$LLAMA_DIR"

GIT_DIR="$(git rev-parse --absolute-git-dir)"
STATE_DIR="$GIT_DIR/llama-patch-stack-state"
STATE_FILE="$STATE_DIR/sycl-patches-candidates.untracked"

# Globs expand in sorted order; reverse so unapply order is the inverse of
# apply order. nullglob makes an empty match a truly empty array.
shopt -s nullglob
PATCHES=("$SCRIPT_DIR"/*.patch)
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

reverse_in_place PATCHES

if [ ${#PATCHES[@]} -eq 0 ]; then
    echo "No patches found in $SCRIPT_DIR"
    exit 0
fi

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

for patch_path in "${PATCHES[@]}"; do
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
        exit 1
    fi
done

rm -f "$STATE_FILE"
rmdir "$STATE_DIR" 2>/dev/null || true

echo ""
echo "Patch reversal complete."

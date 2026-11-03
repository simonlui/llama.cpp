# Vulkan Intel PR Patches

This directory is the Vulkan analog of `sycl-patches/`. It carries open upstream
pull requests that are both **labeled `Vulkan`** and have **`Intel` in the title**,
so a Vulkan build for Intel Arc (tested on Arc A770 / DG2, Mesa driver) can pick up
in-flight Intel-specific work before it merges.

Unlike `sycl-patches/`, there is no fork "base" patch series here (no `pull_base.sh`,
no `00*.patch`): the stack is composed purely of upstream `PR-*.patch` files.

The stack also carries `ZZ-LOCAL-*.patch` files for small validation-harness
repairs that are not upstream PRs. They are applied after every upstream PR and
are intentionally outside the PR synchronizer's cleanup set.

## Scope filter

The scope of this stack is defined by two authoritative filters:

1. the `Vulkan` GitHub label, and
2. the literal token `Intel` in the PR title.

This is a deliberate difference from `sycl-patches/`, whose `pull_prs.sh` uses the
label alone and explicitly does **not** treat the title as a scope test. Here the
title filter is applied client-side (jq `test("Intel"; "i")`) inside `pull_prs.sh`.

Draft PRs are skipped and are not downloaded or applied. If a previously
carried PR becomes a draft, its patch is archived out of the active directory.

## Carried set (checked 2026-11-03)

`pull_prs.sh` downloads open, non-draft Vulkan PRs whose title contains `Intel`.
This snapshot uses upstream `fac3a894` and a 365-day creation-date window
starting 2025-11-03.

| PR | What | State |
|----|------|-------|
| 29357 | Intel Vulkan flash-attention prefill kernel | open; now 5 commits plus a master merge, collapsed into one cumulative diff and pinned. Refreshed 2026-11-03 for `06e3843a` (shader and host refinement, "accuracy issues fixes") and `c083edcc` (prefill FA shared memory capped at 48 KiB) |

PR 29357's later commits edit `flash_attn_prefill.comp`, which its first commit
creates. `git apply -R` of that multi-commit series reverses the edit but leaves
the new shader behind, so `unapply.sh` no longer returned a clean tree. The
collapsed single diff creates the file once and reverses cleanly. `PINNED`
keeps the sync from replacing it with the multi-commit form.

PR 30641 (q8_0/q4_0 KV for the Xe FA decode kernel) is open and in scope but not
carried. It merges cleanly with upstream on its own, but it rewrites the same
Xe FA dispatch block in `ggml_vk_flash_attn` and the same K/V offset math in
`flash_attn_decode_phase_{1,2}.comp` as PR 29357, so the two need a semantic
merge. It is recorded as a file-less `PINNED` entry. That FA path only turns on
for Xe2 and Xe-LPG UMA devices, not the A770.

PR 29835 merged upstream and its patch was retired; PR 30431 (Arc Pro B65/B70
shader count) also merged in this window. Drafts `24408` and `26829` and the
non-draft `30810` closed without merging.

### Local validation patches

`ZZ-LOCAL-0001-vulkan-check-results.patch` is not an upstream feature patch. It
repairs two current result-checker gaps needed to run `GGML_VULKAN_CHECK_RESULTS`:
it supplies the CPU clone for `GET_ROWS_BACK` (using the output tensor for the
shape-only third argument), and skips output types that the checker does not
decode instead of indexing quantized buffers as scalar elements. It remains a
separate, reversible local patch so upstream PR accounting stays accurate.

`ZZ-LOCAL-0002-vulkan-gate-coopmat2-dependency.patch` separately gates
`VK_NV_cooperative_matrix2` when the required KHR cooperative-matrix path is
disabled by the device capability filter, preventing a validation-layer device
creation error on Arc A770/Mesa.

`ZZ-LOCAL-0003-vulkan-serialize-check-results.patch` serializes Vulkan graph
execution only when `GGML_VULKAN_CHECK_RESULTS` is enabled. The optional checker
uses process-global comparison state, so this keeps concurrent backend tests from
overwriting one another's reference buffer and counter.

The former `ZZ-LOCAL-0004-vulkan-debug-cross-tu.patch` was retired on
2026-11-01. Upstream `ggml-vulkan-common.h` now declares the result-checker and
built-in test entry points itself, and the patch's second declaration of
`ggml_vk_test_dequant_matmul(..., bool mmq = false)` repeated the default
argument, which GCC rejects. Upstream alone builds with
`GGML_VULKAN_CHECK_RESULTS=ON` and `GGML_VULKAN_RUN_TESTS=ON`.

### Auto-skipped drafts (not applied)

Still labeled `Vulkan` + `Intel` but left as drafts and not carried:

| PR | What |
|----|------|
| 17374 | Change default subgroup/block size on selected pipelines for Intel Xe2 and newer |
| 18138 | [WIP] Reduce the number of FA rows for Intel |
| 18126 | [WIP] Enable cooperative matrix support for Intel Arrow Lake H GPUs |

## Quick start

Apply to the current checkout (defaults to the parent llama.cpp dir):

```bash
bash vulkan-patches/pull_prs.sh      # refresh PR-*.patch from upstream
bash vulkan-patches/apply.sh .       # apply the carried set
```

Reverse:

```bash
bash vulkan-patches/unapply.sh .
```

The `apply.sh` / `unapply.sh` scripts use the same guarded flow as the SYCL
stack: direct `git apply --check` first, three-way fallback for context drift,
an already-applied patch is a SKIP (not an error), and a genuine conflict stops
the run with the failing diagnostic. Apply snapshots pre-existing untracked
paths in the worktree's Git administrative directory; unapply protects those
paths but removes new files introduced by the current stack. The state file is
removed after a successful reverse.

## Management script

- `pull_prs.sh`: searches open `Vulkan`-labeled PRs from the last 365 days,
  filters to those with `Intel` in the title, auto-skips drafts, and downloads
  the survivors with `curl -z` (If-Modified-Since). One `PINNED` map (it
  replaced the old `SKIP_REASONS`/`DROP_REASONS` split) holds the hands-off
  PRs: never downloaded, never overwritten, never deleted; whether a
  `PR-<n>.patch` file exists on disk is the carry decision. Before retiring a
  stale patch, the cleanup pass verifies the PR's real state in one batched
  GraphQL call and archives merged, closed, and newly drafted patches with a
  SHA-256 prefix. A still-open PR outside the query window/title filter is kept
  with a warning, and an undeterminable state leaves files untouched. The same
  call flags pinned PRs that merged or closed so their entries can be pruned.

The pre-refresh snapshot and retired patches are under
`../patch-backups/upstream-refresh-20261010/` and
`../patch-backups/upstream-refresh-20261003/`.

### Refresh apply/unapply and build check (2026-11-01)

The Vulkan stack (PR 29357 plus three local patches) applied with plain
`git apply` and reversed in a clean worktree at upstream `dde879bc1`, alone and
on top of the SYCL stack. Unapply returned the tracked and untracked tree
exactly to upstream.

The applied tree (SYCL and Vulkan stacks together) was built with GCC 13,
Ubuntu's Vulkan 1.3.275 headers and glslc 2023.8 (`GL_KHR_cooperative_matrix`
supported, so the Xe FA shaders compiled; coopmat2 and bf16 not supported):

- `GGML_VULKAN_CHECK_RESULTS=ON`, `GGML_VULKAN_RUN_TESTS=ON`, tests on: all
  524 targets built. `ctest -L main -LE model` passed 16/17; the failure is
  `test-thread-safety`, which needs `test-download-model` and had no network.
- Default options: `test-backend-ops -b Vulkan0` on Mesa llvmpipe (no Intel
  GPU was available) passed ROPE 484/484, ROPE_SET_ROWS 73/73, SET_ROWS
  319/319, SUM 10/10 and GROUP_NORM 2/2. FLASH_ATTN_EXT was stopped after
  65 cases, all OK; llvmpipe runs about four FA cases a minute.

None of this exercises the Intel-only Xe FA path; that still needs the A770.

## Build and test

The A770 uses the Mesa (RADV/ANV) open-source Vulkan stack, so no proprietary
toolchain is needed - just the Vulkan headers, `glslc`, `mold`, and Ninja.

```bash
cmake -B build_vulkan -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON \
    -DGGML_LTO=ON \
    -DGGML_NATIVE=ON \
    -DCMAKE_LINKER_TYPE=MOLD \
    -DCMAKE_C_FLAGS="-ffunction-sections -fdata-sections" \
    -DCMAKE_CXX_FLAGS="-ffunction-sections -fdata-sections" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--icf=safe,--gc-sections,--as-needed,-z,noexecstack,-z,relro,-z,now,--strip-all"

cmake --build build_vulkan -j$(nproc)
```

## Verification snapshot (2026-10-03)

The active stack was applied and reversed in a clean worktree at
`836d57176dc699a726c55418e4f96b8ca628e1bf`. The Intel Arc A770 (DG2) was
detected with Vulkan 1.4.363 and Mesa 26.3.0-devel.

With `GGML_VULKAN_VALIDATE=ON`, `GGML_VULKAN_RUN_TESTS=OFF`,
`GGML_VULKAN_CHECK_RESULTS=OFF`, `GGML_BUILD_TESTS=OFF`, and
`LLAMA_BUILD_TESTS=ON`, the Vulkan build succeeded and the selected main CTest
set passed 50/50 with `ctest -L main -LE model -E '^test-download-model$'`
(run from the build directory). This includes
`test-recurrent-state-rollback` and `test-save-load-state`.

With result checking enabled, the focused `FLASH_ATTN_EXT` backend test passed
5,301/5,301 cases (`test-backend-ops -b Vulkan0 -o FLASH_ATTN_EXT`) and
`test-backend-sampler` passed. With result checking enabled,
`test-recurrent-state-rollback` and `test-save-load-state` still segfault in
`ggml_vk_compute_forward`; they pass with result checking disabled. This is an
unresolved diagnostic-mode issue, not a passing result for the full checker
suite. No workaround for it is carried in the active patch stack.

## Requirements

- Vulkan loader + headers, `glslc` (shaderc), `mold` linker, Ninja, CMake
- Intel Arc dGPU (tested on Arc A770 / DG2 with the Mesa driver)

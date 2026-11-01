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

## Carried set (checked 2026-10-10)

`pull_prs.sh` downloads open, non-draft Vulkan PRs whose title contains `Intel`.
This snapshot uses upstream `23b0202a189c44a54625aadcb37a946dd1d6278` and a
365-day creation-date window starting 2025-10-10.

| PR | What | State |
|----|------|-------|
| 29357 | Intel Vulkan flash-attention prefill kernel | open; downloaded |
| 29835 | handle free-memory reports greater than total memory on Intel Windows | open; downloaded |

The live query found no other ready, open PRs matching both scope filters.
Drafts remain skipped: `26829`, `24408`, `18138`, `18126`, and `17374`.
PR 24406 has merged and is now upstream, so its local patch was retired. Other
merged Intel Vulkan PRs in the current window are `29476`, `27471`, `27049`,
`25380`, `25192`, `24406`, `24404`, `23964`, `22461`, `20818`, `20672`, and
`20059`. They are not carried as local patches.

The refreshed SYCL stack no longer carries PR 26631, so its former
`test_mul_mat_vec_fusion` abort does not apply to the current combined stack.

### Local validation patch

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

`ZZ-LOCAL-0004-vulkan-debug-cross-tu.patch` exports the result-checker and
optional built-in Vulkan test entry points moved into `ggml-vulkan-debug.cpp`,
and declares them in the shared header. This keeps the current upstream split
translation units linkable when `GGML_VULKAN_CHECK_RESULTS` or
`GGML_VULKAN_RUN_TESTS` is enabled.

### Auto-skipped drafts (not applied)

Still labeled `Vulkan` + `Intel` but left as drafts and not carried:

| PR | What |
|----|------|
| 17374 | Change default subgroup/block size on selected pipelines for Intel Xe2 and newer |
| 18138 | [WIP] Reduce the number of FA rows for Intel |
| 18126 | [WIP] Enable cooperative matrix support for Intel Arrow Lake H GPUs |
| 24408 | Intel Xe flash-attention and GEMM mega-PR |
| 26829 | Intel Vulkan kernel optimizations for LLM inference (Arc B70 Pro) |

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

### Refresh apply/unapply check (2026-10-10)

The current Vulkan stack applied and reversed in a clean worktree at upstream
`23b0202a189c44a54625aadcb37a946dd1d6278`. Unapply returned the tracked and
untracked tree exactly to upstream, and the applied diff passed
`git diff --check`. The upstream `PR-29835.patch` emits a harmless warning for one
blank-at-EOF line while applying and reversing. No compile, build, or backend
tests were run in this refresh; the user will build the applied stack.

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

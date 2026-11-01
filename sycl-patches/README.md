# SYCL Kernel Patches for Intel Arc Pro (Battlemage)

This directory contains a series of patches for the SYCL backend of `llama.cpp`, optimized for Intel Arc Pro B70 (Xe2-HPG, BMG-G31).

## Patch series (apply in order)

| # | Patch | Type | Impact (vs AOT+F16 baseline) |
|---|-------|------|------------------------------|
| ~~0001~~ | ~~BMG AOT + reorder dequant + async~~ | ~~Bugfix~~ | Merged upstream via PR #22147/#22152/#22153/#22156 — retired |
| 0002 | Remove dpct capability checks | Cleanup | Code quality |
| ~~0003~~ | ~~PAD non-contiguous stride~~ | ~~Bugfix~~ | Merged upstream as PR #22148 — retired |
| 0004 | Split GROUP_NORM gate | Bugfix | Gives GROUP_NORM its own full-contiguous gate (its kernel has no stride params). NORM/L2_NORM/RMS_NORM are not touched here - upstream #25044 made those kernels stride-generic, verified 81/81 on A770. The former SUM relaxation to `contiguous_rows` was dropped on 2026-11-01: `ggml_sycl_op_sum` sums `ne` consecutive floats, so the row-contiguous view with gaps that upstream #28851 now tests (`test_sum`, `slice=true`) would compute wrong data. The file name is unchanged so the `pull_base.sh` pin still matches |
| ~~0005~~ | ~~RMS_NORM+MUL fusion~~ | ~~Optimization~~ | Superseded by upstream RMS_NORM+MUL implementation; retired |
| ~~0006~~ | ~~6 new ops + oneMKL TRSM~~ | ~~Feature~~ | Merged upstream as PR #22149 — retired |
| 0007 | Barrier local_space fence | Bugfix | Correctness |
| 0008 | Native SYCL shuffles | Cleanup | Code quality |
| ~~0009~~ | ~~Scratchpad pool + PVC removal~~ | ~~Optimization~~ | Retired |
| ~~0010~~ | ~~Small matmul oneMKL direct~~ | ~~Optimization~~ | Merged upstream as PR #22150 — retired |
| ~~0011~~ | ~~Fuse UNARY+MUL~~ | ~~Optimization~~ | Superseded by the richer open PR #26411; retired |
| ~~0012~~ | ~~RMS_NORM+MUL+ADD 3-op fusion~~ | ~~Optimization~~ | Merged upstream as PR #27610; retired |
| 0014 | Drop Q2_K/Q4_K/Q5_K from GET_ROWS supports on A770 | Correctness | Re-gates types that upstream PR-23710 (merged) enabled but which miss test-backend-ops epsilon on A770 |
| 0015 | Gate CONCAT supports_op to implemented types | Correctness | Gates CONCAT to what `concat.cpp` implements (F32/F16/BF16/I32/I16/I64/I8). Stops the SIGABRT from the quantized-type CONCAT tests added by upstream #25247; unsupported types fall back to CPU. Supersedes the earlier F32/I32-only 0015 (predated #24363/#24011 scalar-type support) |
| ~~0016~~ | ~~quant: use `n_layer_all` for MoE ffn counts~~ | ~~Bugfix (non-SYCL)~~ | Merged upstream as `b3ce5cedf` (`quant : fix quantizing moe with mtp (#24986)`) — retired |
| 0017 | Gate the ternary types out of `supports_op` | Correctness | The SYCL backend implements no kernel for `TQ1_0`/`TQ2_0` anywhere, but `supports_op` only checks operand types per op, so the `TQ2_0` entry in test-backend-ops' `all_types` aborts the run three times over - `set_rows.cpp:547 Unsupported tensor type!`, then `cpy.cpp:1409 fatal error`, then `mmvq.cpp:2681 unsupport data type=tq2_0`. One central type gate (metadata-only ops excepted) replaces what would otherwise be three per-op gates; the types fall back to CPU |

### Fetched upstream PRs (`PR-*.patch`) - checked 2026-11-01

`pull_prs.sh` uses the `SYCL` label as its scope and does not require `SYCL` in
the title. It carries open, non-draft PRs created in the last 60 days. This
snapshot uses upstream `dde879bc1` (`sycl: Try PkgConfig for LevelZero includes
/ libs (#31076)`) and a creation-date window starting 2026-09-02.

| PR | What | State |
|----|------|-------|
| 29506 | build the SYCL backend with a separate compiler | open; re-anchored on the upstream F16 footnote anchor in `docs/backend/SYCL.md` (pinned) |
| 29672 | add PTQ1_0 ternary type | open; re-anchored next to upstream `GGML_TYPE_F8_E4M3` (pinned) |
| 30156 | device-side MoE scheduler | open; was explicitly skipped for its overlap with PR 28725, which has merged; applies cleanly on the current stack |
| 30206 | support `truncate=false` for YaRN | open; re-anchored: `test_rope_set_rows` keeps upstream `n_dims` and adds the PR's `yarn`/`truncate` (pinned) |
| 30606 | route BMG D=256 single-query decode to the FA vector kernel | open |
| 30857 | f16 support for SWIGLU_OAI | open |
| 30956 | fast `ggml_sycl_comm_exchange()` for `-sm tensor` | open; carries two trailing-whitespace lines from the PR |
| 31109 | bound CONV_TRANSPOSE_1D input iteration | open |
| 31126 | cache the current device id | open |
| 31138 | thread-safe oneDNN FA cache, memoized runtime failures | open |
| 31208 | drop the fattn-mkl host-wait check from graph checks | open |
| 31285 | refuse device buffers that would exceed VRAM | open |
| 31325 | dmmv launcher cleanup | open; re-anchored after fork patches 0002/0008 (pinned) |
| 31355 | fuse recurrent convolution windows | open |
| 31460 | accept only implemented types in MUL_MAT, SET_ROWS and CPY | open; re-anchored after PR 29672: its whitelist replaces the per-op TQ1_0/TQ2_0/PTQ1_0 blocklists (pinned) |
| 31465 | exact rounding in CPY to Q4_1, Q5_1 and Q8_0 | open |

Retired in this refresh because upstream merged them: `28725`, `29107`,
`29337`, `29338`, `29696`, `29864`, `29889`, `30271`, and `30286`. PR 30226
closed without merging and was retired. Their pins and the `SKIP_PRS` entries
for merged PR 26631 and now-carried PR 30156 were removed. The retired files
are in this branch's Git history; no `../patch-backups/` copy was made because
the refresh ran outside the usual checkout.

New open, non-draft PRs in the window that are explicitly skipped in
`SKIP_PRS` because they are not SYCL-primary: `29510` (llama + CUDA
`flash_attn_ext_rows`), `29676` (CUDA PTQ1_0), `30559` (CUDA IQ4_NL
GET_ROWS), `30715` (CPU JQ*_K types), `30743` (cross-backend argsort/top-k),
`30967` (docs only), `31066` (nix only), `31139` (cross-backend max
allocation size), and `31236` (CPU reference MXFP4/NVFP4 rounding).

The query skipped drafts `31428`, `31379`, `31313`, `30802`, `30718`, `30717`,
`30716`, `29896`, `29801`, `29367`, `29247`, and `28898`. PR 29367's draft
status still keeps the archived extracted 0013 and local repairs 0002/0004
inactive.

`PINNED` is reserved for carried, locally re-anchored patches. `SKIP_PRS`
records explicit open-PR exclusions with reasons. Drafts are skipped
automatically; merged and closed PRs are retired from GitHub state rather than
maintained in a separate drop list. If GitHub cannot be queried, the script
leaves local patches untouched.

Earlier refresh backups remain under
`../patch-backups/upstream-refresh-20261010/`,
`../patch-backups/upstream-refresh-20261003/`, and
`../patch-backups/upstream-refresh-20260920/`.

### Extracted (experimental, `sycl-patches-extracted/`)

`0015`/`0016` extracted changes live here and are applied by
`sycl-patches-extracted/apply.sh` after the upstream PRs. The former `0012`
fusion is supplied by merged upstream PR 27610 and is not carried locally.
The former 0013 MMVQ multirow-Y SLM optimization depends on draft PR 29367's
integer-tag dispatch, so it is archived and its number is not reused.

No local repairs are active. `ZZ-LOCAL-0001` (CONV_2D strides) was retired on
2026-11-01: upstream #30363 runs WHCN and CWHN input/kernel layouts on SYCL and
gates every other layout to CPU, and CONV_2D's destination is always a new
contiguous tensor, so the output-stride half of the repair had no effect.
The former local repairs 0002 and 0004 depended on draft PR 29367, and local
repair 0003 was superseded by PR 29809; all three stay archived.

## Quick start

To apply the complete stack (fork, upstream PR, extracted, and local patches)
to the current checkout:

```bash
bash sycl-patches-extracted/apply.sh .
```

To reverse patches:

```bash
bash sycl-patches-extracted/unapply.sh .
```

To update against the latest `master`: run the extracted `unapply.sh`,
fast-forward `master`, refresh the patches with `pull_base.sh` / `pull_prs.sh`,
then run the extracted `apply.sh` again. (This is done by hand; there is no
`rebase.sh` wrapper.)

## Management Scripts

The sync scripts separate re-anchored carries, explicit skips, drafts, and
merged/closed PR retirement. The reason string records each pinned carry or
explicit skip.

- `pull_base.sh`: Syncs base patches (`00*.patch`) from the `aicss-genai` fork.
  Pinned-with-file entries are the locally rebased 0002 / 0004 / 0007 / 0008 and
  the **local-only** 0014 / 0015 / 0017 (never published by the fork; pinning is
  what stops cleanup from deleting them). Pinned-without-file entries are the
  retired/merged fork patches the fork still publishes (0001 / 0003 / 0005 /
  0006 / 0010 / 0011 / 0012); pinning stops them from being re-downloaded. When
  a pinned name is neither on the fork nor local, the script notes that the
  entry can be pruned. The script refuses to run cleanup if the fork listing
  comes back empty.
- `pull_prs.sh`: Pulls open, `SYCL`-labeled PRs from upstream that were created
  in the last 60 days. Drafts are skipped. `SKIP_PRS` prevents known open but
  excluded PRs from being downloaded; stale files for those PRs are archived.
  Before retiring other stale patches, the script checks state and draft
  status in one batched GraphQL call. Merged, closed, and newly drafted patches
  are archived with a SHA-256 prefix. An unknown state leaves files untouched.
  Title text is not used as a scope test.

### Locally-rebased patches

Upstream PRs whose patches have been re-anchored locally are pinned, so
`pull_prs.sh` never downloads them again - a later force-push by the PR author
cannot clobber the local re-anchor. Unpinned open PRs are fetched with
`curl -z` (If-Modified-Since), so an unchanged upstream patch is not
re-downloaded.

## Build and test

```bash
source /opt/intel/oneapi/setvars_fixed.sh
cmake -B build_sycl -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_SYCL=ON \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx \
    -DGGML_SYCL_F16=ON \
    -DGGML_LTO=ON \
    -DGGML_NATIVE=ON \
    -DCMAKE_C_FLAGS="-ffunction-sections -fdata-sections" \
    -DCMAKE_CXX_FLAGS="-ffunction-sections -fdata-sections" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--gc-sections,--as-needed,-z,noexecstack,-z,relro,-z,now,--strip-all"

cmake --build build_sycl -j$(nproc)
```

## Requirements

- oneAPI 2025.2+ (icpx compiler, oneDNN, oneMKL)
- Intel Arc Pro dGPU with Xe2 architecture (tested on Arc Pro B70)
- Level Zero runtime

# SYCL Kernel Patches for Intel Arc Pro (Battlemage)

This directory contains a series of patches for the SYCL backend of `llama.cpp`, optimized for Intel Arc Pro B70 (Xe2-HPG, BMG-G31).

## Patch series (apply in order)

| # | Patch | Type | Impact (vs AOT+F16 baseline) |
|---|-------|------|------------------------------|
| ~~0001~~ | ~~BMG AOT + reorder dequant + async~~ | ~~Bugfix~~ | Merged upstream via PR #22147/#22152/#22153/#22156 — retired |
| 0002 | Remove dpct capability checks | Cleanup | Code quality |
| ~~0003~~ | ~~PAD non-contiguous stride~~ | ~~Bugfix~~ | Merged upstream as PR #22148 — retired |
| 0004 | Relax SUM + split GROUP_NORM gate | Bugfix | Relaxes SUM to `contiguous_rows`; splits GROUP_NORM to its own full-contiguous gate. NORM/L2_NORM/RMS_NORM no longer touched here - upstream #25044 made those kernels stride-generic, verified 81/81 on A770 |
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

### Fetched upstream PRs (`PR-*.patch`) - checked 2026-10-10

`pull_prs.sh` uses the `SYCL` label as its scope and does not require `SYCL` in
the title. It carries open, non-draft PRs created in the last 60 days. This
snapshot uses upstream `23b0202a189c44a54625aadcb37a946dd1d6278` and a
creation-date window starting 2026-08-11.

| PR | What | State |
|----|------|-------|
| 28725 | SYCL graph record and replay | open; collapsed from 11 commits into one cumulative diff |
| 29107 | IQ3 code reorder | open; includes the standard IQ3_S multi-column path |
| 29337 | reordered Q8_0 MoE weights | open |
| 29338 | guard DMMV reads at row tails | open; re-anchored after PR 29337 |
| 29506 | build the SYCL backend with a separate compiler | open |
| 29672 | add PTQ1_0 ternary type | open; re-anchored with both quantization regression tests |
| 29696 | wide q4_K/q5_K dequant stores | open; re-anchored after PR 29687 |
| 29864 | multi-column XMX MMVQ and fused gate/up | open; regenerated against the current stack and applied after the other PRs |
| 29889 | fix `mul_mat`, split-buffer, and host-pool memory errors | open; re-anchored after PR 29809 |
| 30206 | support `truncate=false` for YaRN | open |
| 30226 | Q4_K MMVQ wide row pairing | open |
| 30271 | fix SYCL compiler warnings | open |
| 30286 | fix flash-attention crash for query-head dimension mismatch | open |

PR 30156 (device-side MoE scheduler) is open and non-draft but explicitly
skipped. Its branch merges cleanly with current upstream by itself, but its
SYCL dispatch changes conflict with the carried PR 28725 graph-replay changes
when combined. The patch is archived until those changes can be reconciled as
one stack. PR 26631 remains explicitly skipped because its patch conflicts
with current upstream and GitHub reports `mergeable_state=dirty`; its previous
test-matrix defect is documented in `SYCL_DEV_LOG.md`.

The current query skipped drafts `30239`, `29801`, `29676`, `29510`, `29367`,
`29247`, `27983`, `27325`, `27324`, and `27322`. PR 29367's draft status also
means its dependent extracted patch 0013 and local repairs 0002/0004 are not
active. PR 29809 and PR 29500 are merged; local repair 0003 was superseded by
PR 29809 and is archived. PR 29796 is an older open draft and was archived
during stale-patch cleanup.

Previously carried PR patches merged or became drafts and were archived:
`26610`, `27689`, `29071`, `29171`, `29245`, `29367`, `29375`, `29507`,
`29687`, `29796`, and `29809`. Other merged PRs are already part of the
upstream base. Retired patch files are archived with a SHA-256 prefix under
`../patch-backups/upstream-refresh-YYYYMMDD/sycl-patches/`.

`PINNED` is reserved for carried, locally re-anchored patches. `SKIP_PRS`
records explicit open-PR exclusions with reasons. Drafts are skipped
automatically; merged and closed PRs are retired from GitHub state rather than
maintained in a separate drop list. If GitHub cannot be queried, the script
leaves local patches untouched.

Earlier refresh backups remain under
`../patch-backups/upstream-refresh-20261003/` and
`../patch-backups/upstream-refresh-20260920/`.

### Extracted (experimental, `sycl-patches-extracted/`)

`0015`/`0016` extracted changes live here and are applied by
`sycl-patches-extracted/apply.sh` after the upstream PRs. The former `0012`
fusion is supplied by merged upstream PR 27610 and is not carried locally.
The former 0013 MMVQ multirow-Y SLM optimization depends on draft PR 29367's
integer-tag dispatch, so it is archived and its number is not reused.

`ZZ-LOCAL-0001-sycl-conv2d-strided-layout.patch` fixes direct CONV_2D for
non-contiguous tensors. Upstream PR 26786 proposed a narrower input-only fix
and closed without merging. The local patch covers input, kernel, and output
strides, so it remains active.

The former local repairs 0002 and 0004 depended on draft PR 29367 and are
archived. Local repair 0003 was superseded when PR 29809 merged and is archived.
The independently applicable `ZZ-LOCAL-0001` CONV_2D stride repair remains
active and reversible.

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

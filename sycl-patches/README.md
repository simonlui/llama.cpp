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
| 0017 | Gate the ternary types out of `supports_op` | Correctness | The SYCL backend implements no kernel for `TQ1_0`/`TQ2_0` anywhere, but `supports_op` only checks operand types per op, so the `TQ2_0` entry in test-backend-ops' `all_types` aborts the run three times over - `set_rows.cpp:547 Unsupported tensor type!`, then `cpy.cpp:1409 fatal error`, then `mmvq.cpp:2681 unsupport data type=tq2_0`. One central type gate (metadata-only ops excepted) replaces what would otherwise be three per-op gates; the types fall back to CPU. Local patch `ZZ-LOCAL-0005` (in `sycl-patches-extracted/`) adds TQ1_0 / TQ2_0 / PTQ1_0 kernels and narrows this gate to everything but the weight operand of MUL_MAT, MUL_MAT_ID and GET_ROWS |

### Fetched upstream PRs (`PR-*.patch`) - checked 2026-11-03

`pull_prs.sh` uses the `SYCL` label as its scope and does not require `SYCL` in
the title. It carries open, non-draft PRs created in the last 60 days. This
snapshot uses upstream `fac3a894` (`vulkan: skip maxStorageBufferRange
pre-check for non-compute tensors (#31114)`) and a creation-date window starting
2026-09-04. GitHub's `pull/N.patch` download returned 403 through this
environment's proxy, so the unpinned files were written with
`git format-patch` from the fetched PR heads (`pull/N/head`), which gives the
same mbox series.

| PR | What | State |
|----|------|-------|
| 29506 | build the SYCL backend with a separate compiler | open; re-anchored on the upstream F16 footnote anchor in `docs/backend/SYCL.md` (pinned) |
| 29672 | add PTQ1_0 ternary type | open; re-anchored next to upstream `GGML_TYPE_F8_E4M3` (pinned) |
| 30156 | device-side MoE scheduler | open |
| 30206 | support `truncate=false` for YaRN | open; re-anchored: `test_rope_set_rows` keeps upstream `n_dims` and adds the PR's `yarn`/`truncate` (pinned) |
| 30606 | route BMG D=256 single-query decode to the FA vector kernel | open; refreshed to the 2026-11-03 head (3 commits) |
| 30857 | f16 support for SWIGLU_OAI | open |
| 31109 | bound CONV_TRANSPOSE_1D input iteration | open |
| 31126 | cache the current device id | open |
| 31138 | thread-safe oneDNN FA cache, memoized runtime failures | open |
| 31208 | drop the fattn-mkl host-wait check from graph checks | open |
| 31285 | refuse device buffers that would exceed VRAM | open; reworked upstream: the check now runs on BMG-G21/G31 only (no effect on the A770 beyond `GGML_SYCL_MEMTRACE` output), with a free-memory baseline and `GGML_SYCL_ALLOW_VRAM_OVERCOMMIT`; re-anchored after PR 30606, whose `GGML_SYCL_FA_TILE_DEBUG` row sits in its docs context (pinned) |
| 31355 | fuse recurrent convolution windows | open; no longer matches delta-net models after upstream `b74298c1`, see below |
| 31460 | accept only implemented types in MUL_MAT, SET_ROWS and CPY | open; re-anchored after PR 29672: its whitelist replaces the per-op TQ1_0/TQ2_0/PTQ1_0 blocklists (pinned) |
| 31465 | exact rounding in CPY to Q4_1, Q5_1 and Q8_0 | open |
| 31536 | oneMKL flash attention with a mask broadcast over heads or batches; `cache-ram` argument for the example server scripts | open, new; re-anchored: its test struct goes before upstream's new `test_flash_attn_ext_kv_rows` (#29510) and its cases after PR 30606's GQA decode loop (pinned) |
| 31542 | attention sinks (gpt-oss, Flint) may take the oneMKL FA path instead of TILE; the author measures 1.87x / 1.32x per attention op | open, new; re-anchored after PR 31536's test cases (pinned) |

Retired in this refresh because upstream merged them on 2026-11-02: `30956`
(fast `ggml_sycl_comm_exchange()`) and `31325` (dmmv launcher cleanup). Fork
patches 0002 and 0008 were re-anchored onto the merged dmmv launcher templates.
Their files are in this branch's Git history.
The `SKIP_PRS` entry for `29510` was removed: it merged.

Upstream `b74298c1` (`delta_net_base: refactor conv state`, merged
2026-11-03) writes the rollback conv-state windows of delta-net models
(Qwen3.5, Qwen3-Next) with one `SET_ROWS` instead of one `CPY` per slot.
Carried PR 31355 still matches the old `CPY` chain, which its own
test-backend-ops case builds, so its tests pass but it no longer fires on those
models. SYCL runs the new `SET_ROWS` itself (strided source, F32). Carry it
until the PR follows upstream.

New open, non-draft PRs in the window that are explicitly skipped in
`SKIP_PRS` because they are not SYCL-primary: `29676` (CUDA PTQ1_0), `30559`
(CUDA IQ4_NL GET_ROWS), `30715` (CPU JQ*_K types), `30743` (cross-backend
argsort/top-k), `30967` (docs only), `31066` (nix only), `31139` (cross-backend
max allocation size; the SYCL buffer types report no limit, so nothing changes
on SYCL), `31236` (CPU reference MXFP4/NVFP4 rounding), and `31540` (CI, Dockerfile
and docs: oneAPI 2026.2, oneDNN 2026.1, Level Zero 1.34.0; its changelog
reports Qwen3.8-9B Q6_K pp512 565 vs 378 t/s on an Arc B570 from oneAPI
2026.2 alone, against the 2026.1 build).

The query skips drafts automatically. Four SYCL drafts are carried by hand as
opt-in candidates in `../sycl-patches-candidates/`: `31428` (C20), `31574`
(C62), `31379` (C70) and `31313` (C75). Other drafts in the window: `30802`,
`30716`, `30717`, `30718` (not SYCL-primary). PR 31528 (reorder-layout wide
mat-vec kernels for q4_K/q5_K/q6_K) closed unmerged as a draft on 2026-11-02.
PR 29367's draft status still keeps the archived extracted 0013 and local
repairs 0002/0004 inactive.

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

To update against the latest `master`: run the candidates' and then the
extracted `unapply.sh`, fast-forward `master`, run `check_arch_gates.sh`,
refresh the patches with `pull_base.sh` / `pull_prs.sh`, then run the extracted
`apply.sh` again. (This is done by hand; there is no `rebase.sh` wrapper.)

Opt-in patches that may help the A770 but are not proven on it live in
`../sycl-patches-candidates/` and apply on top of the full stack; see its
README for the test plan.

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
- `check_arch_gates.sh`: Lists every Intel architecture gate in `ggml-sycl` at a
  ref (default `upstream/master`) and diffs it against `arch_gates.txt`, the
  snapshot from the last refresh. Gates in files that use DPAS, joint_matrix,
  ESIMD or 2D block loads are tagged `HW`: never add DG2 to those without a
  sub-group-8 port. Untagged gates only pick between kernels built for every
  device; try them on the A770 through `GGML_SYCL_DG2_BMG_TUNING` (see
  `../sycl-patches-candidates/`). Run it after fast-forwarding `master`, then
  `--update` once the new gates are classified.
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
bash sycl-patches-extracted/apply.sh .
bash sycl-patches-candidates/apply.sh .   # optional
OAPI_BIN="$(dirname "$(command -v icpx)")/compiler"
cmake -B build_sycl -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_C_COMPILER_AR="$OAPI_BIN/llvm-ar" -DCMAKE_CXX_COMPILER_AR="$OAPI_BIN/llvm-ar" \
    -DCMAKE_C_COMPILER_RANLIB="$OAPI_BIN/llvm-ranlib" -DCMAKE_CXX_COMPILER_RANLIB="$OAPI_BIN/llvm-ranlib" \
    -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=ON -DGGML_SYCL_GRAPH=ON -DGGML_SYCL_SUPPORT_XMX=ON \
    -DGGML_NATIVE=ON -DGGML_LTO=ON \
    -DCMAKE_C_FLAGS="-ffunction-sections -fdata-sections" \
    -DCMAKE_CXX_FLAGS="-ffunction-sections -fdata-sections" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--gc-sections,--as-needed,-z,noexecstack,-z,relro,-z,now,--strip-all"
cmake --build build_sycl -j$(nproc)
```

The `llvm-ar` / `llvm-ranlib` overrides keep `GGML_LTO` on. CMake 3.30 and older do not look in
oneAPI's `bin/compiler` for them and take the system's, which may not read icx's bitcode (llvm-ar 18 on
icx 2026.2: `Unknown attribute kind (106)`). `check_ipo_supported` then fails and the build goes on
without LTO after one "IPO is not supported" warning. CMake 3.31 and newer find oneAPI's without the
overrides. This line is a JIT build: `GGML_SYCL_DEVICE_ARCH=acm-g10` (AOT) drops
the ESIMD kernels' 256-register request; `ab/` measures which build is faster on the card.

Which patches make the A770 faster is decided on the card, with the A/B plan in `ab/README.md`.
Register files, spills and DPAS counts per kernel need no card: `aot/README.md`.

## Requirements

- oneAPI 2025.2+ (icpx compiler, oneDNN, oneMKL)
- Intel Arc Pro dGPU with Xe2 architecture (tested on Arc Pro B70)
- Level Zero runtime

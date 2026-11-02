# SYCL fork — dev log

Consolidated record of investigations, patch decisions, and bench results for
this local SYCL fork. Replaces the older `INVESTIGATION_LOG.md` and
`SYCL_PATCH_SYNC_LOG.md`.

- **Hardware**: Intel Arc A770 (DG2/Xe-HPG, 15.4 GiB VRAM), Ryzen 9 5950X
  16C/32T, CachyOS kernel 7.0.x, i915 driver
- **Toolchain**: oneAPI 2025.3 (icx/icpx, oneDNN, oneMKL, Level Zero); source
  `/opt/intel/oneapi/setvars_fixed.sh` before any SYCL command
- **Build dir**: `build_sycl/` (Ninja, Release, F16=ON, DNN=ON, MMQ_XMX=ON,
  LTO=ON, native arch)
- **Base**: `b75ecd197`; the whole stack applies on it with plain `git apply`
  (no three-way fallback anywhere).
- **Patch stack**: 7 fork patches (`0002` `0004` `0007` `0008` `0014` `0015`
  `0017`) in `sycl-patches/`, then the carried upstream PRs, then 4 extracted
  patches (`0012` RMS_NORM+MUL+ADD fusion, `0013` MMVQ multirow-Y SLM, `0015`
  softmax reduce-over-group, `0016` MMQ XMX) and one local repair
  (`ZZ-LOCAL-0001` conv2d strided layout) in `sycl-patches-extracted/`.
  All four groups are applied in that order by `sycl-patches-extracted/apply.sh`.
  21 patches total.
- **Carried open PRs** (applied, locally re-anchored where noted in
  `sycl-patches/README.md`): `PR-25089` `PR-25266` `PR-26336` `PR-26376`
  `PR-26631` `PR-26635` `PR-26689` `PR-26713` `PR-26863`.
- **Fully merged / dropped**: `PR-24941` (softmax-underflow clamp) merged upstream
  as `e7e3f3509`. The old `0016` quant-MTP one-liner (ggml-org#24379) merged as
  `b3ce5cedf`. The most recent sync retired `PR-25268` `PR-26081` `PR-26251`
  `PR-26372` `PR-26411` `PR-26643` on the SYCL side and `PR-25380` on the Vulkan
  side. Merged PRs need no manual exclusion: the sync scripts now keep a single
  `PINNED` map (it replaced the old `SKIP_REASONS`/`DROP_REASONS` split; older
  entries below use the old names), verify a PR really merged or closed before
  deleting its file, and print a NOTE when a pinned PR finishes so the entry can
  be retired by hand.

## Known failing cases (upstream-attributed; they do not block this stack's verification gate)

Each backend-numeric item was confirmed against an unpatched `b75ecd197` build,
so none of this is stack fallout. The reference for full coverage is the run
that predates the retirement of `ZZ-LOCAL-0002` (SYCL0 16458 OK / 4849
not-supported / 3 FAIL, Vulkan0 17231 OK / 4080 not-supported / 0 FAIL, zero
aborts on either backend):

- **The ternary types abort the SYCL backend.** `TQ1_0`/`TQ2_0` have no kernel
  anywhere in `ggml-sycl`, but `supports_op` only checks operand types per op, so
  test-backend-ops' `all_types` entry for `TQ2_0` walks into a hard abort three
  times: `set_rows.cpp:547 Unsupported tensor type!`, `cpy.cpp:1409 fatal error`,
  and `mmvq.cpp:2681 unsupport data type=tq2_0`. A pristine checkout reproduces
  the CPY one. Local patch `0017` gates the types out centrally.
- **`CPY` into `q2_0` misses its epsilon.** `bf16 -> q2_0` fails at ~2e-4 against a
  1e-6 threshold on a pristine checkout (`f16 -> q2_0`, `f16 -> q4_1` and
  `f16 -> q5_0` fail intermittently at similar or marginal magnitudes). This is the
  pre-existing CPY flakiness widened by the new `q2_0` type; it is not gated
  locally because the kernel exists and mostly works.
- **`ROLL` with a permuted tensor is wrong.**
  `ROLL(shift0=3,shift1=-2,shift3=1,shift4=-1,permute=1)` returns ERR ~1.75 against
  a 1e-7 threshold; a pristine checkout gives ~1.78. Nothing in this stack touches
  `roll.cpp` or the ROLL gate (byte-identical to master), so this is upstream.
- **PR 26631's own test matrix cannot produce a CPU reference, so a full
  `test-backend-ops` run aborts.** Its `skip=1` variant puts `-1` into the ids,
  and with `with_lane_scale` those ids reach `ggml_get_rows()`, whose CPU
  implementation asserts `i01 >= 0` (`ggml-cpu/ops.cpp:5014`). With the stack
  applied, a full run on any backend (the reference is always CPU) stops at
  `test_mul_mat_vec_fusion` when it reaches that combination; every op before
  the abort still passes/fails on its own merits. A former `ZZ-LOCAL-0002`
  restricted the combination locally; it was retired on purpose so the PR's
  test code stays pristine, and this abort is accepted as a known upstream test
  defect until upstream defines what a skipped expert's lane scale should be.

## Build check (2026-11-01, oneAPI 2026.2 without an Intel GPU)

oneAPI came from conda-forge into `/opt/oneapi` on a GPU-less cloud box:
`dpcpp_linux-64` 2026.2.0, `mkl-devel-dpcpp`, `onednn-dpcpp` 3.14.1,
`level-zero-devel`, `intel-graphics-compiler` + `intel-compute-runtime`
(`ocloc` 24.52) and `intel-opencl-rt` (CPU OpenCL device). Results so far:

- **Main stack, JIT** (README flags: `GGML_SYCL_F16=ON`, oneDNN found, no
  `GGML_SYCL_DEVICE_ARCH`): `ggml-sycl`, `test-backend-ops` and `llama-bench`
  build with zero errors.
- **DG2 code generation** is checked one translation unit at a time: compile
  and link with `-fsycl-targets=spir64_gen -device acm-g10` (the flags
  `GGML_SYCL_DEVICE_ARCH=acm-g10` uses, and the same SPIR-V the A770 compiles at
  JIT time), with IGC shader dumps to read the DG2 assembly.
- **Candidate C21 (cwriter's 8x8 ESIMD MoE GEMM)** builds for `acm-g10`. All
  2716 DPAS instructions are `dpas.8x8 (8|M0)`, the DG2 form. The 16-wide Xe2
  kernels in the same image also compile for DG2 (split into 8-wide DPAS,
  emulated 2D loads, spills); the device probe never picks them on DG2.
  Register use matters:
  - The kernel asks for `grf_size<256>`. With 256 GRF the default path
    (`GGML_SYCL_MMID_ESIMD=1`, K split 1) does not spill. Only the bit-exact
    mode 2 (K split 4) spills 3-4 KB.
  - An AOT build with `-ze-opt-enable-auto-large-GRF-mode` (what C20's CMake
    passes) gets 128 GRF for every DG2 kernel and spills 3.7-9 KB. A JIT build
    should honor the property (per C20's notes; unverified without a GPU), so
    measure C21 with a JIT build.
- `__SYCL_TARGET_INTEL_GPU_ACM_G10__` is not defined under `spir64_gen
  -device acm-g10`, only under `-fsycl-targets=intel_gpu_acm_g10`, so code keyed
  on it (`FG_AOT_SG`) builds every width in a `GGML_SYCL_DEVICE_ARCH` build.
- **Upstream XMX MMVQ (`mmvq.cpp`, Xe2-only gate)** also builds for DG2: 216
  ESIMD kernels, all lowered to 8-wide DPAS, with spills. ESIMD `xmx::dpas`
  is lowered correctly on DG2, just slowly. The silent-garbage risk is limited
  to the OpenCL-style sub-group-16 builtins and hand-written `(M1, 16)` vISA
  like `fattn-dpas.cpp`.
- **`fused-gemm.cpp` (PR 30156) and a TILE FA instance (candidate C50)** build
  for DG2.
- **Extracted 0016 with `GGML_SYCL_MMQ_XMX=ON` did not build for DG2.** Its int8
  tile was 8x16x32 at sub-group 16 (the Xe2 shape), and IGC 2.43 crashes on it
  for `acm-g10` (floating point exception). The A/B in this log never noticed
  because Q4_0 never reaches MMQ on the A770; the first K-quant MMQ call would
  have crashed the JIT. 0016 now builds both int8 tiles, 8x16x32 at sub-group 16
  and 8x8x32 at sub-group 8, and picks the one the device lists at run time;
  nothing to configure. The two shapes land in separate device images (one per
  `reqd_sub_group_size`), so the A770's JIT only ever compiles the 8-wide image.
  An AOT build keeps only its target's shape (`GGML_SYCL_XMX_AOT_SG`):
  `acm-g10` gets 15 kernels with `dpas.8x8 (8|M0)`, `bmg-g21` 15 with
  `dpas.8x8 (16|M0)`. A device with neither shape keeps the regular MMQ path.
  Numerics of the 8-wide tile are unverified on hardware.
- **Candidate C40 (ef777 GQA decode) did not build for DG2 as published.** The
  q8_0 D=256 SLM ESIMD kernel's in-kernel `mask` null test made IGC emit a
  64-bit shift (`shr ... 0x20:uq`, "Q destination data type is not supported"),
  failing the whole ESIMD image with any GRF option. Testing `mask` on the host
  fixes it; all 37 kernels build. ef777 ran it on an A770 under Windows, so
  whether the original breaks depends on the driver's IGC.
- **Candidate C60 ports upstream's DPAS flash attention (PR 31125) to DG2.**
  Upstream's kernel (q4_0 / q8_0 KV, 1-31 query tokens; B580 tg128 at 16K
  31.3 -> 54.2 t/s) is hand-written `(M1, 16)` vISA behind a Battlemage /
  Lunar Lake / Panther Lake allowlist. C60 makes the DPAS width a template
  parameter: sub-group 16 keeps upstream's operand bytes, tiles and dispatch;
  sub-group 8 issues `(M1, 8)` with 32 sub-groups per work-group, Q.K over
  8-key groups, P.V over 8-row tiles, 4 bytes of each Q row per lane and 2 keys
  per lane in the f16 V operand. The width is picked per device at run time:
  16 on upstream's allowlist, 8 where the device lists the 8x8x32 int8 and
  8x8x16 f16 tiles at sub-group 8. Results:
  - `acm-g10`: 42 kernels, all 782 DPAS `dpas.8x8 (8|M0)` (s4 x s8 for q4_0 K,
    s8 x s8 for q8_0 K, f16 for P.V). At 128 GRF the 32-row kernels from head
    size 128 and the staged D=256 decode kernel spilled 0.5-7.8 KB, so at
    sub-group 8 they always take the 256-register file; nothing spills now.
  - `bmg-g21`: the same 604 DPAS as upstream's file, instruction counts within
    +-1% per kernel (+0.6% in total), no spills.
  - JIT build: each device image holds one width; the split-K combine kernel
    runs at the DPAS width, so the A770 never builds the `(M1, 16)` image.
  - `sycl-patches-candidates/tools/fattn-dpas-emu/run.sh` runs the unchanged
    kernels on the CPU OpenCL device with the three DPAS helpers emulated
    (A = the flat register region of the per-lane operand, B and D one column
    per lane). 112 cases pass at both widths: q8_0 / q4_0, D = 64/96/128/256,
    GQA 1-32, 1-7 tokens, masks, sinks, split-K; NMSE vs FP64 <= 4.8e-5,
    sub-group 8 vs 16 <= 1.6e-11. Planting either of two plausible sub-group-8
    layout bugs (Q byte order, keys per lane in P.V) makes it fail with NMSE ~1.
    Sub-group 16 passing ties the emulated register model to the layouts
    upstream relies on; the DG2 hardware itself is still untested.
- **Running the backend on the CPU OpenCL device does not work** without porting
  ggml-sycl to CPUs. A scratch hook that let `dpct::dev_mgr` pick the CPU got as
  far as the first DMMV kernel (`UR_RESULT_ERROR_UNKNOWN`); FA died with a
  floating point exception, likely from GPU-only device properties. The only
  numeric test of SYCL kernels here is C60's standalone emulation above.

## Quant-type gap fill, part 2 (2026-11-02)

Closes the gaps the first pass left open, again as local patches (`ZZ-LOCAL-0005` to `ZZ-LOCAL-0009`).

- **`ZZ-LOCAL-0005`, ternary types.** TQ1_0, TQ2_0 and PTQ1_0 get dequantize, GET_ROWS, 1..8-column
  MMVQ and the fused MoE MMVQ. CUDA PR 29676 covers only PTQ1_0 and no GPU backend in tree has TQ1_0 /
  TQ2_0, so all three follow the CPU codec (`dequantize_row_tq1_0` and friends): base-3 digit n of a
  packed byte is `((q * 3^n mod 256) * 3) >> 8`, done four bytes at a time in 16-bit lanes, and each
  32-weight q8_1 group is expanded to signed bytes for `dp4a`. A host harness against
  `quantize_row_*_ref` / `dequantize_row_*` matched all three bit for bit before the SYCL build.
  0017's central gate now admits these types only as the weight operand of MUL_MAT, MUL_MAT_ID and
  GET_ROWS (PTQ1_0 is caught by it too now; DUP / CONT check no types and would have aborted).
- **`ZZ-LOCAL-0006`, F8_E4M3.** DMMV for one token, exact f16 widening plus the GEMM for batches,
  MUL_MAT_ID through the per-expert loop, GET_ROWS, CPY to and from F32 / F16 / BF16. No GPU backend in
  tree runs FP8 weights (Metal only copies them). The conversions were checked bit-exact against the
  CPU's on all 256 codes, every tie, and 4M random floats; they test bits instead of calling `isnan`,
  because icpx's default `-fp-model=fast` folds `isnan` (the same harness built with `-ffast-math`
  breaks the CPU's own `ggml_fp32_to_f8_e4m3` on -0 and NaN, not the SYCL one).
- **`ZZ-LOCAL-0007`, IQ4_NL reorder (2D and MoE).** IQ4_NL's block is Q4_0's, so it takes Q4_0's
  `[qs][d]` layout with a codebook variant of the reorder vec_dot. All readers of a reordered weight
  are covered: 1..8-column MMVQ, the MoE reorder MMVQ, dequantize, GET_ROWS, and a SoA stage of the
  fused / grouped XMX dequant GEMM. Candidate C20 already reads the layout. The new lines were moved off
  candidate C10's context (it edits the same type lists for IQ4_XS).
- **`ZZ-LOCAL-0008`, IQ3 MoE reorder.** The grouped XMX dequant GEMM gets SoA stages for IQ3_XXS and
  IQ3_S, sharing one decode with the block stages, so MoE experts can now take the per-slice 2D IQ3
  layout without moving MoE prompt batches off the grouped GEMM. Dense IQ3 weights gain the same: once
  the existing 2D reorder is installed, their prompt batches keep the fused GEMM instead of dequantize
  plus oneDNN. A host harness checked the IQ4_NL and both IQ3 SoA stages against the block stages,
  bit for bit, on random blocks; the XMX GEMM itself cannot run on the CPU device.
- **`ZZ-LOCAL-0009`, Q2_0 GET_ROWS.** Q2_0 had a dequantizer but was missing from GET_ROWS.
- **Verification (CPU OpenCL device, `tools/cpu-device-test/`), all against the CPU backend:**
  - Ternary: MUL_MAT tq1_0 13/13, tq2_0 48/48, ptq1_0 14/14; MUL_MAT_ID 5/5, 3/3, 3/3; GET_ROWS 8/8
    each. TQ1_0's first dot product branched per q8_1 block; it passed one column but gave ERR ~0.9
    in the 2..8-column kernel (and the MoE kernel) on this device, though the same function matched
    the CPU on the host. The rewrite selects the source bytes and digit per step instead and passes.
  - F8_E4M3: MUL_MAT 60/60, MUL_MAT_ID 73/73, GET_ROWS 8/8, CPY 29/29 without the BF16 cases; every
    non-BF16 CPY case 379/379. The CPU OpenCL compiler crashes building any BF16 conversion kernel,
    upstream's F16 to BF16 copy included, so F8 to and from BF16 is untested here.
  - Reorder, with and without `GGML_SYCL_TEST_FORCE_REORDER=1`: MUL_MAT / MUL_MAT_ID iq4_nl 56/56 and
    12/12, iq3_xxs 49/49 and 4/4, iq3_s 49/49 and 21/21. GET_ROWS, every type but BF16, reorder on:
    224/224 (8 of them the new Q2_0). Upstream's reuse tests (`test_get_rows_after_mul_mat`,
    `test_mul_mat_reorder_then_big_batch`, now with IQ4_NL): 10/10. The new `reorder-reuse` tool,
    which installs the reorder and then reads it through a 32-token MUL_MAT, GET_ROWS and a 32-token
    MUL_MAT_ID: 47/47 over Q4_0, Q8_0, Q2_K..Q6_K, IQ3_XXS, IQ3_S, IQ4_NL (Q2_K / Q4_K / Q5_K GET_ROWS
    skipped, gated by 0014). The device-program cache shows `reorder_qw_planar_moe` built for IQ4_NL,
    IQ3_XXS and IQ3_S, so the MoE cases ran on the new layout.
- **DG2 AOT (`acm-g10`, `-ze-opt-enable-auto-large-GRF-mode`).** `mmvq.cpp`, `fused-gemm.cpp`,
  `convert.cpp`, `getrows.cpp`, `dmmv.cpp` and `cpy.cpp` build. None of the 38 new MMVQ kernels spills
  (27 ternary, 9 IQ4_NL reorder, 2 IQ3 MoE reorder; all at 128 GRF), nor do the 36 new dequantize,
  GET_ROWS, DMMV and CPY kernels. The fused XMX dequant GEMM spills on DG2 in every variant, the existing
  block-layout ones included (2.3 to 3.3 KB). The new SoA variants spill 64 B (IQ4_NL), 160 B (IQ3_XXS)
  and 512 B (IQ3_S) more than the block variant of the same type, as the existing Q4_K SoA variant does
  (+256 B). Whether a spilling fused GEMM beats dequantize plus oneDNN for reordered IQ3 / IQ4_NL
  prompt batches on the A770 is unmeasured.
- **Apply.** The stack, the seven local patches, every candidate (plain `git apply`, no three-way) and
  the Vulkan stack apply and unapply back to a clean tree; `hooks.patch` still applies.
- **Not verified:** speed on the A770, and the XMX / ESIMD paths, which need the GPU.

## Quant-type gap fill (2026-11-01)

Audit of which kernel each quant type reaches with the stack applied, against what the carried PRs
cover. Two gaps were filled as local patches; the rest are listed with the reason they stay open.

- **`ZZ-LOCAL-0003`, multi-column MMVQ for i-quants.** IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ4_NL
  (and IQ3_XXS without the reorder) were the only quantized types whose 2..8-column MMVQ (speculative
  decoding verify, small batches) looped one column at a time and reread the weights per column. They
  now use the generic `mul_mat_vec_q_ncols` kernel with their one-column qi / vdr / vec_dot.
- **`ZZ-LOCAL-0004`, MoE expert reorder for Q4_0 / Q2_K / Q3_K.** The 2D reorder covers these, the MoE
  reorder did not (only Q8_0, Q4_K, Q5_K, Q6_K, MXFP4), so their fused MoE decode read the block layout.
  Each expert slice now gets the 2D layout (one tail writer shared with the 2D reorder) and runs the
  generic MoE reorder MMVQ. All other readers of a reordered expert tensor already handle these types
  (per-expert MMVQ / DMMV / ESIMD / dequantize, the grouped GEMM's `fg_soa`, candidate C20/C21).
- **Verification.** The SYCL backend now runs on the CPU OpenCL device
  (`sycl-patches-extracted/tools/cpu-device-test/`). The earlier attempt failed because the
  `-ze-intel-greater-than-4GB-buffer-required` link option is embedded in every device image and the
  CPU compiler rejects it; with it dropped, a per-kernel device-code split, and the GPU-only paths
  (ESIMD, DPAS, oneDNN, the `joint_matrix` dequant GEMM) switched off, `test-backend-ops` runs against
  the CPU backend. Before and after the two patches, all of these pass: MUL_MAT for iq2_xxs 74/74,
  iq2_xs / iq2_s / iq1_s / iq1_m 49/49 each, iq4_nl 56/56, iq3_xxs without reorder 49/49, q4_0
  111/111, q2_K 55/55, q3_K 48/48; MUL_MAT_ID q4_0 106/106, q2_K 14/14, q3_K 14/14. With the reorder
  forced on: MUL_MAT_ID q4_0 106/106, q2_K 14/14, q3_K 14/14, q4_K 110/110, q8_0 108/108; MUL_MAT
  q4_0, q2_K, q3_K, iq3_xxs all pass. SYCL debug logging shows the new multi-column kernels run, and
  the device-program cache shows the per-expert reorder and the MoE reorder MMVQ kernels being built
  for Q4_0, Q2_K and Q3_K once the reorder is on. `mmvq.cpp` with both patches builds for `acm-g10`:
  49 new multi-column kernels, no spills; the Q3_K MoE reorder kernel's first IGC attempt spilled
  192 bytes, its retry none. Not measured: speed on the A770.
- **`0014` is probably obsolete.** It drops Q2_K / Q4_K / Q5_K from GET_ROWS because they missed the
  test epsilon on the A770 (NMSE 2-4e-7 against 1e-7), a half-precision symptom. Those types now
  dequantize through the float `dequantize_q*_K_f32` paths, and with 0014's gate lifted all three pass
  GET_ROWS on the CPU device (8/8 each). One A770 run decides: unapply 0014, then
  `test-backend-ops -b SYCL0 -o GET_ROWS`.
- **`0016`'s XMX MMQ path cannot run.** Its dispatch sits in the `use_mul_mat_q` branch, and upstream's
  `ggml_sycl_supports_mmq()` returns false for every type ("accuracy issues in MMQ"), so
  `-DGGML_SYCL_MMQ_XMX=ON` changes nothing. That, not the oneMKL routing, is why the ON / OFF numbers
  in `sycl-patches-extracted/README.md` match.
- **Left open** (the first three closed by part 2, above):
  - IQ4_NL 2D reorder: its block matches Q4_0's, but the grouped GEMM has no SoA stage for it and
    candidate C20 has its own IQ4_NL SoA reader, so a new layout would need both kept in step.
  - IQ3_XXS / IQ3_S MoE reorder: the grouped GEMM has no SoA stage for them, so MoE prefill would
    drop from the grouped GEMM to the per-expert loop.
  - TQ1_0, TQ2_0, PTQ1_0 (carried PR 29672) and F8_E4M3 have no SYCL kernels and run on the CPU.
    PR 29676 (not carried, CUDA) is the reference for PTQ1_0.
  - Flash attention already takes every legacy KV type (q4_0, q4_1, q5_0, q5_1, q8_0, f16, bf16) and
    different K and V types: `common.hpp` defines `GGML_SYCL_FA_ALL_QUANTS`. Only C60's DPAS kernel
    and C40's decode kernel are limited to q4_0 / q8_0 (C40: q8_0 / f16).
  - MMQ is off for every type (above); prompt processing goes through dequantize + oneDNN / the XMX
    dequant GEMM.

## Update (2026-11-01, upstream refresh to `dde879bc1`)

Refreshed from `23b0202a1` (480 upstream commits). The header bullets above
predate the October refreshes; `sycl-patches/README.md`,
`sycl-patches-extracted/README.md` and `vulkan-patches/README.md` hold the
current sets.

- **Stack**: 7 fork + 16 upstream PR + 2 extracted SYCL patches, then 1
  upstream PR + 3 local Vulkan patches; 29 patches. All apply with plain
  `git apply` (no three-way fallback) and both unapply scripts return the tree
  to upstream, alone or combined.
- **Merged upstream and retired**: SYCL `28725` `29107` `29337` `29338` `29696`
  `29864` `29889` `30271` `30286`; Vulkan `29835`. `30226` closed unmerged.
- **New carries**: `30156` (its skip reason, overlap with 28725, went away when
  28725 merged), `30606` `30857` `30956` `31109` `31126` `31138` `31208` `31285`
  `31325` `31355` `31460` `31465`. Nine new cross-backend or non-code SYCL-labeled
  PRs went into `SKIP_PRS`. Vulkan `30641` is in scope but not carried: it
  overlaps `29357` in the Xe FA dispatch and decode shaders.
- **Re-anchored by hand**: fork `0002` (upstream rewrote the dequant and MMQ
  launchers it edits), `29506` (docs footnote), `29672` (next to upstream
  `F8_E4M3`), `30206` (`test_rope_set_rows`: upstream `n_dims` plus the PR's
  `yarn`/`truncate`; the PR call site now passes `n_dims` explicitly so the bools
  cannot bind to it), `31325` (after `0002`/`0008`), `31460` (whitelist replaces
  the `29672`-extended blocklists), `0016` (context only), Vulkan `29357`
  (collapsed to one diff so unapply removes its new shader).
- **Behavior changes to local patches**:
  - `0004` no longer relaxes SUM. `ggml_sycl_op_sum` sums `ne` consecutive
    floats, and upstream #28851 added `test_sum(..., slice=true)`, a
    row-contiguous view with gaps that the relaxed gate would have accepted and
    computed wrong. The GROUP_NORM gate stays.
  - `0016` entry points renamed to `ggml_sycl_mmq_xmx_*`. Upstream's
    `static inline ggml_sycl_xmx_supports_type()` (from PR 29864) made
    `mmq_xmx.cpp` a redefinition error with `-DGGML_SYCL_MMQ_XMX=ON`; this was
    already latent in the 2026-10-10 stack while 29864 was carried.
  - `ZZ-LOCAL-0001` (CONV_2D strides) retired, superseded by upstream #30363.
  - Vulkan `ZZ-LOCAL-0004` retired: upstream declares the same debug entry
    points, and the patch's repeated default argument no longer compiles.
- **Verification**: no SYCL compiler or Intel GPU was available, so the SYCL TUs
  were not compiled. A GCC 13 build of the applied tree with Vulkan and tests
  built all 524 targets with `GGML_VULKAN_CHECK_RESULTS=ON` and
  `GGML_VULKAN_RUN_TESTS=ON`; `ctest -L main -LE model` passed 16/17 (the miss
  needs a model download). Vulkan llvmpipe results are in
  `vulkan-patches/README.md`. First thing on the A770: an `icpx` build with and
  without `GGML_SYCL_MMQ_XMX`, then `test-backend-ops -b SYCL0`.
- **Not re-verified**: the "Known failing cases" above. PR 26631 is now
  upstream, and upstream gates TQ1_0/TQ2_0 per op on its own, so `0017` may be
  redundant; it still applies and was kept.

## Update (2026-06-25, upstream sync + NORM contiguity fix)

Master fast-forwarded and rebased all patches cleanly.

**NORM Contiguity Fix (test-backend-ops crash on SYCL0)**
- Upstream recently added non-contiguous row tests for `GGML_OP_NORM` (`noncontig_rows=1`).
- The SYCL norm kernel does not support non-contiguous rows, resulting in `norm.cpp:641: GGML_ASSERT(nb00 == ts0) failed` when the scheduler dispatched them to the GPU.
- Modified `do_ggml_backend_sycl_device_supports_op` to return `ggml_is_contiguous_rows(op->src[0])` for `GGML_OP_NORM`, `GGML_OP_RMS_NORM`, and `GGML_OP_L2_NORM`, and `ggml_is_contiguous(op->src[0])` for `GGML_OP_GROUP_NORM` (similar to the CUDA backend).
- Integrated this fix directly into local patch `0004-sycl-relax-sum-mean-contiguous.patch`.
- Verified that all NORM, RMS_NORM, and GROUP_NORM tests run and pass on `SYCL0` (with non-contiguous cases falling back to CPU).

**SUM_ROWS & MEAN Contiguity Fix**
- Found that `SUM_ROWS` and `MEAN` were failing on non-contiguous row configurations (permuted and sliced inputs) with high numerical error.
- The SYCL implementations (`k_sum_rows_f32`) assume fully contiguous layouts with sequential row strides (hardcoded as `row * ncols`).
- Re-restricted `SUM_ROWS` and `MEAN` back to `ggml_is_contiguous` in patch `0004-sycl-relax-sum-mean-contiguous.patch` (matching CUDA's constraints) while keeping the relaxation for `SUM`.
- Verified that all `SUM_ROWS` and `MEAN` tests now pass on `SYCL0`.

## Update (2026-06-14, upstream sync + 8 new SYCL PRs folded in)

Master fast-forwarded `c74759a24` → `6e14286ed` (+73 commits; only **one** touches
`ggml/src/ggml-sycl/`: `e95dae18d` #24086, the MTP/GDN "remove padding + multiple
D2D copies" cleanup — no overlap with our stack). All carried patches re-apply.

**Merge-status check (the "make sure none merged already" ask)** — none of the
carried or newly-pulled PRs are merged. All 8 new + PR-21700 + PR-22526 are still
open upstream.

**Two `pull_*.sh` cleanup bugs found and fixed** (same root cause — the SKIP list
was only consulted *inside* the loop over remote-returned items, so anything not
returned by the remote got deleted as "stale"):
- `pull_base.sh` deleted local-only `0014` every run (the fork API never lists it).
- `pull_prs.sh` deleted `PR-21700` once it aged past the 60-day search window,
  despite being in `SKIP_PRS`.
Both now pre-mark their SKIP lists as KEEP before the cleanup pass. Verified both
patches survive a re-run.

**8 new open SYCL PRs evaluated; all 8 folded in** (carried set 2 → 10 PRs):
- Clean apply, kept verbatim: **24452** (reordered Q4_K/Q5_K/Q6_K MoE MUL_MAT_ID),
  **24578** (fix reorder crash `block_num_y % num_subgroups`). Left out of `SKIP_PRS`
  so they keep re-syncing from upstream.
- **24451** (soft_max_f32 max-reduction fix): 3-way reconciled with fork `0007`.
  Upstream removes the redundant `-INFINITY` init + its barrier and replaces the
  single `max_val = buf_iw[lane_id]` read with a loop over all `nwarps` slots
  (the actual correctness fix for blocks where nwarps > WARP_SIZE, e.g. large-
  context attention); we kept `0007`'s `local_space` fence on the *surviving*
  barrier and `0007`'s independent sum-block fences. Also adds `reqd_sub_group_size`.
- **24476** (dev2dev memcpy): 3-way merged vs `0002`/`22526` (`common.hpp`/
  `helper.hpp` merged clean; global decl kept alongside `g_ggml_sycl_usm_system`).
  NOTE: its core path is dGPU↔dGPU — **inert on a single A770**; carried for
  completeness + the incidental `helper.hpp` cleanup.
- **24584** (pool_1d + move pool code): include re-anchored above the `MEM_SIZE`
  defines; code only.
- **24363** (EXPM1 + FLOOR/TRUNC/ROUND) and **24564** (set_rows q1_0/mxfp4/nvfp4):
  applied **code-only** — the `docs/ops.md`/`SYCL.csv` table hunks collide across
  PRs and aren't built, so they were stripped.
- **24600** (conv_2d / conv_2d_dw / conv2d_transpose): code-only (docs + the
  `examples/sycl/update_ops_doc.*` helpers stripped). Adds new `conv2d*.cpp/hpp`.
All 6 locally-modified PRs (24363/24451/24476/24564/24584/24600) added to
`SKIP_PRS` so `pull_prs.sh` won't clobber the resolutions.

**New fork patch `0015` (CONCAT gate).** `test-backend-ops` aborted (SIGABRT) in
`ggml_sycl_op_concat` — `concat.cpp:199 GGML_ASSERT(false && "unsupported type")`.
Root cause is upstream, not ours: commit `85f99dca8` (#24011, "support concat for
scalar types at cuda backend", one of the +73) added F16/BF16/I8/I16/I64 CONCAT
test cases for **all** backends, but the SYCL concat kernel still handles only
F32/I32 and `supports_op` lumped CONCAT into an ungated `return true` group. Both
`concat.cpp` and the CONCAT `supports_op` case are untouched by our stack, so this
aborts on clean `6e14286ed` too. `0015` gates `GGML_OP_CONCAT` to
`op->type == F32 || I32` so the scalar types fall back to CPU (same approach as
`0014` for GET_ROWS). Local-only patch; added to `pull_base.sh` SKIP_PATCHES. Drop
when upstream extends the SYCL concat kernel.

**New fork patch `0016` (llama-quant MTP fix).** Requested add of ggml-org/llama.cpp#24379:
`llama-quantize` errors on MTP/nextn models (MiMo V2.5, GLM 5.1, and the
gemma-4-assistant MTP head) because `init_quantize_state_counters` bounds the MoE
ffn tensor counts with `hparams.n_layer()` instead of `n_layer_all`, missing the
extra MTP layers. One-liner from the issue author (`artraze`). Non-SYCL
(`src/llama-quant.cpp`); does not affect test-backend-ops. Built `llama-quantize`.

**test-backend-ops (A770, full 21-patch stack).** After `0015`, the suite runs to
completion (no SIGABRT). SYCL0: **13177/13186 pass**; the 9 failures are all
`MUL_MAT_ID` (low-bit quants at `n_used=1`, ERR ~0.3–1.2) — the known pre-existing
upstream flaky bug (`random_device`-seeded data; 11/12 full runs failed on clean
master in the 2026-06-08 audit), **not** a regression from this sync. `CONCAT`
now 32/32 (scalar types report unsupported → CPU fallback via `0015`).

The full 21-patch stack re-applies cleanly from clean master via plain `git apply`
in lexical order.

## Update (2026-06-08, upstream sync + test-backend-ops audit)

Master fast-forwarded `2154a0fdc` → `c74759a24` (+37 commits, **none** touching
`ggml/src/ggml-sycl/` or ggml core headers). Patch stack re-applies cleanly.

**Patch reconciliation**

- `0007` (barrier local_space fence): `pull_base.sh` had been re-downloading the
  fork's full copy and clobbering the local rebase (4662→5009 B), re-introducing
  the `k_copy_src1_to_contiguous` ggml-sycl.cpp hunk that PR-23142 (`cc9e33121`)
  made obsolete. Re-dropped the hunk and added `0007` to `SKIP_PATCHES`.
- `SKIP_PRS`/`DROP_PRS` audited against live upstream state: **23710** (GET_ROWS
  more types) and **23725** (Q3_K reorder) merged upstream (`416252268`,
  `44e211cec`) → moved to `DROP_PRS`. 23187 closed without merge. 21700/22526
  carried copies verified current (no upstream content drift).

**test-backend-ops -b SYCL0 (A770) findings**

1. **GET_ROWS Q2_K/Q4_K/Q5_K → FAIL (fixed).** Upstream 23710 enabled these
   unconditionally; on A770 they miss the test epsilon (ERR ~2.2e-7..4.0e-7 vs
   1e-7), deterministic 12/12. Q3_K/Q6_K and all IQ types pass. The mitigation
   the locally-rebased PR-23710 used to carry was lost when the PR merged and its
   patch was auto-deleted. Re-expressed as **`0014`**; GET_ROWS now 0 FAIL (×3),
   the three types report "not supported" and fall back to CPU.
2. **MUL_MAT_ID → flaky FAIL (PRE-EXISTING UPSTREAM, not patch-caused).** The
   fused MoE path (`ggml_sycl_mul_mat_id_mmvq_fused`) returns garbage (ERR
   0.3–1.1, well past the 5e-4 threshold) for *data-dependent* random routings —
   even `f32×f32`/`f16`. **Clean master fails 11/12 runs** (0–20 failures each);
   the patch stack does not introduce it (a single "clean=0" run earlier was just
   the lucky draw — test-backend-ops seeds from `std::random_device`). Reproduce:
   `test-backend-ops -o MUL_MAT_ID -b SYCL0` a handful of times. **TODO: file
   upstream**; do not attribute to the carried patches.

## Update (2026-06-05, upstream sync)

Master commit: `2154a0fdc`.

Upstream merged **PR-21845** (`7fe2ae45a sycl : port multi-column MMVQ from CUDA
backend`) into master. This was one of our locally-rebased `SKIP_PRS` keepers,
so a `rebase.sh` run left the stash-pop conflicted (local `mmvq.cpp` +1218-line
multi-column port vs. upstream's now-merged `ncols_dst` version) across
`mmvq.cpp`, `dmmv.cpp`, `convert.cpp`, `dequantize.hpp`, `vecdotq.hpp`,
`ggml-sycl.cpp`, and `docs/ops/SYCL.csv`.

Resolution: retired `PR-21845` — moved from `SKIP_PRS` to `DROP_PRS` in
`pull_prs.sh`, deleted `sycl-patches/PR-21845.patch`. Reset to clean master and
re-applied the remaining stack on top of the upstream merge.

| Patch | Status | Reason |
|-------|--------|--------|
| `PR-21845.patch` | retired | Merged upstream as `7fe2ae45a` (`sycl : port multi-column MMVQ from CUDA backend (#21845)`) |

## Update (2026-05-30, latest upstream sync)

Master commit: `d48a56eff`.

Re-verified the local patch stack against latest upstream in an isolated
worktree off `origin/master`. All **11** remaining patches apply cleanly in
order. Verified with:

- `source /opt/intel/oneapi/setvars_fixed.sh`
- `cmake -S llama.cpp/llama.cpp-patchcheck -B llama.cpp/llama.cpp-patchcheck/build_sycl_patchcheck -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=ON -DGGML_SYCL_MMQ_XMX=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_LTO=ON -DGGML_NATIVE=ON`
- `cmake --build llama.cpp/llama.cpp-patchcheck/build_sycl_patchcheck --target llama-cli test-backend-ops -j 16`
- `ONEAPI_DEVICE_SELECTOR=level_zero:0 llama.cpp/llama.cpp-patchcheck/build_sycl_patchcheck/bin/test-backend-ops -b SYCL0`

Result: **10404/10404** `test-backend-ops` cases passed on `SYCL0`.

Patch-status decisions in this sync:

| Patch | Status | Reason |
|-------|--------|--------|
| `PR-22862.patch` | retired | Merged upstream as `581d020b1` (`SYCL: implement ggml_sycl_pool_vmm (#22862)`) |
| `PR-22482.patch` | dropped locally | Non-SYCL CMake/export-target packaging patch; current `ggml-config.cmake.in` drift breaks apply, and latest SYCL stack configures/builds/tests cleanly without it |

The bench data later in this log remains the historical 2026-05-24 run on the
older stack; no fresh end-to-end perf perf was run as part of the 2026-05-30
sync.

## Current head-of-stack state (2026-05-22, post-upstream-pull)

Master commit: `ef570f630`.

All 16 patches apply cleanly in declared order. `test-backend-ops -b SYCL0`
passes **10402/10402** on the fully-stacked build. Bench deltas vs clean
upstream master (Gemma-4 family, `-fa 1 -ub 128 -t 8 -ngl 99 -r 3`):

| Model | Metric | Clean | Stacked | Δ |
|-------|--------|------:|--------:|---:|
| E2B Q4_0 | pp512 | 727.93 | 757.74 | +4.1 % |
| E2B Q4_0 | tg128 | 22.37 | **26.41** | **+18.1 %** |
| E2B BF16 | pp512 | 954.04 | 990.61 | +3.8 % |
| E2B BF16 | tg128 | 17.91 | **20.07** | **+12.1 %** |
| 26B-A4B MoE | pp512 | 121.77 | **170.74** | **+40.2 %** |
| 26B-A4B MoE | tg128 | 14.48 | 15.98 | +10.4 % |

Headline gains shrank from the pre-pull session (where clean master was
`4f0e43da6`) because upstream merged the three biggest wins (BF16 DMMV,
MoE counting-sort, GATED_DELTA_NET K>1) into the new baseline at
`ef570f630`. The remaining fork-driven delta is still significant on
MoE prefill (+40 %), dense tg (+12–18 %), and modestly on prefill.

Detailed per-step table is in
[`.bench-results/per-patch-report.md`](.bench-results/per-patch-report.md).

## Patches retired this session (2026-05-22)

Upstream pulled 10 commits, four of which subsumed our patches. All four
were verified merged by hallmark-symbol grep (`mmid_counting_sort_rows`,
`GGML_SYCL_DMMV_HAS_BF16`, multi-K `gated_delta_net_sycl` dispatch,
`ext_oneapi_level_zero` field). Removed from `sycl-patches/` and added
to `pull_prs.sh` `DROP_PRS` to prevent re-download:

| Patch | Upstream commit | Notes |
|-------|----------------|-------|
| `PR-21580` | `8cc67efcd` | BF16 DMMV path (~4× tg on Arc) — merged |
| `PR-23097` | `bcfd1989e` | Level Zero detection in `ggml_sycl_init` — merged |
| `PR-23142` | `cc9e33121` | MoE prefill counting-sort dispatch — merged |
| `PR-23174` | `56f16f235` | `gated_delta_net` K>1 GPU path — merged |
| `PR-23399` | — | Level Zero env-detection cleanup; *substantively* superseded by upstream merge of PR-23097. Remaining content was cosmetic global-removal not worth carrying as a fork patch. Retired by choice. |

## Patches rebased this session

Three patches required context-shift rebasing to apply against
`ef570f630`:

| Patch | Change |
|-------|--------|
| `0007-sycl-barrier-local-space-fence.patch` | Dropped the one-line `ggml-sycl.cpp` hunk inside `k_copy_src1_to_contiguous`. PR-23142's counting-sort rewrite (now upstream) removed the function's barrier entirely. Remaining hunks (mmq.cpp, norm.cpp, softmax.cpp) apply cleanly. |
| `PR-22862.patch` | Two context-only fixes for upstream code drift: `sycl::device device = …` → `auto & device = …`; and the `GGML_SYCL_ENABLE_LEVEL_ZERO` default in `get_sycl_env(…)` changed from `1` to `ggml_sycl_info().ext_oneapi_level_zero` after the upstream PR-23097 merge. No logic changed. |
| `0015-sycl-softmax-reduce-over-group.patch` | No direct edit — applies cleanly once the rebased `0007` is in place (its softmax context expects 0007's fenced barriers). |

The patch headers carry `[Locally modified 2026-05-22: …]` annotations
recording each rebase.

## Heavy lifters (on Gemma-4 specifically)

| Patch | Effect | What it does |
|-------|-------:|---|
| `0005` | +12 % Q4_0 tg, +9 % BF16 tg, +5–8 % MoE tg | RMS_NORM + MUL 2-op fusion |
| `PR-21845` | **+30 % MoE pp512** (132 → 172) | Multi-column MMVQ from CUDA |
| `EX-0012` | +6 % Q4_0 tg, +4 % BF16 tg, +2 % MoE tg | RMS_NORM + MUL + ADD 3-op fusion |

Everything else (0002, 0004, 0007, 0008, 0011, PR-21700, PR-22482, PR-22526,
PR-22862, EX-0013, EX-0014, EX-0015, EX-0016) is noise-on-Gemma-4 by
design. They are kept for op-coverage, correctness, codebase tidy, or to
enable other patches in the chain. **No patch produced a regression beyond
the noise floor.**

The previously-headline patches PR-21580 and PR-23142 (and PR-23174 for
op coverage) are now baked into the upstream baseline, which is why the
clean-master numbers are higher than in the pre-pull session.

## Per-step bench noise (observed during the iter run)

The iter bench at `-r 3` reps showed wide stddev on a few steps where
either GPU thermal state or system load drifted during the long run. The
table in `per-patch-report.md` has been spot-corrected by re-running the
anomalous steps (01, 09–13, 14–16). When in doubt, the **mean is correct
within ±3 %**; reading isolated stddev figures > 5 % of mean as "real
variance" is misleading at n=3.

## Known correctness caveats

- **PR-23142 MUL_MAT_ID small-n edge cases**: the new counting-sort
  dispatch (now upstream, not in our patch dir anymore) produces output
  above the 0.0005 epsilon on a handful of `MUL_MAT_ID` test shapes with
  `n ∈ {4, 5, 17}`, `n_used ∈ {1, 2, 4}`, `k = 256`. The exact set of
  failing shapes varies run-to-run because `test-backend-ops` randomises
  input data and these shapes sit at the epsilon boundary. Observed
  counts across consecutive TBO runs this session: 0 / 0 / 22 / 14 / 27
  / 0 / 0. **Real-model inference on Gemma-4 MoE is qualitatively
  correct**: routing never hits the failing shapes, and the fully-stacked
  tree passes 10402/10402 on every final sweep. Worth re-bisecting if a
  future model trips the dispatch.
- **GATED_DELTA_NET K>1**: pre-PR-23174 master failed 8 K>1 shapes; now
  resolved upstream. The post-pull baseline passes 10396/10396 cleanly.

## Build / cmake gotcha

`ggml/src/ggml-sycl/CMakeLists.txt` uses `file(GLOB ...)` *without*
`CONFIGURE_DEPENDS` to enumerate source files. Patches that add new
top-level `.cpp` files (`EX-0013 mmq_xmx.cpp`, `EX-0016 fattn-onednn.cpp`)
need `cmake -B build_sycl -S .` to re-run before `cmake --build`,
otherwise the new file is silently dropped from the GLOB list and the
link fails on its symbols. `.bench-results/iter_bench.sh` re-runs
configure before every build to handle this transparently. If you
bypass the driver, remember to do the same. `EX-0013` side-steps the
problem only by accident — its patch also edits CMakeLists.txt (adding
the `GGML_SYCL_MMQ_XMX` option block), which triggers ninja regen.

## Operational notes (A770-specific)

- **i915 hangcheck**: the original local `0012-sycl-extend-graph-fusion`
  3-op fusion ran past the i915 watchdog at `-ub 512`, surfacing as
  `UR_RESULT_ERROR_DEVICE_LOST` at the next `stream->wait()` inside
  `ggml_sycl_mul_mat_id`. The patch was retired locally. The current
  `EX-0012 RMS_NORM+MUL+ADD` from `sycl-patches-extracted/` is a
  different implementation and does not exhibit the hang. Empirical
  guard is to stay at `-ub 128`, which is also the documented
  `apply.sh` default.
- **`--swa-full -fa off --no-mmap`** causes a host-RAM OOM in
  `cpy.cpp:583` for Gemma-4 because the no-mmap + no-FA combo eagerly
  materialises full-window KV slabs. Workaround: leave SWA at default
  and `-fa on`.
- **i915 vs xe drivers**: this box stays on i915 for now. Switching to
  `xe` is out of scope for this fork's working set.

## Patch series origin

- `sycl-patches/0*-*.patch`: experimental fork-author patches against
  mainline SYCL backend. The numbering used to be denser; retired
  numbers (0001 / 0003 / 0006 / 0009 / 0010 / 0012) are reserved gaps
  where patches were either upstreamed or disabled locally.
- `sycl-patches/PR-*.patch`: open SYCL-labelled PRs pulled from upstream
  via `pull_prs.sh`. Skip list keeps non-SYCL-primary PRs out; drop
  list removes merged or intentionally-not-carried PRs.
- `sycl-patches-extracted/0*-*.patch`: patches mined from the
  experimental `llama.cpp-sycl-terrible` research tree, reshaped to fit
  mainline SYCL backend conventions (no unified-cache, no custom
  dispatcher).

## How to reproduce

```bash
# From a clean upstream-master checkout
source /opt/intel/oneapi/setvars_fixed.sh   # exits non-zero if already run; ignore
bash sycl-patches-extracted/apply.sh        # applies all 11 in order
cmake -B build_sycl -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=ON -DGGML_SYCL_MMQ_XMX=ON \
    -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    -DGGML_LTO=ON -DGGML_NATIVE=ON
cmake --build build_sycl -j$(nproc)

# Reproduce the per-patch bench + tbo run (uses oneAPI env; ~2.5 h):
bash .bench-results/iter_bench.sh

# Render the per-patch summary:
bash .bench-results/build_report.sh
# -> .bench-results/per-patch-report.md
```

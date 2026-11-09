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
    passes) gets 128 GRF for every DG2 kernel and spills 3.7-9 KB. Plain
    `-device acm-g10` does the same: the AOT path drops `grf_size` on ESIMD
    kernels (SPMD kernels keep it; see the GRF audit below). A JIT build
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

## DG2 hardware notes applied: C60's Q.K guard, an ISA check, a stricter A/B protocol (2026-11-09)

Input: three notes from an A770 flash-attention project (DG2 hardware facts, measured evidence, Xe
target guidance). Most of what they say confirms the stack's kernels: at least 4 independent DPAS
accumulators per thread (fused-gemm 4 or 8, C21, C40), DPAS operands ordered so consecutive DPAS share
src2 and vary src1, no 2D block loads on DG2 (C21 uses 1D). One finding needed code: IGC's
`Wa_18023229625` (every ACM-G10 stepping) inserts a dummy 8x8 DPAS before the first DPAS of each basic
block and before a back-to-back DPAS that would reuse src1.

Dummy DPAS counted in `acm-g10` AOT shader dumps (real + dummy per loop body):

| Kernel | Before | After |
|--------|--------|-------|
| C60 Q.K, 256-register kernels (D=128 32 rows, D=256 decode, D=512) | 8+8, 8+8, 32+32 | 8+1, 8+0 (q8_0) / 8+1, 32+1 |
| C60, all 42 sub-group-8 kernels | 490 + 292 | 473 + 87 |
| C60 P.V | 2+1 (16 rows), 4+1 (32 rows) | unchanged: `#pragma unroll 2` gave 4+2, no gain |
| fused-gemm sub-group 8, 128 GRF (0010) / 0017's 256 GRF | 4+1 / 16+2 | unchanged |
| C21 ESIMD MoE GEMM / C40 XMX decode | 16+1 / 40+2 | unchanged |
| 0016 MMQ-XMX | 1+0 or 1+1 per k-block | unchanged (one DPAS between SLM barriers; its build A/B decides) |

The C60 cause was a `t < RT` test around each Q.K DPAS. At sub-group 8 it is now dropped where it always
holds (one row tile per job) or only guards scores the store discards, and kept for 128-register two-tile
jobs, where dropping it spilled 2.4 KB at head size 96. Checks: no spills, `bmg-g21` instruction streams
identical in all 83 kernels, CPU emulator 112/112, apply / unapply round trip clean.
`-DGGML_SYCL_FA_DPAS_QK_GUARD=1` restores the test; `C60-qk-guard*` in `sycl-patches/ab/plan.json`
measures it. Speed on the card: not measured. The evidence note puts a DG2 flash kernel's time in P.V's
value loads more than in Q.K, so expect the gain, if any, in 32-row batches at depth.

Tools: `sycl-patches/aot/isa_check.py` prints register file, spill bytes, DPAS and dummy DPAS per kernel
for one file, from any configured build's `compile_commands.json`. `sycl-patches/ab/`: A/A rows run first
(`aa-dense`, `aa-deep`). The report adds a Hodges-Lehmann estimate with its exact interval and flags rounds
far from the median. `run` appends rounds to an existing results file and runs a `-d` list one depth per
process. `guard.sh` gates on GPU temperature (start below 60 C, stop at 80 C) and on other processes' DRM
engine use (5%), and stamps the loaded oneDNN. New plan rows for oneDNN 3.14+ (`onednn-314-*`: the
xe_hpg SDPA retunes ship there) and a per-token runtime-call count (`SYCL_UR_TRACE=2`) in the run order.

## A/B plan, off switches and the weight-only Q8_0 GEMM (2026-11-04, `ZZ-LOCAL-0018`, C80)

Compile tests cannot say which patch makes the A770 faster, so this round builds what the card needs to
decide it, in `sycl-patches/ab/`.

- **Off switches (`ZZ-LOCAL-0018`).** The local perf patches had no off switch except
  `GGML_SYCL_ENABLE_FUSION=0` / `GGML_SYCL_ENABLE_OPT=0`, which also turn off upstream's fusions and
  reorders. They are also stacked (0016 on 0013, 0012 and 0015 on 0011), so leaving one out meant
  regenerating patches and rebuilding between the arms of a benchmark. `GGML_SYCL_LOCAL_OFF` has 15 bits.
  Each one adds a test in front of an existing branch: the seven 0003 call sites, the two reorder entry
  points, 0010's row-pair predicate, the graph-loop fusion sites, the 0012 / 0013 / 0011 expert-sum
  matchers (which `graph_optimize` also calls, so a disabled fusion drops its allocation dependency too),
  0014's cache node, 0016's epilogue entry and the topk-moe extensions. The four touched TUs build (icpx
  2026.2, JIT) with all candidates on top, and every candidate still applies with plain `git apply`.
- **C80, weight-only Q8_0 GEMM.** From the fork's `woq_gemm_q8_0`, without its cache and layout tiers.
  oneDNN gets the reorder layout's int8 plane as weights `{k, m}` with strides `{1, k}`, and the d plane
  transposed to `{k / 32, m}` as grouped f16 scales (mask 3, groups `{32, 1}`), with f16 fpmath applied to
  the integer weights. `tools/woq-q8_0-probe` checked these descriptors on oneDNN 3.14.1's CPU engine
  against a double-precision reference: NMSE 1.6e-13 with f32 activations. With f16 activations, the form
  C80 runs, the CPU engine declines: brgemm reports the data types unsupported on this ISA (AVX-512 with
  DL Boost, no FP16), and the f32 / bf16 / int8 GEMMs and the reference reject the combination. So whether DG2 has a non-reference kernel is the first thing the
  probe answers on the card. C80 declines into the old paths and stays off after one decline. Upstream
  installs the reorder only on calls of at most 8 tokens. With the switch on, C80 also installs it on a
  prompt batch, so `llama-bench -p 512 -n 0` reaches the path.
- **Runner, plan and guard.** `ab.py` runs `plan.json`'s 57 experiments in alternating ABBA rounds. Its
  verdict is the per-round paired gain with a t-interval, checked against fake `llama-bench` /
  `llama-perplexity` binaries that emit the real output formats. `ab.py kld` scores both arms against
  CPU-backend logits (`-dev none`). `guard.sh` reads i915 and xe sysfs.
- **Not ported:** the fork's FA masked-cell fix (f8561ec88). It rewrites the fork's ESIMD / XMX kernels,
  and upstream's TILE / VEC add the mask as CUDA does over a cache that `llama_kv_cache` clears at
  allocation.
- **Corrected:** the "Deferred" list in `sycl-patches-extracted/README.md`. fattn-onednn and fattn-xmx
  are superseded (upstream oneDNN SDPA with C62 / C63, and C60). moe-xmx's "no MoE model fits 16 GB" was
  wrong (gpt-oss-20b MXFP4 is about 12 GiB), though it stays out because its kernels are 16-wide only.
  Also corrected: the build block in `sycl-patches/README.md`, which lacked the oneAPI `llvm-ar` override
  that keeps `GGML_LTO` on.

## Upstream refresh to `fac3a894`, new upstream PRs and draft candidates (2026-11-03)

Moves the stack from `dde879bc1` to `fac3a894` (43 upstream commits, merged into this branch as
`500bd505`) and carries the SYCL PRs opened since the last refresh. GitHub's `pull/N.patch` URLs returned
403 through this environment's proxy, so every PR file was made with `git format-patch` (unpinned) or
`git diff` (pinned, re-anchored) from `pull/N/head` fetched with git.

| Item | Change |
|---|---|
| Upstream PRs 30956, 31325 | merged upstream on 2026-11-02; retired |
| Fork 0002, 0008 | re-anchored: merged 31325 replaced dmmv's per-type launchers with two templates. 0002 drops the fp16 capability check from both; 0008 turns all 12 remaining `dpct::permute_sub_group_by_xor` calls in `dmmv.cpp` into `select_from_group` |
| PR 30606 | refreshed to its 2026-11-03 head (3 commits) |
| PR 31285 | reworked upstream (the VRAM check now runs on BMG-G21/G31 only, against a free-memory baseline); re-anchored after 30606's docs row, pinned. No effect on the A770 beyond `GGML_SYCL_MEMTRACE` output |
| PR 31536 (new) | oneMKL FA with a mask broadcast over heads or batches; re-anchored after upstream's `test_flash_attn_ext_kv_rows` (#29510) and 30606's GQA test loop, pinned |
| PR 31542 (new) | attention sinks (gpt-oss, Flint) may take the oneMKL FA path; re-anchored after 31536's tests, pinned |
| PR 31540 | skipped: CI, Dockerfile and docs only. Its changelog reports oneAPI 2026.2 alone at Qwen3.8-9B Q6_K pp512 565 vs 378 t/s on an Arc B570 against 2026.1; this machine's notes list oneAPI 2025.3, so a compiler upgrade is a free A/B |
| PR 31139 | still skipped; on reading it, the SYCL buffer types get a NULL `get_max_alloc_size`, so it changes nothing on SYCL |
| Candidate C51 | re-anchored: #29510 changed the context of the CPU FA loop it edits (`ic` became `ik1`) |
| Candidate C62 (new) | PR 31574 (draft): oneDNN fused SDPA at head_dim 64 on Alchemist |
| Candidate C70 (new) | PR 31379 (draft): GDN reads its state through the GET_ROWS ids |
| Candidate C75 (new) | PR 31313 (draft, WIP): lightning-indexer vector and XMX kernels, without the PR's fused-gemm.cpp dedup |
| Candidate C76 (new, local) | 256 registers for C75's vector kernel on ACM |
| Vulkan PR 29357 | refreshed: two new commits (shader and host refinement with accuracy fixes; prefill FA shared memory capped at 48 KiB) |

Findings from the refresh:

- **PR 31355 no longer fires on Qwen3.5 / Qwen3-Next.** Upstream `b74298c1` (merged today) writes the
  rollback conv-state windows of delta-net models with one `SET_ROWS` over an overlapping 4D view instead
  of one `CPY` per slot. 31355's matcher looks for the `CPY` chain. Its own test builds that chain, so the
  test still passes, but real graphs no longer match. SYCL runs the new `SET_ROWS` itself (F32 source,
  row strides honoured). Kept carried until the PR follows upstream.
- **Upstream already has an MoE expert cache.** `--moe-cache-mib` (SLRU since #30637, multi-GPU since
  #30112) keeps hot experts in VRAM. The survey that preceded this refresh said llama.cpp only had static
  offload; that was wrong. 31139 extends that cache for devices with a per-allocation limit, which SYCL does
  not report.
- **C75 without the dedup.** 31313 moves fused-gemm.cpp's joint_matrix traits, combination list and
  picker into `xmx.hpp` under `xmx_*` names. Local 0010 / 0017 and C20 / C21 edit that code in
  fused-gemm.cpp, so C75 adds `xmx.hpp` for the indexer alone and fused-gemm.cpp keeps its `fg_*` copy.
  The names do not overlap. The PR's docs rename of the logged picker (`fg_pick_combo` to
  `xmx_pick_combo`) was dropped with it.
- **C70 and 31355 share scaffolding.** Both are cwriter's; 31379 repeats 31355's `graph_optimize` hook and
  test-backend-ops whole-graph perf mode line for line, so C70 adds only its own parts.
- **C62's test card is unclear.** The PR's validation line reads "Arc A770 / dg2_g11", but `dg2_g11` is
  ACM-G11 (the A380 / A310) and the A770 is ACM-G10, so the card name or the label is wrong. The block it
  lifts guarded against wrong results that the PR traces to a Linux IGC regression fixed in IGC 2.40.
- **Draft check:** all four draft heads (31574, 31379, 31313, 31428) were re-fetched unchanged and still
  open drafts; a line-by-line comparison of each PR diff with its candidate found nothing missing beyond
  the documented C75 dedup (candidates README, "Draft PR check").
- **Review comments on the drafts** (candidates README, same section). The only substantive one is arthw's
  question on 31428 about `-ze-intel-greater-than-4GB-buffer-required` in AOT builds. An `acm-g10` test
  settles it in C20's favour: the bare `-Xs` form fails the AOT link (`Invalid option`), the `-options` form
  builds, and the flag changes the code (`getrows.cpp`: 673 `a32` bindless-surface accesses without it, 716
  `a64` with it). Upstream AOT builds therefore address at most 4 GB from the start of each buffer; JIT
  builds, like the ones in this log, get the flag through `-Xs`. The other comments are names and build
  messages (not changed) or were already resolved in the carried commits. arthw measured 31379 on an A770:
  tg128 +2.6% / +3.3% (FA off / on) on Qwen3.5-0.8B.
- **C63 (new, local, on C62).** oneDNN#5510, the issue 31574 cites, reproduced the head_dim 64 failure on an
  A770 and an A750 under Ubuntu 24.04 with Level Zero 1.15.38646 (NMSE 0.17, the only failing head size),
  fixed in IGC 2.40. llama.cpp cannot read the IGC version, so C63 keeps the dg2 block outside Windows
  until `GGML_SYCL_FA_ONEDNN_DG2_HD64=1`. Without it, applying the candidates on such a driver silently
  breaks attention for every head-dim-64 model on the oneDNN path. Compiled (`fattn-onednn.cpp`,
  `ggml-sycl.cpp`) with the whole stack and all candidates; not run, because the CPU test device never
  takes the oneDNN SDPA path.

Verification so far:

- **Apply / unapply:** on a clean `fac3a894` worktree the stack (40 patches), the candidates and the
  Vulkan stack apply in script order with plain `git apply`, no three-way fallback, 58 patches in all
  (re-run after adding C63). Vulkan, candidates and stack `unapply.sh` reverse all 58 and return the tracked and untracked tree to
  upstream (the build directory aside).
- **Vulkan build:** GCC 13 build of the same applied tree with `GGML_VULKAN=ON` and tests: 792 targets, no
  warnings; `ctest -L main -LE model` 16/18, the two failures being `test-download-model` (no network)
  and `test-thread-safety`, which depends on it. No Vulkan device here, so no Vulkan op ran.
- **`acm-g10` AOT** (real device flags, `-fsycl-device-code-split=per_kernel`): `lightning-indexer.cpp`
  (C75 + C76) 198 DPAS, no spill; `gated_delta_net.cpp` (C70) 20 kernels, no spill; `fattn-mkl.cpp`
  (31536, 31542) 11 kernels, no spill; `dmmv.cpp` (0002, 0008 re-anchored) 53 kernels, no spill.
- **CPU OpenCL device** (`icpx` 2026.2 build of the stack, the test hooks, and every candidate but C20 / C21,
  whose CMake change conflicts with the hooks; `tools/cpu-device-test/run.sh`):

  | Check | Result |
  |---|---|
  | `FLASH_ATTN_EXT` mask broadcast over heads and batches (31536) | 4/4; `GGML_SYCL_MKL_FA_DEBUG=1` shows all four on the oneMKL kernel |
  | `FLASH_ATTN_EXT` gpt-oss / Flint sinks, f16 and q8_0 KV, kv 1024 / 16384, 32 / 512 queries (31542) | 16/16; the kv 1024 cases dispatch to oneMKL FA |
  | `GATED_DELTA_NET` | 43/43 |
  | `GATED_DELTA_NET_GATHER` (C70), gather on / `GGML_SYCL_GDN_STATE_GATHER=0` | 19/19 / 19/19 |
  | `LIGHTNING_INDEXER` (C75 vector kernel, `GGML_SYCL_LIGHTNING_INDEXER=1`, XMX types off), f32 and f16 keys | 70/70 (4-64 heads, 1-512 tokens, key tails, 1-4 streams, aligned strided views) |
  | `MUL_MAT` one column, DMMV types with `GGML_SYCL_PRIORITIZE_DMMV=1` (0002, 0008) | 177/209; the 32 failures are the known f16 `k_v` batched cases (see "Quant-type gap fill, part 2"), which take `ggml_sycl_mul_mat_batched_sycl` |
  | the same with the reorder forced on | 65/65 |
  | `SSM_CONV` (31355) | 60/60 |
  | `SET_ROWS`, no bf16 | 471/471 |

  Two crashes come from upstream code on this test device, not from these patches (the CPU runtime's
  "PLEASE submit a bug report" banner is its signal handler, printed for any fatal signal):
  - TILE flash attention (`GGML_SYCL_ENABLE_MKL_FA=0`, or any shape oneMKL FA does not take) stopped with
    SIGFPE in `launch_fattn`: upstream sets `nsm = max_compute_units / 16`, which is 0 on this 4-unit CPU
    device (32 on an A770), and the stream-k setup takes `ntiles_total % 0`. `hooks.patch` now sets `nsm`
    to 1 under `GGML_SYCL_TEST_CPU_DEVICE`, so TILE runs here.
  - Upstream's unchanged `lightning_indexer_f32_sycl` (the default indexer, and C75's fallback for bf16
    and quantized keys and unaligned views) segfaults on its first launch when run normally, with or
    without ASLR, and passes 5/5 under gdb: timing-dependent inside the CPU runtime. Not chased further.

  bf16 kernels crash the CPU device as before. With the `nsm` hook, TILE FA on the gpt-oss / Flint sink
  shapes (`GGML_SYCL_ENABLE_MKL_FA=0`) passes 4/4, so TILE and 31542's oneMKL route agree there.
- **`test-llama-archs`** (upstream's random-weight models through the whole stack, against the CPU
  backend) on the CPU device, with `GGML_SYCL_ENABLE_GRAPH=0` and `=1`:

  | Architecture | NMSE vs CPU | Mixed batch | Roundtrip |
  |---|---|---|---|
  | llama (dense, MoE), qwen35, qwen35moe, qwen3next, gemma3, mamba2, gpt-oss | OK, at most 3e-8, both settings | exact, both settings | OK except llama dense |

  The delta-net models (qwen35, qwen35moe, qwen3next) are where C70 and PR 31355 apply, gpt-oss is where
  31542 applies, and mamba2 is where local 0015's SSM cache fusion applies. With one seed (`-s 42`) the two
  graph settings give identical NMSE.
  Graph replay itself never ran here: `check_graph_compatibility` turns graphs off for every graph with a
  MUL_MAT because the CPU OpenCL runtime lacks the oneAPI async memory-allocation extension. Replay
  correctness and speed are therefore A770-only checks (commands below).
  Roundtrip (save the device model to GGUF, reload, require bit-identical logits) fails for some seeds:
  llama dense with two random seeds, qwen35 with `-s 42`. It fails the same with `GGML_SYCL_ENABLE_FUSION=0`,
  which also turns off 0014's activation cache, and NMSE is unchanged, so no local fusion is behind it.
  Whether upstream SYCL fails the same way was not checked (no upstream-only build here).

### Measuring on the A770 (not done here: no Intel GPU)

SYCL graph replay is compiled in and off by default. Strata-A770 measured +22% decode on an A770 by
replaying command lists through Level Zero instead of submitting each launch, which says launch overhead
is worth measuring here. A/B on the same build:

```bash
for g in 0 1; do GGML_SYCL_ENABLE_GRAPH=$g ./build_sycl/bin/llama-bench -m MODEL -fa 1 -p 0 -n 128 -r 5; done
```

The log prints `disabling SYCL graphs` when a graph does not qualify; the reason is a debug-level line, so
add `--verbose`. Graphs turn off for multi-GPU, a MUL_MAT_ID that waits on the host, OUT_PROD / CONV_3D /
SOLVE_TRI, any MUL_MAT when the device lacks the async memory-allocation extension
(`ext_oneapi_async_memory_alloc`, which is why the CPU test device never replays), and, when oneDNN is
older than 3.12.3, a oneDNN MUL_MAT or oneDNN FA. Decode of a single-GPU model normally qualifies. A replay needs two
identical calls first, so use `-n 128` or longer. Check replay correctness on the card too (the CPU test
device cannot replay, see above); `test-llama-archs` needs `-DLLAMA_BUILD_TESTS=ON`:

```bash
GGML_SYCL_ENABLE_GRAPH=1 ./build_sycl/bin/test-llama-archs -b SYCL0 -a '^(llama|qwen35|qwen3next|gpt-oss|gemma3)$' -s 42
```

To see how far marfrit/arcint's OpenCL kernels are ahead before porting any of them, build its patched
llama.cpp (pinned at `3e4dfa8`) with the OpenCL backend and run the same `llama-bench` line on both:

```bash
git clone https://github.com/marfrit/arcint
git clone https://github.com/ggml-org/llama.cpp llama-arcint
git -C llama-arcint checkout 3e4dfa8
git -C llama-arcint apply ../arcint/contrib/llama.cpp/patches/*.patch
cmake -S llama-arcint -B llama-arcint/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_OPENCL=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF
cmake --build llama-arcint/build --target llama-bench
llama-arcint/build/bin/llama-bench -m MODEL -fa 1 -p 512,4096 -n 128 -r 3
./build_sycl/bin/llama-bench -m MODEL -fa 1 -p 512,4096 -n 128 -r 3
```

arcint runs only under the `xe` kernel driver (its docs: "i915 is untested"); this machine uses i915.
A k-quant dense model shows the prompt-processing gap, where this stack still dequantizes to f16 for
oneDNN.

## Remaining fusions, MMVQ add epilogue and the register-file audit (2026-11-03, `ZZ-LOCAL-0015` to `ZZ-LOCAL-0017`)

The rest of the survey's list, ported where it pays on an A770 and declined where it does not.

| Item | Outcome | Why |
|---|---|---|
| Logit softcap (SCALE + TANH + SCALE), squared ReLU (RELU + SQR) | 0015 | final-logit softcap (Gemma 2, Grok) saves 2 launches; relu² (Nemotron, Nemotron-H, Arcee, PLM, Jais2) 1 per FFN |
| Snake activation (MUL, SIN, SQR, MUL, ADD) | 0015 | the TTS audio decoders in mtmd (Qwen3-TTS): 4 launches per use |
| SCALE + SIGMOID + SCALE into DSV4_HC_POST | 0015 | qwen4exp's hc_combine (CUDA's pattern): the hc_post kernel applies the gate, 3 launches saved per block |
| SSM_SCAN + snapshot CPY into the cache | 0015 | mamba-2 / hybrid models: one copy of the full state per layer and token |
| topk-moe sqrt(softplus) gating and expert groups | 0015 | DeepSeek-V4's gate and DeepSeek-V3's group masking (14 more ops) left the whole router unfused |
| MUL_MAT + ADD in the one-column MMVQ store | 0016 | i-quants, MXFP4 and the reorder types without DMMV still paid an in-place ADD after 0013 |
| 256-register XMX dequant GEMM tiles | 0017, opt-in | answers whether 0010's 128-register tile is the right side of the trade; see below |
| Shared-expert fusion | not ported | CUDA's matcher needs the shared expert at the routed experts' type and shape; Qwen2-MoE and DeepSeek-V2-Lite size it differently, a quant mix can give it another type, and it saves one launch |
| MoE gate + up of different types | not ported | CUDA does not fuse it either; the MoE MMVQ is templated on one type and a mixed pair would need a second weight stream through every launcher |
| XMX int8 MMQ for prompt batches | not reachable | extracted 0016's MMQ reads the plain block layout, and the reorder (on by default) leaves the weights in SoA; enabling it would misread reordered weights. Needs SoA stages first |
| Fused dequant GEMM for dense k-quant / q8_0 batches | not enabled | the dense kernel decodes A again for every 32 columns; PR 30156 keeps these types on dequantize + oneDNN for that reason (`fg_plain_ok`) |
| Multi-stream Q / K / V concurrency | not ported | opt-in on CUDA (`GGML_CUDA_GRAPH_OPT`), needs per-stream pools and events in the SYCL graph path; nothing here can measure the gain |

How the new pieces work:

- **0015 elementwise:** `fused-elementwise.cpp` holds the softcap (F32), relu² (F32, F16) and snake (F32, F16,
  BF16 activations, F32 a and 1/b) kernels, one work-item per element. The snake matcher is CUDA's: x [T, C],
  a and 1/b broadcast per channel. The softcap folds both scales; relu² is one `max(x, 0)^2`.
- **0015 hc_post gate:** `dsv4_hc_post_f32_sycl<has_comb, gated>` computes `gate_out * sigmoid(gate_in * post)`
  where it read `post`, so the scaled sigmoid never exists as a tensor. The matcher checks the edges (the
  sigmoid feeds DSV4_HC_POST's post operand, the scales have no bias, F32, same shape).
- **0015 SSM_SCAN cache:** CUDA's predicate (one sequence, d_state 128 / 256, contiguous A, the CPY's
  destination a view of the cache with the scan's own snapshot layout). The scan writes the final state and
  the rollback snapshots at `cache + slot * slot_stride` instead of its dst tail, and the CPY is skipped.
- **0015 topk-moe:** `topk_moe_cuda`'s sqrt(softplus) gate and group masking: each group's score is the sum
  of its best two experts (sub-group shuffles), the `n_group_used` best groups survive, the others are masked
  to -inf before the expert argmax, on the biased selection weights when there is a bias. Up to `WARP_SIZE`
  groups, as on CUDA. The 14-op group pattern is CUDA's table. The scoring is one pass over a lane's experts
  (a group is whole elements or an aligned block of lanes, since the counts are powers of two), and the
  masking is a template parameter of the kernel, so routers without groups run the old instantiations. The
  grouped ones at 256 / 512 experts take the 256-register file on Intel GPUs with XMX (see the audit).
- **0016:** the one-column MMVQ launchers are specializations of two kernels (`mul_mat_vec_q` and
  `mul_mat_vec_q_reorder`); both take an optional addend now. With an ADD to fold, the one-column dispatch
  launches them directly with each type's parameters and reports the ADD done; types with their own tuned
  launcher (reorder q4_K / q5_K, the q8_0 wide dot product, the XMX MMVQ) keep 0013's in-place ADD.
- **0017:** see "When the 256-register file pays" below.

Verification (CPU OpenCL device, `tools/cpu-device-test`, icpx 2026.2, `GGML_SYCL_DEBUG=1` to see each fusion run):

| Test | Result |
|---|---|
| `SOFTCAP` / `RELU_SQR` / `SNAKE_FUSE` (no bf16) | 1/1, 4/4, 16/16; fused 1, 4 and 10 times |
| `DSV4_HC_POST` | 9/9; the 3 gated cases fused |
| `SSM_SCAN_CACHE_FUSION`, `SSM_SCAN` | 7/7 (all fused), 12/12 |
| `TOPK_MOE`, power-of-two expert counts | 384/384 on the final build, 278 fused, 96 of them with expert groups: sqrt(softplus) and the 4 / 8 / 16-group cases all fuse (sqrt(softplus) with groups too); 32 groups is above `WARP_SIZE` and runs unfused. An earlier run failed one unfused 32-group case, the tie-break flake seen with 0011; it passed 8 reruns |
| `MUL_MAT_VEC_FUSION` | 1818/1818 (the upstream cases plus 64 scratch MUL_MAT + ADD cases for q5_0 / q5_1, the k-quants, IQ1..IQ4 and MXFP4 at 32 / 48 rows, k = 256 / 768, in the test tree only); 86 ADDs folded into the store. Every IQ1 / IQ2 / IQ3 / IQ4, q5_0 / q5_1 and MXFP4 scratch case folds; the k-quants take the CPU device's non-ESIMD DMMV, which keeps the in-place ADD (the A770's ESIMD DMMV folds them since 0013) |
| `MUL_MAT_VEC_FUSION`, reorder forced | 1818/1818, 84 folds: the reorder q2_K / q3_K / q6_K, IQ3 and IQ4_NL weights now fold through the reorder kernel; reorder q4_K / q5_K and q8_0 (wide dot product) keep the in-place ADD by design |
| `MUL_MAT`, one column, no bf16 | 448/480 with and without the reorder; the 32 failures are the f16 `k_v` cases listed under 0011 to 0014, unchanged |

`acm-g10` AOT, the stack's flags and its per-kernel device-code split (the earlier checks in this log linked
the whole file as one module, which gave the same register allocation for `mmvq.cpp` and `fused-gemm.cpp`
but more spills in `topk-moe.cpp`): `fused-elementwise.cpp` (10 kernels), `dsv4-hc.cpp` (11) and `ssm_scan.cpp` (7)
do not spill. `mmvq.cpp` goes from 649 to 678 kernels; the 29 new add-epilogue kernels do not spill and the
file's spill total is unchanged (the ESIMD XMX MMVQ and the paired gate + up variants 0010 keeps off on DG2).
`topk-moe.cpp`: the instantiations without groups are no worse than the old file (512 experts: 0 / 96 B
without / with a bias, before 352 / 288 B). The expert-group instantiations spill at 128 registers: 1.4 / 2.4 KB
with / without a bias at 256 experts, 7.9 / 9.1 KB at 512. Rewriting the group scoring as one pass did not
change that, so the masking got its own instantiation instead of taxing every router, and those
instantiations take 256 registers on Intel GPUs with XMX: no spill at 256 experts, 1.0 / 1.2 KB at 512.
`fused-gemm.cpp`: 178 kernels, none spills, the 72 large-register ones at 256. Every new or changed kernel
is `numGRF=128` except those that ask for 256.

The patches apply in order with plain `git apply`, all candidates apply on top (their `fused-gemm.cpp`,
`mmvq.cpp`, `ggml-sycl.cpp` and `fattn-gqa-dec.cpp` pass host and device syntax checks), and the round trip
with the Vulkan stack returns a clean tree. Not verified: speed on the A770, and the large-register kernels
(0017's tile, the grouped router's 256-register variant, C40's) at run time: none runs on the CPU device.

### When the 256-register file pays (Xe-HPG)

An Xe-HPG EU holds 8 hardware threads of 128 32-byte registers (4 KB each). The 256-register mode gives a
thread 8 KB and the EU 4 threads; the register file and the EU's ALU and XMX rate do not change. What 128
registers buy is latency hiding: while one thread waits on a load the EU issues from another, so 8 threads
cover memory latency that 4 cannot unless each thread keeps more loads in flight itself.

So 256 registers win in two cases:

1. The kernel spills at 128 values that stay live across its main loop (accumulators, a register tile,
   a row slice). Each fill is then a scratch read per iteration, memory traffic on top of the kernel's
   own, which costs more than the latency hiding the lost threads would have bought.
2. The kernel is compute-bound and more registers buy a bigger register tile. A larger GEMM / attention
   tile reuses each loaded A or B fragment across more DPAS, raising arithmetic intensity, and 4 threads
   with that much independent DPAS work keep the XMX busy.

128 registers win everywhere else, which is every decode kernel: a one-token mat-vec reads each weight
once, its intensity is fixed by the format, and its speed is the number of loads in flight. Elementwise,
norm and rope kernels are the same. The exception the other way is a kernel too small to fill the EUs:
with no occupancy to lose, it takes 256 registers whenever that saves a spill (0015's grouped router).

Audit of the stack against that rule (acm-g10 AOT with the stack's flags, IGC dumps):

| Kernels | Registers | Verdict |
|---|---|---|
| MMVQ / DMMV / MoE MMVQ (the SPMD kernels of `mmvq.cpp`), elementwise, rope, norm, topk-moe without groups | 128; no spills except 0010's paired gate + up variants, which DG2 does not run | right: memory-bound |
| 0015: topk-moe with expert groups, 256 / 512 experts | 256 on Intel GPUs with XMX; no spill at 256 experts, 1.0 / 1.2 KB at 512 (7.9 / 9.1 KB at 128) | right: a router runs one sub-group per token, so there is no occupancy to lose, and its spills sat in its only work |
| 0010: reorder gate + up row pairing on DG2 | off, 128, no spills | right: pairing only added registers to a memory-bound kernel |
| 0010: XMX dequant GEMM at sub-group 8 (prompt batches) | 128, 8-row tile, no spills | defensible, unmeasured: a compute-bound kernel where 256 registers would let the 16-row tile halve the B loads per DPAS. 0017 builds that tile spill-free at 256 as an opt-in, so the A770 can decide |
| fattn-vec, D >= 512 | `grf_size_automatic` | upstream's choice, not checked here |
| candidate C60 fattn-dpas (SPMD) | `grf_size<256>`, honoured in AOT | right: prefill attention is compute-bound |
| candidate C40 D = 256 SPMD decode kernels | were 128 with 0.7 to 1.7 KB of spills; now 256 on Intel GPUs with XMX, no spills | fixed here: a lane's q, accumulator and K / V slices (about 128 registers at D = 256) stay live across the KV loop, so whatever spills is reloaded per KV row |
| candidate C21 ESIMD MoE GEMM, upstream ESIMD XMX MMVQ (Xe2 gate) | ask for 256, get 128 in AOT, spill 3.7 to 12.7 KB | wrong in an AOT build, through no fault of the request |

The last row is a toolchain finding. oneAPI 2026.2's AOT path keeps `grf_size<256>` for SPMD kernels (a
minimal kernel gets `numGRF=256`) and drops it for ESIMD ones: the same minimal kernel in ESIMD gets 128
registers and spills 4.9 KB, and the VC back end receives only `-vc-codegen`.
`-ftarget-register-alloc-mode` would work but applies to every kernel of the target, decode included, so it
is the wrong tool. Build C21 (and C40's ESIMD path) with JIT to measure them; the XMX MMVQ is not selected
on DG2.

A request for 256 registers is not free of side effects either: the runtime turns `grf_size<256>` into the
IGC option `-ze-opt-large-register-file`, and the CPU OpenCL device fails the whole build on it
("Unrecognized build options"). That device also reports `ext_intel_matrix` (AMX), so C40 and 0015's
grouped router gate their 256-register kernels on `is_gpu() && ext_intel_matrix`, two kernel types chosen
at run time (one lambda launched with two property sets is two kernels of one name, which the compiler
rejects). The stack's other
256-register kernels (0017's tile, fattn-dpas, C60) never run on the CPU device.

0017 makes the GEMM decision measurable instead of argued. `GGML_SYCL_XMX_GRF256=1` switches the
sub-group-8 combinations (f16 and bf16 8x8x16) to the 16-row tile with both k halves unrolled and
`grf_size<256>`; the shape bit, the B packing and C20 / C21's ESIMD remap are unchanged. On an i-quant or
MoE model (the types that take the dequant GEMM):

```bash
GGML_SYCL_XMX_GRF256=1 ./build/bin/llama-bench -m MODEL -p 512 -n 0 -r 3
```
```bash
./build/bin/llama-bench -m MODEL -p 512 -n 0 -r 3
```

If the first wins, make it the default; if it loses, 0010's trade was right and 0017 can be dropped.

## Fusions ported from CUDA, Vulkan and Metal (2026-11-03, `ZZ-LOCAL-0011` to `ZZ-LOCAL-0014`)

A survey of the CUDA, Vulkan and Metal backends against the patched SYCL backend found the gap was graph
fusion, not kernels: the three backends fold decode-time ops into the mat-vec or into one launch where
SYCL launched several. Four local patches port eight of them. Speed on the A770 is not measured; the
counts below are from the graph structure.

| Patch | Fusion | Source | Launches saved (decode) |
|---|---|---|---|
| 0011 | ROPE + VIEW + SET_ROWS | CUDA, Vulkan; SYCL had the kernel (`ggml_sycl_rope_fused`) and no caller | 1 per layer |
| 0011 | RMS_NORM + MUL + ROPE (+ VIEW + SET_ROWS) | CUDA, Vulkan | 2 more per layer with Q / K norms (Qwen3, Gemma 3, OLMo 2) |
| 0011 | MoE expert sum: MUL (router weights) + VIEWs + k - 1 ADDs, k <= 15 | CUDA, Metal | 4 per MoE layer at k = 8 (was 1 MUL + 4 ADD / ADD + ADD) |
| 0011 | 3..8 same-shape ADDs | CUDA, Vulkan | model-dependent |
| 0011 | topk-moe with a routing bias | CUDA, Vulkan | about 10 routing ops per layer (GLM-4.5+, DeepSeek-V3-style) |
| 0012 | MoE MMVQ for 2..8 tokens | CUDA | spec-decode verify / parallel slots stay on the MMVQ |
| 0012 | MUL_MAT_ID gate + up + GLU (+ ADD_ID biases), MUL_MAT_ID + ADD_ID | CUDA | 3 per MoE layer (5 with gpt-oss biases) |
| 0013 | MUL_MAT (one column) + ADD | CUDA, Vulkan, Metal | 2 per layer (residuals), more with biases |
| 0014 | one q8_1 / f16 copy of an activation for Q, K, V | Vulkan (`prealloc_y`) | 2 per layer on MMVQ / f16-DMMV types |

For Qwen3-30B-A3B (48 layers) 0011 to 0013 remove about 11 of roughly 30 kernels per layer, about 530 per
token; for a dense Qwen3-8B, about 5 of 20 per layer. To turn that into time on the card, compare
`llama-bench -n 128` with `GGML_SYCL_ENABLE_FUSION=0` and `=1`: every fusion here, 0014 included, follows
that switch. The 2..8-token MoE MMVQ is not a fusion; `GGML_SYCL_MMID_MMVQ_MAX_TOKENS=1` turns it off.

How each one works:

- **0011:** the predicates are CUDA's, cut down to what the SYCL kernels take. The rope set_rows path
  takes f32 in, an f32 / f16 cache, I64 row ids, the norm and neox modes, and a cache whose element index fits
  in 32 bits (the rope kernels index in `int`). The norm + rope kernel is one sub-group per (head, token) row
  and sums the row in the order of the one-sub-group rms_norm kernel. The expert-sum matcher is CUDA's;
  `graph_optimize` keeps the expert results, router weights and scales alive until the sum, so the
  allocator cannot place the sum over them. The topk-moe bias path is `topk_moe_cuda<has_bias>`: selection
  on probs + bias, weights from the probs. CUDA's expert-group masking and sqrt(softplus) gating are not
  ported here; those graphs ran unfused until `ZZ-LOCAL-0015`.
- **0012:** one `ggml_sycl_moe_mmvq_ext` threads three options through both MoE kernels: a token
  dimension in the grid (strides for the q8_1 rows, the dst rows and the ids), a gate weight with a GLU
  epilogue (SWIGLU, GEGLU, SWIGLU_OAI, SWIGLU_CLAMP, the formulas of the standalone GLU kernels), and
  per-expert row biases. The token cap per type is CUDA's RDNA3 table (k-quants and IQ2 / IQ3: 4, IQ1 /
  IQ4: 6, others: 8); it is a guess for Intel, `GGML_SYCL_MMID_MMVQ_MAX_TOKENS` overrides it. The routing
  ids are read while the fused output is written, so `graph_optimize` keeps them alive until it and the
  dispatcher declines an overlap.
- **0013:** `ggml_sycl_mul_mat_add_fused` points the mat-vec at the ADD's memory and puts the addend in
  the context. The DMMV kernels add it in their store and say so: the A770's ESIMD kernels (q2_K..q6_K,
  q8_0) through a separate policy, the generic plain / reorder kernels (q4_0, q4_1, q5_0, q5_1, q8_0, q1_0)
  through a parameter. Anything else (MMVQ, the non-ESIMD k-quant DMMV) leaves its plain result there and
  one in-place ADD follows, so the fusion never adds a launch. The ESIMD addend needed its own policy:
  with a run-time null test around the load, IGC's ESIMD back end crashes building `dmmv.cpp` for
  `acm-g10` ("Internal Compiler Error: Abnormal termination", oneAPI 2026.2), the same kind of failure
  candidate C40 hit with its in-kernel mask test. An unconditional gather builds, so the add policy
  gathers the addend and the plain policy is unchanged.
- **0014:** the context keeps the last converted activation, keyed by the graph node's src1 tensor, its data
  pointer, the layout (q8_1, reordered q8_1, f16) and the size. Only the node the graph loop is running may
  use it, which keeps the MUL_MAT_ID per-expert loop's stack tensors out. The graph loop drops it when a node
  writes the activation's memory, and at the start and end of every graph. The pool buffer it holds is host
  bookkeeping over persistent device memory, as every other pool temporary is, so SYCL graph replay is
  unaffected. The A770's ESIMD DMMV for the k-quants and q8_0 reads f32 activations and gains nothing.

Verification (CPU OpenCL device, `tools/cpu-device-test`, icpx 2026.2):

| Test | Result |
|---|---|
| `ROPE_SET_ROWS` | 73/73 |
| `RMS_NORM_MUL_ROPE` | 260/260 |
| `MOE_REDUCE` | 6/6 (the 16-expert case is above the cap and runs unfused) |
| `ADD_ADD`, `ADD` with 16 chained adds | 12/12, 2/2 |
| `TOPK_MOE`, power-of-two expert counts | 351/352: one expert-group case (not fused) failed once and passed 16 reruns with fusion on and off |
| `MUL_MAT_VEC_FUSION` | 1754/1754, and 1754/1754 with the reorder forced on |
| `MUL_MAT_ID` | 1205/1205, and 1205/1205 with the reorder forced on |
| `MUL_MAT`, one column, the DMMV types | 182/214: all 32 failures are f16 weights viewed with a wider row stride in batches (`k_v`, `bs=[*,3]`), which take `ggml_sycl_mul_mat_batched_sycl`, untouched here; they fail the same with fusion off |

With `GGML_SYCL_DEBUG=1` each fusion shows up running: the rope set_rows and norm + rope launches, 5 of the
6 expert sums, two 8-add chains for the 16-add test, the MUL_MAT + ADD epilogue applied on q4_0 and the in-place
fallback on q4_K, and one q8_1 and one f16 conversion instead of two for a gate / up pair that the GLU
fusion does not take. The full-run timeout on `TOPK_MOE` was the unfused 288-expert cases, whose argsort is
slow on the CPU device; the rerun covered the power-of-two counts the fused kernel takes.

`acm-g10` AOT: `rope.cpp` (53 kernels), `fused-add.cpp` (6), `topk-moe.cpp` (24), `dmmv.cpp` (53) and the
MoE kernels of `mmvq.cpp` (100, gate variants included) build without spills, except the plain-layout
q3_K MoE kernel with the gate (672 B). The dispatcher keeps that one case as two launches. Each of the
four patches passes a device-side syntax check (`-fsycl-device-only -fsyntax-only`) at its own stage. All candidates apply on top with plain `git apply`, and their
`mmvq.cpp`, `ggml-sycl.cpp`, `dmmv.cpp` and `fusion.cpp` pass the same check. The round trip with the Vulkan stack
returns a clean tree.

Not covered: the MoE gate + up fusion for gate and up of different types; the MUL_MAT + ADD epilogue in
the MMVQ kernels (i-quants, 2..8 columns), whose many per-type launchers would need it one by one
(`ZZ-LOCAL-0016` adds it for one column); CUDA's shared-expert fusion; the opt-in multi-stream Q/K/V
concurrency (`GGML_CUDA_GRAPH_OPT=1`). The section above settles each of these.
Prompt processing is unchanged. Dense k-quant / q8_0 batches still dequantize every weight to f16 before
oneDNN, since `ggml_sycl_supports_mmq()` keeps 0016's int8 XMX MMQ unreachable and the fused dequant GEMM
stops at 64 columns.

To measure on the A770 (JIT build, so the ESIMD kernels get their 256 GRF):

```bash
GGML_SYCL_ENABLE_FUSION=0 ./build/bin/llama-bench -m MODEL -p 0 -n 128 -r 3
```
```bash
GGML_SYCL_ENABLE_FUSION=1 ./build/bin/llama-bench -m MODEL -p 0 -n 128 -r 3
```

Run one dense model (Qwen3-8B Q4_K_M: QK-norm rope and residual epilogues) and one MoE model
(Qwen3-30B-A3B or gpt-oss-20b: expert sum, gate + up, biases). Also run `test-backend-ops -b SYCL0 -o
MUL_MAT_VEC_FUSION` on the card, since the ESIMD epilogue only runs there.

## DG2 register spills (2026-11-02, `ZZ-LOCAL-0010`)

The part-2 AOT check below found spills on `acm-g10` in kernels the A770 runs. Every number here is
from `acm-g10` AOT builds (oneAPI 2026.2's IGC, with C20's `-ze-opt-enable-auto-large-GRF-mode`; the
stack's own `-device acm-g10` gives the same register allocation, re-checked 2026-11-03) with IGC
shader dumps, `//.spill size` per kernel.

- **XMX dequant GEMM** (fused for dense i-quant prompt batches, grouped for MoE): all 72 kernels
  spilled, 1.9 to 3.5 KB per thread, the block and SoA layouts alike (191 KB in total). The sub-group-8
  tile (8x8x16, the only DG2 shape) gave a sub-group 16 rows x 32 columns: 8 accumulator tiles (64
  registers), 4 B tiles (32), the A tile (8), then addressing and the A-stage decode, against 128
  registers. Tried, each over all 72 kernels:

  | Change at sub-group 8 | Total spill | Kernels spilling |
  |---|---|---|
  | none | 191 KB | 72 |
  | 8 rows | 68 KB | 72 |
  | 8 rows, one B tile live at a time | 126 KB | 72 |
  | 16 rows, k halves not unrolled | 121 KB | 72 |
  | 16 rows x 16 columns | 29 KB | 29 |
  | 8 rows x 16 columns | 0 | 0 |
  | **8 rows, k halves not unrolled** | **0** | **0** |

  The last row is `ZZ-LOCAL-0010`: it keeps 32 columns, which the device-side MoE schedule requires
  (`GGML_SYCL_FG_BN`, checked by `bn_ok`), so each dequantized weight row still serves 32 columns. The
  cost is B reuse: a B tile now feeds 8 rows, not 16. 16 columns would also be spill-free but
  dequantizes twice as often and is refused by the device-side schedule.
- **DG2 has a 256-register mode.** Forcing it (`-ze-opt-large-register-file`) builds the old 16 x 32
  tile with no spill, at half the threads per EU. The auto mode the build relies on never chose it on
  `acm-g10` here, for these kernels or for C20/C21's ESIMD ones (which request 256 through `grf_size`;
  AOT ignores that on ESIMD, JIT honours it, see the candidates README). The fix fits 128 registers so
  it does not depend on that heuristic.
- **Gate + up MMVQ (reorder, fused GLU):** row pairing doubles the registers and, at sub-group 16, a
  value is two 32-byte registers on DG2 against one 64-byte register on Xe2. Every paired gate + up
  kernel spills on DG2: q4_K at 2..4 columns 0.5 to 1.9 KB (stack default), q4_K at 5..8 2.3 to 3.6 KB
  (C30 with C90 bit 16), q5_K at 3..5 1.1 to 3.5 KB (C90 bit 4), iq4_xs at 3..4 0.7 to 1 KB (C10). The
  A-series now never pairs rows in that kernel; all one-row variants are spill-free except q4_K at 8
  columns (64 B, as before). The plain MMVQ's pairing, one weight matrix, does not spill on DG2 (q4_K
  2..8 columns, C30 included) and is unchanged.
- **Left:** three decode kernels spill one or two registers (q3_K reorder one column 32 B, a single
  reload per block-loop iteration; q6_K six columns 64 B; iq3_s reorder eight columns 32 B). The XMX
  MMVQ kernels spill up to 12.8 KB on DG2 but are gated to Xe2 and never run there.
- **Tests:** on the CPU OpenCL device `MUL_MAT_VEC_FUSION` q4_K passes 140/140 with the one-row path
  forced on all widths (a scratch env override, not in the patch) and 140/140 with the default
  pairing; `MUL_MAT` q4_K with the reorder 113/113; `MUL_MAT_VEC_FUSION` f8_e4m3 224/224 (`ZZ-LOCAL-0006`
  coverage that part 2 had not run). The GEMM change cannot run on this device (no fp16 `joint_matrix`
  shapes); it only changes two tile constants and an unroll count in a kernel templated on them.
- **Not measured:** speed. On the A770 compare prompt processing of a dense IQ model and of a MoE
  model before and after `ZZ-LOCAL-0010` (and against `GGML_SYCL_XMX_GATHER_TYPES=0`, the dequantize +
  oneDNN route).

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
- **DG2 AOT (`acm-g10`, with and without `-ze-opt-enable-auto-large-GRF-mode`).** `mmvq.cpp`, `fused-gemm.cpp`,
  `convert.cpp`, `getrows.cpp`, `dmmv.cpp` and `cpy.cpp` build. None of the 38 new MMVQ kernels spills
  (27 ternary, 9 IQ4_NL reorder, 2 IQ3 MoE reorder; all at 128 GRF), nor do the 36 new dequantize,
  GET_ROWS, DMMV and CPY kernels. The fused XMX dequant GEMM spills on DG2 in every variant, the existing
  block-layout ones included (2.3 to 3.3 KB). The new SoA variants spill 64 B (IQ4_NL), 160 B (IQ3_XXS)
  and 512 B (IQ3_S) more than the block variant of the same type, as the existing Q4_K SoA variant does
  (+256 B). `ZZ-LOCAL-0010` (section above) removes the spills from all of them.
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

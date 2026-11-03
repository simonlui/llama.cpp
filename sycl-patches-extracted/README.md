# SYCL patches extracted from llama.cpp-sycl-terrible

Patches isolated from the experimental `llama.cpp-sycl-terrible` research tree
and reshaped to fit mainline SYCL backend conventions (no `unified-cache`,
no custom dispatcher, no TLSF allocator, no tiered/eviction runtime).

Sibling to `../sycl-patches/` - the active extracted files use slots `0015` and
`0016`. Retired slots are not reused: 0013 depended on draft PR 29367's
integer-tag dispatch, while the former RMS_NORM fusion patches and the old
SLM-cached 0014 extraction are upstream or dropped.

Only small, kernel-local, mainline-shaped changes live here. The bulk of the
research tree (memory tiering, autotuning dispatch, ESIMD, profiling) is **not**
extractable as idiomatic backend code and is intentionally excluded — see
"What was deliberately left behind".

Local work on top of the stack goes in the separate `ZZ-LOCAL-*` group in this
directory: repairs the current upstream tree needs, and gap fills for quant types
the carried PRs leave out. They are not folded into the numbered extracted
history and remain independently reversible. Fifteen are active, `ZZ-LOCAL-0003` to
`ZZ-LOCAL-0017` (see the patch index); 0011 to 0016 port graph fusions from the CUDA, Vulkan and Metal
backends, and 0017 adds an opt-in 256-register GEMM tile for DG2. They apply in number order, and every candidate in
`../sycl-patches-candidates/` applies on top of them with plain `git apply`.

## Patch index

| # | File | What it does | Type |
|---|------|--------------|------|
| 0015 | `0015-sycl-softmax-reduce-over-group.patch` | Replaces the block max/sum barrier+SLM reductions in `softmax.cpp` with `sycl::reduce_over_group`. Reduction-only; **keeps upstream exp semantics** (`sycl::native::exp` + the `-80.0f` clamp). | cleanup |
| 0016 | `0016-sycl-mmq-xmx.patch` | Experimental XMX (Intel Xe Matrix eXtensions / `joint_matrix`) Int8 MMQ path, gated behind the `GGML_SYCL_MMQ_XMX` CMake option (default OFF). Reshaped from the fork's `mmq_xmx.*`. | perf (build-flag gated) |
| ZZ-LOCAL-0003 | `ZZ-LOCAL-0003-sycl-mmvq-multi-col-iq.patch` | 2..8-column MMVQ for IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ4_NL and non-reordered IQ3_XXS, the only quantized types that reread their weights per column in a speculative-decoding verify or small batch. Generic `mul_mat_vec_q_ncols` with each type's one-column qi / vdr / vec_dot. | perf (gap fill) |
| ZZ-LOCAL-0004 | `ZZ-LOCAL-0004-sycl-moe-reorder-q4_0-q2_k-q3_k.patch` | Reorders Q4_0, Q2_K and Q3_K MoE expert weights per expert slice (the 2D reorder layout, one shared tail writer per type) and runs them through the generic MoE reorder MMVQ. Before, only Q8_0, Q4_K, Q5_K, Q6_K and MXFP4 experts were reordered. IQ3 is left out here; `ZZ-LOCAL-0008` adds it with the grouped GEMM's SoA stage it needs. | perf (gap fill) |
| ZZ-LOCAL-0005 | `ZZ-LOCAL-0005-sycl-ternary-types.patch` | SYCL kernels for the ternary weight types TQ1_0, TQ2_0 and PTQ1_0 (PTQ1_0 from carried PR 29672), which ran on the CPU before. Dequantize to f16 / f32 (prompt batches go through the GEMM), GET_ROWS, MMVQ for 1..8 columns, and the fused MoE MMVQ. The dot product expands each 32-weight group to signed bytes for `dp4a` with the CPU codec's base-3 digit extraction (multiply by 3^n mod 256, then `(q * 3) >> 8`), on packed bytes. Fork patch 0017's central gate now admits these types only as the weight operand of MUL_MAT, MUL_MAT_ID and GET_ROWS; SET_ROWS, CPY and every other op still fall back to the CPU. | feature (gap fill) |
| ZZ-LOCAL-0006 | `ZZ-LOCAL-0006-sycl-f8-e4m3.patch` | F8_E4M3 (OCP FP8) weights, which ran on the CPU before: DMMV for one-token MUL_MAT, exact widening to f16 plus the GEMM for batches, MUL_MAT_ID through the per-expert loop, GET_ROWS, and CPY to and from F32, F16 and BF16. The conversions are bit-exact with the CPU's `ggml_fp32_to_f8_e4m3` / `ggml_f8_e4m3_to_fp32`, and they test bits rather than calling `isnan`, so icpx's default `-fp-model=fast` cannot fold the NaN and saturation cases away. | feature (gap fill) |
| ZZ-LOCAL-0007 | `ZZ-LOCAL-0007-sycl-reorder-iq4_nl.patch` | IQ4_NL in the reorder (SoA) layout, dense and per MoE expert. The layout is Q4_0's `[qs][d]` (the blocks are identical), read by a codebook variant of Q4_0's reorder vec_dot for 1..8 columns and by the MoE reorder MMVQ. Dequantize and GET_ROWS (tied embeddings) read the layout too, and the fused / grouped XMX dequant GEMM gets a SoA stage, so a reordered weight keeps that GEMM for prompt batches. Candidate C20's ESIMD MoE GEMM already reads this layout. | perf (gap fill) |
| ZZ-LOCAL-0008 | `ZZ-LOCAL-0008-sycl-moe-reorder-iq3.patch` | Reorders IQ3_XXS and IQ3_S MoE experts per slice (the existing 2D IQ3 layout) for the MoE reorder MMVQ, and adds SoA stages for both types to the fused / grouped XMX dequant GEMM; the block and SoA stages share one decode. MoE prompt batches therefore keep the grouped GEMM after the reorder. The same stages let dense IQ3 weights keep the fused GEMM once the existing 2D reorder is installed; before, those fell back to dequantize plus oneDNN. Candidates C20 / C21 read the same layout. | perf (gap fill) |
| ZZ-LOCAL-0009 | `ZZ-LOCAL-0009-sycl-getrows-q2_0.patch` | Q2_0 in GET_ROWS, through the generic `get_rows_sycl` and the existing `dequantize_q2_0`. Q2_0 had a dequantizer and MMVQ but fell back to the CPU for row lookups. | feature (gap fill) |
| ZZ-LOCAL-0010 | `ZZ-LOCAL-0010-sycl-dg2-register-spills.patch` | Stops register spills on Arc A-series (Xe-HPG) in kernels the A770 runs. The XMX dequant GEMM (prompt batches of the i-quants, and of every MoE type it covers) spilled 1.9-3.5 KB per thread in all 72 DG2 kernels; at sub-group 8 a sub-group now owns 8 rows instead of 16 (still 32 columns) and the two k halves of a step are not unrolled, and none spills. The reorder gate + up MMVQ no longer pairs rows on the A-series, where every paired variant spilled 0.5-3.6 KB (q4_K at 2..4 columns; also iq4_xs with C10, q4_K at 5..8 with C30, q5_K). | perf (DG2) |
| ZZ-LOCAL-0011 | `ZZ-LOCAL-0011-sycl-fuse-rope-norm-moe-sum.patch` | Graph fusions SYCL lacked: ROPE + VIEW + SET_ROWS (the rope kernels' existing set_rows destination, which nothing called), RMS_NORM + MUL + ROPE (+ VIEW + SET_ROWS) for the Q / K norms of Qwen3, Gemma 3 and OLMo 2, the MoE expert sum (MUL by the router weights, the per-expert VIEWs and k - 1 ADDs, up to 15 experts) in one kernel, chains of 3..8 same-shape ADDs, and topk-moe with a routing bias (DeepSeek-V3-style selection on probs + bias). Ported from CUDA (`ggml_cuda_should_fuse_rope_set_rows`, `rms_norm_mul_rope_f32`, `ggml_cuda_match_moe_weighted_reduction`, multi-add, `topk_moe_cuda<has_bias>`). | perf (port) |
| ZZ-LOCAL-0012 | `ZZ-LOCAL-0012-sycl-moe-mmvq-tokens-glu-bias.patch` | The MoE MMVQ (plain and reorder layout) takes 2..8 tokens (per-type cap, CUDA's RDNA3 table, `GGML_SYCL_MMID_MMVQ_MAX_TOKENS` overrides), runs MUL_MAT_ID gate + MUL_MAT_ID up + GLU in one launch (SWIGLU, GEGLU, SWIGLU_OAI, SWIGLU_CLAMP; with the ADD_ID bias of each lane for gpt-oss), and adds a lone MUL_MAT_ID's ADD_ID bias in its store. Ported from CUDA's `mul_mat_vec_q` with ids and fusion args. | perf (port) |
| ZZ-LOCAL-0013 | `ZZ-LOCAL-0013-sycl-fuse-mul-mat-add.patch` | MUL_MAT (one activation column) + ADD of a same-shape tensor, i.e. the residual after the attention output and the FFN down projection, or a bias: the mat-vec writes the ADD's memory and the DMMV kernels add the addend in their store (the A770's ESIMD kernels for q2_K..q6_K and q8_0, the generic kernels for q4_0, q4_1, q5_0, q5_1, q8_0, q1_0). Kernels without the epilogue fall back to one in-place ADD. CUDA, Vulkan and Metal all run this fusion. | perf (port) |
| ZZ-LOCAL-0014 | `ZZ-LOCAL-0014-sycl-reuse-converted-activation.patch` | Sibling mat-muls of one activation (Q, K, V) share its q8_1 (MMVQ) or f16 (DMMV) copy within a graph instead of converting it once each; Vulkan's `prealloc_y` cache. Dropped when a node writes the activation and at the end of every graph. | perf (port) |
| ZZ-LOCAL-0015 | `ZZ-LOCAL-0015-sycl-fuse-model-specific.patch` | The rest of CUDA's model-specific fusions: logit softcap (SCALE + TANH + SCALE) and squared ReLU as one kernel each, the snake activation of the TTS audio decoders (MUL, SIN, SQR, MUL, ADD), SCALE + SIGMOID + SCALE folded into DSV4_HC_POST's post weights, SSM_SCAN writing its state snapshots straight into the cache (skipping the CPY), and topk-moe with sqrt(softplus) gating and DeepSeek-V3 expert-group masking (up to `WARP_SIZE` groups). Shared-expert fusion and multi-stream concurrency are not ported (see `SYCL_DEV_LOG.md`). | perf (port) |
| ZZ-LOCAL-0016 | `ZZ-LOCAL-0016-sycl-mmvq-add-epilogue.patch` | 0013's MUL_MAT + ADD epilogue in the one-column MMVQ store: the generic plain kernel (q4_0 .. q6_K, q8_0, mxfp4, IQ1 / IQ2 / IQ3 / IQ4) and the generic reorder kernel (q4_0, q8_0, mxfp4, q2_K, q3_K, q6_K, IQ3, IQ4_NL) take the addend. The tuned reorder q4_K / q5_K launchers, the q8_0 wide dot product, the XMX MMVQ and 2..8 columns keep the in-place ADD. | perf (port) |
| ZZ-LOCAL-0017 | `ZZ-LOCAL-0017-sycl-dg2-gemm-grf256-opt-in.patch` | `GGML_SYCL_XMX_GRF256=1` runs the sub-group-8 XMX dequant GEMM (f16 and bf16 8x8x16) with the 16-row, unrolled tile and `grf_size<256>` instead of 0010's 8-row tile at 128 registers: 4 threads per EU and twice the B reuse against 8 threads and twice the B loads. Same shape bit and B packing. An `acm-g10` AOT build gives all 72 kernels 256 registers and none spills. Off by default, for an A770 A/B. | perf (DG2, opt-in) |

The former `ZZ-LOCAL-0001` made direct SYCL `CONV_2D` index input, kernel, and
output by their real strides. It was retired on 2026-11-01 as superseded by
upstream #30363, which handles WHCN and CWHN input/kernel layouts on SYCL and
gates other layouts to CPU; CONV_2D's destination is always a new contiguous
tensor.

A former `ZZ-LOCAL-0002` restricted PR 26631's `skip=1` fusion test so it could
not combine with `with_lane_scale`. It was deliberately retired to keep the PR's
test code pristine; the defect it papered over is now documented as a known
full-suite abort in `SYCL_DEV_LOG.md` instead.

### Numbering note — retired 0013

Earlier revisions of this README described 0013 as an XMX/MMQ patch. That story
belonged to the fork's `mmq_xmx.cpp`, which was never present in this directory
as a file. The XMX work now lives in its own slot, **0016**. The former 0013
was the unrelated MMVQ multirow-Y SLM kernel; it is archived because its
integer-tag dependency is in draft PR 29367. Slot 0013 remains unused.

## 0016 — how the XMX path was reshaped

The fork's `mmq_xmx.cpp` (~4,300 lines of `joint_matrix` kernels) was entangled
with the research tree's runtime. The extraction keeps the kernels and the two
public entry points (`ggml_sycl_mmq_xmx_available()`,
`ggml_sycl_mmq_xmx_supports_type()`) and drops the coupling:

- The whole translation unit is wrapped in `#ifdef GGML_SYCL_MMQ_XMX` (mainline
  globs `*.cpp`, so the file is always compiled; when the option is OFF it is an
  empty TU).
- The per-device `xmx_caps` struct is replaced with the two Int8 DPAS tiles,
  picked per device at run time: 8x16x32 at sub-group 16 (Xe2, Xe-HPC) and
  8x8x32 at sub-group 8 (Xe-HPG, so the A770). The kernels live once in
  `mmq_xmx_tile.inc`, which `mmq_xmx.cpp` includes into one namespace per shape;
  `ggml_sycl_mmq_xmx_available()` queries the device's `matrix_combinations` and
  routes to the shape it lists, or keeps the regular MMQ path if it lists
  neither. The shapes have different `reqd_sub_group_size`, so they land in
  separate device images and a JIT build never asks IGC for the other one (IGC
  2.43 crashes building the sub-group-16 tile for `acm-g10`). An AOT build keeps
  only its target's shape through `GGML_SYCL_XMX_AOT_SG`. Before 2026-11-01 the
  tile was fixed at sub-group 16, so an MMQ_XMX=ON build on the A770 only
  worked because Q4_0 never reached the MMQ path. The 8-wide numerics are
  unverified on hardware.
- The tiered-cache lookup, the `GGML_SYCL_XMX_GEMM` env/global machinery, the
  `GGML_SYCL_KTRACE` profiling, the weight-layout tier check, and the
  `mem-ops` / `unified_allocate` self-test are removed.
- The entry points carry an `mmq_` infix since 2026-11-01. Upstream's XMX MMVQ
  work (PR 29864, now merged) defines `static inline ggml_sycl_xmx_supports_type()`
  in `common.hpp`, so the old names made `mmq_xmx.cpp` a redefinition error
  whenever `GGML_SYCL_MMQ_XMX=ON`. The default OFF build was never affected.
- Dispatch is a single `#ifdef GGML_SYCL_MMQ_XMX` branch in `ggml_sycl_mul_mat`:
  when XMX is available and supports the type, it routes the existing
  `use_mul_mat_q` path through `ggml_sycl_op_mul_mat_q_xmx` (a drop-in for
  `ggml_sycl_op_mul_mat_q`).

This is a **reshape, not a faithful port**: the fork's tuned eligibility ladder
is gone, so runtime behavior/perf differs from the research tree.

## Status / verification

- **Apply / unapply:** on 2026-11-03, the current stack (7 fork patches, 16
  upstream PR patches, 2 numbered extracted patches and 15 local patches)
  applied with plain `git apply` in a clean worktree at upstream `fac3a894`,
  with no three-way fallback, followed by the 14 candidates and the 4 Vulkan
  patches (58 in all). The three `unapply.sh` scripts, Vulkan first, returned
  the tracked and untracked tree to upstream. (On 2026-11-01 the same held at
  `dde879bc1` without the local patches.)
- **Build:** no SYCL compiler was available for this refresh, so the SYCL
  translation units were not compiled. A GCC build of the same applied tree
  with `GGML_VULKAN=ON` and tests compiled everything outside `ggml-sycl`
  (core, CPU backend, tests, tools) and the main CTest set passed. The SYCL
  side was checked statically: every manually resolved hunk was re-read, and a
  symbol scan found no call without a definition, no removed definition that
  is still referenced, and no duplicate definition inside one TU. Build with
  `icpx` before relying on it, both with and without `-DGGML_SYCL_MMQ_XMX=ON`.
- **`ZZ-LOCAL-0003` / `ZZ-LOCAL-0004` (2026-11-01):** built with `icpx` 2026.2 (JIT, and `mmvq.cpp`
  for `acm-g10`) and tested with `test-backend-ops` on the CPU OpenCL device through
  `tools/cpu-device-test/`, with and without the reorder forced on: every MUL_MAT and MUL_MAT_ID case
  of the affected types passes against the CPU backend. Speed on the A770 is not measured. Details
  in `SYCL_DEV_LOG.md` ("Quant-type gap fill").
- **`ZZ-LOCAL-0005` to `ZZ-LOCAL-0009` (2026-11-02):** built with `icpx` 2026.2 and tested on the CPU
  OpenCL device: every MUL_MAT, MUL_MAT_ID, GET_ROWS and (non-BF16) CPY case of the new types passes,
  as do the reorder types with the reorder forced on, upstream's reuse tests and the new
  `tools/cpu-device-test/reorder-reuse.cpp`. The new ternary and FP8 conversions and the SoA GEMM stages
  were also checked bit for bit on the host. The six touched files build for `acm-g10` (AOT); the new MMVQ, dequantize, GET_ROWS, DMMV and CPY kernels do not spill. Not verified: speed on the A770 and the
  XMX / ESIMD paths. Details in `SYCL_DEV_LOG.md` ("Quant-type gap fill, part 2").
- **`ZZ-LOCAL-0010` (2026-11-02):** `acm-g10` AOT builds of `fused-gemm.cpp` and `mmvq.cpp`, with and
  without the candidates: no XMX dequant GEMM kernel and no gate + up kernel the A-series takes spills
  any more (before: all 72 GEMM kernels, and q4_K's gate + up at 2..4 columns). Tested on the CPU OpenCL device with the one-row gate + up
  forced on (`MUL_MAT_VEC_FUSION` q4_K 140/140) and off (140/140); the GEMM tiles cannot run there.
  Speed not measured.
- **`ZZ-LOCAL-0011` to `ZZ-LOCAL-0014` (2026-11-03):** built with `icpx` 2026.2 and tested on the CPU OpenCL
  device: `ROPE_SET_ROWS` 73/73, `RMS_NORM_MUL_ROPE` 260/260, `MOE_REDUCE` 6/6, `ADD_ADD` 12/12,
  `MUL_MAT_VEC_FUSION` 1754/1754 and `MUL_MAT_ID` 1205/1205, each with and without the reorder forced on,
  `TOPK_MOE` 351/352 on the power-of-two expert counts (the one failure, an expert-group case that is
  not fused, passed 16 reruns with fusion on and off). The SYCL debug log shows each fusion running. Every new or changed kernel builds
  for `acm-g10` (AOT) without spills except the plain-layout q3_K MoE kernel with the gate (672 B), which
  the dispatcher keeps off. The four patches apply in order with plain `git apply`, each stage passes a device-side syntax check,
  all candidates still apply on top with plain `git apply`, and the full round trip with the Vulkan
  stack returns a clean tree. Not verified: speed on the A770 and the ESIMD epilogue at run time
  (ESIMD does not run on the CPU device). Details in `SYCL_DEV_LOG.md` ("Fusions ported from CUDA, Vulkan
  and Metal").
- **`ZZ-LOCAL-0015` to `ZZ-LOCAL-0017` (2026-11-03):** built with `icpx` 2026.2 and tested on the CPU OpenCL
  device: `SOFTCAP` 1/1, `RELU_SQR` 4/4, `SNAKE_FUSE` 16/16, `DSV4_HC_POST` 9/9, `SSM_SCAN_CACHE_FUSION` 7/7,
  `SSM_SCAN` 12/12, `TOPK_MOE` 384/384 on the power-of-two expert counts (sqrt(softplus) and up to 16 expert
  groups fused; one earlier run hit the unfused 32-group flake seen with 0011, which passed 8 reruns), and
  `MUL_MAT_VEC_FUSION` 1818/1818 with and without the reorder forced, 64 of those scratch MUL_MAT + ADD cases
  covering the types 0016 adds (IQ1..IQ4, q5_0 / q5_1, MXFP4, the reorder k-quants). The SYCL debug log
  shows each fusion running. `acm-g10` AOT with the per-kernel device-code split: no new kernel spills except
  topk-moe's expert-group instantiations at 512 experts (1.0-1.2 KB, with the 256 registers they take on XMX
  GPUs; 7.9-9.1 KB at 128); 0017's 72 tiles get 256 registers and do not spill. Applies in order with plain `git apply`, every
  candidate applies on top (C40 now also asks for 256 registers in its D=256 SPMD kernels), and the round trip
  with the Vulkan stack is clean. Not verified: speed on the A770, and the 256-register kernels at run time.
  Details and the register-file audit in `SYCL_DEV_LOG.md` ("Remaining fusions, MMVQ add epilogue and the
  register-file audit").
- The scripts snapshot pre-existing untracked paths in the worktree's Git
  administrative directory. Files introduced by an apply are removed on
  unapply, while paths present before apply are protected. The state file is
  removed after a successful reverse.
- **Historical perf measurement (older base):** `llama-bench` on Arc A770 (Alchemist), oneAPI 2026.0,
  `gemma-4-E2B-it-heretic-ara-Q4_0` (3.12 GiB, 4.65 B), `-p 512 -n 128 -r 3`,
  `level_zero:0`, full stack applied on the older base `86d86ed43`:

  | `GGML_SYCL_MMQ_XMX` | pp512 (t/s) | tg128 (t/s) |
  |---|---|---|
  | OFF (default) | 1377.5 ± 2.4 | 29.63 ± 0.03 |
  | ON | 1378.4 ± 4.6 | 29.59 ± 0.02 |

  ON vs OFF is within run-to-run noise: the A770 routes Q4_0 matmuls through
  oneMKL BLAS, so 0016's XMX MMQ branch is rarely (if ever) reached for this
  model — exactly as predicted below. The build compiles and runs correctly in
  both configurations; a measurable XMX delta would require a workload that
  actually dispatches through MMQ (K-quants, or Battlemage). These are
  whole-stack numbers, not per-patch deltas. Both throughputs are ~10-13% higher
  than the previous base (`a646006f0`: 1255/26.21) - attributable to the advanced
  mainline and the re-anchored FA patches, not to XMX.

## Apply order

`apply.sh` applies fork patches, upstream PRs, numbered extracted patches, then
local repairs. `unapply.sh` reverses those groups in the exact inverse order. The
supported flow is apply -> build/test -> unapply -> apply. Re-running `apply.sh`
on a fully applied tree is not supported: it stops at the first fork patch whose
context a later patch changed (0002 today), before writing anything. Run
`unapply.sh` first.

```bash
cd /home/simonlui/Code_Repositories/llama.cpp
./sycl-patches-extracted/apply.sh      # fork -> PR -> extracted -> local repairs
# ... build / test ...
./sycl-patches-extracted/unapply.sh    # local -> extracted -> PR -> fork (back to pristine)
```

Both scripts resolve the fork/PR patches relative to their own location, so they
can be pointed at any checkout: `./sycl-patches-extracted/apply.sh /path/to/other/llama.cpp`.

Dependency summary (each "needs at least"):

| # | Needs |
|---|---|
| 0015 | — (file-disjoint) |
| 0016 | — (new files + one gated dispatch branch; independent) |

Archived patches are under `../patch-backups/upstream-refresh-20261010/sycl-patches/`.
The former 0013 and local repairs 0002/0004 depended on draft PR 29367. Local
repair 0003 was superseded when PR 29809 merged. Their original patch bytes are
retained in the hash-named archive. `ZZ-LOCAL-0001` was retired on 2026-11-01;
its last version is in this branch's Git history.

## What was deliberately left behind

From the review of `llama.cpp-sycl-terrible` against mainline backend
conventions (CUDA is the reference; SYCL mirrors it), these were judged
non-idiomatic and not extracted:

- **The memory/runtime empire** — `unified-cache` (~19k lines), `tlsf-allocator`,
  `vram-pool`, `compute-buffer-manager`, `kv-tier-manager`, eviction/heat
  policies, prefetch schedulers. Mainline already provides ggml-alloc and
  `ggml_sycl_pool`; no other backend reimplements allocation/tiering.
- **The runtime autotuner/dispatcher** — `dispatch-tuning`, `tiered-dispatch`,
  `tuning-engine`, `kernel-selection`, `orchestrator`, `transfer-learning`,
  `crossover-discovery`. Reference backends use a static op switch.
- **ESIMD kernels** (`*-esimd.hpp`) — Intel-only `sycl::ext::intel::esimd`; breaks
  SYCL single-source portability.
- **Research/profiling scaffolding** — `gpu-sampler`, probes, `ab-validation`,
  `e2e-profile`, `sycl-timeline`. Belongs in tooling/tests, not the backend.

### Deferred (real features, but multi-session ports)

Named, not numbered, so this index cannot drift from the files again:

- **fattn-onednn** — oneDNN-graph SDPA path; needs new `sdpa_cache` infra on
  `ggml_backend_sycl_context` and a mainline-shaped `fattn-common.hpp` extension.
- **fattn-xmx** — `joint_matrix` flash-attention; shares fattn params with
  fattn-onednn, extends the FA kernel switch. The fork's fp16 tile shape is
  rejected by oneAPI 2025.3's JIT on A770 (Alchemist); revisit on Battlemage.
- **moe-xmx** — XMX MoE path; clean kernels but ~600 lines of dispatch glue
  referencing a `MoEContext` that does not exist in mainline. Unmeasurable on
  A770 (no MoE model fits 16 GB).

## Why the small patches behave the way they do on A770

`GGML_SYCL_DEBUG=1` shows mainline routes A770 Gemma Q4_0 matmuls through
`ggml_sycl_op_mul_mat_sycl` (oneMKL BLAS with on-the-fly dequant), not MMQ:

- The `RMS_NORM -> MUL -> ADD` fusion is upstream in PR 27610 and is not carried
  as a local extracted patch.
- The former **0013** multirow MMVQ experiment is not active because it depends
  on the integer-tag dispatch from draft PR 29367; its patch remains archived.
- **0015** trims softmax reductions; softmax is a small fraction of Q4_0
  inference time, so end-to-end effect is small but the kernel is simpler.
- **0016** adds an XMX branch to the MMQ path — but the A770 dispatcher rarely
  reaches MMQ for Q4_0 (it prefers BLAS), so any win is more likely on
  Battlemage / K-quant models. Whether Alchemist's JIT accepts the Int8 tile at
  runtime must be confirmed on hardware.

## Validation commands

```bash
source /opt/intel/oneapi/setvars.sh --force
cd /home/simonlui/Code_Repositories/llama.cpp
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL=ON \
    -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build build -j$(nproc) --target llama-bench

ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
    -m ~/LLM/Gemma-4-E2B-It-Heretic-Ara-4.6B-Q4_0.gguf \
    -p 512 -n 128 -r 3

# To opt into 0016's XMX path (the tile shape is picked per device), rebuild with:
cmake -B build -DGGML_SYCL_MMQ_XMX=ON && cmake --build build -j$(nproc) --target llama-bench
```

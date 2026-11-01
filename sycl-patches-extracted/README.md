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

Manual repairs that are required by the current upstream tree are kept in the
separate `ZZ-LOCAL-*` group at the top of this directory. They are not folded
into the numbered extracted history and remain independently reversible.

## Patch index

| # | File | What it does | Type |
|---|------|--------------|------|
| 0015 | `0015-sycl-softmax-reduce-over-group.patch` | Replaces the block max/sum barrier+SLM reductions in `softmax.cpp` with `sycl::reduce_over_group`. Reduction-only; **keeps upstream exp semantics** (`sycl::native::exp` + the `-80.0f` clamp). | cleanup |
| 0016 | `0016-sycl-mmq-xmx.patch` | Experimental XMX (Intel Xe Matrix eXtensions / `joint_matrix`) Int8 MMQ path, gated behind the `GGML_SYCL_MMQ_XMX` CMake option (default OFF). Reshaped from the fork's `mmq_xmx.*`. | perf (build-flag gated) |
| local 0001 | `ZZ-LOCAL-0001-sycl-conv2d-strided-layout.patch` | Makes direct SYCL `CONV_2D` use the actual element strides of the input, kernel, and output tensors, including channel-contiguous (`cwhn`) layouts. | local repair |

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
public entry points (`ggml_sycl_xmx_available()`, `ggml_sycl_xmx_supports_type()`)
and drops the coupling:

- The whole translation unit is wrapped in `#ifdef GGML_SYCL_MMQ_XMX` (mainline
  globs `*.cpp`, so the file is always compiled; when the option is OFF it is an
  empty TU).
- The per-device `xmx_caps` struct is replaced with the compile-time Int8 tile
  (`XMX_M=8, XMX_N=16, XMX_K=32`).
- The tiered-cache lookup, the `GGML_SYCL_XMX_GEMM` env/global machinery, the
  `GGML_SYCL_KTRACE` profiling, the weight-layout tier check, and the
  `mem-ops` / `unified_allocate` self-test are removed.
- Dispatch is a single `#ifdef GGML_SYCL_MMQ_XMX` branch in `ggml_sycl_mul_mat`:
  when XMX is available and supports the type, it routes the existing
  `use_mul_mat_q` path through `ggml_sycl_op_mul_mat_q_xmx` (a drop-in for
  `ggml_sycl_op_mul_mat_q`).

This is a **reshape, not a faithful port**: the fork's tuned eligibility ladder
is gone, so runtime behavior/perf differs from the research tree.

## Status / verification

- **Apply / unapply:** on 2026-10-10, the current stack (7 fork patches, 13
  upstream PR patches, 2 numbered extracted patches, and 1 local repair)
  applied in a clean worktree at upstream
  `23b0202a189c44a54625aadcb37a946dd1d6278`. `unapply.sh` reversed the exact
  order and returned the tracked and untracked tree to upstream. No conflict
  markers remained. No build or tests were run in this refresh; the user will
  build the applied stack.
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

`apply.sh` applies fork patches, upstream PRs, the late upstream dependency
PR 29864, numbered extracted patches, then local repairs. PR 29864 is applied
after the other upstream PRs because its regenerated diff follows PR 29889.
`unapply.sh` reverses those groups in the exact inverse order. The supported
flow is apply -> build/test -> unapply -> apply. Re-running `apply.sh` on a fully
applied tree skips the existing patches; a real context conflict still aborts
with the affected paths printed.

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
| local 0001 | Needs the current numbered extracted stack, because it repairs `conv2d.cpp` after those patches are applied |

Archived patches are under `../patch-backups/upstream-refresh-20261010/sycl-patches/`.
The former 0013 and local repairs 0002/0004 depended on draft PR 29367. Local
repair 0003 was superseded when PR 29809 merged. Their original patch bytes are
retained in the hash-named archive.

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

# To opt into 0016's XMX path, add -DGGML_SYCL_MMQ_XMX=ON and rebuild:
cmake -B build -DGGML_SYCL_MMQ_XMX=ON && cmake --build build -j$(nproc) --target llama-bench
```

# Opt-in SYCL candidates for the Arc A770 (DG2)

Patches that may make the A770 faster but are not proven on it yet. They are kept out of
the main stack: apply them on top of it, measure, and promote the winners into
`sycl-patches/` (upstream PRs) or `sycl-patches-extracted/` (everything else).

```bash
bash sycl-patches-extracted/apply.sh .      # main SYCL stack first
bash sycl-patches-candidates/apply.sh .     # then C10..C90
# build, test, measure
bash sycl-patches-candidates/unapply.sh .   # reverse the candidates first
bash sycl-patches-extracted/unapply.sh .
```

The candidates apply in file-name order with plain `git apply` and reverse cleanly. As with
the main stack, re-running `apply.sh` on an applied tree is not supported.

| # | Source | What | Why it may help the A770 | Off switch |
|---|--------|------|--------------------------|------------|
| C10 | PR 31251 (closed unmerged, no successor) | Reordered (SoA) IQ4_XS layout and MMVQ, multi-column and fused GLU | IQ4_XS GEMV was bound by table lookups: 261 -> 380 GB/s in the author's probe. Runs on every Intel GPU | none (reorder applies to IQ4_XS weights) |
| C20 | PR 31428 (draft, on top of carried PR 30156) | ESIMD grouped MoE GEMM that dequantizes straight into the DPAS operands | Base for C21. On its own it only replaces the sub-group-16 shape, so no change on DG2 | `GGML_SYCL_MMID_ESIMD=0` |
| C21 | cwriter `36b5038` (not in a PR) | The C20 kernel for the Xe-HPG 8x8x16 sub-group-8 DPAS shape: 8-wide DPAS, 16 rows, 1D loads | The only MoE GEMM change written for Arc A-series. 1.6-2.6x per MoE layer for C20 on Xe2; the DG2 variant was only run with software DPAS | `GGML_SYCL_MMID_ESIMD=0` |
| C30 | PR 30226 (closed unmerged) | q4_K MMVQ row pairing up to 8 columns | Speculative-decoding verify; Battlemage-gated, enable on DG2 with C90 bit 16 | C90 bit 16 off |
| C40 | ef777/a770-qwen-xe1 `da7f577`, plus a local DG2 build fix | GQA flash-decode kernel, 1-8 tokens, q8_0/f16 KV, D=64/128/256 | VEC rereads K/V per query head; this reads each KV head once for all its query heads. Author on A770: 24k n1 520 -> 410 us, 12k n4 762 -> 404 us | `GGML_SYCL_FA_GQA_DEC=0` |
| C50 | Frostbite1536/A770 `0a31dc7`, p11c F | TILE FA flushes its FP16 V accumulator to FP32 once per KV tile | Probable long-context accuracy fix: NMSE up to 2.3 at 16k keys per block, CPU OpenCL device only | none |
| C51 | Frostbite1536/A770, patch 0005 | 16 biased-V FA test cases with an FP32 CPU reference | Tests C50; CPU FA changes only under the test reference flag | none |
| C60 | local port of upstream PR 31125 | Upstream's DPAS flash attention for q4_0 / q8_0 KV (1-31 query tokens), with the DPAS width as a template parameter: sub-group 16 as upstream, sub-group 8 for Xe-HPG, picked per device at run time | Upstream measured tg128 at 16K 31.3 -> 54.2 t/s on a B580 (q4_0 KV), but gates it to Xe2. On DG2 it replaces VEC for quantized-KV decode and small batches, and it is the only candidate that covers q4_0 KV. With C40 applied, C40 still takes q8_0 decode at 1-8 tokens and D 64/128/256 | `GGML_SYCL_FA_DPAS=0` |
| C90 | local | `GGML_SYCL_DG2_BMG_TUNING` bit mask that opens five Battlemage-only kernel choices to DG2 | Each choice is between kernels already built for every device, so only speed can change | `GGML_SYCL_DG2_BMG_TUNING=0` (default) |

`GGML_SYCL_DG2_BMG_TUNING` bits:

| Bit | Gate | Upstream default on DG2 |
|-----|------|-------------------------|
| 1 | quantized-KV decode (1-2 tokens) on FA TILE | VEC |
| 2 | FA VEC with 256-thread work-groups | 128 |
| 4 | q5_K multi-column MMVQ reuse and fused gate/up | per-column kernel |
| 8 | D=256 single-query decode on FA VEC (PR 30606) | dispatcher default |
| 16 | q4_K MMVQ row pairing up to 8 columns (needs C30) | pairing up to 4 |

Bits 1 and 8 choose between TILE and VEC, which C60 and C40 run ahead of for q4_0 / q8_0 KV:
measure them with `GGML_SYCL_FA_DPAS=0 GGML_SYCL_FA_GQA_DEC=0`.

## Hardware gates are different

`sycl-patches/check_arch_gates.sh` lists every architecture gate and tags the ones in files
that use DPAS, joint_matrix, ESIMD or 2D block loads as `HW`. Do not add DG2 to an `HW` gate.
The A770 runs DPAS at execution size 8; Xe2 runs it at 16. On `acm-g10` the sub-group-16
DPAS builtin compiles without an error or warning and produces no DPAS instruction
(Frostbite1536/A770 compile probes), so a widened `HW` gate can build and then compute
garbage. Those paths need a sub-group-8 port: C21 is that port for the MoE GEMM and C60 for
the DPAS flash attention. The remaining `HW` gate, upstream's XMX multi-column MMVQ, is not
worth porting: ef777 measured it on the A770 slower than dequantize + GEMM at 9-80 columns,
and their own 8-row XMX matvec slower than dp4a at 1-8. ef777 also found an XMX
flash-attention kernel of their own correct but slower (SLM and barrier stalls). That is a
different kernel from C60, so C60 has to win its benchmark like every other candidate.

## Test plan on the A770

Correctness first, with the same build as for the main stack plus the candidates:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build_sycl/bin/test-backend-ops -b SYCL0 -o FLASH_ATTN_EXT
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build_sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT -p iq4_xs
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build_sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT_ID
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_DG2_BMG_TUNING=31 ./build_sycl/bin/test-backend-ops -b SYCL0 -o FLASH_ATTN_EXT
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_DG2_BMG_TUNING=31 ./build_sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT -p q4_K
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_FA_GQA_DEC=0 ./build_sycl/bin/test-backend-ops -b SYCL0 -o FLASH_ATTN_EXT
```

With C60 the log shows `fattn_dpas_device_sg: device 0: DPAS flash attention at sub-group 8
(Xe-HPG)` on the first quantized-KV FA call. The last command routes q8_0 decode to C60
instead of C40.

Then speed, one switch at a time against the same build (`-d` sweeps the KV depth that
the FA candidates target; use a GQA model with q8_0 KV for C40):

```bash
for t in 0 1 2 4 8 16; do GGML_SYCL_FA_DPAS=0 GGML_SYCL_FA_GQA_DEC=0 GGML_SYCL_DG2_BMG_TUNING=$t ./build_sycl/bin/llama-bench -m MODEL -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -d 0,4096,16384 -r 3; done
GGML_SYCL_FA_GQA_DEC=0 ./build_sycl/bin/llama-bench -m MODEL -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -d 0,4096,16384 -r 3
GGML_SYCL_MMID_ESIMD=0 ./build_sycl/bin/llama-bench -m MOE_MODEL -p 512 -n 128 -r 3
```

C60 against the kernels it replaces, for each KV type (`-p 16 -ub 16` is a 16-token batch,
the shape of a speculative-decoding verify):

```bash
for kv in q4_0 q8_0; do for dpas in 1 0; do GGML_SYCL_FA_GQA_DEC=0 GGML_SYCL_FA_DPAS=$dpas ./build_sycl/bin/llama-bench -m MODEL -fa 1 -ctk $kv -ctv $kv -p 16 -ub 16 -n 128 -d 0,4096,16384 -r 3; done; done
GGML_SYCL_FA_GQA_DEC=1 ./build_sycl/bin/llama-bench -m MODEL -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -d 0,4096,16384 -r 3
```

The last line is C40 on q8_0 decode, to compare with the `dpas=1 q8_0` rows.

Promote a candidate when its tests pass and it wins at the depths you use. A tuning bit that
wins can become a plain DG2 entry in that gate.

## Build status of this group (2026-11-01)

Compiled with oneAPI 2026.2 (conda-forge) on a box without an Intel GPU. Each
risky translation unit was also built for `acm-g10` with `spir64_gen` (the SPIR-V
the A770 compiles at JIT time) and its DG2 assembly read from IGC dumps:

- C20/C21 `fused-gemm-esimd.cpp`: builds; every DPAS is `dpas.8x8 (8|M0)`. The
  default 8-wide path is spill-free at 256 GRF, which the kernel requests. A
  `GGML_SYCL_DEVICE_ARCH=acm-g10` build gets 128 GRF and spills 3.7-9 KB per
  kernel, so measure with a JIT build.
- C10/C30 `mmvq.cpp`: builds. Upstream's Xe2-only XMX MMVQ kernels in the same
  file also compile for DG2, as 8-wide DPAS with spills; they stay gated off.
- Since local patch `ZZ-LOCAL-0010` (2026-11-02), the stack's own `joint_matrix` grouped
  GEMM no longer spills on DG2 at 128 GRF (8 rows a sub-group, k halves not unrolled), so
  `GGML_SYCL_MMID_ESIMD=0` now compares C21 against a spill-free kernel at full occupancy
  rather than the spilling one C21 was written to replace. The same patch stops the gate + up
  MMVQ pairing rows on DG2, where every paired variant spills: that covers C10's iq4_xs pairing
  (3..4 columns, 0.7-1 KB) and C30's 5..8-column q4_K pairing under C90 bit 16 (2.3-3.6 KB),
  and C90 bit 4 now gives q5_K weight reuse without the paired gate + up (1.1-3.5 KB). C30's
  plain-MMVQ pairing does not spill on DG2 and is unaffected.
- Since local patch `ZZ-LOCAL-0012` (2026-11-03), MUL_MAT_ID batches of 2 tokens up to a per-type cap
  (4 for the k-quants, 8 for q4_0 / q8_0 / mxfp4) take the MoE MMVQ instead of the grouped GEMM, so C20 / C21
  only see wider batches. `GGML_SYCL_MMID_MMVQ_MAX_TOKENS=1` restores the old split when comparing them.
- C40 `fattn-gqa-dec.cpp`: the original failed. The q8_0 D=256 SLM ESIMD kernel's
  in-kernel `mask` null test made IGC 2.43 emit a 64-bit shift that DG2 cannot
  execute, and that failed every ESIMD kernel in the file. C40 now tests `mask`
  on the host; all 37 kernels build. At the 128 GRF of an AOT build, the D=256
  SPMD and ESIMD variants spill (0.7-12 KB).
- C50 TILE FA instance: builds.
- C60 `fattn-dpas.cpp`: builds for `acm-g10` (42 kernels, all 782 DPAS `dpas.8x8 (8|M0)`,
  none spill) and `bmg-g21` (the same 604 DPAS as upstream, instruction counts within 1%,
  none spill). The sub-group-8 kernels that spilled at 128 GRF (32 rows from D=128, staged
  D=256 decode) always use 256. In a JIT build each device image holds one DPAS width.
  `tools/fattn-dpas-emu/run.sh` runs the kernels on the CPU OpenCL device with emulated
  DPAS: 112 cases pass at both widths (NMSE vs FP64 <= 4.8e-5, sub-group 8 vs 16 <=
  1.6e-11), and it fails on planted sub-group-8 layout bugs. Rerun it when C60 is
  re-anchored.

What was not checked: any number on real hardware. Apart from C60's emulation,
no kernel here ran: the backend does not run on the CPU OpenCL device (see
`SYCL_DEV_LOG.md`), and ESIMD, DPAS and joint_matrix need the GPU.

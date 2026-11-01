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
| C40 | ef777/a770-qwen-xe1 `da7f577` | GQA flash-decode kernel, 1-8 tokens, q8_0/f16 KV, D=64/128/256 | VEC rereads K/V per query head; this reads each KV head once for all its query heads. Author on A770: 24k n1 520 -> 410 us, 12k n4 762 -> 404 us | `GGML_SYCL_FA_GQA_DEC=0` |
| C50 | Frostbite1536/A770 `0a31dc7`, p11c F | TILE FA flushes its FP16 V accumulator to FP32 once per KV tile | Probable long-context accuracy fix: NMSE up to 2.3 at 16k keys per block, CPU OpenCL device only | none |
| C51 | Frostbite1536/A770, patch 0005 | 16 biased-V FA test cases with an FP32 CPU reference | Tests C50; CPU FA changes only under the test reference flag | none |
| C90 | local | `GGML_SYCL_DG2_BMG_TUNING` bit mask that opens five Battlemage-only kernel choices to DG2 | Each choice is between kernels already built for every device, so only speed can change | `GGML_SYCL_DG2_BMG_TUNING=0` (default) |

`GGML_SYCL_DG2_BMG_TUNING` bits:

| Bit | Gate | Upstream default on DG2 |
|-----|------|-------------------------|
| 1 | quantized-KV decode (1-2 tokens) on FA TILE | VEC |
| 2 | FA VEC with 256-thread work-groups | 128 |
| 4 | q5_K multi-column MMVQ reuse and fused gate/up | per-column kernel |
| 8 | D=256 single-query decode on FA VEC (PR 30606) | dispatcher default |
| 16 | q4_K MMVQ row pairing up to 8 columns (needs C30) | pairing up to 4 |

## Hardware gates are different

`sycl-patches/check_arch_gates.sh` lists every architecture gate and tags the ones in files
that use DPAS, joint_matrix, ESIMD or 2D block loads as `HW`. Do not add DG2 to an `HW` gate.
The A770 runs DPAS at execution size 8; Xe2 runs it at 16. On `acm-g10` the sub-group-16
DPAS builtin compiles without an error or warning and produces no DPAS instruction
(Frostbite1536/A770 compile probes), so a widened `HW` gate can build and then compute
garbage. Those paths need a sub-group-8 port, as C21 is for the MoE GEMM. Measured on the
A770 by ef777: an 8-wide port of the XMX multi-column MMVQ was slower than dp4a at 1-8
columns and slower than dequantize + GEMM at 9-80, and an XMX flash-attention kernel was
correct but slower. Neither is worth porting.

## Test plan on the A770

Correctness first, with the same build as for the main stack plus the candidates:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build_sycl/bin/test-backend-ops -b SYCL0 -o FLASH_ATTN_EXT
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build_sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT -p iq4_xs
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build_sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT_ID
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_DG2_BMG_TUNING=31 ./build_sycl/bin/test-backend-ops -b SYCL0 -o FLASH_ATTN_EXT
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_DG2_BMG_TUNING=31 ./build_sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT -p q4_K
```

Then speed, one switch at a time against the same build (`-d` sweeps the KV depth that
the FA candidates target; use a GQA model with q8_0 KV for C40):

```bash
for t in 0 1 2 4 8 16; do GGML_SYCL_DG2_BMG_TUNING=$t ./build_sycl/bin/llama-bench -m MODEL -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -d 0,4096,16384 -r 3; done
GGML_SYCL_FA_GQA_DEC=0 ./build_sycl/bin/llama-bench -m MODEL -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -d 0,4096,16384 -r 3
GGML_SYCL_MMID_ESIMD=0 ./build_sycl/bin/llama-bench -m MOE_MODEL -p 512 -n 128 -r 3
```

Promote a candidate when its tests pass and it wins at the depths you use. A tuning bit that
wins can become a plain DG2 entry in that gate.

## Build status of this group (2026-11-01)

See the "Build check" section of `SYCL_DEV_LOG.md` for what was compiled for DG2 without
a GPU, and what could only be checked statically.
